"""
cmip6_wbgt_facility_projection.py

Extract projected WBGT (and precipitation) at each health-facility location from
NEX-GDDP-CMIP6 model output, and write per-facility monthly panels for use in the
climate-health regression and in the TLO WeatherDisruptions module.

Pipeline
--------
1. Read the facility registry and keep facilities with coordinates.
2. Restrict to the facility NAMES used by the TLO simulation
   (``ResourceFile_Facility_Characteristics.csv``), so the panel keys match the
   ``level_0..level_3`` values Demography assigns at runtime.
3. For each CMIP6 model x scenario: find the facility's nearest grid cell, extract
   the daily WBGT series, aggregate to monthly mean and monthly extreme indices
   (WBGTx = monthly max; WBGT5x = monthly max of the within-month 5-day rolling
   mean), and attach the matching monthly precipitation.
4. Write one "mean" panel and one "extreme" panel (with lags) per model x scenario,
   plus a single facility covariate table.

Outputs (in ``OUTPUT_DIR``)
--------------------------
  wbgt_monthly_mean_facility_{model}_{scenario}.csv
      long: facility, date, wbgt_day, wbgt_night, precip_month, precip_5day
  wbgt_extreme_indices_facility_{model}_{scenario}.csv
      long: facility, date, {wbgtx,wbgt5x}_{day,night}, precip_month, precip_5day,
            and a lag column per extreme index
  facility_info_projection.csv
      facility characteristics x facility (owner, urban/rural, altitude, type,
      coordinates, nearest-facility and nearest-clinic distances, mean exposures).
      Used by the regression model, not by the disruption module.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist

# ===========================================================================
# CONFIG
# ===========================================================================
WBGT_DIRECTORY = "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6_Split/"
WBGT_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
WBGT_FILE_PREFIX = "wbgt_daynight_"

WBGT_VARS = ["wbgt_day", "wbgt_night"]
EXTREME_VARS = ["wbgtx_day", "wbgtx_night", "wbgt5x_day", "wbgt5x_night"]
LAG_MONTHS = [1, 2, 3, 4, 9]
PRECIP_VARS = ["precip_month", "precip_5day"]
PRECIP_FILE_PREFIX = "precip_monthly_"
ALL_VARS = WBGT_VARS + PRECIP_VARS

WBGT_TIME_COORD = "time"
WBGT_LAT_COORD = "lat"
WBGT_LON_COORD = "lon"

EXCLUDE_MODELS = {"GISS-E2-1-G"}   # all-NaN WBGT; excluded from ensemble
DISTANCE_GUARD_KM = 30.0
ROLLING_WINDOW_DAYS = 5            # window for the WBGT5x index

# Facility registry (projection coordinates + characteristics)
FACILITIES_CSV = (
    "/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/"
    "climate_change_impacts/facilities_with_lat_long_region.csv"
)
FACILITY_NAME_COL = "Fname"
FACILITY_LAT_COL = "A109__Latitude"
FACILITY_LON_COL = "A109__Longitude"

# TLO facility registry — the authoritative set of facility NAMES the simulation
# assigns people to. Demography loads this file (index_col=0, transposed) and uses
# the facility name 'Fname' as the level_0..level_3 category, so population.props
# level_X holds a facility NAME, not a numeric id. The exposure output must be keyed
# by these exact names so WeatherDisruptions can match facility_used -> disruption.
TLO_FACILITY_CHARACTERISTICS_CSV = (
    "/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/demography/"
    "ResourceFile_Facility_Characteristics.csv"
)
RESTRICT_TO_TLO_FACILITIES = True

RESTRICT_TO_FACILITIES_FILE = None   # optional: a historical panel to restrict to

OUTPUT_DIR = "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices"

COVARIATE_COLS = ["Zonename", "Resid", "Dist", "A105", "A114", "A109__Altitude",
                  "Ftype", "A109__Latitude", "A109__Longitude"]

# Registry district-name corrections -> canonical names
DISTRICT_NAME_FIXES = {
    "Blanytyre": "Blantyre",
    "Nkhatabay": "Nkhata Bay",
    "Mzimba North": "Mzimba",
    "Mzimba South": "Mzimba",
}


# ===========================================================================
# Time handling — NEX-GDDP-CMIP6 uses a NoLeap (365-day) calendar, so the time
# coordinate comes back as cftime objects, not numpy.datetime64.
# ===========================================================================
def get_time_index(time_values):
    """Return a monthly 'YYYY-MM' string key per timestep, for cftime OR datetime64
    time coordinates. For DAILY files this yields a repeated month key per day,
    which is exactly the grouping key for the monthly aggregations."""
    keys = []
    for t in time_values:
        if hasattr(t, "year") and hasattr(t, "month"):
            keys.append(f"{t.year:04d}-{t.month:02d}")
        else:
            ts = pd.Timestamp(t)
            keys.append(f"{ts.year:04d}-{ts.month:02d}")
    return pd.Index(keys, name="date")


def assert_daily(time_values, nc_path):
    """GUARD: abort if the file's timesteps are not daily."""
    if len(time_values) < 2:
        raise ValueError(f"{Path(nc_path).name}: fewer than 2 timesteps")
    diffs = []
    for i in range(min(len(time_values) - 1, 366)):
        d = time_values[i + 1] - time_values[i]
        diffs.append(d.days if hasattr(d, "days") else d / np.timedelta64(1, "D"))
    med = float(np.median(diffs))
    if med > 5:
        raise ValueError(
            f"{Path(nc_path).name}: timesteps look MONTHLY (median spacing "
            f"{med:.0f} days). This script expects DAILY {WBGT_FILE_PREFIX}*.nc "
            f"— aborting rather than mislabelling monthly data as daily.")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between arrays of points."""
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def find_model_files(base_dir, scenario):
    """Discover (model_id, path) for {model}/{scenario}/{prefix}*.nc."""
    found = []
    for model_dir in sorted(Path(base_dir).iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name in EXCLUDE_MODELS:
            continue
        scenario_dir = model_dir / scenario
        if not scenario_dir.is_dir():
            continue
        matches = sorted(scenario_dir.glob(f"{WBGT_FILE_PREFIX}*.nc"))
        if not matches:
            continue
        found.append((model_dir.name, matches[0]))
    return found


# ===========================================================================
# Facility registry
# ===========================================================================
def load_tlo_facility_names(path):
    """Return the set of facility NAMES the TLO simulation assigns people to.
    Demography loads this file index_col=0 then transposes, so facility names are
    the (transposed) index."""
    tlo = pd.read_csv(path, index_col=0).T
    tlo.index.name = "Fname"
    return set(tlo.index.astype(str).str.strip())


def load_facilities():
    """Load the projection facility registry: drop duplicates and facilities without
    coordinates, optionally restrict to a historical panel, and (by default) align
    the set to the TLO facility NAMES so every emitted key is one the simulation can
    assign. Returns (facilities_df, names, lats, lons)."""
    facilities = pd.read_csv(FACILITIES_CSV).drop_duplicates(FACILITY_NAME_COL)

    if RESTRICT_TO_FACILITIES_FILE:
        hist = pd.read_csv(RESTRICT_TO_FACILITIES_FILE, index_col=0)
        keep = set(hist.columns)
        before = len(facilities)
        facilities = facilities[facilities[FACILITY_NAME_COL].isin(keep)]
        print(f"Restricted to {len(facilities)}/{before} historically-fitted facilities "
              f"from {Path(RESTRICT_TO_FACILITIES_FILE).name}")

    has_coords = facilities[FACILITY_LAT_COL].notna() & facilities[FACILITY_LON_COL].notna()
    facilities = facilities[has_coords].reset_index(drop=True)

    if RESTRICT_TO_TLO_FACILITIES:
        tlo_names = load_tlo_facility_names(TLO_FACILITY_CHARACTERISTICS_CSV)
        reg_names = set(facilities[FACILITY_NAME_COL].astype(str).str.strip())

        facilities = facilities[
            facilities[FACILITY_NAME_COL].astype(str).str.strip().isin(tlo_names)
        ].reset_index(drop=True)

    n_dropped = len(reg_names - tlo_names)
    n_missing = len(tlo_names - reg_names)
    if n_dropped or n_missing:
        print(
            f"facility name alignment: {n_dropped} registry facilities not in TLO set (dropped); "
            f"{n_missing} TLO facilities absent from registry (no WBGT)"
        )
    names = facilities[FACILITY_NAME_COL].tolist()
    lats = facilities[FACILITY_LAT_COL].astype(float).values
    lons = facilities[FACILITY_LON_COL].astype(float).values
    return facilities, names, lats, lons


# ===========================================================================
# Grid -> facility extraction and monthly aggregation
# ===========================================================================
def open_cftime_dataset(path):
    """Open a netCDF with cftime decoding (NoLeap calendar)."""
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        return xr.open_dataset(path, decode_times=time_coder)
    except AttributeError:
        return xr.open_dataset(path, use_cftime=True)


def nearest_cell_indices(lat_data, lon_data, facility_lats, facility_lons):
    """Nearest-grid-cell indices per facility, and a boolean 'too far' guard mask."""
    ix = np.array([((lon_data - lon) ** 2).argmin() for lon in facility_lons])
    iy = np.array([((lat_data - lat) ** 2).argmin() for lat in facility_lats])
    dist = haversine_km(facility_lats, facility_lons,
                        np.asarray(lat_data)[iy], np.asarray(lon_data)[ix])
    return ix, iy, dist, dist > DISTANCE_GUARD_KM


def _facility_series(arr, iy, ix, far):
    """Extract the per-facility time series from a (time, lat, lon) array, NaNing
    facilities that failed the distance guard."""
    series = arr[:, iy, ix].astype(float)
    if far.any():
        series[:, far] = np.nan
    return series


def compute_monthly_indices(ds, ds_pr, iy, ix, far, date_keys, facility_names, avg_accum):
    """Build the monthly aggregations for one model from the daily WBGT dataset and
    the (already-monthly) precip dataset. Returns a dict of {var: monthly DataFrame}:
      - monthly mean WBGT         -> monthly[wbgt_day], monthly[wbgt_night]
      - monthly extreme indices   -> monthly[wbgtx_*], monthly[wbgt5x_*]
      - monthly precipitation     -> monthly[precip_month], monthly[precip_5day]
    Also accumulates each facility's mean exposure into `avg_accum` for the covariate table."""
    monthly = {}

    # Daily WBGT -> monthly mean, and extreme indices from the same daily series.
    for var in WBGT_VARS:
        bracket = var.split("_")[-1]  # "day" or "night"
        series = _facility_series(ds[var].values, iy, ix, far)
        daily_df = pd.DataFrame(series, index=date_keys, columns=facility_names)

        monthly[var] = daily_df.groupby(level=0).mean()
        monthly[f"wbgtx_{bracket}"] = daily_df.groupby(level=0).max()
        monthly[f"wbgt5x_{bracket}"] = daily_df.groupby(level=0).apply(
            lambda m: m.rolling(ROLLING_WINDOW_DAYS, min_periods=ROLLING_WINDOW_DAYS).mean().max()
        )
        for j, fac in enumerate(facility_names):
            avg_accum[var][fac].append(np.nanmean(series[:, j]))

    # Precipitation (already monthly on the same grid) aligned to the WBGT months.
    wbgt_month_index = monthly[WBGT_VARS[0]].index
    if ds_pr.sizes[WBGT_TIME_COORD] != len(wbgt_month_index):
        raise AssertionError(
            f"precip has {ds_pr.sizes[WBGT_TIME_COORD]} months, "
            f"WBGT has {len(wbgt_month_index)}")
    for var in PRECIP_VARS:
        series = _facility_series(ds_pr[var].values, iy, ix, far)
        monthly[var] = pd.DataFrame(series, index=wbgt_month_index, columns=facility_names)
        for j, fac in enumerate(facility_names):
            avg_accum[var][fac].append(np.nanmean(series[:, j]))

    return monthly


def long_frame(monthly, var_list, dates, source_name):
    """Melt the requested monthly variables into one long (facility, date, vars) frame."""
    parts = []
    for var in var_list:
        mv = monthly[var].copy()
        mv.index = dates
        mv.index.name = "date"
        parts.append(mv.reset_index().melt(id_vars="date", var_name="facility", value_name=var))
    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on=["date", "facility"])
    df = (df[["facility", "date"] + var_list]
          .sort_values(["facility", "date"]).reset_index(drop=True))
    if df.duplicated(["facility", "date"]).any():
        raise RuntimeError(f"{source_name}: duplicate facility-month rows")
    return df


def process_model(model_id, path, scenario, facility_names, facility_lats, facility_lons, avg_accum):
    """Extract, aggregate and write the mean and extreme panels for one model x scenario."""

    ds = open_cftime_dataset(path)
    missing = [v for v in WBGT_VARS if v not in ds]
    if missing:
        ds.close()
        return

    lat_data = ds[WBGT_LAT_COORD].values
    lon_data = ds[WBGT_LON_COORD].values
    times = ds[WBGT_TIME_COORD].values
    assert_daily(times, path)
    date_keys = get_time_index(times)

    ix, iy, dist, far = nearest_cell_indices(lat_data, lon_data, facility_lats, facility_lons)
    precip_path = path.parent / f"{PRECIP_FILE_PREFIX}{model_id}_{scenario}.nc"
    if not precip_path.exists():
        raise FileNotFoundError(f"missing precip file for {model_id}: {precip_path}")
    ds_pr = open_cftime_dataset(precip_path)
    assert np.array_equal(ds_pr[WBGT_LAT_COORD].values, lat_data), \
        f"{model_id}: precip lat grid differs from WBGT lat grid"
    assert np.array_equal(ds_pr[WBGT_LON_COORD].values, lon_data), \
        f"{model_id}: precip lon grid differs from WBGT lon grid"
    missing_pr = [v for v in PRECIP_VARS if v not in ds_pr]
    if missing_pr:
        raise KeyError(f"{precip_path.name}: {missing_pr} not in {list(ds_pr.data_vars)}")

    monthly = compute_monthly_indices(
        ds, ds_pr, iy, ix, far, date_keys, facility_names, avg_accum
    )
    ds.close()
    ds_pr.close()

    dates = (pd.PeriodIndex(monthly[WBGT_VARS[0]].index, freq="M")
             .to_timestamp(how="end").normalize())

    # Mean panel: monthly-mean WBGT + precip
    mean_df = long_frame(monthly, WBGT_VARS + PRECIP_VARS, dates, path.name)
    out_mean = os.path.join(OUTPUT_DIR, f"wbgt_monthly_mean_facility_{model_id}_{scenario}.csv")
    mean_df.to_csv(out_mean, index=False)

    # Extreme panel: WBGTx / WBGT5x + precip + per-facility lags
    extreme_df = long_frame(monthly, EXTREME_VARS + PRECIP_VARS, dates, path.name)
    for col in EXTREME_VARS:
        for lag in LAG_MONTHS:
            extreme_df[f"{col}_lag{lag}"] = extreme_df.groupby("facility")[col].shift(lag)
    out_extreme = os.path.join(OUTPUT_DIR, f"wbgt_extreme_indices_facility_{model_id}_{scenario}.csv")
    extreme_df.to_csv(out_extreme, index=False)


# ===========================================================================
# Facility covariate table (written once; input to the regression, not TLO)
# ===========================================================================
def build_covariate_table(facilities, avg_accum):
    """Build the one-off facility characteristic table: registry covariates, nearest-
    facility and nearest-clinic distances, and mean exposures. Written transposed to
    facility_info_projection.csv."""
    info = facilities.set_index(FACILITY_NAME_COL)[COVARIATE_COLS].copy()
    info.index.name = "facility"
    info["Dist"] = info["Dist"].replace(DISTRICT_NAME_FIXES)

    coords = info[["A109__Latitude", "A109__Longitude"]].astype(float).values
    dmat = cdist(coords, coords, metric="euclidean")
    np.fill_diagonal(dmat, np.inf)

    min_dist = dmat.min(axis=1)
    info["minimum_distance"] = np.where(np.isfinite(min_dist), min_dist, np.nan)

    is_clinic = (info["Ftype"].astype(str).str.strip() == "Clinic").values
    dmat_clinic = dmat.copy()
    dmat_clinic[:, ~is_clinic] = np.inf
    mind_clinic = dmat_clinic.min(axis=1)
    info["distance_nearest_clinic"] = np.where(np.isfinite(mind_clinic), mind_clinic, np.nan)

    for var in ALL_VARS:
        info[f"average_{var}"] = [
            np.nanmean(avg_accum[var][fac]) if avg_accum[var][fac] else np.nan
            for fac in info.index
        ]

    info = info.T
    info_out = os.path.join(OUTPUT_DIR, "facility_info_projection.csv")
    info.to_csv(info_out)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    facilities, facility_names, facility_lats, facility_lons = load_facilities()

    # Per-facility running mean accumulator for the covariate table
    avg_accum = {var: {fac: [] for fac in facility_names} for var in ALL_VARS}

    for scenario in WBGT_SCENARIOS:
        model_files = find_model_files(WBGT_DIRECTORY, scenario)
        if not model_files:
            raise FileNotFoundError(
                f"No '{WBGT_FILE_PREFIX}*.nc' under {WBGT_DIRECTORY}/*/{scenario}/")

        for model_id, path in model_files:
            process_model(model_id, path, scenario,
                          facility_names, facility_lats, facility_lons, avg_accum)

    build_covariate_table(facilities, avg_accum)


if __name__ == "__main__":
    main()
