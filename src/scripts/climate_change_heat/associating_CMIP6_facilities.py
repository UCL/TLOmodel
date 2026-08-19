"""
cmip6_wbgt_facility_projection.py

    CMIP6 WBGT netCDF  ->  extract at each facility grid cell
    facilities registry ->  coordinates + covariates
Outputs
-------
  wbgt_monthly_mean_facility_{model}_{scenario}.csv   long: facility,date,wbgt_day,wbgt_night,precip_month,precip_5day
  facility_info_projection.csv                        covariate x facility   (once)
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
WBGT_DIRECTORY = ("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6_Split/")
WBGT_SCENARIOS =  ["ssp126", "ssp245", "ssp585"]
WBGT_FILE_PREFIX = "wbgt_daynight_"

WBGT_VARS = ["wbgt_day", "wbgt_night"]
PRECIP_VARS = ["precip_month", "precip_5day"]
PRECIP_FILE_PREFIX = "precip_monthly_"
ALL_VARS = WBGT_VARS + PRECIP_VARS

WBGT_TIME_COORD = "time"
WBGT_LAT_COORD = "lat"
WBGT_LON_COORD = "lon"

EXCLUDE_MODELS = {"GISS-E2-1-G"}   # all-NaN WBGT; excluded from ensemble

DISTANCE_GUARD_KM = 30.0

FACILITIES_CSV = ("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/climate_change_impacts/facilities_with_lat_long_region.csv")
FACILITY_NAME_COL = "Fname"
FACILITY_LAT_COL = "A109__Latitude"
FACILITY_LON_COL = "A109__Longitude"

RESTRICT_TO_FACILITIES_FILE = None

OUTPUT_DIR = ("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices")

COVARIATE_COLS = ["Zonename", "Resid", "Dist", "A105", "A109__Altitude",
                  "Ftype", "A109__Latitude", "A109__Longitude"]


# ---------------------------------------------------------------------------
# Time handling — NEX-GDDP-CMIP6 uses a NoLeap (365-day) calendar, so the
# time coordinate comes back as cftime objects, not numpy.datetime64.
# ---------------------------------------------------------------------------
def get_time_index(time_values):
    """Return a monthly 'YYYY-MM' string key per timestep for cftime OR
    datetime64 time coordinates. For DAILY files this yields a repeated month
    key per day, which is exactly the grouping key for the monthly mean."""
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
            print(f"  ⚠ skip {model_dir.name}: in EXCLUDE_MODELS")
            continue
        scenario_dir = model_dir / scenario
        if not scenario_dir.is_dir():
            print(f"  ⚠ skip {model_dir.name}: no directory {scenario_dir}")
            continue
        matches = sorted(scenario_dir.glob(f"{WBGT_FILE_PREFIX}*.nc"))
        if not matches:
            print(f"  ⚠ skip {model_dir.name}: no {WBGT_FILE_PREFIX}*.nc in "
                  f"{scenario_dir}")
            continue
        found.append((model_dir.name, matches[0]))
    return found


# ---------------------------------------------------------------------------
# Load facility list from the registry
# ---------------------------------------------------------------------------
facilities = pd.read_csv(FACILITIES_CSV).drop_duplicates(FACILITY_NAME_COL)

if RESTRICT_TO_FACILITIES_FILE:
    hist = pd.read_csv(RESTRICT_TO_FACILITIES_FILE, index_col=0)
    keep = set(hist.columns)
    before = len(facilities)
    facilities = facilities[facilities[FACILITY_NAME_COL].isin(keep)]
    print(f"Restricted to {len(facilities)}/{before} historically-fitted "
          f"facilities from {Path(RESTRICT_TO_FACILITIES_FILE).name}")

has_coords = facilities[FACILITY_LAT_COL].notna() & facilities[FACILITY_LON_COL].notna()
n_no_coords = (~has_coords).sum()
if n_no_coords:
    print(f"⚠ {n_no_coords} facilities have no coordinates and are dropped")
facilities = facilities[has_coords].reset_index(drop=True)

facility_names = facilities[FACILITY_NAME_COL].tolist()
facility_lats = facilities[FACILITY_LAT_COL].astype(float).values
facility_lons = facilities[FACILITY_LON_COL].astype(float).values
print(f"Projecting for {len(facility_names)} facilities")

# ---------------------------------------------------------------------------
# Find CMIP6 model files
# ---------------------------------------------------------------------------
for WBGT_SCENARIO in WBGT_SCENARIOS:
    model_files = find_model_files(WBGT_DIRECTORY, WBGT_SCENARIO)
    if not model_files:
        raise FileNotFoundError(
            f"No '{WBGT_FILE_PREFIX}*.nc' under {WBGT_DIRECTORY}/*/{WBGT_SCENARIO}/")
    print(f"CMIP6 model files: {len(model_files)} ({[m for m, _ in model_files]})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Per model: extract projected WBGT + precip at each facility's nearest grid cell
# ---------------------------------------------------------------------------
avg_wbgt_accum = {var: {fac: [] for fac in facility_names} for var in ALL_VARS}

for WBGT_SCENARIO in WBGT_SCENARIOS:
    for model_id, path in model_files:
        print(f"\n--- {model_id} ---")

        try:
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            ds = xr.open_dataset(path, decode_times=time_coder)
        except AttributeError:
            ds = xr.open_dataset(path, use_cftime=True)
        missing = [v for v in WBGT_VARS if v not in ds]
        if missing:
            print(f"  ⚠ {missing} not in {path.name} "
                  f"(has {list(ds.data_vars)}) — skipping this file")
            ds.close()
            continue

        lat_data = ds[WBGT_LAT_COORD].values
        long_data = ds[WBGT_LON_COORD].values
        times = ds[WBGT_TIME_COORD].values

        assert_daily(times, path)
        date_keys = get_time_index(times)

        ix = np.array([((long_data - lon) ** 2).argmin() for lon in facility_lons])
        iy = np.array([((lat_data - lat) ** 2).argmin() for lat in facility_lats])

        dist = haversine_km(facility_lats, facility_lons,
                            np.asarray(lat_data)[iy], np.asarray(long_data)[ix])
        far = dist > DISTANCE_GUARD_KM
        if far.any():
            print(f"  ⚠ distance guard: {int(far.sum())} facilit(y/ies) > "
                  f"{DISTANCE_GUARD_KM:.0f} km from any cell — set to NaN:")
            for k in np.where(far)[0]:
                print(f"      {facility_names[k]:<40s} {dist[k]:6.1f} km")
        print(f"  nearest-cell distance: max {dist.max():.1f} km, "
              f"median {np.median(dist):.1f} km")

        # Open the matching precip file once (already monthly, same grid)
        precip_path = path.parent / f"{PRECIP_FILE_PREFIX}{model_id}_{WBGT_SCENARIO}.nc"
        if not precip_path.exists():
            raise FileNotFoundError(f"missing precip file for {model_id}: {precip_path}")
        try:
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            ds_pr = xr.open_dataset(precip_path, decode_times=time_coder)
        except AttributeError:
            ds_pr = xr.open_dataset(precip_path, use_cftime=True)

        assert np.array_equal(ds_pr[WBGT_LAT_COORD].values, lat_data), \
            f"{model_id}: precip lat grid differs from WBGT lat grid"
        assert np.array_equal(ds_pr[WBGT_LON_COORD].values, long_data), \
            f"{model_id}: precip lon grid differs from WBGT lon grid"
        missing_pr = [v for v in PRECIP_VARS if v not in ds_pr]
        if missing_pr:
            raise KeyError(f"{precip_path.name}: {missing_pr} not in "
                           f"{list(ds_pr.data_vars)}")

        monthly = {}

        # daily WBGT -> monthly mean
        for var in WBGT_VARS:
            arr = ds[var].values
            series = arr[:, iy, ix].astype(float)
            if far.any():
                series[:, far] = np.nan
            daily_df = pd.DataFrame(series, index=date_keys, columns=facility_names)
            monthly[var] = daily_df.groupby(level=0).mean()
            for j, fac in enumerate(facility_names):
                avg_wbgt_accum[var][fac].append(np.nanmean(series[:, j]))

        # monthly precip — already monthly, no groupby
        wbgt_month_index = monthly[WBGT_VARS[0]].index
        if ds_pr.sizes[WBGT_TIME_COORD] != len(wbgt_month_index):
            raise AssertionError(
                f"{model_id}: precip has {ds_pr.sizes[WBGT_TIME_COORD]} months, "
                f"WBGT has {len(wbgt_month_index)}")

        for var in PRECIP_VARS:
            arr = ds_pr[var].values
            series = arr[:, iy, ix].astype(float)
            if far.any():
                series[:, far] = np.nan
            monthly[var] = pd.DataFrame(series, index=wbgt_month_index,
                                        columns=facility_names)
            for j, fac in enumerate(facility_names):
                avg_wbgt_accum[var][fac].append(np.nanmean(series[:, j]))

        ds.close()
        ds_pr.close()

        # assemble long: facility, date, wbgt_day, wbgt_night, precip_month, precip_5day
        dates = (pd.PeriodIndex(monthly[WBGT_VARS[0]].index, freq="M")
                 .to_timestamp(how="end").normalize())
        parts = []
        for var in ALL_VARS:
            mv = monthly[var].copy()
            mv.index = dates
            mv.index.name = "date"
            parts.append(mv.reset_index().melt(id_vars="date",
                                               var_name="facility", value_name=var))
        wbgt_df = parts[0]
        for p in parts[1:]:
            wbgt_df = wbgt_df.merge(p, on=["date", "facility"])
        wbgt_df = (wbgt_df[["facility", "date"] + ALL_VARS]
                   .sort_values(["facility", "date"]).reset_index(drop=True))
        if wbgt_df.duplicated(["facility", "date"]).any():
            raise RuntimeError(f"{path.name}: duplicate facility-month rows")

        out = os.path.join(OUTPUT_DIR,
                           f"wbgt_monthly_mean_facility_{model_id}_{WBGT_SCENARIO}.csv")
        wbgt_df.to_csv(out, index=False)
        print(f"  {wbgt_df['facility'].nunique()} facilities x "
              f"{wbgt_df['date'].nunique()} months -> {Path(out).name}")

# ---------------------------------------------------------------------------
# Covariate table (once)
# ---------------------------------------------------------------------------
info = facilities.set_index(FACILITY_NAME_COL)[COVARIATE_COLS].copy()
info.index.name = "facility"
info["Dist"] = info["Dist"].replace({"Blanytyre": "Blantyre",
                                     "Nkhatabay": "Nkhata Bay"})

coords = info[["A109__Latitude", "A109__Longitude"]].astype(float).values
dmat = cdist(coords, coords, metric="euclidean")
np.fill_diagonal(dmat, np.inf)
min_dist = dmat.min(axis=1)
info["minimum_distance"] = np.where(np.isfinite(min_dist), min_dist, np.nan)

for var in ALL_VARS:
    info[f"average_{var}"] = [
        np.nanmean(avg_wbgt_accum[var][fac]) if avg_wbgt_accum[var][fac] else np.nan
        for fac in info.index]

info = info.T
info_out = os.path.join(OUTPUT_DIR, "facility_info_projection.csv")
info.to_csv(info_out)

print(f"\nSaved covariate table -> {Path(info_out).name}")
print("\nProcessing complete!")
print(f"{len(facility_names)} facilities projected across "
      f"{len(model_files)} CMIP6 models; WBGT + precip panels + one covariate "
      f"table written to {OUTPUT_DIR}")
