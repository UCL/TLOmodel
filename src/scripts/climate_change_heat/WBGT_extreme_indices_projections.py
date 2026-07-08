"""
wbgt_extreme_indices.py

Computes ETCCDI-style climate extreme indices for WBGT, applied to the
day/night bracketed WBGT output.

Indices computed (day and night bracket, separately):

  WBGTx   - Monthly maximum of daily WBGT.
  WBGT5x  - Monthly maximum of the WITHIN-MONTH 5-day rolling mean WBGT.
             The rolling window resets at each month boundary so a month's
             value is driven entirely by that month's days (no cross-boundary
             contamination).

Lagged versions of each index are also computed (configurable via LAG_MONTHS),
so the facility index file is self-contained for the regression.

Three levels of output:

  1. Country-wide: unweighted spatial mean across in-country grid cells.
  2. Facility-level: same indices at the single grid cell nearest each
     health facility, via nearest-neighbour (argmin) matching.
  3. Multi-model: loops over all CMIP6 model folders, writing per-model
     facility CSVs plus an ensemble summary (median, p25, p75).

Outputs
-------
Per model:
  wbgt_extreme_indices_{model}_{scenario}.csv
      Country-wide monthly WBGTx / WBGT5x (day + night).
  wbgt_extreme_indices_facility_{model}_{scenario}.csv
      Facility-level monthly indices + lags, long format.

Ensemble:
  wbgt_extreme_indices_facility_ensemble_{scenario}.csv
      Median, p25, p75 across models per facility-month.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset, num2date
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WBGT_SCENARIO = "ssp245"

WBGT_BASE_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/"
    "Thermofeel_WBGT/NASA_GDDP_CMIP6_Split"
)

# All models to process. Set to None to auto-discover from WBGT_BASE_DIR.
MODELS = None  # or e.g. ["ACCESS-CM2"] for a single-model test run

ROLLING_WINDOW_DAYS = 5

# Lag months to include in the facility index file. These are the prior
# months' index values, shifted forward. From the regression script:
# lags 1-4 capture short-term carryover, lag 9 captures prior-season.
LAG_MONTHS = [1, 2, 3, 4, 9]

MALAWI_BOUNDARY_PATH = Path(
    "/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/mapping/"
    "ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
)

OUTPUT_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices/"
)

FACILITIES_CSV_PATH = Path(
    "/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/"
    "climate_change_impacts/facilities_with_lat_long_region.csv"
)
FACILITY_ID_COL = "Fname"
LAT_COL = "A109__Latitude"
LON_COL = "A109__Longitude"

COMPUTE_FACILITY_LEVEL = True
COMPUTE_COUNTRY_LEVEL = True


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def find_model_files(base_dir, scenario, models=None):
    """Discover wbgt_daynight files under <base>/<model>/<scenario>/."""
    found = []
    if models is None:
        model_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())
    else:
        model_dirs = [base_dir / m for m in models]

    for model_dir in model_dirs:
        if not model_dir.is_dir():
            print(f"  ⚠ {model_dir.name}: directory not found — skipping")
            continue
        scenario_dir = model_dir / scenario
        if not scenario_dir.exists():
            print(f"  ⚠ {model_dir.name}: no '{scenario}' folder — skipping")
            continue
        matches = sorted(scenario_dir.glob("wbgt_daynight_*.nc"))
        if not matches:
            print(f"  ⚠ {model_dir.name}: no wbgt_daynight file — skipping")
            continue
        found.append((model_dir.name, matches[0]))
    return found


def load_daynight_wbgt(nc_path: Path):
    nc = Dataset(nc_path)
    wbgt_day = nc.variables['wbgt_day'][:]
    wbgt_night = nc.variables['wbgt_night'][:]
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]

    time_var = nc.variables['time']
    times = num2date(
        time_var[:], units=time_var.units,
        calendar=getattr(time_var, 'calendar', 'standard')
    )
    return wbgt_day, wbgt_night, lat, lon, times


def _times_to_datetimeindex(times) -> pd.DatetimeIndex:
    """Shared date builder — same across country and facility level."""
    return pd.to_datetime(
        [f"{t.year}-{t.month:02d}-{t.day:02d}" for t in times])


def build_country_mask(lat, lon, boundary_gdf):
    """Boolean 2D mask: grid cell centroids inside Malawi."""
    boundary_union = (boundary_gdf.union_all()
                      if hasattr(boundary_gdf, "union_all")
                      else boundary_gdf.unary_union)
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for i, y in enumerate(lat):
        for j, x in enumerate(lon):
            if boundary_union.contains(Point(x, y)):
                mask[i, j] = True
    return mask


def nearest_grid_index(lat, lon, facility_lat, facility_lon):
    """Argmin nearest-neighbour on squared distance."""
    lat_idx = int(np.argmin((np.asarray(lat) - facility_lat) ** 2))
    lon_idx = int(np.argmin((np.asarray(lon) - facility_lon) ** 2))
    return lat_idx, lon_idx


def monthly_max(daily_series: pd.Series) -> pd.Series:
    """WBGTx: monthly maximum of the daily series."""
    return daily_series.resample("ME").max()


def monthly_max_of_rolling_within_month(daily_series: pd.Series,
                                         window: int) -> pd.Series:
    """WBGT5x: monthly max of within-month rolling mean.

    The rolling window resets at each month boundary, so a month's WBGT5x
    is driven entirely by that month's days — no cross-boundary leakage.
    The first (window-1) days of each month are NaN (no full window yet),
    so they cannot inflate the max.
    """
    return daily_series.groupby(daily_series.index.to_period("M")).apply(
        lambda month: month.rolling(window, min_periods=window).mean().max()
    )


def add_lags(df, index_cols, lag_months):
    """Add lagged versions of each index column, per facility.

    Lags are computed per facility so a facility's lag-1 is its own prior
    month, not a different facility's value. NaN for months where the lag
    reaches before the start of the record.
    """
    for col in index_cols:
        for lag in lag_months:
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df.groupby("facility_id")[col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Country-wide indices
# ---------------------------------------------------------------------------

def compute_country_indices(wbgt_day, wbgt_night, country_mask, times,
                            window):
    """Country-mean daily series -> monthly WBGTx / WBGT5x."""
    dates = _times_to_datetimeindex(times)

    results = {}
    for bracket, data in [("day", wbgt_day), ("night", wbgt_night)]:
        masked = data[:, country_mask]
        daily = pd.Series(np.mean(masked, axis=1), index=dates).sort_index()
        results[f"wbgtx_{bracket}"] = monthly_max(daily)
        results[f"wbgt{window}x_{bracket}"] = \
            monthly_max_of_rolling_within_month(daily, window)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Facility-level indices
# ---------------------------------------------------------------------------

def compute_facility_level_indices(wbgt_day, wbgt_night, lat, lon, times,
                                    window, lag_months):
    """Per-facility monthly WBGTx / WBGT5x (day + night) + lags."""
    facilities = pd.read_csv(FACILITIES_CSV_PATH)
    facilities = facilities.dropna(subset=[LAT_COL, LON_COL])
    dates = _times_to_datetimeindex(times)

    rows = []
    n_facilities = len(facilities)
    for count, (_, row) in enumerate(facilities.iterrows(), start=1):
        facility_id = row[FACILITY_ID_COL]
        lat_idx, lon_idx = nearest_grid_index(
            lat, lon, row[LAT_COL], row[LON_COL])

        fac_data = {}
        for bracket, data in [("day", wbgt_day), ("night", wbgt_night)]:
            daily = pd.Series(data[:, lat_idx, lon_idx], index=dates).sort_index()
            fac_data[f"wbgtx_{bracket}"] = monthly_max(daily)
            fac_data[f"wbgt{window}x_{bracket}"] = \
                monthly_max_of_rolling_within_month(daily, window)

        # Use WBGTx_day's index as the canonical date column
        ref_index = fac_data["wbgtx_day"].index
        facility_df = pd.DataFrame({
            "facility_id": facility_id,
            "date": ref_index,
            **{col: vals.values for col, vals in fac_data.items()},
        })
        rows.append(facility_df)

        if count % 200 == 0 or count == n_facilities:
            print(f"  facility indices: {count}/{n_facilities}")

    result = pd.concat(rows, ignore_index=True)

    # Add lags per facility
    index_cols = [c for c in result.columns
                  if c.startswith("wbgt") and "lag" not in c]
    if lag_months:
        result = add_lags(result, index_cols, lag_months)

    return result


# ---------------------------------------------------------------------------
# Ensemble summary
# ---------------------------------------------------------------------------

def compute_ensemble_summary(all_model_dfs):
    """Median, p25, p75 across models per facility-month."""
    stacked = pd.concat(all_model_dfs, ignore_index=True)
    index_cols = [c for c in stacked.columns
                  if c.startswith("wbgt")]
    grouped = stacked.groupby(["facility_id", "date"])[index_cols]
    summary = grouped.agg(["median", lambda x: x.quantile(0.25),
                           lambda x: x.quantile(0.75)])
    # flatten multi-level column names
    summary.columns = [f"{col}_{stat}" if stat != "median"
                       else f"{col}_median"
                       for col, stat in summary.columns]
    # rename the lambda columns
    summary.columns = [c.replace("<lambda_0>", "p25").replace("<lambda_1>", "p75")
                       for c in summary.columns]
    return summary.reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=ROLLING_WINDOW_DAYS)
    parser.add_argument("--skip-facility-level", action="store_true",
                        default=not COMPUTE_FACILITY_LEVEL)
    parser.add_argument("--skip-country-level", action="store_true",
                        default=not COMPUTE_COUNTRY_LEVEL)
    parser.add_argument("--models", nargs="*", default=MODELS,
                        help="Model names to process (default: all found)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover model files
    model_files = find_model_files(WBGT_BASE_DIR, WBGT_SCENARIO, args.models)
    if not model_files:
        raise FileNotFoundError(
            f"No wbgt_daynight files under {WBGT_BASE_DIR}/*/{WBGT_SCENARIO}/")
    print(f"Models to process: {[m for m, _ in model_files]}")

    # Country mask (shared across models — same Malawi boundary)
    boundary_gdf = None
    country_mask = None
    if not args.skip_country_level:
        print(f"Loading Malawi boundary: {MALAWI_BOUNDARY_PATH}")
        boundary_gdf = gpd.read_file(MALAWI_BOUNDARY_PATH)

    # Per-model loop
    all_facility_dfs = []

    for model_id, nc_path in model_files:
        print(f"\n{'='*60}")
        print(f"  {model_id}")
        print(f"{'='*60}")
        print(f"Loading {nc_path.name}")
        wbgt_day, wbgt_night, lat, lon, times = load_daynight_wbgt(nc_path)
        print(f"  {len(times)} daily timesteps")

        # --- Country-wide ---
        if not args.skip_country_level:
            if country_mask is None:
                country_mask = build_country_mask(lat, lon, boundary_gdf)
                print(f"  {country_mask.sum()}/{country_mask.size} grid cells "
                      "inside Malawi")

            country_df = compute_country_indices(
                wbgt_day, wbgt_night, country_mask, times, args.window)
            csv_path = (args.output_dir /
                        f"wbgt_extreme_indices_{model_id}_{WBGT_SCENARIO}.csv")
            country_df.to_csv(csv_path)
            print(f"  Country-wide saved: {csv_path.name}")

        # --- Facility-level ---
        if not args.skip_facility_level:
            print(f"  Computing facility-level indices + lags {LAG_MONTHS}")
            facility_df = compute_facility_level_indices(
                wbgt_day, wbgt_night, lat, lon, times,
                args.window, LAG_MONTHS)
            facility_df["model"] = model_id

            fac_path = (args.output_dir /
                        f"wbgt_extreme_indices_facility_"
                        f"{model_id}_{WBGT_SCENARIO}.csv")
            facility_df.to_csv(fac_path, index=False)
            print(f"  Facility-level saved: {fac_path.name}")
            print(f"    {facility_df['facility_id'].nunique()} facilities x "
                  f"{facility_df['date'].nunique()} months = "
                  f"{len(facility_df)} rows, "
                  f"{len([c for c in facility_df.columns if 'lag' in c])} "
                  f"lag columns")
            all_facility_dfs.append(facility_df)

    # --- Ensemble summary ---
    if len(all_facility_dfs) > 1:
        print(f"\nComputing ensemble summary across {len(all_facility_dfs)} "
              "models...")
        ensemble = compute_ensemble_summary(all_facility_dfs)
        ens_path = (args.output_dir /
                    f"wbgt_extreme_indices_facility_ensemble_"
                    f"{WBGT_SCENARIO}.csv")
        ensemble.to_csv(ens_path, index=False)
        print(f"Ensemble summary: {ens_path.name}")
        print(f"  {ensemble['facility_id'].nunique()} facilities x "
              f"{ensemble['date'].nunique()} months")

    print("\nDone!")


if __name__ == "__main__":
    main()
