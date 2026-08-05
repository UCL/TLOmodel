"""
cmip6_precip_facility_projection.py

    CMIP6 daily `pr` netCDF  ->  extract at each facility grid cell
                            ->  monthly total (mm/month)
                                monthly Rx5day (max within-month 5-day sum)

Reads combined per-model files from your collate script:
  {PRECIP_DIRECTORY}/pr_day_{model}_{scenario}_malawi_*.nc

Outputs, per model, in the projection block's expected WIDE format
(rows = 'YYYY-M' strings, cols = facility names):

  precip_monthly_total_facility_{model}_{scenario}.csv
  precip_5day_max_facility_{model}_{scenario}.csv

After running, rank models (or reuse model_ranking_{scenario}.csv from
wbgt_extreme_indices.py) and copy the low/median/high tier files to the
projection block's expected filenames, e.g. for tier=highest and model
MPI-ESM1-2-HR:

  cp precip_monthly_total_facility_MPI-ESM1-2-HR_ssp245.csv \\
     ResourceFile_Precipitation_Disruptions_ssp245_highest_monthly_total_weather_by_facility.csv

  cp precip_5day_max_facility_MPI-ESM1-2-HR_ssp245.csv \\
     ResourceFile_Precipitation_Disruptions_ssp245_highest_window_prediction_weather_by_facility.csv

The optional block at the end of this script does that rename automatically
if a ranking file is found.
"""

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# CONFIG (mirrors cmip6_wbgt_facility_projection.py)
# ---------------------------------------------------------------------------
PRECIP_DIRECTORY = ("/Users/rachelmurray-watson/Documents/Heat_data/"
                    "NASA_GDDP-CMIP6/Combined")
SCENARIO         = "ssp245"
FILE_PREFIX      = "pr_day_"       # combined per-model files
PRECIP_VAR       = "pr"
ROLLING_WINDOW_DAYS = 5

# CMIP6 pr: kg m-2 s-1 -> mm/day (1 kg/m2 = 1 mm depth of water)
PR_TO_MM_DAY = 86400.0

# Same distance guard as WBGT projection: CMIP6 downscaled to ~0.25 deg
# (~28 km), so anything >30 km from the nearest cell centre is suspect.
DISTANCE_GUARD_KM = 30.0

FACILITIES_CSV = ("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/"
                  "resources/climate_change_impacts/facilities_with_lat_long_region.csv")
FACILITY_NAME_COL = "Fname"
FACILITY_LAT_COL  = "A109__Latitude"
FACILITY_LON_COL  = "A109__Longitude"

# Point to one historical expanded_facility_info_*.csv to project only for
# the facilities used in the fit; leave None for every registry facility.
RESTRICT_TO_FACILITIES_FILE = None

OUTPUT_DIR = ("/Users/rachelmurray-watson/Documents/Heat_data/"
              "Thermofeel_WBGT/Indices")

# Optional rename step at the end of the script
MODEL_RANKING_CSV = None  # e.g. f"{OUTPUT_DIR}/model_ranking_{SCENARIO}.csv"
TIER_ROLE_MAP = {"low": "lowest", "median": "median", "high": "highest"}


# ---------------------------------------------------------------------------
# Helpers (matching the WBGT projection script's conventions)
# ---------------------------------------------------------------------------
def get_month_periods(time_values):
    """Return a PeriodIndex('M') for cftime OR datetime64 time coords."""
    periods = []
    for t in time_values:
        if hasattr(t, "year") and hasattr(t, "month"):    # cftime.Datetime*
            periods.append(pd.Period(year=t.year, month=t.month, freq="M"))
        else:                                             # numpy.datetime64
            ts = pd.Timestamp(t)
            periods.append(pd.Period(year=ts.year, month=ts.month, freq="M"))
    return pd.PeriodIndex(periods, name="date")


def assert_daily(time_values, nc_path):
    """GUARD: abort if the file's timesteps aren't daily, so a monthly file
    can never be silently summed as if daily."""
    if len(time_values) < 2:
        raise ValueError(f"{Path(nc_path).name}: fewer than 2 timesteps")
    diffs = []
    for i in range(min(len(time_values) - 1, 366)):
        d = time_values[i + 1] - time_values[i]
        diffs.append(d.days if hasattr(d, "days")
                     else d / np.timedelta64(1, "D"))
    med = float(np.median(diffs))
    if med > 5:
        raise ValueError(
            f"{Path(nc_path).name}: median spacing {med:.0f} days - "
            f"expects DAILY {FILE_PREFIX}*.nc, aborting.")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def fmt_period(p):
    """'YYYY-M' — unpadded month, matching the historical precip file's
    index format that _load_precip_wide expects."""
    return f"{p.year}-{p.month}"


# ---------------------------------------------------------------------------
# Load facilities (same as WBGT projection script)
# ---------------------------------------------------------------------------
facilities = pd.read_csv(FACILITIES_CSV).drop_duplicates(FACILITY_NAME_COL)

if RESTRICT_TO_FACILITIES_FILE:
    hist = pd.read_csv(RESTRICT_TO_FACILITIES_FILE, index_col=0)
    keep = set(hist.columns)
    before = len(facilities)
    facilities = facilities[facilities[FACILITY_NAME_COL].isin(keep)]
    print(f"Restricted to {len(facilities)}/{before} fitted facilities")

has_coords = (facilities[FACILITY_LAT_COL].notna()
              & facilities[FACILITY_LON_COL].notna())
n_no_coords = (~has_coords).sum()
if n_no_coords:
    print(f"⚠ {n_no_coords} facilities have no coordinates - dropping")
facilities = facilities[has_coords].reset_index(drop=True)

facility_names = facilities[FACILITY_NAME_COL].tolist()
facility_lats  = facilities[FACILITY_LAT_COL].astype(float).values
facility_lons  = facilities[FACILITY_LON_COL].astype(float).values
print(f"Projecting precip for {len(facility_names)} facilities")

# ---------------------------------------------------------------------------
# Find combined per-model pr files
# ---------------------------------------------------------------------------
pr_dir = Path(PRECIP_DIRECTORY)
if not pr_dir.exists():
    raise FileNotFoundError(f"{pr_dir}: does not exist")

pat = re.compile(
    rf"^{FILE_PREFIX}(?P<model>[A-Za-z0-9\-\.]+)_{SCENARIO}_malawi_.+\.nc$")
pr_files = []
for f in sorted(pr_dir.glob(f"{FILE_PREFIX}*.nc")):
    m = pat.match(f.name)
    if m:
        pr_files.append((m.group("model"), f))

if not pr_files:
    raise FileNotFoundError(
        f"No {FILE_PREFIX}*_{SCENARIO}_malawi_*.nc under {pr_dir}")
print(f"Precip files: {len(pr_files)} models: "
      f"{[m for m, _ in pr_files]}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Per model: extract precip at each facility, compute two indices, save WIDE
# ---------------------------------------------------------------------------
for model_id, path in pr_files:
    print(f"\n--- {model_id} ---")

    # Keep the NoLeap calendar intact — same handling as WBGT script.
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(path, decode_times=time_coder)
    except AttributeError:
        ds = xr.open_dataset(path, use_cftime=True)

    if PRECIP_VAR not in ds:
        print(f"  ⚠ '{PRECIP_VAR}' not in {path.name} "
              f"(has {list(ds.data_vars)}) - skipping")
        ds.close()
        continue

    lat_data = ds["lat"].values
    lon_data = ds["lon"].values
    times    = ds["time"].values

    assert_daily(times, path)                       # GUARD 1
    months = get_month_periods(times)

    # nearest grid index per facility (once per model — grids can differ)
    ix = np.array([((lon_data - lon) ** 2).argmin() for lon in facility_lons])
    iy = np.array([((lat_data - lat) ** 2).argmin() for lat in facility_lats])

    dist = haversine_km(
        facility_lats, facility_lons,
        np.asarray(lat_data)[iy], np.asarray(lon_data)[ix])
    far = dist > DISTANCE_GUARD_KM                  # GUARD 2
    if far.any():
        print(f"  ⚠ distance guard: {int(far.sum())} facilit(y/ies) > "
              f"{DISTANCE_GUARD_KM:.0f} km from nearest cell — set to NaN")
    print(f"  nearest-cell distance: max {dist.max():.1f} km, "
          f"median {np.median(dist):.1f} km")

    # daily nearest-cell series, unit-convert to mm/day
    pr_arr = ds[PRECIP_VAR].values * PR_TO_MM_DAY   # (time, lat, lon)
    series = pr_arr[:, iy, ix].astype(float)        # (time, n_facilities)
    if far.any():
        series[:, far] = np.nan

    ds.close()

    daily = pd.DataFrame(series, index=months, columns=facility_names)
    daily.index.name = "date"

    # (a) monthly total (mm/month): sum of daily mm within month
    monthly_total = daily.groupby(level=0).sum(min_count=1)

    # (b) Rx5day: monthly max of within-month 5-day rolling SUM.
    # Rolling resets at each month boundary — no cross-boundary leakage.
    # First (window-1) days of each month have no full window, so they
    # cannot spuriously inflate the max.
    def _rx5day(month_df):
        return (month_df
                .rolling(ROLLING_WINDOW_DAYS,
                         min_periods=ROLLING_WINDOW_DAYS)
                .sum()
                .max())

    rx5day = daily.groupby(level=0).apply(_rx5day)

    # 'YYYY-M' index (unpadded month), matching historical precip file format
    monthly_total.index = [fmt_period(p) for p in monthly_total.index]
    rx5day.index        = [fmt_period(p) for p in rx5day.index]
    monthly_total.index.name = ""
    rx5day.index.name        = ""

    out_total  = Path(OUTPUT_DIR) / (
        f"precip_monthly_total_facility_{model_id}_{SCENARIO}.csv")
    out_rx5day = Path(OUTPUT_DIR) / (
        f"precip_5day_max_facility_{model_id}_{SCENARIO}.csv")

    monthly_total.to_csv(out_total)
    rx5day.to_csv(out_rx5day)

    print(f"  {monthly_total.shape[1]} facilities × "
          f"{monthly_total.shape[0]} months")
    print(f"  monthly total: mean {monthly_total.stack().mean():.1f} mm/month, "
          f"max {monthly_total.stack().max():.0f}")
    print(f"  Rx5day      : mean {rx5day.stack().mean():.1f} mm, "
          f"max {rx5day.stack().max():.0f}")
    print(f"  Saved: {out_total.name}")
    print(f"  Saved: {out_rx5day.name}")


# ---------------------------------------------------------------------------
# Optional: copy the low / median / high tier files to the projection block's
# expected filenames. Runs only if MODEL_RANKING_CSV is set and exists.
# ---------------------------------------------------------------------------
if MODEL_RANKING_CSV and Path(MODEL_RANKING_CSV).exists():
    print(f"\nRenaming tier files from {Path(MODEL_RANKING_CSV).name} ...")
    rank = pd.read_csv(MODEL_RANKING_CSV)
    if "role" not in rank.columns:
        print(f"  ⚠ 'role' column not found — skipping rename step")
    else:
        tier_reps = rank[rank["role"].isin(TIER_ROLE_MAP)]
        for _, row in tier_reps.iterrows():
            role  = row["role"]
            model = row["model"]
            tier  = TIER_ROLE_MAP[role]

            for src_pattern, dst_pattern in [
                (f"precip_monthly_total_facility_{model}_{SCENARIO}.csv",
                 f"ResourceFile_Precipitation_Disruptions_{SCENARIO}_{tier}"
                 f"_monthly_total_weather_by_facility.csv"),
                (f"precip_5day_max_facility_{model}_{SCENARIO}.csv",
                 f"ResourceFile_Precipitation_Disruptions_{SCENARIO}_{tier}"
                 f"_window_prediction_weather_by_facility.csv"),
            ]:
                src = Path(OUTPUT_DIR) / src_pattern
                dst = Path(OUTPUT_DIR) / dst_pattern
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"  {tier:<7s} <- {model:<20s}: {dst.name}")
                else:
                    print(f"  ⚠ {tier} tier source missing: {src.name}")

print("\nDone.")
