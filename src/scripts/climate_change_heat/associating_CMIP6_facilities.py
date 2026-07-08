"""
cmip6_wbgt_facility_projection.py

    CMIP6 WBGT netCDF  ->  extract at each facility grid cell
    facilities registry ->  coordinates + covariates
Outputs
-------
  projected_{var}_by_facility_{model}.csv   date x facility   (per model, per WBGT var)
  facility_info_projection.csv              covariate x facility   (once)
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
# Directory of per-model CMIP6 WBGT files. Each file = one model's projection.
WBGT_DIRECTORY = ("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6_Split/")
# Files matched by this prefix; the model id is the filename between the
# prefix and '.nc' (adjust EXTRACT_MODEL if your naming differs).
WBGT_FILE_PREFIX = "wbgt_monthly_"

# Current CMIP6 variables (day/night bracketed). One panel written per var.
WBGT_VARS = ["wbgt_day", "wbgt_night"]
WBGT_TIME_COORD = "time"        # CMIP6 convention (NoLeap calendar -> cftime)
WBGT_LAT_COORD = "lat"
WBGT_LON_COORD = "lon"

FACILITIES_CSV = ("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/climate_change_impacts/facilities_with_lat_long_region.csv")
FACILITY_NAME_COL = "Fname"
FACILITY_LAT_COL = "A109__Latitude"
FACILITY_LON_COL = "A109__Longitude"

# Optional: restrict projection to the facilities used in the historical fit,
# for like-for-like comparison. Point this at one historical
# expanded_facility_info_*.csv (its COLUMNS are the fitted facilities);
# leave as None to project for every registry facility with coordinates.
RESTRICT_TO_FACILITIES_FILE = None

OUTPUT_DIR = ("/Users/rachelmurray-watson/Desktop/Climate_change_health/"
              "Data/Temperature_data/WBGT/Projection/")

# Covariates carried through for the fitted model (same as historical
# expanded_facility_info, so the projection is drop-in for the regression).
COVARIATE_COLS = ["Zonename", "Resid", "Dist", "A105", "A109__Altitude",
                  "Ftype", "A109__Latitude", "A109__Longitude"]


# ---------------------------------------------------------------------------
# Time handling — NEX-GDDP-CMIP6 uses a NoLeap (365-day) calendar, so the
# time coordinate comes back as cftime objects, not numpy.datetime64.
# ---------------------------------------------------------------------------
def get_time_index(time_values):
    """Return a monthly 'YYYY-MM' string index for cftime OR datetime64 time
    coordinates (monthly WBGT files: one timestep per month)."""
    keys = []
    for t in time_values:
        if hasattr(t, "year") and hasattr(t, "month"):   # cftime.Datetime*
            keys.append(f"{t.year:04d}-{t.month:02d}")
        else:                                            # numpy.datetime64
            ts = pd.Timestamp(t)
            keys.append(f"{ts.year:04d}-{ts.month:02d}")
    return pd.Index(keys, name="date")


def extract_model_id(filename):
    """Model identifier = filename between the prefix and '.nc'."""
    stem = filename[:-3] if filename.endswith(".nc") else filename
    if stem.startswith(WBGT_FILE_PREFIX):
        stem = stem[len(WBGT_FILE_PREFIX):]
    return stem or "model"


# ---------------------------------------------------------------------------
# Load facility list from the registry (this IS the facility set — no matching)
# ---------------------------------------------------------------------------
facilities = pd.read_csv(FACILITIES_CSV).drop_duplicates(FACILITY_NAME_COL)

# Optionally restrict to the historically-fitted facilities
if RESTRICT_TO_FACILITIES_FILE:
    hist = pd.read_csv(RESTRICT_TO_FACILITIES_FILE, index_col=0)
    keep = set(hist.columns)   # columns of expanded_facility_info = facilities
    before = len(facilities)
    facilities = facilities[facilities[FACILITY_NAME_COL].isin(keep)]
    print(f"Restricted to {len(facilities)}/{before} historically-fitted "
          f"facilities from {Path(RESTRICT_TO_FACILITIES_FILE).name}")

# Drop facilities without usable coordinates
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
model_files = sorted(f for f in os.listdir(WBGT_DIRECTORY)
                     if f.startswith(WBGT_FILE_PREFIX) and f.endswith(".nc"))
if not model_files:
    raise FileNotFoundError(f"No '{WBGT_FILE_PREFIX}*.nc' in {WBGT_DIRECTORY}")
print(f"CMIP6 model files: {len(model_files)} "
      f"({[extract_model_id(f) for f in model_files]})")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Per model: extract projected WBGT at each facility's nearest grid cell
# ---------------------------------------------------------------------------
# Accumulate ensemble-mean average WBGT per var for the covariate table
avg_wbgt_accum = {var: {fac: [] for fac in facility_names} for var in WBGT_VARS}

for filename in model_files:
    model_id = extract_model_id(filename)
    path = os.path.join(WBGT_DIRECTORY, filename)
    print(f"\n--- {model_id} ---")

    # Keep the NoLeap calendar intact rather than erroring. Newer xarray
    # deprecates the use_cftime kwarg in favour of a CFDatetimeCoder, so try
    # the modern path and fall back for older versions.
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(path, decode_times=time_coder)
    except AttributeError:
        ds = xr.open_dataset(path, use_cftime=True)
    missing = [v for v in WBGT_VARS if v not in ds]
    if missing:
        print(f"  ⚠ {missing} not in {filename} "
              f"(has {list(ds.data_vars)}) — skipping this file")
        ds.close()
        continue

    lat_data = ds[WBGT_LAT_COORD].values
    long_data = ds[WBGT_LON_COORD].values
    date_index = get_time_index(ds[WBGT_TIME_COORD].values)

    # nearest grid index per facility (once per model — grids can differ)
    ix = np.array([((long_data - lon) ** 2).argmin() for lon in facility_lons])
    iy = np.array([((lat_data - lat) ** 2).argmin() for lat in facility_lats])

    for var in WBGT_VARS:
        arr = ds[var].values                     # (time, lat, lon)
        # gather all facilities' series at once: (time, n_facilities)
        series = arr[:, iy, ix]
        wbgt_df = pd.DataFrame(series, index=date_index, columns=facility_names)

        out = os.path.join(OUTPUT_DIR,
                           f"projected_{var}_by_facility_{model_id}.csv")
        wbgt_df.to_csv(out)
        print(f"  {var}: {wbgt_df.shape[0]} months x "
              f"{wbgt_df.shape[1]} facilities -> {Path(out).name}")

        for j, fac in enumerate(facility_names):
            avg_wbgt_accum[var][fac].append(np.nanmean(series[:, j]))

    ds.close()

# ---------------------------------------------------------------------------
# Covariate table (once) — same shape/rows as historical expanded_facility_info
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

# ensemble-mean average WBGT per var (descriptive; mirrors historical average_wbgt)
for var in WBGT_VARS:
    info[f"average_{var}"] = [
        np.nanmean(avg_wbgt_accum[var][fac]) if avg_wbgt_accum[var][fac] else np.nan
        for fac in info.index]

info = info.T   # covariate x facility, matching historical orientation
info_out = os.path.join(OUTPUT_DIR, "facility_info_projection.csv")
info.to_csv(info_out)

print(f"\nSaved covariate table -> {Path(info_out).name}")
print("\nProcessing complete!")
print(f"{len(facility_names)} facilities projected across "
      f"{len(model_files)} CMIP6 models; WBGT panels + one covariate table "
      f"written to {OUTPUT_DIR}")
