"""
wbgt_facility_panels_all_indicators.py

    match + extract WBGT           -> once, for the union of facilities
    for each indicator:
        slice reporting panel      -> date x facility (that indicator's facilities)
        align WBGT + covariates     -> same facility set/order
        write three files           -> per indicator

Reads the combined wide panel from combine_dhis2_data.py
(one row per facility-month, one column per indicator).

WBGT sources (all ERA5, historical — the fit stays entirely on ERA5):
    means    : wbgt_monthly_ERA5_historical.nc          -> wbgt_day, wbgt_night
    extremes : wbgt_extreme_indices_ERA5_historical.nc  -> wbgt5x_day, ...
The extreme index is extracted here so it becomes a native column in
regression_panel_{INDICATOR}.csv (no downstream merge, no CMIP6 files).
CMIP6/ssp extreme indices belong only to the PROJECTION stage.

Outputs, per indicator (columns identical and identically ordered across the three):
    historical_{var}_by_facility_{INDICATOR}.csv        date x facility
    monthly_reporting_{INDICATOR}_by_facility_wbgt.csv  date x facility
    expanded_facility_info_wbgt_{INDICATOR}.csv         covariate x facility
    regression_panel_{INDICATOR}.csv                    long: facility x date
"""

import difflib
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Combined wide panel from combine_dhis2_data.py:
#   columns = [facility, period_parsed, <one column per indicator>]
DHIS2_WIDE_PATH = ("/Users/rachelmurray-watson/Documents/Heat_data/"
                   "DHIS2_Malawi/combined/dhis2_panel_wide_facility_month.csv")
DHIS2_FACILITY_COL = "facility"
DHIS2_PERIOD_COL = "period_parsed"
# Which indicator columns to process. None = every column except the two
# structural columns above. Set a list to restrict, e.g. ["anc1_coverage"].
INDICATORS = None

WBGT_DIRECTORY = ("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/ERA5")
WBGT_FILE_PREFIX = "wbgt_monthly_"
WBGT_VARS = ['wbgt_day', 'wbgt_night']    # monthly MEAN vars
WBGT_TIME_COORD = "time"
WBGT_LAT_COORD = "lat"
WBGT_LON_COORD = "lon"

# --- ERA5 extreme indices: adds wbgt5x_day (and any others) to the panel ----
# Same ERA5 source as the means, so the regression stays consistent. The
# variable names below MUST match what's inside the file — the script prints
# the available names on load, so run once and correct WBGT_EXTREME_VARS if
# the KeyError fires.
WBGT_EXTREME_FILE = "wbgt_extreme_indices_ERA5_historical.nc"   # in WBGT_DIRECTORY
WBGT_EXTREME_VARS = ["wbgt5x_day"]        # e.g. + "wbgtx_day", "wbgt5x_night"
# Coord names inside the extreme file (default: same as the means file)
WBGT_EXTREME_TIME_COORD = WBGT_TIME_COORD
WBGT_EXTREME_LAT_COORD = WBGT_LAT_COORD
WBGT_EXTREME_LON_COORD = WBGT_LON_COORD

FACILITIES_CSV = ("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/climate_change_impacts/facilities_with_lat_long_region.csv")

FACILITIES_SHP = ("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/climate_change_impacts/facilities_with_districts.shp")
MALAWI_GRID_SHP = ("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/climate_change_impacts/"
                   "Data/malawi_grid.shp")

OUTPUT_DIR = ("/Users/rachelmurray-watson/Documents/Heat_data/All_predictors_processed/")

COVARIATE_COLS = ["Zonename", "Resid", "Dist", "A105", "A109__Altitude",
                  "Ftype", "A109__Latitude", "A109__Longitude"]

FUZZY_CUTOFF = 0.70

# Stub — populate from real mismatches in the unmatched list, not assumptions.
ABBREVIATIONS = {
    r"\bhc\b": "health centre",
    r"\bh/c\b": "health centre",
    r"\bdist\b": "district",
    r"\bhosp\b": "hospital",
}

# Reporting names resolved by district rule rather than name matching.
SPECIAL_CASES = {
    "Central East Zone": "Nkhotakota",
    "Central Hospital": "Lilongwe City",
}

# ---------------------------------------------------------------------------
# Load combined DHIS2 wide panel
# ---------------------------------------------------------------------------
print(f"DHIS2 wide panel: {DHIS2_WIDE_PATH}")
dhis2 = pd.read_csv(DHIS2_WIDE_PATH)
for col in (DHIS2_FACILITY_COL, DHIS2_PERIOD_COL):
    if col not in dhis2.columns:
        raise KeyError(f"'{col}' not in wide panel (found {list(dhis2.columns)[:8]}...)")

dhis2[DHIS2_PERIOD_COL] = pd.to_datetime(dhis2[DHIS2_PERIOD_COL], errors="coerce")
if dhis2[DHIS2_PERIOD_COL].isna().any():
    raise ValueError("Some period values didn't parse as dates in the wide panel")

indicator_cols = ([c for c in dhis2.columns
                   if c not in (DHIS2_FACILITY_COL, DHIS2_PERIOD_COL)]
                  if INDICATORS is None else list(INDICATORS))
missing = [c for c in indicator_cols if c not in dhis2.columns]
if missing:
    raise KeyError(f"Requested indicators not in panel: {missing}")
print(f"Indicators to process ({len(indicator_cols)}): {indicator_cols}")

# Union of all reporting facilities (matching is done on this set, once)
all_reporting_facilities = dhis2[DHIS2_FACILITY_COL].dropna().unique().tolist()
print(f"Reporting facilities (union across indicators): {len(all_reporting_facilities)}")

# ---------------------------------------------------------------------------
# Load WBGT netCDF — monthly MEANS (once)
# ---------------------------------------------------------------------------
wbgt_files = sorted(f for f in os.listdir(WBGT_DIRECTORY)
                    if f.startswith(WBGT_FILE_PREFIX) and f.endswith(".nc"))
if not wbgt_files:
    raise FileNotFoundError(f"No '{WBGT_FILE_PREFIX}*.nc' in {WBGT_DIRECTORY}")
if len(wbgt_files) > 1:
    print(f"⚠ {len(wbgt_files)} WBGT files found; using {wbgt_files[0]}")
wbgt_file_path = os.path.join(WBGT_DIRECTORY, wbgt_files[0])
print(f"WBGT means:       {wbgt_file_path}")

ds_wbgt = xr.open_dataset(wbgt_file_path)
missing_vars = [v for v in WBGT_VARS if v not in ds_wbgt]
if missing_vars:
    raise KeyError(f"{missing_vars} not in file "
                   f"(available: {list(ds_wbgt.data_vars)}) — update WBGT_VARS")
wbgt_data = {var: ds_wbgt[var].values for var in WBGT_VARS}  # (time, lat, lon)
lat_data = ds_wbgt[WBGT_LAT_COORD].values
long_data = ds_wbgt[WBGT_LON_COORD].values
time_data = pd.to_datetime(ds_wbgt[WBGT_TIME_COORD].values)

# ---------------------------------------------------------------------------
# Load WBGT netCDF — ERA5 EXTREME indices (once)
# ---------------------------------------------------------------------------
ext_path = os.path.join(WBGT_DIRECTORY, WBGT_EXTREME_FILE)
print(f"WBGT extremes:    {ext_path}")
ds_ext = xr.open_dataset(ext_path)
print(f"  available extreme vars: {list(ds_ext.data_vars)}")
missing_ext = [v for v in WBGT_EXTREME_VARS if v not in ds_ext]
if missing_ext:
    raise KeyError(f"{missing_ext} not in {WBGT_EXTREME_FILE} "
                   f"(available: {list(ds_ext.data_vars)}) — update WBGT_EXTREME_VARS")
ext_data = {v: ds_ext[v].values for v in WBGT_EXTREME_VARS}
for v in WBGT_EXTREME_VARS:                       # expect (time, lat, lon)
    if ext_data[v].ndim != 3:
        raise ValueError(f"'{v}' has shape {ext_data[v].shape}; expected 3-D "
                         "(time, lat, lon). Check the extreme file's layout.")
ext_lat = ds_ext[WBGT_EXTREME_LAT_COORD].values
ext_lon = ds_ext[WBGT_EXTREME_LON_COORD].values
ext_time = pd.to_datetime(ds_ext[WBGT_EXTREME_TIME_COORD].values)
print(f"  extreme grid {ext_data[WBGT_EXTREME_VARS[0]].shape}, "
      f"{ext_time.min():%Y-%m} to {ext_time.max():%Y-%m}")

# All WBGT variables, and the per-source (grid + time) extraction definitions.
ALL_VARS = WBGT_VARS + WBGT_EXTREME_VARS
WBGT_SOURCES = [
    dict(data=wbgt_data, lat=lat_data, lon=long_data, time=time_data),  # means
    dict(data=ext_data,  lat=ext_lat,  lon=ext_lon,   time=ext_time),   # extremes
]
VAR_TIME = {}
for src in WBGT_SOURCES:
    for v in src["data"]:
        VAR_TIME[v] = src["time"]


def nearest_series(data_dict, lat_arr, lon_arr, fac_lat, fac_lon):
    """Nearest grid cell (argmin on lat and lon of THIS source's grid)."""
    iy = int(((np.asarray(lat_arr) - fac_lat) ** 2).argmin())
    ix = int(((np.asarray(lon_arr) - fac_lon) ** 2).argmin())
    return {v: data_dict[v][:, iy, ix].tolist() for v in data_dict}


# ---------------------------------------------------------------------------
# Match facilities + extract WBGT — ONCE, for the union of facilities
# ---------------------------------------------------------------------------
facilities_with_lat_long = pd.read_csv(FACILITIES_CSV)


def clean_name(name):
    """Lowercase, strip parenthetical suffixes, expand abbreviations with
    word-boundary regex, drop stray punctuation."""
    name = str(name).lower().strip()
    name = re.sub(r"\s*\([^)]*\)", "", name)
    name = re.sub(r"[.,]", "", name)
    name = re.sub(r"\s+", " ", name)
    for pattern, expansion in ABBREVIATIONS.items():
        name = re.sub(pattern, expansion, name)
    return name.strip()


facilities_clean = {clean_name(f): f
                    for f in facilities_with_lat_long["Fname"].dropna().unique()}
print(f"\nRegistry facilities: {len(facilities_clean)}")


def get_special_case_wbgt(district):
    """District-rule exposure: WBGT at the malawi_grid cell for that district,
    for BOTH mean and extreme variables (each read off its own source grid).
    Shapefiles are loaded lazily, only if a special case is present."""
    import geopandas as gpd
    if not hasattr(get_special_case_wbgt, "_cache"):
        general_facilities = gpd.read_file(FACILITIES_SHP)
        malawi_grid = gpd.read_file(MALAWI_GRID_SHP)
        wbgt_by_grid = {}
        for grid_idx, polygon in enumerate(malawi_grid["geometry"]):
            minx, miny, maxx, maxy = polygon.bounds
            ix = ((long_data - minx) ** 2).argmin()
            iy = ((lat_data - miny) ** 2).argmin()
            cell = {var: wbgt_data[var][:, iy, ix] for var in WBGT_VARS}
            # extreme vars off the extreme grid (nearest cell to same corner)
            ex = ((ext_lon - minx) ** 2).argmin()
            ey = ((ext_lat - miny) ** 2).argmin()
            cell.update({v: ext_data[v][:, ey, ex] for v in WBGT_EXTREME_VARS})
            wbgt_by_grid[grid_idx] = cell
        get_special_case_wbgt._cache = (general_facilities, wbgt_by_grid)
    general_facilities, wbgt_by_grid = get_special_case_wbgt._cache
    grid = general_facilities[
        general_facilities["District"] == district]["Grid_Index"].iloc[0]
    return wbgt_by_grid[grid]


wbgt_data_by_facility = {var: {} for var in ALL_VARS}
matched_facilities = []            # matched, in first-seen order
facility_name_mapping = {}         # reporting name -> registry name (or itself)
unmatched_facilities = []
match_stats = {"exact": 0, "fuzzy": 0, "special_case": 0, "failed": 0}

print("\n" + "=" * 80)
print("MATCHING FACILITIES (once, for all indicators)")
print("=" * 80)

for reporting_facility in all_reporting_facilities:

    if reporting_facility in SPECIAL_CASES:
        district = SPECIAL_CASES[reporting_facility]
        grid_wbgt = get_special_case_wbgt(district)
        for var in ALL_VARS:
            wbgt_data_by_facility[var][reporting_facility] = grid_wbgt[var].tolist()
        matched_facilities.append(reporting_facility)
        facility_name_mapping[reporting_facility] = reporting_facility
        match_stats["special_case"] += 1
        print(f"★ SPECIAL: '{reporting_facility}' -> grid rule ({district})")
        continue

    reporting_clean = clean_name(reporting_facility)
    original_facility_name = None

    if reporting_clean in facilities_clean:
        original_facility_name = facilities_clean[reporting_clean]
        match_stats["exact"] += 1
    else:
        close = difflib.get_close_matches(
            reporting_clean, facilities_clean.keys(), n=1, cutoff=FUZZY_CUTOFF)
        if close:
            original_facility_name = facilities_clean[close[0]]
            match_stats["fuzzy"] += 1
            print(f"≈ FUZZY: '{reporting_facility}' -> '{original_facility_name}'")

    if original_facility_name is None:
        unmatched_facilities.append(reporting_facility)
        match_stats["failed"] += 1
        continue

    facility_row = facilities_with_lat_long[
        facilities_with_lat_long["Fname"] == original_facility_name].iloc[0]
    lat_for_facility = facility_row["A109__Latitude"]
    long_for_facility = facility_row["A109__Longitude"]

    if pd.isna(lat_for_facility) or pd.isna(long_for_facility):
        unmatched_facilities.append(reporting_facility)
        match_stats["failed"] += 1
        continue

    matched_facilities.append(reporting_facility)
    facility_name_mapping[reporting_facility] = original_facility_name

    # Extract every WBGT variable from its own source grid (means + extremes)
    for src in WBGT_SOURCES:
        series = nearest_series(src["data"], src["lat"], src["lon"],
                                lat_for_facility, long_for_facility)
        for v, s in series.items():
            wbgt_data_by_facility[v][reporting_facility] = s

print(f"\nMatched {len(matched_facilities)}/{len(all_reporting_facilities)} "
      f"(exact {match_stats['exact']}, fuzzy {match_stats['fuzzy']}, "
      f"special {match_stats['special_case']}); failed {match_stats['failed']}")
if unmatched_facilities:
    print(f"Unmatched ({len(unmatched_facilities)}): {unmatched_facilities[:20]}"
          + (" ..." if len(unmatched_facilities) > 20 else ""))

# ---------------------------------------------------------------------------
# Build full WBGT panels + covariate table — ONCE, for all matched facilities
# ---------------------------------------------------------------------------
wbgt_dfs_full = {}
for var in ALL_VARS:
    df = pd.DataFrame.from_dict(wbgt_data_by_facility[var], orient="index").T
    df.columns = list(wbgt_data_by_facility[var].keys())
    tv = VAR_TIME[var]
    df.index = tv[: len(df)]                 # each var uses its own time axis
    df.index.name = "date"
    wbgt_dfs_full[var] = df[matched_facilities]   # consistent column order

registry_indexed = facilities_with_lat_long.drop_duplicates(
    "Fname").set_index("Fname")

info_rows = []
for reporting_facility in matched_facilities:
    registry_name = facility_name_mapping[reporting_facility]
    if registry_name in registry_indexed.index:
        row = registry_indexed.loc[registry_name, COVARIATE_COLS].copy()
    else:
        row = pd.Series(index=COVARIATE_COLS, dtype=object)
    row.name = reporting_facility
    info_rows.append(row)

facility_info_full = pd.DataFrame(info_rows)
facility_info_full.index.name = "facility"
facility_info_full["Dist"] = facility_info_full["Dist"].replace(
    {"Blanytyre": "Blantyre", "Nkhatabay": "Nkhata Bay"})

# minimum distance uses the FULL matched set (nearest neighbour among all
# matched facilities, not just one indicator's reporters)
coords = facility_info_full[
    ["A109__Latitude", "A109__Longitude"]].astype(float).values
dmat = cdist(coords, coords, metric="euclidean")
np.fill_diagonal(dmat, np.inf)
dmat[np.isnan(dmat)] = np.inf
min_dist = dmat.min(axis=1)
facility_info_full["minimum_distance"] = np.where(
    np.isfinite(min_dist), min_dist, np.nan)

for var in ALL_VARS:
    facility_info_full[f"average_{var}"] = facility_info_full.index.map(
        {fac: np.nanmean(vals)
         for fac, vals in wbgt_data_by_facility[var].items()})

# ---------------------------------------------------------------------------
# Per-indicator loop: slice reporting panel, align, write files
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
matched_set = set(matched_facilities)

print("\n" + "=" * 80)
print("WRITING PER-INDICATOR FILE SETS")
print("=" * 80)
for indicator in indicator_cols:
    # Pivot this indicator to date x facility from the long wide panel
    reporting_wide = dhis2.pivot_table(
        index=DHIS2_PERIOD_COL, columns=DHIS2_FACILITY_COL,
        values=indicator, aggfunc="first").sort_index()
    reporting_wide.index.name = "date"

    reports_this = reporting_wide.columns[
        reporting_wide.notna().any(axis=0)]
    facs = [f for f in matched_facilities
            if f in matched_set and f in set(reports_this)]

    if not facs:
        print(f"  {indicator}: no matched facilities report this — skipping")
        continue

    reporting_out = reporting_wide[facs]
    wbgt_out = {var: wbgt_dfs_full[var][facs] for var in ALL_VARS}
    info_out = facility_info_full.loc[facs].T

    # ----------------------------------------------------------------
    # Long-format regression panel
    #   rows = (facility x date), columns = wbgt means + extremes + indicator
    # Dates are normalised to month-start so mean/extreme/indicator align on
    # month regardless of each source's day-of-month convention.
    # ----------------------------------------------------------------
    def _to_month_start(s):
        return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()

    wbgt_long_parts = []
    for var in ALL_VARS:
        part = (
            wbgt_out[var]
            .stack(future_stack=True)                 # (date, facility) MultiIndex
            .rename(var)
            .reset_index()
            .rename(columns={"level_1": "facility"})  # date already named
        )
        part["date"] = _to_month_start(part["date"])
        wbgt_long_parts.append(part.set_index(["facility", "date"]))

    wbgt_long = pd.concat(wbgt_long_parts, axis=1).reset_index()

    indicator_long = (
        reporting_out.stack(future_stack=True)
        .rename(indicator).reset_index().rename(columns={"level_1": "facility"})
    )
    indicator_long["date"] = _to_month_start(indicator_long["date"])

    panel = pd.merge(wbgt_long, indicator_long,
                     on=["facility", "date"], how="left")

    static_cols = COVARIATE_COLS + ["minimum_distance"]
    panel = pd.merge(
        panel,
        facility_info_full[static_cols].reset_index(),   # index = facility
        on="facility", how="left"
    )

    panel = panel.sort_values(["facility", "date"]).reset_index(drop=True)

    panel.to_csv(os.path.join(OUTPUT_DIR,
                              f"regression_panel_{indicator}.csv"), index=False)

    # --- wide outputs (now include the extreme vars too) ---
    for var in ALL_VARS:
        wbgt_out[var].to_csv(os.path.join(
            OUTPUT_DIR, f"historical_{var}_by_facility_{indicator}.csv"))
    reporting_out.to_csv(os.path.join(
        OUTPUT_DIR, f"monthly_reporting_{indicator}_by_facility_wbgt.csv"))
    info_out.to_csv(os.path.join(
        OUTPUT_DIR, f"expanded_facility_info_wbgt_{indicator}.csv"))

    print(f"  {indicator}: {len(facs)} facilities, "
          f"{reporting_out.shape[0]} reporting months, "
          f"vars={ALL_VARS}  -> regression_panel written")

# ---------------------------------------------------------------------------
# Alignment note (WBGT vs reporting month coverage)
# ---------------------------------------------------------------------------
dhis2_months = pd.to_datetime(dhis2[DHIS2_PERIOD_COL]).dt.to_period("M")
mean_months = pd.Series(time_data).dt.to_period("M")
ext_months = pd.Series(ext_time).dt.to_period("M")
if set(dhis2_months) - set(mean_months):
    print("\n⚠ Some DHIS2 months have no WBGT-mean coverage.")
if set(dhis2_months) - set(ext_months):
    print("⚠ Some DHIS2 months have no WBGT-extreme coverage — wbgt5x_day will "
          "be NaN there and those rows drop out of the regression sample.")

print("\nProcessing complete!")
print(f"WBGT (means + extremes) extracted once for {len(matched_facilities)} "
      f"facilities; {len(indicator_cols)} indicator file sets written to {OUTPUT_DIR}")

ds_wbgt.close()
ds_ext.close()
