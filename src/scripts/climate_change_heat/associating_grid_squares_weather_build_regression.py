"""
wbgt_facility_panels_all_indicators.py

    match + extract WBGT           -> once, for the union of facilities
    for each indicator:
        slice reporting panel      -> date x facility (that indicator's facilities)
        align WBGT + covariates     -> same facility set/order
        write three files           -> per indicator

Reads the combined wide panel from combine_dhis2_data.py
(one row per facility-month, one column per indicator).

Outputs, per indicator (columns identical and identically ordered across the three):
    historical_{var}_by_facility_{INDICATOR}.csv        date x facility
    monthly_reporting_{INDICATOR}_by_facility_wbgt.csv  date x facility
    expanded_facility_info_wbgt_{INDICATOR}.csv         covariate x facility
    regression_panel_{INDICATOR}.csv                    long: facility x date

Plus, if CF_ENABLED:
    wbgt_monthly_mean_facility_{CF_LABEL}.csv           long: facility x date
    wbgt_extreme_indices_facility_{CF_LABEL}.csv        long: facility x date
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

WBGT_DIRECTORY = ("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/ERA5/")
WBGT_FILE_PREFIX = "wbgt_monthly_"
WBGT_MONTHLY_VARS = ['wbgt_day', 'wbgt_night']     # from wbgt_monthly_*.nc
WBGT_EXTREME_VARS = ['wbgt5x_day']                 # from the extreme file
WBGT_EXTREME_FILE = "wbgt_extreme_indices_ERA5_historical.nc"   # native grid (matches monthly)

WBGT_VARS = WBGT_MONTHLY_VARS + WBGT_EXTREME_VARS   # everything downstream loops over this
WBGT_TIME_COORD = "time"
WBGT_LAT_COORD  = "lat"
WBGT_LON_COORD  = "lon"

# --- Counterfactual (periindustrial ERA5) ----------------------------------
# Reuses matched_facilities + facility_name_mapping from the historical pass:
# facility coords don't change, only the WBGT time series does.
CF_ENABLED           = True
CF_LABEL             = "ERA5_periindustrial_1940_1948"
CF_WBGT_DIRECTORY    = ("/Users/rachelmurray-watson/Documents/Heat_data/"
                        "Thermofeel_WBGT/ERA5/Periindustrial/")
CF_MONTHLY_FILE      = "wbgt_monthly_ERA5_periindustrial_1940_1948.nc"
CF_EXTREME_FILE      = "wbgt_extreme_indices_ERA5_periindustrial_1940_1948.nc"
# If you renamed the periindustrial precip file, update this. Filename only —
# it's joined to CF_WBGT_DIRECTORY below.
CF_PRECIP_FILE = "precip_monthly_ERA5_periindustrial_1940_1948.nc"
CF_MONTHLY_VARS      = WBGT_MONTHLY_VARS       # wbgt_day, wbgt_night
CF_EXTREME_VARS      = WBGT_EXTREME_VARS       # wbgt5x_day
# CF_PRECIP_VARS is set below, after PRECIP_VARS is defined.

# Files
FACILITIES_CSV = ("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/climate_change_impacts/facilities_with_lat_long_region.csv")

FACILITIES_SHP = ("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/climate_change_impacts/facilities_with_districts.shp")
MALAWI_GRID_SHP = ("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/climate_change_impacts/"
                   "Data/malawi_grid.shp")

OUTPUT_DIR = ("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices/")

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
# If these no longer appear in your DHIS2 org-unit names, leave empty and the
# shapefile dependency is never touched.
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
# Load WBGT netCDF (once)
# ---------------------------------------------------------------------------
wbgt_files = sorted(f for f in os.listdir(WBGT_DIRECTORY)
                    if f.startswith(WBGT_FILE_PREFIX) and f.endswith(".nc"))
if not wbgt_files:
    raise FileNotFoundError(f"No '{WBGT_FILE_PREFIX}*.nc' in {WBGT_DIRECTORY}")
if len(wbgt_files) > 1:
    print(f"⚠ {len(wbgt_files)} WBGT files found; using {wbgt_files[0]}")
wbgt_file_path = os.path.join(WBGT_DIRECTORY, wbgt_files[0])
print(f"WBGT data:        {wbgt_file_path}")

ds_wbgt = xr.open_dataset(wbgt_file_path)
missing_vars = [v for v in WBGT_MONTHLY_VARS if v not in ds_wbgt]
if missing_vars:
    raise KeyError(f"{missing_vars} not in {os.path.basename(wbgt_file_path)} "
                   f"(available: {list(ds_wbgt.data_vars)})")
wbgt_data = {var: ds_wbgt[var].values for var in WBGT_MONTHLY_VARS}  # (time, lat, lon)
lat_data = ds_wbgt[WBGT_LAT_COORD].values
long_data = ds_wbgt[WBGT_LON_COORD].values
time_data = pd.to_datetime(ds_wbgt[WBGT_TIME_COORD].values)

# --- Add the extreme index (wbgt5x_day) from the ERA5 extreme file ----------
ds_ext = xr.open_dataset(os.path.join(WBGT_DIRECTORY, WBGT_EXTREME_FILE))
assert ds_ext.sizes.get(WBGT_TIME_COORD) == len(time_data), \
    "extreme file has a different number of months than the monthly-mean file"
for var in WBGT_EXTREME_VARS:
    if var not in ds_ext:
        raise KeyError(f"{var} not in {WBGT_EXTREME_FILE} "
                       f"(has {list(ds_ext.data_vars)})")
    wbgt_data[var] = ds_ext[var].values     # (time, lat, lon), same grid
ds_ext.close()

PRECIP_FILE = "precip_monthly_ERA5_historical.nc"
PRECIP_VARS = ["precip_month", "precip_5day"]
CF_PRECIP_VARS = PRECIP_VARS               # CF carries the same precip vars

ds_precip = xr.open_dataset(os.path.join(WBGT_DIRECTORY, PRECIP_FILE))
assert ds_precip.sizes.get(WBGT_TIME_COORD) == len(time_data), \
    "precip file has a different number of months than the monthly-mean file"
for var in PRECIP_VARS:
    if var not in ds_precip:
        raise KeyError(f"{var} not in {PRECIP_FILE} (has {list(ds_precip.data_vars)})")
    wbgt_data[var] = ds_precip[var].values      # (time, lat, lon), same grid
ds_precip.close()

WBGT_VARS = WBGT_VARS + PRECIP_VARS

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
    """District-rule exposure: WBGT at the malawi_grid cell for that district.
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
            wbgt_by_grid[grid_idx] = {var: wbgt_data[var][:, iy, ix]
                                      for var in WBGT_VARS}
        get_special_case_wbgt._cache = (general_facilities, wbgt_by_grid)
    general_facilities, wbgt_by_grid = get_special_case_wbgt._cache
    grid = general_facilities[
        general_facilities["District"] == district]["Grid_Index"].iloc[0]
    return wbgt_by_grid[grid]


wbgt_data_by_facility = {var: {} for var in WBGT_VARS}
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
        for var in WBGT_VARS:
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

    index_for_x = ((long_data - long_for_facility) ** 2).argmin()
    index_for_y = ((lat_data - lat_for_facility) ** 2).argmin()
    for var in WBGT_VARS:
        wbgt_data_by_facility[var][reporting_facility] = \
            wbgt_data[var][:, index_for_y, index_for_x].tolist()

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
for var in WBGT_VARS:
    df = pd.DataFrame.from_dict(wbgt_data_by_facility[var], orient="index").T
    df.columns = list(wbgt_data_by_facility[var].keys())
    df.index = time_data[: len(df)]
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
    {"Blanytyre": "Blantyre",
     "Nkhatabay": "Nkhata Bay",
     "Mzimba North": "Mzimba",
     "Mzimba South": "Mzimba"})

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

for var in WBGT_VARS:
    facility_info_full[f"average_{var}"] = facility_info_full.index.map(
        {fac: np.nanmean(vals)
         for fac, vals in wbgt_data_by_facility[var].items()})

# ---------------------------------------------------------------------------
# Per-indicator loop: slice reporting panel, align, write three files
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
    wbgt_out = {var: wbgt_dfs_full[var][facs] for var in WBGT_VARS}
    info_out = facility_info_full.loc[facs].T

    # ----------------------------------------------------------------
    # Long-format regression panel
    #   rows = (facility x date), columns = wbgt vars + indicator value
    # ----------------------------------------------------------------
    wbgt_long_parts = []
    for var in WBGT_VARS:
        part = (
            wbgt_out[var]
            .stack(future_stack=True)                     # (date, facility) MultiIndex
            .rename(var)
            .reset_index()
            .rename(columns={"level_1": "facility"})  # date already named
        )
        wbgt_long_parts.append(part.set_index(["facility", "date"]))

    wbgt_long = pd.concat(wbgt_long_parts, axis=1).reset_index()

    indicator_long = (
        reporting_out.stack(future_stack=True).rename(indicator).reset_index().rename(columns={"level_1": "facility"})
    )

    panel = pd.merge(wbgt_long, indicator_long,
                     on=["facility", "date"], how="left")

    static_cols = COVARIATE_COLS + ["minimum_distance"]
    panel = pd.merge(
        panel,
        facility_info_full[static_cols].reset_index(),   # index = facility
        on="facility", how="left"
    )

    panel = panel.sort_values(["facility", "date"]).reset_index(drop=True)

    p_panel = os.path.join(OUTPUT_DIR,
                           f"regression_panel_{indicator}.csv")
    panel.to_csv(p_panel, index=False)

    # --- existing outputs unchanged ---
    for var in WBGT_VARS:
        p = os.path.join(OUTPUT_DIR,
                         f"historical_{var}_by_facility_{indicator}.csv")
        wbgt_out[var].to_csv(p)
    reporting_out.to_csv(os.path.join(
        OUTPUT_DIR, f"monthly_reporting_{indicator}_by_facility_wbgt.csv"))
    info_out.to_csv(os.path.join(
        OUTPUT_DIR, f"expanded_facility_info_wbgt_{indicator}.csv"))

    print(f"  {indicator}: {len(facs)} facilities, "
          f"{reporting_out.shape[0]} reporting months, "
          f"{len(time_data)} WBGT months  -> regression_panel written")

# ---------------------------------------------------------------------------
# Alignment note (WBGT vs reporting month coverage)
# ---------------------------------------------------------------------------
dhis2_months = pd.to_datetime(dhis2[DHIS2_PERIOD_COL]).dt.to_period("M")
wbgt_months = pd.Series(time_data).dt.to_period("M")
if set(dhis2_months) - set(wbgt_months):
    print("\n⚠ Some DHIS2 months have no WBGT coverage — the regression script "
          "aligns panels by row position, so confirm the date ranges match "
          "before flattening.")

print("\nProcessing complete!")
print(f"WBGT extracted once for {len(matched_facilities)} facilities; "
      f"{len(indicator_cols)} indicator file sets written to {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# Counterfactual: extract periindustrial WBGT (+ precip) at the same facility
# locations and write the two long-format CSVs the model script expects.
# ---------------------------------------------------------------------------
if CF_ENABLED:
    print("\n" + "=" * 80)
    print(f"COUNTERFACTUAL EXTRACTION ({CF_LABEL})")
    print("=" * 80)

    cf_monthly_path = os.path.join(CF_WBGT_DIRECTORY, CF_MONTHLY_FILE)
    cf_extreme_path = os.path.join(CF_WBGT_DIRECTORY, CF_EXTREME_FILE)
    cf_precip_path  = os.path.join(CF_WBGT_DIRECTORY, CF_PRECIP_FILE)

    if not (os.path.exists(cf_monthly_path) and os.path.exists(cf_extreme_path)):
        print(f"  ⚠ Missing counterfactual NetCDFs — skipping.")
        print(f"    monthly:  {cf_monthly_path}  exists={os.path.exists(cf_monthly_path)}")
        print(f"    extreme:  {cf_extreme_path}  exists={os.path.exists(cf_extreme_path)}")
    else:
        # ---- load counterfactual NCs and stack vars into one dict ----------
        ds_cf_m = xr.open_dataset(cf_monthly_path)
        ds_cf_x = xr.open_dataset(cf_extreme_path)
        if not os.path.exists(cf_precip_path):
            raise FileNotFoundError(f"CF precip file missing: {cf_precip_path}")
        ds_cf_p = xr.open_dataset(cf_precip_path)

        missing_m = [v for v in CF_MONTHLY_VARS if v not in ds_cf_m]
        missing_x = [v for v in CF_EXTREME_VARS if v not in ds_cf_x]
        missing_p = [v for v in CF_PRECIP_VARS  if v not in ds_cf_p]
        if missing_m or missing_x or missing_p:
            raise KeyError(f"CF vars missing: monthly={missing_m}, "
                           f"extreme={missing_x}, precip={missing_p}")

        cf_lat  = ds_cf_m[WBGT_LAT_COORD].values
        cf_lon  = ds_cf_m[WBGT_LON_COORD].values
        cf_time = pd.to_datetime(ds_cf_m[WBGT_TIME_COORD].values)

        # sanity: precip should share time axis with the monthly means
        if ds_cf_p.sizes.get(WBGT_TIME_COORD) != len(cf_time):
            raise ValueError("CF precip time axis differs from CF monthly time axis")

        cf_data = {v: ds_cf_m[v].values for v in CF_MONTHLY_VARS}
        for v in CF_EXTREME_VARS:
            cf_data[v] = ds_cf_x[v].values
        for v in CF_PRECIP_VARS:
            cf_data[v] = ds_cf_p[v].values

        ds_cf_m.close(); ds_cf_x.close(); ds_cf_p.close()

        # sanity: same grid orientation as historical (this doesn't require
        # identical cells — periindustrial is on the same ERA5 grid — but
        # catches accidental transposes)
        cf_all_vars = CF_MONTHLY_VARS + CF_EXTREME_VARS + CF_PRECIP_VARS
        for v in cf_all_vars:
            if cf_data[v].shape[1:] != (len(cf_lat), len(cf_lon)):
                raise ValueError(f"CF var {v!r} shape {cf_data[v].shape} "
                                 f"inconsistent with ({len(cf_lat)}, {len(cf_lon)})")

        # ---- extract at each matched facility ------------------------------
        # Skip special-case facilities: their historical exposure uses a
        # district grid rule, not lat/lon; keeping counterfactual on the same
        # rule would need a CF-aware grid cache. Small in number, warned below.
        cf_by_facility = {v: {} for v in cf_all_vars}
        skipped_special, skipped_nocoord = [], []

        for reporting_facility in matched_facilities:
            registry_name = facility_name_mapping[reporting_facility]

            if reporting_facility in SPECIAL_CASES:
                skipped_special.append(reporting_facility)
                continue

            frow = facilities_with_lat_long[
                facilities_with_lat_long["Fname"] == registry_name]
            if frow.empty:
                skipped_nocoord.append(reporting_facility)
                continue
            lat_f = frow.iloc[0]["A109__Latitude"]
            lon_f = frow.iloc[0]["A109__Longitude"]
            if pd.isna(lat_f) or pd.isna(lon_f):
                skipped_nocoord.append(reporting_facility)
                continue

            ix = ((cf_lon - lon_f) ** 2).argmin()
            iy = ((cf_lat - lat_f) ** 2).argmin()
            for v in cf_by_facility:
                cf_by_facility[v][reporting_facility] = cf_data[v][:, iy, ix]

        if skipped_special:
            print(f"  ⚠ CF skipped {len(skipped_special)} special-case facilities "
                  f"(district-rule exposure not implemented for CF): "
                  f"{skipped_special}")
        if skipped_nocoord:
            print(f"  ⚠ CF skipped {len(skipped_nocoord)} facilities with no coords "
                  f"in registry (unexpected — they matched historical): "
                  f"{skipped_nocoord[:10]}")

        cf_facilities = list(cf_by_facility[CF_MONTHLY_VARS[0]].keys())
        print(f"  CF extracted for {len(cf_facilities)} facilities x "
              f"{len(cf_time)} months ({cf_time.min():%Y-%m} to {cf_time.max():%Y-%m})")

        # ---- write the two long-format CSVs --------------------------------
        # Long: one row per (facility, date). Precip is written to BOTH files
        # so the model script finds it whether it uses the monthly or extreme
        # branch.
        def _to_long(vars_):
            frames = []
            for v in vars_:
                wide = pd.DataFrame(cf_by_facility[v], index=cf_time)
                wide.index.name = "date"
                long = (wide.stack(future_stack=True)
                            .rename(v)
                            .reset_index()
                            .rename(columns={"level_1": "facility"}))
                frames.append(long.set_index(["facility", "date"]))
            out = pd.concat(frames, axis=1).reset_index()
            return out.sort_values(["facility", "date"]).reset_index(drop=True)

        cf_monthly_long = _to_long(CF_MONTHLY_VARS + CF_PRECIP_VARS)
        cf_extreme_long = _to_long(CF_EXTREME_VARS + CF_PRECIP_VARS)

        p_cf_m = os.path.join(OUTPUT_DIR,
            f"wbgt_monthly_mean_facility_{CF_LABEL}.csv")
        p_cf_x = os.path.join(OUTPUT_DIR,
            f"wbgt_extreme_indices_facility_{CF_LABEL}.csv")
        cf_monthly_long.to_csv(p_cf_m, index=False)
        cf_extreme_long.to_csv(p_cf_x, index=False)
        print(f"  wrote {p_cf_m}")
        print(f"  wrote {p_cf_x}")

ds_wbgt.close()

# ---------------------------------------------------------------------------
# Indicator summary
# ---------------------------------------------------------------------------
def in_scope_denominator(sub, period_col):
    """For each facility, count months in the panel between its first and last
    non-null report for this indicator, and sum across facilities."""
    spans = sub.groupby("facility")[period_col].agg(["min", "max"])
    all_months = dhis2[period_col].drop_duplicates().sort_values().values
    total = 0
    for _, (lo, hi) in spans.iterrows():
        total += int(((all_months >= lo) & (all_months <= hi)).sum())
    return total


print("\n=== INDICATOR SUMMARY ===")
for ind in indicator_cols:
    sub = dhis2[["facility", DHIS2_PERIOD_COL, ind]].dropna(subset=[ind])
    n_facilities = sub["facility"].nunique()
    years = sorted(sub[DHIS2_PERIOD_COL].dt.year.unique())
    total_obs = len(sub)
    possible_obs = in_scope_denominator(sub, DHIS2_PERIOD_COL)
    completeness = round(100 * total_obs / possible_obs, 1) if possible_obs else float("nan")
    print(f"{ind} | facilities={n_facilities} | obs={total_obs:,} | "
          f"in-scope-months={possible_obs:,} | completeness={completeness}% | "
          f"years={years[0]}-{years[-1]}")
