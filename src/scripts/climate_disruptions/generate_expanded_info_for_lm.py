import difflib
import os
import re
import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial.distance import cdist
from collections import defaultdict

# Data accessed from https://dhis2.health.gov.mw/dhis-web-data-visualizer/#/YiQK65skxjz
# Reporting rate is expected reporting vs actual reporting
ANC = True
Inpatient = False
multiplier = 1000
baseline = False
baseline_all_years = False

if ANC:
    reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_Change_Health/Data/ANC_data/ANC_data_2011_2024.csv')
elif Inpatient:
    reporting_data = pd.read_csv(
        '/Users/rem76/Desktop/Climate_Change_Health/Data/Inpatient_Data/HMIS_Total_Number_Admissions.csv')
else:
    reporting_data = pd.read_csv(
        '/Users/rem76/Desktop/Climate_Change_Health/Data/Reporting_Rate/Reporting_Rate_by_smaller_facilities_2011_2024.csv')

# Drop NAs
reporting_data = reporting_data.dropna(subset=reporting_data.columns[3:], how='all')

### Aggregate over months
monthly_reporting_data_by_facility = {}
if ANC:
    months = set(col.split("HMIS Total Antenatal Visits ")[1] for col in reporting_data.columns if
                 "HMIS Total Antenatal Visits " in col)
elif Inpatient:
    months = set(col.split("HMIS Total # of Admissions (including Maternity) ")[1] for col in reporting_data.columns if
                 "HMIS Total # of Admissions (including Maternity) " in col)
else:
    months = set(col.split(" - Reporting rate ")[1] for col in reporting_data.columns if " - Reporting rate " in col)

months = [date.strip() for date in months]
dates = pd.to_datetime(months, format='%B %Y', errors='coerce')
months = dates.sort_values().strftime('%B %Y').tolist()

for month in months:
    columns_of_interest_all_metrics = [reporting_data.columns[1]] + reporting_data.columns[
        reporting_data.columns.str.endswith(month)].tolist()
    data_of_interest_by_month = reporting_data[columns_of_interest_all_metrics]
    numeric_data = data_of_interest_by_month.select_dtypes(include='number')
    monthly_mean_by_facility = numeric_data.mean(axis=1)
    monthly_reporting_data_by_facility[month] = monthly_mean_by_facility

monthly_reporting_by_facility = pd.DataFrame(monthly_reporting_data_by_facility)
monthly_reporting_by_facility["facility"] = reporting_data["organisationunitname"].values

# Weather data
if baseline:
    if baseline_all_years:
        directory = "/Users/rem76/Desktop/Climate_Change_Health/Data/Precipitation_data/Historical/monthly_data/Baseline/All_years"
    else:
        directory = "/Users/rem76/Desktop/Climate_Change_Health/Data/Precipitation_data/Historical/monthly_data/Baseline"
else:
    directory = "/Users/rem76/Desktop/Climate_Change_Health/Data/Precipitation_data/Historical/monthly_data"

malawi_grid = gpd.read_file("/Users/rem76/Desktop/Climate_Change_Health/Data/malawi_grid.shp")

files = os.listdir(directory)
weather_by_grid = {}
for file in files:
    if file.endswith('.nc'):
        file_path = os.path.join(directory, file)
        weather_monthly_all_grids = Dataset(file_path, mode='r')

pr_data = weather_monthly_all_grids.variables['tp'][:]
lat_data = weather_monthly_all_grids.variables['latitude'][:]
long_data = weather_monthly_all_grids.variables['longitude'][:]
grid = 0
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

for polygon in malawi_grid["geometry"]:
    minx, miny, maxx, maxy = polygon.bounds
    index_for_x_min = ((long_data - minx) ** 2).argmin()
    index_for_y_min = ((lat_data - miny) ** 2).argmin()

    precip_data_for_grid = pr_data[:, index_for_y_min, index_for_x_min] * multiplier
    precip_data_monthly = [precip_data_for_grid[i] * days_in_month[i % 12]
                           for i in range(len(precip_data_for_grid))]
    weather_by_grid[grid] = precip_data_monthly
    grid += 1

############### FACILITY LAT/LONG #####################
general_facilities = gpd.read_file("/Users/rem76/Desktop/Climate_Change_Health/Data/facilities_with_districts.shp")
facilities_with_lat_long = pd.read_csv(
    "/Users/rem76/Desktop/Climate_Change_Health/Data/facilities_with_lat_long_region.csv")


def clean_name(name):
    """Standardize facility names for matching - never used for display, only for keying."""
    name = str(name).lower().strip()
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = name.replace('pvt', 'private')
    name = name.replace('hc', 'health centre')
    name = name.replace('h/c', 'health centre')
    name = name.replace('dist', 'district')
    name = name.replace('.', '').replace(',', '')
    return name.strip()


# -----------------------------------------------------------------------
# Build normalized lookups ONCE upfront
# -----------------------------------------------------------------------
facilities_with_lat_long['Fname_clean'] = facilities_with_lat_long['Fname'].apply(clean_name)

# If there are duplicate cleaned names, keep the first occurrence
facilities_with_lat_long_deduped = facilities_with_lat_long.drop_duplicates(subset='Fname_clean', keep='first')

# Dict: cleaned name -> original Fname string
facilities_clean = dict(zip(facilities_with_lat_long_deduped['Fname_clean'],
                            facilities_with_lat_long_deduped['Fname']))

# Dict: cleaned name -> full row (as a Series) for direct metadata access
facilities_row_lookup = {row['Fname_clean']: row
                         for _, row in facilities_with_lat_long_deduped.iterrows()}

print(f"Total facilities in reference dataset: {len(facilities_clean)}")

# -----------------------------------------------------------------------
# Matching loop
# -----------------------------------------------------------------------
weather_data_by_facility = {}
facilities_with_location = []
facility_name_mapping = {}  # reporting name -> original Fname
matched_facility_rows = {}  # reporting name -> row Series
unmatched_facilities = []
match_stats = {'exact': 0, 'fuzzy': 0, 'special_case': 0, 'failed': 0, 'duplicate': 0}

# Guard: track which original facility names have already been claimed
# so that two different reporting names cannot map to the same original
already_matched_originals = {}  # original Fname -> first reporting name that claimed it

print("\n" + "=" * 80)
print("MATCHING FACILITIES")
print("=" * 80)

for reporting_facility in monthly_reporting_by_facility["facility"]:
    reporting_facility_clean = clean_name(reporting_facility)
    matched_clean_key = None
    is_exact = False

    # 1. Exact match on cleaned name
    if reporting_facility_clean in facilities_clean:
        matched_clean_key = reporting_facility_clean
        is_exact = True
        print(f"✓ EXACT: '{reporting_facility}' -> '{facilities_clean[matched_clean_key]}'")

    # 2. Fuzzy match
    else:
        close = difflib.get_close_matches(
            reporting_facility_clean, facilities_clean.keys(), n=1, cutoff=0.7)
        if close:
            matched_clean_key = close[0]
            print(f"≈ FUZZY: '{reporting_facility}' -> '{facilities_clean[matched_clean_key]}'")

    # If matched, check for duplicates before accepting
    if matched_clean_key is not None:
        original_facility_name = facilities_clean[matched_clean_key]

        # Duplicate guard: skip if this original has already been claimed
        if original_facility_name in already_matched_originals:
            prior = already_matched_originals[original_facility_name]
            print(f"  ⚠ DUPLICATE SKIPPED: '{reporting_facility}' would duplicate "
                  f"'{prior}' -> '{original_facility_name}'")
            unmatched_facilities.append(reporting_facility)
            match_stats['duplicate'] += 1
            continue

        facility_row = facilities_row_lookup[matched_clean_key]
        lat_for_facility = facility_row["A109__Latitude"]
        long_for_facility = facility_row["A109__Longitude"]

        if pd.isna(lat_for_facility) or pd.isna(long_for_facility):
            print(f"  ⚠ No lat/long for: {reporting_facility}")
            unmatched_facilities.append(reporting_facility)
            match_stats['failed'] += 1
            continue

        # Accept the match
        already_matched_originals[original_facility_name] = reporting_facility
        facilities_with_location.append(reporting_facility)
        facility_name_mapping[reporting_facility] = original_facility_name
        matched_facility_rows[reporting_facility] = facility_row

        if is_exact:
            match_stats['exact'] += 1
        else:
            match_stats['fuzzy'] += 1

        # Find nearest grid point and extract precipitation
        index_for_x = ((long_data - long_for_facility) ** 2).argmin()
        index_for_y = ((lat_data - lat_for_facility) ** 2).argmin()

        precip_data_for_facility = pr_data[:, index_for_y, index_for_x]
        precip_data_monthly_for_facility = [
            precip_data_for_facility[i] * days_in_month[i % 12] * multiplier
            for i in range(len(precip_data_for_facility))
        ]
        weather_data_by_facility[reporting_facility] = precip_data_monthly_for_facility

    # Special cases not in facilities file
    elif reporting_facility == "Central East Zone":
        grid_idx = general_facilities[general_facilities["District"] == "Nkhotakota"]["Grid_Index"].iloc[0]
        weather_data_by_facility[reporting_facility] = weather_by_grid[grid_idx]
        facilities_with_location.append(reporting_facility)
        facility_name_mapping[reporting_facility] = reporting_facility
        match_stats['special_case'] += 1
        print(f"★ SPECIAL: '{reporting_facility}' -> Grid-based (Nkhotakota)")

    elif reporting_facility == "Central Hospital":
        grid_idx = general_facilities[general_facilities["District"] == "Lilongwe City"]["Grid_Index"].iloc[0]
        weather_data_by_facility[reporting_facility] = weather_by_grid[grid_idx]
        facilities_with_location.append(reporting_facility)
        facility_name_mapping[reporting_facility] = reporting_facility
        match_stats['special_case'] += 1
        print(f"★ SPECIAL: '{reporting_facility}' -> Grid-based (Lilongwe City)")

    else:
        unmatched_facilities.append(reporting_facility)
        match_stats['failed'] += 1
        close = difflib.get_close_matches(reporting_facility_clean, facilities_clean.keys(), n=3, cutoff=0.4)
        if close:
            print(f"✗ FAILED: '{reporting_facility}'")
            print(f"    Closest candidates: {[facilities_clean[c] for c in close]}")
        else:
            print(f"✗ FAILED: '{reporting_facility}' (no close matches found)")

print("\n" + "=" * 80)
print("MATCHING SUMMARY")
print("=" * 80)
print(f"Total facilities to match:     {len(monthly_reporting_by_facility['facility'])}")
print(f"Successfully matched:          {len(facilities_with_location)}")
print(f"  - Exact matches:             {match_stats['exact']}")
print(f"  - Fuzzy matches:             {match_stats['fuzzy']}")
print(f"  - Special cases:             {match_stats['special_case']}")
print(f"Duplicate originals skipped:   {match_stats['duplicate']}")
print(f"Failed to match:               {match_stats['failed']}")

if unmatched_facilities:
    print(f"\nUnmatched/skipped facilities ({len(unmatched_facilities)}):")
    for f in unmatched_facilities:
        print(f"  - {f}")
print("=" * 80 + "\n")

# Hard assertion: no duplicate IDs should survive to the output
assert len(facilities_with_location) == len(set(facilities_with_location)), \
    "ERROR: Duplicate facility IDs in facilities_with_location!"
print("✓ No duplicate facility IDs confirmed.")

### Prepare dataframes for regression
weather_df = pd.DataFrame.from_dict(weather_data_by_facility, orient='index').T
weather_df.columns = facilities_with_location

monthly_reporting_by_facility = monthly_reporting_by_facility.set_index('facility').T
monthly_reporting_by_facility.index.name = "date"
monthly_reporting_by_facility = monthly_reporting_by_facility.loc[:,
                                monthly_reporting_by_facility.columns.isin(facilities_with_location)]
monthly_reporting_by_facility = monthly_reporting_by_facility[facilities_with_location]

# -----------------------------------------------------------------------
# Build expanded_facility_info directly from matched rows
# -----------------------------------------------------------------------
additional_rows = ["Zonename", "Resid", "Dist", "A105", "A109__Altitude", "Ftype",
                   'A109__Latitude', 'A109__Longitude']

regular_facilities = [f for f in facilities_with_location if f in matched_facility_rows]
special_case_facilities = [f for f in facilities_with_location if f not in matched_facility_rows]

print(f"Building expanded facility info:")
print(f"  Regular facilities:      {len(regular_facilities)}")
print(f"  Special case facilities: {len(special_case_facilities)}")

records = []
for rep_name in regular_facilities:
    row = matched_facility_rows[rep_name]
    record = {'reporting_name': rep_name}
    for col in additional_rows:
        record[col] = row.get(col, np.nan)
    records.append(record)

expanded_facility_info = pd.DataFrame(records).set_index('reporting_name').T
expanded_facility_info.index.name = None

# Fix known district name typos
if 'Dist' in expanded_facility_info.index:
    expanded_facility_info.loc['Dist'] = expanded_facility_info.loc['Dist'].replace(
        {"Blanytyre": "Blantyre", "Nkhatabay": "Nkhata Bay"})

# Add average precipitation
average_precipitation_by_facility = {
    facility: np.mean(precipitation)
    for facility, precipitation in weather_data_by_facility.items()
}
average_precipitation_df = pd.DataFrame.from_dict(
    average_precipitation_by_facility, orient='index', columns=['average_precipitation'])

precip_row = {
    rep_name: average_precipitation_df.loc[rep_name, 'average_precipitation']
    if rep_name in average_precipitation_df.index else np.nan
    for rep_name in facilities_with_location
}
expanded_facility_info.loc['average_precipitation'] = precip_row

# Minimum distance between facilities
coords = expanded_facility_info.loc[
    ['A109__Latitude', 'A109__Longitude'], regular_facilities].T.values.astype(float)
distances = cdist(coords, coords, metric='euclidean')
np.fill_diagonal(distances, np.inf)
min_distances = np.nanmin(distances, axis=1)

min_dist_row = dict(zip(regular_facilities, min_distances))
for sf in special_case_facilities:
    min_dist_row[sf] = np.nan
expanded_facility_info.loc['minimum_distance'] = min_dist_row

# Add empty columns for special case facilities
for sf in special_case_facilities:
    if sf not in expanded_facility_info.columns:
        expanded_facility_info[sf] = np.nan
        if sf in average_precipitation_df.index:
            expanded_facility_info.loc['average_precipitation', sf] = \
                average_precipitation_df.loc[sf, 'average_precipitation']

# Reindex to preserve original facility order
expanded_facility_info = expanded_facility_info.reindex(columns=facilities_with_location)

print(f"\nFinal expanded_facility_info shape: {expanded_facility_info.shape}")

# Sanity check: any regular facilities still missing lat/long?
missing_meta = [col for col in regular_facilities
                if pd.isna(expanded_facility_info.loc['A109__Latitude', col])]
if missing_meta:
    print(f"\n⚠ WARNING: {len(missing_meta)} regular facilities have missing metadata:")
    for f in missing_meta:
        print(f"  - {f}")
else:
    print("✓ All regular facilities have metadata populated correctly.")

# -----------------------------------------------------------------------
# Save CSVs
# -----------------------------------------------------------------------
if baseline:
    if baseline_all_years:
        if ANC:
            weather_df.to_csv(
                "/Users/rem76/Desktop/Climate_Change_Health/Data/historical_weather_by_smaller_facilities_with_ANC_lm_baseline_all_years.csv")
            expanded_facility_info.to_csv(
                "/Users/rem76/Desktop/Climate_Change_Health/Data/expanded_facility_info_by_smaller_facility_lm_with_ANC_baseline_all_years.csv")
        if Inpatient:
            weather_df.to_csv(
                "/Users/rem76/Desktop/Climate_Change_Health/Data/historical_weather_by_smaller_facilities_with_Inpatient_lm_baseline_all_years.csv")
            expanded_facility_info.to_csv(
                "/Users/rem76/Desktop/Climate_Change_Health/Data/expanded_facility_info_by_smaller_facility_lm_with_Inpatient_baseline_all_years.csv")
    else:
        if ANC:
            weather_df.to_csv(
                "/Users/rem76/Desktop/Climate_Change_Health/Data/historical_weather_by_smaller_facilities_with_ANC_lm_baseline.csv")
            expanded_facility_info.to_csv(
                "/Users/rem76/Desktop/Climate_Change_Health/Data/expanded_facility_info_by_smaller_facility_lm_with_ANC_baseline.csv")
        if Inpatient:
            weather_df.to_csv(
                "/Users/rem76/Desktop/Climate_Change_Health/Data/historical_weather_by_smaller_facilities_with_Inpatient_lm_baseline.csv")
            expanded_facility_info.to_csv(
                "/Users/rem76/Desktop/Climate_Change_Health/Data/expanded_facility_info_by_smaller_facility_lm_with_Inpatient_baseline.csv")
        else:
            weather_df.to_csv(
                "/Users/rem76/Desktop/Climate_Change_Health/Data/historical_weather_by_smaller_facility_lm_baseline.csv")
            expanded_facility_info.to_csv(
                "/Users/rem76/Desktop/Climate_Change_Health/Data/expanded_facility_info_by_smaller_facility_lm_baseline.csv")
else:
    if ANC:
        weather_df.to_csv(
            "/Users/rem76/Desktop/Climate_Change_Health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv")
        expanded_facility_info.to_csv(
            "/Users/rem76/Desktop/Climate_Change_Health/Data/expanded_facility_info_by_smaller_facility_lm_with_ANC.csv")
        monthly_reporting_by_facility.to_csv(
            "/Users/rem76/Desktop/Climate_Change_Health/Data/monthly_reporting_ANC_by_smaller_facility_lm.csv")
    if Inpatient:
        weather_df.to_csv(
            "/Users/rem76/Desktop/Climate_Change_Health/Data/historical_weather_by_smaller_facilities_with_Inpatient_lm.csv")
        expanded_facility_info.to_csv(
            "/Users/rem76/Desktop/Climate_Change_Health/Data/expanded_facility_info_by_smaller_facility_lm_with_inpatient_days.csv")
        monthly_reporting_by_facility.to_csv(
            "/Users/rem76/Desktop/Climate_Change_Health/Data/monthly_reporting_Inpatient_by_smaller_facility_lm.csv")
    else:
        weather_df.to_csv(
            "/Users/rem76/Desktop/Climate_Change_Health/Data/historical_weather_by_smaller_facility_lm.csv")
        expanded_facility_info.to_csv(
            "/Users/rem76/Desktop/Climate_Change_Health/Data/expanded_facility_info_by_smaller_facility_lm.csv")
        monthly_reporting_by_facility.to_csv(
            "/Users/rem76/Desktop/Climate_Change_Health/Data/monthly_reporting_by_smaller_facility_lm.csv")

print("\n✓ Script completed successfully!")
print(f"Files saved to: /Users/rem76/Desktop/Climate_Change_Health/Data/")
