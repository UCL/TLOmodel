import difflib
import os
import re
import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial.distance import cdist

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

# drop NAs
reporting_data = reporting_data.dropna(subset=reporting_data.columns[3:], how='all')

### now aggregate over months
monthly_reporting_data_by_facility = {}
if ANC:
    months = set(col.split("HMIS Total Antenatal Visits ")[1] for col in reporting_data.columns if
                 "HMIS Total Antenatal Visits " in col)
elif Inpatient:
    months = set(col.split("HMIS Total # of Admissions (including Maternity) ")[1] for col in reporting_data.columns if
                 "HMIS Total # of Admissions (including Maternity) " in col)
else:
    months = set(col.split(" - Reporting rate ")[1] for col in reporting_data.columns if " - Reporting rate " in col)

# put in order
months = [date.strip() for date in months]
dates = pd.to_datetime(months, format='%B %Y', errors='coerce')
months = dates.sort_values().strftime('%B %Y').tolist()

for month in months:
    columns_of_interest_all_metrics = [reporting_data.columns[1]] + reporting_data.columns[
        reporting_data.columns.str.endswith(month)].tolist()
    print(columns_of_interest_all_metrics)
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
regridded_weather_data = {}
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

for polygon in malawi_grid["geometry"]:
    month = 0
    minx, miny, maxx, maxy = polygon.bounds
    index_for_x_min = ((long_data - minx) ** 2).argmin()
    index_for_y_min = ((lat_data - miny) ** 2).argmin()
    index_for_x_max = ((long_data - maxx) ** 2).argmin()
    index_for_y_max = ((lat_data - maxy) ** 2).argmin()

    precip_data_for_grid = pr_data[:, index_for_y_min, index_for_x_min]
    precip_data_for_grid = precip_data_for_grid * multiplier
    precip_data_monthly = []
    for i in range(len(precip_data_for_grid)):
        month = i % 12
        precip_total_for_month = precip_data_for_grid[i] * days_in_month[month]
        precip_data_monthly.append(precip_total_for_month)
    weather_by_grid[grid] = precip_data_monthly
    grid += 1

############### NOW HAVE LAT/LONG OF FACILITIES #####################
general_facilities = gpd.read_file("/Users/rem76/Desktop/Climate_Change_Health/Data/facilities_with_districts.shp")
facilities_with_lat_long = pd.read_csv(
    "/Users/rem76/Desktop/Climate_Change_Health/Data/facilities_with_lat_long_region.csv")


def clean_name(name):
    """Standardize facility names for better matching"""
    name = str(name).lower().strip()
    name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single

    # Remove parenthetical suffixes first (e.g., (COM), (MOH), (CHAM))
    name = re.sub(r'\s*\([^)]*\)', '', name)

    # Standardize common abbreviations
    name = name.replace('pvt', 'private')
    name = name.replace('hc', 'health centre')
    name = name.replace('h/c', 'health centre')
    name = name.replace('dist', 'district')

    # Remove common punctuation
    name = name.replace('.', '').replace(',', '')

    return name.strip()


# CREATE THE CLEANED MAPPING HERE (maps cleaned name -> original name)
facilities_clean = {clean_name(f): f for f in facilities_with_lat_long['Fname']}
print(f"\nTotal facilities in reference dataset: {len(facilities_clean)}")

weather_data_by_facility = {}
facilities_with_location = []
facility_name_mapping = {}  # maps reporting name -> original name
unmatched_facilities = []
match_stats = {'exact': 0, 'fuzzy': 0, 'special_case': 0, 'failed': 0}

print("\n" + "=" * 80)
print("MATCHING FACILITIES")
print("=" * 80)

for reporting_facility in monthly_reporting_by_facility["facility"]:
    # Clean the reporting facility name before matching
    reporting_facility_clean = clean_name(reporting_facility)
    original_facility_name = None
    match_type = None

    # Try exact matching first
    if reporting_facility_clean in facilities_clean.keys():
        original_facility_name = facilities_clean[reporting_facility_clean]
        match_type = 'exact'
        match_stats['exact'] += 1
        print(f"✓ EXACT: '{reporting_facility}' -> '{original_facility_name}'")

    # Fall back to fuzzy matching
    else:
        matching_facility_name = difflib.get_close_matches(
            reporting_facility_clean,
            facilities_clean.keys(),
            n=1,
            cutoff=0.7
        )

        if matching_facility_name:
            original_facility_name = facilities_clean[matching_facility_name[0]]
            match_type = 'fuzzy'
            match_stats['fuzzy'] += 1
            print(f"≈ FUZZY: '{reporting_facility}' -> '{original_facility_name}'")

    # If we found a match (either exact or fuzzy), get the location data
    if original_facility_name:
        # Look up lat/long using the ORIGINAL facility name
        facility_data = facilities_with_lat_long[facilities_with_lat_long['Fname'] == original_facility_name]

        if len(facility_data) == 0:
            print(f"  ⚠ Warning: Matched name not found in dataset: {original_facility_name}")
            unmatched_facilities.append(reporting_facility)
            match_stats['failed'] += 1
            continue

        lat_for_facility = facility_data["A109__Latitude"].iloc[0]
        long_for_facility = facility_data["A109__Longitude"].iloc[0]

        if pd.isna(lat_for_facility) or pd.isna(long_for_facility):
            print(f"  ⚠ No lat/long for: {reporting_facility}")
            unmatched_facilities.append(reporting_facility)
            match_stats['failed'] += 1
            continue

        facilities_with_location.append(reporting_facility)
        facility_name_mapping[reporting_facility] = original_facility_name

        # Find nearest grid point
        index_for_x = ((long_data - long_for_facility) ** 2).argmin()
        index_for_y = ((lat_data - lat_for_facility) ** 2).argmin()

        precip_data_for_facility = pr_data[:, index_for_y, index_for_x]
        precip_data_monthly_for_facility = []

        # Convert from daily means to monthly totals
        for i in range(len(precip_data_for_facility)):
            month = i % 12
            precip_total_for_month = precip_data_for_facility[i] * days_in_month[month] * multiplier
            precip_data_monthly_for_facility.append(precip_total_for_month)

        weather_data_by_facility[reporting_facility] = precip_data_monthly_for_facility

    # Handle special cases not in facilities file
    elif reporting_facility == "Central East Zone":
        grid = general_facilities[general_facilities["District"] == "Nkhotakota"]["Grid_Index"].iloc[0]
        weather_data_by_facility[reporting_facility] = weather_by_grid[grid]
        facilities_with_location.append(reporting_facility)
        facility_name_mapping[reporting_facility] = reporting_facility  # Special case maps to itself
        match_stats['special_case'] += 1
        print(f"★ SPECIAL: '{reporting_facility}' -> Grid-based (Nkhotakota)")

    elif reporting_facility == "Central Hospital":
        grid = general_facilities[general_facilities["District"] == "Lilongwe City"]["Grid_Index"].iloc[0]
        weather_data_by_facility[reporting_facility] = weather_by_grid[grid]
        facilities_with_location.append(reporting_facility)
        facility_name_mapping[reporting_facility] = reporting_facility  # Special case maps to itself
        match_stats['special_case'] += 1
        print(f"★ SPECIAL: '{reporting_facility}' -> Grid-based (Lilongwe City)")

    else:
        unmatched_facilities.append(reporting_facility)
        match_stats['failed'] += 1

        # Show closest matches for debugging
        close = difflib.get_close_matches(reporting_facility_clean, facilities_clean.keys(), n=3, cutoff=0.5)
        if close:
            print(f"✗ FAILED: '{reporting_facility}'")
            print(f"    Closest candidates (cutoff=0.5): {[facilities_clean[c] for c in close]}")
        else:
            print(f"✗ FAILED: '{reporting_facility}' (no close matches found)")

# Print summary statistics
print("\n" + "=" * 80)
print("MATCHING SUMMARY")
print("=" * 80)
print(f"Total facilities to match: {len(monthly_reporting_by_facility['facility'])}")
print(f"Successfully matched: {len(facilities_with_location)}")
print(f"  - Exact matches: {match_stats['exact']}")
print(f"  - Fuzzy matches: {match_stats['fuzzy']}")
print(f"  - Special cases: {match_stats['special_case']}")
print(f"Failed to match: {match_stats['failed']}")

if unmatched_facilities:
    print(f"\nUnmatched facilities ({len(unmatched_facilities)}):")
    for facility in unmatched_facilities:
        print(f"  - {facility}")

print("=" * 80 + "\n")

### Get data ready for linear regression between reporting and weather data
weather_df = pd.DataFrame.from_dict(weather_data_by_facility, orient='index').T
weather_df.columns = facilities_with_location
monthly_reporting_by_facility = monthly_reporting_by_facility.set_index('facility').T
monthly_reporting_by_facility.index.name = "date"

monthly_reporting_by_facility = monthly_reporting_by_facility.loc[:,
                                monthly_reporting_by_facility.columns.isin(facilities_with_location)]
monthly_reporting_by_facility = monthly_reporting_by_facility[facilities_with_location]

# NOW BUILD expanded_facility_info using the ORIGINAL facility names
# Separate regular facilities from special cases
regular_facilities = [f for f in facilities_with_location
                      if f in facility_name_mapping and facility_name_mapping[f] != f]
special_case_facilities = [f for f in facilities_with_location if f not in regular_facilities]

print(f"\nBuilding expanded facility info:")
print(f"  Regular facilities: {len(regular_facilities)}")
print(f"  Special case facilities: {len(special_case_facilities)}")

# Get the original facility names for the matched regular facilities
original_facility_names = [facility_name_mapping[f] for f in regular_facilities]

included_facilities_with_lat_long = facilities_with_lat_long[
    facilities_with_lat_long["Fname"].isin(original_facility_names)
].copy()  # Use .copy() to avoid SettingWithCopyWarning

additional_rows = ["Zonename", "Resid", "Dist", "A105", "A109__Altitude", "Ftype", 'A109__Latitude',
                   'A109__Longitude']
expanded_facility_info = included_facilities_with_lat_long[["Fname"] + additional_rows].copy()
expanded_facility_info['Dist'] = expanded_facility_info['Dist'].replace("Blanytyre", "Blantyre")
expanded_facility_info['Dist'] = expanded_facility_info['Dist'].replace("Nkhatabay", "Nkhata Bay")

expanded_facility_info.columns = ["Fname"] + additional_rows
expanded_facility_info.set_index("Fname", inplace=True)

# minimum distances between facilities
coordinates = expanded_facility_info[['A109__Latitude', 'A109__Longitude']].values
distances = cdist(coordinates, coordinates, metric='euclidean')
np.fill_diagonal(distances, np.inf)
expanded_facility_info['minimum_distance'] = np.nanmin(distances, axis=1)

# Calculate average precipitation
if baseline:
    average_precipitation_by_facility = {
        facility: np.mean(precipitation)
        for facility, precipitation in weather_data_by_facility.items()
    }
else:
    average_precipitation_by_facility = {
        facility: np.mean(precipitation)
        for facility, precipitation in weather_data_by_facility.items()
    }

average_precipitation_df = pd.DataFrame.from_dict(
    average_precipitation_by_facility, orient='index', columns=['average_precipitation']
)

# Add average precipitation for each original facility name
for orig_name in expanded_facility_info.index:
    # Find the reporting name that corresponds to this original name
    reporting_name = None
    for rep, orig in facility_name_mapping.items():
        if orig == orig_name:
            reporting_name = rep
            break

    if reporting_name and reporting_name in average_precipitation_df.index:
        expanded_facility_info.loc[orig_name, 'average_precipitation'] = average_precipitation_df.loc[
            reporting_name, 'average_precipitation']

# Transpose
expanded_facility_info = expanded_facility_info.T

# Create column mapping: original name -> reporting name
# Build this carefully to avoid duplicates
column_rename = {}
for rep_name in regular_facilities:
    orig_name = facility_name_mapping[rep_name]
    if orig_name in expanded_facility_info.columns:
        # Check if we're about to create a duplicate
        if rep_name in column_rename.values():
            print(f"WARNING: Duplicate reporting name detected: {rep_name}")
        column_rename[orig_name] = rep_name

# Debug: Check for duplicate values in column_rename
if len(column_rename.values()) != len(set(column_rename.values())):
    print("ERROR: Duplicate values in column_rename!")
    print(f"column_rename: {column_rename}")

print(f"\nRenaming {len(column_rename)} columns from original to reporting names")
expanded_facility_info = expanded_facility_info.rename(columns=column_rename)

print(f"After rename, columns: {list(expanded_facility_info.columns)}")
print(f"Number of unique columns: {len(set(expanded_facility_info.columns))}")

# Check for duplicates before adding special cases
if len(expanded_facility_info.columns) != len(set(expanded_facility_info.columns)):
    print("ERROR: Duplicate columns detected after rename!")
    duplicates = [col for col in expanded_facility_info.columns if list(expanded_facility_info.columns).count(col) > 1]
    print(f"Duplicate columns: {set(duplicates)}")
    # Remove duplicate columns, keeping the first occurrence
    expanded_facility_info = expanded_facility_info.loc[:, ~expanded_facility_info.columns.duplicated()]

# For special case facilities, add empty columns with NaN values
for special_facility in special_case_facilities:
    if special_facility not in expanded_facility_info.columns:
        expanded_facility_info[special_facility] = np.nan
        # Add average precipitation for special cases
        if special_facility in average_precipitation_df.index:
            expanded_facility_info.loc['average_precipitation', special_facility] = average_precipitation_df.loc[
                special_facility, 'average_precipitation']

# Now reindex to include all facilities in the correct order
print(f"\nReindexing with {len(facilities_with_location)} facilities")
expanded_facility_info = expanded_facility_info.reindex(columns=facilities_with_location)

print(f"\nFinal expanded_facility_info shape: {expanded_facility_info.shape}")
print(f"Columns: {list(expanded_facility_info.columns)[:5]}... (showing first 5)")

# Save CSVs
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
