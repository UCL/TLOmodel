import difflib
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist

# Configuration
ANC = True
Inpatient = False
baseline = False
baseline_all_years = False

# Load health facility data
if ANC:
    reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_change_health/Data/ANC_data/ANC_data_2011_2024.csv')
elif Inpatient:
    reporting_data = pd.read_csv(
        '/Users/rem76/Desktop/Climate_change_health/Data/Inpatient_Data/HMIS_Total_Number_Admissions.csv')
else:
    reporting_data = pd.read_csv(
        '/Users/rem76/Desktop/Climate_change_health/Data/Reporting_Rate/Reporting_Rate_by_smaller_facilities_2011_2024.csv')

# Drop facilities with all NAs
reporting_data = reporting_data.dropna(subset=reporting_data.columns[3:], how='all')

# Aggregate over months
monthly_reporting_data_by_facility = {}
if ANC:
    months = set(col.split("HMIS Total Antenatal Visits ")[1] for col in reporting_data.columns if
                 "HMIS Total Antenatal Visits " in col)
elif Inpatient:
    months = set(col.split("HMIS Total # of Admissions (including Maternity) ")[1] for col in reporting_data.columns if
                 "HMIS Total # of Admissions (including Maternity) " in col)
else:
    months = set(col.split(" - Reporting rate ")[1] for col in reporting_data.columns if " - Reporting rate " in col)

# Sort months chronologically
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

# Load relative humidity data
wbgt_directory = "/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/relative_humidity/Historical/"
wbgt_files = [f for f in os.listdir(wbgt_directory) if f.startswith('wbgt_monthly_') and f.endswith('.nc')]

if not wbgt_files:
    raise FileNotFoundError(f"No wbgt monthly files found in {wbgt_directory}")

wbgt_file_path = os.path.join(wbgt_directory, wbgt_files[0])
print(f"Loading wbgt data from {wbgt_file_path}")

# Open wbgt dataset
ds_wbgt = xr.open_dataset(wbgt_file_path)
wbgt_data = ds_wbgt['wbgt'].values  # shape: (time, lat, lon)
lat_data = ds_wbgt['latitude'].values
long_data = ds_wbgt['longitude'].values
time_data = pd.to_datetime(ds_wbgt['time'].values)

print(f"wbgt data shape: {wbgt_data.shape}")
print(f"Time range: {time_data[0]} to {time_data[-1]}")

# Load Malawi grid
malawi_grid = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/malawi_grid.shp")

# Extract wbgt by grid square
wbgt_by_grid = {}
for grid_idx, polygon in enumerate(malawi_grid["geometry"]):
    minx, miny, maxx, maxy = polygon.bounds

    # Find closest indices for grid bounds
    index_for_x_min = ((long_data - minx) ** 2).argmin()
    index_for_y_min = ((lat_data - miny) ** 2).argmin()

    # Extract wbgt for this grid (using corner point for simplicity)
    wbgt_data_for_grid = wbgt_data[:, index_for_y_min, index_for_x_min]
    wbgt_by_grid[grid_idx] = wbgt_data_for_grid.tolist()

print(f"Extracted wbgt data for {len(wbgt_by_grid)} grid squares")

# Load facility location data
general_facilities = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.shp")
facilities_with_lat_long = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_lat_long_region.csv")

# Match facilities to wbgt data
wbgt_data_by_facility = {}
facilities_with_location = []

for reporting_facility in monthly_reporting_by_facility["facility"]:
    # Try to match facility name
    matching_facility_name = difflib.get_close_matches(
        reporting_facility,
        facilities_with_lat_long['Fname'],
        n=3,
        cutoff=0.90
    )

    if matching_facility_name:
        match_name = matching_facility_name[0]
        lat_for_facility = facilities_with_lat_long.loc[
            facilities_with_lat_long['Fname'] == match_name, "A109__Latitude"
        ].iloc[0]
        long_for_facility = facilities_with_lat_long.loc[
            facilities_with_lat_long['Fname'] == match_name, "A109__Longitude"
        ].iloc[0]

        if pd.isna(lat_for_facility):
            print(f"No coordinates for {reporting_facility}")
            continue

        facilities_with_location.append(reporting_facility)

        # Find closest grid point
        index_for_x = ((long_data - long_for_facility) ** 2).argmin()
        index_for_y = ((lat_data - lat_for_facility) ** 2).argmin()

        # Extract wbgt for this facility
        wbgt_data_for_facility = wbgt_data[:, index_for_y, index_for_x]
        wbgt_data_by_facility[reporting_facility] = wbgt_data_for_facility.tolist()

    # Handle special cases
    elif reporting_facility == "Central East Zone":
        grid = general_facilities[general_facilities["District"] == "Nkhotakota"]["Grid_Index"].iloc[0]
        wbgt_data_by_facility[reporting_facility] = wbgt_by_grid[grid]
        facilities_with_location.append(reporting_facility)

    elif reporting_facility == "Central Hospital":
        grid = general_facilities[general_facilities["District"] == "Lilongwe City"]["Grid_Index"].iloc[0]
        wbgt_data_by_facility[reporting_facility] = wbgt_by_grid[grid]
        facilities_with_location.append(reporting_facility)

    else:
        continue

print(f"Matched {len(facilities_with_location)} facilities to wbgt data")

# Create DataFrame with wbgt data
wbgt_df = pd.DataFrame.from_dict(wbgt_data_by_facility, orient='index').T
wbgt_df.columns = facilities_with_location

# Add time index
wbgt_df.index = time_data[:len(wbgt_df)]
wbgt_df.index.name = "date"

# Prepare reporting data
monthly_reporting_by_facility = monthly_reporting_by_facility.set_index('facility').T
monthly_reporting_by_facility.index.name = "date"
monthly_reporting_by_facility = monthly_reporting_by_facility.loc[
                                :, monthly_reporting_by_facility.columns.isin(facilities_with_location)
                                ]
monthly_reporting_by_facility = monthly_reporting_by_facility[facilities_with_location]

# Get additional facility information
included_facilities_with_lat_long = facilities_with_lat_long[
    facilities_with_lat_long["Fname"].isin(facilities_with_location)
]

additional_rows = ["Zonename", "Resid", "Dist", "A105", "A109__Altitude", "Ftype",
                   'A109__Latitude', 'A109__Longitude']
expanded_facility_info = included_facilities_with_lat_long[["Fname"] + additional_rows]

# Clean district names
expanded_facility_info['Dist'] = expanded_facility_info['Dist'].replace("Blanytyre", "Blantyre")
expanded_facility_info['Dist'] = expanded_facility_info['Dist'].replace("Nkhatabay", "Nkhata Bay")

expanded_facility_info.set_index("Fname", inplace=True)

# Calculate minimum distances between facilities
coordinates = expanded_facility_info[['A109__Latitude', 'A109__Longitude']].values
distances = cdist(coordinates, coordinates, metric='euclidean')
np.fill_diagonal(distances, np.inf)
expanded_facility_info['minimum_distance'] = np.nanmin(distances, axis=1)

# Calculate average wbgt by facility
average_wbgt_by_facility = {
    facility: np.mean(wbgt_values)
    for facility, wbgt_values in wbgt_data_by_facility.items()
}

average_wbgt_df = pd.DataFrame.from_dict(
    average_wbgt_by_facility, orient='index', columns=['average_wbgt']
)
average_wbgt_df.index.name = "Fname"

expanded_facility_info['average_wbgt'] = expanded_facility_info.index.map(
    average_wbgt_df['average_wbgt']
)

expanded_facility_info = expanded_facility_info.T
expanded_facility_info = expanded_facility_info.reindex(columns=facilities_with_location)

# Save outputs
output_dir = "/Users/rem76/Desktop/Climate_change_health/Data/"

if ANC:
    wbgt_df.to_csv(os.path.join(output_dir, "historical_wbgt_by_smaller_facilities_with_ANC_lm.csv"))
    expanded_facility_info.to_csv(
        os.path.join(output_dir, "expanded_facility_info_wbgt_by_smaller_facility_lm_with_ANC.csv"))
    monthly_reporting_by_facility.to_csv(
        os.path.join(output_dir, "monthly_reporting_ANC_by_smaller_facility_lm_wbgt.csv"))
    print("Saved ANC outputs")

elif Inpatient:
    wbgt_df.to_csv(os.path.join(output_dir, "historical_wbgt_by_smaller_facilities_with_Inpatient_lm.csv"))
    expanded_facility_info.to_csv(
        os.path.join(output_dir, "expanded_facility_info_wbgt_by_smaller_facility_lm_with_Inpatient.csv"))
    monthly_reporting_by_facility.to_csv(
        os.path.join(output_dir, "monthly_reporting_Inpatient_by_smaller_facility_lm_wbgt.csv"))
    print("Saved Inpatient outputs")

else:
    wbgt_df.to_csv(os.path.join(output_dir, "historical_wbgt_by_smaller_facility_lm.csv"))
    expanded_facility_info.to_csv(os.path.join(output_dir, "expanded_facility_info_wbgt_by_smaller_facility_lm.csv"))
    monthly_reporting_by_facility.to_csv(os.path.join(output_dir, "monthly_reporting_by_smaller_facility_lm_wbgt.csv"))
    print("Saved reporting rate outputs")

print("\nProcessing complete!")
print(f"wbgt data saved with {len(facilities_with_location)} facilities")
print(f"Time range: {wbgt_df.index[0]} to {wbgt_df.index[-1]}")

# Clean up
ds_wbgt.close()
