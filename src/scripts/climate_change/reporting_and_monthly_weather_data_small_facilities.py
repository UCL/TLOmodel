import difflib
import os

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
if ANC:
    reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_change_health/Data/ANC_data/ANC_data_2011_2024.csv')
elif Inpatient:
    reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_change_health/Data/Inpatient_Data/HMIS_Total_Number_Admissions.csv')
else:
    reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_change_health/Data/Reporting_Rate/Reporting_Rate_by_smaller_facilities_2011_2024.csv') #January 2011 - January 2024

# ANALYSIS DONE IN OCTOBER 2024 - so drop October, November, December 2024
columns_to_drop = reporting_data.columns[reporting_data.columns.str.endswith(('October 2024', 'November 2024', 'December 2024'))]
reporting_data = reporting_data.drop(columns=columns_to_drop)
# drop NAs
reporting_data = reporting_data.dropna(subset = reporting_data.columns[3:], how='all') # drops 90 clinics
### now aggregate over months
monthly_reporting_data_by_facility =  {}
if ANC:
    months = set(col.split("HMIS Total Antenatal Visits ")[1] for col in reporting_data.columns if "HMIS Total Antenatal Visits " in col)
if Inpatient:
    months = set(col.split("HMIS Total # of Admissions (including Maternity) ")[1] for col in reporting_data.columns if "HMIS Total # of Admissions (including Maternity) " in col)
else:
    months = set(col.split(" - Reporting rate ")[1] for col in reporting_data.columns if " - Reporting rate " in col)
months = set(col.split("HMIS Total Antenatal Visits ")[1] for col in reporting_data.columns if "HMIS Total Antenatal Visits " in col)

# put in order
months = [date.strip() for date in months] # extra spaces??
dates = pd.to_datetime(months, format='%B %Y', errors='coerce')
months = dates.sort_values().strftime('%B %Y').tolist() # puts them in ascending order
print(months)
for month in months:
    columns_of_interest_all_metrics = [reporting_data.columns[1]] + reporting_data.columns[reporting_data.columns.str.endswith(month)].tolist()
    data_of_interest_by_month = reporting_data[columns_of_interest_all_metrics]
    numeric_data = data_of_interest_by_month.select_dtypes(include='number')
    monthly_mean_by_facility = numeric_data.mean(axis=1)
    monthly_reporting_data_by_facility[month] = monthly_mean_by_facility
monthly_reporting_by_facility = pd.DataFrame(monthly_reporting_data_by_facility)
monthly_reporting_by_facility["facility"] = reporting_data["organisationunitname"].values

# Weather data
directory = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/monthly_data" # from 2011 on
malawi_grid = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/malawi_grid.shp")
# find indices of interest from the malawi file

files = os.listdir(directory)
weather_by_grid = {}
for file in files:
    if file.endswith('.nc'):
        file_path = os.path.join(directory, file)
        # Open the NetCDF file - unsure of name, should only be one though
        weather_monthly_all_grids = Dataset(file_path, mode='r')
# the historical data is at a different resolution to the projections. so try and find the closest possible indicses
# to create a new grid for the historical data
pr_data = weather_monthly_all_grids.variables['tp'][:]  # total precipitation in kg m-2 s-1 = mm s-1 x 86400 to get to day
lat_data = weather_monthly_all_grids.variables['latitude'][:]
long_data = weather_monthly_all_grids.variables['longitude'][:]
date = weather_monthly_all_grids['date'][:]
grid = 0
regridded_weather_data = {}
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

for polygon in malawi_grid["geometry"]:
    month = 0
    minx, miny, maxx, maxy = polygon.bounds
    index_for_x_min = ((long_data - minx)**2).argmin()
    index_for_y_min = ((lat_data - miny)**2).argmin()
    index_for_x_max = ((long_data - maxx)**2).argmin()
    index_for_y_max = ((lat_data - maxy)**2).argmin()

    precip_data_for_grid = pr_data[:, index_for_y_min,index_for_x_min]  # across all time points
    precip_data_for_grid = precip_data_for_grid * multiplier  # tday OR m to mm (daily max data). Monthly data is average per day...
    precip_data_monthly = []
    for i in range(len(precip_data_for_grid)):  #from ECMWF website: monthly means is daily means, so need to multiply by number of days in a month
        month = i % 12
        precip_total_for_month = precip_data_for_grid[i] * days_in_month[month]
        precip_data_monthly.append(precip_total_for_month)
    weather_by_grid[grid] = precip_data_monthly
    grid += 1

#
############### NOW HAVE LAT/LONG OF FACILITIES #####################
general_facilities = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.shp")

facilities_with_lat_long = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_lat_long_region.csv")
print(facilities_with_lat_long.columns)
weather_data_by_facility = {}
facilities_with_location = []
for reporting_facility in monthly_reporting_by_facility["facility"]:
    matching_facility_name = difflib.get_close_matches(reporting_facility, facilities_with_lat_long['Fname'], n=3, cutoff=0.90)
    if matching_facility_name:
        match_name = matching_facility_name[0]  # Access the string directly
        lat_for_facility = facilities_with_lat_long.loc[
            facilities_with_lat_long['Fname'] == match_name, "A109__Latitude"].iloc[0]
        long_for_facility = facilities_with_lat_long.loc[
            facilities_with_lat_long['Fname'] == match_name, "A109__Longitude"].iloc[0]
        if pd.isna(lat_for_facility):
            print(reporting_facility)
            continue
        facilities_with_location.append(reporting_facility)

        index_for_x = ((long_data - long_for_facility)**2).argmin()
        index_for_y= ((lat_data - lat_for_facility)**2).argmin()

        precip_data_for_facility = pr_data[:, index_for_y,index_for_x]  # across all time points
        precip_data_monthly_for_facility = []

        for i in range(len(precip_data_for_facility)):  # from ECMWF website: monthly means is daily means, so need to multiply by number of days in a month
            month = i % 12
            precip_total_for_month = precip_data_for_facility[i] * days_in_month[month] * multiplier
            precip_data_monthly_for_facility.append(precip_total_for_month)
        weather_data_by_facility[reporting_facility]  = precip_data_monthly_for_facility  # to get from per second to per day
## below are not in facilities file?
    elif reporting_facility == "Central East Zone":
        grid = general_facilities[general_facilities["District"] == "Nkhotakota"]["Grid_Index"].iloc[0] # furtherst east zone
        weather_data_by_facility[reporting_facility]  = weather_by_grid[grid]
    elif (reporting_facility == "Central Hospital"):
         grid = general_facilities[general_facilities["District"] == "Lilongwe City"]["Grid_Index"].iloc[0] # all labelled X City will be in the same grid
         weather_data_by_facility[reporting_facility]  = weather_by_grid[grid]
    else:
        continue


### Get data ready for linear regression between reporting and weather data
weather_df = pd.DataFrame.from_dict(weather_data_by_facility, orient='index').T
weather_df.columns = facilities_with_location
monthly_reporting_by_facility = monthly_reporting_by_facility.set_index('facility').T
monthly_reporting_by_facility.index.name = "date"
# ### Save CSVs
monthly_reporting_by_facility = monthly_reporting_by_facility.loc[:, monthly_reporting_by_facility.columns.isin(facilities_with_location)]
monthly_reporting_by_facility = monthly_reporting_by_facility[facilities_with_location]

#monthly_reporting_by_facility.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_by_smaller_facility_lm.csv")
if ANC:
    monthly_reporting_by_facility.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_ANC_by_smaller_facility_lm.csv")
    weather_df.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv")
if Inpatient:
    monthly_reporting_by_facility.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_Inpatient_by_smaller_facility_lm.csv")
    weather_df.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_Inpatient_lm.csv")

else:
    monthly_reporting_by_facility.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_by_smaller_facility_lm.csv")
    weather_df.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facility_lm.csv")


## Get additional data - e.g. which zone it is in, altitude
included_facilities_with_lat_long = facilities_with_lat_long[
    facilities_with_lat_long["Fname"].isin(facilities_with_location)
]

additional_rows = ["Zonename", "Resid", "A105", "A109__Altitude", "Ftype", 'A109__Latitude', 'A109__Longitude']
expanded_facility_info = included_facilities_with_lat_long[["Fname"] + additional_rows]

expanded_facility_info.columns = ["Fname"] + additional_rows
expanded_facility_info.set_index("Fname", inplace=True)
# minimum distances between facilities
coordinates = expanded_facility_info[['A109__Latitude', 'A109__Longitude']].values
distances = cdist(coordinates, coordinates, metric='euclidean')
np.fill_diagonal(distances, np.inf)
expanded_facility_info['minimum_distance'] = np.nanmin(distances, axis=1)


expanded_facility_info = expanded_facility_info.T
expanded_facility_info = expanded_facility_info.reindex(columns=facilities_with_location)

if ANC:
    expanded_facility_info.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm_with_ANC.csv")
elif Inpatient:
    expanded_facility_info.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm_with_inpatient_days.csv")

else:
    expanded_facility_info.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm.csv")


