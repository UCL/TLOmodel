import os

import geopandas as gpd
import pandas as pd
from netCDF4 import Dataset
from random import randint
# Data accessed from https://dhis2.health.gov.mw/dhis-web-data-visualizer/#/YiQK65skxjz
# Reporting rate is expected reporting vs actual reporting - for DHO ("by district")
reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_change_health/Data/Reporting_Rate/Reporting_Rate_by_District_DHO_2015_2024.csv') #January 2000 - January 2024
# ANALYSIS DONE IN OCTOBER 2024 - so drop October, November, December 2024
columns_to_drop = reporting_data.columns[reporting_data.columns.str.endswith(('October 2024', 'November 2024', 'December 2024'))]

reporting_data = reporting_data.drop(columns=columns_to_drop)

### But need to drop mental health, as that is only relevant for the  Zomba Mental Hospital and otherwise brings down averages
# extract mental health data
mental_health_columns = reporting_data.columns[reporting_data.columns.str.startswith("Mental")].tolist()
reporting_data_no_mental = reporting_data.drop(mental_health_columns, axis = 1)
mental_health_data = reporting_data[[reporting_data.columns[1]] + mental_health_columns]
all_columns_no_mental_health = reporting_data_no_mental.columns

### now aggregate over months
monthly_reporting_data_by_facility =  {}
months = set(col.split(" - Reporting rate ")[1] for col in all_columns_no_mental_health if " - Reporting rate " in col)

# put in order
months = [date.strip() for date in months] # extra spaces??
dates = pd.to_datetime(months, format='%B %Y', errors='coerce')
months = dates.sort_values().strftime('%B %Y').tolist()
months = months[12*11:] # only want from 2011 on
for month in months:
    columns_of_interest_all_metrics = [reporting_data_no_mental.columns[1]] + reporting_data_no_mental.columns[reporting_data_no_mental.columns.str.endswith(month)].tolist()
    data_of_interest_by_month = reporting_data_no_mental[columns_of_interest_all_metrics]
    numeric_data = data_of_interest_by_month.select_dtypes(include='number')
    monthly_mean_by_facility = numeric_data.mean(axis=1)
    monthly_reporting_data_by_facility[month] = monthly_mean_by_facility
monthly_reporting_by_DHO = pd.DataFrame(monthly_reporting_data_by_facility)
monthly_reporting_by_DHO["facility"] = reporting_data_no_mental["organisationunitname"].values
monthly_reporting_by_DHO["region"] = reporting_data_no_mental["organisationunitcode"].values

# Weather data
directory = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical" # from 2011 on
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
grid = 0

regridded_weather_data = {}
for polygon in malawi_grid["geometry"]:
    minx, miny, maxx, maxy = polygon.bounds
    index_for_x_min = ((long_data - minx)**2).argmin()
    index_for_y_min = ((lat_data - miny)**2).argmin()
    index_for_x_max = ((long_data - maxx)**2).argmin()
    index_for_y_max = ((lat_data - maxy)**2).argmin()

    precip_data_for_grid = pr_data[:, index_for_y_min,index_for_x_min]  # across all time points
    precip_data_for_grid = precip_data_for_grid * 86400  # to get from per second to per day
    weather_by_grid[grid] = precip_data_for_grid
    grid += 1
# Load mapped facilities and find relevant shap file - Malawi grid goes from SE -> NE -> SW -> NW
weather_data_by_region = {}
for reporting_facility in range(len(monthly_reporting_by_DHO)):
    facility_data = monthly_reporting_by_DHO.loc[reporting_facility]
    region = facility_data["region"]

    if region in ["Central East Zone", "Central West Zone"]:
        grid = 2 # correspond directly to the grids in malawi_grid

    elif region == "North Zone":
        grid = randint(3,6)

    elif region == "South East Zone":
        grid = randint(7,9)

    elif region == "South West Zone":
        grid = 0

    weather_data_by_region[facility_data["facility"]] = weather_by_grid[grid]

### Get data ready for linear regression between reporting and weather data
weather_df = pd.DataFrame.from_dict(weather_data_by_region, orient='index').T
weather_df.columns = monthly_reporting_by_DHO["facility"]
monthly_reporting_by_DHO = monthly_reporting_by_DHO.set_index('facility').T

### Save CSVs
monthly_reporting_by_DHO.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_by_DHO_lm.csv")
weather_df.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_DHO_lm.csv")
