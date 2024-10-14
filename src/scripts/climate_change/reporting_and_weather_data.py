import os
import geopandas as gpd
import pandas as pd
from netCDF4 import Dataset
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data accessed from https://dhis2.health.gov.mw/dhis-web-data-visualizer/#/YiQK65skxjz
# Reporting rate is expected reporting vs actual reporting
reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_change_health/Data/Reporting_Rate/Reporting_Rate_Central_Hospital_2000_2024.csv')

### Actually don't want by metric - instead look across all dates for a given row and average (i.e. want to average by month by facility, not by metric by facility)
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
monthly_reporting_by_facility = pd.DataFrame(monthly_reporting_data_by_facility)
monthly_reporting_by_facility["facility"] = reporting_data_no_mental["organisationunitname"].values

# Weather data
directory = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical"
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
# Load facilities file
general_facilities = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.shp")
# find relavent shap file
weather_data_by_facility = {}
for reporting_facility in monthly_reporting_by_facility["facility"]:
    if (reporting_facility == "Central Hospital") or reporting_facility == "Kamuzu Central Hospital":
        # which malawi grid this is
        grid = general_facilities[general_facilities["District"] == "Lilongwe City"]["Grid_Index"].iloc[0] # all labelled X City will be in the same grid
    elif reporting_facility == "Mzuzu Central Hospital":
        grid = general_facilities[general_facilities["District"] == "Mzuzu City"]["Grid_Index"].iloc[0]
    elif reporting_facility == "Queen Elizabeth Central Hospital":
        grid = general_facilities[general_facilities["District"] == "Blantyre City"]["Grid_Index"].iloc[0]
    elif (reporting_facility == "Zomba Central Hospital") or (reporting_facility == "Zomba Mental Hospital"):
        grid = general_facilities[general_facilities["District"] == "Zomba City"]["Grid_Index"].iloc[0]
    elif reporting_facility == "Central East Zone":
        grid = general_facilities[general_facilities["District"] == "Nkhotakota"]["Grid_Index"].iloc[0] # furtherst east zone

    weather_data_by_facility[reporting_facility] = weather_by_grid[grid]


### Linear regression between reporting and weather data
# prep for linear regression
weather_df = pd.DataFrame.from_dict(weather_data_by_facility, orient='index').T
weather_df.columns = monthly_reporting_by_facility["facility"]
monthly_reporting_by_facility = monthly_reporting_by_facility.set_index('facility').T

X = weather_df.values.flatten()
y = monthly_reporting_by_facility.values.flatten()
if X.ndim == 1:
    X = X.reshape(-1, 1)
if y.ndim == 1:
    y = y.reshape(-1, 1)

print(len(X), len(y))

# Perform linear regression
model = LinearRegression()
model.fit(X[0:len(y)], y)
y_pred = model.predict(X[0:len(y)])

# Evaluate the model
r2 = r2_score(y, y_pred)
print(f'R-squared: {r2:.2f}')
print(f'Coefficient: {model.coef_[0]:.2f}')
print(f'Intercept: {model.intercept_:.2f}')
