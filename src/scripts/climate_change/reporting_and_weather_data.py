import os
import geopandas as gpd
import pandas as pd
from netCDF4 import Dataset
from shapely.geometry import Polygon

# Data accessed from https://dhis2.health.gov.mw/dhis-web-data-visualizer/#/YiQK65skxjz
# Reporting rate is expected reporting vs actual reporting
reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_change_health/Data/Reporting_Rate/Reporting_Rate_Central_Hospital_2000_2024.csv')

# Divide dataset based on what is being reported
# get metrics recorded
all_columns = reporting_data.columns
metrics = set([col.split(" - Reporting rate")[0] for col in all_columns])
metrics = {metric for metric in metrics if not metric.startswith("organisation")} # inlcude only reporting data


monthly_reporting_data_by_metric =  {}

for metric in metrics:
    columns_of_interest = [reporting_data.columns[1]] + reporting_data.columns[reporting_data.columns.str.startswith(metric)].tolist()
    data_of_interest = reporting_data[columns_of_interest]
    data_of_interest.columns = [col.replace(metric, "") for col in data_of_interest.columns]
    data_of_interest.columns = [col.replace(" - Reporting rate ", "") for col in data_of_interest.columns]
    monthly_reporting_data_by_metric[metric] = data_of_interest

### Actually don't want by metric - instead look across all dates for a given row and average (i.e. want to average by month by facility, not by metric by facility)

monthly_reporting_data_by_facility =  {}
months = set(col.split(" - Reporting rate ")[1] for col in all_columns if " - Reporting rate " in col)
# put in order
months = [date.strip() for date in months] # extra spaces??
dates = pd.to_datetime(months, format='%B %Y', errors='coerce')
months = dates.sort_values().strftime('%B %Y').tolist()

for month in months:
    columns_of_interest_all_metrics = [reporting_data.columns[1]] + reporting_data.columns[reporting_data.columns.str.endswith(month)].tolist()
    data_of_interest_by_month = reporting_data[columns_of_interest_all_metrics]
    numeric_data = data_of_interest_by_month.select_dtypes(include='number')
    monthly_mean_by_facility = numeric_data.mean(axis=1)
    monthly_reporting_data_by_facility[month] = monthly_mean_by_facility

monthly_reporting_by_facility = pd.DataFrame(monthly_reporting_data_by_facility)

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
regridded_lat_long = []
for polygon in malawi_grid["geometry"]:
    minx, miny, maxx, maxy = polygon.bounds
    index_for_x_min = ((long_data - minx)**2).argmin()
    index_for_y_min = ((lat_data - miny)**2).argmin()
    index_for_x_max = ((long_data - maxx)**2).argmin()
    index_for_y_max = ((lat_data - maxy)**2).argmin()
    print(lat_data[index_for_y_min])
    polygon_new = Polygon([
        (long_data[index_for_x_min], lat_data[index_for_y_min]),
        (long_data[index_for_x_max], lat_data[index_for_y_min]),
        (long_data[index_for_x_max], lat_data[index_for_y_max]),
        (long_data[index_for_x_min], lat_data[index_for_y_max]),
        (long_data[index_for_x_min], lat_data[index_for_y_min])
    ])
    regridded_lat_long.append(polygon_new)

regridded_lat_long = gpd.GeoDataFrame({'geometry': regridded_lat_long}, crs=malawi_grid.crs)

grid = 0
for i in range(len(long_data) - 1):
            for j in range(len(lat_data) - 1):
                precip_data_for_grid = pr_data[:, j, i]  # across all time points
                precip_data_for_grid = precip_data_for_grid * 86400  # to get from per second to per day
                weather_by_grid[grid] = precip_data_for_grid
                grid += 1



#
