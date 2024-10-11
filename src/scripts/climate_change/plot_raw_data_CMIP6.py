import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
from netCDF4 import Dataset

# Load the dataset and the variable
file_path = "/Users/rem76/Downloads/821bebfbcee0609d233c09e8b2bbc1f3/pr_Amon_UKESM1-0-LL_ssp119_r1i1p1f2_gn_20150116-20991216.nc"
dataset = Dataset(file_path, mode='r')
pr_data = dataset.variables['pr'][:] # in kg m-2 s-1 = mm s-1 x 86400 to get to day
time_data = dataset.variables['time'][:]
lat_data = dataset.variables['lat'][:]
long_data = dataset.variables['lon'][:]

## Initial plot
pr_data_time_series_grid_1 = pr_data[:,5,1]
pr_data_time_series_grid_1 *= 86400 # to get to days

# Plot the 2D data
plt.plot(pr_data_time_series_grid_1)

plt.title('Average Precipitation Over Time - Grid 1')
plt.ylabel('Precip (mm)')
plt.xlabel('Time')
plt.show()

################ Now do it by specific regions #################
# get regions from facilities file
facilities_with_grid = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.csv")
facilities_with_grid = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.shp")
print(facilities_with_grid)
bounds_data = []
for geometry in facilities_with_grid["geometry"]:
    minx, miny, maxx, maxy = geometry.bounds  # Get bounding box
    bounds_data.append({'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy})
bounds_df = pd.DataFrame(bounds_data)
facilities_with_bounds = pd.concat([facilities_with_grid.reset_index(drop=True), bounds_df], axis=1)

# could get centroids of the grid and see if they overlap with
centroid_data = []
for geometry in facilities_with_grid["geometry"]:
    central_x = geometry.centroid.x
    central_y = geometry.centroid.y
    centroid_data.append({'central_x': central_x, 'central_y': central_y})


## can find min square distance and index to get relavent indices
diff_northern_lat = []
diff_southern_lat = []
diff_eastern_lon = []
diff_western_lon = []

max_latitudes = facilities_with_bounds['maxy'].values
min_latitudes = facilities_with_bounds['miny'].values
max_longitudes = facilities_with_bounds['maxx'].values
min_longitudes = facilities_with_bounds['minx'].values
# Iterate over each facility to calculate differences
for i in range(len(facilities_with_bounds)):
    # Calculate the square differences for latitude and longitude, to then find nearest index
    northern_diff = (lat_data - max_latitudes[i]) ** 2
    southern_diff = (lat_data - min_latitudes[i]) ** 2
    eastern_diff = (long_data - max_longitudes[i]) ** 2
    western_diff = (long_data - min_longitudes[i]) ** 2

    diff_northern_lat.append(northern_diff.argmin())
    diff_southern_lat.append(southern_diff.argmin())
    diff_eastern_lon.append(eastern_diff.argmin())
    diff_western_lon.append(western_diff.argmin())

print(diff_northern_lat)
print(diff_southern_lat)
print(diff_eastern_lon)
