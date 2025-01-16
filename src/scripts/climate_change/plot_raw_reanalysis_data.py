import glob
import os
import re

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from netCDF4 import Dataset
from shapely.geometry import Polygon

# Load the dataset and the variable
file_path_historical_data = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/2011/60ab007aa16d679a32f9c3e186d2f744.nc"
dataset = Dataset(file_path_historical_data, mode='r')
print(dataset.variables.keys())
pr_data = dataset.variables['tp'][:]  # ['pr'][:] pr for projections, tp for historical
lat_data = dataset.variables['latitude'][:]  # ['lat'][:]
long_data = dataset.variables['longitude'][:]  # ['lon'][:]
meshgrid_from_netCDF = np.meshgrid(long_data, lat_data)

# Load Malawi shapefile
malawi = gpd.read_file(
    "/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm0_nso_20181016.shp")
malawi_admin1 = gpd.read_file(
    "/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm1_nso_20181016.shp")
malawi_admin2 = gpd.read_file(
    "/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")

difference_lat = lat_data[1] - lat_data[0]  # as is a grid, the difference is the same for all sequential coordinates
difference_long = long_data[1] - long_data[0]

polygons = []
for x in long_data:
    for y in lat_data:
        bottom_left = (x, y)
        bottom_right = (x + difference_long, y)
        top_right = (x + difference_long, y + difference_lat)
        top_left = (x, y + difference_lat)
        polygon = Polygon([bottom_left, bottom_right, top_right, top_left])
        polygons.append(polygon)

grid = gpd.GeoDataFrame({'geometry': polygons}, crs=malawi.crs)
grid_clipped = gpd.overlay(grid, malawi, how='intersection')  # for graphing
grid_clipped_ADM1 = gpd.overlay(grid, malawi_admin1, how='intersection')  # for graphing
grid_clipped_ADM2 = gpd.overlay(grid, malawi_admin2, how='intersection')  # for graphing

# Setup color map for plotting grid lines
colors = cm.get_cmap("tab20", 20)

# Corrected part for processing model files
nc_file_directory = "/path/to/your/nc_files"  # Define this correctly
fig, ax = plt.subplots(figsize=(10, 10))  # Ensure you create the axis before plotting
for idx, file in enumerate(glob.glob(os.path.join(nc_file_directory, "*.nc"))):
    data_per_model = Dataset(file, mode='r')
    pr_data_model = data_per_model.variables['pr'][:]  # in kg m-2 s-1 = mm s-1 x 86400 to get to day
    lat_data_model = data_per_model.variables['lat'][:]
    long_data_model = data_per_model.variables['lon'][:]

    # Plot grid lines for this model file
    for lon in long_data_model:
        ax.axvline(x=lon, color=colors(idx), linestyle='--', linewidth=0.5)
    for lat in lat_data_model:
        ax.axhline(y=lat, color=colors(idx), linestyle='--', linewidth=0.5)

# Add in facility information
expanded_facility_info = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm_with_ANC.csv",
    index_col=0
)

long_format = expanded_facility_info.T.reset_index()
long_format.columns = [
    'Facility', 'Zonename', 'Resid', 'Dist', 'A105', 'A109__Altitude', 'Ftype',
    'A109__Latitude', 'A109__Longitude', 'minimum_distance', 'average_precipitation'
]

long_format = long_format.dropna(subset=['A109__Latitude'])

facilities_gdf = gpd.GeoDataFrame(
    long_format,
    geometry=gpd.points_from_xy(long_format['A109__Longitude'], long_format['A109__Latitude']),
    crs="EPSG:4326"
)

facilities_gdf['average_precipitation'] = pd.to_numeric(facilities_gdf['average_precipitation'], errors='coerce')

norm = mcolors.Normalize(vmin=facilities_gdf['average_precipitation'].min(),
                         vmax=facilities_gdf['average_precipitation'].max())
cmap_facilities = plt.cm.YlOrBr
facilities_gdf['color'] = facilities_gdf['average_precipitation'].apply(lambda x: cmap_facilities(norm(x)))

# Plotting facilities on the map
malawi_admin2.plot(ax=ax, edgecolor='black', color='white')
grid_clipped_ADM2.plot(ax=ax, edgecolor='#1C6E8C', alpha=0.4)
grid_clipped_ADM1.plot(column='ADM1_EN', ax=ax, cmap=colors, edgecolor='#1C6E8C', alpha=0.7)

facilities_gdf.plot(ax=ax, color=facilities_gdf['color'], markersize=10)

sm = plt.cm.ScalarMappable(cmap=cmap_facilities, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
cbar.set_label('Mean Monthly Precipitation (mm)')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig('/Users/rem76/Desktop/Climate_change_health/Results/ANC_disruptions/historical_weather.png')

plt.show()
