import glob
import os
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from shapely.geometry import Polygon
import matplotlib.cm as cm

# Load netCDF data for gridding info
#file_path = "/Users/rem76/Downloads/821bebfbcee0609d233c09e8b2bbc1f3/pr_Amon_UKESM1-0-LL_ssp119_r1i1p1f2_gn_20150116-20991216.nc"
file_path_historical_data = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/139ef85ab4df0a12fc01854395fc9a6d.nc"
dataset = Dataset(file_path_historical_data, mode='r')
print(dataset.variables.keys())
pr_data = dataset.variables['tp'][:] # ['pr'][:] pr for projections, tp for historical
lat_data = dataset.variables['latitude'][:] #['lat'][:]
long_data = dataset.variables['longitude'][:] #['lon'][:]
meshgrid_from_netCDF = np.meshgrid(long_data, lat_data)

# Load Malawi shapefile
malawi = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm0_nso_20181016.shp")
malawi_admin1 = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm1_nso_20181016.shp")
malawi_admin2 = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
#grid_size = 1
#minx, miny, maxx, maxy = malawi.total_bounds
#x_coords = np.arange(minx, maxx, grid_size) my gridding doesn't work - based on a different projection, maybe?
#y_coords = np.arange(miny, maxy, grid_size)
#polygons = [Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]) for x in x_coords for y in y_coords]

difference_lat = lat_data[1] - lat_data[0] # as is a grid, the difference is the same for all sequential coordinates
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
grid_clipped = gpd.overlay(grid, malawi, how='intersection') # for graphing
grid_clipped_ADM1 = gpd.overlay(grid, malawi_admin1, how='intersection') # for graphing
grid_clipped_ADM2 = gpd.overlay(grid, malawi_admin2, how='intersection') # for graphing
cmap = plt.cm.get_cmap('tab20', len(grid_clipped_ADM1['ADM1_EN'].unique()))
grid.to_file("/Users/rem76/Desktop/Climate_change_health/Data/malawi_grid_0_025.shp")

fig, ax = plt.subplots(figsize=(10, 10))
malawi_admin2.plot(ax=ax, edgecolor='black', color='white')
grid.plot(ax=ax, edgecolor='#1C6E8C',  color='white')
grid_clipped_ADM2.plot(ax=ax,edgecolor='#1C6E8C', alpha=0.4)
grid_clipped_ADM1.plot(column='ADM1_EN', ax=ax, cmap=cmap, edgecolor='#1C6E8C', alpha=0.7)

# Finalize plot
plt.title("Malawi with Overlaying Grids")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
#plt.show()


### SSP25 model grid
base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/"
scenarios = ["ssp2_4_5"]

file_list = glob.glob(os.path.join(base_dir, "*.nc"))
colors = cm.get_cmap("tab20", 20)

data_by_model_and_grid = {}
for scenario in scenarios:
    print(scenario)
    scenario_directory = os.path.join(base_dir, scenario)
    nc_file_directory = os.path.join(scenario_directory, 'nc_files')
    for idx, file in enumerate(glob.glob(os.path.join(nc_file_directory, "*.nc"))):
        model = re.search(r'pr_day_(.*?)_' + scenario.replace('_', ''), file).group(1)
        data_per_model = Dataset(file, mode='r')
        pr_data = data_per_model.variables['pr'][:]  # in kg m-2 s-1 = mm s-1 x 86400 to get to day
        lat_data = data_per_model.variables['lat'][:]
        long_data = data_per_model.variables['lon'][:]
        print(colors(idx))
        for lon in long_data:
            ax.axvline(x=lon, color=colors(idx), linestyle='--', linewidth=0.5)
        for lat in lat_data:
            ax.axhline(y=lat, color=colors(idx), linestyle='--', linewidth=0.5)

        # Customize your plot as needed
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')


plt.show()
