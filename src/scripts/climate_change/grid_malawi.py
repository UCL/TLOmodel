import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from shapely.geometry import Polygon

# Load netCDF data for gridding info
file_path = "/Users/rem76/Downloads/821bebfbcee0609d233c09e8b2bbc1f3/pr_Amon_UKESM1-0-LL_ssp119_r1i1p1f2_gn_20150116-20991216.nc"
dataset = Dataset(file_path, mode='r')
print(dataset.variables.keys())
pr_data = dataset.variables['pr'][:]
time_data = dataset.variables['time'][:]
lat_data = dataset.variables['lat'][:]
long_data = dataset.variables['lon'][:]
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
grid.to_file("/Users/rem76/Desktop/Climate_change_health/Data/malawi_grid.shp")

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

### Intersection between the grid and the admin areas ###
intersections = gpd.overlay(grid, malawi_admin2, how='intersection') # 56 intersections between districts and major grid squares

admin_area_with_major_grid = gpd.overlay( grid, malawi_admin2, how='intersection') # finding which grid each facility falls into

########### Create new table with facilities and add coordinates to each facility #########

facilities_by_area = pd.read_csv("/Users/rem76/PycharmProjects/TLOmodel/resources/healthsystem/organisation/ResourceFile_Master_Facilities_List.csv")

# Referral hospitals have no assigned district - assign biggest city in region?
facilities_by_area.loc[facilities_by_area['Facility_Name'] == 'Referral Hospital_Southern', 'District'] = 'Blantyre City'
facilities_by_area.loc[facilities_by_area['Facility_Name'] == 'Referral Hospital_Central', 'District'] = 'Lilongwe City'
facilities_by_area.loc[facilities_by_area['Facility_Name'] == 'Referral Hospital_Northern', 'District'] = 'Mzuzu City'

# Mental hospital is in Zomba
facilities_by_area.loc[facilities_by_area['Facility_Name'] == 'Zomba Mental Hospital', 'District'] = 'Zomba City'
facilities_by_area.loc[facilities_by_area['Facility_Name'] == 'Zomba Mental Hospital', 'Region'] = 'Southern'

# HQ based in Lilongwe?
facilities_by_area.loc[facilities_by_area['Facility_Name'] == 'Headquarter', 'District'] = 'Lilongwe City'
facilities_by_area.loc[facilities_by_area['Facility_Name'] == 'Headquarter', 'Region'] = 'Central'
# join so each facility has a grid

facilities_with_districts_shap_files = facilities_by_area.merge(admin_area_with_major_grid,
                                                     how='left',
                                                     left_on='District',
                                                     right_on = 'ADM2_EN') # will have what grid cell they're in
# Facilities do not go smaller than region... Which may have multiple polygons
# So, because of the fact one district may overlap with many grids, there are many "duplicates"
# in facilities_with_districts_shap_files (as each facility is paired with any matching grid)
# removing the duplicates PENDING a better system (e.g. assigning based on size)

facilities_with_districts_shap_files_no_duplicates = facilities_with_districts_shap_files.drop_duplicates(subset=['District', 'Facility_Level', 'Region', 'Facility_ID', 'Facility_Name'])
facilities_with_districts_shap_files_no_duplicates.reset_index(drop=True, inplace=True)


# write csv file of facilities with districts
facilities_with_districts_shap_files_no_duplicates.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.csv")

facilities_gdf = gpd.GeoDataFrame(facilities_with_districts_shap_files_no_duplicates,
                                   geometry='geometry',
                                   crs="EPSG:4326")
facilities_gdf.to_file("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.shp")
