import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# Load Malawi shapefile
malawi = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm0_nso_20181016.shp")
malawi_admin1 = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm1_nso_20181016.shp")
malawi_admin2 = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
grid_size = 1
minx, miny, maxx, maxy = malawi.total_bounds
x_coords = np.arange(minx, maxx, grid_size)
y_coords = np.arange(miny, maxy, grid_size)
polygons = [Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]) for x in x_coords for y in y_coords]

grid = gpd.GeoDataFrame({'geometry': polygons}, crs=malawi.crs)
grid_clipped = gpd.overlay(grid, malawi, how='intersection')
grid_clipped_ADM1 = gpd.overlay(grid, malawi_admin1, how='intersection')

cmap = plt.cm.get_cmap('tab20', len(grid_clipped_ADM1['ADM1_EN'].unique()))

fig, ax = plt.subplots(figsize=(10, 10))
malawi_admin2.plot(ax=ax, edgecolor='black', color='white')
grid_clipped.plot(ax=ax, edgecolor='#1C6E8C', color='#9AC4F8', alpha=0.5)
grid_clipped_ADM1.plot(column='ADM1_EN', ax=ax, cmap=cmap, edgecolor='#1C6E8C', alpha=0.7, legend=True)

# Finalize plot
plt.title("Malawi with Overlaying Grids - 1 degree")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

### Intersection between the grid and the admin areas ###
grid_with_admin_areas = gpd.sjoin(malawi_admin2, grid_clipped, how='left', predicate='intersects')

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
# max/min of each polygon

# join so each facility has a grid
facilities_with_districts_shap_files = facilities_by_area.merge(grid_with_admin_areas,
                                                     how='left',
                                                     left_on='District',
                                                     right_on='ADM2_EN') # will have what grid cell they're in

# write csv file of facilities with districts
facilities_with_districts_shap_files.to_csv("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.csv")
