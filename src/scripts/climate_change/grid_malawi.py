import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt

# Load Malawi shapefile
malawi = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm0_nso_20181016.shp")
malawi_admin1 = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm1_nso_20181016.shp")
malawi_admin2 = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
print(malawi_admin2)
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


########### Create new table with facilities and add coordinates to each facility

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


facilities_with_districts = facilities_by_area.merge(malawi_admin2,
                                                     how='left',
                                                     left_on='District',
                                                     right_on='ADM2_EN')

print(facilities_with_districts)
