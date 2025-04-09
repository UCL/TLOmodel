import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import os


resourcefilepath = '/Users/rem76/PycharmProjects/TLOmodel/resources'


worldpop_density_data = pd.read_csv(
    Path(resourcefilepath) /'climate_change_impacts' / 'mwi_pd_2020_1km_ASCII_XYZ_worldpop.csv',
)

print(worldpop_density_data.head())

worldpop_density_data['Z'] = pd.to_numeric(worldpop_density_data['X'], errors='coerce')
worldpop_density_data['Y'] = pd.to_numeric(worldpop_density_data['Y'], errors='coerce')
worldpop_density_data['Z'] = pd.to_numeric(worldpop_density_data['Z'], errors='coerce')

geometry = [Point(xy) for xy in zip(worldpop_density_data['X'], worldpop_density_data['Y'])]
worldpop_gdf = gpd.GeoDataFrame(worldpop_density_data, geometry=geometry, crs="EPSG:4326")

malawi_admin2 = gpd.read_file(
    Path(resourcefilepath) / 'mapping' / 'ResourceFile_mwi_admbnda_adm2_nso_20181016.shp')

worldpop_gdf.set_crs("EPSG:4326", inplace=True)
malawi_admin2.set_crs("EPSG:4326", inplace=True)

print(worldpop_gdf.crs)
print(malawi_admin2.crs)
worldpop_gdf = worldpop_gdf.to_crs(malawi_admin2.crs)

joined = gpd.sjoin(worldpop_gdf, malawi_admin2[['ADM2_EN', 'geometry']], how='left', predicate='within')

joined['Z_proportion'] = joined.groupby('ADM2_EN')['Z'].transform(lambda x: x / x.sum())

print(joined)

joined.to_file(
    Path(resourcefilepath) / 'climate_change_impacts'/'worldpop_density_with_districts.shp',
    driver='ESRI Shapefile'
)
