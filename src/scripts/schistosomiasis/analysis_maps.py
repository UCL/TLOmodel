

import pyproj
# You can also call this for safety (redundant if env var is set correctly):
pyproj.datadir.set_data_dir('/Users/tmangal/anaconda3/envs/tlo/share/proj')

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pandas import read_csv

from tlo.analysis.utils import (
    get_scenario_outputs,
)

# --- Load shapefile of Malawi districts ---
# Adjust this path to point to your .shp file (the others must be in the same folder)
shapefile_path = "resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
gdf = gpd.read_file(shapefile_path)

shapefile_path_national = "resources/mapping/ResourceFile_mwi_admbnda_adm0_nso_20181016.shp"
gdf_national = gpd.read_file(shapefile_path_national)

output_folder = Path("./outputs/t.mangal@imperial.ac.uk")
results_folder = get_scenario_outputs("schisto_scenarios.py", output_folder)[-1]


# --- Create or load data with values per district ---
# Example: dummy data for 3 districts; you should replace this with your actual data
nhb = read_csv(results_folder / f'nhb_district2024-2040.csv')



df = pd.DataFrame({
    'district': ['Blantyre', 'Lilongwe', 'Mzimba'],
    'value': [0.8, 0.5, 0.6]
})

# --- Prepare for merge ---
# rename admin2 column
gdf.rename(columns={'ADM2_EN': 'district'}, inplace=True)
merged = gdf.merge(df, on='district', how='left')  # 'left' keeps all districts


# --- Plot  map ---
fig, ax = plt.subplots(figsize=(8, 10))
merged.plot(
    column='value',
    cmap='viridis',
    linewidth=0.8,
    edgecolor='0.8',
    ax=ax,
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "white",
        "hatch": "///",
        "label": "No data"
    }
)

gdf_national.boundary.plot(ax=ax, edgecolor='grey', linewidth=1.5)

ax.set_title("District-level Values in Malawi", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()
