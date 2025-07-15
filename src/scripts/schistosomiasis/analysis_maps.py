

import pyproj
# You can also call this for safety (redundant if env var is set correctly):
pyproj.datadir.set_data_dir('/Users/tmangal/anaconda3/envs/tlo/share/proj')

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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
pause_wash_best_strategy = read_csv(results_folder / f'pause_wash_best_strategy2024-2040.csv')
continue_wash_best_strategy = read_csv(results_folder / f'continue_wash_best_strategy2024-2040.csv')
scaleup_wash_best_strategy = read_csv(results_folder / f'scaleup_wash_best_strategy2024-2040.csv')



df = pd.DataFrame({
    'district': ['Blantyre', 'Lilongwe', 'Mzimba'],
    'value': [0.8, 0.5, 0.6]
})

# --- Prepare for merge ---
# rename admin2 column
gdf.rename(columns={'ADM2_EN': 'district'}, inplace=True)
map_with_nhb_diffs_pause = gdf.merge(pause_wash_best_strategy, on='district', how='left')  # 'left' keeps all districts
map_with_nhb_diffs_continue = gdf.merge(continue_wash_best_strategy, on='district', how='left')  # 'left' keeps all districts
map_with_nhb_diffs_scaleup = gdf.merge(scaleup_wash_best_strategy, on='district', how='left')  # 'left' keeps all districts


# --- Plot  map ---

fig, axs = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

maps = [map_with_nhb_diffs_pause, map_with_nhb_diffs_continue, map_with_nhb_diffs_scaleup]
titles = [
    "Pause WASH",
    "Continue WASH",
    "Scale-up WASH"
]

# Determine global min and max for consistent color scaling
vmin = min(gdf['mean'].min() for gdf in maps)
vmax = max(gdf['mean'].max() for gdf in maps)

# Define colormap and normalizer
cmap = 'viridis'
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

for ax, gdf, title in zip(axs, maps, titles):
    base = gdf.plot(
        column='mean',
        cmap=cmap,
        linewidth=0.8,
        edgecolor='0.8',
        ax=ax,
        legend=False,  # Disable individual legend
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data"
        }
    )

    gdf_national.boundary.plot(ax=ax, edgecolor='grey', linewidth=1.5)

    # highlight districts if preferred strategy MDA All
    # highlight = gdf[gdf['draw'].str.contains("MDA All", na=False)]
    # highlight.plot(
    #     hatch='///',
    #     edgecolor='black',
    #     linewidth=1.5,
    #     facecolor="none",
    #     ax=base,
    #     legend=False
    # )
    # add asterisk if preferred strategy is MDA All
    for idx, row in gdf.iterrows():
        if isinstance(row['draw'], str) and "MDA All" in row['draw']:
            centroid = row['geometry'].centroid
            ax.text(centroid.x, centroid.y, '*', fontsize=24, ha='center', va='center', color='red')

    ax.set_title(title, fontsize=14)
    ax.axis('off')

# Add one shared colorbar for all three maps
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height] (adjust as needed)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array for colorbar
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Net Health Benefit (difference)', fontsize=12)

plt.show()

"""
The map shows the district-level net health benefit (NHB) of alternative MDA strategies
relative to the current policy of targeting school-age children (MDA SAC).
NHB is calculated as the difference between health gains (e.g., DALYs averted) and
the health opportunity costs of the intervention, where costs are converted to
health units using a predefined willingness-to-pay threshold.
Positive NHB values indicate that the alternative strategy provides greater net
health gains after considering both its benefits and costs, whereas negative values
suggest that the costs may outweigh the health benefits compared to the baseline.
"""
