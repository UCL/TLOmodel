

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

shapefile_path_regional = "resources/mapping/ResourceFile_mwi_admbnda_adm1_nso_20181016.shp"
gdf_regional = gpd.read_file(shapefile_path_regional)

# Load Natural Earth lake polygons (downloaded beforehand)
# https://www.naturalearthdata.com/downloads/50m-physical-vectors/
lakes = gpd.read_file("resources/mapping/ne_50m_lakes.shp")  # from Natural Earth 50 m lakes dataset
# Filter for Lake Malawi
# Drop rows where 'name' is NaN before filtering
lake_malawi = lakes[lakes['name'].notna() & lakes['name'].str.contains("Malawi", case=False)]




output_folder = Path("./outputs/t.mangal@imperial.ac.uk")
results_folder = get_scenario_outputs("schisto_scenarios-2025.py", output_folder)[-1]


# --- load data with values per district ---
pause_wash_best_strategy = read_csv(results_folder / f'pause_wash_nhb_vs_SAC_2024-2050.csv')
continue_wash_best_strategy = read_csv(results_folder / f'continue_wash_nhb_vs_SAC_2024-2050.csv')
scaleup_wash_best_strategy = read_csv(results_folder / f'scaleup_wash_nhb_vs_SAC_2024-2050.csv')


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

####################################################################################
# %%  MAP: NHB with preferred policy + YEAR REACHING EPHP haem and mansoni
####################################################################################


# match draw names from preferred stratgy to draw name first_years_ephp_df_haem
# todo these are probably 2% limits like the KM plots
first_years_ephp_df_haem = pd.read_excel(results_folder / 'first_years_haem HML_2percent_2024-2050.xlsx')
first_years_ephp_df_mansoni = pd.read_excel(results_folder / 'first_years_mansoni HML_2percent 2024-2050.xlsx')

# Merge to get year_ephp for the best strategy under Continue WASH
continue_wash_best_strategy_with_ehph = continue_wash_best_strategy.merge(
    first_years_ephp_df_haem.rename(columns={"year_ephp": "year_ephp_haem"}),
    on=["district", "draw"],
    how="left"
)

continue_wash_best_strategy_with_ehph = continue_wash_best_strategy_with_ehph.merge(
    first_years_ephp_df_mansoni.rename(columns={"year_ephp": "year_ephp_mansoni"}),
    on=["district", "draw"],
    how="left"
)

# add these data to the map file
map_with_nhb_diffs_ephp = map_with_nhb_diffs_continue.merge(
    continue_wash_best_strategy_with_ehph, on='district', how='left')  # 'left' keeps all districts


#---------------- maps for paper


fig, axs = plt.subplots(1, 3, figsize=(15, 8))

# Define maps and their respective columns and titles
maps = [
    (map_with_nhb_diffs_ephp, 'mean_x', "Net Health Benefit", 'viridis'),
    (map_with_nhb_diffs_ephp, 'year_ephp_haem', "Elimination Year: S. Haematobium", 'plasma'),
    (map_with_nhb_diffs_ephp, 'year_ephp_mansoni', "Elimination Year: S. Mansoni", 'plasma'),
]

# Align CRS for all datasets
target_crs = "EPSG:4326"

for g in [map_with_nhb_diffs_ephp, lake_malawi, gdf_national]:
    if g.crs is None:
        g.set_crs(target_crs, inplace=True)   # assign if missing (assumes it's already lat/lon)
    else:
        g.to_crs(target_crs, inplace=True)    # reproject if necessary

for i, (ax, (gdf, column, title, cmap)) in enumerate(zip(axs, maps)):
    # Set normalisation
    if i == 0:
        # Dynamic range for first map
        vmin = gdf[column].min()
        vmax = gdf[column].max()
    else:
        # Fixed range for years on second and third maps
        vmin, vmax = 2024, 2050

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Plot lake with hatch and border
    lake_malawi.plot(
        ax=ax,
        facecolor='none',
        edgecolor='lightblue',
        linewidth=0.5,
        hatch='///',
        zorder=1
    )
    # lake_malawi.boundary.plot(
    #     ax=ax,
    #     edgecolor='slategrey',
    #     linewidth=1.2,
    #     zorder=2
    # )

    # Prepare missing data style for second and third maps
    if i == 0:
        missing_kwds = {
            "color": "white",
            "edgecolor": "lightgrey",
            "hatch": "///",
            "label": "No data"
        }
    else:
        # For maps 2 & 3: no shading, leave white for missing data
        missing_kwds = {
            "color": "white",
            "edgecolor": "lightgrey",
            "label": "No data"
        }

    # Plot districts
    gdf.plot(
        column=column,
        cmap=cmap,
        linewidth=0.8,
        edgecolor='0.8',
        ax=ax,
        legend=False,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        zorder=3,
        missing_kwds=missing_kwds
    )

    # Plot national boundaries
    gdf_national.boundary.plot(ax=ax, edgecolor='grey', linewidth=1.5, zorder=4)

    # Add asterisks only on first map
    # if i == 0:
    #     for idx, row in gdf.iterrows():
    #         if isinstance(row['draw_x'], str) and "MDA All" in row['draw_x']:
    #             centroid = row['geometry'].centroid
    #             ax.text(centroid.x, centroid.y, '*', fontsize=24, ha='center', va='center', color='red', zorder=5)
    if "district" in gdf.columns:
        islands = gdf[gdf["district"].str.contains("Likoma|Chizumulu", case=False, na=False)]
        if not islands.empty:
            islands.plot(
                column=column,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                ax=ax,
                edgecolor="none",
                linewidth=0,
                legend=False,
                zorder=9,  # on top of everything
            )

    ax.set_title(title, fontsize=14)
    ax.axis('off')

# Add individual colourbars per map on top of each axis
for ax, (gdf, column, title, cmap) in zip(axs, maps):
    if title == "Net Health Benefit":
        vmin = gdf[column].min()
        vmax = gdf[column].max()
    else:
        vmin, vmax = 2024, 2050
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    # cbar.set_label(title, fontsize=12)
plt.subplots_adjust(wspace=0.1)  # reduce horizontal space between plots (default is ~0.2)
fig.savefig('maps_nhb_ephp.png', dpi=300, bbox_inches='tight')

plt.show()
