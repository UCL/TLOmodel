

# import pyproj
# # You can also call this for safety (redundant if env var is set correctly):
# pyproj.datadir.set_data_dir('/Users/tmangal/anaconda3/envs/tlo/share/proj')

import os
os.environ["PROJ_LIB"] = "/opt/homebrew/Caskroom/miniforge/base/envs/tlo_arm/share/proj"

import pyproj
from pyproj import CRS
print(CRS.from_epsg(4326))


import geopandas as gpd
import pandas as pd
import numpy as np
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
pause_wash_best_strategy = read_csv(results_folder / f'pause_wash_nhb_vs_SAC_financial_2024-2050.csv')
continue_wash_best_strategy = read_csv(results_folder / f'continue_wash_nhb_vs_SAC_financial_2024-2050.csv')
scaleup_wash_best_strategy = read_csv(results_folder / f'scaleup_wash_nhb_vs_SAC_financial_2024-2050.csv')


# --- Prepare for merge ---
# rename admin2 column
gdf.rename(columns={'ADM2_EN': 'district'}, inplace=True)
map_with_nhb_diffs_pause = gdf.merge(pause_wash_best_strategy, on='district', how='left')  # 'left' keeps all districts
map_with_nhb_diffs_continue = gdf.merge(continue_wash_best_strategy, on='district', how='left')  # 'left' keeps all districts
map_with_nhb_diffs_scaleup = gdf.merge(scaleup_wash_best_strategy, on='district', how='left')  # 'left' keeps all districts


# --- Plot  NHB map with different WASH assumptions ---

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
first_years_ephp_df_mansoni = pd.read_excel(results_folder / 'first_years_mansoni_HML_2percent 2024-2050.xlsx')



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

for i, (ax, (gdf_in, column, title, cmap)) in enumerate(zip(axs, maps)):

    gdf = gdf_in.copy()

    # Treat -99 as missing for elimination-year maps (plots 2 & 3)
    if i in (1, 2):
        gdf[column] = pd.to_numeric(gdf[column], errors="coerce")
        gdf[column] = gdf[column].where(gdf[column] != -99, np.nan)

    # Set normalisation
    if i == 0:
        # Dynamic range for first map
        vmin = gdf[column].min()
        vmax = gdf[column].max()
    else:
        # Fixed range for years on second and third maps
        vmin, vmax = 2010, 2050

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
            # "hatch": "///",
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
        vmin, vmax = 2010, 2050
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    # cbar.set_label(title, fontsize=12)
plt.subplots_adjust(wspace=0.1)  # reduce horizontal space between plots (default is ~0.2)
fig.savefig('maps_nhb_ephp.png', dpi=300, bbox_inches='tight')

plt.show()



#---------------- plot set of maps for each draw

continue_wash_all_strategy = read_csv(
    results_folder / f'nhb_district_vs_noMDA_financial_costs_88_2024-2050.csv')


# Merge to get year_ephp for the best strategy under Continue WASH
continue_wash_all_strategy_with_ehph = continue_wash_all_strategy.merge(
    first_years_ephp_df_haem.rename(columns={"year_ephp": "year_ephp_haem"}),
    on=["district", "draw"],
    how="left"
)

continue_wash_all_strategy_with_ehph = continue_wash_all_strategy_with_ehph.merge(
    first_years_ephp_df_mansoni.rename(columns={"year_ephp": "year_ephp_mansoni"}),
    on=["district", "draw"],
    how="left"
)


df_sac = continue_wash_all_strategy_with_ehph.loc[
    continue_wash_all_strategy_with_ehph["draw"].eq("Continue WASH, MDA SAC")
].copy()

gdf_sac = gdf.merge(
    df_sac,
    how="left",
    left_on="district",     # or "District"
    right_on="district"     # or "District"
)

df_psac_sac = continue_wash_all_strategy_with_ehph.loc[
    continue_wash_all_strategy_with_ehph["draw"].eq("Continue WASH, MDA PSAC+SAC")
].copy()

gdf_psac_sac = gdf.merge(
    df_psac_sac,
    how="left",
    left_on="district",     # or "District"
    right_on="district"     # or "District"
)

df_all= continue_wash_all_strategy_with_ehph.loc[
    continue_wash_all_strategy_with_ehph["draw"].eq("Continue WASH, MDA All")
].copy()

gdf_all = gdf.merge(
    df_all,
    how="left",
    left_on="district",     # or "District"
    right_on="district"     # or "District"
)


gdfs = (gdf_sac, gdf_psac_sac, gdf_all)






def plot_3x3_maps(
    gdfs: tuple["gpd.GeoDataFrame", "gpd.GeoDataFrame", "gpd.GeoDataFrame"],
    lake_malawi: "gpd.GeoDataFrame",
    gdf_national: "gpd.GeoDataFrame",
    row_titles: tuple[str, str, str] = (
        "MDA SAC",
        "MDA PSAC+SAC",
        "MDA All",
    ),
    target_crs: str = "EPSG:4326",
    nhb_col: str = "mean",
    haem_col: str = "year_ephp_haem",
    mansoni_col: str = "year_ephp_mansoni",
    nhb_cmap: str = "viridis",
    nhb_pct_cmap: str = "RdBu_r",
    year_cmap: str = "plasma",
    year_vmin: int = 2010,
    year_vmax: int = 2050,
    figsize: tuple[int, int] = (16, 18),
    panel_labels: list[str] | None = None,   # e.g. list("ABCDEFGHI")
    label_kwargs: dict | None = None,
    savepath: str | None = None,
    pct_eps: float = 1e-9,
    # NEW: cap % maps and label colourbars with ≤/≥ at the ends
    pct_cap: float = 100.0,
):
    """
    3x3 grid:
      rows = (SAC, PSAC+SAC, All) provided as three GeoDataFrames
      cols = NHB, elim year haem, elim year mansoni

    NHB panels:
      - Row 1, Col 1: absolute NHB for SAC
      - Row 2, Col 1: % change in NHB for PSAC+SAC vs SAC (baseline SAC)
      - Row 3, Col 1: % change in NHB for All vs SAC       (baseline SAC)

    % maps:
      - allow positive/negative values
      - cap to [-pct_cap, +pct_cap]
      - colourbar tick labels show ≤-pct_cap and ≥+pct_cap at the ends
    """

    if len(gdfs) != 3:
        raise ValueError("gdfs must contain exactly three GeoDataFrames (SAC, PSAC+SAC, All).")

    if panel_labels is None:
        panel_labels = [chr(ord("A") + i) for i in range(9)]
    if len(panel_labels) < 9:
        raise ValueError("panel_labels must have length >= 9 (e.g. list('ABCDEFGHI')).")

    if label_kwargs is None:
        label_kwargs = dict(fontsize=14, fontweight="bold")

    # --- CRS alignment (copy so we don't mutate inputs)
    gdfs = tuple(g.copy() for g in gdfs)
    lake = lake_malawi.copy()
    nat = gdf_national.copy()

    for g in (*gdfs, lake, nat):
        if g.crs is None:
            g.set_crs(target_crs, inplace=True)
        else:
            g.to_crs(target_crs, inplace=True)

    gdf_sac, gdf_psac, gdf_all = gdfs

    # --- required merge key
    for name, g in [("SAC", gdf_sac), ("PSAC+SAC", gdf_psac), ("All", gdf_all)]:
        if "district" not in g.columns:
            raise ValueError(f"{name} GeoDataFrame is missing 'district' column.")
        if g["district"].duplicated().any():
            dups = g.loc[g["district"].duplicated(), "district"].unique()[:10]
            raise ValueError(f"{name} GeoDataFrame has duplicated district rows (e.g. {dups}). Expect 1 row per district.")
        if nhb_col not in g.columns:
            raise ValueError(f"{name} GeoDataFrame is missing NHB column {nhb_col!r}.")

    # --- compute % changes using SAC as baseline for BOTH comparisons
    sac = gdf_sac[["district", nhb_col]].rename(columns={nhb_col: "nhb_sac"})
    ps  = gdf_psac[["district", nhb_col]].rename(columns={nhb_col: "nhb_psac"})
    al  = gdf_all[["district", nhb_col]].rename(columns={nhb_col: "nhb_all"})

    tmp = sac.merge(ps, on="district", how="inner").merge(al, on="district", how="inner")
    for c in ["nhb_sac", "nhb_psac", "nhb_all"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # raw % changes
    tmp["pct_psac_vs_sac_raw"] = 100.0 * (tmp["nhb_psac"] - tmp["nhb_sac"]) / (tmp["nhb_sac"] + pct_eps)
    tmp["pct_all_vs_sac_raw"]  = 100.0 * (tmp["nhb_all"]  - tmp["nhb_sac"]) / (tmp["nhb_sac"] + pct_eps)

    # cap to [-pct_cap, +pct_cap] for plotting
    tmp["pct_psac_vs_sac"] = tmp["pct_psac_vs_sac_raw"].clip(-pct_cap, pct_cap)
    tmp["pct_all_vs_sac"]  = tmp["pct_all_vs_sac_raw"].clip(-pct_cap, pct_cap)

    # attach pct columns back to the geo frames
    gdf_psac = gdf_psac.merge(tmp[["district", "pct_psac_vs_sac"]], on="district", how="left")
    gdf_all  = gdf_all.merge(tmp[["district", "pct_all_vs_sac"]],  on="district", how="left")

    # --- NHB scale for absolute NHB map (SAC)
    nhb_abs_vals = pd.to_numeric(gdf_sac[nhb_col], errors="coerce")
    nhb_vmin = float(np.nanmin(nhb_abs_vals.values))
    nhb_vmax = float(np.nanmax(nhb_abs_vals.values))

    # % scale fixed by cap (shared across both % panels)
    pct_vmin, pct_vmax = -float(pct_cap), float(pct_cap)

    fig, axs = plt.subplots(3, 3, figsize=figsize)

    label_idx = 0
    for r in range(3):
        for c in range(3):
            ax = axs[r, c]

            gdf_row = gdf_sac if r == 0 else (gdf_psac if r == 1 else gdf_all)

            if c == 0:
                if r == 0:
                    col = nhb_col
                    title = "Net Health Benefit"
                    cmap = nhb_cmap
                    vmin, vmax = nhb_vmin, nhb_vmax
                    treat_minus99 = False
                    is_pct_panel = False
                elif r == 1:
                    col = "pct_psac_vs_sac"
                    title = "Δ Net Health Benefit (%)"
                    cmap = nhb_pct_cmap
                    vmin, vmax = pct_vmin, pct_vmax
                    treat_minus99 = False
                    is_pct_panel = True
                else:
                    col = "pct_all_vs_sac"
                    title = "Δ Net Health Benefit (%)"
                    cmap = nhb_pct_cmap
                    vmin, vmax = pct_vmin, pct_vmax
                    treat_minus99 = False
                    is_pct_panel = True

            elif c == 1:
                col = haem_col
                title = "Elimination Year: S. Haematobium"
                cmap = year_cmap
                vmin, vmax = year_vmin, year_vmax
                treat_minus99 = True
                is_pct_panel = False
            else:
                col = mansoni_col
                title = "Elimination Year: S. Mansoni"
                cmap = year_cmap
                vmin, vmax = year_vmin, year_vmax
                treat_minus99 = True
                is_pct_panel = False

            gdf = gdf_row.copy()

            if treat_minus99:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
                gdf[col] = gdf[col].where(gdf[col] != -99, np.nan)

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            lake.plot(
                ax=ax,
                facecolor="none",
                edgecolor="lightblue",
                linewidth=0.5,
                hatch="///",
                zorder=1,
            )

            missing_kwds = {"color": "white", "edgecolor": "lightgrey", "label": "No data"}

            gdf.plot(
                column=col,
                cmap=cmap,
                linewidth=0.8,
                edgecolor="0.8",
                ax=ax,
                legend=False,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                zorder=3,
                missing_kwds=missing_kwds,
            )

            nat.boundary.plot(ax=ax, edgecolor="grey", linewidth=1.5, zorder=4)

            # islands overlay
            if "district" in gdf.columns:
                islands = gdf[gdf["district"].str.contains("Likoma|Chizumulu", case=False, na=False)]
                if not islands.empty:
                    islands.plot(
                        column=col,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        norm=norm,
                        ax=ax,
                        edgecolor="none",
                        linewidth=0,
                        legend=False,
                        zorder=9,
                    )

            ax.set_title(title, fontsize=13)
            ax.axis("off")

            ax.text(
                -0.2, 0.98, panel_labels[label_idx],
                transform=ax.transAxes,
                ha="left", va="top",
                zorder=20,
                **label_kwargs
            )
            label_idx += 1

            # colourbar per panel
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.02)

            # customise % colourbar ticks and end labels
            if is_pct_panel:
                ticks = [-pct_cap, -50, 0, 50, pct_cap] if pct_cap >= 50 else [-pct_cap, 0, pct_cap]
                cbar.set_ticks(ticks)
                tick_labels = []
                for t in ticks:
                    if np.isclose(t, -pct_cap):
                        tick_labels.append(f"≤{-pct_cap:.0f}%")
                    elif np.isclose(t, pct_cap):
                        tick_labels.append(f"≥{pct_cap:.0f}%")
                    else:
                        tick_labels.append(f"{t:.0f}%")
                cbar.set_ticklabels(tick_labels)

        # row title on left margin
        axs[r, 0].text(
            -0.05, 0.5, row_titles[r],
            transform=axs[r, 0].transAxes,
            rotation=90,
            ha="right", va="center",
            fontsize=12
        )

    plt.subplots_adjust(wspace=0.08, hspace=0.12)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axs



fig, axs = plot_3x3_maps(
    gdfs=gdfs,
    lake_malawi=lake_malawi,
    gdf_national=gdf_national,
    panel_labels=list("ABCDEFGHI"),
    savepath=results_folder / "maps_3x3.png",
)
plt.show()




############################
# NHB, Delta NHB and elimination 3x3 plot

def plot_3x3_maps_delta_nhb(
    gdfs: tuple["gpd.GeoDataFrame", "gpd.GeoDataFrame", "gpd.GeoDataFrame"],
    lake_malawi: "gpd.GeoDataFrame",
    gdf_national: "gpd.GeoDataFrame",
    row_titles: tuple[str, str, str] = (
        "MDA SAC",
        "MDA PSAC+SAC",
        "MDA All",
    ),
    target_crs: str = "EPSG:4326",
    nhb_col: str = "mean",
    haem_col: str = "year_ephp_haem",
    mansoni_col: str = "year_ephp_mansoni",
    nhb_cmap: str = "viridis",
    dnhb_cmap: str = "RdBu_r",
    year_cmap: str = "plasma",
    year_vmin: int = 2010,
    year_vmax: int = 2050,
    figsize: tuple[int, int] = (16, 18),
    panel_labels: list[str] | None = None,
    label_kwargs: dict | None = None,
    savepath: str | None = None,
    # NEW: fixed symmetric cap for ΔNHB
    dnhb_cap: float = 30000.0,
):
    """
    Left column:
        Row 1: absolute NHB (SAC)
        Row 2: ΔNHB (PSAC+SAC − SAC)
        Row 3: ΔNHB (All − SAC)

    ΔNHB panels:
        - symmetric fixed cap ±dnhb_cap
        - colourbar labels show ≤ and ≥ at limits
    """

    if len(gdfs) != 3:
        raise ValueError("gdfs must contain exactly three GeoDataFrames.")

    if panel_labels is None:
        panel_labels = [chr(ord("A") + i) for i in range(9)]
    if label_kwargs is None:
        label_kwargs = dict(fontsize=14, fontweight="bold")

    # --- CRS alignment
    gdfs = tuple(g.copy() for g in gdfs)
    lake = lake_malawi.copy()
    nat = gdf_national.copy()

    for g in (*gdfs, lake, nat):
        if g.crs is None:
            g.set_crs(target_crs, inplace=True)
        else:
            g.to_crs(target_crs, inplace=True)

    gdf_sac, gdf_psac, gdf_all = gdfs

    # --- compute absolute ΔNHB
    sac = gdf_sac[["district", nhb_col]].rename(columns={nhb_col: "nhb_sac"})
    ps  = gdf_psac[["district", nhb_col]].rename(columns={nhb_col: "nhb_psac"})
    al  = gdf_all[["district", nhb_col]].rename(columns={nhb_col: "nhb_all"})

    tmp = sac.merge(ps, on="district").merge(al, on="district")

    for c in ["nhb_sac", "nhb_psac", "nhb_all"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp["dnhb_psac_vs_sac"] = tmp["nhb_psac"] - tmp["nhb_sac"]
    tmp["dnhb_all_vs_sac"]  = tmp["nhb_all"]  - tmp["nhb_sac"]

    gdf_psac = gdf_psac.merge(tmp[["district", "dnhb_psac_vs_sac"]], on="district", how="left")
    gdf_all  = gdf_all.merge(tmp[["district", "dnhb_all_vs_sac"]],  on="district", how="left")

    # --- NHB absolute scale (SAC only)
    nhb_vals = pd.to_numeric(gdf_sac[nhb_col], errors="coerce")
    nhb_vmin = float(np.nanmin(nhb_vals))
    nhb_vmax = float(np.nanmax(nhb_vals))

    # --- fixed symmetric ΔNHB scale
    dnhb_vmin = -float(dnhb_cap)
    dnhb_vmax = float(dnhb_cap)

    fig, axs = plt.subplots(3, 3, figsize=figsize)

    label_idx = 0
    for r in range(3):
        for c in range(3):
            ax = axs[r, c]
            gdf_row = gdf_sac if r == 0 else (gdf_psac if r == 1 else gdf_all)

            if c == 0:
                if r == 0:
                    col = nhb_col
                    title = "Net Health Benefit"
                    cmap = nhb_cmap
                    vmin, vmax = nhb_vmin, nhb_vmax
                    is_delta = False
                elif r == 1:
                    col = "dnhb_psac_vs_sac"
                    title = "Δ Net Health Benefit (vs SAC)"
                    cmap = dnhb_cmap
                    vmin, vmax = dnhb_vmin, dnhb_vmax
                    is_delta = True
                else:
                    col = "dnhb_all_vs_sac"
                    title = "Δ Net Health Benefit (vs SAC)"
                    cmap = dnhb_cmap
                    vmin, vmax = dnhb_vmin, dnhb_vmax
                    is_delta = True

            elif c == 1:
                col = haem_col
                title = "Elimination Year: S. Haematobium"
                cmap = year_cmap
                vmin, vmax = year_vmin, year_vmax
                is_delta = False
            else:
                col = mansoni_col
                title = "Elimination Year: S. Mansoni"
                cmap = year_cmap
                vmin, vmax = year_vmin, year_vmax
                is_delta = False

            gdf = gdf_row.copy()
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

            if "year_ephp" in col:
                gdf[col] = gdf[col].where(gdf[col] != -99, np.nan)

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            lake.plot(ax=ax, facecolor="none", edgecolor="lightblue",
                      linewidth=0.5, hatch="///", zorder=1)

            gdf.plot(
                column=col,
                cmap=cmap,
                linewidth=0.8,
                edgecolor="0.8",
                ax=ax,
                legend=False,
                norm=norm,
                zorder=3,
                missing_kwds={"color": "white", "edgecolor": "lightgrey"},
            )

            nat.boundary.plot(ax=ax, edgecolor="grey", linewidth=1.5, zorder=4)

            ax.set_title(title, fontsize=13)
            ax.axis("off")

            ax.text(
                -0.2, 0.98, panel_labels[label_idx],
                transform=ax.transAxes,
                ha="left", va="top",
                zorder=20,
                **label_kwargs
            )
            label_idx += 1

            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                                fraction=0.046, pad=0.02)

            # --- rounded legend ticks with ≥ / ≤ labels for delta panels
            if is_delta:
                ticks = [dnhb_vmin, 0, dnhb_vmax]
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([
                    f"≤{int(dnhb_vmin):,}",
                    "0",
                    f"≥{int(dnhb_vmax):,}",
                ])

        axs[r, 0].text(
            -0.05, 0.5, row_titles[r],
            transform=axs[r, 0].transAxes,
            rotation=90,
            ha="right", va="center",
            fontsize=12
        )

    plt.subplots_adjust(wspace=0.08, hspace=0.12)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axs


# save with different name
fig, axs = plot_3x3_maps_delta_nhb(
    gdfs=gdfs,
    lake_malawi=lake_malawi,
    gdf_national=gdf_national,
    panel_labels=list("ABCDEFGHI"),
    savepath=results_folder / "maps_3x3_deltaNHB_capped.png",
    dnhb_cap=30000,
)
plt.show()
