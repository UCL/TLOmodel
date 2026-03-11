"""
plot_population_and_facilities.py

Standalone script to produce the TLO Model – Malawi population placement
and health facilities figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

resourcefilepath = Path(__file__).parent.parent.parent.parent / "resources"
path_to_climate = resourcefilepath / "climate_change_impacts"

# ── Load data ─────────────────────────────────────────────────────────────────
worldpop_gdf = gpd.read_file(path_to_climate / "worldpop_density_with_districts.shp")
worldpop_gdf["Z_prop"] = pd.to_numeric(worldpop_gdf["Z_prop"], errors="coerce").fillna(0)

facilities_info = pd.read_csv(path_to_climate / "facilities_with_lat_long_region.csv")

# ── Sample 10,000 worldpop-weighted population points ─────────────────────────
sampled = worldpop_gdf.sample(n=100_000, weights="Z_prop", replace=True, random_state=42)
pop_x, pop_y = sampled.geometry.x.values, sampled.geometry.y.values

# ── Facility level config ─────────────────────────────────────────────────────
LEVEL_CONFIG = {
    "Level 0 – Health post": (["Health Post"], "#a8d8ea", 18),
    "Level 1a – Dispensary / Clinic": (["Dispensary", "Clinic"], "#3aafa9", 25),
    "Level 1b – Health centre / Rural hospital": (["Health Centre", "Rural/Community Hospital"], "#f4a261", 35),
    "Level 2 – District hospital": (["District Hospital"], "#c0392b", 80),
    "Level 3 – Central hospital": (["Central Hospital"], "#922b21", 160),
}

# ── Plot ──────────────────────────────────────────────────────────────────────
BG = "white"
fig, (ax_pop, ax_fac) = plt.subplots(1, 2, figsize=(16, 11), facecolor=BG)
district_boundaries = worldpop_gdf.dissolve("ADM2_EN")

for ax in (ax_pop, ax_fac):
    district_boundaries.boundary.plot(ax=ax, linewidth=0.4, edgecolor="#aaaaaa", zorder=1, aspect=None)
    ax.set_aspect("equal")
    ax.axis("off")
for ax in (ax_pop, ax_fac):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

# Panel A
ax_pop.set_title("(A) Population locations (n = 100,000)", fontsize=10, loc="left")
ax_pop.scatter(pop_x, pop_y, s=1.5, color="#7bafd4", alpha=0.4, linewidths=0)
for name, pt in worldpop_gdf.dissolve("ADM2_EN").centroid.items():
    ax_pop.text(pt.x, pt.y, name, fontsize=5.5, ha="center", color="#555", alpha=0.8)

# Panel B
ax_fac.set_title("(B) Health facilities in model\nby level of care", fontsize=10, loc="center")
legend_handles = []
for label, (ftypes, color, size) in LEVEL_CONFIG.items():
    sub = facilities_info[facilities_info["Ftype"].isin(ftypes)].dropna(
        subset=["A109__Longitude", "A109__Latitude"])
    ax_fac.scatter(sub["A109__Longitude"], sub["A109__Latitude"],
                   s=size, c=color, alpha=0.85, linewidths=0.3, edgecolors="white", zorder=3)
    legend_handles.append(mpatches.Patch(facecolor=color, label=label))
for name, pt in worldpop_gdf.dissolve("ADM2_EN").centroid.items():
    ax_fac.text(pt.x, pt.y, name, fontsize=5.5, ha="center", color="#555", alpha=0.8)

legend = ax_fac.legend(handles=legend_handles, title="Facility level", title_fontsize=9,
                       fontsize=8, loc="lower left", framealpha=0.9, edgecolor="#ccc")
legend.get_frame().set_facecolor(BG)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    "/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/baseline_run_with_pop-2026-03-05T101702Z/tlo_population_facilities.png",
    dpi=200, bbox_inches="tight")
plt.show()
