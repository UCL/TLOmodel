"""
combined_district_maps.py

Combines the two district choropleth figures into a single 2x2 panel:

    Top row    (A, B): % of district POPULATION with >=1 weather disruption/year
                       (source: proportion_disrupted_by_district.py)
    Bottom row (C, D): % of HSIs disrupted per district
                       (source: the main comparison script, "Figure 6")

    Columns:           Default | Worst Case

Both source scripts already write the per-district values to CSV, so this
script does NOT re-run any extraction — it just reads those two CSVs and the
admin-2 shapefile.

IMPORTANT — the two rows are different metrics on different denominators:
    Top row    = person-level (share of people disrupted at least once)
    Bottom row = event-level  (share of HSIs disrupted, volume-weighted)
They are NOT directly comparable, so each row gets its own colourbar and label.
Make this explicit in the figure caption.

Inputs expected in <output_folder> (with matching <suffix>):
    proportion_disrupted_by_district_<suffix>.csv
    district_hsi_disruption_percentage_<suffix>.csv
and the shapefile in <resourcefilepath>/mapping/.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

FS_TICK = 13
FS_LABEL = 15
FS_TITLE = 16

CMAP = "Oranges"
SCEN_ORDER = ["Default", "Worst Case"]
HIGHLIGHT_DISTRICTS = ["Nkhata Bay", "Rumphi", "Nkhotakota"]
HSI_VMAX = 4.0  # matches Figure 6 in the comparison script


def _load_shapefile(resourcefilepath: Path) -> gpd.GeoDataFrame:
    malawi = gpd.read_file(
        resourcefilepath / "mapping" / "ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
    )
    for old, new in [
        ("Blantyre City", "Blantyre"), ("Mzuzu City", "Mzuzu"),
        ("Lilongwe City", "Lilongwe"), ("Zomba City", "Zomba"),
    ]:
        malawi["ADM2_EN"] = malawi["ADM2_EN"].replace(old, new)
    return malawi


def _plot_row(fig, axes_row, malawi, lookup, vmax, cbar_label,
              panel_letters, highlight):
    """Draw the two scenario maps for one row and attach a shared colourbar."""
    for ax, scen, letter in zip(axes_row, SCEN_ORDER, panel_letters):
        malawi["val"] = malawi["ADM2_EN"].map(lookup[scen])
        malawi.plot(
            column="val", ax=ax, cmap=CMAP,
            edgecolor="black", vmin=0, vmax=vmax, legend=False,
            missing_kwds={"color": "lightgrey"},
        )
        if highlight:
            hl = malawi[malawi["ADM2_EN"].isin(HIGHLIGHT_DISTRICTS)]
            hl.plot(ax=ax, facecolor="none", edgecolor="#22577A", linewidth=2.0)
            for _, row in hl.iterrows():
                centroid = row.geometry.centroid
                ax.annotate(
                    row["ADM2_EN"], xy=(centroid.x, centroid.y),
                    fontsize=FS_TICK - 1, color="black", fontweight="bold",
                    ha="center", va="center",
                )
        ax.set_title(f"({letter}) {scen}", fontsize=FS_TITLE, fontweight="bold")
        ax.axis("off")

    # one shared colourbar per row, spanning both panels
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=mpl.colors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes_row), shrink=0.7, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=FS_LABEL, fontweight="bold")
    cbar.ax.tick_params(labelsize=FS_TICK)


def apply(output_folder: Path, resourcefilepath: Path, suffix: str = "main_text_mode_1"):
    malawi = _load_shapefile(resourcefilepath)

    # ── Top row: population proportion with >=1 disruption/year ───────────────
    pop_csv = output_folder / f"proportion_disrupted_by_district_{suffix}.csv"
    if not pop_csv.exists():
        raise FileNotFoundError(
            f"Missing {pop_csv.name} — run proportion_disrupted_by_district.py first."
        )
    pop_df = pd.read_csv(pop_csv)
    pop_lookup = {
        scen: pop_df.loc[pop_df["Scenario"] == scen].set_index("district")["pct_disrupted_mean"]
        for scen in SCEN_ORDER
    }
    pop_vmax = max(
        np.ceil(max(pop_lookup[s].max() for s in SCEN_ORDER) * 10) / 10, 0.1
    )

    # ── Bottom row: % of HSIs disrupted ───────────────────────────────────────
    hsi_csv = output_folder / f"district_hsi_disruption_percentage_{suffix}.csv"
    if not hsi_csv.exists():
        raise FileNotFoundError(
            f"Missing {hsi_csv.name} — run the comparison script (Figure 6) first."
        )
    hsi_df = pd.read_csv(hsi_csv, index_col=0)  # index = district, cols = scenarios
    hsi_lookup = {scen: hsi_df[scen] for scen in SCEN_ORDER}

    # ── Assemble 2x2 ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 14))

    _plot_row(
        fig, axes[0], malawi, pop_lookup, pop_vmax,
        cbar_label="% of district population\nwith \u22651 disruption/year",
        panel_letters=("A", "B"), highlight=True,
    )
    _plot_row(
        fig, axes[1], malawi, hsi_lookup, HSI_VMAX,
        cbar_label="% HSIs disrupted",
        panel_letters=("C", "D"), highlight=True,
    )

    out_path = output_folder / f"map_combined_population_and_hsi_{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined map saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder", type=Path,
                        help="Folder containing the two source CSVs and where the figure is written.")
    parser.add_argument("--resourcefilepath", type=Path, default=Path("./resources"))
    parser.add_argument("--suffix", type=str, default="main_text_mode_1")
    args = parser.parse_args()
    apply(args.output_folder, args.resourcefilepath, args.suffix)
