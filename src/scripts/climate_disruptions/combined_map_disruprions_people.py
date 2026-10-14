"""
combined_district_maps.py

Combines three district choropleth metrics into a single 3x2 panel:

    Row 1 (A, B): % of district POPULATION with >=1 weather disruption/year
                  (person-level; source: proportion_disrupted_by_district.py)
    Row 2 (C, D): % HSI DEFICIT per district — total shortfall in HSIs delivered
                  vs. the No Disruption scenario (direct + indirect/cascade
                  losses). Source: district_hsi_deficit_percentage_<suffix>.csv
    Row 3 (E, F): % HSIs DIRECTLY weather-disrupted (delayed + cancelled), the
                  volume-weighted direct disruption rate.
                  Source: district_hsi_disruption_percentage_<suffix>.csv

    Columns:      Default | Worst Case

This script does NOT re-run extraction — it reads three per-district CSVs and
the admin-2 shapefile. All three must share the same <suffix>.

IMPORTANT — the three rows are different metrics on different denominators and
are NOT directly comparable, so each row gets its own colourbar, scale, and
label. Make this explicit in the figure caption:
    Row 1 = person-level (share of people disrupted at least once)
    Row 2 = event-level deficit (% fewer HSIs delivered vs. No Disruption)
    Row 3 = event-level direct rate (% of HSIs registered as disrupted)

Inputs expected in <output_folder> (with matching <suffix>):
    proportion_disrupted_by_district_<suffix>.csv
    district_hsi_deficit_percentage_<suffix>.csv
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

# Highlights differ by metric, because the leading districts differ:
#   - direct rate  -> northern/lakeshore districts
#   - deficit      -> Chitipa + southern districts (contraception-volume driven)
HIGHLIGHT_RATE = ["Nkhata Bay", "Rumphi", "Nkhotakota"]
HIGHLIGHT_DEFICIT = ["Chitipa", "Mulanje", "Nkhata Bay"]
HIGHLIGHT_POP = ["Nkhata Bay", "Rumphi", "Nkhotakota"]


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
              panel_letters, highlight_districts):
    """Draw the two scenario maps for one row and attach a shared colourbar."""
    for ax, scen, letter in zip(axes_row, SCEN_ORDER, panel_letters):
        malawi["val"] = malawi["ADM2_EN"].map(lookup[scen])
        malawi.plot(
            column="val", ax=ax, cmap=CMAP,
            edgecolor="black", vmin=0, vmax=vmax, legend=False,
            missing_kwds={"color": "lightgrey"},
        )
        if highlight_districts:
            hl = malawi[malawi["ADM2_EN"].isin(highlight_districts)]
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


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name} — {hint}")
    return path


def apply(output_folder: Path, resourcefilepath: Path, suffix: str = "main_text_mode_1"):
    malawi = _load_shapefile(resourcefilepath)

    # ── Row 1: population proportion with >=1 disruption/year ─────────────────
    pop_csv = _require(
        output_folder / f"proportion_disrupted_by_district_{suffix}.csv",
        "run proportion_disrupted_by_district.py first.",
    )
    pop_df = pd.read_csv(pop_csv)
    pop_lookup = {
        scen: pop_df.loc[pop_df["Scenario"] == scen].set_index("district")["pct_disrupted_mean"]
        for scen in SCEN_ORDER
    }
    pop_vmax = max(np.ceil(max(pop_lookup[s].max() for s in SCEN_ORDER) * 10) / 10, 0.1)

    # ── Row 2: % HSI DEFICIT (total shortfall vs No Disruption) ───────────────
    deficit_csv = _require(
        output_folder / f"district_hsi_deficit_percentage_{suffix}.csv",
        "re-run the comparison script with the district-deficit export added.",
    )
    deficit_df = pd.read_csv(deficit_csv, index_col=0)  # index=district, cols=scenarios
    deficit_lookup = {scen: deficit_df[scen] for scen in SCEN_ORDER}
    deficit_vmax = max(np.ceil(max(deficit_lookup[s].max() for s in SCEN_ORDER)), 1.0)

    # ── Row 3: % HSIs DIRECTLY weather-disrupted (rate) ───────────────────────
    rate_csv = _require(
        output_folder / f"district_hsi_disruption_percentage_{suffix}.csv",
        "run the main comparison script (writes the district disruption-rate CSV).",
    )
    rate_df = pd.read_csv(rate_csv, index_col=0)  # index=district, cols=scenarios
    rate_lookup = {scen: rate_df[scen] for scen in SCEN_ORDER}
    rate_vmax = max(np.ceil(max(rate_lookup[s].max() for s in SCEN_ORDER)), 1.0)

    # ── Assemble 3x2 ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 20))

    _plot_row(
        fig, axes[0], malawi, pop_lookup, pop_vmax,
        cbar_label="% of district population\nwith \u22651 disruption/year",
        panel_letters=("A", "B"), highlight_districts=HIGHLIGHT_POP,
    )
    _plot_row(
        fig, axes[1], malawi, deficit_lookup, deficit_vmax,
        cbar_label="% HSI deficit\n(vs. No Disruption)",
        panel_letters=("C", "D"), highlight_districts=HIGHLIGHT_DEFICIT,
    )
    _plot_row(
        fig, axes[2], malawi, rate_lookup, rate_vmax,
        cbar_label="% HSIs directly\nweather-disrupted",
        panel_letters=("E", "F"), highlight_districts=HIGHLIGHT_RATE,
    )

    out_path = output_folder / f"map_combined_population_and_hsi_{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined 3x2 map saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder", type=Path,
                        help="Folder containing the three source CSVs and where the figure is written.")
    parser.add_argument("--resourcefilepath", type=Path, default=Path("./resources"))
    parser.add_argument("--suffix", type=str, default="main_text_mode_1")
    args = parser.parse_args()
    apply(args.output_folder, args.resourcefilepath, args.suffix)
