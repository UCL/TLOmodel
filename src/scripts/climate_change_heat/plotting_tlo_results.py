"""
Publication figure: HSIs lost to heat 2025–2040.

Panel A: small-multiples choropleth — rows = indicators, cols = SSPs.
         Colour = total HSIs lost per district (median tier).
Panel B: national totals by indicator × SSP, tier range as error bars.

Reads combined_district_*.csv from the pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---- Config ----
COMBINED_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/combined_wbgt_tlo")
OUT_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/figures")
OUT_DIR.mkdir(exist_ok=True)

SHAPEFILE = Path("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/"
                 "resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
DIST_COL_SHP = "ADM2_EN"

PREFIX = "1"

# Indicators to include (order top→bottom in map grid)
INDICATORS = [
    "vmmc_first_visits",
    "anc_total_visits",
    "bcg_under1",
    "measles1_under1",
    "penta3_under1",
    "pnc_within_2wks",
    "fp_total_clients",
    "opd_attendance",
]
INDICATOR_LABELS = {
    "vmmc_first_visits":      "VMMC",
    "anc_total_visits":       "ANC visits",
    "bcg_under1":             "BCG (<1)",
    "measles1_under1":        "Measles 1 (<1)",
    "penta3_under1":          "Penta3 (<1)",
    "pnc_within_2wks":        "PNC ≤2 wks",
    "fp_total_clients":       "FP clients",
    "opd_attendance":         "OPD",
}

SSPS = ["ssp126", "ssp245", "ssp585"]
SSP_LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
TIERS = ["lowest", "median", "highest"]
MAIN_TIER = "median"


def load_combined(indicator, ssp, tier):
    """Return district totals across 2025–2040 for one (ind, ssp, tier)."""
    path = COMBINED_DIR / f"{PREFIX}_combined_district_{indicator}_{ssp}_{tier}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return (df.groupby("District")
              .agg(HSIs_expected=("HSIs_expected", "sum"),
                   HSIs_lost=("HSIs_lost", "sum"))
              .reset_index())


# ------------------------------------------------------------------
# Panel A: choropleth grid
# ------------------------------------------------------------------
def make_map_grid():
    gdf = gpd.read_file(SHAPEFILE)

    n_rows, n_cols = len(INDICATORS), len(SSPS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    # Compute a per-indicator vmax so each row has its own scale
    # (indicators span very different HSI volumes — a shared scale hides pattern)
    row_maxes = {}
    for ind in INDICATORS:
        vals = []
        for ssp in SSPS:
            d = load_combined(ind, ssp, MAIN_TIER)
            if d is not None:
                vals.append(d["HSIs_lost"].abs().max())
        row_maxes[ind] = max(vals) if vals else 1.0

    for i, ind in enumerate(INDICATORS):
        vmax = row_maxes[ind] or 1.0
        # Diverging if any negatives (services gained under heat), else sequential
        has_neg = False
        cell_data = {}
        for ssp in SSPS:
            d = load_combined(ind, ssp, MAIN_TIER)
            cell_data[ssp] = d
            if d is not None and (d["HSIs_lost"] < 0).any():
                has_neg = True

        cmap = "RdBu_r" if has_neg else "YlOrRd"
        vmin = -vmax if has_neg else 0

        for j, ssp in enumerate(SSPS):
            ax = axes[i, j]
            d = cell_data[ssp]
            if d is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
            else:
                g = gdf.merge(d, left_on=DIST_COL_SHP, right_on="District", how="left")
                g.plot(column="HSIs_lost", ax=ax, vmin=vmin, vmax=vmax,
                       cmap=cmap, missing_kwds={"color": "lightgrey"},
                       edgecolor="white", linewidth=0.3)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(SSP_LABELS[ssp], fontsize=11, fontweight="bold")

        # Row label + colourbar on the right of each row
        axes[i, 0].text(-0.08, 0.5, INDICATOR_LABELS.get(ind, ind),
                        transform=axes[i, 0].transAxes,
                        ha="right", va="center", fontsize=10, fontweight="bold")

        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[i, :], fraction=0.02, pad=0.02, shrink=0.9)
        cbar.ax.tick_params(labelsize=7)
        # Compact tick format
        cbar.formatter.set_scientific(False)
        cbar.formatter.set_useOffset(False)

    fig.suptitle(f"HSIs lost to heat by district, 2025–2040 (tier: {MAIN_TIER})",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "Grey = no facility-year match. Row-specific colour scales (HSI volumes differ by orders of magnitude).",
             ha="center", fontsize=8, style="italic", color="grey")
    fig.tight_layout(rect=[0, 0.01, 1, 0.99])

    out = OUT_DIR / "fig_A_map_grid.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------
# Panel B: national totals with tier range as error bars
# ------------------------------------------------------------------
def make_dot_plot():
    rows = []
    for ind in INDICATORS:
        for ssp in SSPS:
            per_tier = {}
            for tier in TIERS:
                d = load_combined(ind, ssp, tier)
                if d is None:
                    continue
                per_tier[tier] = d["HSIs_lost"].sum()
            if MAIN_TIER not in per_tier:
                continue
            lo = min(per_tier.values())
            hi = max(per_tier.values())
            rows.append({
                "indicator": ind,
                "ssp": ssp,
                "mid": per_tier[MAIN_TIER],
                "lo": lo, "hi": hi,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print("no data for dot plot"); return

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(INDICATORS) + 1.5))

    y = np.arange(len(INDICATORS))
    ssp_offset = {"ssp126": -0.22, "ssp245": 0.0, "ssp585": 0.22}
    ssp_colour = {"ssp126": "#2b7bba", "ssp245": "#f4a261", "ssp585": "#c1121f"}

    for ssp in SSPS:
        sub = df[df["ssp"] == ssp]
        yy = np.array([INDICATORS.index(i) for i in sub["indicator"]]) + ssp_offset[ssp]
        ax.errorbar(sub["mid"], yy,
                    xerr=[sub["mid"] - sub["lo"], sub["hi"] - sub["mid"]],
                    fmt="o", markersize=6, capsize=3, linewidth=1.2,
                    color=ssp_colour[ssp], label=SSP_LABELS[ssp])

    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([INDICATOR_LABELS.get(i, i) for i in INDICATORS])
    ax.invert_yaxis()
    ax.set_xlabel("National HSIs lost to heat, 2025–2040\n(dot = median tier, bars = lowest–highest range)",
                  fontsize=10)
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = OUT_DIR / "fig_B_national_dots.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_map_grid()
    make_dot_plot()
