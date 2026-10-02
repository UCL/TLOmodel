"""
Figures: heat-attributable missed services, TLO volume × WBGT district rates, 2025–2040.

Reads combined_district_scaled_<ind>_<ssp>_<tier>.csv from reading_tlo_results.py
(WBGT district loss rate × full TLO district volume).

Outputs:
    fig_tlo_maps_<ssp>.png       main text: one map per indicator, MAIN_SSP / MAIN_TIER
    fig_tlo_national_totals.png  main text: national totals by indicator × SSP,
                                 dot = median tier, bars = lowest–highest tier
    fig_tlo_maps_all_ssp.png     supplement: indicators (rows) × SSPs (cols), MAIN_TIER
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter

hex_colors = ["#FAF0F1", "#E5B3BA", "#DB7E8B", "#D48E95", "#D66473", "#721D28"]
custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", hex_colors)
if "custom_cmap" not in colormaps:
    colormaps.register(custom_cmap, name="custom_cmap")

# ---- Config ----
COMBINED_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/combined_wbgt_tlo")
OUT_DIR = COMBINED_DIR
OUT_DIR.mkdir(exist_ok=True)

SHAPEFILE = Path("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/"
                 "resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
DIST_COL_SHP = "ADM2_EN"

# City polygons in the shapefile take their parent district's value
# (the facility registry assigns city facilities to the parent district)
SHP_DISTRICT_MAP = {
    "Blantyre City": "Blantyre",
    "Lilongwe City": "Lilongwe",
    "Zomba City":    "Zomba",
    "Mzuzu City":    "Mzimba",
}

YEAR_MIN, YEAR_MAX = 2025, 2040

INDICATORS = [
    "opd_attendance",
    "anc_total_visits",
    "pnc_within_2wks",
    "fp_total_clients",
    "cervical_screening_total",
]
INDICATOR_LABELS = {
    "opd_attendance":           "OPD",
    "anc_total_visits":         "ANC visits",
    "pnc_within_2wks":          "PNC ≤2 wks",
    "fp_total_clients":         "FP clients",
    "cervical_screening_total": "Cervical screening",
}

SSPS = ["ssp126", "ssp245", "ssp585"]
SSP_LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
SSP_COLOUR = {"ssp126": "#9BB29E", "ssp245": "#F1DCBA", "ssp585": "#D45C5D"}
SSP_EDGE = {"ssp126": "#5E7A62", "ssp245": "#B08F55", "ssp585": "#8E2F30"}  # darker outline so pale amber reads on white
TIERS = ["lowest", "median", "highest"]
MAIN_TIER = "median"
MAIN_SSP = "ssp245"

LOST_COL = "HSIs_lost_only"     # or "HSIs_lost_net"
MAP_METRIC = "pct"              # "pct": % of expected services lost; "count": HSIs lost


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load_district(indicator, ssp, tier):
    """District totals over YEAR_MIN–YEAR_MAX; None if the file is missing."""
    path = COMBINED_DIR / f"combined_district_scaled_{indicator}_{ssp}_{tier}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)]
    d = (df.groupby("District")
           .agg(HSIs_expected=("HSIs_expected", "sum"),
                HSIs_lost=(LOST_COL, "sum"))
           .reset_index())
    d["pct_lost"] = d["HSIs_lost"] / d["HSIs_expected"] * 100.0
    d["value"] = d["pct_lost"] if MAP_METRIC == "pct" else d["HSIs_lost"]
    return d


def load_shapes():
    gdf = gpd.read_file(SHAPEFILE)
    gdf["District"] = gdf[DIST_COL_SHP].replace(SHP_DISTRICT_MAP)
    return gdf


_checked_names = set()


def check_names(gdf, d, indicator):
    """Print district names that don't merge, once per indicator."""
    if indicator in _checked_names:
        return
    _checked_names.add(indicator)
    shp, dat = set(gdf["District"]), set(d["District"])
    if dat - shp:
        print(f"  [warn] {indicator}: data districts not in shapefile: {sorted(dat - shp)}")
    if shp - dat:
        print(f"  [info] {indicator}: shapefile districts with no data (grey): {sorted(shp - dat)}")


def metric_label():
    return "% of services lost" if MAP_METRIC == "pct" else "Services lost"


def pct_formatter(vmax):
    dp = 2 if vmax < 1 else 1
    return FuncFormatter(lambda x, _: f"{x:.{dp}f}%")


def fmt_count(x, _=None):
    if x >= 1e6:
        return f"{x/1e6:.3g}M"
    if x >= 1e3:
        return f"{x/1e3:.3g}k"
    return f"{x:.3g}"


# ------------------------------------------------------------------
# Maps
# ------------------------------------------------------------------
def _draw_map(ax, gdf, d, vmin, vmax):
    ax.set_axis_off()
    if d is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="grey")
        return
    g = gdf.merge(d[["District", "value"]], on="District", how="left")
    g.plot(column="value", ax=ax, vmin=vmin, vmax=vmax, cmap="custom_cmap",
           missing_kwds={"color": "lightgrey", "edgecolor": "white", "linewidth": 0.3},
           edgecolor="white", linewidth=0.3)


def _colourbar(fig, ax, vmin, vmax):
    sm = ScalarMappable(cmap="custom_cmap", norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=7)
    if MAP_METRIC == "pct":
        cb.ax.xaxis.set_major_formatter(pct_formatter(vmax))
    else:
        cb.ax.xaxis.set_major_formatter(FuncFormatter(fmt_count))
    cb.ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    return cb


def make_main_maps(gdf):
    """One map per indicator for MAIN_SSP / MAIN_TIER, each with its own scale."""
    n = len(INDICATORS)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 5.2), squeeze=False)
    for j, ind in enumerate(INDICATORS):
        ax = axes[0, j]
        d = load_district(ind, MAIN_SSP, MAIN_TIER)
        if d is not None:
            check_names(gdf, d, ind)
            vmin, vmax = 0.0, max(d["value"].max(), 1e-9)
            _draw_map(ax, gdf, d, vmin, vmax)
            _colourbar(fig, ax, vmin, vmax)
        else:
            _draw_map(ax, gdf, None, 0, 1)
        ax.set_title(INDICATOR_LABELS.get(ind, ind), fontsize=10, fontweight="bold")

    fig.suptitle(f"Heat-attributable missed services by district, {YEAR_MIN}–{YEAR_MAX} "
                 f"({SSP_LABELS[MAIN_SSP]}, {MAIN_TIER} WBGT model)",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, 0.01,
             f"Colour: {metric_label().lower()}. Each panel has its own scale. "
             "Grey: no WBGT-modelled facility in district.",
             ha="center", fontsize=7.5, style="italic", color="dimgrey")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = OUT_DIR / f"fig_tlo_maps_{MAIN_SSP}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def make_supp_maps(gdf):
    """Indicators (rows) × SSPs (cols), MAIN_TIER; shared scale within each row."""
    n_rows, n_cols = len(INDICATORS), len(SSPS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols + 1.2, 3.4 * n_rows),
                             squeeze=False)
    for i, ind in enumerate(INDICATORS):
        cells = {ssp: load_district(ind, ssp, MAIN_TIER) for ssp in SSPS}
        vals = [c["value"].max() for c in cells.values() if c is not None]
        vmin, vmax = 0.0, (max(vals) if vals else 1.0) or 1.0
        for j, ssp in enumerate(SSPS):
            _draw_map(axes[i, j], gdf, cells[ssp], vmin, vmax)
            if i == 0:
                axes[i, j].set_title(SSP_LABELS[ssp], fontsize=11, fontweight="bold")
        axes[i, 0].text(-0.08, 0.5, INDICATOR_LABELS.get(ind, ind),
                        transform=axes[i, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=10, fontweight="bold")
        sm = ScalarMappable(cmap="custom_cmap", norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes[i, :], fraction=0.025, pad=0.02, shrink=0.8)
        cb.ax.tick_params(labelsize=7)
        if MAP_METRIC == "pct":
            cb.ax.yaxis.set_major_formatter(pct_formatter(vmax))
        else:
            cb.ax.yaxis.set_major_formatter(FuncFormatter(fmt_count))

    fig.suptitle(f"Heat-attributable missed services by district and scenario, "
                 f"{YEAR_MIN}–{YEAR_MAX} ({MAIN_TIER} WBGT model)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.5, 0.003,
             f"Colour: {metric_label().lower()}; scale shared within each row. "
             "Grey: no WBGT-modelled facility in district.",
             ha="center", fontsize=8, style="italic", color="dimgrey")
    out = OUT_DIR / "fig_tlo_maps_all_ssp.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------
# National totals
# ------------------------------------------------------------------
def national_table():
    rows = []
    for ind in INDICATORS:
        for ssp in SSPS:
            per_tier = {}
            for tier in TIERS:
                d = load_district(ind, ssp, tier)
                if d is not None:
                    per_tier[tier] = (d["HSIs_lost"].sum(), d["HSIs_expected"].sum())
            if MAIN_TIER not in per_tier:
                continue
            lost = {t: v[0] for t, v in per_tier.items()}
            exp = per_tier[MAIN_TIER][1]
            rows.append({
                "indicator": ind, "ssp": ssp,
                "lost_mid": lost[MAIN_TIER],
                "lost_lo": min(lost.values()), "lost_hi": max(lost.values()),
                "expected": exp,
                "pct_mid": lost[MAIN_TIER] / exp * 100.0,
            })
    df = pd.DataFrame(rows)
    out = OUT_DIR / "tlo_national_totals.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    return df


def make_national_totals(df):
    if df.empty:
        print("no data for national totals")
        return
    inds = [i for i in INDICATORS if i in set(df["indicator"])]
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(inds) + 1.4))
    offset = {"ssp126": -0.22, "ssp245": 0.0, "ssp585": 0.22}

    for ssp in SSPS:
        sub = df[df["ssp"] == ssp].set_index("indicator").reindex(inds).dropna(subset=["lost_mid"])
        y = np.array([inds.index(i) for i in sub.index]) + offset[ssp]
        ax.errorbar(sub["lost_mid"], y,
                    xerr=[sub["lost_mid"] - sub["lost_lo"], sub["lost_hi"] - sub["lost_mid"]],
                    fmt="o", markersize=7, capsize=3, linewidth=1.4,
                    color=SSP_EDGE[ssp], markerfacecolor=SSP_COLOUR[ssp],
                    markeredgecolor=SSP_EDGE[ssp], markeredgewidth=1.0,
                    label=SSP_LABELS[ssp])

    # Label the main SSP's median value on each row
    main = df[df["ssp"] == MAIN_SSP].set_index("indicator")
    for i, ind in enumerate(inds):
        if ind in main.index:
            r = main.loc[ind]
            ax.text(r["lost_hi"] * 1.15, i, f"{fmt_count(r['lost_mid'])} ({r['pct_mid']:.1f}%)",
                    va="center", fontsize=7.5, color="dimgrey")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_count))
    ax.set_yticks(range(len(inds)))
    ax.set_yticklabels([INDICATOR_LABELS.get(i, i) for i in inds])
    ax.invert_yaxis()
    ax.set_xlabel(f"Services lost nationally, {YEAR_MIN}–{YEAR_MAX} (log scale)", fontsize=10)
    lo, hi = df["lost_lo"].min(), df["lost_hi"].max()
    ax.set_xlim(lo / 2, hi * 6)   # room for the value labels
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", which="major", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005,
             f"Dot: {MAIN_TIER} WBGT model; bars: lowest–highest. "
             f"Labels: {SSP_LABELS[MAIN_SSP]} {MAIN_TIER} (% of expected services).",
             fontsize=7.5, style="italic", color="dimgrey")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = OUT_DIR / "fig_tlo_national_totals.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    gdf = load_shapes()
    make_main_maps(gdf)
    make_national_totals(national_table())
    make_supp_maps(gdf)
