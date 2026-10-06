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

import glob
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter

custom_cmap = LinearSegmentedColormap.from_list(
    "custom_diverging",
    ["#4D7799", "#7FA4C4", "#FFFFFF", "#D48E95", "#B5515B"],
)
if "custom_cmap" in colormaps:
    colormaps.unregister("custom_cmap")
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

# Must match HOT_ONLY in reading_tlo_results.py: True reads the "_hot" outputs
# (services lost in months above the historical WBGT p95).
HOT_ONLY = True
SCALED_SUFFIX = "_hot" if HOT_ONLY else ""
PERIOD_LABEL = "hot months (WBGT > historical p95)" if HOT_ONLY else "all months"

# WBGT projection outputs (for the hot-month vs all-month comparison)
HEAT_OUT_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs")
WBGT_VAR = "wbgt5x_day"
LAG_SUFFIX = "_with_lags"

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

LOST_COL = "HSIs_lost_net"      # net: gains (excess services) kept as negatives; or "HSIs_lost_only"
MAP_METRIC = "pct"              # "pct": % of expected services lost; "count": HSIs lost


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load_district(indicator, ssp, tier):
    """District totals over YEAR_MIN–YEAR_MAX; None if the file is missing."""
    path = COMBINED_DIR / f"combined_district_scaled{SCALED_SUFFIX}_{indicator}_{ssp}_{tier}.csv"
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
    base = "% of services lost" if MAP_METRIC == "pct" else "Services lost"
    if LOST_COL == "HSIs_lost_net":
        base += " (red = deficit, blue = excess)"
    return base


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
            vmax = max(d["value"].abs().max(), 1e-9)   # symmetric: white = 0
            vmin = -vmax
            _draw_map(ax, gdf, d, vmin, vmax)
            _colourbar(fig, ax, vmin, vmax)
        else:
            _draw_map(ax, gdf, None, 0, 1)
        ax.set_title(INDICATOR_LABELS.get(ind, ind), fontsize=10, fontweight="bold")

    fig.suptitle(f"Heat-attributable missed services by district, {YEAR_MIN}–{YEAR_MAX}, {PERIOD_LABEL}\n"
                 f"({SSP_LABELS[MAIN_SSP]}, {MAIN_TIER} WBGT model)",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, 0.01,
             f"Colour: {metric_label().lower()}. Each panel has its own scale. "
             "Grey: no WBGT-modelled facility in district.",
             ha="center", fontsize=7.5, style="italic", color="dimgrey")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = OUT_DIR / f"fig_tlo_maps{SCALED_SUFFIX}_{MAIN_SSP}.png"
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
        vals = [c["value"].abs().max() for c in cells.values() if c is not None]
        vmax = (max(vals) if vals else 1.0) or 1.0   # symmetric: white = 0
        vmin = -vmax
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
                 f"{YEAR_MIN}–{YEAR_MAX}, {PERIOD_LABEL} ({MAIN_TIER} WBGT model)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.5, 0.003,
             f"Colour: {metric_label().lower()}; scale shared within each row. "
             "Grey: no WBGT-modelled facility in district.",
             ha="center", fontsize=8, style="italic", color="dimgrey")
    out = OUT_DIR / f"fig_tlo_maps_all_ssp{SCALED_SUFFIX}.png"
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
                "pct_lo": min(lost.values()) / exp * 100.0,
                "pct_hi": max(lost.values()) / exp * 100.0,
            })
    df = pd.DataFrame(rows)
    out = OUT_DIR / f"tlo_national_totals{SCALED_SUFFIX}.csv"
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
    ax.set_xlabel(f"Services lost nationally, {YEAR_MIN}–{YEAR_MAX}, {PERIOD_LABEL} (log scale)", fontsize=10)
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
    out = OUT_DIR / f"fig_tlo_national_totals{SCALED_SUFFIX}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------
# Numbers for the text
# ------------------------------------------------------------------
def sig(x, n=2):
    """Round to n significant figures for reporting (e.g. 93,256 -> 93,000)."""
    if x == 0 or not np.isfinite(x):
        return x
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def write_text_summary(nat, n_top=3):
    """Text-ready counts: national totals per indicator × SSP, plus top districts
    for MAIN_SSP / MAIN_TIER. Writes a LaTeX-ready .txt and prints a plain version."""
    n_years = YEAR_MAX - YEAR_MIN + 1
    lines_tex, lines_plain = [], []

    hdr = (f"National services lost, {YEAR_MIN}-{YEAR_MAX}, {PERIOD_LABEL} ({LOST_COL}); "
           f"{MAIN_TIER} tier (lowest-highest tier range)")
    lines_plain += [hdr, "-" * len(hdr)]
    lines_tex += [f"% {hdr}"]
    for ind in INDICATORS:
        sub = nat[nat["indicator"] == ind].set_index("ssp")
        if sub.empty:
            continue
        label = INDICATOR_LABELS.get(ind, ind)
        lines_plain.append(f"{label}  (expected over period: {sig(sub['expected'].iloc[0], 3):,.0f})")
        for ssp in SSPS:
            if ssp not in sub.index:
                continue
            r = sub.loc[ssp]
            mid, lo, hi = sig(r["lost_mid"]), sig(r["lost_lo"]), sig(r["lost_hi"])
            per_yr = sig(r["lost_mid"] / n_years)
            lines_plain.append(
                f"  {SSP_LABELS[ssp]}: {mid:,.0f} ({lo:,.0f}-{hi:,.0f}); "
                f"{r['pct_mid']:.1f}% ({r['pct_lo']:.1f}-{r['pct_hi']:.1f}%); ~{per_yr:,.0f}/yr")
            lines_tex.append(
                f"{label}, {SSP_LABELS[ssp]}: {mid:,.0f} ({lo:,.0f}--{hi:,.0f}) services "
                f"({r['pct_mid']:.1f}\\%, {r['pct_lo']:.1f}--{r['pct_hi']:.1f}\\%), "
                f"approximately {per_yr:,.0f} per year")
        lines_plain.append("")

    hdr2 = f"Top {n_top} districts, {SSP_LABELS[MAIN_SSP]} {MAIN_TIER} ({LOST_COL})"
    lines_plain += [hdr2, "-" * len(hdr2)]
    lines_tex += ["", f"% {hdr2}"]
    for ind in INDICATORS:
        d = load_district(ind, MAIN_SSP, MAIN_TIER)
        if d is None:
            continue
        label = INDICATOR_LABELS.get(ind, ind)
        by_n = d.nlargest(n_top, "HSIs_lost")
        by_p = d.nlargest(n_top, "pct_lost")
        n_str = ", ".join(f"{r.District} ({sig(r.HSIs_lost):,.0f})" for r in by_n.itertuples())
        p_str = ", ".join(f"{r.District} ({r.pct_lost:.1f}%)" for r in by_p.itertuples())
        lines_plain += [f"{label}", f"  most services lost: {n_str}", f"  highest %: {p_str}"]
        lines_tex.append(f"{label}: most services lost in {n_str}; highest percentage in "
                         f"{p_str.replace('%', chr(92) + '%')}")
        n_excess = (d["HSIs_lost"] < 0).sum()
        if n_excess:
            lines_plain.append(f"  districts with net excess: {n_excess}")
    print("\n".join(lines_plain))

    out = OUT_DIR / f"tlo_text_summary{SCALED_SUFFIX}.txt"
    out.write_text("\n".join(lines_tex) + "\n")
    print(f"wrote {out}")

    # full district table for reference
    rows = []
    for ind in INDICATORS:
        for ssp in SSPS:
            d = load_district(ind, ssp, MAIN_TIER)
            if d is not None:
                rows.append(d.assign(indicator=ind, ssp=ssp, tier=MAIN_TIER))
    if rows:
        out2 = OUT_DIR / f"tlo_district_totals_summary{SCALED_SUFFIX}.csv"
        pd.concat(rows)[["indicator", "ssp", "tier", "District", "HSIs_expected",
                         "HSIs_lost", "pct_lost"]].to_csv(out2, index=False)
        print(f"wrote {out2}")


# ------------------------------------------------------------------
# Hot-month vs all-month deficits (from WBGT projection files)
# ------------------------------------------------------------------
def _hot_thresholds():
    """Historical p95 per indicator, as used for hot_deficit_pct in the projection script."""
    files = sorted(glob.glob(str(HEAT_OUT_DIR / f"projection_summary_{WBGT_VAR}*{LAG_SUFFIX}.csv")))
    if not files:
        print("  [skip] hot-month comparison: projection_summary_*.csv not found")
        return None, None
    s = pd.read_csv(files[0])
    thr = s.groupby("indicator")["hist_p95_threshold"].first()
    check = s.set_index(["indicator", "ssp", "tier"])[["deficit_pct", "hot_deficit_pct"]]
    return thr, check


def _hot_split(fac, p95):
    fac = fac[fac["year"].between(YEAR_MIN, YEAR_MAX) & (fac["mu_b"] > 0)]
    hot = fac[WBGT_VAR] > p95
    d = fac["mu_b"] - fac["mu_a"]
    tot_b = fac["mu_b"].sum()

    def pct(m):
        return 100 * d[m].sum() / fac.loc[m, "mu_b"].sum() if m.any() else np.nan

    return {
        "frac_hot": hot.mean(),
        "all_pct": 100 * d.sum() / tot_b,             # what the TLO combination uses
        "hot_pct": pct(hot),                          # = forest-plot hot_deficit_pct
        "nonhot_pct": pct(~hot),
        "hot_share_pct": 100 * d[hot].sum() / tot_b,  # hot-month losses as % of annual volume
        "nonhot_share_pct": 100 * d[~hot].sum() / tot_b,
    }


def write_hot_comparison():
    """Split each projection's deficit into hot (WBGT > historical p95) and non-hot months.
    all_pct = hot_share_pct + nonhot_share_pct."""
    thr, check = _hot_thresholds()
    if thr is None:
        return None
    rows = []
    for ind in INDICATORS:
        if ind not in thr.index:
            print(f"  [skip] hot-month comparison: no threshold for {ind}")
            continue
        for ssp in SSPS:
            for tier in TIERS:
                p = HEAT_OUT_DIR / f"projection_facility_{ind}_{ssp}_{tier}_{WBGT_VAR}{LAG_SUFFIX}.csv"
                if not p.exists():
                    continue
                r = {"indicator": ind, "ssp": ssp, "tier": tier, "p95": thr[ind]}
                r.update(_hot_split(pd.read_csv(p), thr[ind]))
                if (ind, ssp, tier) in check.index:
                    r["summary_deficit_pct"] = check.loc[(ind, ssp, tier), "deficit_pct"]
                    r["summary_hot_deficit_pct"] = check.loc[(ind, ssp, tier), "hot_deficit_pct"]
                rows.append(r)
    if not rows:
        return None
    out = pd.DataFrame(rows)
    path = OUT_DIR / "hot_vs_all_month_deficits.csv"
    out.to_csv(path, index=False)
    print(f"\nHot vs all-month deficits ({SSP_LABELS[MAIN_SSP]}, {MAIN_TIER}):")
    show = out[(out["ssp"] == MAIN_SSP) & (out["tier"] == MAIN_TIER)].drop(columns=["ssp", "tier"])
    with pd.option_context("display.width", 200):
        print(show.round(2).to_string(index=False))
    print(f"wrote {path}")
    return out


def make_hot_plot(hc):
    """Dot plot: all-month vs hot-month vs non-hot-month deficit per indicator, MAIN_SSP.
    Dot = MAIN_TIER; bars = lowest–highest tier."""
    if hc is None or hc.empty:
        return
    sub = hc[hc["ssp"] == MAIN_SSP]
    inds = [i for i in INDICATORS if i in set(sub["indicator"])]
    series = [("all_pct", "All months", "#5A5A5A", -0.22),
              ("hot_pct", "Hot months (> hist. p95)", "#B5515B", 0.0),
              ("nonhot_pct", "Non-hot months", "#4D7799", 0.22)]

    fig, ax = plt.subplots(figsize=(7.5, 0.75 * len(inds) + 1.4))
    for col, label, colour, off in series:
        mid, lo, hi, ys = [], [], [], []
        for k, ind in enumerate(inds):
            r = sub[sub["indicator"] == ind].set_index("tier")[col]
            if MAIN_TIER not in r.index or pd.isna(r[MAIN_TIER]):
                continue
            m = r[MAIN_TIER]
            mid.append(m); lo.append(m - r.min()); hi.append(r.max() - m); ys.append(k + off)
        ax.errorbar(mid, ys, xerr=[lo, hi], fmt="o", markersize=7, capsize=3, linewidth=1.4,
                    color=colour, markeredgecolor="white", markeredgewidth=0.8, label=label)

    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(inds)))
    ax.set_yticklabels([INDICATOR_LABELS.get(i, i) for i in inds])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%"))
    ax.set_xlabel(f"Deficit (% of expected services), {YEAR_MIN}–{YEAR_MAX}", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.005,
             f"{SSP_LABELS[MAIN_SSP]}. Dot: {MAIN_TIER} WBGT model; bars: lowest–highest. "
             "Positive = fewer services than expected without heat.",
             fontsize=7.5, style="italic", color="dimgrey")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = OUT_DIR / f"fig_hot_vs_all_months_{MAIN_SSP}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------
# Services lost per hot month (independent of how many hot months occur)
# ------------------------------------------------------------------
def write_per_hot_month(nat, hc):
    """Services lost nationally in ONE month exceeding the historical WBGT p95.

    = hot_deficit_pct (deficit within hot months, as in the forest plot)
      × mean monthly expected national TLO volume over YEAR_MIN–YEAR_MAX.
    Dot = MAIN_TIER; range = lowest–highest tier. Does not depend on frac_hot.
    """
    if hc is None or hc.empty or nat is None or nat.empty:
        print("  [skip] per-hot-month: needs national totals and hot comparison")
        return None
    n_months = (YEAR_MAX - YEAR_MIN + 1) * 12
    monthly = (nat.drop_duplicates("indicator").set_index("indicator")["expected"] / n_months)

    rows = []
    for (ind, ssp), g in hc.groupby(["indicator", "ssp"]):
        if ind not in monthly.index:
            continue
        pct = g.set_index("tier")["hot_pct"].dropna()
        if MAIN_TIER not in pct.index:
            continue
        m = monthly[ind]
        rows.append({
            "indicator": ind, "ssp": ssp,
            "monthly_expected": m,
            "hot_pct_mid": pct[MAIN_TIER], "hot_pct_lo": pct.min(), "hot_pct_hi": pct.max(),
            "lost_per_hot_month_mid": m * pct[MAIN_TIER] / 100,
            "lost_per_hot_month_lo": m * pct.min() / 100,
            "lost_per_hot_month_hi": m * pct.max() / 100,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return None
    path = OUT_DIR / "tlo_lost_per_hot_month.csv"
    out.to_csv(path, index=False)

    lines_plain = [f"Services lost nationally per month above historical WBGT p95 "
                   f"(monthly volume = mean {YEAR_MIN}-{YEAR_MAX}); {MAIN_TIER} tier (lowest-highest)"]
    lines_tex = [f"% {lines_plain[0]}"]
    for ind in INDICATORS:
        sub = out[out["indicator"] == ind].set_index("ssp")
        if sub.empty:
            continue
        label = INDICATOR_LABELS.get(ind, ind)
        lines_plain.append(f"{label}  (expected per month: {sig(sub['monthly_expected'].iloc[0], 3):,.0f})")
        for ssp in SSPS:
            if ssp not in sub.index:
                continue
            r = sub.loc[ssp]
            mid, lo, hi = (sig(r[c]) for c in ["lost_per_hot_month_mid", "lost_per_hot_month_lo",
                                                "lost_per_hot_month_hi"])
            neg = min(lo, r["hot_pct_lo"]) < 0          # "a to b" reads better than "a--b" with negatives
            sp, st = (" to ", " to ") if neg else ("-", "--")
            lines_plain.append(f"  {SSP_LABELS[ssp]}: {mid:,.0f} ({lo:,.0f}{sp}{hi:,.0f}) per hot month; "
                               f"{r['hot_pct_mid']:.1f}% ({r['hot_pct_lo']:.1f}{sp}{r['hot_pct_hi']:.1f}%)")
            lines_tex.append(f"{label}, {SSP_LABELS[ssp]}: {mid:,.0f} ({lo:,.0f}{st}{hi:,.0f}) services "
                             f"per hot month ({r['hot_pct_mid']:.1f}\\%, "
                             f"{r['hot_pct_lo']:.1f}{st}{r['hot_pct_hi']:.1f}\\%)")
    print("\n" + "\n".join(lines_plain))
    tex = OUT_DIR / "tlo_lost_per_hot_month.txt"
    tex.write_text("\n".join(lines_tex) + "\n")
    print(f"wrote {path}\nwrote {tex}")
    return out


if __name__ == "__main__":
    gdf = load_shapes()
    make_main_maps(gdf)
    nat = national_table()
    make_national_totals(nat)
    write_text_summary(nat)
    hc = write_hot_comparison()
    make_hot_plot(hc)
    write_per_hot_month(nat, hc)
    make_supp_maps(gdf)
