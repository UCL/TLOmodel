

"""
plot_all_figures.py

Regenerates every figure produced by loop_all_indicators_two_model_NB.py
from the CSVs it writes, without refitting the NB models.

Design principle: every plot function reads its inputs from disk and writes
its output to disk. You can call one function to redo a single figure, or
run the whole script to redraw everything. No pickle, no rpy2, no fixest.

Assumes the model script has already run and populated OUT_DIR. Filenames
match the *writer* paths in the model script (WBGT_VAR-suffixed), which
means the timeseries panel here reads the correct file — unlike the reader
in the model script's own plotting block, which is missing the suffix.
"""
import pandas as pd
from pathlib import Path
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
WBGT_VAR = "wbgt_day"
paths = sorted(Path(OUT_DIR).glob(f"exposure_response_curve_*_{WBGT_VAR}.csv"))
#assert paths, "no per-indicator curve files found"
pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).to_csv(
    Path(OUT_DIR) / f"exposure_response_curves_{WBGT_VAR}.csv", index=False)
print(f"wrote {len(paths)} indicators")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

SHAPEFILE_PATH    = ("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/"
                     "resources/mapping/"
                     "ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
DISTRICT_NAME_COL = "ADM2_EN"
CLUSTER_COL       = "Dist"

FDR_ALPHA     = 0.05
IRR_HIGH      = 32.0   # IRR contrast upper bound; low is read from results_df

SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
MODEL_TIERS   = ["lowest", "median", "highest"]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ONLY_DEFICITS = False
SUFFIX = "_onlydeficits" if ONLY_DEFICITS else ""

# Subset + column order for the district × indicator heatmap (Fig. 11).
# Deliberately narrower than the full indicator list — VMMC, skilled
# deliveries, ANC 1st trimester, ANC new attendees, and PNC first-visit <2wks
# are omitted to keep the heatmap readable and consistent with the version
# in plot_district_indicator_heatmap.py.
HEATMAP_INDICATOR_ORDER = [
    "vmmc_first_visits",
    "opd_attendance",
    "ipd_total_admissions",
    "measles1_under1",
    "fully_immunised_under1",
    "fp_total_clients",
    "pnc_mother_checked_48h",
    "penta3_under1",
    "live_births_total",
    "bcg_under1",
    "pnc_within_2wks",
    "anc_total_visits"
]

# Rough Malawi north → south latitude order; cities placed last, separated by
# a dashed rule in the heatmap.
DISTRICT_ORDER = [
    "Chitipa", "Karonga", "Likoma", "Rumphi", "Mzimba", "Nkhata Bay",
    "Kasungu", "Nkhotakota", "Ntchisi", "Dowa", "Salima", "Lilongwe",
    "Mchinji", "Dedza", "Ntcheu", "Mangochi", "Balaka", "Machinga",
    "Zomba", "Chiradzulu", "Blantyre", "Mwanza", "Neno", "Phalombe",
    "Mulanje", "Thyolo", "Chikwawa", "Nsanje",
    "Mzuzu City", "Lilongwe City", "Blantyre City", "Zomba City",
]

INDICATOR_LABELS: dict[str, str] = {
    "fp_total_clients":           "FP Total Clients",
    "opd_attendance":             "OPD Attendance",
    "ipd_total_admissions":       "IPD Total Admissions",
    "vmmc_first_visits":          "VMMC First Visits",
    "pnc_mother_checked_48h":     "PNC Mother <48h",
    "anc_new_attendees":          "ANC New Attendees",
    "anc_first_trimester_starts": "ANC 1st Trimester Starts",
    "bcg_under1":                 "BCG Under-1",
    "penta3_under1":              "Penta3 Under-1",
    "measles1_under1":            "Measles 1st Dose Under-1",
    "fully_immunised_under1":     "Fully Immunised Under-1",
    "pnc_within_2wks":            "PNC Within 2 Weeks",
    "pnc_first_visit_2wks":       "PNC First Visit <2 Weeks",
    "live_births_total":          "Live Births Total",
    "skilled_deliveries":         "Skilled Deliveries",
    "anc_total_visits": "ANC Total Visits",
}

labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"]


# =====================================================================
# HELPERS
# =====================================================================
def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg. Mirrors the model script exactly so re-running
    the plotting recovers the same significance flags."""
    p   = np.asarray(pvals, dtype=float)
    ok  = ~np.isnan(p)
    q   = np.full_like(p, np.nan)
    rej = np.zeros(p.shape, dtype=bool)
    if ok.sum() == 0:
        return q, rej
    p_ok  = p[ok]
    n     = len(p_ok)
    order = np.argsort(p_ok)
    adj   = p_ok[order] * n / np.arange(1, n + 1)
    adj   = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0, 1)
    q_ok        = np.empty(n)
    q_ok[order] = adj
    q[ok]  = q_ok
    rej[ok] = q_ok <= alpha
    return q, rej


def _monthly_jackknife_ci(mu_a, mu_b, facility_ids):
    """Leave-one-facility-out CI on (sum_B - sum_A)/sum_B * 100."""
    sum_a = float(mu_a.sum())
    sum_b = float(mu_b.sum())
    if sum_b <= 0:
        return np.nan, np.nan, np.nan
    deficit_pct = 100.0 * (sum_b - sum_a) / sum_b
    facs = np.unique(facility_ids)
    n    = len(facs)
    if n < 3:
        return deficit_pct, np.nan, np.nan
    jack = []
    for fac in facs:
        keep = facility_ids != fac
        sa_j = float(mu_a[keep].sum())
        sb_j = float(mu_b[keep].sum())
        if sb_j <= 0:
            continue
        jack.append(100.0 * (sa_j - sb_j) / sb_j)
    if len(jack) < 3:
        return deficit_pct, np.nan, np.nan
    jack = np.asarray(jack)
    jack_se = np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))
    return deficit_pct, deficit_pct - 1.96 * jack_se, deficit_pct + 1.96 * jack_se


def _require(path: str) -> str:
    """Loud failure if a required CSV is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input missing: {path}\n"
            f"Run loop_all_indicators_two_model_NB.py first."
        )
    return path


def _label(ind: str) -> str:
    return INDICATOR_LABELS.get(ind, ind)


# =====================================================================
# 1. PER-INDICATOR EXPOSURE-RESPONSE CURVE
# =====================================================================
def plot_exposure_response_curve(indicator: str, out_dir: str = OUT_DIR) -> str:
    """Reads exposure_response_curve_{indicator}_{WBGT_VAR}.csv."""
    csv_path = _require(
        f"{out_dir}exposure_response_curve_{indicator}_{WBGT_VAR}{SUFFIX}.csv")
    curve_df = pd.read_csv(csv_path)
    x_ref = float(curve_df["wbgt_ref"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if curve_df["rr_lo"].notna().any():
        ax.fill_between(
            curve_df["wbgt"], 1/curve_df["rr_lo"], 1/curve_df["rr_hi"],
            color="#2f5d80", alpha=0.2, linewidth=0)
    ax.plot(curve_df["wbgt"], 1/curve_df["rr_vs_ref"], color="#2f5d80", lw=2)
    ax.axhline(1.0, color="black", ls="--", lw=0.9)
    ax.axvline(x_ref, color="#888888", ls=":", lw=1.0)
    ax.set_xlabel("WBGT (°C)")
    ax.set_ylabel("Relative rate vs reference WBGT")
    ax.grid(axis="both", ls=":", alpha=0.4)
    plt.tight_layout()
    out_path = f"{out_dir}exposure_response_curve_{indicator}_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 2 + 3. MAIN FOREST PLOT (aggregate deficit) AND HOT MONTH FOREST PLOT
# =====================================================================
def plot_main_forest(results_df: pd.DataFrame, out_dir: str = OUT_DIR) -> str:
    from scipy.stats import norm

    def _bh_from_jack(pt_col, se_col, lo_col, hi_col):
        pt = results_df[pt_col].values
        if se_col in results_df.columns:
            se = results_df[se_col].values
        else:
            # Derive SE from the CI when the SE column isn't written by the
            # model script: CI = pt ± 1.96 * SE  ⇒  SE = (hi - lo) / (2*1.96)
            se = (results_df[hi_col].values - results_df[lo_col].values) / (2 * 1.96)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, np.abs(pt / se), np.nan)
            p = 2 * (1 - norm.cdf(z))
        _, rej = bh_fdr(p, alpha=FDR_ALPHA)
        return p, rej

    results_df = results_df.copy()
    p_all, rej_all = _bh_from_jack("deficit_pct", "se_jackknife", "ci_lo", "ci_hi")
    results_df["p_bh"] = p_all
    results_df["sig_bh"] = rej_all
    p_hot, rej_hot = _bh_from_jack(
        "hot_deficit_pct", "hot_se_jackknife", "hot_ci_lo", "hot_ci_hi"
    )
    results_df["p_hot_bh"] = p_hot
    results_df["sig_hot_bh"] = rej_hot

    plot_df = results_df.sort_values("hot_deficit_pct").reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    has_ci = plot_df["ci_lo"].notna().any()
    colors = [
        "#823038" if bool(r.get("sig_bh", False))
        else ("#888888" if has_ci else "#4a7298")
        for _, r in plot_df.iterrows()
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, max(4, len(plot_df) * 0.55 + 1.5)))
    axes = axes.flatten()

    # ---- Panel A: aggregate deficit -------------------------------------
    for i, row in plot_df.iterrows():
        if pd.notna(row["ci_lo"]):
            axes[0].plot(
                [row["ci_lo"], row["ci_hi"]], [i, i],
                color=colors[i], lw=1.4, zorder=1,
            )
    axes[0].scatter(plot_df["deficit_pct"], y_pos, color=colors, s=55, zorder=2)
    axes[0].axvline(0, color="black", ls="--", lw=0.9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(plot_df["label"], fontsize=9)
    axes[0].set_xlabel("% change in appointments associated with WBGT", fontsize=10)
    axes[0].grid(axis="x", ls=":", alpha=0.5)
    if has_ci:
        axes[0].legend(
            handles=[
                mpatches.Patch(color="#823038", label=f"BH-FDR q≤{FDR_ALPHA}"),
                mpatches.Patch(color="#888888", label="not significant"),
            ],
            loc="lower right", fontsize=9, frameon=False,
        )
    ax2 = axes[0].twinx()
    ax2.set_ylim(axes[0].get_ylim())
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"θ={r['alpha']:.1f}" for _, r in plot_df.iterrows()],
        fontsize=7, color="#666666",
    )
    ax2.tick_params(axis="y", length=0)

    # ---- Panel B: hot-month deficit -------------------------------------
    ph = (
        results_df.dropna(subset=["hot_deficit_pct"])
        .sort_values("hot_deficit_pct")
        .reset_index(drop=True)
    )
    y_ph = np.arange(len(ph))
    hot_colors = ["#823038" if bool(s) else "#888888" for s in ph["sig_hot_bh"]]

    hot_thresh_vals = ph["hot_threshold"].dropna()
    if hot_thresh_vals.empty:
        hot_thresh_label = "hot months"
    elif hot_thresh_vals.nunique() == 1:
        hot_thresh_label = f">{hot_thresh_vals.iloc[0]:.1f}°C"
    else:
        hot_thresh_label = f">{hot_thresh_vals.min():.1f}–{hot_thresh_vals.max():.1f}°C"

    for i, (_, row) in enumerate(ph.iterrows()):
        pt = row["hot_deficit_pct"]
        lo = row["hot_ci_lo"]
        hi = row["hot_ci_hi"]
        if pd.notna(lo) and pd.notna(hi):
            axes[1].errorbar(
                pt, i,
                xerr=[[pt - lo], [hi - pt]],
                fmt="o", markersize=7, capsize=4, capthick=1.4,
                elinewidth=1.4, color=hot_colors[i], zorder=2,
            )
        else:
            axes[1].scatter(pt, i, color=hot_colors[i], s=55, zorder=2)

    axes[1].axvline(0, color="black", ls="--", lw=0.9)
    axes[1].set_yticks(y_ph)
    axes[1].set_yticklabels(ph["label"], fontsize=9)
    axes[1].set_xlabel(
        f"% Deficit in appointments during hottest months ({hot_thresh_label})",
        fontsize=10,
    )
    axes[1].grid(axis="x", ls=":", alpha=0.5)
    axes[1].legend(
        handles=[
            mpatches.Patch(color="#823038", label=f"BH-FDR q ≤ {FDR_ALPHA}"),
            mpatches.Patch(color="#888888", label="not significant"),
        ],
        loc="lower right", fontsize=9, frameon=False,
    )

    plt.tight_layout()
    axes[0].text(-0.1, 1.05, "(A)", transform=axes[0].transAxes,
                 fontsize=18, va="top", ha="right")
    axes[1].text(-0.1, 1.05, "(B)", transform=axes[1].transAxes,
                 fontsize=18, va="top", ha="right")

    out_path = f"{out_dir}forest_plot_hot_deficit_NB_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 4. IRR FOREST PLOT (spline contrast)
# =====================================================================
def plot_irr_forest(results_df: pd.DataFrame, out_dir: str = OUT_DIR) -> str:
    irr_df = pd.read_csv(_require(f"{out_dir}irr_contrast_{WBGT_VAR}.csv"))
    if irr_df.empty:
        print("  IRR csv empty — skipping IRR forest plot")
        return ""

    # Sort and reset index so position i == row i throughout.
    irr_df = irr_df.sort_values("irr").reset_index(drop=True)

    # The "low" reference WBGT is stored per-indicator in irr_df itself.
    # Use the median across indicators for the axis label (or the first row
    # if all are identical, which is the common case).
    irr_low = float(irr_df["reference_wbgt"].median())

    irr_colors = [
        "#823038" if (row["irr_hi"] < 1.0 or row["irr_lo"] > 1.0) else "#888888"
        for _, row in irr_df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(7, max(4, len(irr_df) * 0.55 + 1.5)))

    for i, (_, row) in enumerate(irr_df.iterrows()):   # enumerate gives correct y
        ax.errorbar(
            row["irr"], i,
            xerr=[[row["irr"] - row["irr_lo"]],
                  [row["irr_hi"] - row["irr"]]],
            fmt="o", markersize=7, capsize=8, capthick=1.4,
            elinewidth=1.4, color=irr_colors[i], zorder=2,
            ecolor="black",
        )

    ax.axvline(1.0, color="black", ls="--", lw=0.9)
    ax.set_yticks(range(len(irr_df)))
    ax.set_yticklabels(irr_df["label"], fontsize=9)   # FIX 3: read from irr_df, not results_df
    ax.set_xlabel(
        f"IRR: WBGT {IRR_HIGH:.0f}°C vs {irr_low:.0f}°C (25th pctile)",
        fontsize=10,
    )
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.legend(
        handles=[
            mpatches.Patch(color="#823038", label="CI excludes 1.0"),
            mpatches.Patch(color="#888888", label="CI includes 1.0"),
        ],
        loc="lower right", fontsize=9, frameon=False,
    )
    plt.tight_layout()
    out_path = (
        f"{out_dir}forest_plot_IRR_{irr_low:.0f}_{IRR_HIGH:.0f}_NB_{WBGT_VAR}.png"
    )
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 5. EXPOSURE-RESPONSE PANEL (all indicators)
# =====================================================================
def plot_exposure_response_panel(fitted: list[str], out_dir: str = OUT_DIR) -> str:
    curves_df = pd.read_csv(_require(f"{out_dir}exposure_response_curves_{WBGT_VAR}.csv"))
    inds = [i for i in fitted if i in curves_df["indicator"].unique()]
    if not inds:
        print("  no exposure-response rows — skipping panel")
        return ""

    n_ind = len(inds)
    n_cols = 3
    n_rows = int(np.ceil(n_ind / n_cols))

    # Share x across all; y only among non-VMMC panels (added manually below).
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3 * n_rows),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Find the first non-VMMC axis and use it as the shared-y anchor.
    non_vmmc_idx = [i for i, ind in enumerate(inds) if ind != "vmmc_first_visits"]
    anchor_idx = non_vmmc_idx[0] if non_vmmc_idx else 0
    anchor_ax = axes_flat[anchor_idx]

    for idx, ind in enumerate(inds):
        ax = axes_flat[idx]
        if ind != "vmmc_first_visits" and ax is not anchor_ax:
            ax.sharey(anchor_ax)

        sub = curves_df[curves_df["indicator"] == ind].sort_values("wbgt")
        ref_w = float(sub["wbgt_ref"].iloc[0])

        ax.fill_between(sub["wbgt"], 1 / sub["rr_lo"], 1 / sub["rr_hi"],
                        color="#B17776", alpha=0.25, linewidth=0)
        ax.plot(sub["wbgt"], 1 / sub["rr_vs_ref"], color="#CEB5C8", lw=1.5)
        ax.axhline(1.0, color="black", lw=0.5, ls="--")
        ax.axvline(ref_w, color="grey", lw=0.5, ls=":")
        ax.set_title(_label(ind), fontsize=9, fontweight="bold")

        # y-labels: leftmost column, plus VMMC always (its scale is its own).
        if idx % n_cols == 0 or ind == "vmmc_first_visits":
            ax.set_ylabel("IRR", fontsize=8)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("WBGT (°C)", fontsize=8)
        ax.tick_params(labelsize=7)
        # Force tick labels on VMMC since it's not on the shared scale.
        if ind == "vmmc_first_visits":
            ax.tick_params(labelleft=True)
        ax.annotate(labels[idx], xy=(0.05, 1.05), xycoords="axes fraction", size = 14)

    for idx in range(len(inds), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    out_path = f"{out_dir}exposure_response_panel_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path

# =====================================================================
# 6. MONTHLY DEFICIT PANEL (jackknife CI)
# =====================================================================
def plot_monthly_deficit_panel(fitted: list[str], out_dir: str = OUT_DIR) -> str:
    n_ind = len(fitted)
    n_cols = 3
    n_rows = int(np.ceil(n_ind / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows), sharex=True, squeeze=False
    )  # Force axes to always be a 2D array

    # Now axes is guaranteed to be a 2D array, flatten it for easy iteration
    axes_flat = axes.flatten()

    for idx, ind in enumerate(fitted):
        ax = axes_flat[idx]  # Get individual Axes object
        csv_path = f"{out_dir}historical_burden_{ind}_{WBGT_VAR}.csv"
        if not os.path.exists(csv_path):
            ax.set_visible(False)
            continue

        bdf = pd.read_csv(csv_path)
        fac_ids = bdf["facility"].values
        months = bdf["month"].values
        mu_a = bdf["mu_a"].values
        mu_b = bdf["mu_b"].values

        pcts, los, his = [], [], []
        for m in range(1, 13):
            mask = months == m
            if not mask.any():
                pcts.append(0)
                los.append(np.nan)
                his.append(np.nan)
                continue
            pt, lo, hi = _monthly_jackknife_ci(mu_a[mask], mu_b[mask], fac_ids[mask])
            pcts.append(pt)
            los.append(lo)
            his.append(hi)

        pcts_a = np.asarray(pcts, dtype=float)
        los_a = np.asarray(los, dtype=float)
        his_a = np.asarray(his, dtype=float)
        bar_c = ["#823038" if p > 0 else "#2a78d6" for p in pcts_a]
        yerr = np.array(
            [
                np.nan_to_num(pcts_a - los_a, nan=0.0),
                np.nan_to_num(his_a - pcts_a, nan=0.0),
            ]
        )

        # Now ax is a proper Axes object, not an array
        ax.bar(
            range(12), pcts_a, color=bar_c, alpha=0.8, yerr=yerr, error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": "#333"}
        )
        ax.set_xticks(range(12))
        ax.set_xticklabels(MONTH_NAMES, fontsize=6, rotation=45)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(_label(ind), fontsize=9, fontweight="bold")

        if idx % n_cols == 0:
            ax.set_ylabel("% deficit", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for idx in range(n_ind, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    out_path = f"{out_dir}deficit_by_month_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 7. TIMESERIES BURDEN PANEL
# =====================================================================
def plot_timeseries_panel(fitted: list[str], out_dir: str = OUT_DIR) -> str:
    n_ind = len(fitted)
    nc = 3
    nr = int(np.ceil(n_ind / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5.5 * nc, 3.5 * nr), sharex=True)
    af = axes.flatten() if n_ind > 1 else [axes]

    for idx, ind in enumerate(fitted):
        ax  = af[idx]
        # NB: WBGT_VAR-suffixed path (writer path), unlike the model script's
        # own timeseries block which reads the unsuffixed path.
        csv_path = f"{out_dir}historical_burden_{ind}_{WBGT_VAR}.csv"
        if not os.path.exists(csv_path):
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv_path, parse_dates=["date"])
        m  = (df.groupby("date")
              .agg(obs=("y_int", "sum"),
                   mu_a=("mu_a", "sum"),
                   mu_b=("mu_b", "sum"))
              .sort_index())
        ax.plot(m.index, m["mu_b"], color="#2a78d6", lw=1.0, ls="--",
                alpha=0.8, label="Model B (no weather)")
        ax.plot(m.index, m["obs"],  color="#333", lw=1.0, label="Observed")
        ax.fill_between(m.index, m["mu_a"], m["mu_b"],
                        where=m["mu_a"] < m["mu_b"],
                        color="#823038", alpha=0.25, label="Heat deficit")
        ax.set_title(_label(ind), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6)

    for idx in range(n_ind, len(af)):
        af[idx].set_visible(False)
    plt.tight_layout()
    out_path = f"{out_dir}timeseries_burden_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 8. DISTRICT CHOROPLETH MAPS
# =====================================================================
def plot_district_maps(fitted: list[str], out_dir: str = OUT_DIR) -> list[str]:
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not installed — skipping maps.")
        return []
    if not os.path.exists(SHAPEFILE_PATH):
        print(f"Shapefile not found at {SHAPEFILE_PATH} — skipping maps.")
        return []

    shp = gpd.read_file(SHAPEFILE_PATH)
    shp[DISTRICT_NAME_COL] = (
        shp[DISTRICT_NAME_COL].astype(str).str.strip().str.title())

    # Load per-indicator district CSVs — including the CI file for sig.
    frames = []
    for ind in fitted:
        p_def = f"{out_dir}district_burden_{ind}_{WBGT_VAR}{SUFFIX}.csv"
        p_ci  = f"{out_dir}district_burden_ci_{ind}_{WBGT_VAR}{SUFFIX}.csv"
        if not os.path.exists(p_def):
            print(f"  {ind}: district csv missing — skipping in maps")
            continue
        d = pd.read_csv(p_def)
        d["indicator"] = ind
        if os.path.exists(p_ci):
            ci = pd.read_csv(p_ci)[["district", "sig"]].rename(
                columns={"district": CLUSTER_COL})
            d = d.merge(ci, on=CLUSTER_COL, how="left")
        else:
            d["sig"] = False
        frames.append(d)
    if not frames:
        return []
    dist_all = pd.concat(frames, ignore_index=True)
    dist_all[CLUSTER_COL] = (
        dist_all[CLUSTER_COL].astype(str).str.strip().str.title())

    # Custom diverging colormap.
    hex_colors = ['#4D7799', '#7FA4C4', '#C5C8D4', '#D48E95', '#B5515B']
    custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', hex_colors)

    # Panel labels for individual maps.
    labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"]

    out_paths = []

    # ---- one map per indicator ----
    for idx, ind in enumerate(fitted):
        sub = dist_all[dist_all["indicator"] == ind].copy()
        if sub.empty:
            continue
        merged = shp.merge(
            sub[[CLUSTER_COL, "deficit_pct", "sig"]],
            left_on=DISTRICT_NAME_COL,
            right_on=CLUSTER_COL, how="left")
        n_matched = merged["deficit_pct"].notna().sum()
        n_sig = merged["sig"].fillna(False).sum()
        print(f"  {ind}: {n_matched}/{len(merged)} districts matched, "
              f"{n_sig} significant")

        vmax = max(merged["deficit_pct"].abs().quantile(0.95), 0.01)
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        ax.annotate(labels[idx] if idx < len(labels) else "",
                    xy=(0.05, 1.05), xycoords="axes fraction", size=14)

        merged.plot(
            column="deficit_pct", ax=ax,
            cmap=custom_cmap, vmin=-vmax, vmax=vmax,
            edgecolor="white", linewidth=0.4,
            missing_kwds={"color": "#cccccc", "label": "No data"},
            legend=True,
            legend_kwds={"label": "% deficit (Model A vs B)",
                         "orientation": "horizontal",
                         "shrink": 0.7, "pad": 0.02})

        # Hatch NON-significant districts (fade them to highlight sig ones).
        sig_mask = merged["sig"].fillna(False)
        non_sig = merged[~sig_mask & merged["deficit_pct"].notna()]
        if not non_sig.empty:
            non_sig.plot(
                ax=ax, facecolor="none", edgecolor="black",
                linewidth=0.4, hatch="///")

        ax.set_axis_off()
        fig.text(0.5, 0.02,
                 "Hatched: 95% jackknife CI includes zero (not significant)",
                 ha="center", fontsize=9, style="italic")
        plt.tight_layout()
        out_path = f"{out_dir}map_district_deficit_{ind}_{WBGT_VAR}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        out_paths.append(out_path)

    # ---- summary panel (per-map scales, with hatching) ----
    n_ind = len(fitted)
    n_cols = 3
    n_rows = int(np.ceil(n_ind / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows))
    af = axes.flatten() if n_ind > 1 else [axes]

    for idx, ind in enumerate(fitted):
        ax = af[idx]
        sub = dist_all[dist_all["indicator"] == ind].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        merged = shp.merge(
            sub[[CLUSTER_COL, "deficit_pct", "sig"]],
            left_on=DISTRICT_NAME_COL,
            right_on=CLUSTER_COL, how="left")
        local_vmax = max(merged["deficit_pct"].abs().quantile(0.95), 0.01)

        merged.plot(
            column="deficit_pct", ax=ax,
            cmap=custom_cmap, vmin=-local_vmax, vmax=local_vmax,
            edgecolor="white", linewidth=0.3,
            missing_kwds={"color": "#cccccc"},
            legend=True,
            legend_kwds={"shrink": 0.5, "orientation": "horizontal", "pad": 0.02})

        # Hatch NON-significant, excluding no-data cells.
        sig_mask = merged["sig"].fillna(False)
        non_sig = merged[~sig_mask & merged["deficit_pct"].notna()]
        if not non_sig.empty:
            non_sig.plot(
                ax=ax, facecolor="none", edgecolor="black",
                linewidth=0.3, hatch="///")

        ax.set_axis_off()
        ax.set_title(f"{_label(ind)}  (±{local_vmax:.2f}%)",
                     fontsize=9, fontweight="bold")

    for idx in range(n_ind, len(af)):
        af[idx].set_visible(False)

    fig.text(0.5, 0.005,
             "Hatched: 95% jackknife CI includes zero (not significant). "
             "Grey: no data.",
             ha="center", fontsize=10, style="italic")

    panel_path = f"{out_dir}map_district_deficit_panel_{WBGT_VAR}.png"
    plt.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close()
    out_paths.append(panel_path)
    return out_paths


# =====================================================================
# 9. PROJECTION HEATMAPS
# =====================================================================
def plot_projection_heatmaps(fitted: list[str],
                              out_dir: str = OUT_DIR) -> list[str]:
    csv_path = f"{out_dir}projection_summary_{WBGT_VAR}{SUFFIX}.csv"
    if not os.path.exists(csv_path):
        print(f"  projection summary missing at {csv_path} — skipping")
        return []
    proj_df = pd.read_csv(csv_path)

    out_paths = []
    for ind in fitted:
        sub = proj_df[proj_df["indicator"] == ind]
        if sub.empty:
            continue
        grid = pd.DataFrame(
            index=SSP_SCENARIOS, columns=MODEL_TIERS, dtype=float)
        for _, p in sub.iterrows():
            if p["ssp"] in SSP_SCENARIOS and p["tier"] in MODEL_TIERS:
                grid.loc[p["ssp"], p["tier"]] = p["deficit_pct"]
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(grid.values.astype(float), cmap="RdBu_r",
                       aspect="auto")
        ax.set_xticks(range(len(MODEL_TIERS)))
        ax.set_xticklabels(MODEL_TIERS, fontsize=9)
        ax.set_yticks(range(len(SSP_SCENARIOS)))
        ax.set_yticklabels(SSP_SCENARIOS, fontsize=9)
        for i in range(len(SSP_SCENARIOS)):
            for j in range(len(MODEL_TIERS)):
                val = grid.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:+.2f}%",
                            ha="center", va="center", fontsize=10)
        plt.colorbar(im, ax=ax).set_label(
            "Δ deficit (proj−hist, %)", fontsize=9)
        plt.tight_layout()
        out_path = f"{out_dir}projection_heatmap_{ind}_{WBGT_VAR}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        out_paths.append(out_path)
    return out_paths


# =====================================================================
# 10. TLO DISRUPTION CURVES
# =====================================================================
def plot_tlo_curves(results_df: pd.DataFrame, out_dir: str = OUT_DIR) -> str:
    tlo_path = f"{out_dir}tlo_wbgt_lookup{SUFFIX}.csv"  # writer path (no suffix)
    if not os.path.exists(tlo_path):
        # tolerate the alternative suffix in case you fix the writer later
        alt = f"{out_dir}tlo_wbgt_lookup_{WBGT_VAR}{SUFFIX}.csv"
        if os.path.exists(alt):
            tlo_path = alt
        else:
            print(f"  TLO lookup missing at {tlo_path} — skipping")
            return ""
    tlo = pd.read_csv(tlo_path)
    ref_wbgt = float(results_df["reference_wbgt"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    for ind in tlo["indicator"].unique():
        sub = tlo[tlo["indicator"] == ind]
        ax.plot(sub["wbgt"], sub["disruption_probability"], lw=1.3,
                label=_label(ind))
    ax.set_xlabel("WBGT (°C)")
    ax.set_ylabel(f"Disruption probability (vs {ref_wbgt:.1f}°C)")
    ax.legend(fontsize=7)
    ax.grid(ls=":", alpha=0.4)
    ax.set_ylim(bottom=-0.01)
    plt.tight_layout()
    out_path = f"{out_dir}tlo_disruption_curves_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 11. DISTRICT × INDICATOR HEATMAP
# =====================================================================
def plot_district_indicator_heatmap(
    out_dir: str = OUT_DIR,
    indicator_order: list[str] = HEATMAP_INDICATOR_ORDER,
    district_order: list[str] = DISTRICT_ORDER,
    vabs: float = 2.0,
) -> str:
    """Reads district_burden_ci_{ind}_{WBGT_VAR}.csv per indicator and plots
    a district × indicator heatmap of services-lost (%) with delta-method
    significance flagged by a black border on each cell.
    """
    rows = []
    for ind in indicator_order:
        # Writer path uses WBGT_VAR suffix; the standalone heatmap script
        # was reading without the suffix and quietly getting nothing.
        path = f"{out_dir}district_burden_ci_{ind}_{WBGT_VAR}{SUFFIX}.csv"
        if not os.path.exists(path):
            print(f"  [{ind}] no district CI CSV — skipping")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "district":          r["district"],
                "indicator":         ind,
                "services_lost_pct": -r["deficit_pct"],   # sign flip: see docstring
                "sig":               bool(r["sig"]),
            })
    if not rows:
        print("  no district CI rows found — skipping heatmap")
        return ""

    long_df  = pd.DataFrame(rows)
    wide     = (long_df.pivot(index="district", columns="indicator",
                              values="services_lost_pct")
                .reindex(columns=indicator_order))
    sig_wide = (long_df.pivot(index="district", columns="indicator",
                              values="sig")
                .reindex(index=wide.index, columns=wide.columns))

    # Reorder rows N → S, then cities; drop anything not in DISTRICT_ORDER.
    ordered_rows = [d for d in district_order if d in wide.index]
    missing = set(wide.index) - set(district_order)
    if missing:
        print(f"  heatmap: {len(missing)} district(s) not in DISTRICT_ORDER "
              f"— dropped: {sorted(missing)}")
    wide     = wide.reindex(ordered_rows)
    sig_wide = sig_wide.reindex(ordered_rows)

    fig, ax = plt.subplots(figsize=(11, max(6, 0.4 * len(wide) + 2)))
    im = ax.imshow(wide.values, cmap="RdBu_r",
                    vmin=-vabs, vmax=vabs, aspect="auto")

    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels(
        [_label(c) for c in wide.columns],
        rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels(wide.index, fontsize=8)

    # Numeric annotations (skip near-zero to reduce clutter).
    for i in range(len(wide.index)):
        for j in range(len(wide.columns)):
            v = wide.values[i, j]
            if pd.notna(v) and abs(v) >= 0.05:
                ax.text(j, i, f"{v:+.1f}",
                        ha="center", va="center", fontsize=6.5,
                        color="black" if abs(v) < vabs * 0.7 else "white")

    # Bold outline on cells where the delta-method CI excludes 0.
    for i in range(len(wide.index)):
        for j in range(len(wide.columns)):
            if sig_wide.values[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="black", lw=1.5, zorder=3))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=25, pad=0.02)
    cbar.set_label("% services lost to heat", fontsize=9)

    # Separator between rural districts and cities.
    n_rural = sum(1 for d in wide.index if "City" not in d)
    if 0 < n_rural < len(wide.index):
        ax.axhline(n_rural - 0.5, color="black", lw=1.2, linestyle="--")

    plt.tight_layout()
    out_path = f"{out_dir}district_indicator_heatmap_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path

SSP_COLOURS = {
    "ssp126": "#9BB29E",  # green: low emissions
    "ssp245": "#F1DCBA",  # amber: middle
    "ssp585": "#D45C5D",  # red: high emissions
}
SSP_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5",
}


# =====================================================================
# 12. PROJECTION FOREST — end-of-period % deficit per indicator × SSP
# =====================================================================
def plot_projection_forest(
    fitted: list[str],
    out_dir: str = None,
    wbgt_var: str = None,
    window: tuple[int, int] = (2025, 2040),
) -> str:
    """End-of-period projection forest.

    One row per indicator. Three points per row (one per SSP, colour-coded),
    horizontally offset. Error bar = range across GCM tiers (lowest/median/
    highest). This isn't a formal CI — it's the modelling uncertainty from
    the climate ensemble, which is what actually dominates on this horizon.

    Reads projection_annual_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv and averages
    Deficit_Pct over `window` (calendar years, inclusive).
    """

    out_dir = out_dir or OUT_DIR
    wbgt_var = wbgt_var or WBGT_VAR

    rows = []
    for ind in fitted:
        for ssp in ["ssp126", "ssp245", "ssp585"]:
            tier_vals = {}
            for tier in ["lowest", "median", "highest"]:
                p = f"{out_dir}projection_annual_{ind}_{ssp}_{tier}_{wbgt_var}{SUFFIX}.csv"
                if not os.path.exists(p):
                    continue
                df = pd.read_csv(p)
                sel = df[df["year"].between(*window)]
                if sel.empty:
                    continue
                # Volume-weighted mean deficit over the window: sum mu_a
                # and mu_b, THEN compute pct. This is the same aggregation
                # the historical figures use — averaging monthly percentages
                # would weight low-service months equally.
                tot_a = float(sel["mu_a"].sum())
                tot_b = float(sel["mu_b"].sum())
                if tot_b > 0:
                    tier_vals[tier] = 100.0 * (tot_b - tot_a) / tot_b
            if not tier_vals:
                continue
            vals = list(tier_vals.values())
            rows.append(
                {
                    "indicator": ind,
                    "ssp": ssp,
                    "median": tier_vals.get("median", np.median(vals)),
                    "lo": min(vals),
                    "hi": max(vals),
                    "n_tiers": len(vals),
                }
            )
    if not rows:
        print("  no projection rows found for forest — skipping")
        return ""

    df = pd.DataFrame(rows)

    # Sort indicators by SSP245 median (central scenario), descending —
    # most-affected at the top.
    ranker = df[df["ssp"] == "ssp245"].set_index("indicator")["median"].reindex(fitted).fillna(-np.inf)
    ind_order = ranker.sort_values(ascending=True).index.tolist()

    y_pos = {ind: i for i, ind in enumerate(ind_order)}
    ssp_offset = {"ssp126": -0.22, "ssp245": 0.0, "ssp585": +0.22}

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(ind_order) + 1.5)))

    for _, r in df.iterrows():
        y = y_pos[r["indicator"]] + ssp_offset[r["ssp"]]
        colour = SSP_COLOURS[r["ssp"]]
        pt = r["median"]
        ax.plot([r["lo"], r["hi"]], [y, y], color=colour, lw=1.4, alpha=0.8)
        ax.scatter(pt, y, color=colour, s=45, zorder=3, edgecolor="white", linewidth=0.6)

    ax.axvline(0, color="black", ls="--", lw=0.9)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([_label(i) for i in ind_order], fontsize=9)
    ax.set_xlabel(
        f"Projected % difference, mean over {window[0]}–{window[1]}",
        fontsize=10,
    )
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.legend(
        handles=[mpatches.Patch(color=SSP_COLOURS[s], label=SSP_LABELS[s]) for s in ["ssp126", "ssp245", "ssp585"]],
        loc="lower right",
        fontsize=9,
        frameon=False,
        title="Error bars: GCM tier range",
        title_fontsize=8,
    )
    plt.tight_layout()
    out_path = f"{out_dir}projection_forest_{window[0]}_{window[1]}_{wbgt_var}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 13. PROJECTION ANNUAL TRAJECTORY PANEL
# =====================================================================
def plot_projection_annual_panel(
    fitted: list[str],
    out_dir: str = None,
    wbgt_var: str = None,
) -> str:
    """Annual % deficit trajectory panel.

    One subplot per indicator. For each SSP: median-tier line, with a shaded
    ribbon spanning the lowest–highest tier range. This is the main
    projection figure — the forest above is its end-of-period summary.
    """

    out_dir = out_dir or OUT_DIR
    wbgt_var = wbgt_var or WBGT_VAR

    # Prefer the concatenated file; fall back to per-file glob.
    concat_path = f"{out_dir}projection_annual_all_{wbgt_var}{SUFFIX}.csv"
    if os.path.exists(concat_path):
        big = pd.read_csv(concat_path)
    else:
        parts = []
        for p in sorted(Path(out_dir).glob(f"projection_annual_*_{wbgt_var}{SUFFIX}.csv")):
            parts.append(pd.read_csv(p))
        if not parts:
            print("  no annual projection files — skipping trajectory panel")
            return ""
        big = pd.concat(parts, ignore_index=True)

    inds = [i for i in fitted if i in big["indicator"].unique()]
    if not inds:
        print("  no matching indicators in annual projection file")
        return ""

    n = len(inds)
    nc = min(3, n)
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.2 * nr), sharex=True, squeeze=False)
    axes_flat = axes.flatten()

    for idx, ind in enumerate(inds):
        ax = axes_flat[idx]
        sub = big[big["indicator"] == ind]
        for ssp in ["ssp126", "ssp245", "ssp585"]:
            ssp_sub = sub[sub["ssp"] == ssp]
            if ssp_sub.empty:
                continue
            wide = ssp_sub.pivot_table(index="year", columns="tier", values="Deficit_Pct")
            colour = SSP_COLOURS[ssp]
            years = wide.index.values
            if {"lowest", "highest"}.issubset(wide.columns):
                ax.fill_between(
                    years,
                    wide["lowest"],
                    wide["highest"],
                    color=colour,
                    alpha=0.15,
                    linewidth=0,
                )
            if "median" in wide.columns:
                ax.plot(years, wide["median"], color=colour, lw=1.6, label=SSP_LABELS[ssp])
            else:
                # No median tier — plot the row-wise mean of what we have.
                ax.plot(years, wide.mean(axis=1), color=colour, lw=1.6, label=SSP_LABELS[ssp], ls=":")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(_label(ind), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(ls=":", alpha=0.3)
        if idx % nc == 0:
            ax.set_ylabel("% deficit", fontsize=8)
        if idx // nc == nr - 1:
            ax.set_xlabel("Year", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    out_path = f"{out_dir}projection_annual_panel_{wbgt_var}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 14. SEASONAL AMPLIFICATION PANEL
# =====================================================================
def plot_seasonal_amplification(
    fitted: list[str],
    out_dir: str = None,
    wbgt_var: str = None,
    ssp: str = "ssp245",
    tier: str = "median",
    early: tuple[int, int] = (2025, 2029),
    late: tuple[int, int] = (2036, 2040),
) -> str:
    """Early vs late projection deficit by calendar month.

    Under one SSP/tier combination, compare mean deficit-by-month in an early
    window vs a late window. If heat exposure is concentrating rather than
    spreading uniformly across the year, the hot-season peak will grow
    disproportionately.

    Uses projection_monthly_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv.
    """
    out_dir = out_dir or OUT_DIR
    wbgt_var = wbgt_var or WBGT_VAR

    inds_present = []
    for ind in fitted:
        p = f"{out_dir}projection_monthly_{ind}_{ssp}_{tier}_{wbgt_var}{SUFFIX}.csv"
        if os.path.exists(p):
            inds_present.append(ind)
    if not inds_present:
        print(f"  no monthly projection files for {ssp}/{tier} — skipping")
        return ""

    n = len(inds_present)
    nc = min(3, n)
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.2 * nr), sharex=True, squeeze=False)
    axes_flat = axes.flatten()

    x = np.arange(12)
    for idx, ind in enumerate(inds_present):
        ax = axes_flat[idx]
        df = pd.read_csv(f"{out_dir}projection_monthly_{ind}_{ssp}_{tier}_{wbgt_var}{SUFFIX}.csv")

        # Volume-weighted per-month across each window.
        def _by_month(window):
            sel = df[df["year"].between(*window)]
            agg = sel.groupby("month").agg(mu_a=("mu_a", "sum"), mu_b=("mu_b", "sum"))
            agg["pct"] = np.where(
                agg["mu_b"] > 0,
                100.0 * (agg["mu_b"] - agg["mu_a"]) / agg["mu_b"],
                np.nan,
            )
            return agg["pct"].reindex(range(1, 13)).values

        early_pct = _by_month(early)
        late_pct = _by_month(late)

        ax.plot(x, early_pct, marker="o", color="#4a7298", lw=1.4, ms=4, label=f"{early[0]}–{early[1]}")
        ax.plot(x, late_pct, marker="o", color="#d7191c", lw=1.4, ms=4, label=f"{late[0]}–{late[1]}")
        ax.fill_between(x, early_pct, late_pct, where=late_pct > early_pct, color="#d7191c", alpha=0.12, linewidth=0)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_NAMES, fontsize=6, rotation=45)
        ax.set_title(_label(ind), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(ls=":", alpha=0.3)
        if idx % nc == 0:
            ax.set_ylabel("% deficit", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Seasonal deficit — {SSP_LABELS.get(ssp, ssp)} / {tier} GCM",
        fontsize=10,
        y=1.00,
    )
    plt.tight_layout()
    out_path = f"{out_dir}seasonal_amplification_{ssp}_{tier}_{wbgt_var}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path

# =====================================================================
# 15/16. SUMMARY STATISTICS — respects ONLY_DEFICITS toggle
# =====================================================================
def _row_mask(df: pd.DataFrame) -> pd.Series:
    """ON → only rows where baseline > weather; OFF → all rows."""
    if ONLY_DEFICITS:
        return df["mu_b"] > df["mu_a"]
    return pd.Series(True, index=df.index)


def _agg_deficit(df: pd.DataFrame) -> tuple[float, float, float]:
    sub = df[_row_mask(df)]
    b = float(sub["mu_b"].sum())
    a = float(sub["mu_a"].sum())
    pct = 100.0 * (b - a) / b if b > 0 else np.nan
    return b, a, pct


def _jackknife_deficit_ci(df: pd.DataFrame) -> tuple[float, float, float]:
    """Leave-one-facility-out CI on the volume-weighted deficit %.
    Returns (pct, ci_lo, ci_hi). Filtering (ONLY_DEFICITS) is applied
    inside every jackknife replicate, matching the point estimate."""
    b, a, pct = _agg_deficit(df)
    if not np.isfinite(pct):
        return pct, np.nan, np.nan
    facs = df["facility"].unique()
    n = len(facs)
    if n < 3:
        return pct, np.nan, np.nan
    jack = []
    for f in facs:
        sub = df[df["facility"] != f]
        _, _, p = _agg_deficit(sub)
        if np.isfinite(p):
            jack.append(p)
    if len(jack) < 3:
        return pct, np.nan, np.nan
    jack = np.asarray(jack)
    se = np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))
    return pct, pct - 1.96 * se, pct + 1.96 * se


def calculate_summary_statistics_by_indicator(
    fitted: list[str], out_dir: str = OUT_DIR
) -> pd.DataFrame:
    ssp, tier, window = "ssp245", "median", (2036, 2040)
    rows = []

    for ind in fitted:
        row = {"indicator": ind, "label": _label(ind), "only_deficits": ONLY_DEFICITS}

        # ---- historical -------------------------------------------------
        hist_path = f"{out_dir}historical_burden_{ind}_{WBGT_VAR}.csv"
        if os.path.exists(hist_path):
            hist = pd.read_csv(hist_path, parse_dates=["date"])
            b, a, pct = _agg_deficit(hist)
            _, lo, hi = _jackknife_deficit_ci(hist)
            row["hist_mu_b_sum"]     = b
            row["hist_mu_a_sum"]     = a
            row["hist_appts_missed"] = max(0.0, b - a)
            row["hist_deficit_pct"]  = pct
            row["hist_ci_lo"]        = lo
            row["hist_ci_hi"]        = hi

            hist = hist.copy()
            hist["row_pct"] = np.where(
                hist["mu_b"] > 0,
                100.0 * (hist["mu_b"] - hist["mu_a"]) / hist["mu_b"],
                np.nan,
            )
            loss = hist[hist["mu_b"] > hist["mu_a"]]
            if not loss.empty and loss["row_pct"].notna().any():
                pk = loss["row_pct"].idxmax()
                row["peak_row_pct"]      = hist.loc[pk, "row_pct"]
                row["peak_row_date"]     = hist.loc[pk, "date"]
                row["peak_row_district"] = hist.loc[pk, CLUSTER_COL]
                row["peak_row_facility"] = hist.loc[pk, "facility"]
            else:
                row["peak_row_pct"] = np.nan
                row["peak_row_date"] = row["peak_row_district"] = row["peak_row_facility"] = None

            dm = (hist.groupby([CLUSTER_COL, "date"])
                       .agg(mu_a=("mu_a", "sum"), mu_b=("mu_b", "sum"))
                       .reset_index())
            dm_loss = dm[dm["mu_b"] > dm["mu_a"]].copy()
            if not dm_loss.empty:
                dm_loss["pct"] = 100.0 * (dm_loss["mu_b"] - dm_loss["mu_a"]) / dm_loss["mu_b"]
                pk = dm_loss["pct"].idxmax()
                row["peak_district_pct"]  = dm_loss.loc[pk, "pct"]
                row["peak_district_date"] = dm_loss.loc[pk, "date"]
                row["peak_district_name"] = dm_loss.loc[pk, CLUSTER_COL]
            else:
                row["peak_district_pct"] = np.nan
                row["peak_district_date"] = row["peak_district_name"] = None
        else:
            for k in ["hist_mu_b_sum", "hist_mu_a_sum", "hist_appts_missed",
                      "hist_deficit_pct", "hist_ci_lo", "hist_ci_hi",
                      "peak_row_pct", "peak_district_pct"]:
                row[k] = np.nan

        # ---- future (SSP245 / median, window) --------------------------
        # For the CI we need the facility-level projection file (the
        # annual file is pre-pooled). It's written WITHOUT SUFFIX, so
        # apply the mask on read.
        fac_path = f"{out_dir}projection_facility_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv"
        if os.path.exists(fac_path):
            fut = pd.read_csv(fac_path)
            fut = fut[fut["year"].between(*window)]
            if not fut.empty:
                _, _, pct_f = _agg_deficit(fut)
                _, lo_f, hi_f = _jackknife_deficit_ci(fut)
                # Missed appts: use the point estimate of net loss over window.
                sub = fut[_row_mask(fut)]
                row["future_deficit_pct"]  = pct_f
                row["future_ci_lo"]        = lo_f
                row["future_ci_hi"]        = hi_f
                row["future_appts_missed"] = max(0.0, float(sub["mu_b"].sum() - sub["mu_a"].sum()))
            else:
                for k in ("future_deficit_pct", "future_ci_lo", "future_ci_hi"):
                    row[k] = np.nan
                row["future_appts_missed"] = 0.0
        else:
            # Fall back to the pooled annual file — no CI possible then.
            fut_path = f"{out_dir}projection_annual_{ind}_{ssp}_{tier}_{WBGT_VAR}{SUFFIX}.csv"
            if os.path.exists(fut_path):
                fut = pd.read_csv(fut_path)
                sel = fut[fut["year"].between(*window)]
                if not sel.empty and sel["mu_b"].sum() > 0:
                    b = float(sel["mu_b"].sum()); a = float(sel["mu_a"].sum())
                    row["future_deficit_pct"]  = 100.0 * (b - a) / b
                    row["future_appts_missed"] = max(0.0, b - a)
                else:
                    row["future_deficit_pct"] = np.nan
                    row["future_appts_missed"] = 0.0
            else:
                row["future_deficit_pct"] = np.nan
                row["future_appts_missed"] = 0.0
            row["future_ci_lo"] = np.nan
            row["future_ci_hi"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = f"{out_dir}summary_stats_by_indicator_{WBGT_VAR}{SUFFIX}.csv"
    df.to_csv(out_path, index=False)
    return df


def calculate_summary_statistics(fitted: list[str], out_dir: str = OUT_DIR) -> dict:
    """Overall rollup with a CI on the volume-weighted overall deficit,
    computed by pooling facility-month rows across all indicators and
    jackknifing over (indicator, facility) pairs — so the CI reflects
    facility-level variability the same way per-indicator CIs do."""
    by_ind = calculate_summary_statistics_by_indicator(fitted, out_dir)
    if by_ind.empty:
        return {}

    # ---- pooled overall CI --------------------------------------------
    pool = []
    for ind in by_ind["indicator"]:
        p = f"{out_dir}historical_burden_{ind}_{WBGT_VAR}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p, usecols=["facility", "mu_a", "mu_b"])
            d["indicator"] = ind
            # Namespace facility per-indicator so the jackknife drops
            # (indicator, facility) rather than every indicator's copy
            # of that facility name simultaneously.
            d["facility"] = d["indicator"] + "::" + d["facility"].astype(str)
            pool.append(d)
    if pool:
        pool_df = pd.concat(pool, ignore_index=True)
        overall_pct, overall_lo, overall_hi = _jackknife_deficit_ci(pool_df)
    else:
        overall_pct = overall_lo = overall_hi = np.nan

    stats = {
        "only_deficits": ONLY_DEFICITS,
        "n_indicators": len(by_ind),
        "mean_deficit_pct_across_indicators": by_ind["hist_deficit_pct"].mean(),
        "overall_deficit_pct_volume_weighted": overall_pct,
        "overall_ci_lo": overall_lo,
        "overall_ci_hi": overall_hi,
        "total_appts_missed": by_ind["hist_appts_missed"].sum(),
        "peak_district_month_pct":  by_ind["peak_district_pct"].max(),
        "peak_district_month_name": (
            by_ind.loc[by_ind["peak_district_pct"].idxmax(), "peak_district_name"]
            if by_ind["peak_district_pct"].notna().any() else None
        ),
        "future_mean_deficit_pct_across_indicators": by_ind["future_deficit_pct"].mean(),
        "future_total_appts_missed": by_ind["future_appts_missed"].sum(),
        "future_max_indicator": (
            by_ind.loc[by_ind["future_deficit_pct"].idxmax(), "indicator"]
            if by_ind["future_deficit_pct"].notna().any() else None
        ),
        "future_max_indicator_pct": by_ind["future_deficit_pct"].max(),
    }
    return stats


def print_summary_statistics(stats: dict, by_ind: pd.DataFrame):
    view = "LOSS-ONLY (missed appointments)" if stats.get("only_deficits") else "NET (gains and losses)"
    print("\n" + "=" * 88)
    print(f"SUMMARY — view: {view}")
    print("=" * 88)

    print(f"\nHistorical, across {stats['n_indicators']} indicators:")
    print(f"  Mean deficit % (equal-weighted):    {stats['mean_deficit_pct_across_indicators']:+.2f}%")
    print(f"  Overall deficit % (vol-weighted):   {stats['overall_deficit_pct_volume_weighted']:+.2f}% "
          f"(95% CI {stats['overall_ci_lo']:+.2f}..{stats['overall_ci_hi']:+.2f})")
    print(f"  Total appointments missed:          {stats['total_appts_missed']:,.0f}")
    print(f"  Peak district-month:                {stats['peak_district_month_pct']:.2f}%  in  {stats['peak_district_month_name']}")

    print(f"\nFuture (SSP245 / median, 2036–2040):")
    print(f"  Mean deficit % across indicators:   {stats['future_mean_deficit_pct_across_indicators']:+.2f}%")
    print(f"  Total appointments missed:          {stats['future_total_appts_missed']:,.0f}")
    print(f"  Worst indicator:                    {stats['future_max_indicator']} ({stats['future_max_indicator_pct']:+.2f}%)")

    def _fmt_ci(lo, hi):
        if pd.isna(lo) or pd.isna(hi):
            return "  (n/a)         "
        return f"[{lo:+6.2f},{hi:+6.2f}]"

    print(f"\nBy indicator:")
    print("-" * 108)
    print(f"{'Indicator':<28} {'Hist %':>8} {'Hist 95% CI':>18} "
          f"{'Missed':>12} {'Peak D-M %':>11} {'Future %':>9} {'Future 95% CI':>18}")
    print("-" * 108)
    for _, r in by_ind.iterrows():
        print(f"{r['label']:<28} "
              f"{r['hist_deficit_pct']:>+7.2f}% "
              f"{_fmt_ci(r.get('hist_ci_lo'), r.get('hist_ci_hi')):>18} "
              f"{r['hist_appts_missed']:>12,.0f} "
              f"{r['peak_district_pct']:>+10.2f}% "
              f"{r['future_deficit_pct']:>+8.2f}% "
              f"{_fmt_ci(r.get('future_ci_lo'), r.get('future_ci_hi')):>18}")
    print("-" * 108)

# =====================================================================
# MAIN — run everything
# =====================================================================
def load_results_df(out_dir: str = OUT_DIR) -> pd.DataFrame:
    p = _require(f"{out_dir}summary_all_indicators_{WBGT_VAR}{SUFFIX}.csv")
    return pd.read_csv(p)


if __name__ == "__main__":
    print("=" * 60)
    print("Plotting from CSVs in", OUT_DIR)
    print("=" * 60)

    results_df = load_results_df()
    fitted = list(results_df["indicator"])

    print("\n[1] per-indicator exposure-response curves")
    for ind in fitted:
        try:
            plot_exposure_response_curve(ind)
        except FileNotFoundError as e:
            print(f"  skip {ind}: {e}")

    print("\n[2] main forest plot")
    print("  ->", plot_main_forest(results_df))

    print("\n[4] IRR forest plot")
    print("  ->", plot_irr_forest(results_df))

    print("\n[5] exposure-response panel")
    print("  ->", plot_exposure_response_panel(fitted))

    print("\n[6] monthly deficit panel")
    print("  ->", plot_monthly_deficit_panel(fitted))

    print("\n[7] timeseries burden panel")
    print("  ->", plot_timeseries_panel(fitted))

    print("\n[8] district choropleth maps")
    for p in plot_district_maps(fitted):
        print("  ->", p)

    print("\n[9] projection heatmaps")
    for p in plot_projection_heatmaps(fitted):
        print("  ->", p)

    print("\n[10] TLO disruption curves")
    print("  ->", plot_tlo_curves(results_df))

    print("\n[11] district × indicator heatmap")
    print("  ->", plot_district_indicator_heatmap())

    print("\\n[12] projection forest")
    print("  ->", plot_projection_forest(fitted))

    print("\\n[13] projection annual trajectory panel")
    print("  ->", plot_projection_annual_panel(fitted))

    print("\\n[14] seasonal amplification (SSP245 / median)")
    print("  ->", plot_seasonal_amplification(fitted))

    print("\\n[15] ")
    print("  ->", calculate_summary_statistics(fitted))

    print("\\n[16] ")
    print("  ->", calculate_summary_statistics_by_indicator(fitted))
    print("\nDone.")


