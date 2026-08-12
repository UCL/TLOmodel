"""
plot_all_figures_poisson.py

Plotter-only adaptation of plot_all_figures.py for the Poisson predict-twice
pipeline (model_of_wbgt_dhis2_poisson_predicttwice.py). Reads the CSVs the
Poisson script writes; produces the four figures its outputs can support:

  [1] main forest plot (aggregate deficit + CI + BH-FDR flag)
  [2] hot-month forest plot (hot deficit + hot CI)
  [3] monthly deficit panel across facilities (jackknife CI)
  [4] observed vs counterfactual timeseries panel

Not produced here (would require additions to the model script):
  - exposure-response curves / panel  (needs per-indicator curve CSV export)
  - IRR forest                        (needs vcov extraction from fepois)
  - district choropleth + heatmap     (needs district CI roll-up + shapefile)
  - projection heatmaps               (Poisson script has no projection block)
  - TLO disruption curves             (needs tlo_wbgt_lookup export)

Filename/column differences vs the NB plotter:
  - summary CSV        deficit_summary_predicttwice_{V}.csv (not two_model_deficit_results_NB_{V}.csv)
  - predictions CSV    predicttwice_predictions_{ind}_{V}.csv (not historical_burden_{ind}_{V}.csv)
  - column names       mu_obs / mu_ref (not mu_a / mu_b);  hot_ci_lo/hi (not hot_deficit_ci_lo/hi)
  - reference_wbgt is NOT written to the summary CSV — the Poisson script uses
    REFERENCE_WBGT_PERCENTILE (default 90) on each indicator's own WBGT
    distribution. We recompute the hot threshold from the per-indicator
    predictions CSV for labelling purposes only.
  - sig_bh column      absent in the summary; recomputed here from p_boot
                       (BH-FDR is applied inside the Poisson script but only
                        the q-value + boolean go to disk — we defensively
                        re-derive so a stale summary can't lie to us).

Sign convention: matches the Poisson script's _pct — positive `deficit_pct`
= services lost to heat. No sign flip anywhere in this file.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================================
# CONFIG — keep in sync with model_of_wbgt_dhis2_poisson_predicttwice.py
# =====================================================================
WBGT_VAR = "wbgt_day"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs_poisson/Test"

# Percentile used inside the Poisson script to define "hot month". Kept in
# sync so hot-threshold labels are accurate; not required for correctness of
# the hot-forest plot itself (that CI is already computed at fit time).
REFERENCE_WBGT_PERCENTILE = 90
FDR_ALPHA = 0.05

CLUSTER_COL = "Dist"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

INDICATOR_LABELS: dict[str, str] = {
    "fp_total_clients":           "FP Total Clients",
    "opd_attendance":             "OPD Attendance",
    "ipd_total_admissions":       "IPD Total Admissions",
    "vmmc_first_visits":          "VMMC First Visits",
    "pnc_mother_checked_48h":     "PNC Mother <48h",
    "anc_new_attendees":          "ANC New Attendees",
    "anc_first_trimester_starts": "ANC 1st Trimester Starts",
    "anc4_coverage":              "ANC4 Coverage",
    "bcg_under1":                 "BCG Under-1",
    "penta3_under1":              "Penta3 Under-1",
    "measles1_under1":            "Measles 1st Dose Under-1",
    "measles_under1":             "Measles Under-1",
    "fully_immunised_under1":     "Fully Immunised Under-1",
    "pnc_within_2wks":            "PNC Within 2 Weeks",
    "pnc_first_visit_2wks":       "PNC First Visit <2 Weeks",
    "live_births_total":          "Live Births Total",
    "skilled_deliveries":         "Skilled Deliveries",
}


# =====================================================================
# HELPERS
# =====================================================================
def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg. Mirrors the Poisson script's bh_fdr exactly so
    the plot's significance flags agree with what the script would flag,
    without relying on the summary CSV to carry the boolean.

    Returns (q_values, reject_mask).
    """
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


def _monthly_jackknife_ci(mu_obs, mu_ref, facility_ids):
    """Leave-one-facility-out CI on the Poisson deficit
        pct = 100 * (sum_ref - sum_obs) / sum_ref
    matching the sign convention in the Poisson script's `_pct`. Positive
    = services lost to heat.
    """
    sum_obs = float(mu_obs.sum())
    sum_ref = float(mu_ref.sum())
    if sum_ref <= 0:
        return np.nan, np.nan, np.nan
    deficit_pct = 100.0 * (sum_ref - sum_obs) / sum_ref
    facs = np.unique(facility_ids)
    n    = len(facs)
    if n < 3:
        return deficit_pct, np.nan, np.nan
    jack = []
    for fac in facs:
        keep = facility_ids != fac
        so_j = float(mu_obs[keep].sum())
        sr_j = float(mu_ref[keep].sum())
        if sr_j <= 0:
            continue
        jack.append(100.0 * (sr_j - so_j) / sr_j)
    if len(jack) < 3:
        return deficit_pct, np.nan, np.nan
    jack = np.asarray(jack)
    jack_se = np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))
    return deficit_pct, deficit_pct - 1.96 * jack_se, deficit_pct + 1.96 * jack_se


def _require(path: str) -> str:
    """Loud failure if a required CSV is missing. The Poisson script must
    have run to completion first."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input missing: {path}\n"
            f"Run model_of_wbgt_dhis2_poisson_predicttwice.py first."
        )
    return path


def _label(ind: str) -> str:
    return INDICATOR_LABELS.get(ind, ind)


def _predictions_path(ind: str, out_dir: str = OUT_DIR) -> str:
    return f"{out_dir}predicttwice_predictions_{ind}_{WBGT_VAR}.csv"


# =====================================================================
# 1. MAIN FOREST PLOT (aggregate deficit)
# =====================================================================
def plot_main_forest(results_df: pd.DataFrame, out_dir: str = OUT_DIR) -> str:
    plot_df = results_df.sort_values("deficit_pct").reset_index(drop=True)
    y_pos   = np.arange(len(plot_df))
    has_ci  = plot_df["ci_lo"].notna().any()
    colors = [
        "#823038" if bool(r.get("sig_bh", False))
        else ("#888888" if has_ci else "#4a7298")
        for _, r in plot_df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(7, max(4, len(plot_df) * 0.55 + 1.5)))
    for i, row in plot_df.iterrows():
        if pd.notna(row["ci_lo"]):
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                    color=colors[i], lw=1.4, zorder=1)
    ax.scatter(plot_df["deficit_pct"], y_pos, color=colors, s=55, zorder=2)
    ax.axvline(0, color="black", ls="--", lw=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [_label(i) for i in plot_df["indicator"]], fontsize=9)
    ax.set_xlabel("% appointments lost to heat (positive = loss)", fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.5)
    if has_ci:
        ax.legend(handles=[
            mpatches.Patch(color="#823038", label=f"BH-FDR q≤{FDR_ALPHA}"),
            mpatches.Patch(color="#888888", label="not significant"),
        ], loc="lower right", fontsize=9, frameon=False)
    plt.tight_layout()
    out_path = f"{out_dir}forest_plot_deficit_poisson_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 2. HOT-MONTH FOREST PLOT
# =====================================================================
def plot_hot_forest(results_df: pd.DataFrame, out_dir: str = OUT_DIR) -> str:
    """Hot-month deficit + CI. Unlike the NB script, the Poisson script
    doesn't write a per-indicator p-value for the hot deficit — the hot CI
    is the bootstrap percentile interval and there's no p_hot column. So
    the significance flag here comes from whether the hot CI excludes 0,
    not from BH-FDR on a hot p-value. Different flag → different colour
    rule than the NB version, deliberately."""
    ph = (results_df.dropna(subset=["hot_deficit_pct"])
          .sort_values("hot_deficit_pct").reset_index(drop=True))
    if ph.empty:
        print("  no hot deficit rows — skipping hot forest")
        return ""
    y_ph = np.arange(len(ph))

    def _hot_sig(row):
        lo, hi = row.get("hot_ci_lo"), row.get("hot_ci_hi")
        if pd.isna(lo) or pd.isna(hi):
            return False
        return (lo > 0) or (hi < 0)

    hot_sig    = ph.apply(_hot_sig, axis=1).values
    hot_colors = ["#823038" if bool(s) else "#888888" for s in hot_sig]

    # Try to label the panel with the hot threshold. Reference is not on the
    # summary CSV (Poisson script uses a percentile per indicator, not a
    # single value); if all indicators used the same threshold at fit time
    # we can recover it from any predictions CSV. If they differ, we say so.
    thresholds = []
    for ind in ph["indicator"]:
        pp = _predictions_path(ind)
        if os.path.exists(pp):
            w = pd.read_csv(pp)[WBGT_VAR].values
            thresholds.append(np.percentile(w, REFERENCE_WBGT_PERCENTILE))
    if thresholds and np.allclose(thresholds, thresholds[0], atol=0.05):
        thr_label = f"WBGT > {thresholds[0]:.1f}°C"
    elif thresholds:
        thr_label = (f"WBGT > per-indicator P{REFERENCE_WBGT_PERCENTILE} "
                     f"(~{np.mean(thresholds):.1f}°C)")
    else:
        thr_label = f"WBGT > P{REFERENCE_WBGT_PERCENTILE}"

    fig, ax = plt.subplots(figsize=(7, max(4, len(ph) * 0.55 + 1.5)))
    for i, row in ph.iterrows():
        lo, hi, pt = (row.get("hot_ci_lo"),
                      row.get("hot_ci_hi"),
                      row["hot_deficit_pct"])
        if pd.notna(lo) and pd.notna(hi):
            ax.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]],
                        fmt="o", markersize=7, capsize=4, capthick=1.4,
                        elinewidth=1.4, color=hot_colors[i], zorder=2)
        else:
            ax.scatter(pt, i, color=hot_colors[i], s=55, zorder=2)
    ax.axvline(0, color="black", ls="--", lw=0.9)
    ax.set_yticks(y_ph)
    ax.set_yticklabels(
        [_label(i) for i in ph["indicator"]], fontsize=9)
    ax.set_xlabel(f"% appointments lost ({thr_label})", fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color="#823038", label="CI excludes 0"),
        mpatches.Patch(color="#888888", label="CI includes 0"),
    ], loc="lower right", fontsize=9, frameon=False)
    plt.tight_layout()
    out_path = f"{out_dir}forest_plot_hot_deficit_poisson_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 3. MONTHLY DEFICIT PANEL (jackknife CI)
# =====================================================================
def plot_monthly_deficit_panel(fitted: list[str],
                                out_dir: str = OUT_DIR) -> str:
    """One subplot per indicator, bars = deficit_pct by calendar month with
    jackknife CI across facilities. Reads predicttwice_predictions_{ind}_{V}.csv
    (columns: facility, date, mu_obs, mu_ref, …) and derives `month` from
    `date` — the Poisson predictions CSV doesn't carry a month column."""
    n_ind = len(fitted)
    if n_ind == 0:
        print("  no indicators — skipping monthly panel")
        return ""
    n_cols = 3
    n_rows = int(np.ceil(n_ind / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 3 * n_rows),
                              sharex=True)
    af = axes.flatten() if n_ind > 1 else [axes]

    for idx, ind in enumerate(fitted):
        ax = af[idx]
        csv_path = _predictions_path(ind, out_dir)
        if not os.path.exists(csv_path):
            ax.set_visible(False)
            continue
        bdf = pd.read_csv(csv_path, parse_dates=["date"])
        ratio = bdf["mu_obs"].sum() / bdf["mu_ref"].sum()
        if not (0.5 < ratio < 2.0):
            print(
                f"  WARNING [{ind}]: mu_obs/mu_ref ratio = {ratio:.3f} — "
                f"possible column mismatch or offset scaling issue"
            )
        bdf["month"] = bdf["date"].dt.month
        fac_ids = bdf["facility"].values
        months  = bdf["month"].values
        mu_obs  = bdf["mu_obs"].values
        mu_ref  = bdf["mu_ref"].values

        pcts, los, his = [], [], []
        for m in range(1, 13):
            mask = months == m
            if not mask.any():
                pcts.append(0.0); los.append(np.nan); his.append(np.nan)
                continue
            pt, lo, hi = _monthly_jackknife_ci(
                mu_obs[mask], mu_ref[mask], fac_ids[mask])
            pcts.append(pt); los.append(lo); his.append(hi)

        pcts_a = np.asarray(pcts, dtype=float)
        los_a  = np.asarray(los,  dtype=float)
        his_a  = np.asarray(his,  dtype=float)
        # Positive = loss under heat, red. Negative = gain under heat, blue.
        bar_c  = ["#823038" if p > 0 else "#2a78d6" for p in pcts_a]
        yerr = np.array([
            np.nan_to_num(pcts_a - los_a, nan=0.0),
            np.nan_to_num(his_a  - pcts_a, nan=0.0),
        ])
        ax.bar(range(12), pcts_a, color=bar_c, alpha=0.8, yerr=yerr,
               error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": "#333"})
        ax.set_xticks(range(12))
        ax.set_xticklabels(MONTH_NAMES, fontsize=6, rotation=45)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(_label(ind), fontsize=9, fontweight="bold")
        if idx % n_cols == 0:
            ax.set_ylabel("% deficit", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n_ind, len(af)):
        af[idx].set_visible(False)
    plt.tight_layout()
    out_path = f"{out_dir}deficit_by_month_poisson_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 4. TIMESERIES BURDEN PANEL
# =====================================================================
def plot_timeseries_panel(fitted: list[str], out_dir: str = OUT_DIR) -> str:
    """Observed vs counterfactual monthly totals across facilities, one
    subplot per indicator. The Poisson CSV doesn't carry `y_int` (only
    `y_obs`), so we plot y_obs as the observed line."""
    n_ind = len(fitted)
    if n_ind == 0:
        print("  no indicators — skipping timeseries panel")
        return ""
    nc = 3
    nr = int(np.ceil(n_ind / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5.5 * nc, 3.5 * nr), sharex=True)
    af = axes.flatten() if n_ind > 1 else [axes]

    for idx, ind in enumerate(fitted):
        ax  = af[idx]
        csv_path = _predictions_path(ind, out_dir)
        if not os.path.exists(csv_path):
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv_path, parse_dates=["date"])
        m  = (df.groupby("date")
              .agg(obs=("y_obs", "sum"),
                   mu_obs=("mu_obs", "sum"),
                   mu_ref=("mu_ref", "sum"))
              .sort_index())
        # Reference (no-heat) counterfactual — dashed line. Observed — solid.
        # Shade the gap where mu_obs < mu_ref (services lost to heat).
        ax.plot(m.index, m["mu_ref"], color="#2a78d6", lw=1.0, ls="--",
                alpha=0.8, label="Counterfactual (WBGT@ref)")
        ax.plot(m.index, m["obs"],    color="#333", lw=1.0, label="Observed")
        ax.fill_between(m.index, m["mu_obs"], m["mu_ref"],
                        where=m["mu_obs"] < m["mu_ref"],
                        color="#823038", alpha=0.25, label="Heat deficit")
        ax.set_title(_label(ind), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6)

    for idx in range(n_ind, len(af)):
        af[idx].set_visible(False)
    plt.tight_layout()
    out_path = f"{out_dir}timeseries_burden_poisson_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# MAIN
# =====================================================================
def load_results_df(out_dir: str = OUT_DIR) -> pd.DataFrame:
    """Load the Poisson summary CSV and defensively re-derive `sig_bh`.

    The Poisson script writes `q_boot` and `sig_bh` — but `sig_bh` there
    is `q_boot < FDR_ALPHA` (strict), while the plot's legend copy says
    "q ≤ α". Re-derive with `≤` to match the legend, and to guard against
    a stale summary being read alongside a plotter that expects the flag."""
    p = _require(f"{out_dir}deficit_summary_predicttwice_{WBGT_VAR}.csv")
    df = pd.read_csv(p)
    if "p_boot" in df.columns:
        q, rej = bh_fdr(df["p_boot"].values, alpha=FDR_ALPHA)
        df["q_bh"]   = q
        df["sig_bh"] = rej
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Plotting Poisson predict-twice CSVs from", OUT_DIR)
    print("=" * 60)

    results_df = load_results_df()
    fitted = list(results_df["indicator"])

    print("\n[1] main forest plot")
    print("  ->", plot_main_forest(results_df))

    print("\n[2] hot-month forest plot")
    print("  ->", plot_hot_forest(results_df))

    print("\n[3] monthly deficit panel")
    print("  ->", plot_monthly_deficit_panel(fitted))

    print("\n[4] timeseries burden panel")
    print("  ->", plot_timeseries_panel(fitted))

    print("\nDone.")
