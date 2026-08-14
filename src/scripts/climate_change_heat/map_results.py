

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
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/OLD_ANALYSIS/"
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
}


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
        f"{out_dir}exposure_response_curve_{indicator}_{WBGT_VAR}.csv")
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
# 2. MAIN FOREST PLOT (aggregate deficit)
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
    ax.set_yticklabels(plot_df["label"], fontsize=9)
    ax.set_xlabel("% change in appointments associated with WBGT", fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.5)
    if has_ci:
        ax.legend(handles=[
            mpatches.Patch(color="#823038", label=f"BH-FDR q≤{FDR_ALPHA}"),
            mpatches.Patch(color="#888888", label="not significant"),
        ], loc="lower right", fontsize=9, frameon=False)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"θ={r['alpha']:.1f}" for _, r in plot_df.iterrows()],
        fontsize=7, color="#666666")
    ax2.tick_params(axis="y", length=0)
    plt.tight_layout()
    out_path = f"{out_dir}forest_plot_two_model_deficit_NB_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 3. HOT-MONTH FOREST PLOT
# =====================================================================
def plot_hot_forest(results_df: pd.DataFrame, out_dir: str = OUT_DIR) -> str:
    # Re-compute the FDR flag locally so the CSV doesn't have to carry it
    if "sig_hot_bh" not in results_df.columns:
        _, rej_hot = bh_fdr(results_df["pval"].values,
                            alpha=FDR_ALPHA)
        results_df = results_df.copy()
        results_df["sig_hot_bh"] = rej_hot

    ph = (results_df.dropna(subset=["hot_deficit_pct"])
          .sort_values("hot_deficit_pct").reset_index(drop=True))
    y_ph       = np.arange(len(ph))
    hot_colors = ["#823038" if bool(s) else "#888888" for s in ph["sig_hot_bh"]]
    ref_wbgt   = float(results_df["reference_wbgt"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, max(4, len(ph) * 0.55 + 1.5)))
    for i, row in ph.iterrows():
        lo, hi, pt = (row["hot_ci_lo"],
                      row["hot_ci_hi"],
                      row["hot_deficit_pct"])
        if pd.notna(lo) and pd.notna(hi):
            ax.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]],
                        fmt="o", markersize=7, capsize=4, capthick=1.4,
                        elinewidth=1.4, color=hot_colors[i], zorder=2)
        else:
            ax.scatter(pt, i, color=hot_colors[i], s=55, zorder=2)
    ax.axvline(0, color="black", ls="--", lw=0.9)
    ax.set_yticks(y_ph)
    ax.set_yticklabels(ph["label"], fontsize=9)
    ax.set_xlabel(
        f"% change in appointments (WBGT > {ref_wbgt:.1f}°C)", fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color="#823038", label=f"BH-FDR q≤{FDR_ALPHA}"),
        mpatches.Patch(color="#888888", label="not significant"),
    ], loc="lower right", fontsize=9, frameon=False)
    plt.tight_layout()
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
    irr_df = irr_df.sort_values("irr").reset_index(drop=True)
    irr_low = float(results_df["reference_wbgt"].iloc[0])

    irr_colors = [
        "#823038" if (r["irr_hi"] < 1 or r["irr_lo"] > 1) else "#888888"
        for _, r in irr_df.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(7, max(4, len(irr_df) * 0.55 + 1.5)))
    for i, row in irr_df.iterrows():
        ax.errorbar(
            row["irr"], i,
            xerr=[[row["irr"] - row["irr_lo"]],
                  [row["irr_hi"] - row["irr"]]],
            fmt="o", markersize=7, capsize=8, capthick=1.4,
            elinewidth=1.4, color=irr_colors[i], zorder=2, ecolor="black")
    ax.axvline(1.0, color="black", ls="--", lw=0.9)
    ax.set_yticks(range(len(irr_df)))
    ax.set_yticklabels(irr_df["label"], fontsize=9)
    ax.set_xlabel(
        f"IRR: WBGT {IRR_HIGH:.0f}°C vs {irr_low:.0f}°C", fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.5)
    plt.tight_layout()
    out_path = (f"{out_dir}forest_plot_IRR_{irr_low:.0f}_{IRR_HIGH:.0f}_NB_"
                f"{WBGT_VAR}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 5. EXPOSURE-RESPONSE PANEL (all indicators)
# =====================================================================
def plot_exposure_response_panel(fitted: list[str],
                                  out_dir: str = OUT_DIR) -> str:
    curves_df = pd.read_csv(
        _require(f"{out_dir}exposure_response_curves_{WBGT_VAR}.csv"))
    inds = [i for i in fitted if i in curves_df["indicator"].unique()]
    if not inds:
        print("  no exposure-response rows — skipping panel")
        return ""
    n_ind  = len(inds)
    n_cols = 3
    n_rows = int(np.ceil(n_ind / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 3 * n_rows),
                              sharex=True)
    af = axes.flatten() if n_ind > 1 else [axes]

    for idx, ind in enumerate(inds):
        ax  = af[idx]
        sub = curves_df[curves_df["indicator"] == ind].sort_values("wbgt")
        ref_w = float(sub["wbgt_ref"].iloc[0])
        ax.fill_between(sub["wbgt"], sub["rr_lo"], sub["rr_hi"],
                        color="#4a7298", alpha=0.25, linewidth=0)
        ax.plot(sub["wbgt"], 1/sub["rr_vs_ref"], color="#2a4d70", lw=1.5)
        ax.axhline(1.0, color="black", lw=0.5, ls="--")
        ax.axvline(ref_w, color="grey", lw=0.5, ls=":")
        ax.set_title(_label(ind), fontsize=9, fontweight="bold")
        if idx % n_cols == 0:
            ax.set_ylabel("IRR", fontsize=8)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("WBGT (°C)", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n_ind, len(af)):
        af[idx].set_visible(False)

    plt.tight_layout()
    out_path = f"{out_dir}exposure_response_panel_{WBGT_VAR}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# 6. MONTHLY DEFICIT PANEL (jackknife CI)
# =====================================================================
def plot_monthly_deficit_panel(fitted: list[str],
                                out_dir: str = OUT_DIR) -> str:
    n_ind  = len(fitted)
    n_cols = 3
    n_rows = int(np.ceil(n_ind / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 3 * n_rows),
                              sharex=True)
    af = axes.flatten() if n_ind > 1 else [axes]

    for idx, ind in enumerate(fitted):
        ax = af[idx]
        csv_path = f"{out_dir}historical_burden_{ind}_{WBGT_VAR}.csv"
        if not os.path.exists(csv_path):
            ax.set_visible(False)
            continue
        bdf = pd.read_csv(csv_path)
        fac_ids = bdf["facility"].values
        months  = bdf["month"].values
        mu_a    = bdf["mu_a"].values
        mu_b    = bdf["mu_b"].values

        pcts, los, his = [], [], []
        for m in range(1, 13):
            mask = months == m
            if not mask.any():
                pcts.append(0); los.append(np.nan); his.append(np.nan)
                continue
            pt, lo, hi = _monthly_jackknife_ci(
                mu_a[mask], mu_b[mask], fac_ids[mask])
            pcts.append(pt); los.append(lo); his.append(hi)

        pcts_a = np.asarray(pcts, dtype=float)
        los_a  = np.asarray(los,  dtype=float)
        his_a  = np.asarray(his,  dtype=float)
        bar_c  = ["#823038" if p < 0 else "#2a78d6" for p in pcts_a]
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

    # Load per-indicator district CSVs.
    frames = []
    for ind in fitted:
        p = f"{out_dir}district_burden_{ind}_{WBGT_VAR}.csv"
        if not os.path.exists(p):
            print(f"  {ind}: district csv missing — skipping in maps")
            continue
        d = pd.read_csv(p)
        d["indicator"] = ind
        frames.append(d)
    if not frames:
        return []
    dist_all = pd.concat(frames, ignore_index=True)
    dist_all[CLUSTER_COL] = (
        dist_all[CLUSTER_COL].astype(str).str.strip().str.title())

    out_paths = []

    # ---- one map per indicator ----
    for ind in fitted:
        sub = dist_all[dist_all["indicator"] == ind].copy()
        if sub.empty:
            continue
        merged = shp.merge(
            sub[[CLUSTER_COL, "deficit_pct"]],
            left_on=DISTRICT_NAME_COL,
            right_on=CLUSTER_COL, how="left")
        n_matched = merged["deficit_pct"].notna().sum()
        print(f"  {ind}: {n_matched}/{len(merged)} districts matched")

        vmax = max(merged["deficit_pct"].abs().quantile(0.95), 1.0)
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        merged.plot(
            column="deficit_pct", ax=ax,
            cmap="RdBu", vmin=-vmax, vmax=vmax,
            edgecolor="white", linewidth=0.4,
            missing_kwds={"color": "#cccccc", "label": "No data"},
            legend=True,
            legend_kwds={"label": "% deficit (Model A vs B)",
                          "orientation": "horizontal",
                          "shrink": 0.7, "pad": 0.02})
        ax.set_axis_off()
        plt.tight_layout()
        out_path = f"{out_dir}map_district_deficit_{ind}_{WBGT_VAR}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        out_paths.append(out_path)

    # ---- summary panel ----
    global_vmax = max(dist_all["deficit_pct"].abs().quantile(0.95), 1.0)
    n_ind = len(fitted)
    n_cols = 3
    n_rows = int(np.ceil(n_ind / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows))
    af = axes.flatten() if n_ind > 1 else [axes]

    for idx, ind in enumerate(fitted):
        ax  = af[idx]
        sub = dist_all[dist_all["indicator"] == ind].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        merged = shp.merge(
            sub[[CLUSTER_COL, "deficit_pct"]],
            left_on=DISTRICT_NAME_COL,
            right_on=CLUSTER_COL, how="left")
        merged.plot(
            column="deficit_pct", ax=ax,
            cmap="RdBu", vmin=-global_vmax, vmax=global_vmax,
            edgecolor="white", linewidth=0.3,
            missing_kwds={"color": "#cccccc"},
            legend=False)
        ax.set_axis_off()
        ax.set_title(_label(ind), fontsize=8, fontweight="bold")

    for idx in range(n_ind, len(af)):
        af[idx].set_visible(False)

    sm = plt.cm.ScalarMappable(
        cmap="RdBu",
        norm=plt.Normalize(vmin=-global_vmax, vmax=global_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=af, orientation="horizontal",
                        fraction=0.02, pad=0.02, shrink=0.6)
    cbar.set_label("% deficit (Model A vs B)", fontsize=9)
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
    csv_path = f"{out_dir}projection_summary_{WBGT_VAR}.csv"
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
                grid.loc[p["ssp"], p["tier"]] = p["delta_deficit"]
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
    tlo_path = f"{out_dir}tlo_wbgt_lookup.csv"  # writer path (no suffix)
    if not os.path.exists(tlo_path):
        # tolerate the alternative suffix in case you fix the writer later
        alt = f"{out_dir}tlo_wbgt_lookup_{WBGT_VAR}.csv"
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

    SIGN CONVENTION — READ THIS BEFORE EDITING:
    The CI file is written by district_deficit_analytical(), which computes
    delta = 100 * (sum_A − sum_B) / sum_B, so positive `deficit_pct` in
    that CSV means services GAINED under heat. The sign fix documented in
    the model script propagated to district_burden_{ind}_{WBGT_VAR}.csv
    (which uses (mu_B − mu_A)/mu_B) but NOT to the CI file. So we negate
    on read to get "services lost" (positive = loss), matching every
    other figure in the paper. If district_deficit_analytical is ever
    updated to use (sum_B − sum_A), remove this negation.
    """
    rows = []
    for ind in indicator_order:
        # Writer path uses WBGT_VAR suffix; the standalone heatmap script
        # was reading without the suffix and quietly getting nothing.
        path = f"{out_dir}district_burden_ci_{ind}_{WBGT_VAR}.csv"
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


# =====================================================================
# MAIN — run everything
# =====================================================================
def load_results_df(out_dir: str = OUT_DIR) -> pd.DataFrame:
    p = _require(f"{out_dir}summary_all_indicators_{WBGT_VAR}.csv")
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

    print("\n[3] hot-month forest plot")
    print("  ->", plot_hot_forest(results_df))

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

    print("\nDone.")
