"""
loop_all_indicators_forest.py  (pyfixest / Poisson-FE version)

For every count indicator:
  1. Fit a Poisson model (facility + month FE, WBGT natural cubic spline).
  2. Report the cumulative IRR and a cluster-robust Wald block test.
  3. Run robustness checks (harmonic season, different df, different contrast,
     sustained lags) and collect them into one panel.
  4. Forest plot of the primary results; robustness CSV alongside.
"""

import os, warnings
import numpy as np
import pandas as pd
import patsy
import scipy.stats as stats
import pyfixest as pf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# 1. SETTINGS — PRIMARY SPECIFICATION
# ===========================================================================
# These define the MAIN model. Robustness checks vary one thing at a time
# away from these, and are defined in section 8.

COUNT_INDICATORS = [
    "fp_total_clients", "opd_attendance",
    "ipd_total_admissions",
    "vmmc_first_visits",
    "pnc_mother_checked_48h",
    #"anc_new_attendees",
    "anc_first_trimester_starts",
    "bcg_under1", "penta3_under1", "measles1_under1",
    "fully_immunised_under1", "pnc_within_2wks", "pnc_first_visit_2wks",
    "live_births_total",
    "skilled_deliveries",
]

INDICATOR_LABELS = {
    "fp_total_clients":           "FP Total Clients",
    "opd_attendance":             "OPD Attendance",
    "vmmc_first_visits":          "VMMC First Visits",
    "pnc_mother_checked_48h":     "PNC Mother <48h",
    #"anc_first_trimester_starts": "ANC 1st Trimester Starts",
    "bcg_under1":                 "BCG Under-1",
    "penta3_under1":              "Penta3 Under-1",
    "measles1_under1":            "Measles 1st Dose Under-1",
    "fully_immunised_under1":     "Fully Immunised Under-1",
    "pnc_within_2wks":            "PNC Within 2 Weeks",
    "pnc_first_visit_2wks":       "PNC First Visit <2 Weeks",
    "skilled_deliveries":         "Skilled Deliveries",
}

WBGT_VAR   = "wbgt5x_day"
SPLINE_DF  = 3
LAG_MONTHS = [1, 2, 3, 9]

# Primary specification choices.
SEASON_CONTROL = "month_fe"      # "month_fe" or "harmonic"
N_HARMONICS    = 2               # only used when "harmonic"
CONTRAST_MODE  = "percentile"         # "fixed" or "percentile"
CONTRAST_PCTL  = (50, 90)
CONTRAST_FIXED = (25.0, 32.0)
INCLUDE_LAGS   = False           # contemporaneous only

# df selection.
RUN_DF_SELECTION = True
DF_CANDIDATES    = (3, 4, 5, 6)

MIN_YEAR, MAX_YEAR, MIN_OBS = 2015, 2025, 10

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# 2. HELPERS — SPLINE BASIS, FORMULA, HARMONICS
# ===========================================================================

def spline_design_info(data, df_spline):
    return patsy.dmatrix(
        f"cr({WBGT_VAR}, df={df_spline})", data, return_type="dataframe"
    ).design_info


def add_harmonic_columns(data, n_harmonics):
    names = []
    for k in range(1, n_harmonics + 1):
        s, c = f"sin{k}", f"cos{k}"
        data[s] = np.sin(2 * np.pi * k * data["month"] / 12)
        data[c] = np.cos(2 * np.pi * k * data["month"] / 12)
        names += [s, c]
    return names


def build_formula(df_spline, lag_columns, season="month_fe", harm_cols=None):
    spline_term = f"cr({WBGT_VAR}, df={df_spline})"
    rhs = [spline_term] + lag_columns + ["year_c"]
    if season == "harmonic" and harm_cols:
        rhs += harm_cols
        fe = "facility"
    else:
        fe = "facility + month"
    return f"y_int ~ {' + '.join(rhs)} | {fe}"


# ===========================================================================
# 3. CUMULATIVE IRR (accepts settings as arguments, not globals)
# ===========================================================================

def get_contrast_bounds(data, mode, pctl, fixed):
    if mode == "fixed":
        return fixed
    return (float(np.nanpercentile(data[WBGT_VAR], pctl[0])),
            float(np.nanpercentile(data[WBGT_VAR], pctl[1])))


def cumulative_irr(model, di, wbgt_mean, cool, hot, include_lags):
    names = list(model.coef().index)
    beta  = model.coef().values
    vcov  = model._vcov

    basis = patsy.build_design_matrices(
        [di], pd.DataFrame({WBGT_VAR: [cool, hot]}), return_type="dataframe"
    )[0]

    contrast = np.zeros(len(names))
    for j, name in enumerate(names):
        if name in basis.columns:
            contrast[j] = basis[name].iloc[1] - basis[name].iloc[0]
        elif "_lag" in name:
            contrast[j] = (hot - cool) if include_lags else 0.0

    log_irr = float(contrast @ beta)
    se      = float(np.sqrt(contrast @ vcov @ contrast))
    z       = log_irr / se
    return {"irr": np.exp(log_irr),
            "irr_lo": np.exp(log_irr - 1.96 * se),
            "irr_hi": np.exp(log_irr + 1.96 * se),
            "pval": float(2 * stats.norm.sf(abs(z)))}


# ===========================================================================
# 4. WALD BLOCK TEST
# ===========================================================================

def weather_block_pvalue(model):
    names = list(model.coef().index)
    beta  = model.coef().values
    vcov  = model._vcov
    w = [i for i, n in enumerate(names) if WBGT_VAR in n]
    if not w:
        return np.nan
    R = np.zeros((len(w), len(names)))
    for r, i in enumerate(w):
        R[r, i] = 1.0
    Rb, RVR = R @ beta, R @ vcov @ R.T
    return float(stats.chi2.sf(float(Rb @ np.linalg.solve(RVR, Rb)), len(w)))


# ===========================================================================
# 5. df SELECTION
# ===========================================================================

def compare_spline_df(data, lag_columns, wbgt_mean, cool, hot,
                      season, harm_cols, include_lags):
    rows = []
    for df_s in DF_CANDIDATES:
        formula = build_formula(df_s, lag_columns, season, harm_cols)
        try:
            m = pf.fepois(formula, data=data, vcov={"CRV1": "facility"})
        except Exception as e:
            rows.append({"df": df_s, "loglik": np.nan, "k": np.nan,
                         "AIC": np.nan, "BIC": np.nan, "irr": np.nan,
                         "note": str(e)[:30]})
            continue
        ll, k, n = m._loglik, len(m.coef()), m._N
        di = spline_design_info(data, df_s)
        irr = cumulative_irr(m, di, wbgt_mean, cool, hot, include_lags)["irr"]
        rows.append({"df": df_s, "loglik": ll, "k": k,
                     "AIC": -2*ll + 2*k, "BIC": -2*ll + k*np.log(n),
                     "irr": irr, "note": ""})
    return pd.DataFrame(rows)


# ===========================================================================
# 6. PREPARE DATA (shared across primary + robustness)
# ===========================================================================

def prepare_data(indicator):
    path = f"{DATA_DIR}/All_predictors_processed/regression_panel_{indicator}.csv"
    if not os.path.exists(path):
        return None, None, None, None

    data = pd.read_csv(path, parse_dates=["date"]).rename(columns={indicator: "y"})
    if WBGT_VAR not in data.columns:
        return None, None, None, None

    data.loc[data["date"].between("2020-04-01", "2021-12-01"), "y"] = np.nan
    data.loc[data["date"].between("2023-04-01", "2024-06-01")
             & (data["facility"] == "Phalombe Health Centre"), "y"] = 0
    data.loc[data["date"].between("2023-03-01", "2024-03-01")
             & (data["facility"] == "Thumbwe Health Centre"), "y"] = 0

    data["year"]  = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data = data[data["year"].between(MIN_YEAR, MAX_YEAR - 1)]

    wbgt_mean = data[WBGT_VAR].mean()
    data["year_c"] = data["year"] - data["year"].mean()

    data = data.sort_values(["facility", "date"])
    lag_columns = []
    for lag in LAG_MONTHS:
        lc = f"{WBGT_VAR}_lag{lag}_c"
        data[lc] = data.groupby("facility")[WBGT_VAR].shift(lag) - wbgt_mean
        lag_columns.append(lc)

    # Always build harmonic columns so they're available for robustness.
    harm_cols = add_harmonic_columns(data, N_HARMONICS)

    needed = ["y", "facility", "month", "year_c", WBGT_VAR] + lag_columns
    data = data.dropna(subset=needed).copy()
    data["y_int"] = data["y"].round().clip(lower=0).astype(int)
    data["facility"] = data["facility"].astype(str)
    data["month"] = data["month"].astype(int)

    good = data.groupby("facility").size()
    data = data[data["facility"].isin(good[good >= MIN_OBS].index)].copy()

    if data["facility"].nunique() < 2:
        return None, None, None, None

    return data, lag_columns, harm_cols, wbgt_mean


# ===========================================================================
# 7. FIT + EVALUATE (one spec, one indicator)
# ===========================================================================

def fit_and_evaluate(data, lag_columns, harm_cols, wbgt_mean,
                     df_spline, season, contrast_mode, contrast_pctl,
                     contrast_fixed, include_lags):
    """Fit one model and return its IRR. Returns None on failure."""
    formula = build_formula(df_spline, lag_columns, season, harm_cols)
    try:
        model = pf.fepois(formula, data=data, vcov={"CRV1": "facility"})
    except Exception:
        return None

    cool, hot = get_contrast_bounds(data, contrast_mode, contrast_pctl, contrast_fixed)
    di = spline_design_info(data, df_spline)
    effect = cumulative_irr(model, di, wbgt_mean, cool, hot, include_lags)
    block_p = weather_block_pvalue(model)

    return {"irr": effect["irr"], "irr_lo": effect["irr_lo"],
            "irr_hi": effect["irr_hi"], "pval_contrast": effect["pval"],
            "pval_block": block_p, "n_obs": int(model._N),
            "cool": cool, "hot": hot, "model": model}


# ===========================================================================
# 8. MAIN LOOP — PRIMARY + ROBUSTNESS
# ===========================================================================

print("=" * 60)
print(f"Primary spec: season={SEASON_CONTROL}, df={SPLINE_DF}, "
      f"contrast={CONTRAST_MODE}, lags={'on' if INCLUDE_LAGS else 'off'}")
print("=" * 60)

primary_results = []
robustness_rows = []
df_tables = []

for indicator in COUNT_INDICATORS:
    print(f"\n-> {indicator}")

    # --- Prepare data (shared) ---------------------------------------------
    data, lag_columns, harm_cols, wbgt_mean = prepare_data(indicator)
    if data is None:
        print(f"  [{indicator}] data not available — skipping.")
        continue

    cool, hot = get_contrast_bounds(data, CONTRAST_MODE, CONTRAST_PCTL, CONTRAST_FIXED)

    # --- df selection (optional) -------------------------------------------
    if RUN_DF_SELECTION:
        df_table = compare_spline_df(data, lag_columns, wbgt_mean, cool, hot,
                                     SEASON_CONTROL, harm_cols, INCLUDE_LAGS)
        df_table.insert(0, "indicator", indicator)
        df_tables.append(df_table)
        best_aic = df_table.loc[df_table["AIC"].idxmin(), "df"] if df_table["AIC"].notna().any() else "?"
        best_bic = df_table.loc[df_table["BIC"].idxmin(), "df"] if df_table["BIC"].notna().any() else "?"
        print(f"  df selection (AIC->{best_aic}, BIC->{best_bic}):")
        for _, r in df_table.iterrows():
            if pd.isna(r["AIC"]):
                print(f"       df={int(r['df'])}  {r['note']}")
            else:
                print(f"       df={int(r['df'])}  AIC={r['AIC']:.1f}  BIC={r['BIC']:.1f}  IRR={r['irr']:.3f}")

    # --- PRIMARY model -----------------------------------------------------
    primary = fit_and_evaluate(
        data, lag_columns, harm_cols, wbgt_mean,
        df_spline=SPLINE_DF, season=SEASON_CONTROL,
        contrast_mode=CONTRAST_MODE, contrast_pctl=CONTRAST_PCTL,
        contrast_fixed=CONTRAST_FIXED, include_lags=INCLUDE_LAGS)

    if primary is None:
        print(f"  [{indicator}] PRIMARY FAILED — skipping.")
        continue

    print(f"  PRIMARY  IRR({primary['cool']:.0f}->{primary['hot']:.0f})="
          f"{primary['irr']:.3f}  block p={primary['pval_block']:.3g}")

    primary_results.append({
        "indicator": indicator,
        "label": INDICATOR_LABELS.get(indicator, indicator),
        "irr": primary["irr"], "irr_lo": primary["irr_lo"],
        "irr_hi": primary["irr_hi"],
        "pval_contrast": primary["pval_contrast"],
        "pval_block": primary["pval_block"],
        "n_obs": primary["n_obs"],
        "n_facilities": data["facility"].nunique(),
    })

    # --- ROBUSTNESS CHECKS -------------------------------------------------
    # Each check changes ONE thing from the primary spec. Checks that only
    # change the contrast or the lag inclusion reuse the primary model (no
    # refit needed). Checks that change the model spec require a refit.

    robustness_specs = {
        # --- Contrast-only (no refit, same model) ---
        "percentile contrast (10-90)": dict(
            refit=False,
            contrast_mode="percentile", contrast_pctl=(10, 90),
            contrast_fixed=CONTRAST_FIXED, include_lags=INCLUDE_LAGS),
        "percentile contrast (5-95)": dict(
            refit=False,
            contrast_mode="percentile", contrast_pctl=(5, 95),
            contrast_fixed=CONTRAST_FIXED, include_lags=INCLUDE_LAGS),
        "sustained lags": dict(
            refit=False,
            contrast_mode=CONTRAST_MODE, contrast_pctl=CONTRAST_PCTL,
            contrast_fixed=CONTRAST_FIXED, include_lags=True),

        # --- Refit needed ---
        "harmonic season": dict(
            refit=True, season="harmonic", df_spline=SPLINE_DF),
        "df = 4": dict(
            refit=True, season=SEASON_CONTROL, df_spline=4),
        "df = 5": dict(
            refit=True, season=SEASON_CONTROL, df_spline=5),
    }

    # Skip the harmonic check if primary is already harmonic.
    if SEASON_CONTROL == "harmonic":
        robustness_specs["month FE season"] = robustness_specs.pop("harmonic season")
        robustness_specs["month FE season"]["season"] = "month_fe"

    # Skip df checks that match the primary.
    robustness_specs = {k: v for k, v in robustness_specs.items()
                        if not (k.startswith("df =") and v.get("df_spline") == SPLINE_DF)}

    primary_model = primary["model"]
    primary_di    = spline_design_info(data, SPLINE_DF)

    for check_name, spec in robustness_specs.items():
        if spec.get("refit"):
            # Refit with a different model spec.
            r = fit_and_evaluate(
                data, lag_columns, harm_cols, wbgt_mean,
                df_spline=spec["df_spline"], season=spec["season"],
                contrast_mode=CONTRAST_MODE, contrast_pctl=CONTRAST_PCTL,
                contrast_fixed=CONTRAST_FIXED, include_lags=INCLUDE_LAGS)
            if r is None:
                robustness_rows.append({
                    "indicator": indicator, "check": check_name,
                    "irr": np.nan, "irr_lo": np.nan, "irr_hi": np.nan,
                    "note": "fit failed"})
                continue
            irr_row = r
        else:
            # Same model, different contrast evaluation.
            c, h = get_contrast_bounds(data, spec["contrast_mode"],
                                       spec["contrast_pctl"], spec["contrast_fixed"])
            effect = cumulative_irr(primary_model, primary_di, wbgt_mean,
                                    c, h, spec["include_lags"])
            irr_row = {"irr": effect["irr"], "irr_lo": effect["irr_lo"],
                       "irr_hi": effect["irr_hi"], "cool": c, "hot": h}

        robustness_rows.append({
            "indicator": indicator, "check": check_name,
            "irr": irr_row["irr"], "irr_lo": irr_row["irr_lo"],
            "irr_hi": irr_row["irr_hi"], "note": ""})

    # Print robustness summary for this indicator.
    ind_rob = [r for r in robustness_rows if r["indicator"] == indicator]
    print(f"  robustness:")
    for r in ind_rob:
        if r["note"]:
            print(f"       {r['check']:30s}  {r['note']}")
        else:
            print(f"       {r['check']:30s}  IRR={r['irr']:.3f}")


# ===========================================================================
# 9. RECONCILE + SAVE
# ===========================================================================

fitted = [r["indicator"] for r in primary_results]
skipped = [i for i in COUNT_INDICATORS if i not in fitted]
print("\n" + "=" * 60)
print(f"Fitted {len(fitted)} of {len(COUNT_INDICATORS)} indicators")
if skipped:
    print(f"SKIPPED: {', '.join(skipped)}")
print("=" * 60)

if df_tables:
    pd.concat(df_tables, ignore_index=True).to_csv(
        f"{OUT_DIR}spline_df_selection.csv", index=False)

if not primary_results:
    raise RuntimeError("No indicators fitted — check panel paths.")

results = pd.DataFrame(primary_results)
results["pval_block_bh"] = multipletests(results["pval_block"], method="fdr_bh")[1]
results["significant"]   = results["pval_block_bh"] < 0.05
results = results.sort_values("irr").reset_index(drop=True)
results.to_csv(f"{OUT_DIR}all_indicators_cumulative_irr.csv", index=False)
print(f"Primary results -> {OUT_DIR}all_indicators_cumulative_irr.csv")

robustness = pd.DataFrame(robustness_rows)
robustness.to_csv(f"{OUT_DIR}robustness_checks.csv", index=False)
print(f"Robustness panel -> {OUT_DIR}robustness_checks.csv")


# ===========================================================================
# 10. FOREST PLOT — PRIMARY
# ===========================================================================

RED, GREY = "#823038", "#888888"
y = np.arange(len(results))
colors = [RED if s else GREY for s in results["significant"]]

fig, ax = plt.subplots(figsize=(7.5, max(4, len(results) * 0.5 + 1.5)))
for i in range(len(results)):
    row = results.iloc[i]
    ax.plot([row.irr_lo, row.irr_hi], [i, i], color=colors[i], lw=1.6, zorder=1)
    if row.significant:
        ax.axhspan(i-0.4, i+0.4, color="#f7e0e2", alpha=0.35, zorder=0)
ax.scatter(results.irr, y, color=colors, s=60, zorder=2)
ax.axvline(1.0, color="black", ls="--", lw=0.9)
ax.set_yticks(y); ax.set_yticklabels(results.label, fontsize=9)
ctext = (f"{CONTRAST_FIXED[0]:.0f} to {CONTRAST_FIXED[1]:.0f}C WBGT"
         if CONTRAST_MODE == "fixed"
         else f"{CONTRAST_PCTL[0]}th-{CONTRAST_PCTL[1]}th pctl WBGT")
season_desc = "month FE" if SEASON_CONTROL == "month_fe" else f"harmonic ({N_HARMONICS})"
ax.set_xlabel(f"Cumulative IRR for {ctext} (95% CI)", fontsize=10)
ax.set_title(f"Primary: Poisson, facility FE + {season_desc}, df={SPLINE_DF}\n"
             f"cluster-robust CIs; red = significant block after BH",
             fontsize=10, fontweight="bold")
ax.grid(axis="x", ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}forest_plot_primary.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Primary forest plot -> {OUT_DIR}forest_plot_primary.png")


# ===========================================================================
# 11. ROBUSTNESS PANEL PLOT
# ===========================================================================
# Each indicator gets a group of rows: the primary (bold) on top, then each
# robustness check below it with its name labelled. Clear vertical separation
# between indicators. Easy to read top-to-bottom.

if not robustness.empty:
    check_names = robustness["check"].unique().tolist()
    indicators_order = results["indicator"].tolist()

    # Layout: each indicator takes (1 primary + n checks) rows, plus a gap.
    n_checks = len(check_names)
    rows_per_indicator = 1 + n_checks
    gap = 0.6    # vertical gap between indicator groups

    # Assign y-positions bottom-to-top so first indicator is at the top.
    y_map = {}   # (indicator, check_name_or_"primary") -> y position
    y_cursor = 0
    for indicator in reversed(indicators_order):
        # Robustness checks go at the bottom of the group (lower y).
        for chk_i, check_name in enumerate(reversed(check_names)):
            y_map[(indicator, check_name)] = y_cursor
            y_cursor += 1
        # Primary goes at the top of the group (highest y in the group).
        y_map[(indicator, "primary")] = y_cursor
        y_cursor += 1 + gap   # gap before next indicator group

    fig_height = max(5, y_cursor * 0.28 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    # Draw each indicator group.
    for indicator in indicators_order:
        label = INDICATOR_LABELS.get(indicator, indicator)

        # Primary.
        prow = results[results["indicator"] == indicator].iloc[0]
        yp = y_map[(indicator, "primary")]
        ax.plot([prow.irr_lo, prow.irr_hi], [yp, yp],
                color="#2a78d6", lw=2.0, zorder=2)
        ax.scatter([prow.irr], [yp], color="#2a78d6", s=55, zorder=3)
        ax.text(-0.02, yp, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, fontweight="bold")

        # Robustness checks.
        ind_rob = robustness[robustness["indicator"] == indicator]
        for check_name in check_names:
            yc = y_map[(indicator, check_name)]
            row = ind_rob[ind_rob["check"] == check_name]
            if row.empty or pd.isna(row.iloc[0]["irr"]):
                ax.text(-0.02, yc, f"  {check_name}", fontsize=7, color="#999",
                        transform=ax.get_yaxis_transform(), ha="right", va="center")
                continue
            r = row.iloc[0]
            ax.plot([r.irr_lo, r.irr_hi], [yc, yc],
                    color="#888888", lw=1.0, zorder=1)
            ax.scatter([r.irr], [yc], color="#888888", s=25, zorder=2)
            ax.text(-0.02, yc, f"  {check_name}", fontsize=7, color="#555",
                    transform=ax.get_yaxis_transform(), ha="right", va="center")

        # Light horizontal line separating indicator groups.
        group_bottom = y_map[(indicator, check_names[-1])] - 0.4
        ax.axhline(group_bottom - gap/2, color="#e0e0e0", lw=0.5, zorder=0)

    ax.axvline(1.0, color="black", ls="--", lw=0.9, zorder=0)
    ax.set_yticks([])
    ax.set_xlabel("Cumulative IRR (95% CI)", fontsize=10)
    ax.set_title("Primary specification (blue) vs robustness checks (grey)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", ls=":", alpha=0.4)
    plt.subplots_adjust(left=0.35)

    # Tighten x margins.
    all_irrs = list(results["irr"]) + list(robustness["irr"].dropna())
    all_los  = list(results["irr_lo"]) + list(robustness["irr_lo"].dropna())
    all_his  = list(results["irr_hi"]) + list(robustness["irr_hi"].dropna())
    x_lo = min(all_los) - 0.02
    x_hi = max(all_his) + 0.02
    ax.set_xlim(x_lo, x_hi)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}robustness_panel.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Robustness panel -> {OUT_DIR}robustness_panel.png")

print("\nDone.")

# ===========================================================================
# 12. EXPOSURE-RESPONSE CURVES — ALL INDICATORS IN ONE PANEL FIGURE
# ===========================================================================

def exposure_response_curve(model, di, wbgt_mean, data, n_points=80):
    """
    Evaluate the fitted spline at a grid of WBGT values, returning IRR and
    CI relative to the median WBGT of the data.
    """
    wbgt_lo = float(np.nanpercentile(data[WBGT_VAR], 2))
    wbgt_hi = float(np.nanpercentile(data[WBGT_VAR], 98))
    wbgt_grid = np.linspace(wbgt_lo, wbgt_hi, n_points)
    wbgt_ref  = float(np.nanmedian(data[WBGT_VAR]))

    names = list(model.coef().index)
    beta  = model.coef().values
    vcov  = model._vcov

    all_points = np.append(wbgt_grid, wbgt_ref)
    basis = patsy.build_design_matrices(
        [di], pd.DataFrame({WBGT_VAR: all_points}), return_type="dataframe"
    )[0]

    ref_row    = basis.iloc[-1]
    grid_basis = basis.iloc[:-1]

    irrs, los, his = [], [], []
    for i in range(len(wbgt_grid)):
        contrast = np.zeros(len(names))
        for j, name in enumerate(names):
            if name in basis.columns:
                contrast[j] = grid_basis[name].iloc[i] - ref_row[name]

        log_irr = float(contrast @ beta)
        se      = float(np.sqrt(contrast @ vcov @ contrast))
        irrs.append(np.exp(log_irr))
        los.append(np.exp(log_irr - 1.96 * se))
        his.append(np.exp(log_irr + 1.96 * se))

    return wbgt_grid, np.array(irrs), np.array(los), np.array(his), wbgt_ref


# ---------------------------------------------------------------------------
# Layout: compute grid dimensions dynamically from number of fitted indicators
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Generating combined exposure-response panel")
print("=" * 60)

N_PANELS   = len(fitted)
N_COLS     = 4                                      # adjust to taste
N_ROWS     = int(np.ceil(N_PANELS / N_COLS))

DF_TO_PLOT = [d for d in [3, 4, 5] if d >= 3]
CURVE_COLORS = {3: "#2a78d6", 4: "#eb6834", 5: "#7cb342"}

fig, axes = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(N_COLS * 4.5, N_ROWS * 3.8),
    constrained_layout=True,
)

# Flatten axes array for easy indexing; hide any unused panels at the end.
axes_flat = axes.flatten() if N_PANELS > 1 else [axes]
for ax in axes_flat[N_PANELS:]:
    ax.set_visible(False)

for panel_idx, indicator in enumerate(fitted):
    ax = axes_flat[panel_idx]
    label = INDICATOR_LABELS.get(indicator, indicator)

    # Re-load data for this indicator.
    data, lag_columns, harm_cols, wbgt_mean = prepare_data(indicator)
    if data is None:
        ax.set_visible(False)
        continue

    cool, hot = get_contrast_bounds(
        data, CONTRAST_MODE, CONTRAST_PCTL, CONTRAST_FIXED
    )

    plotted_any = False
    for df_s in DF_TO_PLOT:
        formula = build_formula(df_s, lag_columns, SEASON_CONTROL, harm_cols)
        try:
            model = pf.fepois(formula, data=data, vcov={"CRV1": "facility"})
        except Exception:
            continue

        di = spline_design_info(data, df_s)
        wgrid, irrs, los, his, ref = exposure_response_curve(
            model, di, wbgt_mean, data
        )

        c          = CURVE_COLORS.get(df_s, "#999999")
        is_primary = (df_s == SPLINE_DF)

        if is_primary:
            ax.fill_between(wgrid, los, his, alpha=0.15, color=c)
            ax.plot(wgrid, irrs, color=c, lw=2.0,
                    label=f"df={df_s} (primary)")
        else:
            ax.plot(wgrid, irrs, color=c, lw=1.1, ls="--",
                    alpha=0.8, label=f"df={df_s}")

        plotted_any = True

    if not plotted_any:
        ax.set_visible(False)
        continue

    # Reference line and contrast shading.
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ymin, ymax = ax.get_ylim()
    ax.axvspan(cool, hot, alpha=0.07, color="#823038", zorder=0)
    ax.text(
        cool + (hot - cool) / 2, ymax * 0.97,
        f"{cool:.0f}–{hot:.0f}°C",
        ha="center", va="top", fontsize=7, color="#823038",
    )

    # Axis labels — only on edge subplots to reduce clutter.
    row_idx = panel_idx // N_COLS
    col_idx = panel_idx  % N_COLS
    if row_idx == N_ROWS - 1 or panel_idx + N_COLS >= N_PANELS:
        ax.set_xlabel("WBGT (°C)", fontsize=8)
    if col_idx == 0:
        ax.set_ylabel("IRR (vs median WBGT)", fontsize=8)

    ax.set_title(label, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)
    ax.grid(axis="both", ls=":", alpha=0.3)
    ax.legend(fontsize=6, loc="best", framealpha=0.6)

# ---------------------------------------------------------------------------
# Shared super-title and save
# ---------------------------------------------------------------------------
season_desc = "month FE" if SEASON_CONTROL == "month_fe" else f"harmonic ({N_HARMONICS})"
fig.suptitle(
    f"Exposure-response curves — Poisson FE, {season_desc}\n"
    f"IRR relative to median WBGT  |  shaded band = 95% CI (primary df only)",
    fontsize=11, fontweight="bold", y=1.01,
)

out_path = f"{OUT_DIR}exposure_response_panel_all.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Combined panel -> {out_path}")
print("\nAll done.")
