"""
loop_all_indicators_complete.py  (pyfixest / Poisson-FE version)

ONE script, ONE model per indicator, ALL outputs:
  1. Primary cumulative IRR + Wald block test + BH correction
  2. df selection (AIC/BIC)
  3. Robustness checks (harmonic season, different df, contrasts, lags)
  4. Forest plot + robustness panel plot
  5. Exposure-response curves (per indicator, overlaying df=3/4/5)
  6. Historical burden (observed vs counterfactual at reference WBGT)
  7. Forward projection (CMIP6: historical vs projected WBGT)
  8. Time-series gap plot (observed vs counterfactual)
  9. District-level aggregation + choropleth maps
  10. TLO model lookup table (WBGT -> disruption probability)
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
# 1. SETTINGS
# ===========================================================================

COUNT_INDICATORS = [
    "fp_total_clients", "opd_attendance",
    #"ipd_total_admissions",
    "vmmc_first_visits",
    "pnc_mother_checked_48h",
    #"anc_new_attendees",
    "anc_first_trimester_starts",
    "bcg_under1", "penta3_under1", "measles1_under1",
    "fully_immunised_under1", "pnc_within_2wks", "pnc_first_visit_2wks",
    #"live_births_total",
    "skilled_deliveries",
]

INDICATOR_LABELS = {
    "fp_total_clients":           "FP Total Clients",
    "opd_attendance":             "OPD Attendance",
    "vmmc_first_visits":          "VMMC First Visits",
    "pnc_mother_checked_48h":     "PNC Mother <48h",
    "anc_first_trimester_starts": "ANC 1st Trimester Starts",
    "bcg_under1":                 "BCG Under-1",
    "penta3_under1":              "Penta3 Under-1",
    "measles1_under1":            "Measles 1st Dose Under-1",
    "fully_immunised_under1":     "Fully Immunised Under-1",
    "pnc_within_2wks":            "PNC Within 2 Weeks",
    "pnc_first_visit_2wks":       "PNC First Visit <2 Weeks",
    "skilled_deliveries":         "Skilled Deliveries",
}

WBGT_VAR   = "wbgt5x_day"
SPLINE_DF  = 4              # used when DF_MODE = "fixed"
LAG_MONTHS = [1, 2, 3, 9]

# How to choose spline df.
# "fixed"         : use SPLINE_DF for every indicator
# "per_indicator" : use AIC-best df from DF_CANDIDATES for each indicator
DF_MODE = "fixed"

# Fixed knot positions for the spline. Set to None to let patsy choose by
# quantiles (the old behaviour). Set to a list of WBGT values to pin the
# knots — this eliminates knot-wandering as a source of df sensitivity.
# Compute once from the pooled WBGT distribution across all panels, e.g.:
#   FIXED_KNOTS = [25.0, 27.5, 30.0]
# With 3 internal knots + natural boundary constraints, this gives df=4.
FIXED_KNOTS = None   # set to e.g. [25.0, 27.5, 30.0] after checking your data

# Piecewise linear (hockey stick) threshold for the robustness check.
# Below this: flat (no effect). Above this: linear slope per degree.
PIECEWISE_THRESHOLD = 28.0

SEASON_CONTROL = "month_fe"
N_HARMONICS    = 2
CONTRAST_MODE  = "fixed"
CONTRAST_PCTL  = (50, 90)
CONTRAST_FIXED = (25.0, 32.0)
INCLUDE_LAGS   = False

RUN_DF_SELECTION = True
DF_CANDIDATES    = (3, 4, 5, 6)

MIN_YEAR, MAX_YEAR, MIN_OBS = 2015, 2025, 10

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

# Historical burden.
REFERENCE_WBGT = 25.0

# CMIP6 projections.
PROJECTION_DIR = f"{DATA_DIR}/CMIP6_facility_projections"
SSP_SCENARIOS  = ["ssp126", "ssp245", "ssp585"]
MODEL_TIERS    = ["lowest", "median", "highest"]

# District maps.
SHAPEFILE_PATH    = None   # set to your Malawi districts shapefile path
DISTRICT_NAME_COL = "ADM2_EN"

# TLO lookup.
WBGT_GRID = np.arange(20.0, 37.0, 0.5)


# ===========================================================================
# 2. HELPERS
# ===========================================================================

def spline_design_info(data, df_spline):
    if FIXED_KNOTS is not None:
        knot_str = ", ".join(str(k) for k in FIXED_KNOTS)
        formula_str = f"cr({WBGT_VAR}, knots=[{knot_str}])"
    else:
        formula_str = f"cr({WBGT_VAR}, df={df_spline})"
    return patsy.dmatrix(formula_str, data, return_type="dataframe").design_info


def add_harmonic_columns(data, n_harmonics):
    names = []
    for k in range(1, n_harmonics + 1):
        s, c = f"sin{k}", f"cos{k}"
        data[s] = np.sin(2 * np.pi * k * data["month"] / 12)
        data[c] = np.cos(2 * np.pi * k * data["month"] / 12)
        names += [s, c]
    return names


def build_formula(df_spline, lag_columns, season="month_fe", harm_cols=None):
    if FIXED_KNOTS is not None:
        knot_str = ", ".join(str(k) for k in FIXED_KNOTS)
        spline_term = f"cr({WBGT_VAR}, knots=[{knot_str}])"
    else:
        spline_term = f"cr({WBGT_VAR}, df={df_spline})"
    rhs = [spline_term] + lag_columns + ["year_c"]
    if season == "harmonic" and harm_cols:
        rhs += harm_cols
        fe = "facility"
    else:
        fe = "facility + month"
    return f"y_int ~ {' + '.join(rhs)} | {fe}"


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
        [di], pd.DataFrame({WBGT_VAR: [cool, hot]}), return_type="dataframe")[0]
    contrast = np.zeros(len(names))
    for j, name in enumerate(names):
        if name in basis.columns:
            contrast[j] = basis[name].iloc[1] - basis[name].iloc[0]
        elif "_lag" in name:
            contrast[j] = (hot - cool) if include_lags else 0.0
    log_irr = float(contrast @ beta)
    se = float(np.sqrt(contrast @ vcov @ contrast))
    z = log_irr / se
    return {"irr": np.exp(log_irr), "irr_lo": np.exp(log_irr - 1.96*se),
            "irr_hi": np.exp(log_irr + 1.96*se),
            "pval": float(2 * stats.norm.sf(abs(z)))}


def weather_block_pvalue(model):
    names = list(model.coef().index)
    beta, vcov = model.coef().values, model._vcov
    w = [i for i, n in enumerate(names) if WBGT_VAR in n]
    if not w: return np.nan
    R = np.zeros((len(w), len(names)))
    for r, i in enumerate(w): R[r, i] = 1.0
    Rb, RVR = R @ beta, R @ vcov @ R.T
    return float(stats.chi2.sf(float(Rb @ np.linalg.solve(RVR, Rb)), len(w)))


def compare_spline_df(data, lag_columns, wbgt_mean, cool, hot,
                      season, harm_cols, include_lags):
    rows = []
    for df_s in DF_CANDIDATES:
        formula = build_formula(df_s, lag_columns, season, harm_cols)
        try:
            m = pf.fepois(formula, data=data, vcov={"CRV1": "facility"})
        except Exception as e:
            rows.append({"df": df_s, "loglik": np.nan, "k": np.nan,
                         "AIC": np.nan, "BIC": np.nan, "irr": np.nan, "note": str(e)[:30]})
            continue
        ll, k, n = m._loglik, len(m.coef()), m._N
        di = spline_design_info(data, df_s)
        irr = cumulative_irr(m, di, wbgt_mean, cool, hot, include_lags)["irr"]
        rows.append({"df": df_s, "loglik": ll, "k": k,
                     "AIC": -2*ll+2*k, "BIC": -2*ll+k*np.log(n), "irr": irr, "note": ""})
    return pd.DataFrame(rows)


def compute_row_irrs(model, di, wbgt_from, wbgt_to):
    names = list(model.coef().index)
    beta = model.coef().values
    irrs = np.ones(len(wbgt_from))
    for i in range(len(wbgt_from)):
        if np.isnan(wbgt_from[i]) or np.isnan(wbgt_to[i]): continue
        basis = patsy.build_design_matrices(
            [di], pd.DataFrame({WBGT_VAR: [wbgt_from[i], wbgt_to[i]]}),
            return_type="dataframe")[0]
        c = np.zeros(len(names))
        for j, nm in enumerate(names):
            if nm in basis.columns: c[j] = basis[nm].iloc[1] - basis[nm].iloc[0]
        irrs[i] = np.exp(float(c @ beta))
    return irrs


def exposure_response_curve(model, di, data, n_points=80):
    wbgt_lo = float(np.nanpercentile(data[WBGT_VAR], 2))
    wbgt_hi = float(np.nanpercentile(data[WBGT_VAR], 98))
    wgrid = np.linspace(wbgt_lo, wbgt_hi, n_points)
    wref = float(np.nanmedian(data[WBGT_VAR]))
    names, beta, vcov = list(model.coef().index), model.coef().values, model._vcov
    all_pts = np.append(wgrid, wref)
    basis = patsy.build_design_matrices(
        [di], pd.DataFrame({WBGT_VAR: all_pts}), return_type="dataframe")[0]
    ref_row = basis.iloc[-1]
    irrs, los, his = [], [], []
    for i in range(len(wgrid)):
        c = np.zeros(len(names))
        for j, nm in enumerate(names):
            if nm in basis.columns: c[j] = basis.iloc[i][nm] - ref_row[nm]
        li = float(c @ beta); se = float(np.sqrt(c @ vcov @ c))
        irrs.append(np.exp(li)); los.append(np.exp(li-1.96*se)); his.append(np.exp(li+1.96*se))
    return wgrid, np.array(irrs), np.array(los), np.array(his), wref


# ===========================================================================
# 3. PREPARE DATA
# ===========================================================================

def prepare_data(indicator):
    path = f"{DATA_DIR}/All_predictors_processed/regression_panel_{indicator}.csv"
    if not os.path.exists(path): return None, None, None, None
    data = pd.read_csv(path, parse_dates=["date"]).rename(columns={indicator: "y"})
    if WBGT_VAR not in data.columns: return None, None, None, None

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

    harm_cols = add_harmonic_columns(data, N_HARMONICS)
    needed = ["y", "facility", "month", "year_c", WBGT_VAR] + lag_columns
    data = data.dropna(subset=needed).copy()
    data["y_int"] = data["y"].round().clip(lower=0).astype(int)
    data["facility"] = data["facility"].astype(str)
    data["month"] = data["month"].astype(int)

    good = data.groupby("facility").size()
    data = data[data["facility"].isin(good[good >= MIN_OBS].index)].copy()
    if data["facility"].nunique() < 2: return None, None, None, None
    return data, lag_columns, harm_cols, wbgt_mean


# ===========================================================================
# 4. FIT + EVALUATE
# ===========================================================================

def fit_and_evaluate(data, lag_columns, harm_cols, wbgt_mean,
                     df_spline, season, contrast_mode, contrast_pctl,
                     contrast_fixed, include_lags):
    formula = build_formula(df_spline, lag_columns, season, harm_cols)
    try:
        model = pf.fepois(formula, data=data, vcov={"CRV1": "facility"})
    except Exception:
        return None
    cool, hot = get_contrast_bounds(data, contrast_mode, contrast_pctl, contrast_fixed)
    di = spline_design_info(data, df_spline)
    effect = cumulative_irr(model, di, wbgt_mean, cool, hot, include_lags)
    block_p = weather_block_pvalue(model)

    # Goodness of fit.
    ll      = model._loglik
    ll_null = model._loglik_null if hasattr(model, '_loglik_null') else np.nan
    k       = len(model.coef())
    n       = int(model._N)
    aic     = -2*ll + 2*k
    bic     = -2*ll + k*np.log(n)
    pseudo_r2 = model._pseudo_r2 if hasattr(model, '_pseudo_r2') else np.nan
    deviance  = model.deviance if hasattr(model, 'deviance') else np.nan

    return {"irr": effect["irr"], "irr_lo": effect["irr_lo"],
            "irr_hi": effect["irr_hi"], "pval_contrast": effect["pval"],
            "pval_block": block_p, "n_obs": n,
            "cool": cool, "hot": hot, "model": model,
            "loglik": ll, "loglik_null": ll_null,
            "aic": aic, "bic": bic,
            "pseudo_r2": pseudo_r2, "deviance": deviance, "k": k}


# ===========================================================================
# 5. MAIN LOOP — PRIMARY + ROBUSTNESS + STORE FOR DOWNSTREAM
# ===========================================================================

print("=" * 60)
print(f"Primary: season={SEASON_CONTROL}, df={DF_MODE} "
      f"{'('+str(SPLINE_DF)+')' if DF_MODE=='fixed' else '(AIC-selected)'}, "
      f"contrast={CONTRAST_MODE}, lags={'on' if INCLUDE_LAGS else 'off'}")
print("=" * 60)

primary_results = []
robustness_rows = []
df_tables = []
indicator_state = {}

for indicator in COUNT_INDICATORS:
    print(f"\n-> {indicator}")
    data, lag_columns, harm_cols, wbgt_mean = prepare_data(indicator)
    if data is None:
        print(f"  [{indicator}] data not available — skipping.")
        continue

    cool, hot = get_contrast_bounds(data, CONTRAST_MODE, CONTRAST_PCTL, CONTRAST_FIXED)

    # --- df selection: always run so we have the table -----------------------
    df_table = compare_spline_df(data, lag_columns, wbgt_mean, cool, hot,
                                 SEASON_CONTROL, harm_cols, INCLUDE_LAGS)
    df_table.insert(0, "indicator", indicator)
    df_tables.append(df_table)

    valid_df = df_table.dropna(subset=["AIC"])
    if valid_df.empty:
        print(f"  [{indicator}] all df candidates failed — skipping.")
        continue

    best_aic_df = int(valid_df.loc[valid_df["AIC"].idxmin(), "df"])
    best_bic_df = int(valid_df.loc[valid_df["BIC"].idxmin(), "df"])

    # Choose the df for this indicator's primary model.
    if DF_MODE == "per_indicator":
        chosen_df = best_aic_df
    else:
        chosen_df = SPLINE_DF

    print(f"  df selection (AIC->{best_aic_df}, BIC->{best_bic_df})  "
          f"-> using df={chosen_df}:")
    for _, r in df_table.iterrows():
        if pd.isna(r["AIC"]):
            print(f"       df={int(r['df'])}  {r['note']}")
        else:
            marker = " <-- PRIMARY" if int(r["df"]) == chosen_df else ""
            print(f"       df={int(r['df'])}  AIC={r['AIC']:.1f}  "
                  f"BIC={r['BIC']:.1f}  IRR={r['irr']:.3f}{marker}")

    # --- Primary model at the chosen df ------------------------------------
    primary = fit_and_evaluate(data, lag_columns, harm_cols, wbgt_mean,
        df_spline=chosen_df, season=SEASON_CONTROL, contrast_mode=CONTRAST_MODE,
        contrast_pctl=CONTRAST_PCTL, contrast_fixed=CONTRAST_FIXED, include_lags=INCLUDE_LAGS)

    if primary is None:
        print(f"  [{indicator}] PRIMARY FAILED — skipping.")
        continue

    print(f"  PRIMARY (df={chosen_df})  IRR({primary['cool']:.0f}->{primary['hot']:.0f})="
          f"{primary['irr']:.3f}  block p={primary['pval_block']:.3g}  "
          f"AIC={primary['aic']:.0f}  BIC={primary['bic']:.0f}  "
          f"pseudo-R2={primary['pseudo_r2']:.4f}  loglik={primary['loglik']:.0f}")

    di = spline_design_info(data, chosen_df)
    indicator_state[indicator] = {
        "model": primary["model"], "data": data, "di": di,
        "lag_columns": lag_columns, "harm_cols": harm_cols,
        "wbgt_mean": wbgt_mean, "chosen_df": chosen_df}

    primary_results.append({
        "indicator": indicator, "label": INDICATOR_LABELS.get(indicator, indicator),
        "spline_df": chosen_df,
        "irr": primary["irr"], "irr_lo": primary["irr_lo"], "irr_hi": primary["irr_hi"],
        "pval_contrast": primary["pval_contrast"], "pval_block": primary["pval_block"],
        "n_obs": primary["n_obs"], "n_facilities": data["facility"].nunique(),
        "loglik": primary["loglik"], "loglik_null": primary["loglik_null"],
        "aic": primary["aic"], "bic": primary["bic"],
        "pseudo_r2": primary["pseudo_r2"], "deviance": primary["deviance"],
        "n_params": primary["k"]})

    # --- Robustness checks ------------------------------------------------
    # df checks: test df±1 from the chosen df (if they're in range).
    df_minus = chosen_df - 1 if chosen_df - 1 >= 3 else None
    df_plus  = chosen_df + 1 if chosen_df + 1 <= max(DF_CANDIDATES) else None

    robustness_specs = {
        "percentile (10-90)": dict(refit=False, contrast_mode="percentile",
            contrast_pctl=(10,90), contrast_fixed=CONTRAST_FIXED, include_lags=INCLUDE_LAGS),
        "percentile (5-95)": dict(refit=False, contrast_mode="percentile",
            contrast_pctl=(5,95), contrast_fixed=CONTRAST_FIXED, include_lags=INCLUDE_LAGS),
        "sustained lags": dict(refit=False, contrast_mode=CONTRAST_MODE,
            contrast_pctl=CONTRAST_PCTL, contrast_fixed=CONTRAST_FIXED, include_lags=True),
        "harmonic season": dict(refit=True, season="harmonic", df_spline=chosen_df),
    }
    if df_minus is not None:
        robustness_specs[f"df = {df_minus}"] = dict(
            refit=True, season=SEASON_CONTROL, df_spline=df_minus)
    if df_plus is not None:
        robustness_specs[f"df = {df_plus}"] = dict(
            refit=True, season=SEASON_CONTROL, df_spline=df_plus)

    if SEASON_CONTROL == "harmonic":
        robustness_specs["month FE"] = robustness_specs.pop("harmonic season")
        robustness_specs["month FE"]["season"] = "month_fe"

    for cn, spec in robustness_specs.items():
        if spec.get("refit"):
            r = fit_and_evaluate(data, lag_columns, harm_cols, wbgt_mean,
                df_spline=spec["df_spline"], season=spec["season"],
                contrast_mode=CONTRAST_MODE, contrast_pctl=CONTRAST_PCTL,
                contrast_fixed=CONTRAST_FIXED, include_lags=INCLUDE_LAGS)
            if r is None:
                robustness_rows.append({"indicator": indicator, "check": cn,
                    "irr": np.nan, "irr_lo": np.nan, "irr_hi": np.nan, "note": "failed"})
                continue
            irr_row = r
        else:
            c2, h2 = get_contrast_bounds(data, spec["contrast_mode"],
                                          spec["contrast_pctl"], spec["contrast_fixed"])
            eff = cumulative_irr(primary["model"], di, wbgt_mean, c2, h2, spec["include_lags"])
            irr_row = {"irr": eff["irr"], "irr_lo": eff["irr_lo"], "irr_hi": eff["irr_hi"]}
        robustness_rows.append({"indicator": indicator, "check": cn,
            "irr": irr_row["irr"], "irr_lo": irr_row["irr_lo"],
            "irr_hi": irr_row["irr_hi"], "note": ""})

    # --- Piecewise linear (hockey stick) robustness check -------------------
    # Flat below the threshold, linear slope above. Two parameters, no df/knot
    # choice, directly interpretable. The IRR is exp(beta * (hot - threshold))
    # for the contrast window above the threshold.
    try:
        data_pw = data.copy()
        data_pw["wbgt_below"] = np.minimum(data_pw[WBGT_VAR], PIECEWISE_THRESHOLD)
        data_pw["wbgt_above"] = np.maximum(data_pw[WBGT_VAR] - PIECEWISE_THRESHOLD, 0.0)

        pw_rhs = ["wbgt_below", "wbgt_above"] + lag_columns + ["year_c"]
        if SEASON_CONTROL == "harmonic" and harm_cols:
            pw_rhs += harm_cols
            pw_fe = "facility"
        else:
            pw_fe = "facility + month"
        pw_formula = f"y_int ~ {' + '.join(pw_rhs)} | {pw_fe}"

        model_pw = pf.fepois(pw_formula, data=data_pw, vcov={"CRV1": "facility"})
        beta_above = model_pw.coef()["wbgt_above"]
        se_above = np.sqrt(model_pw._vcov[
            list(model_pw.coef().index).index("wbgt_above"),
            list(model_pw.coef().index).index("wbgt_above")])

        # IRR for the contrast window (only the part above the threshold).
        cool, hot = get_contrast_bounds(data, CONTRAST_MODE, CONTRAST_PCTL, CONTRAST_FIXED)
        degrees_above = max(0, hot - max(cool, PIECEWISE_THRESHOLD))
        log_irr_pw = beta_above * degrees_above
        se_pw = se_above * degrees_above

        irr_pw = np.exp(log_irr_pw)
        irr_pw_lo = np.exp(log_irr_pw - 1.96 * se_pw)
        irr_pw_hi = np.exp(log_irr_pw + 1.96 * se_pw)

        robustness_rows.append({"indicator": indicator,
            "check": f"piecewise linear (>{PIECEWISE_THRESHOLD}°C)",
            "irr": irr_pw, "irr_lo": irr_pw_lo, "irr_hi": irr_pw_hi, "note": ""})

        # Also report the per-degree IRR for interpretability.
        irr_per_deg = np.exp(beta_above)
        irr_per_deg_lo = np.exp(beta_above - 1.96 * se_above)
        irr_per_deg_hi = np.exp(beta_above + 1.96 * se_above)
        pval_above = float(2 * stats.norm.sf(abs(beta_above / se_above)))

        print(f"  piecewise linear (>{PIECEWISE_THRESHOLD}°C): "
              f"per-degree IRR={irr_per_deg:.3f} [{irr_per_deg_lo:.3f}, {irr_per_deg_hi:.3f}] "
              f"p={pval_above:.3g}")

    except Exception as e:
        robustness_rows.append({"indicator": indicator,
            "check": f"piecewise linear (>{PIECEWISE_THRESHOLD}°C)",
            "irr": np.nan, "irr_lo": np.nan, "irr_hi": np.nan, "note": f"failed: {e}"})

    ind_rob = [r for r in robustness_rows if r["indicator"] == indicator]
    print("  robustness:")
    for r in ind_rob:
        val = f"IRR={r['irr']:.3f}" if not r["note"] else r["note"]
        print(f"       {r['check']:25s}  {val}")


# ===========================================================================
# 6. SAVE
# ===========================================================================

fitted = list(indicator_state.keys())
skipped = [i for i in COUNT_INDICATORS if i not in fitted]
print("\n" + "=" * 60)
print(f"Fitted {len(fitted)} of {len(COUNT_INDICATORS)}")
if skipped: print(f"SKIPPED: {', '.join(skipped)}")
print("=" * 60)

if df_tables:
    pd.concat(df_tables, ignore_index=True).to_csv(f"{OUT_DIR}spline_df_selection.csv", index=False)

if not primary_results:
    raise RuntimeError("No indicators fitted.")

results = pd.DataFrame(primary_results)
results["pval_block_bh"] = multipletests(results["pval_block"], method="fdr_bh")[1]
results["significant"] = results["pval_block_bh"] < 0.05
results = results.sort_values("irr").reset_index(drop=True)
results.to_csv(f"{OUT_DIR}all_indicators_cumulative_irr.csv", index=False)

robustness = pd.DataFrame(robustness_rows)
robustness.to_csv(f"{OUT_DIR}robustness_checks.csv", index=False)
print("Primary + robustness saved.")


# ===========================================================================
# 7. FOREST PLOT
# ===========================================================================

RED, GREY = "#823038", "#888888"
y = np.arange(len(results))
colors = [RED if s else GREY for s in results["significant"]]
fig, ax = plt.subplots(figsize=(7.5, max(4, len(results)*0.5+1.5)))
for i in range(len(results)):
    row = results.iloc[i]
    ax.plot([row.irr_lo, row.irr_hi], [i, i], color=colors[i], lw=1.6, zorder=1)
    if row.significant: ax.axhspan(i-0.4, i+0.4, color="#f7e0e2", alpha=0.35, zorder=0)
ax.scatter(results.irr, y, color=colors, s=60, zorder=2)
ax.axvline(1.0, color="black", ls="--", lw=0.9)
ax.set_yticks(y)
if DF_MODE == "per_indicator":
    ylabels = [f"{row.label} (df={int(row.spline_df)})" for _, row in results.iterrows()]
else:
    ylabels = [row.label for _, row in results.iterrows()]
ax.set_yticklabels(ylabels, fontsize=9)
ctext = (f"{CONTRAST_FIXED[0]:.0f}-{CONTRAST_FIXED[1]:.0f}C" if CONTRAST_MODE == "fixed"
         else f"{CONTRAST_PCTL[0]}th-{CONTRAST_PCTL[1]}th pctl")
sd = "month FE" if SEASON_CONTROL == "month_fe" else f"harmonic({N_HARMONICS})"
df_desc = "AIC-selected df" if DF_MODE == "per_indicator" else f"df={SPLINE_DF}"
ax.set_xlabel(f"Cumulative IRR for {ctext} WBGT (95% CI)", fontsize=10)
ax.set_title(f"Poisson FE + {sd}, {df_desc}; red=significant after BH",
             fontsize=10, fontweight="bold")
ax.grid(axis="x", ls=":", alpha=0.5); plt.tight_layout()
plt.savefig(f"{OUT_DIR}forest_plot_primary.png", dpi=180, bbox_inches="tight"); plt.close()


# ===========================================================================
# 8. ROBUSTNESS PANEL
# ===========================================================================

if not robustness.empty:
    check_names = robustness["check"].unique().tolist()
    ind_order = results["indicator"].tolist()
    gap = 0.6; y_map = {}; yc = 0
    for ind in reversed(ind_order):
        for cn in reversed(check_names): y_map[(ind, cn)] = yc; yc += 1
        y_map[(ind, "primary")] = yc; yc += 1 + gap

    fig, ax = plt.subplots(figsize=(9, max(5, yc*0.28+1.5)))
    for ind in ind_order:
        lbl = INDICATOR_LABELS.get(ind, ind)
        prow = results[results["indicator"] == ind].iloc[0]
        yp = y_map[(ind, "primary")]
        ax.plot([prow.irr_lo, prow.irr_hi], [yp, yp], color="#2a78d6", lw=2.0, zorder=2)
        ax.scatter([prow.irr], [yp], color="#2a78d6", s=55, zorder=3)
        ax.text(-0.02, yp, lbl, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, fontweight="bold")
        ir = robustness[robustness["indicator"] == ind]
        for cn in check_names:
            ypos = y_map[(ind, cn)]
            row = ir[ir["check"] == cn]
            if row.empty or pd.isna(row.iloc[0]["irr"]):
                ax.text(-0.02, ypos, f"  {cn}", fontsize=7, color="#999",
                        transform=ax.get_yaxis_transform(), ha="right", va="center")
                continue
            r = row.iloc[0]
            ax.plot([r.irr_lo, r.irr_hi], [ypos, ypos], color="#888", lw=1.0, zorder=1)
            ax.scatter([r.irr], [ypos], color="#888", s=25, zorder=2)
            ax.text(-0.02, ypos, f"  {cn}", fontsize=7, color="#555",
                    transform=ax.get_yaxis_transform(), ha="right", va="center")
        gb = y_map[(ind, check_names[-1])] - 0.4
        ax.axhline(gb - gap/2, color="#e0e0e0", lw=0.5, zorder=0)
    ax.axvline(1.0, color="black", ls="--", lw=0.9)
    ax.set_yticks([]); ax.set_xlabel("Cumulative IRR (95% CI)")
    ax.set_title("Primary (blue) vs robustness (grey)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", ls=":", alpha=0.4); plt.subplots_adjust(left=0.35)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}robustness_panel.png", dpi=180, bbox_inches="tight"); plt.close()


# ===========================================================================
# 9. EXPOSURE-RESPONSE CURVES
# ===========================================================================

print("\nExposure-response curves...")
CCOL = {3: "#2a78d6", 4: "#eb6834", 5: "#7cb342", 6: "#9c27b0"}
NC = 4; NR = int(np.ceil(len(fitted)/NC))
fig, axes = plt.subplots(NR, NC, figsize=(NC*4.5, NR*3.8), constrained_layout=True)
af = axes.flatten() if len(fitted) > 1 else [axes]
for ax in af[len(fitted):]: ax.set_visible(False)

for pi, ind in enumerate(fitted):
    ax = af[pi]; st = indicator_state[ind]
    chosen_df = st["chosen_df"]
    cool, hot = get_contrast_bounds(st["data"], CONTRAST_MODE, CONTRAST_PCTL, CONTRAST_FIXED)

    # Plot df-1, chosen, df+1 to show sensitivity around the selected df.
    dfs_to_plot = sorted(set([
        d for d in [chosen_df - 1, chosen_df, chosen_df + 1]
        if 3 <= d <= max(DF_CANDIDATES)
    ]))

    for df_s in dfs_to_plot:
        fm = build_formula(df_s, st["lag_columns"], SEASON_CONTROL, st["harm_cols"])
        try: m = pf.fepois(fm, data=st["data"], vcov={"CRV1": "facility"})
        except: continue
        di2 = spline_design_info(st["data"], df_s)
        wg, ir, lo, hi, ref = exposure_response_curve(m, di2, st["data"])
        c = CCOL.get(df_s, "#999")
        if df_s == chosen_df:
            ax.fill_between(wg, lo, hi, alpha=0.15, color=c)
            ax.plot(wg, ir, color=c, lw=2.0, label=f"df={df_s} (primary)")
        else:
            ax.plot(wg, ir, color=c, lw=1.1, ls="--", alpha=0.8, label=f"df={df_s}")
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.axvspan(cool, hot, alpha=0.07, color="#823038", zorder=0)
    ax.set_title(f"{INDICATOR_LABELS.get(ind, ind)} (df={chosen_df})",
                 fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7); ax.grid(ls=":", alpha=0.3); ax.legend(fontsize=6)
    if pi % NC == 0: ax.set_ylabel("IRR", fontsize=8)
    if pi >= len(fitted)-NC: ax.set_xlabel("WBGT (°C)", fontsize=8)
fig.savefig(f"{OUT_DIR}exposure_response_panel.png", dpi=180, bbox_inches="tight"); plt.close()
print(f"  -> {OUT_DIR}exposure_response_panel.png")


# ===========================================================================
# 10. HISTORICAL BURDEN
# ===========================================================================

print("\nHistorical burden...")

def compute_burden_analytical(model, di, data, reference_wbgt):
    """Compute the burden with analytical CIs from the cluster-robust covariance.

    The aggregate hot-months deficit % is derived from a count-weighted
    average of per-row log-IRRs. Because this is linear in beta, its SE
    comes directly from the delta method on the cluster-robust vcov —
    no bootstrap needed.
    """
    names = list(model.coef().index)
    beta  = model.coef().values
    vcov  = model._vcov

    actual_wbgt = data[WBGT_VAR].values
    obs = data["y_int"].values.astype(float)
    hmask = actual_wbgt > reference_wbgt

    # Build the spline basis at every observed WBGT and at the reference.
    # For hot months only — cool months contribute zero to the deficit.
    hot_idx = np.where(hmask)[0]

    if len(hot_idx) == 0:
        return {
            "obs": obs, "irr_h": np.ones(len(obs)), "hmask": hmask,
            "cf": obs.copy(), "imp": np.zeros(len(obs)),
            "pct_hot": 0.0, "pct_hot_lo": 0.0, "pct_hot_hi": 0.0,
            "aggregate_irr": 1.0, "aggregate_irr_lo": 1.0, "aggregate_irr_hi": 1.0,
        }

    hot_wbgt = actual_wbgt[hot_idx]
    hot_obs  = obs[hot_idx]

    # Build basis at all hot WBGT values + the reference, in one call.
    all_pts = np.append(hot_wbgt, reference_wbgt)
    basis = patsy.build_design_matrices(
        [di], pd.DataFrame({WBGT_VAR: all_pts}), return_type="dataframe"
    )[0]
    ref_row = basis.iloc[-1]       # the reference row
    hot_basis = basis.iloc[:-1]    # one row per hot facility-month

    # Per-row contrast vectors and log-IRRs.
    n_coefs = len(names)
    spline_cols = [n for n in names if n in basis.columns]

    # Build the per-row log-IRRs and the aggregate weighted contrast.
    irr_all = np.ones(len(obs))
    aggregate_contrast = np.zeros(n_coefs)
    total_hot_obs = hot_obs.sum()

    for row_i in range(len(hot_idx)):
        c = np.zeros(n_coefs)
        for j, nm in enumerate(names):
            if nm in basis.columns:
                c[j] = hot_basis.iloc[row_i][nm] - ref_row[nm]

        log_irr_i = float(c @ beta)
        irr_all[hot_idx[row_i]] = np.exp(log_irr_i)

        # Weight by observed count for the aggregate.
        aggregate_contrast += hot_obs[row_i] * c

    # The aggregate log-IRR is the count-weighted mean.
    if total_hot_obs > 0:
        aggregate_contrast /= total_hot_obs
    agg_log_irr = float(aggregate_contrast @ beta)
    agg_se = float(np.sqrt(aggregate_contrast @ vcov @ aggregate_contrast))

    agg_irr    = np.exp(agg_log_irr)
    agg_irr_lo = np.exp(agg_log_irr - 1.96 * agg_se)
    agg_irr_hi = np.exp(agg_log_irr + 1.96 * agg_se)

    # Deficit % = (1 - aggregate_IRR) * 100 when IRR < 1 (heat reduces services).
    # More precisely: counterfactual = obs / IRR, impact = cf - obs.
    pct_hot    = (1 - agg_irr)    * 100
    pct_hot_lo = (1 - agg_irr_hi) * 100   # note: inverted because lower IRR = bigger deficit
    pct_hot_hi = (1 - agg_irr_lo) * 100

    # Per-row counterfactual and impact (for saving and plotting).
    cf  = np.where(hmask, obs / irr_all, obs)
    imp = np.where(hmask, cf - obs, 0.0)

    return {
        "obs": obs, "irr_h": irr_all, "hmask": hmask, "cf": cf, "imp": imp,
        "pct_hot": pct_hot, "pct_hot_lo": pct_hot_lo, "pct_hot_hi": pct_hot_hi,
        "aggregate_irr": agg_irr, "aggregate_irr_lo": agg_irr_lo,
        "aggregate_irr_hi": agg_irr_hi,
    }


burden_summaries = []
for ind in fitted:
    st = indicator_state[ind]
    model, data, di = st["model"], st["data"], st["di"]

    result = compute_burden_analytical(model, di, data, REFERENCE_WBGT)
    obs, irr_h, hmask = result["obs"], result["irr_h"], result["hmask"]
    cf, imp = result["cf"], result["imp"]

    # Unconditional: across ALL facility-months.
    tot_imp = imp.sum(); tot_cf = cf.sum()
    pct_all = tot_imp / tot_cf * 100 if tot_cf > 0 else 0

    # Conditional: hot months only, with analytical CI.
    n_hot = hmask.sum(); n_total = len(hmask)
    pct_hot = result["pct_hot"]
    pct_hot_lo = result["pct_hot_lo"]
    pct_hot_hi = result["pct_hot_hi"]

    # Mean deficit per facility-month (only where there IS a deficit).
    deficit_mask = imp > 0
    n_deficit_months = deficit_mask.sum()
    mean_deficit_per_fm = imp[deficit_mask].mean() if n_deficit_months > 0 else 0
    median_deficit_per_fm = np.median(imp[deficit_mask]) if n_deficit_months > 0 else 0
    pct_per_fm = (imp[deficit_mask] / cf[deficit_mask] * 100) if n_deficit_months > 0 else np.array([0])
    mean_pct_per_fm = pct_per_fm.mean()

    print(f"  {ind}:")
    print(f"    all months:     {tot_imp:>12,.0f} total  ({pct_all:.2f}%)")
    print(f"    hot months:     {pct_hot:.2f}% [{pct_hot_lo:.2f}%, {pct_hot_hi:.2f}%]  "
          f"[{n_hot:,} of {n_total:,} above {REFERENCE_WBGT}°C]")
    print(f"    aggregate IRR:  {result['aggregate_irr']:.4f} "
          f"[{result['aggregate_irr_lo']:.4f}, {result['aggregate_irr_hi']:.4f}]")
    print(f"    deficit months: {n_deficit_months:,} facility-months with a loss")
    print(f"    mean deficit:   {mean_deficit_per_fm:.1f} appts/facility-month  "
          f"({mean_pct_per_fm:.2f}%)")

    burden_summaries.append({
        "indicator": ind,
        "total_impact": tot_imp,
        "pct_all_months": pct_all,
        "pct_hot_months_only": pct_hot,
        "pct_hot_ci_lo": pct_hot_lo,
        "pct_hot_ci_hi": pct_hot_hi,
        "aggregate_irr": result["aggregate_irr"],
        "aggregate_irr_lo": result["aggregate_irr_lo"],
        "aggregate_irr_hi": result["aggregate_irr_hi"],
        "n_hot_months": int(n_hot),
        "n_total_months": int(n_total),
        "n_deficit_months": int(n_deficit_months),
        "mean_deficit_per_fm": mean_deficit_per_fm,
        "median_deficit_per_fm": median_deficit_per_fm,
        "mean_pct_per_fm": mean_pct_per_fm,
        "pct_months_above_ref": n_hot / n_total * 100 if n_total > 0 else 0,
    })

    # Save facility-month detail.
    bdf = data[["facility","date","year","month",WBGT_VAR,"y_int"]].copy()
    bdf["irr"] = irr_h; bdf["counterfactual"] = cf; bdf["impact_count"] = imp
    bdf["above_reference"] = hmask
    bdf.to_csv(f"{OUT_DIR}historical_burden_{ind}.csv", index=False)

burden_df = pd.DataFrame(burden_summaries)
burden_df.to_csv(f"{OUT_DIR}historical_burden_summary.csv", index=False)

# --- Per-month deficit bar chart ------------------------------------------
# Shows the deficit by calendar month, so the seasonal concentration is visible.
print("\nPer-month deficit charts...")
n_ind = len(fitted); n_cols = 3; n_rows = int(np.ceil(n_ind / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3*n_rows), sharex=True)
af = axes.flatten() if n_ind > 1 else [axes]
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

for idx, ind in enumerate(fitted):
    ax = af[idx]
    bp = f"{OUT_DIR}historical_burden_{ind}.csv"
    if not os.path.exists(bp): continue
    df = pd.read_csv(bp)
    monthly = df.groupby("month").agg(
        impact=("impact_count", "sum"),
        counterfactual=("counterfactual", "sum"),
    ).reindex(range(1, 13))
    monthly["pct"] = monthly["impact"] / monthly["counterfactual"] * 100
    monthly["pct"] = monthly["pct"].fillna(0)

    colors = ["#823038" if p > 0 else "#2a78d6" for p in monthly["pct"]]
    ax.bar(range(12), monthly["pct"], color=colors, alpha=0.8)
    ax.set_xticks(range(12)); ax.set_xticklabels(month_names, fontsize=6, rotation=45)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(INDICATOR_LABELS.get(ind, ind), fontsize=9, fontweight="bold")
    ax.set_ylabel("% deficit", fontsize=7) if idx % n_cols == 0 else None
    ax.tick_params(labelsize=6)

for idx in range(n_ind, len(af)): af[idx].set_visible(False)
fig.suptitle(f"Heat deficit by calendar month (vs {REFERENCE_WBGT}°C)\n"
             "Red = services lost to heat; concentrated in hot months",
             fontsize=10, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}deficit_by_month.png", dpi=180, bbox_inches="tight"); plt.close()
print(f"  -> {OUT_DIR}deficit_by_month.png")

# --- Deficit forest plot: all indicators on one figure ---------------------
# Shows two dots per indicator: the "all months" % deficit (diluted, includes
# cool months where nothing happens) and the "hot months only" % deficit
# (meaningful, restricted to months above the reference WBGT).
print("\nDeficit forest plot...")
bdf = burden_df.sort_values("pct_hot_months_only").reset_index(drop=True)
y = np.arange(len(bdf))

fig, ax = plt.subplots(figsize=(8, max(4, len(bdf) * 0.55 + 1.5)))

# Bootstrap CIs on the hot-months deficit (if available).
has_ci = bdf["pct_hot_ci_lo"].notna().any()
if has_ci:
    for i in range(len(bdf)):
        row = bdf.iloc[i]
        if pd.notna(row["pct_hot_ci_lo"]):
            ax.plot([row["pct_hot_ci_lo"], row["pct_hot_ci_hi"]], [i, i],
                    color="#823038", lw=1.4, zorder=2, alpha=0.6)

# Hot-months-only dots (the meaningful number).
ci_label = " with 95% bootstrap CI" if has_ci else ""
ax.scatter(bdf["pct_hot_months_only"], y, color="#823038", s=60, zorder=3,
           label=f"Hot months only (WBGT > {REFERENCE_WBGT}°C){ci_label}")

# All-months dots (the diluted number).
ax.scatter(bdf["pct_all_months"], y, color="#888888", s=30, zorder=2, marker="D",
           label="All months (diluted)")

# Connect the two with a line so the dilution is visible.
for i in range(len(bdf)):
    ax.plot([bdf.iloc[i]["pct_all_months"], bdf.iloc[i]["pct_hot_months_only"]],
            [i, i], color="#cccccc", lw=1.0, zorder=1)

ax.axvline(0, color="black", ls="--", lw=0.9)
ax.set_yticks(y)
labels = [INDICATOR_LABELS.get(ind, ind) for ind in bdf["indicator"]]
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel(f"% of appointments attributable to heat (vs {REFERENCE_WBGT}°C)", fontsize=10)
ax.set_title("Heat-attributable deficit by indicator\n"
             "Red = hot months only; grey diamond = all months (diluted)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="best")
ax.grid(axis="x", ls=":", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}forest_plot_deficit.png", dpi=180, bbox_inches="tight"); plt.close()
print(f"  -> {OUT_DIR}forest_plot_deficit.png")


# ===========================================================================
# 11. FORWARD PROJECTION (CMIP6)
# ===========================================================================

print("\nForward projections...")
all_proj = []
for ind in fitted:
    st = indicator_state[ind]
    model, data, di = st["model"], st["data"], st["di"]
    v = WBGT_VAR
    hc = data.groupby(["facility","month"])[v].mean().reset_index().rename(columns={v:"wbgt_hist"})
    hcounts = data.groupby(["facility","month"])["y_int"].mean().reset_index().rename(columns={"y_int":"baseline"})

    for ssp in SSP_SCENARIOS:
        for tier in MODEL_TIERS:
            pf_path = f"{PROJECTION_DIR}/{ssp}/{tier}.csv"
            if not os.path.exists(pf_path):
                print(f"  {ind} {ssp}/{tier}: not found — skipping")
                continue
            proj = pd.read_csv(pf_path)
            proj["facility"] = proj["facility"].astype(str)
            if v not in proj.columns: continue
            proj = proj.rename(columns={v: "wbgt_proj"})
            proj = proj.merge(hc, on=["facility","month"], how="inner")
            proj = proj.merge(hcounts, on=["facility","month"], how="inner")
            if proj.empty: continue

            irr_p = compute_row_irrs(model, di, proj["wbgt_hist"].values, proj["wbgt_proj"].values)
            proj["irr"] = irr_p
            proj["proj_count"] = proj["baseline"] * irr_p
            proj["change"] = proj["proj_count"] - proj["baseline"]
            proj["pct_change"] = proj["change"] / proj["baseline"] * 100

            tc = proj["change"].sum(); tb = proj["baseline"].sum()
            pct = tc/tb*100 if tb > 0 else 0
            wd = (proj["wbgt_proj"]-proj["wbgt_hist"]).mean()
            print(f"  {ind} {ssp}/{tier}: dWBGT={wd:+.2f}, change={tc:+,.0f} ({pct:+.1f}%)")

            proj["indicator"]=ind; proj["ssp"]=ssp; proj["tier"]=tier
            proj.to_csv(f"{OUT_DIR}projection_{ind}_{ssp}_{tier}.csv", index=False)
            all_proj.append({"indicator":ind,"ssp":ssp,"tier":tier,
                "n_fac":len(proj),"mean_wbgt_diff":wd,"total_change":tc,"pct_change":pct})

if all_proj:
    pd.DataFrame(all_proj).to_csv(f"{OUT_DIR}projection_summary.csv", index=False)


# ===========================================================================
# 11b. CMIP6 PROJECTION MAPS (SSP × tier grid per indicator)
# ===========================================================================
# For each indicator, aggregate facility-level projections to district and
# plot a 3×3 grid of maps: rows = SSP scenarios, columns = model tiers.
# This produces the figure from your reference image.

print("\nProjection maps...")

# We need the facility-to-district mapping from the original panel.
for ind in fitted:
    panel_path = f"{DATA_DIR}/All_predictors_processed/regression_panel_{ind}.csv"
    if not os.path.exists(panel_path):
        continue
    panel = pd.read_csv(panel_path)
    if "Dist" not in panel.columns:
        print(f"  {ind}: no Dist column — skipping projection maps.")
        continue

    fac_dist = panel[["facility", "Dist"]].drop_duplicates()
    fac_dist["facility"] = fac_dist["facility"].astype(str)

    # Collect district-level results for each SSP × tier.
    grid_data = {}
    for ssp in SSP_SCENARIOS:
        for tier in MODEL_TIERS:
            proj_path = f"{OUT_DIR}projection_{ind}_{ssp}_{tier}.csv"
            if not os.path.exists(proj_path):
                continue
            proj = pd.read_csv(proj_path)
            proj["facility"] = proj["facility"].astype(str)
            proj = proj.merge(fac_dist, on="facility", how="left")

            dist = proj.groupby("Dist").agg(
                baseline=("baseline", "sum"),
                proj_count=("proj_count", "sum"),
                change=("change", "sum"),
            ).reset_index()
            dist["pct_change"] = dist["change"] / dist["baseline"] * 100
            grid_data[(ssp, tier)] = dist

    if not grid_data:
        print(f"  {ind}: no projection data available for maps.")
        continue

    # Save a combined district projection table.
    dist_rows = []
    for (ssp, tier), dist in grid_data.items():
        d = dist.copy()
        d["ssp"] = ssp; d["tier"] = tier; d["indicator"] = ind
        dist_rows.append(d)
    if dist_rows:
        pd.concat(dist_rows, ignore_index=True).to_csv(
            f"{OUT_DIR}projection_district_{ind}.csv", index=False)
    if SHAPEFILE_PATH and os.path.exists(SHAPEFILE_PATH):
        import geopandas as gpd

        gdf = gpd.read_file(SHAPEFILE_PATH)
        for res in all_results:
            ind = res["indicator"]
            csv = f"{OUT_DIR}district_burden_{ind}.csv"
            if not os.path.exists(csv):
                continue
            d = pd.read_csv(csv)
            merged = gdf.merge(d, left_on=DISTRICT_NAME_COL, right_on=CLUSTER_COL, how="left")
            fig, ax = plt.subplots(figsize=(6, 8))
            merged.plot(
                column="deficit_pct",
                cmap="RdBu_r",
                linewidth=0.5,
                edgecolor="black",
                legend=True,
                ax=ax,
                missing_kwds={"color": "lightgrey", "hatch": "///"},
            )
            ax.set_title(
                f"{INDICATOR_LABELS.get(ind, ind)}\nHistorical two-model deficit by district",
                fontsize=11,
                fontweight="bold",
            )
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}map_burden_{ind}.png", dpi=180, bbox_inches="tight")
            plt.close()
            print(f"  {ind}: -> map_burden_{ind}.png")
    # --- Plot the grid (without shapefile: bar chart; with shapefile: choropleth) ---

    if SHAPEFILE_PATH and os.path.exists(SHAPEFILE_PATH):
        import geopandas as gpd
        gdf = gpd.read_file(SHAPEFILE_PATH)

        n_ssp = len(SSP_SCENARIOS); n_tier = len(MODEL_TIERS)
        fig, axes = plt.subplots(n_ssp, n_tier, figsize=(5*n_tier, 6*n_ssp))
        if n_ssp == 1 and n_tier == 1:
            axes = np.array([[axes]])
        elif n_ssp == 1:
            axes = axes[np.newaxis, :]
        elif n_tier == 1:
            axes = axes[:, np.newaxis]

        # Find a common colour range across all panels.
        all_pct = [d["pct_change"].values for d in grid_data.values()]
        if all_pct:
            all_pct = np.concatenate(all_pct)
            vmin = np.nanpercentile(all_pct, 2)
            vmax = np.nanpercentile(all_pct, 98)
            vabs = max(abs(vmin), abs(vmax))
        else:
            vabs = 5

        for row_i, ssp in enumerate(SSP_SCENARIOS):
            for col_i, tier in enumerate(MODEL_TIERS):
                ax = axes[row_i, col_i]
                key = (ssp, tier)
                if key not in grid_data:
                    ax.set_title(f"{ssp}: {tier}\n(no data)", fontsize=9)
                    ax.axis("off")
                    continue

                dist = grid_data[key]
                merged = gdf.merge(dist, left_on=DISTRICT_NAME_COL,
                                   right_on="Dist", how="left")
                merged.plot(column="pct_change", cmap="RdBu_r",
                            linewidth=0.5, edgecolor="black",
                            legend=False, ax=ax, vmin=-vabs, vmax=vabs,
                            missing_kwds={"color": "lightgrey", "hatch": "///"})

                mean_pct = dist["pct_change"].mean()
                sd_pct = dist["pct_change"].std()
                ax.set_title(f"{ssp}: {tier}", fontsize=10)
                ax.text(0.02, 0.02, f"Mean: {mean_pct:.2f}%\nSD: {sd_pct:.2f}%",
                        transform=ax.transAxes, fontsize=7, verticalalignment="bottom",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                ax.axis("off")

        # Add a shared colorbar.
        sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                    norm=plt.Normalize(vmin=-vabs, vmax=vabs))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.6, aspect=30, pad=0.02)
        cbar.set_label("Projected % change in services", fontsize=10)

        label = INDICATOR_LABELS.get(ind, ind)
        fig.suptitle(f"{label}\nProjected change under CMIP6 scenarios",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.savefig(f"{OUT_DIR}projection_map_grid_{ind}.png",
                    dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  {ind}: map grid saved -> projection_map_grid_{ind}.png")

    else:
        # No shapefile: produce a summary table plot instead.
        n_ssp = len(SSP_SCENARIOS); n_tier = len(MODEL_TIERS)
        summary_grid = pd.DataFrame(index=SSP_SCENARIOS, columns=MODEL_TIERS, dtype=float)
        for (ssp, tier), dist in grid_data.items():
            summary_grid.loc[ssp, tier] = dist["pct_change"].mean()

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(summary_grid.values.astype(float), cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(n_tier)); ax.set_xticklabels(MODEL_TIERS, fontsize=9)
        ax.set_yticks(range(n_ssp)); ax.set_yticklabels(SSP_SCENARIOS, fontsize=9)

        # Annotate cells.
        for i in range(n_ssp):
            for j in range(n_tier):
                val = summary_grid.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}%", ha="center", va="center", fontsize=10)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean % change", fontsize=9)
        label = INDICATOR_LABELS.get(ind, ind)
        ax.set_title(f"{label}\nMean projected % change by scenario",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}projection_heatmap_{ind}.png", dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  {ind}: heatmap saved -> projection_heatmap_{ind}.png")
        print(f"    (set SHAPEFILE_PATH for choropleth maps)")


# ===========================================================================
# 12. TIME-SERIES: OBSERVED vs COUNTERFACTUAL
# ===========================================================================

print("\nTime-series plots...")
nc = 3; ni = len(fitted); nr = int(np.ceil(ni/nc))
fig, axes = plt.subplots(nr, nc, figsize=(5.5*nc, 3.5*nr), sharex=True)
af = axes.flatten() if ni > 1 else [axes]
for idx, ind in enumerate(fitted):
    ax = af[idx]
    p = f"{OUT_DIR}historical_burden_{ind}.csv"
    if not os.path.exists(p): continue
    df = pd.read_csv(p, parse_dates=["date"])
    m = df.groupby("date").agg(obs=("y_int","sum"), cf=("counterfactual","sum")).sort_index()
    ax.plot(m.index, m.cf, color="#2a78d6", lw=1.0, ls="--", alpha=0.8, label="Without heat")
    ax.plot(m.index, m.obs, color="#333", lw=1.0, label="Observed")
    ax.fill_between(m.index, m.obs, m.cf, where=m.obs<m.cf, color="#823038", alpha=0.25, label="Lost")
    ax.fill_between(m.index, m.obs, m.cf, where=m.obs>m.cf, color="#2a78d6", alpha=0.25, label="Added")
    ax.set_title(INDICATOR_LABELS.get(ind,ind), fontsize=9, fontweight="bold"); ax.tick_params(labelsize=7)
    if idx == 0: ax.legend(fontsize=6)
for idx in range(ni, len(af)): af[idx].set_visible(False)
fig.suptitle(f"Observed vs counterfactual (ref={REFERENCE_WBGT}°C)", fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}observed_vs_counterfactual.png", dpi=180, bbox_inches="tight"); plt.close()


# ===========================================================================
# 13. DISTRICT AGGREGATION + MAP
# ===========================================================================

print("\nDistrict aggregation...")
all_dist = []
for ind in fitted:
    bp = f"{OUT_DIR}historical_burden_{ind}.csv"
    pp = f"{DATA_DIR}/All_predictors_processed/regression_panel_{ind}.csv"
    if not os.path.exists(bp) or not os.path.exists(pp): continue
    burden = pd.read_csv(bp); panel = pd.read_csv(pp)
    if "Dist" not in panel.columns:
        print(f"  {ind}: no Dist column — skipping map."); continue
    fd = panel[["facility","Dist"]].drop_duplicates()
    burden = burden.merge(fd, on="facility", how="left")
    d = burden.groupby("Dist").agg(obs=("y_int","sum"),cf=("counterfactual","sum"),
                                   imp=("impact_count","sum")).reset_index()
    d["pct_disruption"] = d["imp"]/d["cf"]*100; d["indicator"] = ind
    all_dist.append(d); d.to_csv(f"{OUT_DIR}district_burden_{ind}.csv", index=False)
    print(f"  {ind}: {len(d)} districts, mean {d['pct_disruption'].mean():.2f}%")

    if SHAPEFILE_PATH and os.path.exists(SHAPEFILE_PATH):
        import geopandas as gpd
        gdf = gpd.read_file(SHAPEFILE_PATH).merge(d, left_on=DISTRICT_NAME_COL, right_on="Dist", how="left")
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        gdf.plot(column="pct_disruption", cmap="RdBu_r", linewidth=0.5, edgecolor="black",
                 legend=True, ax=ax, missing_kwds={"color":"lightgrey","hatch":"///"})
        ax.set_title(f"{INDICATOR_LABELS.get(ind,ind)}\n% disrupted (vs {REFERENCE_WBGT}°C)",
                     fontsize=11, fontweight="bold"); ax.axis("off"); plt.tight_layout()
        plt.savefig(f"{OUT_DIR}map_burden_{ind}.png", dpi=180, bbox_inches="tight"); plt.close()

# -----------------------------------------------------------------------
    # HISTORICAL BURDEN MAPS (per indicator)
    # -----------------------------------------------------------------------
    if SHAPEFILE_PATH and os.path.exists(SHAPEFILE_PATH):
        import geopandas as gpd
        gdf = gpd.read_file(SHAPEFILE_PATH)

        for ind in fitted:
            csv = f"{OUT_DIR}district_burden_{ind}.csv"
            if not os.path.exists(csv):
                continue
            d = pd.read_csv(csv)
            merged = gdf.merge(
                d, left_on=DISTRICT_NAME_COL, right_on=CLUSTER_COL, how="left")

            vabs = float(np.nanpercentile(np.abs(d["deficit_pct"]), 98))

            fig, ax = plt.subplots(figsize=(6, 8))
            merged.plot(column="deficit_pct", cmap="RdBu_r",
                        vmin=-vabs, vmax=vabs,
                        linewidth=0.5, edgecolor="black", legend=True, ax=ax,
                        missing_kwds={"color": "lightgrey", "hatch": "///"})
            ax.set_title(
                f"{INDICATOR_LABELS.get(ind, ind)}\n"
                "Historical two-model deficit by district (%)",
                fontsize=11, fontweight="bold")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}map_burden_{ind}.png",
                        dpi=180, bbox_inches="tight")
            plt.close()
            print(f"  {ind}: -> map_burden_{ind}.png")

    # -----------------------------------------------------------------------
    # PROJECTION MAPS (SSP × tier grid per indicator)
    # -----------------------------------------------------------------------
    if SHAPEFILE_PATH and os.path.exists(SHAPEFILE_PATH):
        print("\nProjection maps...")
        for res in all_results:
            ind = res["indicator"]
            fac_dist = res["_nb_data"][["facility", CLUSTER_COL]].drop_duplicates()
            fac_dist["facility"] = fac_dist["facility"].astype(str)

            grid_data = {}
            for ssp in SSP_SCENARIOS:
                for tier in MODEL_TIERS:
                    proj_path = f"{OUT_DIR}projection_{ind}_{ssp}_{tier}.csv"
                    if not os.path.exists(proj_path):
                        continue
                    proj = pd.read_csv(proj_path)
                    proj["facility"] = proj["facility"].astype(str)
                    proj = proj.merge(fac_dist, on="facility", how="left")
                    if CLUSTER_COL not in proj.columns:
                        continue
                    dist = (proj.groupby(CLUSTER_COL)
                            .agg(mu_a_proj=("mu_a_proj", "sum"),
                                 mu_b_proj=("mu_b_proj", "sum"))
                            .reset_index())
                    dist["delta_deficit"] = 100 * (dist["mu_a_proj"] - dist["mu_b_proj"]) / dist["mu_b_proj"]
                    grid_data[(ssp, tier)] = dist

            if not grid_data:
                continue

            n_ssp, n_tier = len(SSP_SCENARIOS), len(MODEL_TIERS)
            fig, axes = plt.subplots(n_ssp, n_tier, figsize=(5*n_tier, 6*n_ssp))
            axes = np.atleast_2d(axes)

            all_pct = np.concatenate([d["delta_deficit"].values for d in grid_data.values()])
            vabs = float(np.nanpercentile(np.abs(all_pct), 98))

            for row_i, ssp in enumerate(SSP_SCENARIOS):
                for col_i, tier in enumerate(MODEL_TIERS):
                    ax = axes[row_i, col_i]
                    key = (ssp, tier)
                    if key not in grid_data:
                        ax.set_title(f"{ssp}: {tier}\n(no data)", fontsize=9)
                        ax.axis("off")
                        continue
                    dist = grid_data[key]
                    merged = gdf.merge(dist, left_on=DISTRICT_NAME_COL,
                                       right_on=CLUSTER_COL, how="left")
                    merged.plot(column="delta_deficit", cmap="RdBu_r",
                                vmin=-vabs, vmax=vabs,
                                linewidth=0.5, edgecolor="black",
                                legend=False, ax=ax,
                                missing_kwds={"color": "lightgrey", "hatch": "///"})
                    ax.set_title(f"{ssp}: {tier}\nmean {dist['delta_deficit'].mean():+.2f}%",
                                 fontsize=10)
                    ax.axis("off")

            sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                        norm=plt.Normalize(vmin=-vabs, vmax=vabs))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, shrink=0.6, aspect=30, pad=0.02)
            cbar.set_label("Projected two-model deficit (%)", fontsize=10)

            fig.suptitle(f"{INDICATOR_LABELS.get(ind, ind)}\n"
                          "District-level projected deficit under CMIP6",
                          fontsize=12, fontweight="bold", y=1.02)
            plt.savefig(f"{OUT_DIR}projection_map_grid_{ind}.png",
                        dpi=180, bbox_inches="tight")
            plt.close()
            print(f"  {ind}: -> projection_map_grid_{ind}.png")
# ===========================================================================
# 14. TLO LOOKUP TABLE
# ===========================================================================

print("\nTLO lookup tables...")
tlo_rows = []
for ind in fitted:
    st = indicator_state[ind]
    model, di = st["model"], st["di"]
    names, beta, vcov = list(model.coef().index), model.coef().values, model._vcov
    all_pts = np.append(WBGT_GRID, REFERENCE_WBGT)
    basis = patsy.build_design_matrices(
        [di], pd.DataFrame({WBGT_VAR: all_pts}), return_type="dataframe")[0]
    ref_row = basis.iloc[-1]
    for i, w in enumerate(WBGT_GRID):
        c = np.zeros(len(names))
        for j, nm in enumerate(names):
            if nm in basis.columns: c[j] = basis.iloc[i][nm] - ref_row[nm]
        li = float(c @ beta); se = float(np.sqrt(c @ vcov @ c))
        irr = np.exp(li)
        tlo_rows.append({"indicator":ind, "label":INDICATOR_LABELS.get(ind,ind),
            "wbgt":w, "irr":irr, "irr_lo":np.exp(li-1.96*se), "irr_hi":np.exp(li+1.96*se),
            "disruption_probability":max(0.0,1.0-irr), "demand_multiplier":max(1.0,irr)})
    print(f"  {ind}: {len(WBGT_GRID)} values")

tlo = pd.DataFrame(tlo_rows)
tlo.to_csv(f"{OUT_DIR}tlo_wbgt_lookup.csv", index=False)
for col, nm in [("disruption_probability","disruption"),("demand_multiplier","demand")]:
    tlo.pivot(index="wbgt", columns="indicator", values=col).to_csv(f"{OUT_DIR}tlo_{nm}_wide.csv")

fig, ax = plt.subplots(figsize=(8, 5))
for ind in tlo["indicator"].unique():
    sub = tlo[tlo["indicator"]==ind]
    ax.plot(sub.wbgt, sub.disruption_probability, lw=1.3, label=INDICATOR_LABELS.get(ind,ind))
ax.set_xlabel("WBGT (°C)"); ax.set_ylabel(f"Disruption prob (vs {REFERENCE_WBGT}°C)")
ax.set_title("Heat disruption for TLO model", fontsize=11, fontweight="bold")
ax.legend(fontsize=7); ax.grid(ls=":", alpha=0.4); ax.set_ylim(bottom=-0.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}tlo_disruption_curves.png", dpi=180, bbox_inches="tight"); plt.close()

print(f"\nAll outputs in {OUT_DIR}")
print("Done.")
