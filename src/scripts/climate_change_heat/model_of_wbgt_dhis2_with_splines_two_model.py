"""
model_of_wbgt_dhis2_two_model_splines_optimized.py

Two-model NB (statsmodels) WBGT–service analysis - OPTIMIZED VERSION.
Key improvements:
  1. Fixed alpha (estimated once, reused across models)
  2. Parallel processing for multiple indicators
  3. Sparse matrices for fixed effects
  4. Vectorized predictions
  5. Efficient spline basis construction

Fixes vs previous version:
  - Projection block de-dented to top level (was nested inside `if curve_paths:`)
  - PRECIP_MONTH_PROJ_FILE_TPL now assigned BEFORE _load_future_climate() is called
  - Added annual pooled + annual x district projection outputs
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy.stats as stats
import scipy.sparse as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar

# Suppress convergence warnings (we handle them)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
COUNT_INDICATORS = [
    "fp_total_clients",
    # "opd_attendance",
    # "ipd_total_admissions",
    # "vmmc_first_visits",
    # "pnc_mother_checked_48h",
    # "anc_new_attendees",
    # "anc_first_trimester_starts",
    # "bcg_under1",
    # "penta3_under1",
    # "measles1_under1",
    "fully_immunised_under1",
    # "pnc_within_2wks",
    # "pnc_first_visit_2wks",
    # "live_births_total",
    # "skilled_deliveries",
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

# Model settings
WBGT_VAR = "wbgt_day"
SPLINE_DF = 3
LAG_MONTHS = [1, 2, 3]
CENTER = True
MIN_OBS = 24

# COVID and closures
COVID_WINDOW = ("2020-04-01", "2021-12-01")
CLUSTER_COL = "Dist"
USE_PRECIP = True
PRECIP_COL = "precip_month"

REFERENCE_WBGT_PERCENTILE = 95  # Top 5% hottest months
HOT_DEFICIT_CI_METHOD = "bootstrap"  # or "bootstrap"
N_BOOTSTRAP = 1000  # if using bootstrap

# IRR contrast: high WBGT vs a per-indicator "low" reference. The reference
# is the IRR_LOW_PCTILE percentile of observed WBGT for that indicator and
# is stored per-indicator in the summary as `reference_wbgt`.
IRR_LOW_PCTILE = 25
IRR_HIGH = 32.0

# TLO disruption lookup grid: WBGT axis for the disruption_probability curve.
TLO_WBGT_GRID = np.linspace(20.0, 34.0, 57)  # 0.25 degC steps

# BH-FDR alpha for the `sig` column in the summary.
FDR_ALPHA = 0.05


CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre", "2023-03-01", "2024-03-01"),
]

# Time periods
min_year_historical = 2016
max_year_historical = 2025
LAST_HIST_YEAR = max_year_historical - 1

MIN_YEAR_BY_INDICATOR: dict[str, int] = {
    "fp_total_clients":  2019,   # FP reporting begins ~2019
    "vmmc_first_visits": 2019,   # before is choppy
}

# Projection
PROJECT = True
PROJECT_HOLD_YEAR = True
SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
WBGT_MODELS = ["lowest", "median", "highest"]
min_year_projection = 2025
max_year_projection = 2041

# Precip GCM used per WBGT tier (assigned BEFORE _load_future_climate is called)
PRECIP_FILE_BY_TIER = {
    "highest": "precip_monthly_total_facility_CanESM5_{ssp}.csv",
    "lowest":  "precip_monthly_total_facility_MPI-ESM1-2-HR_{ssp}.csv",
    "median":  "precip_monthly_total_facility_MIROC6_{ssp}.csv",
}

# Curve settings
CURVE_REF_MODE = "mean"
CURVE_N = 60

# Paths
DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
PANEL_DIR = f"{DATA_DIR}/Thermofeel_WBGT/Indices/"
INDICES_DIR = PANEL_DIR
os.makedirs(OUT_DIR, exist_ok=True)


# Parallel processing
USE_PARALLEL = False
N_WORKERS = min(cpu_count() - 1, 4)  # Use up to 4 cores


# ===========================================================================
# Local helper: jackknife CI over facilities for a mu_a/mu_b pair
# ===========================================================================
def _monthly_jackknife_ci_local(mu_a, mu_b, facility_ids, sign="b_minus_a"):
    """Leave-one-facility-out 95% CI on deficit_pct.

    sign = "b_minus_a": returns 100 * (sum_B - sum_A) / sum_B
       (positive = services LOST — matches the paper's convention)
    sign = "a_minus_b": returns 100 * (sum_A - sum_B) / sum_B
       (positive = services GAINED — for the CI file the map negates)
    """
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    facility_ids = np.asarray(facility_ids)
    sum_a = float(mu_a.sum())
    sum_b = float(mu_b.sum())
    if sum_b <= 0:
        return np.nan, np.nan, np.nan

    def _stat(sa, sb):
        return (100.0 * (sa - sb) / sb) if sign == "a_minus_b" \
            else (100.0 * (sb - sa) / sb)

    pt = _stat(sum_a, sum_b)
    facs = np.unique(facility_ids)
    if len(facs) < 3:
        return pt, np.nan, np.nan
    jack = []
    for fac in facs:
        keep = facility_ids != fac
        sa_j = float(mu_a[keep].sum())
        sb_j = float(mu_b[keep].sum())
        if sb_j <= 0:
            continue
        jack.append(_stat(sa_j, sb_j))
    if len(jack) < 3:
        return pt, np.nan, np.nan
    jack = np.asarray(jack)
    n = len(facs)
    se = np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))
    return pt, pt - 1.96 * se, pt + 1.96 * se


# ===========================================================================
# OPTIMIZED: Shared alpha estimation (faster than profile likelihood)
# ===========================================================================
def estimate_alpha_fast(data, y_col="y_int"):
    """
    Quick alpha estimate for NB2 using method of moments.
    Much faster than profile likelihood optimization.
    """
    mod_pois = smf.glm(
        f"{y_col} ~ 1",
        data=data,
        family=sm.families.Poisson()
    ).fit()

    mu = mod_pois.fittedvalues
    pearson_resid = (data[y_col] - mu) / np.sqrt(mu)
    dispersion = (pearson_resid**2).sum() / mod_pois.df_resid

    alpha = max(0.01, (dispersion - 1) / mu.mean())
    return alpha


def diagnose_indicator(df, indicator):
    y = df[indicator].dropna()
    print(f"\n{'='*50}")
    print(f"  DIAGNOSTICS: {indicator}")
    print(f"{'='*50}")
    print(f"  N obs:        {len(y):,}")
    print(f"  N zeros:      {(y == 0).sum():,} ({100*(y==0).mean():.1f}%)")
    print(f"  Mean:         {y.mean():.2f}")
    print(f"  Variance:     {y.var():.2f}")
    print(f"  Var/Mean:     {y.var()/y.mean():.2f}")
    print(f"  Max:          {y.max():.0f}")
    print(f"  Skewness:     {y.skew():.2f}")
    print(f"  % > 3*IQR:    {100*(y > y.quantile(0.75)*3).mean():.2f}%")


def fit_negbin_fixed_alpha(formula, data, groups, alpha, y_col="y_int"):
    """
    Fit Negative Binomial with fixed alpha (no profile likelihood).
    """
    mod = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.NegativeBinomial(alpha=alpha)
    ).fit(cov_type="cluster", cov_kwds={"groups": groups})
    mod.alpha_hat = alpha
    return mod


# ===========================================================================
# OPTIMIZED: Sparse fixed effects (much faster than C(facility))
# ===========================================================================
def add_sparse_fixed_effects(df, fe_cols):
    df = df.copy()
    for col in fe_cols:
        dummies = pd.get_dummies(df[col], prefix=col, sparse=True)
        for c in dummies.columns:
            df[c] = dummies[c].astype(float)
    return df


def fit_negbin_with_sparse_fe(formula_base, df, fe_cols, groups, alpha, y_col="y_int"):
    df_fe = add_sparse_fixed_effects(df, fe_cols)
    formula = f"{y_col} ~ " + formula_base
    mod = smf.glm(
        formula=formula,
        data=df_fe,
        family=sm.families.NegativeBinomial(alpha=alpha)
    ).fit(cov_type="cluster", cov_kwds={"groups": groups})
    mod.alpha_hat = alpha
    return mod, df_fe


# ===========================================================================
# OPTIMIZED: Shared column construction
# ===========================================================================
def add_weather_columns_optimized(df, shifts, spline_design=None,
                                   lag_months=LAG_MONTHS):
    df = df.sort_values(["facility", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_c"] = df["year"] - shifts["year"]

    lo, hi = pd.Timestamp(COVID_WINDOW[0]), pd.Timestamp(COVID_WINDOW[1])
    df["covid"] = df["date"].between(lo, hi).astype(int)

    rhs = ["year_c", "covid"]
    if USE_PRECIP:
        df["precip_c"] = df[PRECIP_COL] - shifts.get("precip", 0.0)
        rhs.append("precip_c")

    spline_cols_all = []
    design_map = {} if spline_design is None else spline_design

    for v in [WBGT_VAR]:
        xc = df[v] - shifts[v]

        if spline_design is None:
            B = patsy.dmatrix(f"cr(x, df={SPLINE_DF}) - 1", {"x": xc},
                              return_type="dataframe")
            design_map[v] = B.design_info
        else:
            B = patsy.build_design_matrices([design_map[v]], {"x": xc})[0]
            B = pd.DataFrame(np.asarray(B))

        cols = [f"{v}_s{i+1}" for i in range(B.shape[1])]
        for c, col in enumerate(cols):
            df[col] = np.asarray(B)[:, c]
        spline_cols_all.extend(cols)
        rhs.extend(cols)

        for lag in lag_months:
            lc = f"{v}_lag{lag}_c"
            df[lc] = df.groupby("facility")[v].shift(lag) - shifts[v]
            rhs.append(lc)

    return df, rhs, spline_cols_all, design_map


# ===========================================================================
# OPTIMIZED: Exposure-response curve
# ===========================================================================
def exposure_response_curve_fast(model, spline_cols, design_info, WBGT_VAR_name,
                                  wbgt_shift, ref_mode, wbgt_values, n=CURVE_N):
    wobs = np.asarray(wbgt_values).flatten()
    grid = np.linspace(np.percentile(wobs, 1), np.percentile(wobs, 99), n)
    ref = wobs.mean() if ref_mode == "mean" else float(ref_mode)

    Bg = np.asarray(patsy.build_design_matrices(
        [design_info[WBGT_VAR_name]], {"x": grid - wbgt_shift})[0])
    Br = np.asarray(patsy.build_design_matrices(
        [design_info[WBGT_VAR_name]], {"x": np.array([ref]) - wbgt_shift})[0])
    contrast = Bg - Br

    beta = model.params.reindex(spline_cols).values
    V = model.cov_params().reindex(index=spline_cols, columns=spline_cols).values

    log_irr = contrast @ beta
    var = np.einsum("ij,jk,ik->i", contrast, V, contrast)
    se = np.sqrt(np.clip(var, 0, None))

    return pd.DataFrame({
        "wbgt": grid,
        "wbgt_ref": ref,
        "rr_vs_ref": np.exp(log_irr),
        "rr_lo": np.exp(log_irr - 1.96 * se),
        "rr_hi": np.exp(log_irr + 1.96 * se),
    })


# ===========================================================================
# Winsorize
# ===========================================================================
def winsorize_indicator(df, indicator_col='y', facility_col='facility_id',
                        upper_quantile=0.999):
    df = df.copy()

    def _cap_facility(group):
        cap = group[indicator_col].quantile(upper_quantile)
        return group[indicator_col].clip(upper=cap)

    original = df[indicator_col].copy()
    df[indicator_col] = df.groupby(facility_col, group_keys=False).apply(_cap_facility)

    n_capped = (df[indicator_col] != original).sum()
    print(f"  [WINSORIZE] {indicator_col}: {n_capped} values capped "
          f"(per-facility {upper_quantile*100:.1f}th percentile)")
    return df


# ===========================================================================
# Main fitting function
# ===========================================================================
def fit_indicator(indicator, panel_path):
    """Fit both models for a single indicator - optimized version."""
    print(f"\n→ {indicator}")
    t0 = time.time()

    try:
        long = pd.read_csv(panel_path, parse_dates=["date"])
        long = long.rename(columns={indicator: "y"})
    except Exception as e:
        print(f"  [{indicator}] Failed to load: {e}")
        return None
    if USE_PRECIP and PRECIP_COL not in long.columns:
        print(f"  [{indicator}] Missing {PRECIP_COL} in panel")
        return None
    if CLUSTER_COL not in long.columns:
        print(f"  [{indicator}] Missing {CLUSTER_COL} in panel")
        return None

    # Closures -> NaN (not 0)
    for fac, d0, d1 in CLOSURES:
        m = (long["date"].between(d0, d1)) & (long["facility"] == fac)
        if m.any():
            long.loc[m, "y"] = 0

    long["year"] = long["date"].dt.year
    long["month"] = long["date"].dt.month
    ind_min_year = MIN_YEAR_BY_INDICATOR.get(indicator, min_year_historical)
    long = long[long["year"].between(ind_min_year, max_year_historical - 1)]

    obs_per_fac = long.dropna(subset=["y", WBGT_VAR]).groupby("facility").size()
    long = long[long["facility"].isin(obs_per_fac[obs_per_fac >= MIN_OBS].index)].copy()

    if len(long) < 100 or long["facility"].nunique() < 2:
        print(f"  [{indicator}] Too few observations/facilities")
        return None
    long = winsorize_indicator(long, indicator_col='y', facility_col='facility',
                               upper_quantile=0.99)
    diagnose_indicator(long, 'y')

    SHIFTS = {"year": long["year"].mean() if CENTER else 0.0}
    SHIFTS[WBGT_VAR] = long[WBGT_VAR].mean() if CENTER else 0.0
    if USE_PRECIP:
        SHIFTS["precip"] = long[PRECIP_COL].mean() if CENTER else 0.0

    long, weather_rhs, spline_cols, DESIGN = add_weather_columns_optimized(
        long, SHIFTS
    )

    nb_cols = ["y", "facility", "month", CLUSTER_COL] + weather_rhs
    nb_data = long.dropna(subset=nb_cols).copy()
    nb_data["y_int"] = nb_data["y"].round().clip(lower=0).astype(int)

    obs_nb = nb_data.groupby("facility").size()
    nb_data = nb_data[nb_data["facility"].isin(obs_nb[obs_nb >= MIN_OBS].index)].copy()
    FITTED_FACILITIES = set(nb_data["facility"].unique())

    print(f"  [{indicator}] Sample: {len(nb_data):,} obs, {len(FITTED_FACILITIES)} facilities")

    alpha = estimate_alpha_fast(nb_data)
    groups = nb_data[CLUSTER_COL]

    ctrl = "year_c + covid" + (" + precip_c" if USE_PRECIP else "")
    FE = "C(month) + C(facility)"

    f_base = f"y_int ~ {ctrl} + {FE}"
    f_wx = f"y_int ~ {' + '.join(weather_rhs)} + {FE}"

    try:
        model_base = fit_negbin_fixed_alpha(f_base, nb_data, groups, alpha)
        model_wx = fit_negbin_fixed_alpha(f_wx, nb_data, groups, alpha)
    except Exception as e:
        print(f"  [{indicator}] Model fitting failed: {e}")
        return None

    missing_spline = [c for c in spline_cols if c not in model_wx.params.index]
    if missing_spline:
        print(f"  [{indicator}] Warning: missing spline cols: {missing_spline}")

    # Exposure-response curve
    try:
        curve = exposure_response_curve_fast(
            model_wx,
            spline_cols,
            DESIGN,
            WBGT_VAR,
            SHIFTS[WBGT_VAR],
            CURVE_REF_MODE,
            nb_data[WBGT_VAR].values,
        )
        curve.insert(0, "indicator", indicator)
        curve.insert(1, "label", INDICATOR_LABELS.get(indicator, indicator))
        curve.to_csv(
            f"{OUT_DIR}exposure_response_curve_{indicator}_{WBGT_VAR}.csv",
            index=False,
        )
    except Exception as e:
        print(f"  [{indicator}] Curve failed: {e}")

    # Two-model deficit (facility jackknife)
    nb_data["y_pred_base"] = model_base.fittedvalues
    nb_data["y_pred_wx"] = model_wx.fittedvalues
    nb_data["difference"] = nb_data["y_pred_base"] - nb_data["y_pred_wx"]

    def aggregate_deficit_pct(df):
        b, w = df["y_pred_base"].sum(), df["y_pred_wx"].sum()
        return 100.0 * (b - w) / b if b > 0 else np.nan

    deficit_pt = aggregate_deficit_pct(nb_data)

    facs = nb_data["facility"].unique()
    jack = np.array([aggregate_deficit_pct(nb_data[nb_data["facility"] != f])
                     for f in facs])
    jack = jack[np.isfinite(jack)]
    n = len(jack)
    if n > 1:
        jbar = jack.mean()
        se_jack = np.sqrt((n - 1) / n * np.sum((jack - jbar) ** 2))
        deficit_ci = (deficit_pt - 1.96 * se_jack, deficit_pt + 1.96 * se_jack)
    else:
        deficit_ci = (np.nan, np.nan)
        se_jack = np.nan

    print(f"  [{indicator}] Deficit: {deficit_pt:+.2f}% "
          f"(CI: {deficit_ci[0]:+.2f}..{deficit_ci[1]:+.2f})")

    # HOT-MONTH DEFICIT
    hot_threshold = np.percentile(nb_data[WBGT_VAR], REFERENCE_WBGT_PERCENTILE)
    hot_mask = nb_data[WBGT_VAR] > hot_threshold
    hot_data = nb_data[hot_mask].copy()

    if len(hot_data) > 10:
        hot_deficit_pt = aggregate_deficit_pct(hot_data)
        hot_facs = hot_data["facility"].unique()
        hot_jack = np.array([aggregate_deficit_pct(hot_data[hot_data["facility"] != f])
                             for f in hot_facs])
        hot_jack = hot_jack[np.isfinite(hot_jack)]
        n_hot = len(hot_jack)
        if n_hot > 1:
            hot_jbar = hot_jack.mean()
            hot_se_jack = np.sqrt((n_hot - 1) / n_hot * np.sum((hot_jack - hot_jbar) ** 2))
            hot_deficit_ci = (hot_deficit_pt - 1.96 * hot_se_jack,
                              hot_deficit_pt + 1.96 * hot_se_jack)
        else:
            hot_deficit_ci = (np.nan, np.nan)
            hot_se_jack = np.nan

        print(
            f"  [{indicator}] HOT months (>{hot_threshold:.1f}°C): "
            f"{hot_deficit_pt:+.2f}% (CI: {hot_deficit_ci[0]:+.2f}..{hot_deficit_ci[1]:+.2f})"
        )
    else:
        hot_deficit_pt = np.nan
        hot_deficit_ci = (np.nan, np.nan)
        hot_se_jack = np.nan
        print(f"  [{indicator}] Not enough hot months ({len(hot_data)} observations)")

    # EXTRAS for map_results.py compatibility
    reference_wbgt = float(np.percentile(nb_data[WBGT_VAR], IRR_LOW_PCTILE))

    try:
        design_info_wbgt = DESIGN[WBGT_VAR]
        B_hi = np.asarray(patsy.build_design_matrices(
            [design_info_wbgt],
            {"x": np.array([IRR_HIGH]) - SHIFTS[WBGT_VAR]},
        )[0])
        B_ref = np.asarray(patsy.build_design_matrices(
            [design_info_wbgt],
            {"x": np.array([reference_wbgt]) - SHIFTS[WBGT_VAR]},
        )[0])
        contrast_hl = (B_hi - B_ref).ravel()
        beta_s = model_wx.params.reindex(spline_cols).values
        V_s = model_wx.cov_params().reindex(index=spline_cols,
                                            columns=spline_cols).values
        log_irr_hl = float(contrast_hl @ beta_s)
        var_hl = float(contrast_hl @ V_s @ contrast_hl)
        se_hl = float(np.sqrt(max(var_hl, 0.0)))
        irr_pt = float(np.exp(log_irr_hl))
        irr_lo = float(np.exp(log_irr_hl - 1.96 * se_hl))
        irr_hi = float(np.exp(log_irr_hl + 1.96 * se_hl))
    except Exception as e:
        print(f"  [{indicator}] IRR contrast failed: {e}")
        irr_pt = irr_lo = irr_hi = np.nan

    # Joint Wald p-value on the WBGT spline block
    try:
        beta_s = model_wx.params.reindex(spline_cols).values
        V_s = model_wx.cov_params().reindex(index=spline_cols,
                                            columns=spline_cols).values
        chi2 = float(beta_s @ np.linalg.pinv(V_s) @ beta_s)
        df_test = int(np.linalg.matrix_rank(V_s))
        pval = float(1.0 - stats.chi2.cdf(chi2, df=df_test)) if df_test > 0 else np.nan
    except Exception as e:
        print(f"  [{indicator}] Wald test failed: {e}")
        pval = np.nan

    # historical_burden_{ind}_{WBGT_VAR}.csv
    hb = pd.DataFrame({
        "date": nb_data["date"].values,
        "facility": nb_data["facility"].values,
        "month": nb_data["month"].values,
        CLUSTER_COL: nb_data[CLUSTER_COL].values,
        "y_int": nb_data["y_int"].values,
        "mu_a": nb_data["y_pred_wx"].values,
        "mu_b": nb_data["y_pred_base"].values,
    })
    hb.to_csv(
        f"{OUT_DIR}historical_burden_{indicator}_{WBGT_VAR}.csv",
        index=False,
    )

    # district_burden_{ind}_{WBGT_VAR}.csv
    district_agg = (
        hb.groupby(CLUSTER_COL)[["mu_a", "mu_b"]].sum().reset_index()
    )
    district_agg["deficit_pct"] = np.where(
        district_agg["mu_b"] > 0,
        100.0 * (district_agg["mu_b"] - district_agg["mu_a"]) / district_agg["mu_b"],
        np.nan,
    )
    district_agg[[CLUSTER_COL, "deficit_pct"]].to_csv(
        f"{OUT_DIR}district_burden_{indicator}_{WBGT_VAR}.csv",
        index=False,
    )

    # district_burden_ci_{ind}_{WBGT_VAR}.csv (OPPOSITE sign convention)
    dist_rows = []
    for dist, sub in hb.groupby(CLUSTER_COL):
        pt, lo, hi = _monthly_jackknife_ci_local(
            sub["mu_a"].values, sub["mu_b"].values, sub["facility"].values,
            sign="a_minus_b",
        )
        sig = bool(pd.notna(lo) and pd.notna(hi) and (lo * hi > 0))
        dist_rows.append({
            "district": dist,
            "deficit_pct": pt,
            "ci_lo": lo,
            "ci_hi": hi,
            "sig": sig,
        })
    pd.DataFrame(dist_rows).to_csv(
        f"{OUT_DIR}district_burden_ci_{indicator}_{WBGT_VAR}.csv",
        index=False,
    )

    # TLO disruption curve
    try:
        Bg = np.asarray(patsy.build_design_matrices(
            [design_info_wbgt],
            {"x": TLO_WBGT_GRID - SHIFTS[WBGT_VAR]},
        )[0])
        Br = np.asarray(patsy.build_design_matrices(
            [design_info_wbgt],
            {"x": np.array([reference_wbgt]) - SHIFTS[WBGT_VAR]},
        )[0])
        rr_grid = np.exp((Bg - Br) @ beta_s)
        tlo_rows = pd.DataFrame({
            "indicator": indicator,
            "wbgt": TLO_WBGT_GRID,
            "rr_vs_ref": rr_grid,
            "disruption_probability": np.clip(1.0 - rr_grid, 0.0, None),
        })
    except Exception as e:
        print(f"  [{indicator}] TLO lookup rows failed: {e}")
        tlo_rows = pd.DataFrame(columns=[
            "indicator", "wbgt", "rr_vs_ref", "disruption_probability"])

    # Save per-indicator deficit row
    pd.DataFrame(
        [
            {
                "indicator": indicator,
                "label": INDICATOR_LABELS.get(indicator, indicator),
                "deficit_pct": deficit_pt,
                "ci_lo": deficit_ci[0],
                "ci_hi": deficit_ci[1],
                "se_jackknife": se_jack,
                "hot_deficit_pct": hot_deficit_pt,
                "hot_ci_lo": hot_deficit_ci[0],
                "hot_ci_hi": hot_deficit_ci[1],
                "hot_se_jackknife": hot_se_jack,
                "hot_threshold": hot_threshold if len(hot_data) > 10 else np.nan,
                "n_hot_obs": len(hot_data),
                "n_facilities": len(FITTED_FACILITIES),
                "n_obs": len(nb_data),
                "n_districts": nb_data[CLUSTER_COL].nunique(),
                "alpha": alpha,
                "reference_wbgt": reference_wbgt,
                "irr_hi_vs_low": irr_pt,
                "irr_lo_bound": irr_lo,
                "irr_hi_bound": irr_hi,
                "pval": pval,
                "time_seconds": time.time() - t0,
            }
        ]
    ).to_csv(f"{OUT_DIR}deficit_{indicator}.csv", index=False)

    pred_cols = ["year", "month", "facility", "date", "y_int", "covid",
                 "y_pred_base", "y_pred_wx", "difference"]
    nb_data[pred_cols].to_csv(
        f"{OUT_DIR}predictions_{indicator}.csv", index=False
    )

    print(f"  [{indicator}] Done in {time.time() - t0:.1f}s")

    return {
        "indicator": indicator,
        "label":     INDICATOR_LABELS.get(indicator, indicator),
        "deficit_pct": deficit_pt,
        "ci_lo":       deficit_ci[0],
        "ci_hi":       deficit_ci[1],
        "hot_deficit_pct": hot_deficit_pt,
        "hot_ci_lo":       hot_deficit_ci[0],
        "hot_ci_hi":       hot_deficit_ci[1],
        "hot_threshold":   hot_threshold if len(hot_data) > 10 else np.nan,
        "n_hot_obs":       len(hot_data),
        "n_facilities": len(FITTED_FACILITIES),
        "n_obs":        len(nb_data),
        "n_districts":  nb_data[CLUSTER_COL].nunique(),
        "alpha":        alpha,
        "reference_wbgt": reference_wbgt,
        "irr_hi_vs_low":  irr_pt,
        "irr_lo_bound":   irr_lo,
        "irr_hi_bound":   irr_hi,
        "pval":           pval,
        "time":           time.time() - t0,
        "_shifts":      SHIFTS,
        "_design_map":  DESIGN,
        "_spline_cols": spline_cols,
        "_train_facs":  list(FITTED_FACILITIES),
        "_model_wx":    model_wx,
        "_model_base":  model_base,
        "_fac_district": nb_data[["facility", CLUSTER_COL]]
                          .drop_duplicates().reset_index(drop=True),
        "_tlo_rows": tlo_rows,
    }


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED Two-Model NB Analysis")
    print(f"Indicators: {len(COUNT_INDICATORS)}")
    print(f"Parallel: {USE_PARALLEL} (workers={N_WORKERS})")
    print(f"Spline DF: {SPLINE_DF}, Lags: {LAG_MONTHS}")
    print("=" * 60)

    panel_paths = {
        ind: f"{PANEL_DIR}regression_panel_{ind}.csv"
        for ind in COUNT_INDICATORS
    }

    t_start = time.time()

    if USE_PARALLEL and len(COUNT_INDICATORS) > 1:
        print(f"\nRunning {len(COUNT_INDICATORS)} indicators in parallel...")
        with Pool(processes=N_WORKERS) as pool:
            results = pool.starmap(
                fit_indicator,
                [(ind, panel_paths[ind]) for ind in COUNT_INDICATORS]
            )
        for i, result in enumerate(results):
            if result is not None:
                print(
                    f"  [{i + 1}/{len(COUNT_INDICATORS)}] {result['indicator']}: "
                    f"hot deficit = {result['hot_deficit_pct']:+.2f}%"
                )
    else:
        print("\nRunning indicators sequentially...")
        results = []
        for ind in COUNT_INDICATORS:
            result = fit_indicator(ind, panel_paths[ind])
            results.append(result)

    results = [r for r in results if r is not None]

    if not results:
        raise RuntimeError("No indicators fitted successfully!")

    print(f"\n{'='*60}")
    print(f"All {len(results)} indicators fitted in {time.time() - t_start:.1f}s")
    print(f"Average per indicator: {(time.time() - t_start)/len(results):.1f}s")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Summary table with both aggregate and hot-month deficits
    # -----------------------------------------------------------------------
    summary_df = pd.DataFrame(
        [
            {
                "indicator": r["indicator"],
                "label": r["label"],
                "deficit_pct": r["deficit_pct"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "hot_deficit_pct": r["hot_deficit_pct"],
                "hot_ci_lo": r["hot_ci_lo"],
                "hot_ci_hi": r["hot_ci_hi"],
                "hot_threshold": r["hot_threshold"],
                "n_hot_obs": r["n_hot_obs"],
                "n_facilities": r["n_facilities"],
                "n_obs": r["n_obs"],
                "n_districts": r["n_districts"],
                "alpha": r["alpha"],
                "reference_wbgt": r["reference_wbgt"],
                "pval": r["pval"],
                "time": r["time"],
            }
            for r in results
        ]
    )

    def _bh_fdr(pvals, alpha):
        p = np.asarray(pvals, dtype=float)
        ok = ~np.isnan(p)
        q = np.full_like(p, np.nan)
        rej = np.zeros(p.shape, dtype=bool)
        if ok.sum() == 0:
            return q, rej
        p_ok = p[ok]
        n = len(p_ok)
        order = np.argsort(p_ok)
        adj = p_ok[order] * n / np.arange(1, n + 1)
        adj = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0, 1)
        q_ok = np.empty(n)
        q_ok[order] = adj
        q[ok] = q_ok
        rej[ok] = q_ok <= alpha
        return q, rej

    qvals, rej = _bh_fdr(summary_df["pval"].values, FDR_ALPHA)
    summary_df["qval"] = qvals
    summary_df["sig"] = rej

    summary_df = summary_df.sort_values("hot_deficit_pct", na_position="last")

    summary_df.to_csv(
        f"{OUT_DIR}two_model_deficit_results_NB_{WBGT_VAR}.csv",
        index=False,
    )
    summary_df.to_csv(f"{OUT_DIR}summary_all_indicators_{WBGT_VAR}.csv", index=False)

    # -----------------------------------------------------------------------
    # irr_contrast_{WBGT_VAR}.csv
    # -----------------------------------------------------------------------
    irr_df = pd.DataFrame([
        {
            "indicator": r["indicator"],
            "label": r["label"],
            "reference_wbgt": r["reference_wbgt"],
            "irr_high_wbgt": IRR_HIGH,
            "irr": r["irr_hi_vs_low"],
            "irr_lo": r["irr_lo_bound"],
            "irr_hi": r["irr_hi_bound"],
        }
        for r in results
    ])
    irr_df.to_csv(f"{OUT_DIR}irr_contrast_{WBGT_VAR}.csv", index=False)

    # -----------------------------------------------------------------------
    # tlo_wbgt_lookup.csv
    # -----------------------------------------------------------------------
    tlo_frames = [r["_tlo_rows"] for r in results
                  if isinstance(r.get("_tlo_rows"), pd.DataFrame)
                  and not r["_tlo_rows"].empty]
    if tlo_frames:
        tlo_df = pd.concat(tlo_frames, ignore_index=True)
        tlo_df.to_csv(f"{OUT_DIR}tlo_wbgt_lookup.csv", index=False)
    else:
        print("  no TLO rows to write")

    # -----------------------------------------------------------------------
    # exposure_response_curves_{WBGT_VAR}.csv
    # -----------------------------------------------------------------------
    curve_paths = sorted(
        Path(OUT_DIR).glob(f"exposure_response_curve_*_{WBGT_VAR}.csv")
    )
    if curve_paths:
        pd.concat(
            [pd.read_csv(p) for p in curve_paths], ignore_index=True
        ).to_csv(
            Path(OUT_DIR) / f"exposure_response_curves_{WBGT_VAR}.csv",
            index=False,
        )

    # =======================================================================
    # FORWARD PROJECTIONS  (top-level, guarded by PROJECT)
    # =======================================================================
    if PROJECT:
        print(f"\nForward projections ({min_year_projection}–{max_year_projection}) ...")
        PROJECTION_DIR = f"{DATA_DIR}/Thermofeel_WBGT/Indices"
        if WBGT_VAR == "wbgt_day":
            WBGT_PROJ_FILE_TPL = "wbgt_monthly_mean_facility_{tier}_{ssp}.csv"
        elif WBGT_VAR == "wbgt5x_day":
            WBGT_PROJ_FILE_TPL = "wbgt_extreme_indices_facility_{tier}_{ssp}.csv"
        else:
            raise ValueError(f"Unknown WBGT_VAR for projection: {WBGT_VAR}")

        LAG_COLS = [f"{WBGT_VAR}_lag{k}_c" for k in LAG_MONTHS]

        def _load_precip_wide(path, value_name):
            if not os.path.exists(path):
                return None
            wide = pd.read_csv(path, index_col=0)
            wide.index = (
                pd.to_datetime(wide.index.astype(str).str.strip(),
                               format="%Y-%m", errors="coerce")
                .to_period("M")
                .to_timestamp()
            )
            assert not wide.index.isna().any(), f"{path}: unparseable dates"
            wide.index.name = "date"
            wide.columns = wide.columns.astype(str).str.strip()
            return (
                wide.stack(future_stack=True)
                .rename(value_name)
                .rename_axis(index=["date", "facility"])
                .reset_index()
            )

        def _load_future_climate(ssp, tier, precip_file_tpl):
            wbgt_path = os.path.join(PROJECTION_DIR,
                                     WBGT_PROJ_FILE_TPL.format(ssp=ssp, tier=tier))
            pm_path = os.path.join(PROJECTION_DIR,
                                   precip_file_tpl.format(ssp=ssp))
            missing = [p for p in (wbgt_path, pm_path) if not os.path.exists(p)]
            if missing:
                return None, missing
            wbgt_df = pd.read_csv(wbgt_path, parse_dates=["date"])
            wbgt_df["facility"] = wbgt_df["facility"].astype(str).str.strip()
            wbgt_df["date"] = wbgt_df["date"].dt.to_period("M").dt.to_timestamp()
            pm_df = _load_precip_wide(pm_path, PRECIP_COL)
            clim = wbgt_df.merge(pm_df, on=["facility", "date"], how="inner")
            clim = clim.sort_values(["facility", "date"]).reset_index(drop=True)
            for k in LAG_MONTHS:
                clim[f"{WBGT_VAR}_lag{k}"] = clim.groupby("facility")[WBGT_VAR].shift(k)
            clim = clim[clim["date"].dt.year.between(
                min_year_projection, max_year_projection)].copy()
            clim["year"] = clim["date"].dt.year
            clim["month"] = clim["date"].dt.month
            return clim, None

        all_proj_summary = []
        all_annual_pooled = []

        for ssp in SSP_SCENARIOS:
            for tier in WBGT_MODELS:
                # FIX: pick precip template BEFORE calling _load_future_climate
                precip_file_tpl = PRECIP_FILE_BY_TIER.get(tier)
                if precip_file_tpl is None:
                    print(f"  SKIP {ssp}/{tier}: no precip template for tier")
                    continue

                clim, missing = _load_future_climate(ssp, tier, precip_file_tpl)
                if clim is None:
                    print(f"  SKIP {ssp}/{tier}: missing {missing}")
                    continue
                print(f"  {ssp}/{tier}: {len(clim):,} rows, "
                      f"{clim['facility'].nunique()} facilities")

                for res in results:
                    ind = res["indicator"]
                    shifts = res["_shifts"]
                    design_map = res["_design_map"]
                    spline_cols = res["_spline_cols"]
                    train_facs = set(res["_train_facs"])
                    model_wx = res["_model_wx"]
                    model_base = res["_model_base"]

                    df = clim[clim["facility"].isin(train_facs)].copy()
                    if df.empty:
                        print(f"    {ind}: no overlap — skipping")
                        continue

                    df["covid"] = 0
                    df["year_c"] = (
                        (LAST_HIST_YEAR - shifts["year"])
                        if PROJECT_HOLD_YEAR
                        else (df["year"] - shifts["year"])
                    )
                    df["precip_c"] = df[PRECIP_COL] - shifts.get("precip", 0.0)

                    xc = df[WBGT_VAR].values - shifts[WBGT_VAR]
                    B = np.asarray(
                        patsy.build_design_matrices(
                            [design_map[WBGT_VAR]], {"x": xc})[0],
                        dtype=float,
                    )
                    for i_col, c in enumerate(spline_cols):
                        df[c] = B[:, i_col]

                    for k in LAG_MONTHS:
                        df[f"{WBGT_VAR}_lag{k}_c"] = (
                            df[f"{WBGT_VAR}_lag{k}"] - shifts[WBGT_VAR]
                        )

                    need = ["covid", "year_c", "precip_c"] + list(spline_cols) + LAG_COLS
                    n_before = len(df)
                    df = df.dropna(subset=need).reset_index(drop=True)
                    n_dropped = n_before - len(df)
                    if df.empty:
                        print(f"    {ind}: no rows after covariate build")
                        continue

                    mu_wx = np.asarray(model_wx.predict(df), dtype=float)
                    mu_base = np.asarray(model_base.predict(df), dtype=float)

                    ok = np.isfinite(mu_wx) & np.isfinite(mu_base)
                    n_pred_nan = int((~ok).sum())
                    df = df.loc[ok].reset_index(drop=True)
                    mu_wx = mu_wx[ok]
                    mu_base = mu_base[ok]
                    if df.empty:
                        print(f"    {ind}: all predictions NaN")
                        continue

                    df["mu_a"] = mu_wx
                    df["mu_b"] = mu_base
                    df["Disruption"] = df["mu_b"] - df["mu_a"]
                    df["Deficit_Pct"] = np.where(
                        df["mu_b"] > 0,
                        100.0 * df["Disruption"] / df["mu_b"],
                        np.nan,
                    )
                    df = df.merge(res["_fac_district"], on="facility", how="left")
                    df["indicator"], df["ssp"], df["tier"] = ind, ssp, tier

                    # facility × month
                    df[
                        [
                            "indicator", "ssp", "tier", "facility", CLUSTER_COL,
                            "year", "month", "date", WBGT_VAR, PRECIP_COL,
                            "mu_a", "mu_b", "Disruption", "Deficit_Pct",
                        ]
                    ].to_csv(
                        f"{OUT_DIR}projection_facility_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
                        index=False,
                    )

                    # district × year × month
                    dist_agg = (
                        df.groupby([CLUSTER_COL, "year", "month"])
                        .agg(
                            mu_a=("mu_a", "sum"),
                            mu_b=("mu_b", "sum"),
                            Mean_WBGT=(WBGT_VAR, "mean"),
                            Mean_Precip_Month=(PRECIP_COL, "mean"),
                            N_Facilities=("facility", "nunique"),
                        )
                        .reset_index()
                    )
                    dist_agg["Disruption"] = dist_agg["mu_b"] - dist_agg["mu_a"]
                    dist_agg["Deficit_Pct"] = np.where(
                        dist_agg["mu_b"] > 0,
                        100.0 * dist_agg["Disruption"] / dist_agg["mu_b"],
                        np.nan,
                    )
                    dist_agg["indicator"], dist_agg["ssp"], dist_agg["tier"] = ind, ssp, tier
                    dist_agg.to_csv(
                        f"{OUT_DIR}projection_district_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
                        index=False,
                    )

                    # pooled monthly time series
                    mon_agg = (
                        df.groupby(["year", "month"])
                        .agg(
                            mu_a=("mu_a", "sum"),
                            mu_b=("mu_b", "sum"),
                            Mean_WBGT=(WBGT_VAR, "mean"),
                            Mean_Precip_Month=(PRECIP_COL, "mean"),
                            N_Facilities=("facility", "nunique"),
                        )
                        .reset_index()
                    )
                    mon_agg["Disruption"] = mon_agg["mu_b"] - mon_agg["mu_a"]
                    mon_agg["Deficit_Pct"] = np.where(
                        mon_agg["mu_b"] > 0,
                        100.0 * mon_agg["Disruption"] / mon_agg["mu_b"],
                        np.nan,
                    )
                    mon_agg["Year_Month"] = (
                        mon_agg["year"].astype(str) + "-" + mon_agg["month"].astype(str)
                    )
                    mon_agg["indicator"], mon_agg["ssp"], mon_agg["tier"] = ind, ssp, tier
                    mon_agg.to_csv(
                        f"{OUT_DIR}projection_monthly_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
                        index=False,
                    )

                    # ─── ANNUAL: pooled across facilities ──────────────────
                    ann_agg = (
                        df.groupby("year")
                        .agg(
                            mu_a=("mu_a", "sum"),
                            mu_b=("mu_b", "sum"),
                            Mean_WBGT=(WBGT_VAR, "mean"),
                            Mean_Precip_Month=(PRECIP_COL, "mean"),
                            N_Facilities=("facility", "nunique"),
                        )
                        .reset_index()
                    )
                    ann_agg["Disruption"] = ann_agg["mu_b"] - ann_agg["mu_a"]
                    ann_agg["Deficit_Pct"] = np.where(
                        ann_agg["mu_b"] > 0,
                        100.0 * ann_agg["Disruption"] / ann_agg["mu_b"],
                        np.nan,
                    )
                    ann_agg["Mean_Monthly_Disruption"] = (
                        ann_agg["Disruption"] / 12.0
                    )
                    ann_agg["indicator"], ann_agg["ssp"], ann_agg["tier"] = ind, ssp, tier
                    ann_agg.to_csv(
                        f"{OUT_DIR}projection_annual_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
                        index=False,
                    )
                    all_annual_pooled.append(ann_agg)

                    # ─── ANNUAL: district × year ───────────────────────────
                    dist_ann = (
                        df.groupby([CLUSTER_COL, "year"])
                        .agg(
                            mu_a=("mu_a", "sum"),
                            mu_b=("mu_b", "sum"),
                            Mean_WBGT=(WBGT_VAR, "mean"),
                            N_Facilities=("facility", "nunique"),
                        )
                        .reset_index()
                    )
                    dist_ann["Disruption"] = dist_ann["mu_b"] - dist_ann["mu_a"]
                    dist_ann["Deficit_Pct"] = np.where(
                        dist_ann["mu_b"] > 0,
                        100.0 * dist_ann["Disruption"] / dist_ann["mu_b"],
                        np.nan,
                    )
                    dist_ann["indicator"], dist_ann["ssp"], dist_ann["tier"] = ind, ssp, tier
                    dist_ann.to_csv(
                        f"{OUT_DIR}projection_district_annual_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
                        index=False,
                    )

                    tot_a, tot_b = float(mu_wx.sum()), float(mu_base.sum())
                    deficit_proj = (100.0 * (tot_b - tot_a) / tot_b) if tot_b > 0 else np.nan
                    mean_annual_disruption = (
                        (tot_b - tot_a) / df["year"].nunique()
                        if df["year"].nunique() > 0 else np.nan
                    )
                    all_proj_summary.append(
                        {
                            "indicator": ind,
                            "ssp": ssp,
                            "tier": tier,
                            "period_start": min_year_projection,
                            "period_end": max_year_projection,
                            "n_facility_months": len(df),
                            "n_input_dropped": n_dropped,
                            "n_pred_nan": n_pred_nan,
                            "mean_wbgt": float(df[WBGT_VAR].mean()),
                            "mean_precip_month": float(df[PRECIP_COL].mean()),
                            "total_A_projected": tot_a,
                            "total_B_projected": tot_b,
                            "deficit_pct": deficit_proj,
                            "mean_annual_disruption": mean_annual_disruption,
                        }
                    )
                    print(
                        f"    {ind}: deficit_proj={deficit_proj:+.2f}% "
                        f"(n={len(df):,}, dropped {n_dropped:,}, "
                        f"{n_pred_nan:,} pred NaN)"
                    )

        if all_proj_summary:
            pd.DataFrame(all_proj_summary).to_csv(
                f"{OUT_DIR}projection_summary_{WBGT_VAR}.csv", index=False
            )
            print(f"\nProjection summary → {OUT_DIR}projection_summary_{WBGT_VAR}.csv")
        else:
            print("\nNo projections produced — check file templates against actual filenames in PROJECTION_DIR.")

        # Concatenate all annual pooled files into one convenient long file
        if all_annual_pooled:
            pd.concat(all_annual_pooled, ignore_index=True).to_csv(
                f"{OUT_DIR}projection_annual_all_{WBGT_VAR}.csv", index=False
            )
            print(f"Annual pooled (all indicators/SSPs/tiers) → "
                  f"{OUT_DIR}projection_annual_all_{WBGT_VAR}.csv")

    print("\nSummary (HOT MONTHS ONLY):")
    print(summary_df[["indicator", "hot_deficit_pct", "hot_ci_lo",
                      "hot_ci_hi", "n_hot_obs"]].to_string())
