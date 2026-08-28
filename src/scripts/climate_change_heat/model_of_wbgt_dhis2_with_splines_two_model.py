"""
model_of_wbgt_dhis2_two_model_splines_optimized.py

Two-model NB (statsmodels) WBGT–service analysis - OPTIMIZED VERSION.

New in this version: ONLY_DEFICITS toggle
  When True, aggregations only consider observations where the baseline
  (mu_b) exceeds the weather model (mu_a) — i.e. months where services
  were LOST to heat. Months where the weather model predicts MORE
  services than baseline (services gained) are excluded from the mean.
  The point estimate answers a different question:
    OFF: "net effect of weather on services"
    ON:  "when services are disrupted, how much are they disrupted by"
  Output CSV names carry an `_onlydeficits` suffix when the toggle is on,
  so the two sets of results coexist in OUT_DIR without overwriting.
"""

import os

import numpy.random

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

warnings.filterwarnings("ignore", category=UserWarning)
numpy.random.seed(42)
# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
COUNT_INDICATORS = [
    "opd_attendance",
    "ipd_total_admissions",
    "fp_total_clients",
    "fp_subsequent_clients_total",
    "bcg_under1",
    "penta3_under1",
    "measles1_under1",
    "fully_immunised_under1",
    "live_births_total",
    "htc_results_new_negative",
    "htc_results_new_positive",
    "anc_total_visits",
    "cervical_screening_total",
    "pnc_within_2wks",
]

INDICATOR_LABELS: dict[str, str] = {
    # existing — kept for the ones we still fit; older entries kept for plot backwards-compat
    "fp_total_clients": "FP Total Clients",
    "opd_attendance": "OPD Attendance",
    "ipd_total_admissions": "IPD Total Admissions",
    "vmmc_first_visits": "VMMC First Visits",
    "pnc_mother_checked_48h": "PNC Mother <48h",
    "anc_new_attendees": "ANC New Attendees",
    "anc_total_visits": "ANC Total Visits",
    "anc_first_trimester_starts": "ANC 1st Trimester Starts",
    "bcg_under1": "BCG Under-1",
    "penta3_under1": "Penta3 Under-1",
    "measles1_under1": "Measles 1st Dose Under-1",
    "fully_immunised_under1": "Fully Immunised Under-1",
    "pnc_within_2wks": "PNC Within 2 Weeks",
    "pnc_first_visit_2wks": "PNC First Visit <2 Weeks",
    "live_births_total": "Live Births Total",
    "skilled_deliveries": "Skilled Deliveries",
    "htc_results_new_negative":    "HTC New Negative",
    "htc_results_new_positive":    "HTC New Positive",
    "fp_subsequent_clients_total": "FP Subsequent Clients",
    "fully_immunised_outreach":    "Fully Immunised Under-1 (Outreach)",
    "cervical_screening_total":  "Cervical Screening (Total)",
}

HIGH_OVERDISPERSION_INDICATORS = [
    "ipd_total_admissions",
    "opd_attendance",
    "anc_total_visits",
    # new — cervical screening will be sparse and pulsed post-rollout;
    # defensive add so alpha estimation uses the intercept-only path.
    "cervical_screening_initial",
]

MIN_YEAR_BY_INDICATOR: dict[str, int] = {
    "fp_total_clients": 2019,
    "vmmc_first_visits": 2019,
    # new — DHIS2 forms introduced later; earlier years are structural zeros.
    # VERIFY these against the pulled data (plot national monthly totals; the
    # hockey-stick shows the true rollout month).
    "htc_results_new_negative":   2019,
    "htc_results_new_positive":   2019,
    "cervical_screening_initial": 2020,  # CECAP form rollout — verify
    "fully_immunised_outreach":   2018,
}


# --- Only-deficits toggle --------------------------------------------------
# When True, restrict every aggregation (historical deficit, hot-month
# deficit, projection deficits, district aggregations) to rows where
# baseline > weather (services lost). Suffix all output CSVs accordingly.
ONLY_DEFICITS = True
SUFFIX = "_onlydeficits" if ONLY_DEFICITS else ""


def _apply_deficit_filter(df, base_col, wx_col):
    """Return `df` filtered to rows where the baseline exceeds the weather
    model (services lost). If ONLY_DEFICITS is False, return unchanged."""
    if not ONLY_DEFICITS:
        return df
    return df[df[base_col] > df[wx_col]].copy()


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

REFERENCE_WBGT_PERCENTILE = 95
HOT_DEFICIT_CI_METHOD = "bootstrap"
N_BOOTSTRAP = 1000

IRR_LOW_PCTILE = 25
IRR_HIGH = 32.0

TLO_WBGT_GRID = np.linspace(20.0, 34.0, 57)
FDR_ALPHA = 0.05

CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre", "2023-03-01", "2024-03-01"),
]

min_year_historical = 2016
max_year_historical = 2025
LAST_HIST_YEAR = max_year_historical - 1

MIN_YEAR_BY_INDICATOR: dict[str, int] = {
    "fp_total_clients": 2019,
    "vmmc_first_visits": 2019,
}

PROJECT = True
PROJECT_HOLD_YEAR = True
SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
WBGT_MODELS = ["lowest", "median", "highest"]
min_year_projection = 2025
max_year_projection = 2041

PRECIP_FILE_BY_TIER = {
    "highest": "precip_monthly_total_facility_CanESM5_{ssp}.csv",
    "lowest": "precip_monthly_total_facility_MPI-ESM1-2-HR_{ssp}.csv",
    "median": "precip_monthly_total_facility_MIROC6_{ssp}.csv",
}

CURVE_REF_MODE = "mean"
CURVE_N = 60

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
PANEL_DIR = f"{DATA_DIR}/Thermofeel_WBGT/Indices/"
INDICES_DIR = PANEL_DIR
os.makedirs(OUT_DIR, exist_ok=True)

USE_PARALLEL = False
N_WORKERS = min(cpu_count() - 1, 4)

WINSORIZE_BY_INDICATOR: dict[str, float] = {
    "opd_attendance": 0.99,
    "ipd_total_admissions": 1,
    "fully_immunised_under1": 0.95,
    "pnc_within_2wks": 0.90,
    "measles1_under1": 0.90,
    "anc_total_visits": 1,
}
WINSORIZE_DEFAULT = 0.999


# ===========================================================================
# Jackknife CI helper (respects ONLY_DEFICITS via aggregator arg)
# ===========================================================================
def _monthly_jackknife_ci_local(mu_a, mu_b, facility_ids, sign="b_minus_a"):
    """Leave-one-facility-out 95% CI on deficit_pct.

    NOTE ON ONLY_DEFICITS: this helper receives (mu_a, mu_b) arrays that
    have ALREADY been filtered by the caller when the toggle is on. So the
    jackknife is over the same restricted set that produced the point
    estimate — internally consistent.
    """
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    facility_ids = np.asarray(facility_ids)
    sum_a = float(mu_a.sum())
    sum_b = float(mu_b.sum())
    if sum_b <= 0:
        return np.nan, np.nan, np.nan

    def _stat(sa, sb):
        return (100.0 * (sa - sb) / sb) if sign == "a_minus_b" else (100.0 * (sb - sa) / sb)

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
# IMPROVED: Alpha estimation with robust fallback
# ===========================================================================
def estimate_alpha_fast(data, y_col="y_int"):
    """
    Quick alpha estimate for NB2 using method of moments from intercept-only model.
    This is the preferred method for high-overdispersion indicators because it
    always converges and provides an unbiased estimate.
    """
    try:
        mod_pois = smf.glm(f"{y_col} ~ 1", data=data, family=sm.families.Poisson()).fit()
        mu = mod_pois.fittedvalues
        pearson_resid = (data[y_col] - mu) / np.sqrt(mu + 1e-10)
        dispersion = (pearson_resid**2).sum() / mod_pois.df_resid
        alpha = max(0.01, (dispersion - 1) / mu.mean())
        ALPHA_STABILITY_CAP = 5.0
        alpha = min(alpha, ALPHA_STABILITY_CAP)
        if alpha == ALPHA_STABILITY_CAP:
            print(f"   alpha capped at {ALPHA_STABILITY_CAP} for stability (raw estimate was higher)")
        return alpha
    except Exception as e:
        print(f"  Alpha estimation via intercept-only failed: {e}")
        # Fallback: use variance/mean ratio directly
        mean_y = data[y_col].mean()
        var_y = data[y_col].var()
        alpha = max(0.01, (var_y - mean_y) / (mean_y**2)) if mean_y > 0 else 0.1
        ALPHA_STABILITY_CAP = 5.0
        alpha = min(alpha, ALPHA_STABILITY_CAP)
        if alpha == ALPHA_STABILITY_CAP:
            print(f"   alpha capped at {ALPHA_STABILITY_CAP} for stability (raw estimate was higher)")
        return alpha


def estimate_alpha_from_poisson(poisson_result, y, max_alpha=50.0):
    """
    Estimate alpha from a fitted Poisson model.
    This method is less stable for high-overdispersion indicators.

    Parameters
    ----------
    poisson_result : statsmodels GLM result
        Fitted Poisson model
    y : array-like
        Observed counts
    max_alpha : float
        Upper bound for alpha (increased from 10 to 50 for better handling
        of high-overdispersion indicators)
    """
    mu = np.clip(poisson_result.fittedvalues, 1e-6, None)
    pearson_chi2 = ((y - mu) ** 2 / mu  + 1e-10).sum()
    dispersion = pearson_chi2 / poisson_result.df_resid
    alpha = (dispersion - 1) / mu.mean()
    ALPHA_STABILITY_CAP = 5.0
    alpha = min(alpha, ALPHA_STABILITY_CAP)
    if alpha == ALPHA_STABILITY_CAP:
        print(f"   alpha capped at {ALPHA_STABILITY_CAP} for stability (raw estimate was higher)")
    return alpha

    return float(np.clip(alpha, 1e-4, max_alpha))


def estimate_alpha_robust(data, indicator=None, y_col="y_int"):
    """
    Robust alpha estimation with indicator-specific strategy.

    For high-overdispersion indicators (Var/Mean >> 10), use the intercept-only
    method which is more stable. For others, use the full-model method if
    available, but fall back to intercept-only if it fails.
    """
    # Check if this is a high-overdispersion indicator
    if indicator in HIGH_OVERDISPERSION_INDICATORS:
        print(f"  [{indicator}] Using intercept-only alpha (high-overdispersion)")
        return estimate_alpha_fast(data, y_col)

    # For other indicators, try the full-model method first
    try:
        mod_pois = smf.glm(f"{y_col} ~ year_c + covid", data=data, family=sm.families.Poisson()).fit()
        alpha = estimate_alpha_from_poisson(mod_pois, data[y_col])
        return alpha
    except Exception as e:
        print(f"  Full-model alpha failed ({e}), falling back to intercept-only")
        return estimate_alpha_fast(data, y_col)


def fit_negbin_warmstart(formula, data, groups, alpha, start_params, y_col="y_int"):
    mod = smf.glm(formula=formula, data=data, family=sm.families.NegativeBinomial(alpha=alpha))
    result = mod.fit(cov_type="cluster", cov_kwds={"groups": groups}, start_params=start_params, maxiter=200)
    result.alpha_hat = alpha
    return result


def fit_negbin_fixed_alpha(formula, data, groups, alpha, y_col="y_int"):
    mod = smf.glm(formula=formula, data=data, family=sm.families.NegativeBinomial(alpha=alpha)).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    mod.alpha_hat = alpha
    return mod


def diagnose_indicator(df, indicator):
    y = df[indicator].dropna()
    print(f"\n{'=' * 50}")
    print(f"  DIAGNOSTICS: {indicator}")
    print(f"{'=' * 50}")
    print(f"  N obs:        {len(y):,}")
    print(f"  N zeros:      {(y == 0).sum():,} ({100 * (y == 0).mean():.1f}%)")
    print(f"  Mean:         {y.mean():.2f}")
    print(f"  Variance:     {y.var():.2f}")
    print(f"  Var/Mean:     {y.var() / y.mean():.2f}")
    print(f"  Max:          {y.max():.0f}")
    print(f"  Skewness:     {y.skew():.2f}")
    print(f"  % > 3*IQR:    {100 * (y > y.quantile(0.75) * 3).mean():.2f}%")


# ===========================================================================
# Weather column construction
# ===========================================================================
def add_weather_columns_optimized(df, shifts, spline_design=None, lag_months=LAG_MONTHS):
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
            B = patsy.dmatrix(f"cr(x, df={SPLINE_DF}) - 1", {"x": xc}, return_type="dataframe")
            design_map[v] = B.design_info
        else:
            B = patsy.build_design_matrices([design_map[v]], {"x": xc})[0]
            B = pd.DataFrame(np.asarray(B))
        cols = [f"{v}_s{i + 1}" for i in range(B.shape[1])]
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
# Exposure-response curve
# ===========================================================================
def exposure_response_curve_fast(
    model, spline_cols, design_info, WBGT_VAR_name, wbgt_shift, ref_mode, wbgt_values, n=CURVE_N
):
    wobs = np.asarray(wbgt_values).flatten()
    grid = np.linspace(np.percentile(wobs, 1), np.percentile(wobs, 99), n)
    ref = wobs.mean() if ref_mode == "mean" else float(ref_mode)

    Bg = np.asarray(patsy.build_design_matrices([design_info[WBGT_VAR_name]], {"x": grid - wbgt_shift})[0])
    Br = np.asarray(patsy.build_design_matrices([design_info[WBGT_VAR_name]], {"x": np.array([ref]) - wbgt_shift})[0])
    contrast = Bg - Br
    beta = model.params.reindex(spline_cols).values
    V = model.cov_params().reindex(index=spline_cols, columns=spline_cols).values
    log_irr = contrast @ beta
    var = np.einsum("ij,jk,ik->i", contrast, V, contrast)
    se = np.sqrt(np.clip(var, 0, None))
    return pd.DataFrame(
        {
            "wbgt": grid,
            "wbgt_ref": ref,
            "rr_vs_ref": np.exp(log_irr),
            "rr_lo": np.exp(log_irr - 1.96 * se),
            "rr_hi": np.exp(log_irr + 1.96 * se),
        }
    )


# ===========================================================================
# Winsorize
# ===========================================================================
def winsorize_indicator(df, indicator_col="y", facility_col="facility_id", upper_quantile=0.999):
    df = df.copy()

    def _cap_facility(group):
        cap = group[indicator_col].quantile(upper_quantile)
        return group[indicator_col].clip(upper=cap)

    original = df[indicator_col].copy()
    df[indicator_col] = df.groupby(facility_col, group_keys=False).apply(_cap_facility)
    n_capped = (df[indicator_col] != original).sum()
    print(
        f"  [WINSORIZE] {indicator_col}: {n_capped} values capped "
        f"(per-facility {upper_quantile * 100:.1f}th percentile)"
    )
    return df


# ===========================================================================
# Main fitting function
# ===========================================================================
def fit_indicator(indicator, panel_path):
    print(f"\n→ {indicator}")
    if ONLY_DEFICITS:
        print(f"  ONLY_DEFICITS = True — aggregations restricted to loss-of-service rows")
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

    # Closures -> 0 (keep as 0, not NaN)
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
    wq = WINSORIZE_BY_INDICATOR.get(indicator, WINSORIZE_DEFAULT)
    if wq != WINSORIZE_DEFAULT:
        print(f"  [{indicator}] using tighter winsorization: {wq}")
    long = winsorize_indicator(long, indicator_col="y", facility_col="facility", upper_quantile=wq)
    diagnose_indicator(long, "y")

    # Optional scaling for extreme indicators (kept for compatibility)
    if indicator == "ipd_total_admissions":
        # Scale down extreme values for numerical stability
        scale_factor = 100
        long["y"] = long["y"] / scale_factor
        print(f"  [{indicator}] Scaling response by {scale_factor}")
    else:
        scale_factor = 1

    SHIFTS = {"year": long["year"].mean() if CENTER else 0.0}
    SHIFTS[WBGT_VAR] = long[WBGT_VAR].mean() if CENTER else 0.0
    if USE_PRECIP:
        SHIFTS["precip"] = long[PRECIP_COL].mean() if CENTER else 0.0

    long, weather_rhs, spline_cols, DESIGN = add_weather_columns_optimized(long, SHIFTS)

    nb_cols = ["y", "facility", "month", CLUSTER_COL] + weather_rhs
    nb_data = long.dropna(subset=nb_cols).copy()
    nb_data["y_int"] = nb_data["y"].round().clip(lower=0).astype(int)
    obs_nb = nb_data.groupby("facility").size()
    nb_data = nb_data[nb_data["facility"].isin(obs_nb[obs_nb >= MIN_OBS].index)].copy()
    FITTED_FACILITIES = set(nb_data["facility"].unique())

    print(f"  [{indicator}] Sample: {len(nb_data):,} obs, {len(FITTED_FACILITIES)} facilities")

    groups = nb_data[CLUSTER_COL]
    ctrl = "year_c + covid" + (" + precip_c" if USE_PRECIP else "")
    FE = "C(month) + C(facility)"
    f_base = f"y_int ~ {ctrl} + {FE}"
    f_wx = f"y_int ~ {' + '.join(weather_rhs)} + {FE}"

    # -----------------------------------------------------------------------
    # Estimate alpha with robust method (IMPROVED)
    # -----------------------------------------------------------------------
    alpha = estimate_alpha_robust(nb_data, indicator=indicator)
    print(f"  [{indicator}] Alpha: {alpha:.4f}")

    try:
        # Use the estimated alpha directly - no Poisson fitting needed
        model_base = fit_negbin_fixed_alpha(f_base, nb_data, groups, alpha)
        model_wx = fit_negbin_fixed_alpha(f_wx, nb_data, groups, alpha)
    except Exception as e:
        print(f"  [{indicator}] Model fitting failed: {e}")
        return None

    # Rescale fitted values back to original scale if needed
    if scale_factor > 1:
        model_base.fittedvalues = model_base.fittedvalues * scale_factor
        model_wx.fittedvalues = model_wx.fittedvalues * scale_factor

    missing_spline = [c for c in spline_cols if c not in model_wx.params.index]
    if missing_spline:
        print(f"  [{indicator}] Warning: missing spline cols: {missing_spline}")

    # Exposure-response curve (not affected by ONLY_DEFICITS — this is
    # a model-derived curve, not an aggregation over observed rows)
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

    nb_data["y_pred_base"] = model_base.fittedvalues
    nb_data["y_pred_wx"] = model_wx.fittedvalues
    nb_data["difference"] = nb_data["y_pred_base"] - nb_data["y_pred_wx"]

    # ---- Aggregate deficit (respects ONLY_DEFICITS) --------------------
    def aggregate_deficit_pct(df):
        d = _apply_deficit_filter(df, "y_pred_base", "y_pred_wx")
        b, w = d["y_pred_base"].sum(), d["y_pred_wx"].sum()
        return 100.0 * (b - w) / b if b > 0 else np.nan

    deficit_pt = aggregate_deficit_pct(nb_data)

    facs = nb_data["facility"].unique()
    jack = np.array([aggregate_deficit_pct(nb_data[nb_data["facility"] != f]) for f in facs])
    jack = jack[np.isfinite(jack)]
    n = len(jack)
    if n > 1:
        jbar = jack.mean()
        se_jack = np.sqrt((n - 1) / n * np.sum((jack - jbar) ** 2))
        deficit_ci = (deficit_pt - 1.96 * se_jack, deficit_pt + 1.96 * se_jack)
    else:
        deficit_ci = (np.nan, np.nan)
        se_jack = np.nan

    print(f"  [{indicator}] Deficit: {deficit_pt:+.2f}% (CI: {deficit_ci[0]:+.2f}..{deficit_ci[1]:+.2f})")

    # ---- Hot-month deficit ---------------------------------------------
    hot_threshold = np.percentile(nb_data[WBGT_VAR], REFERENCE_WBGT_PERCENTILE)
    hot_mask = nb_data[WBGT_VAR] > hot_threshold
    hot_data = nb_data[hot_mask].copy()

    if len(hot_data) > 10:
        hot_deficit_pt = aggregate_deficit_pct(hot_data)
        hot_facs = hot_data["facility"].unique()
        hot_jack = np.array([aggregate_deficit_pct(hot_data[hot_data["facility"] != f]) for f in hot_facs])
        hot_jack = hot_jack[np.isfinite(hot_jack)]
        n_hot = len(hot_jack)
        if n_hot > 1:
            hot_jbar = hot_jack.mean()
            hot_se_jack = np.sqrt((n_hot - 1) / n_hot * np.sum((hot_jack - hot_jbar) ** 2))
            hot_deficit_ci = (hot_deficit_pt - 1.96 * hot_se_jack, hot_deficit_pt + 1.96 * hot_se_jack)
        else:
            hot_deficit_ci = (np.nan, np.nan)
            hot_se_jack = np.nan

        print(
            f"  [{indicator}] HOT months (>{hot_threshold:.1f}°C): "
            f"{hot_deficit_pt:+.2f}% "
            f"(CI: {hot_deficit_ci[0]:+.2f}..{hot_deficit_ci[1]:+.2f})"
        )
    else:
        hot_deficit_pt = np.nan
        hot_deficit_ci = (np.nan, np.nan)
        hot_se_jack = np.nan
        print(f"  [{indicator}] Not enough hot months ({len(hot_data)} observations)")

    # ---- IRR contrast (model-derived; unaffected by ONLY_DEFICITS) -----
    reference_wbgt = float(np.percentile(nb_data[WBGT_VAR], IRR_LOW_PCTILE))
    try:
        design_info_wbgt = DESIGN[WBGT_VAR]
        B_hi = np.asarray(
            patsy.build_design_matrices([design_info_wbgt], {"x": np.array([IRR_HIGH]) - SHIFTS[WBGT_VAR]})[0]
        )
        B_ref = np.asarray(
            patsy.build_design_matrices([design_info_wbgt], {"x": np.array([reference_wbgt]) - SHIFTS[WBGT_VAR]})[0]
        )
        contrast_hl = (B_hi - B_ref).ravel()
        beta_s = model_wx.params.reindex(spline_cols).values
        V_s = model_wx.cov_params().reindex(index=spline_cols, columns=spline_cols).values
        log_irr_hl = float(contrast_hl @ beta_s)
        var_hl = float(contrast_hl @ V_s @ contrast_hl)
        se_hl = float(np.sqrt(max(var_hl, 0.0)))
        irr_pt = float(np.exp(log_irr_hl))
        irr_lo = float(np.exp(log_irr_hl - 1.96 * se_hl))
        irr_hi = float(np.exp(log_irr_hl + 1.96 * se_hl))
    except Exception as e:
        print(f"  [{indicator}] IRR contrast failed: {e}")
        irr_pt = irr_lo = irr_hi = np.nan

    try:
        # statsmodels handles the covariance properly and uses F-test when possible
        from statsmodels.iolib.smpickle import load_pickle  # noqa (just to check import)

        wt = model_wx.wald_test(
            [c for c in spline_cols if c in model_wx.params.index],
            use_f=True,  # better calibrated for cluster-robust with small G
            scalar=True,
        )
        pval = float(wt.pvalue)
        chi2 = float(wt.statistic)
    except Exception as e:
        print(f"  [{indicator}] Wald test failed: {e}")
        pval = np.nan

    # ---- Historical burden file ----------------------------------------
    # Always write the full facility-month table; downstream aggregations
    # filter on read. This keeps the file useful for both toggle states.
    hb = pd.DataFrame(
        {
            "date": nb_data["date"].values,
            "facility": nb_data["facility"].values,
            "month": nb_data["month"].values,
            CLUSTER_COL: nb_data[CLUSTER_COL].values,
            "y_int": nb_data["y_int"].values,
            "mu_a": nb_data["y_pred_wx"].values,
            "mu_b": nb_data["y_pred_base"].values,
        }
    )
    hb.to_csv(
        f"{OUT_DIR}historical_burden_{indicator}_{WBGT_VAR}.csv",
        index=False,
    )

    # ---- District burden (respects ONLY_DEFICITS in the aggregation) ---
    hb_for_agg = _apply_deficit_filter(hb, "mu_b", "mu_a")
    district_agg = hb_for_agg.groupby(CLUSTER_COL)[["mu_a", "mu_b"]].sum().reset_index()
    district_agg["deficit_pct"] = np.where(
        district_agg["mu_b"] > 0,
        100.0 * (district_agg["mu_b"] - district_agg["mu_a"]) / district_agg["mu_b"],
        np.nan,
    )
    district_agg[[CLUSTER_COL, "deficit_pct"]].to_csv(
        f"{OUT_DIR}district_burden_{indicator}_{WBGT_VAR}{SUFFIX}.csv",
        index=False,
    )

    dist_rows = []
    for dist, sub in hb_for_agg.groupby(CLUSTER_COL):
        pt, lo, hi = _monthly_jackknife_ci_local(
            sub["mu_a"].values,
            sub["mu_b"].values,
            sub["facility"].values,
            sign="a_minus_b",
        )
        sig = bool(pd.notna(lo) and pd.notna(hi) and (lo * hi > 0))
        dist_rows.append(
            {
                "district": dist,
                "deficit_pct": pt,
                "ci_lo": lo,
                "ci_hi": hi,
                "sig": sig,
            }
        )
    pd.DataFrame(dist_rows).to_csv(
        f"{OUT_DIR}district_burden_ci_{indicator}_{WBGT_VAR}{SUFFIX}.csv",
        index=False,
    )

    # ---- TLO disruption curve (model-derived; unaffected) --------------
    try:
        Bg = np.asarray(patsy.build_design_matrices([design_info_wbgt], {"x": TLO_WBGT_GRID - SHIFTS[WBGT_VAR]})[0])
        Br = np.asarray(
            patsy.build_design_matrices([design_info_wbgt], {"x": np.array([reference_wbgt]) - SHIFTS[WBGT_VAR]})[0]
        )
        rr_grid = np.exp((Bg - Br) @ beta_s)
        tlo_rows = pd.DataFrame(
            {
                "indicator": indicator,
                "wbgt": TLO_WBGT_GRID,
                "rr_vs_ref": rr_grid,
                "disruption_probability": np.clip(1.0 - rr_grid, 0.0, None),
            }
        )
    except Exception as e:
        print(f"  [{indicator}] TLO lookup rows failed: {e}")
        tlo_rows = pd.DataFrame(columns=["indicator", "wbgt", "rr_vs_ref", "disruption_probability"])

    pd.DataFrame(
        [
            {
                "indicator": indicator,
                "label": INDICATOR_LABELS.get(indicator, indicator),
                "only_deficits": ONLY_DEFICITS,
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
    ).to_csv(f"{OUT_DIR}deficit_{indicator}{SUFFIX}.csv", index=False)

    pred_cols = ["year", "month", "facility", "date", "y_int", "covid", "y_pred_base", "y_pred_wx", "difference"]
    nb_data[pred_cols].to_csv(f"{OUT_DIR}predictions_{indicator}.csv", index=False)

    print(f"  [{indicator}] Done in {time.time() - t0:.1f}s")

    return {
        "indicator": indicator,
        "label": INDICATOR_LABELS.get(indicator, indicator),
        "deficit_pct": deficit_pt,
        "ci_lo": deficit_ci[0],
        "ci_hi": deficit_ci[1],
        "hot_deficit_pct": hot_deficit_pt,
        "hot_ci_lo": hot_deficit_ci[0],
        "hot_ci_hi": hot_deficit_ci[1],
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
        "time": time.time() - t0,
        "_shifts": SHIFTS,
        "_design_map": DESIGN,
        "_spline_cols": spline_cols,
        "_train_facs": list(FITTED_FACILITIES),
        "_model_wx": model_wx,
        "_model_base": model_base,
        "_fac_district": nb_data[["facility", CLUSTER_COL]].drop_duplicates().reset_index(drop=True),
        "_tlo_rows": tlo_rows,
    }


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED Two-Model NB Analysis")
    print(f"Indicators: {len(COUNT_INDICATORS)}")
    print(f"Parallel: {USE_PARALLEL} (workers={N_WORKERS})")
    print(f"Spline DF: {SPLINE_DF}, Lags: {LAG_MONTHS}")
    print(f"ONLY_DEFICITS: {ONLY_DEFICITS}  (file suffix: {SUFFIX or '<none>'})")
    print(f"High-overdispersion indicators: {HIGH_OVERDISPERSION_INDICATORS}")
    print("=" * 60)

    panel_paths = {ind: f"{PANEL_DIR}regression_panel_{ind}.csv" for ind in COUNT_INDICATORS}

    t_start = time.time()

    if USE_PARALLEL and len(COUNT_INDICATORS) > 1:
        print(f"\nRunning {len(COUNT_INDICATORS)} indicators in parallel...")
        with Pool(processes=N_WORKERS) as pool:
            results = pool.starmap(fit_indicator, [(ind, panel_paths[ind]) for ind in COUNT_INDICATORS])
        for i, result in enumerate(results):
            if result is not None:
                print(
                    f"  [{i + 1}/{len(COUNT_INDICATORS)}] "
                    f"{result['indicator']}: "
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

    print(f"\n{'=' * 60}")
    print(f"All {len(results)} indicators fitted in {time.time() - t_start:.1f}s")
    print(f"Average per indicator: {(time.time() - t_start) / len(results):.1f}s")
    print("=" * 60)

    summary_df = pd.DataFrame(
        [
            {
                "indicator": r["indicator"],
                "label": r["label"],
                "only_deficits": ONLY_DEFICITS,
                "deficit_pct": r["deficit_pct"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "se_jackknife": r.get("se_jackknife"),
                "hot_deficit_pct": r["hot_deficit_pct"],
                "hot_ci_lo": r["hot_ci_lo"],
                "hot_ci_hi": r["hot_ci_hi"],
                "hot_threshold": r["hot_threshold"],
                "hot_se_jackknife": r.get("hot_se_jackknife"),
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
        f"{OUT_DIR}two_model_deficit_results_NB_{WBGT_VAR}{SUFFIX}.csv",
        index=False,
    )
    summary_df.to_csv(
        f"{OUT_DIR}summary_all_indicators_{WBGT_VAR}{SUFFIX}.csv",
        index=False,
    )

    irr_df = pd.DataFrame(
        [
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
        ]
    )
    # IRR is model-derived — no suffix needed, same values under both toggles.
    irr_df.to_csv(f"{OUT_DIR}irr_contrast_{WBGT_VAR}.csv", index=False)

    tlo_frames = [
        r["_tlo_rows"] for r in results if isinstance(r.get("_tlo_rows"), pd.DataFrame) and not r["_tlo_rows"].empty
    ]
    if tlo_frames:
        pd.concat(tlo_frames, ignore_index=True).to_csv(f"{OUT_DIR}tlo_wbgt_lookup.csv", index=False)
    else:
        print("  no TLO rows to write")

    curve_paths = sorted(Path(OUT_DIR).glob(f"exposure_response_curve_*_{WBGT_VAR}.csv"))
    if curve_paths:
        pd.concat([pd.read_csv(p) for p in curve_paths], ignore_index=True).to_csv(
            Path(OUT_DIR) / f"exposure_response_curves_{WBGT_VAR}.csv",
            index=False,
        )

    # =======================================================================
    # FORWARD PROJECTIONS
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
                pd.to_datetime(wide.index.astype(str).str.strip(), format="%Y-%m", errors="coerce")
                .to_period("M")
                .to_timestamp()
            )
            assert not wide.index.isna().any(), f"{path}: unparseable dates"
            wide.index.name = "date"
            wide.columns = wide.columns.astype(str).str.strip()
            return (
                wide.stack(future_stack=True).rename(value_name).rename_axis(index=["date", "facility"]).reset_index()
            )

        def _load_future_climate(ssp, tier):
            wbgt_path = os.path.join(PROJECTION_DIR, WBGT_PROJ_FILE_TPL.format(ssp=ssp, tier=tier))
            if not os.path.exists(wbgt_path):
                return None, [wbgt_path]
            clim = pd.read_csv(wbgt_path, parse_dates=["date"])
            clim["facility"] = clim["facility"].astype(str).str.strip()
            clim["date"] = clim["date"].dt.to_period("M").dt.to_timestamp()
            for col in (WBGT_VAR, PRECIP_COL):
                if col not in clim.columns:
                    raise KeyError(
                        f"{wbgt_path}: {col!r} missing (have {list(clim.columns)}). "
                        f"Re-run the panel producer to include it."
                    )
            clim = clim.sort_values(["facility", "date"]).reset_index(drop=True)
            for k in LAG_MONTHS:
                clim[f"{WBGT_VAR}_lag{k}"] = clim.groupby("facility")[WBGT_VAR].shift(k)
            clim = clim[clim["date"].dt.year.between(min_year_projection, max_year_projection)].copy()
            clim["year"] = clim["date"].dt.year
            clim["month"] = clim["date"].dt.month
            return clim, None

        all_proj_summary = []
        all_annual_pooled = []

        for ssp in SSP_SCENARIOS:
            for tier in WBGT_MODELS:
                clim, missing = _load_future_climate(ssp, tier)
                if clim is None:
                    print(f"  SKIP {ssp}/{tier}: missing {missing}")
                    continue
                print(f"  {ssp}/{tier}: {len(clim):,} rows, {clim['facility'].nunique()} facilities")

                if clim is None:
                    print(f"  SKIP {ssp}/{tier}: missing {missing}")
                    continue
                print(f"  {ssp}/{tier}: {len(clim):,} rows, {clim['facility'].nunique()} facilities")

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

                    wbgt_needed = [WBGT_VAR, PRECIP_COL] + [f"{WBGT_VAR}_lag{k}" for k in LAG_MONTHS]
                    n_before_wbgt = len(df)
                    df = df.dropna(subset=wbgt_needed).reset_index(drop=True)
                    n_dropped_wbgt = n_before_wbgt - len(df)
                    if df.empty:
                        print(f"    {ind}: all rows dropped for missing climate inputs")
                        continue

                    df["covid"] = 0
                    df["year_c"] = (
                        (LAST_HIST_YEAR - shifts["year"]) if PROJECT_HOLD_YEAR else (df["year"] - shifts["year"])
                    )
                    df["precip_c"] = df[PRECIP_COL] - shifts.get("precip", 0.0)

                    xc = df[WBGT_VAR].values - shifts[WBGT_VAR]
                    B = np.asarray(
                        patsy.build_design_matrices([design_map[WBGT_VAR]], {"x": xc})[0],
                        dtype=float,
                    )
                    for i_col, c in enumerate(spline_cols):
                        df[c] = B[:, i_col]
                    for k in LAG_MONTHS:
                        df[f"{WBGT_VAR}_lag{k}_c"] = df[f"{WBGT_VAR}_lag{k}"] - shifts[WBGT_VAR]

                    need = ["covid", "year_c", "precip_c"] + list(spline_cols) + LAG_COLS
                    n_before = len(df)
                    df = df.dropna(subset=need).reset_index(drop=True)
                    n_dropped = n_before - len(df) + n_dropped_wbgt
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

                    # --- FULL facility-month table always written --------
                    # so downstream code can re-derive either view.
                    df[
                        [
                            "indicator",
                            "ssp",
                            "tier",
                            "facility",
                            CLUSTER_COL,
                            "year",
                            "month",
                            "date",
                            WBGT_VAR,
                            PRECIP_COL,
                            "mu_a",
                            "mu_b",
                            "Disruption",
                            "Deficit_Pct",
                        ]
                    ].to_csv(
                        f"{OUT_DIR}projection_facility_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
                        index=False,
                    )

                    # --- Aggregation frame: apply toggle if on -----------
                    df_agg = _apply_deficit_filter(df, "mu_b", "mu_a")
                    if df_agg.empty:
                        print(f"    {ind} [{ssp}/{tier}]: no deficit rows under ONLY_DEFICITS — skipping aggregates")
                        continue

                    # district × year × month
                    dist_agg = (
                        df_agg.groupby([CLUSTER_COL, "year", "month"])
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
                        f"{OUT_DIR}projection_district_{ind}_{ssp}_{tier}_{WBGT_VAR}{SUFFIX}.csv",
                        index=False,
                    )

                    # pooled monthly time series
                    mon_agg = (
                        df_agg.groupby(["year", "month"])
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
                    mon_agg["Year_Month"] = mon_agg["year"].astype(str) + "-" + mon_agg["month"].astype(str)
                    mon_agg["indicator"], mon_agg["ssp"], mon_agg["tier"] = ind, ssp, tier
                    mon_agg.to_csv(
                        f"{OUT_DIR}projection_monthly_{ind}_{ssp}_{tier}_{WBGT_VAR}{SUFFIX}.csv",
                        index=False,
                    )

                    # ANNUAL: pooled across facilities
                    ann_agg = (
                        df_agg.groupby("year")
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
                    ann_agg["Mean_Monthly_Disruption"] = ann_agg["Disruption"] / 12.0
                    ann_agg["indicator"], ann_agg["ssp"], ann_agg["tier"] = ind, ssp, tier
                    ann_agg.to_csv(
                        f"{OUT_DIR}projection_annual_{ind}_{ssp}_{tier}_{WBGT_VAR}{SUFFIX}.csv",
                        index=False,
                    )
                    all_annual_pooled.append(ann_agg)

                    # ANNUAL: district × year
                    dist_ann = (
                        df_agg.groupby([CLUSTER_COL, "year"])
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
                        f"{OUT_DIR}projection_district_annual_{ind}_{ssp}_{tier}_{WBGT_VAR}{SUFFIX}.csv",
                        index=False,
                    )

                    tot_a = float(df_agg["mu_a"].sum())
                    tot_b = float(df_agg["mu_b"].sum())
                    deficit_proj = (100.0 * (tot_b - tot_a) / tot_b) if tot_b > 0 else np.nan
                    mean_annual_disruption = (
                        (tot_b - tot_a) / df_agg["year"].nunique() if df_agg["year"].nunique() > 0 else np.nan
                    )
                    all_proj_summary.append(
                        {
                            "indicator": ind,
                            "ssp": ssp,
                            "tier": tier,
                            "only_deficits": ONLY_DEFICITS,
                            "period_start": min_year_projection,
                            "period_end": max_year_projection,
                            "n_facility_months_total": len(df),
                            "n_facility_months_used": len(df_agg),
                            "n_input_dropped": n_dropped,
                            "n_pred_nan": n_pred_nan,
                            "mean_wbgt": float(df_agg[WBGT_VAR].mean()),
                            "mean_precip_month": float(df_agg[PRECIP_COL].mean()),
                            "total_A_projected": tot_a,
                            "total_B_projected": tot_b,
                            "deficit_pct": deficit_proj,
                            "mean_annual_disruption": mean_annual_disruption,
                        }
                    )
                    print(
                        f"    {ind}: deficit_proj={deficit_proj:+.2f}% "
                        f"(rows used {len(df_agg):,}/{len(df):,}, "
                        f"dropped {n_dropped:,})"
                    )

        if all_proj_summary:
            pd.DataFrame(all_proj_summary).to_csv(
                f"{OUT_DIR}projection_summary_{WBGT_VAR}{SUFFIX}.csv",
                index=False,
            )
            print(f"\nProjection summary → {OUT_DIR}projection_summary_{WBGT_VAR}{SUFFIX}.csv")
        else:
            print("\nNo projections produced.")

        if all_annual_pooled:
            pd.concat(all_annual_pooled, ignore_index=True).to_csv(
                f"{OUT_DIR}projection_annual_all_{WBGT_VAR}{SUFFIX}.csv",
                index=False,
            )
            print(f"Annual pooled → {OUT_DIR}projection_annual_all_{WBGT_VAR}{SUFFIX}.csv")

    # ===========================================================================
    # COUNTERFACTUAL: 1940-1948 ERA5 climate ("if warming hadn't continued")
    # ===========================================================================
    # Predicts facility-month services under early-20th-century climate while
    # holding the secular trend at LAST_HIST_YEAR and covid at 0. The contrast
    # with the observed historical deficit isolates the burden attributable to
    # warming since ~1948, conditional on today's baseline demand and facility mix.
    COUNTERFACTUAL = True
    CF_LABEL = "ERA5_periindustrial_1940_1948"
    CF_WBGT_FILE = (
        f"/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices/wbgt_extreme_indices_facility_{CF_LABEL}.csv"
        if WBGT_VAR == "wbgt5x_day"
        else f"/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices/wbgt_monthly_mean_facility_{CF_LABEL}.csv"
    )
    if COUNTERFACTUAL:
        print(f"\nCounterfactual scenario: {CF_LABEL}")
        cf_path = CF_WBGT_FILE
        if not os.path.exists(cf_path):
            print(f"  SKIP counterfactual: missing {cf_path}")
        else:
            cf = pd.read_csv(cf_path, parse_dates=["date"])
            cf["facility"] = cf["facility"].astype(str).str.strip()
            cf["date"] = cf["date"].dt.to_period("M").dt.to_timestamp()
            for col in (WBGT_VAR, PRECIP_COL):
                if col not in cf.columns:
                    raise KeyError(f"{cf_path}: {col!r} missing (have {list(cf.columns)})")
            cf = cf.sort_values(["facility", "date"]).reset_index(drop=True)
            for k in LAG_MONTHS:
                cf[f"{WBGT_VAR}_lag{k}"] = cf.groupby("facility")[WBGT_VAR].shift(k)
            cf["year"], cf["month"] = cf["date"].dt.year, cf["date"].dt.month
            print(
                f"  {CF_LABEL}: {len(cf):,} rows, {cf['facility'].nunique()} facilities, "
                f"{cf['year'].min()}–{cf['year'].max()}"
            )

            cf_rows = []
            for res in results:
                ind = res["indicator"]
                shifts = res["_shifts"]
                design_map = res["_design_map"]
                spline_cols = res["_spline_cols"]
                train_facs = set(res["_train_facs"])
                model_wx = res["_model_wx"]
                model_base = res["_model_base"]

                df = cf[cf["facility"].isin(train_facs)].copy()
                wbgt_needed = [WBGT_VAR, PRECIP_COL] + [f"{WBGT_VAR}_lag{k}" for k in LAG_MONTHS]
                df = df.dropna(subset=wbgt_needed).reset_index(drop=True)
                if df.empty:
                    print(f"    {ind}: no facility overlap in counterfactual — skipping")
                    continue

                # Diagnostic: how far outside the fit range is the counterfactual WBGT?
                fit_lo, fit_hi = shifts[WBGT_VAR] - 5, shifts[WBGT_VAR] + 5  # approx; splines cr(df=3)
                frac_below = float((df[WBGT_VAR] < fit_lo).mean())
                if frac_below > 0.05:
                    print(
                        f"    {ind}: {100 * frac_below:.1f}% of CF WBGT below ~{fit_lo:.1f}°C "
                        f"(fit centre {shifts[WBGT_VAR]:.1f}°C) — spline extrapolation, treat with care"
                    )

                df["covid"] = 0
                df["year_c"] = LAST_HIST_YEAR - shifts["year"]  # freeze secular trend
                df["precip_c"] = df[PRECIP_COL] - shifts.get("precip", 0.0)

                xc = df[WBGT_VAR].values - shifts[WBGT_VAR]
                B = np.asarray(patsy.build_design_matrices([design_map[WBGT_VAR]], {"x": xc})[0], dtype=float)
                for i_col, c in enumerate(spline_cols):
                    df[c] = B[:, i_col]
                for k in LAG_MONTHS:
                    df[f"{WBGT_VAR}_lag{k}_c"] = df[f"{WBGT_VAR}_lag{k}"] - shifts[WBGT_VAR]

                need = (
                    ["covid", "year_c", "precip_c"] + list(spline_cols) + [f"{WBGT_VAR}_lag{k}_c" for k in LAG_MONTHS]
                )
                df = df.dropna(subset=need).reset_index(drop=True)
                if df.empty:
                    continue

                mu_wx = np.asarray(model_wx.predict(df), dtype=float)
                mu_base = np.asarray(model_base.predict(df), dtype=float)
                ok = np.isfinite(mu_wx) & np.isfinite(mu_base)
                df = df.loc[ok].reset_index(drop=True)
                mu_wx, mu_base = mu_wx[ok], mu_base[ok]
                if df.empty:
                    continue

                df["mu_a"], df["mu_b"] = mu_wx, mu_base
                df = df.merge(res["_fac_district"], on="facility", how="left")
                df["indicator"] = ind

                df[
                    [
                        "indicator",
                        "facility",
                        CLUSTER_COL,
                        "year",
                        "month",
                        "date",
                        WBGT_VAR,
                        PRECIP_COL,
                        "mu_a",
                        "mu_b",
                    ]
                ].to_csv(
                    f"{OUT_DIR}counterfactual_facility_{ind}_{CF_LABEL}_{WBGT_VAR}.csv",
                    index=False,
                )

                df_agg = _apply_deficit_filter(df, "mu_b", "mu_a")
                if df_agg.empty:
                    continue
                tot_a, tot_b = float(df_agg["mu_a"].sum()), float(df_agg["mu_b"].sum())
                cf_deficit = (100.0 * (tot_b - tot_a) / tot_b) if tot_b > 0 else np.nan

                # Jackknife CI on the counterfactual deficit
                _, cf_lo, cf_hi = _monthly_jackknife_ci_local(
                    df_agg["mu_a"].values,
                    df_agg["mu_b"].values,
                    df_agg["facility"].values,
                    sign="a_minus_b",
                )

                hist_row = summary_df.loc[summary_df["indicator"] == ind]
                hist_deficit = float(hist_row["deficit_pct"].iloc[0]) if not hist_row.empty else np.nan

                cf_rows.append(
                    {
                        "indicator": ind,
                        "scenario": CF_LABEL,
                        "only_deficits": ONLY_DEFICITS,
                        "n_facility_months": len(df_agg),
                        "n_facilities": df_agg["facility"].nunique(),
                        "mean_wbgt_cf": float(df_agg[WBGT_VAR].mean()),
                        "mean_precip_cf": float(df_agg[PRECIP_COL].mean()),
                        "deficit_pct_cf": cf_deficit,
                        "cf_ci_lo": cf_lo,
                        "cf_ci_hi": cf_hi,
                        "deficit_pct_historical": hist_deficit,
                        "excess_deficit_attributable_to_warming_pp": hist_deficit - cf_deficit,
                        "frac_wbgt_below_fit_range": frac_below,
                    }
                )
                print(
                    f"    {ind}: cf={cf_deficit:+.2f}%  hist={hist_deficit:+.2f}%  "
                    f"excess={hist_deficit - cf_deficit:+.2f}pp"
                )

            if cf_rows:
                pd.DataFrame(cf_rows).to_csv(
                    f"{OUT_DIR}counterfactual_summary_{CF_LABEL}_{WBGT_VAR}{SUFFIX}.csv",
                    index=False,
                )
                print(f"\nCounterfactual summary → counterfactual_summary_{CF_LABEL}_{WBGT_VAR}{SUFFIX}.csv")
    print(f"\nSummary (HOT MONTHS ONLY, ONLY_DEFICITS={ONLY_DEFICITS}):")
    print(summary_df[["indicator", "hot_deficit_pct", "hot_ci_lo", "hot_ci_hi", "n_hot_obs"]].to_string())
