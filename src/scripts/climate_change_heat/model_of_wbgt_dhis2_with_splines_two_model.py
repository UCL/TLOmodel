"""
model_of_wbgt_dhis2_climatology.py

Two-model NB WBGT-service analysis, revised to identify heat effects off
WITHIN-FACILITY-MONTH ANOMALIES rather than raw WBGT levels.

KEY MODELLING CHOICE
--------------------
For each facility, compute a climatology: the long-run mean of WBGT (and
precipitation) for each calendar month. Add these as continuous controls
to the model. The WBGT spline coefficient is then identified off
DEVIATIONS of this-year's WBGT from this-facility's typical value in this
calendar month — i.e., heat anomalies — rather than off cross-month or
cross-facility variation, which is confounded with seasonal accessibility.

    y ~ spline(wbgt) + spline(wbgt_climatology) + lags
        + precip_c + precip_climatology + controls + FE

Reference: Deschênes & Greenstone (2011, AEJ Applied); Auffhammer et al.
(2013, REEP). The exposure is still WBGT in physical units (so
exposure-response curves and projections remain interpretable), but the
identifying variation is the anomaly.

This replaces the season × spline parameterisation of the previous
version. Season interactions doubled the spline parameter count and
caused most indicators to fail during NB fitting. This version uses ONE
extra spline (climatology) instead of doubling the exposure spline, and
achieves the same accessibility-adjustment purpose more parsimoniously.
"""

import os
import warnings
from multiprocessing import Pool, cpu_count
import time
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

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
]

INDICATOR_LABELS: dict[str, str] = {
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
    "htc_results_new_negative": "HTC New Negative",
    "htc_results_new_positive": "HTC New Positive",
    "fp_subsequent_clients_total": "FP Subsequent Clients",
    "fully_immunised_outreach": "Fully Immunised Under-1 (Outreach)",
    "cervical_screening_total": "Cervical Screening (Total)",
}

HIGH_OVERDISPERSION_INDICATORS = [
    "ipd_total_admissions",
    "opd_attendance",
    "anc_total_visits",
    "cervical_screening_initial",
]

MIN_YEAR_BY_INDICATOR: dict[str, int] = {
    "fp_total_clients": 2019,
    "vmmc_first_visits": 2019,
    "htc_results_new_negative": 2019,
    "htc_results_new_positive": 2019,
    "cervical_screening_initial": 2020,
    "fully_immunised_outreach": 2018,
}

ONLY_DEFICITS = False
SUFFIX = "_onlydeficits" if ONLY_DEFICITS else ""

# --- Measles outbreak control ------------------------------------------------
MEASLES_OUTBREAK_START = "2019-09-01"
MEASLES_OUTBREAK_END = "2020-08-01"
MEASLES_OUTBREAK_2_START = "2022-01-01"
MEASLES_OUTBREAK_2_END = "2022-06-01"

# --- Climatology settings ----------------------------------------------------
# Use facility × month-of-year climatology as a control. This means we
# identify the WBGT effect off within-facility-month variation across years.
# When a facility has too few observations for a given calendar month to
# form a reliable climatology, we fall back to facility-mean across all
# months.
MIN_CLIMATOLOGY_OBS = 3  # min years of data per facility-month cell

# --- Model settings ----------------------------------------------------------
WBGT_VAR = "wbgt_day"
SPLINE_DF = 2
LAG_MONTHS = [1, 2, 3]
CENTER = True
MIN_OBS = 24

COVID_WINDOW = ("2020-04-01", "2021-12-01")
CLUSTER_COL = "Dist"
USE_PRECIP = True
PRECIP_COL = "precip_month"

REFERENCE_WBGT_PERCENTILE = 95
IRR_LOW_PCTILE = 25
IRR_HIGH = 32.0
TLO_WBGT_GRID = np.linspace(20.0, 34.0, 57)
FDR_ALPHA = 0.05
CURVE_N = 60

CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre", "2023-03-01", "2024-03-01"),
]

min_year_historical = 2016
max_year_historical = 2025
LAST_HIST_YEAR = max_year_historical - 1

PROJECT = True
PROJECT_HOLD_YEAR = True
SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
WBGT_MODELS = ["lowest", "median", "highest"]
min_year_projection = 2025
max_year_projection = 2041

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
PANEL_DIR = f"{DATA_DIR}/Thermofeel_WBGT/Indices/"
os.makedirs(OUT_DIR, exist_ok=True)

USE_PARALLEL = False
N_WORKERS = min(cpu_count() - 1, 4)

WINSORIZE_BY_INDICATOR: dict[str, float] = {
    "opd_attendance": 0.99,
    "ipd_total_admissions": 1,
    "fully_immunised_under1": 0.95,
    "pnc_within_2wks": 0.95,
    "measles1_under1": 0.95,
    "anc_total_visits": 1,
}
WINSORIZE_DEFAULT = 0.999

CLIP_PROJECTION_TO_SUPPORT = True
SUPPORT_LOW_PCTILE = 1.0
SUPPORT_HIGH_PCTILE = 99.0


# ===========================================================================
# Helpers
# ===========================================================================

def _apply_deficit_filter(df, base_col, wx_col):
    if not ONLY_DEFICITS:
        return df
    return df[df[base_col] > df[wx_col]].copy()


def _clip_wbgt_to_support(df, support, wbgt_var=WBGT_VAR, lag_months=LAG_MONTHS):
    if not CLIP_PROJECTION_TO_SUPPORT:
        return df, {"frac_clipped_lo": 0.0, "frac_clipped_hi": 0.0,
                    "n_clipped_lo": 0, "n_clipped_hi": 0}
    df = df.copy()
    lo, hi = support["p_lo"], support["p_hi"]
    n_lo = int((df[wbgt_var] < lo).sum())
    n_hi = int((df[wbgt_var] > hi).sum())
    n_tot = max(len(df), 1)
    df[wbgt_var] = df[wbgt_var].clip(lower=lo, upper=hi)
    for k in lag_months:
        lag_col = f"{wbgt_var}_lag{k}"
        if lag_col in df.columns:
            df[lag_col] = df[lag_col].clip(lower=lo, upper=hi)
    return df, {
        "frac_clipped_lo": n_lo / n_tot, "frac_clipped_hi": n_hi / n_tot,
        "n_clipped_lo": n_lo, "n_clipped_hi": n_hi,
        "clip_lo_bound": lo, "clip_hi_bound": hi,
    }


def _facility_jackknife_ci(mu_a, mu_b, facility_ids, sign="b_minus_a"):
    """Vectorised leave-one-facility-out 95% CI on deficit_pct."""
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    facility_ids = np.asarray(facility_ids)
    total_a, total_b = mu_a.sum(), mu_b.sum()
    if total_b <= 0:
        return np.nan, np.nan, np.nan

    def _stat(sa, sb):
        return (100.0 * (sa - sb) / sb) if sign == "a_minus_b" else (100.0 * (sb - sa) / sb)

    pt = _stat(total_a, total_b)
    facs = np.unique(facility_ids)
    if len(facs) < 3:
        return pt, np.nan, np.nan

    fac_df = pd.DataFrame({"fac": facility_ids, "a": mu_a, "b": mu_b})
    fac_sums = fac_df.groupby("fac")[["a", "b"]].sum()
    loo_a = total_a - fac_sums["a"].values
    loo_b = total_b - fac_sums["b"].values
    valid = loo_b > 0
    if valid.sum() < 3:
        return pt, np.nan, np.nan
    if sign == "a_minus_b":
        jack = 100.0 * (loo_a[valid] - loo_b[valid]) / loo_b[valid]
    else:
        jack = 100.0 * (loo_b[valid] - loo_a[valid]) / loo_b[valid]
    n = len(jack)
    se = np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))
    return pt, pt - 1.96 * se, pt + 1.96 * se


def estimate_alpha_fast(data, y_col="y_int"):
    try:
        mod_pois = smf.glm(f"{y_col} ~ 1", data=data, family=sm.families.Poisson()).fit()
        mu = mod_pois.fittedvalues
        pearson_resid = (data[y_col] - mu) / np.sqrt(mu + 1e-10)
        dispersion = (pearson_resid ** 2).sum() / mod_pois.df_resid
        alpha = max(0.01, (dispersion - 1) / mu.mean())
    except Exception:
        mean_y = data[y_col].mean()
        var_y = data[y_col].var()
        alpha = max(0.01, (var_y - mean_y) / (mean_y ** 2)) if mean_y > 0 else 0.1
    alpha = min(alpha, 5.0)
    if alpha == 5.0:
        print("   alpha capped at 5.0 for stability")
    return alpha


def estimate_alpha_robust(data, indicator=None, y_col="y_int"):
    if indicator in HIGH_OVERDISPERSION_INDICATORS:
        print(f"  [{indicator}] Using intercept-only alpha")
        return estimate_alpha_fast(data, y_col)
    try:
        mod_pois = smf.glm(f"{y_col} ~ year_c + covid",
                           data=data, family=sm.families.Poisson()).fit()
        mu = np.clip(mod_pois.fittedvalues, 1e-6, None)
        pearson_chi2 = ((data[y_col] - mu) ** 2 / mu + 1e-10).sum()
        dispersion = pearson_chi2 / mod_pois.df_resid
        alpha = max(0.01, (dispersion - 1) / mu.mean())
        alpha = min(alpha, 5.0)
        if alpha == 5.0:
            print("   alpha capped at 5.0 for stability")
        return alpha
    except Exception:
        return estimate_alpha_fast(data, y_col)


def fit_negbin_fixed_alpha(formula, data, groups, alpha):
    mod = smf.glm(formula=formula, data=data,
                  family=sm.families.NegativeBinomial(alpha=alpha)).fit(
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
    print(f"  Var/Mean:     {y.var() / y.mean():.2f}")
    print(f"  Max:          {y.max():.0f}")


def winsorize_indicator(df, indicator_col="y", facility_col="facility",
                        upper_quantile=0.999):
    df = df.copy()
    def _cap(g):
        return g[indicator_col].clip(upper=g[indicator_col].quantile(upper_quantile))
    original = df[indicator_col].copy()
    df[indicator_col] = df.groupby(facility_col, group_keys=False).apply(_cap)
    print(f"  [WINSORIZE] {indicator_col}: {(df[indicator_col] != original).sum()} capped")
    return df


# ===========================================================================
# Climatology construction
# ===========================================================================

def add_climatology(df, indicator=None):
    """Add facility × month-of-year long-run means for WBGT and precipitation.

    For each facility-month cell with at least MIN_CLIMATOLOGY_OBS years of
    data, compute the mean. For cells with fewer observations, fall back to
    the facility's overall mean (across all months) so we don't lose those
    rows. The fallback rate is reported.

    Also creates ANOMALY columns (raw − climatology) for reporting and
    downstream inspection, though the model uses climatology as a control
    with raw WBGT in the spline (which is equivalent for coefficient
    identification but keeps units interpretable).
    """
    df = df.copy()

    # Per facility × month climatology
    fm_stats = (df.groupby(["facility", "month"])
                .agg(wbgt_fm=(WBGT_VAR, "mean"),
                     wbgt_fm_n=(WBGT_VAR, "count"),
                     precip_fm=(PRECIP_COL, "mean"),
                     precip_fm_n=(PRECIP_COL, "count"))
                .reset_index())
    # Facility fallback (all months pooled)
    f_stats = (df.groupby("facility")
               .agg(wbgt_f=(WBGT_VAR, "mean"),
                    precip_f=(PRECIP_COL, "mean"))
               .reset_index())

    df = df.merge(fm_stats, on=["facility", "month"], how="left")
    df = df.merge(f_stats, on="facility", how="left")

    # Use facility-month climatology where reliable; fallback otherwise
    use_fm = df["wbgt_fm_n"] >= MIN_CLIMATOLOGY_OBS
    df["wbgt_climatology"] = np.where(use_fm, df["wbgt_fm"], df["wbgt_f"])
    df["precip_climatology"] = np.where(use_fm, df["precip_fm"], df["precip_f"])

    # Anomalies (for diagnostic / reporting only; the model uses raw WBGT +
    # climatology as a control)
    df["wbgt_anomaly"] = df[WBGT_VAR] - df["wbgt_climatology"]
    df["precip_anomaly"] = df[PRECIP_COL] - df["precip_climatology"]

    fallback_rate = 1.0 - use_fm.mean()
    tag = f"[{indicator}] " if indicator else ""
    print(f"  {tag}Climatology: {(1 - fallback_rate) * 100:.1f}% of rows use "
          f"facility-month mean (≥{MIN_CLIMATOLOGY_OBS} yrs), "
          f"{fallback_rate * 100:.1f}% fall back to facility mean")
    print(f"  {tag}WBGT anomaly range: [{df['wbgt_anomaly'].min():.2f}, "
          f"{df['wbgt_anomaly'].max():.2f}] °C  "
          f"(sd={df['wbgt_anomaly'].std():.2f})")

    # Drop intermediate helper columns
    return df.drop(columns=["wbgt_fm", "wbgt_fm_n", "precip_fm",
                            "precip_fm_n", "wbgt_f", "precip_f"])


# ===========================================================================
# Weather column construction (climatology-controlled)
# ===========================================================================

def add_weather_columns_climatology(df, shifts, spline_design=None,
                                    lag_months=LAG_MONTHS):
    """Build weather RHS.

    Includes:
      - WBGT spline (main exposure; coefficient identified off anomalies
        because climatology is also in the model)
      - WBGT climatology spline (absorbs seasonal accessibility gradient)
      - Lagged WBGT (linear, main effect only)
      - Precipitation main effect + climatology
      - COVID / measles / year_c controls

    Returns
    -------
    df : DataFrame with all constructed columns
    rhs : list of column names for the full RHS
    spline_cols : main WBGT spline column names (the exposure of interest)
    clim_spline_cols : WBGT climatology spline column names (control)
    lag_cols : lag column names (centred)
    design_map : dict {name: patsy design_info}
    """
    df = df.sort_values(["facility", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_c"] = df["year"] - shifts["year"]

    # COVID
    lo, hi = pd.Timestamp(COVID_WINDOW[0]), pd.Timestamp(COVID_WINDOW[1])
    df["covid"] = df["date"].between(lo, hi).astype(int)

    # Measles outbreak
    df["measles_outbreak"] = df["date"].between(
        MEASLES_OUTBREAK_START, MEASLES_OUTBREAK_END).astype(int)
    df["measles_outbreak_2022"] = df["date"].between(
        MEASLES_OUTBREAK_2_START, MEASLES_OUTBREAK_2_END).astype(int)
    df["measles_outbreak_any"] = (
        df["measles_outbreak"] | df["measles_outbreak_2022"]
    ).astype(int)

    # Centred precipitation
    if USE_PRECIP:
        df["precip_c"] = df[PRECIP_COL] - shifts.get("precip", 0.0)
        # Precipitation climatology also centred
        df["precip_clim_c"] = df["precip_climatology"] - shifts.get("precip", 0.0)

    # Controls: raw covariates + climatology controls for precip
    rhs = ["year_c", "covid", "measles_outbreak_any"]
    if USE_PRECIP:
        rhs.extend(["precip_c", "precip_clim_c"])

    spline_cols: list[str] = []
    clim_spline_cols: list[str] = []
    lag_cols: list[str] = []
    design_map = {} if spline_design is None else spline_design

    v = WBGT_VAR
    xc = df[v] - shifts[v]
    xc_clim = df["wbgt_climatology"] - shifts[v]  # centre with same shift

    # Main WBGT spline
    if "wbgt_main" not in design_map:
        B = patsy.dmatrix(f"cr(x, df={SPLINE_DF}) - 1", {"x": xc},
                          return_type="dataframe")
        design_map["wbgt_main"] = B.design_info
    else:
        B = patsy.build_design_matrices([design_map["wbgt_main"]], {"x": xc})[0]
        B = pd.DataFrame(np.asarray(B))
    B_arr = np.asarray(B)
    for i in range(B_arr.shape[1]):
        col = f"{v}_s{i + 1}"
        df[col] = B_arr[:, i]
        spline_cols.append(col)
    rhs.extend(spline_cols)

    # WBGT CLIMATOLOGY spline (control) — same basis so units line up
    B_clim = patsy.build_design_matrices([design_map["wbgt_main"]],
                                          {"x": xc_clim})[0]
    B_clim_arr = np.asarray(B_clim)
    for i in range(B_clim_arr.shape[1]):
        col = f"{v}_clim_s{i + 1}"
        df[col] = B_clim_arr[:, i]
        clim_spline_cols.append(col)
    rhs.extend(clim_spline_cols)

    # Lags (main-effect linear)
    for lag in lag_months:
        lc = f"{v}_lag{lag}_c"
        df[lc] = df.groupby("facility")[v].shift(lag) - shifts[v]
        rhs.append(lc)
        lag_cols.append(lc)

    return df, rhs, spline_cols, clim_spline_cols, lag_cols, design_map


# ===========================================================================
# Exposure-response curve
# ===========================================================================

def exposure_response_curve(model, spline_cols, design_info_key, design_map,
                            wbgt_shift, ref_mode, wbgt_values, n=CURVE_N):
    """RR(wbgt) vs a reference, computed from the spline contrast on the
    WBGT exposure coefficients. With climatology in the model, this is now
    the effect of a *within-facility-month anomaly* of the given magnitude:
    holding climatology fixed, moving WBGT to X changes services by RR(X).
    """
    wobs = np.asarray(wbgt_values).flatten()
    grid = np.linspace(np.percentile(wobs, 1), np.percentile(wobs, 99), n)
    ref = wobs.mean() if ref_mode == "mean" else float(ref_mode)

    Bg = np.asarray(patsy.build_design_matrices(
        [design_map[design_info_key]], {"x": grid - wbgt_shift})[0])
    Br = np.asarray(patsy.build_design_matrices(
        [design_map[design_info_key]], {"x": np.array([ref]) - wbgt_shift})[0])
    contrast = Bg - Br

    beta = model.params.reindex(spline_cols).values
    V = model.cov_params().reindex(index=spline_cols,
                                   columns=spline_cols).values
    log_rr = contrast @ beta
    var = np.einsum("ij,jk,ik->i", contrast, V, contrast)
    se = np.sqrt(np.clip(var, 0, None))
    return pd.DataFrame({
        "wbgt": grid,
        "wbgt_ref": ref,
        "rr_vs_ref": np.exp(log_rr),
        "rr_lo": np.exp(log_rr - 1.96 * se),
        "rr_hi": np.exp(log_rr + 1.96 * se),
    })


# ===========================================================================
# Main fitting function
# ===========================================================================

def fit_indicator(indicator, panel_path):
    print(f"\n→ {indicator}")
    t0 = time.time()

    try:
        long = pd.read_csv(panel_path, parse_dates=["date"])
        long = long.rename(columns={indicator: "y"})
    except Exception as e:
        print(f"  [{indicator}] Failed to load: {e}")
        return None
    if USE_PRECIP and PRECIP_COL not in long.columns:
        print(f"  [{indicator}] Missing {PRECIP_COL}")
        return None
    if CLUSTER_COL not in long.columns:
        print(f"  [{indicator}] Missing {CLUSTER_COL}")
        return None

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
    long = winsorize_indicator(long, "y", "facility", wq)
    diagnose_indicator(long, "y")

    scale_factor = 100 if indicator == "ipd_total_admissions" else 1
    if scale_factor > 1:
        long["y"] = long["y"] / scale_factor
        print(f"  [{indicator}] Scaling response by {scale_factor}")

    # Compute climatology BEFORE centring, so climatology is in raw units
    long = add_climatology(long, indicator=indicator)

    SHIFTS = {
        "year": long["year"].mean() if CENTER else 0.0,
        WBGT_VAR: long[WBGT_VAR].mean() if CENTER else 0.0,
    }
    if USE_PRECIP:
        SHIFTS["precip"] = long[PRECIP_COL].mean() if CENTER else 0.0

    (long, weather_rhs, spline_cols, clim_spline_cols, lag_cols,
     DESIGN) = add_weather_columns_climatology(long, SHIFTS)

    nb_cols = ["y", "facility", "month", CLUSTER_COL] + weather_rhs
    nb_data = long.dropna(subset=nb_cols).copy()
    nb_data["y_int"] = nb_data["y"].round().clip(lower=0).astype(int)
    obs_nb = nb_data.groupby("facility").size()
    nb_data = nb_data[nb_data["facility"].isin(obs_nb[obs_nb >= MIN_OBS].index)].copy()
    FITTED_FACILITIES = set(nb_data["facility"].unique())

    print(f"  [{indicator}] Sample: {len(nb_data):,} obs, "
          f"{len(FITTED_FACILITIES)} facilities")

    wbgt_train = nb_data[WBGT_VAR].values
    WBGT_SUPPORT = {
        "min": float(np.min(wbgt_train)), "max": float(np.max(wbgt_train)),
        "p_lo": float(np.percentile(wbgt_train, SUPPORT_LOW_PCTILE)),
        "p_hi": float(np.percentile(wbgt_train, SUPPORT_HIGH_PCTILE)),
        "p05": float(np.percentile(wbgt_train, 5)),
        "p95": float(np.percentile(wbgt_train, 95)),
    }

    groups = nb_data[CLUSTER_COL]

    # --- Model formulas ------------------------------------------------------
    # Base: controls + climatology (both WBGT and precip) + FE. This is the
    # counterfactual "what services would we predict without any anomaly?".
    ctrl_base = ("year_c + covid + measles_outbreak_any"
                 + (" + precip_c + precip_clim_c" if USE_PRECIP else "")
                 + " + " + " + ".join(clim_spline_cols))
    FE = "C(month) + C(facility)"
    f_base = f"y_int ~ {ctrl_base} + {FE}"

    # Weather: adds the exposure spline + lags on top of base
    wx_extra = " + ".join(spline_cols + lag_cols)
    f_wx = f"y_int ~ {ctrl_base} + {wx_extra} + {FE}"

    print(f"  base: {f_base[:120]}...")
    print(f"  wx  : {f_wx[:120]}...")

    alpha = estimate_alpha_robust(nb_data, indicator=indicator)
    print(f"  [{indicator}] Alpha: {alpha:.4f}")

    try:
        model_base = fit_negbin_fixed_alpha(f_base, nb_data, groups, alpha)
        model_wx = fit_negbin_fixed_alpha(f_wx, nb_data, groups, alpha)
    except Exception as e:
        print(f"  [{indicator}] Model fitting failed: {e}")
        return None

    if scale_factor > 1:
        model_base.fittedvalues = model_base.fittedvalues * scale_factor
        model_wx.fittedvalues = model_wx.fittedvalues * scale_factor

    # --- Exposure-response curve (heat anomaly effect) ----------------------
    try:
        curve = exposure_response_curve(
            model_wx, spline_cols, "wbgt_main", DESIGN,
            SHIFTS[WBGT_VAR], "mean", nb_data[WBGT_VAR].values,
        )
        curve.insert(0, "indicator", indicator)
        curve.insert(1, "label", INDICATOR_LABELS.get(indicator, indicator))
        curve.to_csv(
            f"{OUT_DIR}exposure_response_curve_{indicator}_{WBGT_VAR}.csv",
            index=False,
        )
    except Exception as e:
        print(f"  [{indicator}] Curve failed: {e}")

    # --- Predictions ---------------------------------------------------------
    nb_data["y_pred_base"] = model_base.fittedvalues
    nb_data["y_pred_wx"] = model_wx.fittedvalues
    nb_data["difference"] = nb_data["y_pred_base"] - nb_data["y_pred_wx"]

    # --- Overall pooled deficit ---------------------------------------------
    deficit_pt, deficit_lo, deficit_hi = _facility_jackknife_ci(
        nb_data["y_pred_wx"].values, nb_data["y_pred_base"].values,
        nb_data["facility"].values, sign="b_minus_a"
    )
    print(f"  [{indicator}] Deficit (all): {deficit_pt:+.2f}% "
          f"(CI: {deficit_lo:+.2f}..{deficit_hi:+.2f})")

    # --- Hot-anomaly deficits ------------------------------------------------
    # Define "hot" as top REFERENCE_WBGT_PERCENTILE of the ANOMALY
    # distribution (not raw WBGT), because that's what climatology-adjusted
    # heat means. Also report raw-WBGT hot for comparability with prior runs.
    anom_threshold = np.percentile(nb_data["wbgt_anomaly"],
                                   REFERENCE_WBGT_PERCENTILE)
    hot_anom = nb_data[nb_data["wbgt_anomaly"] > anom_threshold]
    if len(hot_anom) > 10:
        hot_anom_pt, hot_anom_lo, hot_anom_hi = _facility_jackknife_ci(
            hot_anom["y_pred_wx"].values, hot_anom["y_pred_base"].values,
            hot_anom["facility"].values, sign="b_minus_a"
        )
        print(f"  [{indicator}] HOT ANOMALY (>+{anom_threshold:.2f}°C): "
              f"{hot_anom_pt:+.2f}% "
              f"(CI: {hot_anom_lo:+.2f}..{hot_anom_hi:+.2f})  n={len(hot_anom):,}")
    else:
        hot_anom_pt = hot_anom_lo = hot_anom_hi = np.nan

    # Also raw-WBGT hot (for continuity with previous runs)
    raw_threshold = np.percentile(nb_data[WBGT_VAR], REFERENCE_WBGT_PERCENTILE)
    hot_raw = nb_data[nb_data[WBGT_VAR] > raw_threshold]
    if len(hot_raw) > 10:
        hot_raw_pt, hot_raw_lo, hot_raw_hi = _facility_jackknife_ci(
            hot_raw["y_pred_wx"].values, hot_raw["y_pred_base"].values,
            hot_raw["facility"].values, sign="b_minus_a"
        )
        print(f"  [{indicator}] HOT RAW (>{raw_threshold:.2f}°C): "
              f"{hot_raw_pt:+.2f}% "
              f"(CI: {hot_raw_lo:+.2f}..{hot_raw_hi:+.2f})  n={len(hot_raw):,}")
    else:
        hot_raw_pt = hot_raw_lo = hot_raw_hi = np.nan

    # --- IRR contrast: WBGT=IRR_HIGH vs reference ---------------------------
    reference_wbgt = float(np.percentile(nb_data[WBGT_VAR], IRR_LOW_PCTILE))
    irr_pt = irr_lo = irr_hi = np.nan
    try:
        B_hi = np.asarray(patsy.build_design_matrices(
            [DESIGN["wbgt_main"]],
            {"x": np.array([IRR_HIGH]) - SHIFTS[WBGT_VAR]})[0])
        B_ref = np.asarray(patsy.build_design_matrices(
            [DESIGN["wbgt_main"]],
            {"x": np.array([reference_wbgt]) - SHIFTS[WBGT_VAR]})[0])
        contrast = (B_hi - B_ref).ravel()
        beta_s = model_wx.params.reindex(spline_cols).values
        V_s = model_wx.cov_params().reindex(
            index=spline_cols, columns=spline_cols).values
        log_irr = float(contrast @ beta_s)
        se_irr = float(np.sqrt(max(contrast @ V_s @ contrast, 0.0)))
        irr_pt = float(np.exp(log_irr))
        irr_lo = float(np.exp(log_irr - 1.96 * se_irr))
        irr_hi = float(np.exp(log_irr + 1.96 * se_irr))
    except Exception as e:
        print(f"  [{indicator}] IRR failed: {e}")

    # --- Wald test on WBGT anomaly terms ------------------------------------
    try:
        test_cols = ([c for c in spline_cols if c in model_wx.params.index]
                     + [c for c in lag_cols if c in model_wx.params.index])
        wt = model_wx.wald_test(test_cols, use_f=True, scalar=True)
        pval = float(wt.pvalue)
    except Exception as e:
        print(f"  [{indicator}] Wald test failed: {e}")
        pval = np.nan

    # --- Historical burden CSV ----------------------------------------------
    hb = pd.DataFrame({
        "date": nb_data["date"].values,
        "facility": nb_data["facility"].values,
        "month": nb_data["month"].values,
        CLUSTER_COL: nb_data[CLUSTER_COL].values,
        "y_int": nb_data["y_int"].values,
        "wbgt": nb_data[WBGT_VAR].values,
        "wbgt_climatology": nb_data["wbgt_climatology"].values,
        "wbgt_anomaly": nb_data["wbgt_anomaly"].values,
        "mu_a": nb_data["y_pred_wx"].values,
        "mu_b": nb_data["y_pred_base"].values,
    })
    hb.to_csv(f"{OUT_DIR}historical_burden_{indicator}_{WBGT_VAR}.csv",
              index=False)

    # --- District aggregate --------------------------------------------------
    hb_for_agg = _apply_deficit_filter(hb, "mu_b", "mu_a")
    district_agg = hb_for_agg.groupby(CLUSTER_COL)[["mu_a", "mu_b"]].sum().reset_index()
    district_agg["deficit_pct"] = np.where(
        district_agg["mu_b"] > 0,
        100.0 * (district_agg["mu_b"] - district_agg["mu_a"]) / district_agg["mu_b"],
        np.nan,
    )
    district_agg[[CLUSTER_COL, "deficit_pct"]].to_csv(
        f"{OUT_DIR}district_burden_{indicator}_{WBGT_VAR}{SUFFIX}.csv", index=False)

    dist_rows = []
    for dist, sub in hb_for_agg.groupby(CLUSTER_COL):
        pt, lo, hi = _facility_jackknife_ci(
            sub["mu_a"].values, sub["mu_b"].values,
            sub["facility"].values, sign="a_minus_b"
        )
        sig = bool(pd.notna(lo) and pd.notna(hi) and (lo * hi > 0))
        dist_rows.append({"district": dist, "deficit_pct": pt,
                          "ci_lo": lo, "ci_hi": hi, "sig": sig})
    pd.DataFrame(dist_rows).to_csv(
        f"{OUT_DIR}district_burden_ci_{indicator}_{WBGT_VAR}{SUFFIX}.csv",
        index=False)

    # --- TLO lookup rows ----------------------------------------------------
    try:
        Bg = np.asarray(patsy.build_design_matrices(
            [DESIGN["wbgt_main"]],
            {"x": TLO_WBGT_GRID - SHIFTS[WBGT_VAR]})[0])
        Br = np.asarray(patsy.build_design_matrices(
            [DESIGN["wbgt_main"]],
            {"x": np.array([reference_wbgt]) - SHIFTS[WBGT_VAR]})[0])
        beta_s = model_wx.params.reindex(spline_cols).values
        rr_grid = np.exp((Bg - Br) @ beta_s)
        tlo_rows = pd.DataFrame({
            "indicator": indicator,
            "wbgt": TLO_WBGT_GRID,
            "rr_vs_ref": rr_grid,
            "disruption_probability": np.clip(1.0 - rr_grid, 0.0, None),
        })
    except Exception as e:
        print(f"  [{indicator}] TLO rows failed: {e}")
        tlo_rows = pd.DataFrame()

    # --- Summary row ---------------------------------------------------------
    summary_row = {
        "indicator": indicator,
        "label": INDICATOR_LABELS.get(indicator, indicator),
        "only_deficits": ONLY_DEFICITS,
        "deficit_pct": deficit_pt, "ci_lo": deficit_lo, "ci_hi": deficit_hi,
        "hot_anom_deficit_pct": hot_anom_pt,
        "hot_anom_ci_lo": hot_anom_lo,
        "hot_anom_ci_hi": hot_anom_hi,
        "hot_anom_threshold": anom_threshold,
        "n_hot_anom_obs": len(hot_anom),
        "hot_raw_deficit_pct": hot_raw_pt,
        "hot_raw_ci_lo": hot_raw_lo,
        "hot_raw_ci_hi": hot_raw_hi,
        "hot_raw_threshold": raw_threshold,
        "n_hot_raw_obs": len(hot_raw),
        "n_facilities": len(FITTED_FACILITIES),
        "n_obs": len(nb_data),
        "n_districts": nb_data[CLUSTER_COL].nunique(),
        "alpha": alpha,
        "reference_wbgt": reference_wbgt,
        "irr_hi_vs_low": irr_pt,
        "irr_lo_bound": irr_lo,
        "irr_hi_bound": irr_hi,
        "pval": pval,
        "wbgt_train_min": WBGT_SUPPORT["min"],
        "wbgt_train_max": WBGT_SUPPORT["max"],
        "wbgt_train_p_lo": WBGT_SUPPORT["p_lo"],
        "wbgt_train_p_hi": WBGT_SUPPORT["p_hi"],
        "wbgt_anom_sd": float(nb_data["wbgt_anomaly"].std()),
        "wbgt_anom_p5": float(nb_data["wbgt_anomaly"].quantile(0.05)),
        "wbgt_anom_p95": float(nb_data["wbgt_anomaly"].quantile(0.95)),
        "time_seconds": time.time() - t0,
    }
    pd.DataFrame([summary_row]).to_csv(
        f"{OUT_DIR}deficit_{indicator}{SUFFIX}.csv", index=False)

    pred_cols = ["year", "month", "facility", "date", "y_int", "covid",
                 "measles_outbreak_any", "wbgt_anomaly",
                 "y_pred_base", "y_pred_wx", "difference"]
    keep = [c for c in pred_cols if c in nb_data.columns]
    nb_data[keep].to_csv(f"{OUT_DIR}predictions_{indicator}.csv", index=False)

    print(f"  [{indicator}] Done in {time.time() - t0:.1f}s")

    return {
        **summary_row,
        "time": time.time() - t0,
        "_shifts": SHIFTS,
        "_design_map": DESIGN,
        "_spline_cols": spline_cols,
        "_clim_spline_cols": clim_spline_cols,
        "_lag_cols": lag_cols,
        "_train_facs": list(FITTED_FACILITIES),
        "_model_wx": model_wx,
        "_model_base": model_base,
        "_fac_district": nb_data[["facility", CLUSTER_COL]].drop_duplicates().reset_index(drop=True),
        "_tlo_rows": tlo_rows,
        "_wbgt_support": WBGT_SUPPORT,
        # Store the fitted climatology for reuse in projection/CF
        "_fac_month_climatology": (
            nb_data[["facility", "month", "wbgt_climatology",
                     "precip_climatology"]]
            .drop_duplicates(subset=["facility", "month"])
            .reset_index(drop=True)
        ),
    }


# ===========================================================================
# Projection / counterfactual data builder
# ===========================================================================

def _build_prediction_df(clim_df, res, ssp=None, tier=None, scenario_label=None):
    """Assemble a prediction-ready dataframe using the HISTORICAL climatology.

    The climatology used as a control in the projection/CF is the SAME one
    estimated from historical data — the model predicts what would happen
    given historical baseline seasonality plus the new WBGT values. This
    means future WBGT values with the historical climatology absorbed
    contribute their marginal effect via the WBGT spline.
    """
    shifts = res["_shifts"]
    design_map = res["_design_map"]
    spline_cols = res["_spline_cols"]
    clim_spline_cols = res["_clim_spline_cols"]
    lag_cols = res["_lag_cols"]
    train_facs = set(res["_train_facs"])
    support = res["_wbgt_support"]
    fac_month_clim = res["_fac_month_climatology"]

    df = clim_df[clim_df["facility"].isin(train_facs)].copy()
    if df.empty:
        return None, None, 0, 0

    wbgt_needed = [WBGT_VAR, PRECIP_COL] + \
                  [f"{WBGT_VAR}_lag{k}" for k in LAG_MONTHS]
    n_before = len(df)
    df = df.dropna(subset=wbgt_needed).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if df.empty:
        return None, None, n_dropped, 0

    df, clip_diag = _clip_wbgt_to_support(df, support)

    df["covid"] = 0
    df["measles_outbreak_any"] = 0
    df["year_c"] = (LAST_HIST_YEAR - shifts["year"]) if PROJECT_HOLD_YEAR \
        else (df["year"] - shifts["year"])

    # Merge in HISTORICAL climatology by facility-month; if a facility-month
    # is missing from historical climatology (shouldn't be, but robust),
    # drop those rows
    df = df.merge(fac_month_clim, on=["facility", "month"], how="left")
    df = df.dropna(subset=["wbgt_climatology", "precip_climatology"]).reset_index(drop=True)
    if df.empty:
        return None, None, n_dropped, 0

    df["precip_c"] = df[PRECIP_COL] - shifts.get("precip", 0.0)
    df["precip_clim_c"] = df["precip_climatology"] - shifts.get("precip", 0.0)

    # Main WBGT spline
    xc = df[WBGT_VAR].values - shifts[WBGT_VAR]
    B = np.asarray(patsy.build_design_matrices(
        [design_map["wbgt_main"]], {"x": xc})[0], dtype=float)
    for i in range(B.shape[1]):
        df[f"{WBGT_VAR}_s{i + 1}"] = B[:, i]

    # WBGT climatology spline
    xc_clim = df["wbgt_climatology"].values - shifts[WBGT_VAR]
    B_clim = np.asarray(patsy.build_design_matrices(
        [design_map["wbgt_main"]], {"x": xc_clim})[0], dtype=float)
    for i in range(B_clim.shape[1]):
        df[f"{WBGT_VAR}_clim_s{i + 1}"] = B_clim[:, i]

    # Lags
    for k in LAG_MONTHS:
        df[f"{WBGT_VAR}_lag{k}_c"] = df[f"{WBGT_VAR}_lag{k}"] - shifts[WBGT_VAR]

    need = (["covid", "year_c", "precip_c", "precip_clim_c",
             "measles_outbreak_any"]
            + spline_cols + clim_spline_cols + lag_cols)
    df = df.dropna(subset=need).reset_index(drop=True)
    return df, clip_diag, n_dropped, 0


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("REVISED: WBGT anomaly identification via climatology control")
    print(f"Indicators: {len(COUNT_INDICATORS)}")
    print(f"Spline DF: {SPLINE_DF}, Lags: {LAG_MONTHS}")
    print(f"Min climatology obs per facility-month: {MIN_CLIMATOLOGY_OBS}")
    print(f"ONLY_DEFICITS: {ONLY_DEFICITS}")
    print("=" * 60)

    panel_paths = {ind: f"{PANEL_DIR}regression_panel_{ind}.csv"
                   for ind in COUNT_INDICATORS}

    t_start = time.time()
    if USE_PARALLEL and len(COUNT_INDICATORS) > 1:
        with Pool(processes=N_WORKERS) as pool:
            results = pool.starmap(fit_indicator,
                                   [(ind, panel_paths[ind])
                                    for ind in COUNT_INDICATORS])
    else:
        results = [fit_indicator(ind, panel_paths[ind])
                   for ind in COUNT_INDICATORS]

    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("No indicators fitted successfully!")

    print(f"\n{'=' * 60}")
    print(f"All {len(results)} indicators fitted in {time.time() - t_start:.1f}s")
    print("=" * 60)

    # --- Combined summary ---------------------------------------------------
    summary_cols = [
        "indicator", "label", "only_deficits",
        "deficit_pct", "ci_lo", "ci_hi",
        "hot_anom_deficit_pct", "hot_anom_ci_lo", "hot_anom_ci_hi",
        "hot_anom_threshold", "n_hot_anom_obs",
        "hot_raw_deficit_pct", "hot_raw_ci_lo", "hot_raw_ci_hi",
        "hot_raw_threshold", "n_hot_raw_obs",
        "n_facilities", "n_obs", "n_districts",
        "alpha", "reference_wbgt",
        "irr_hi_vs_low", "irr_lo_bound", "irr_hi_bound",
        "pval",
        "wbgt_anom_sd", "wbgt_anom_p5", "wbgt_anom_p95",
        "time",
    ]
    summary_df = pd.DataFrame([{k: r.get(k) for k in summary_cols}
                               for r in results])

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
    summary_df = summary_df.sort_values("hot_anom_deficit_pct", na_position="last")

    summary_df.to_csv(
        f"{OUT_DIR}two_model_deficit_results_NB_{WBGT_VAR}{SUFFIX}.csv",
        index=False)
    summary_df.to_csv(
        f"{OUT_DIR}summary_all_indicators_{WBGT_VAR}{SUFFIX}.csv",
        index=False)

    # --- Pooled IRR CSV -----------------------------------------------------
    irr_df = pd.DataFrame([{
        "indicator": r["indicator"],
        "label": r["label"],
        "reference_wbgt": r["reference_wbgt"],
        "irr_high_wbgt": IRR_HIGH,
        "irr": r["irr_hi_vs_low"],
        "irr_lo": r["irr_lo_bound"],
        "irr_hi": r["irr_hi_bound"],
    } for r in results])
    irr_df.to_csv(f"{OUT_DIR}irr_contrast_{WBGT_VAR}.csv", index=False)

    # --- Pooled TLO + curves ------------------------------------------------
    tlo_frames = [r["_tlo_rows"] for r in results
                  if isinstance(r.get("_tlo_rows"), pd.DataFrame)
                  and not r["_tlo_rows"].empty]
    if tlo_frames:
        pd.concat(tlo_frames, ignore_index=True).to_csv(
            f"{OUT_DIR}tlo_wbgt_lookup.csv", index=False)

    curve_paths = sorted(Path(OUT_DIR).glob(
        f"exposure_response_curve_*_{WBGT_VAR}.csv"))
    if curve_paths:
        pd.concat([pd.read_csv(p) for p in curve_paths],
                  ignore_index=True).to_csv(
            Path(OUT_DIR) / f"exposure_response_curves_{WBGT_VAR}.csv",
            index=False)

    # =======================================================================
    # FORWARD PROJECTIONS
    # =======================================================================
    if PROJECT:
        print(f"\nForward projections ({min_year_projection}-{max_year_projection})")
        PROJECTION_DIR = f"{DATA_DIR}/Thermofeel_WBGT/Indices"
        if WBGT_VAR == "wbgt_day":
            TPL = "wbgt_monthly_mean_facility_{tier}_{ssp}.csv"
        elif WBGT_VAR == "wbgt5x_day":
            TPL = "wbgt_extreme_indices_facility_{tier}_{ssp}.csv"
        else:
            raise ValueError(f"Unknown WBGT_VAR: {WBGT_VAR}")

        def _load_future(ssp, tier):
            path = os.path.join(PROJECTION_DIR, TPL.format(ssp=ssp, tier=tier))
            if not os.path.exists(path):
                return None, path
            clim = pd.read_csv(path, parse_dates=["date"])
            clim["facility"] = clim["facility"].astype(str).str.strip()
            clim["date"] = clim["date"].dt.to_period("M").dt.to_timestamp()
            for col in (WBGT_VAR, PRECIP_COL):
                if col not in clim.columns:
                    raise KeyError(f"{path}: {col} missing")
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
                clim, missing = _load_future(ssp, tier)
                if clim is None:
                    print(f"  SKIP {ssp}/{tier}: {missing}")
                    continue
                print(f"  {ssp}/{tier}: {len(clim):,} rows")

                for res in results:
                    ind = res["indicator"]
                    built = _build_prediction_df(clim, res, ssp=ssp, tier=tier)
                    df, clip_diag, n_dropped, _ = built
                    if df is None or df.empty:
                        continue

                    mu_wx = np.asarray(res["_model_wx"].predict(df), dtype=float)
                    mu_base = np.asarray(res["_model_base"].predict(df), dtype=float)
                    ok = np.isfinite(mu_wx) & np.isfinite(mu_base)
                    df = df.loc[ok].reset_index(drop=True)
                    mu_wx, mu_base = mu_wx[ok], mu_base[ok]
                    if df.empty:
                        continue

                    df["mu_a"] = mu_wx
                    df["mu_b"] = mu_base
                    df["Disruption"] = df["mu_b"] - df["mu_a"]
                    df["Deficit_Pct"] = np.where(
                        df["mu_b"] > 0,
                        100.0 * df["Disruption"] / df["mu_b"], np.nan)
                    df = df.merge(res["_fac_district"], on="facility", how="left")
                    df["indicator"], df["ssp"], df["tier"] = ind, ssp, tier

                    df_agg = _apply_deficit_filter(df, "mu_b", "mu_a")
                    if df_agg.empty:
                        continue

                    ann = (df_agg.groupby("year")
                           .agg(mu_a=("mu_a", "sum"), mu_b=("mu_b", "sum"),
                                Mean_WBGT=(WBGT_VAR, "mean"),
                                Mean_Precip_Month=(PRECIP_COL, "mean"),
                                N_Facilities=("facility", "nunique"))
                           .reset_index())
                    ann["Disruption"] = ann["mu_b"] - ann["mu_a"]
                    ann["Deficit_Pct"] = np.where(
                        ann["mu_b"] > 0,
                        100.0 * ann["Disruption"] / ann["mu_b"], np.nan)
                    ann["indicator"], ann["ssp"], ann["tier"] = ind, ssp, tier
                    ann.to_csv(
                        f"{OUT_DIR}projection_annual_{ind}_{ssp}_{tier}_{WBGT_VAR}{SUFFIX}.csv",
                        index=False)
                    all_annual_pooled.append(ann)

                    tot_a, tot_b = float(df_agg["mu_a"].sum()), float(df_agg["mu_b"].sum())
                    def_all = (100.0 * (tot_b - tot_a) / tot_b) if tot_b > 0 else np.nan
                    all_proj_summary.append({
                        "indicator": ind, "ssp": ssp, "tier": tier,
                        "period_start": min_year_projection,
                        "period_end": max_year_projection,
                        "n_facility_months": len(df_agg),
                        "mean_wbgt": float(df_agg[WBGT_VAR].mean()),
                        "total_A_projected": tot_a,
                        "total_B_projected": tot_b,
                        "deficit_pct": def_all,
                        "wbgt_train_p_lo": res["_wbgt_support"]["p_lo"],
                        "wbgt_train_p_hi": res["_wbgt_support"]["p_hi"],
                        "frac_clipped_lo": clip_diag["frac_clipped_lo"],
                        "frac_clipped_hi": clip_diag["frac_clipped_hi"],
                    })
                    print(f"    {ind} {ssp}/{tier}: overall={def_all:+.2f}%")

        if all_proj_summary:
            pd.DataFrame(all_proj_summary).to_csv(
                f"{OUT_DIR}projection_summary_{WBGT_VAR}{SUFFIX}.csv",
                index=False)
        if all_annual_pooled:
            pd.concat(all_annual_pooled, ignore_index=True).to_csv(
                f"{OUT_DIR}projection_annual_all_{WBGT_VAR}{SUFFIX}.csv",
                index=False)

    # =======================================================================
    # COUNTERFACTUAL
    # =======================================================================
    COUNTERFACTUAL = True
    CF_LABEL = "ERA5_periindustrial_1940_1948"
    CF_WBGT_FILE = (
        f"{DATA_DIR}/Thermofeel_WBGT/Indices/wbgt_extreme_indices_facility_{CF_LABEL}.csv"
        if WBGT_VAR == "wbgt5x_day"
        else f"{DATA_DIR}/Thermofeel_WBGT/Indices/wbgt_monthly_mean_facility_{CF_LABEL}.csv"
    )
    if COUNTERFACTUAL and os.path.exists(CF_WBGT_FILE):
        print(f"\nCounterfactual: {CF_LABEL}")
        cf = pd.read_csv(CF_WBGT_FILE, parse_dates=["date"])
        cf["facility"] = cf["facility"].astype(str).str.strip()
        cf["date"] = cf["date"].dt.to_period("M").dt.to_timestamp()
        cf = cf.sort_values(["facility", "date"]).reset_index(drop=True)
        for k in LAG_MONTHS:
            cf[f"{WBGT_VAR}_lag{k}"] = cf.groupby("facility")[WBGT_VAR].shift(k)
        cf["year"], cf["month"] = cf["date"].dt.year, cf["date"].dt.month

        cf_rows = []
        for res in results:
            ind = res["indicator"]
            built = _build_prediction_df(cf, res, scenario_label=CF_LABEL)
            df, clip_diag, _, _ = built
            if df is None or df.empty:
                continue

            mu_wx = np.asarray(res["_model_wx"].predict(df), dtype=float)
            mu_base = np.asarray(res["_model_base"].predict(df), dtype=float)
            ok = np.isfinite(mu_wx) & np.isfinite(mu_base)
            df = df.loc[ok].reset_index(drop=True)
            mu_wx, mu_base = mu_wx[ok], mu_base[ok]
            if df.empty:
                continue

            df["mu_a"], df["mu_b"] = mu_wx, mu_base
            df = df.merge(res["_fac_district"], on="facility", how="left")
            df["indicator"] = ind

            df_agg = _apply_deficit_filter(df, "mu_b", "mu_a")
            if df_agg.empty:
                continue

            tot_a, tot_b = float(df_agg["mu_a"].sum()), float(df_agg["mu_b"].sum())
            cf_deficit = (100.0 * (tot_b - tot_a) / tot_b) if tot_b > 0 else np.nan
            _, cf_lo, cf_hi = _facility_jackknife_ci(
                df_agg["mu_a"].values, df_agg["mu_b"].values,
                df_agg["facility"].values, sign="a_minus_b"
            )
            hist_deficit = float(summary_df.loc[
                summary_df["indicator"] == ind, "deficit_pct"].iloc[0])

            cf_rows.append({
                "indicator": ind, "scenario": CF_LABEL,
                "n_facility_months": len(df_agg),
                "mean_wbgt_cf": float(df_agg[WBGT_VAR].mean()),
                "deficit_pct_cf": cf_deficit,
                "cf_ci_lo": cf_lo, "cf_ci_hi": cf_hi,
                "deficit_pct_historical": hist_deficit,
                "excess_deficit_attributable_to_warming_pp":
                    hist_deficit - cf_deficit,
                "frac_clipped_lo": clip_diag["frac_clipped_lo"],
                "frac_clipped_hi": clip_diag["frac_clipped_hi"],
            })
            print(f"    {ind}: cf={cf_deficit:+.2f}%  hist={hist_deficit:+.2f}%  "
                  f"excess={hist_deficit - cf_deficit:+.2f}pp")

        if cf_rows:
            pd.DataFrame(cf_rows).to_csv(
                f"{OUT_DIR}counterfactual_summary_{CF_LABEL}_{WBGT_VAR}{SUFFIX}.csv",
                index=False)

    print("\nHot-anomaly summary (top 5% of WBGT anomalies vs facility-month climatology):")
    print(summary_df[[
        "indicator", "hot_anom_deficit_pct", "hot_anom_ci_lo",
        "hot_anom_ci_hi", "n_hot_anom_obs",
    ]].to_string())
