"""
model_of_wbgt_dhis2_poisson_predicttwice.py

SINGLE-MODEL Poisson FE with a PREDICT-TWICE counterfactual — the clean
alternative to the two-model NB differencing when fenegbin won't converge.

Why this design
---------------
  * ONE Poisson FE model per indicator (fixest::fepois via rpy2 — no new
    dependency). Poisson QMLE + cluster-robust SEs is consistent under any
    conditional variance, has NO dispersion (theta) parameter to destabilise,
    and converges where fenegbin fails.
  * The "deficit" is a WITHIN-MODEL counterfactual, not a two-model subtraction:
    from the one fitted model we predict each facility-month twice —
        mu_obs = exp(alpha_f + gamma_m + delta*year + f(WBGT_observed) + lags)
        mu_ref = exp(alpha_f + gamma_m + delta*year + f(WBGT_reference) + lags)
    and take (mu_ref - mu_obs). Because BOTH come from the same model, the
    facility + month + year effects CANCEL exactly — the deficit depends only on
    the spline (and lag) coefficients and the WBGT contrast. This is why it works
    under Poisson (the two-model subtraction does NOT: Poisson FE forces fitted
    totals to match observed totals within absorbed groups, so M0 and M1 predict
    near-identical totals and the difference collapses).

Deficit sign convention (matches the NB script after the sign fix):
    deficit_pct = 100 * (sum(mu_ref) - sum(mu_obs)) / sum(mu_ref)
    POSITIVE = services LOST to heat (observed WBGT predicts fewer than reference).

Reference WBGT (config REF_MODE):
    "facility_mean" : each facility vs its own typical WBGT — within-facility,
                      matches the anomaly identification of the rest of the model.
    "fixed"         : all facilities vs REF_WBGT_FIXED (e.g. 25 C) — comparable
                      across facilities, matches the IRR-contrast framing.

"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
from collections import Counter
from contextvars import ContextVar

import numpy as np
import pandas as pd
import patsy
from pathlib import Path

os.environ.setdefault("R_HOME", "/Library/Frameworks/R.framework/Resources")
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

base    = importr("base")
stats_r = importr("stats")
try:
    fixest = importr("fixest")
except Exception as e:
    raise RuntimeError("R package 'fixest' not installed.") from e

ro.conversion.converter_ctx = ContextVar("converter", default=ro.default_converter)

# ===========================================================================
# CONFIG  (mirror your NB script; only the estimator + counterfactual differ)
# ===========================================================================
COUNT_INDICATORS = [
    "fp_total_clients", "opd_attendance", "bcg_under1",
    "fully_immunised_under1", "ipd_total_admissions", "measles_under1",
    "penta3_under1", "pnc_mother_checked_48h",
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
    "measles_under1":             "Measles Under-1",
    "fully_immunised_under1":     "Fully Immunised Under-1",
    "pnc_within_2wks":            "PNC Within 2 Weeks",
    "pnc_first_visit_2wks":       "PNC First Visit <2 Weeks",
    "live_births_total":          "Live Births Total",
    "skilled_deliveries":         "Skilled Deliveries",
}

WBGT_VAR   = "wbgt_day"
SPLINE_DF  = 3                # fixed a priori
LAG_MONTHS = [1, 2, 3]
CENTER     = True
WINSOR_K   = 5.0
apply_cap  = True

REFERENCE_WBGT_PERCENTILE = 95     # hot-month threshold (for the hot deficit)

# --- counterfactual reference ---
REF_MODE       = "fixed"   # "facility_mean" | "fixed"
REF_WBGT_FIXED = 23.0

# --- exposure-response curve ---
N_CURVE_POINTS = 200
CURVE_REF      = "mean"    # "mean" | "median" | "min" — reference point the curve is drawn against

min_year_historical = 2019
max_year_historical = 2025
MIN_OBS    =  (max_year_historical - min_year_historical)*12 * 0.5

COVID_START = "2020-04-01"
COVID_END   = "2021-06-01"

CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre",  "2023-03-01", "2024-03-01"),
]

CLUSTER_COL = "Dist"
FE_SPEC     = ["facility", "month"]     # real FE terms in the formula
FE_COLS     = ["facility", "month", "Dist"]   # columns to factor-convert

N_BOOTSTRAP      = 500
BOOT_SEED        = 42
BOOT_CI_LEVEL    = 0.95
BOOT_MIN_SUCCESS = 0.80
FDR_ALPHA        = 0.05

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs_poisson/2019/"
PANEL_TMPL = (f"{DATA_DIR}/All_predictors_processed/"
              "regression_panel_{indicator}.csv")
PANEL_DIST_COL = "Dist"
os.makedirs(OUT_DIR, exist_ok=True)

PRECIP_TERMS = ["precip_month"]

# ---- Projection config (ported from the NB script) ------------------------
# Because this is a ONE-model predict-twice design, projection also uses
# predict-twice: future WBGT vs the SAME reference WBGT. There is no Model B.
PROJECT = True
SSP_SCENARIOS = ["ssp245"]                 # ["ssp126", "ssp245", "ssp585"]
MODEL_TIERS   = ["lowest", "median", "highest"]
PROJ_PERIOD_START = 2025
PROJ_PERIOD_END   = 2040

THERMOFEEL_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices")
PROJECTION_DIR = str(THERMOFEEL_DIR)

# Long: rows are (facility, date), cols include wbgt_day, wbgt_night
WBGT_PROJ_FILE_TPL         = "wbgt_monthly_mean_facility_{tier}_{ssp}.csv"
# Wide: rows are 'YYYY-M' strings, columns are facility names, values = precip
PRECIP_5DAY_PROJ_FILE_TPL  = ("ResourceFile_Precipitation_Disruptions_{ssp}_{tier}_"
                              "window_prediction_weather_by_facility.csv")
PRECIP_MONTH_PROJ_FILE_TPL = ("ResourceFile_Precipitation_Disruptions_{ssp}_{tier}_"
                              "monthly_prediction_weather_by_facility.csv")


# ===========================================================================
# Fit + vcov
# ===========================================================================
def fit_fepois(df, rhs_terms, fe_spec, fe_cols, cluster_col, y_col="y_int"):
    """Fit one Poisson FE model; return (r_model, coef_dict)."""
    rhs = " + ".join(rhs_terms) if rhs_terms else "1"
    fe  = " + ".join(fe_spec)
    fml = ro.Formula(f"{y_col} ~ {rhs} | {fe}")
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    r_df = base.as_data_frame(r_df)
    for col in set(fe_cols + [cluster_col]):
        r_df.rx2[col] = base.as_factor(r_df.rx2(col))
    r_model = ro.r("suppressWarnings")(
        fixest.fepois(fml=fml, data=r_df, cluster=ro.StrVector([cluster_col])))
    # coefficient vector (slopes only; FE are absorbed and not needed here)
    co = stats_r.coef(r_model)
    names = list(ro.r("names")(co))
    vals  = np.asarray(co, dtype=float)
    return r_model, dict(zip(names, vals))


def get_beta_vcov(r_model):
    """Name-aligned (beta, vcov) from the fepois clustered covariance.
    Same contract as the NB script's helper. Names come from coef(); the
    vcov is matched by name so a dropped-collinear term can't misalign."""
    coef_r = ro.r("coef")(r_model)
    with localconverter(ro.default_converter + pandas2ri.converter):
        beta_full = np.asarray(ro.conversion.rpy2py(coef_r), dtype=float)
    coef_names = list(coef_r.names)

    vcov_r = ro.r("vcov")(r_model)
    with localconverter(ro.default_converter + pandas2ri.converter):
        vcov_full = np.asarray(ro.conversion.rpy2py(vcov_r), dtype=float)

    rn = ro.r("rownames")(vcov_r)
    if rn == ro.NULL or vcov_full.shape[0] == len(coef_names):
        return coef_names, beta_full, vcov_full
    vcov_names = list(rn)
    common = [n for n in coef_names if n in vcov_names]
    ci = [coef_names.index(n) for n in common]
    vi = [vcov_names.index(n) for n in common]
    return common, beta_full[ci], vcov_full[np.ix_(vi, vi)]


def spline_basis_at(wbgt_values, wbgt_shift, design_info):
    """Build the SAME spline basis (from the fitted design_info) at arbitrary
    WBGT values, centred with the training shift. Returns an (n, df) array."""
    xc = np.asarray(wbgt_values, dtype=float) - wbgt_shift
    return np.asarray(
        patsy.build_design_matrices([design_info], {"x": xc})[0], dtype=float)


# ===========================================================================
# Exposure-response curve (ported from NB script; delta-method band, fail-loud)
# ===========================================================================
def make_exposure_response_curve(
    coef, spline_cols, wbgt_shift, observed_wbgt, indicator,
    design_info, names_a=None, vcov_a=None,
) -> pd.DataFrame:
    """Curve of exp((basis(w) - basis(ref)) @ beta_spline) over the observed
    WBGT range. The LINE is rr_vs_ref (NOT its inverse). The band is the
    delta method on the spline block of the clustered vcov, indexed by name;
    if that block is non-finite (e.g. collinearity made it singular) this
    RAISES rather than silently emitting a zero-width band."""
    x_min  = float(observed_wbgt.min())
    x_max  = float(observed_wbgt.max())
    x_grid   = np.linspace(x_min, x_max, N_CURVE_POINTS)
    x_grid_c = x_grid - wbgt_shift

    basis_grid = np.asarray(
        patsy.build_design_matrices([design_info], {"x": x_grid_c})[0],
        dtype=float)

    if CURVE_REF == "mean":
        x_ref = float(observed_wbgt.mean())
    elif CURVE_REF == "median":
        x_ref = float(observed_wbgt.median())
    elif CURVE_REF == "min":
        x_ref = float(observed_wbgt.min())
    else:
        raise ValueError(f"Unknown CURVE_REF='{CURVE_REF}'.")

    x_ref_c   = x_ref - wbgt_shift
    ref_row = np.asarray(
        patsy.build_design_matrices([design_info], {"x": np.array([x_ref_c])})[0],
        dtype=float)[0]

    beta     = np.array([coef.get(c, 0.0) for c in spline_cols], dtype=float)
    contrast = basis_grid - ref_row[None, :]
    eta_grid = contrast @ beta
    rr       = np.exp(eta_grid)

    rr_lo = np.full_like(rr, np.nan)
    rr_hi = np.full_like(rr, np.nan)
    if vcov_a is not None and names_a is not None:
        idx = [names_a.index(c) for c in spline_cols if c in names_a]
        if len(idx) == len(spline_cols):
            V = vcov_a[np.ix_(idx, idx)]
            # Fail loud: a non-finite spline vcov means the clustered cov
            # couldn't be recovered (collinearity/singularity). Do NOT scrub
            # to zero — that produces a fake zero-width band.
            assert np.isfinite(V).all(), (
                f"[{indicator}] spline vcov non-finite — clustered cov "
                f"singular (check WBGT-spline/lag collinearity).")
            var_grid = np.einsum("ij,jk,ik->i", contrast, V, contrast)
            se_grid  = np.sqrt(np.maximum(var_grid, 0.0))
            rr_lo = np.exp(eta_grid - 1.96 * se_grid)
            rr_hi = np.exp(eta_grid + 1.96 * se_grid)
        else:
            print(f"  [{indicator}] curve: spline names not all found in vcov "
                  f"— band left as NaN")

    return pd.DataFrame({
        "indicator":         indicator,
        "label":             INDICATOR_LABELS.get(indicator, indicator),
        "wbgt":              x_grid,
        "wbgt_c":            x_grid_c,
        "rr_vs_ref":         rr,
        "rr_lo":             rr_lo,
        "rr_hi":             rr_hi,
        "pct_change_vs_ref": 100.0 * (rr - 1.0),
        "wbgt_ref":          x_ref,
        "curve_ref":         CURVE_REF,
    })


# ===========================================================================
# Data prep  (same logic as the NB script; condensed)
# ===========================================================================
def _resolve_ref(df):
    if REF_MODE == "fixed":
        return pd.Series(REF_WBGT_FIXED, index=df.index)
    elif REF_MODE == "facility_mean":
        return df.groupby("facility")[WBGT_VAR].transform("mean")
    raise ValueError(f"REF_MODE must be 'facility_mean' or 'fixed'.")


def winsorise_by_facility(df, col, k=WINSOR_K):
    def _w(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return s.clip(upper=q3 + k * iqr)
    df[col] = df.groupby("facility")[col].transform(_w)
    return df


def prepare_indicator(indicator):
    path = PANEL_TMPL.format(indicator=indicator)
    if not os.path.exists(path):
        print(f"  [{indicator}] no panel — skip"); return None
    long = pd.read_csv(path, parse_dates=["date"])
    if indicator not in long.columns or WBGT_VAR not in long.columns:
        print(f"  [{indicator}] missing cols — skip"); return None
    long = long.rename(columns={indicator: "y"})

    for fac, d0, d1 in CLOSURES:
        m = (long["date"].between(d0, d1)) & (long["facility"] == fac)
        if m.any():
            print(f"  [{indicator}] Masked {int(m.sum())} closure months for {fac}.")
            long.loc[m, "y"] = np.nan

    long["year"]  = long["date"].dt.year
    long["month"] = long["date"].dt.month
    long = long[long["year"].between(min_year_historical, max_year_historical - 1)]

    long["date"] = long["date"].dt.to_period("M").dt.to_timestamp()

    # Precip is a column in the panel now
    missing_precip = [c for c in PRECIP_TERMS if c not in long.columns]
    if missing_precip:
        print(f"  [{indicator}] panel missing precip columns {missing_precip} — "
              f"regenerate regression_panel with the updated extraction script; "
              f"skip"); return None
    n_precip_missing = long[PRECIP_TERMS].isna().any(axis=1).sum()
    if n_precip_missing:
        print(
            f"  [{indicator}] {n_precip_missing:,} rows with missing precip "
            f"({100 * n_precip_missing / len(long):.1f}%) — will be dropped."
        )

    obs = long.dropna(subset=["y", WBGT_VAR]).groupby("facility").size()
    long = long[long["facility"].isin(obs[obs >= MIN_OBS].index)].copy()
    if long.empty:
        print(f"  [{indicator}] empty after MIN_OBS — skip"); return None

    long = long.sort_values(["facility", "date"]).reset_index(drop=True)
    long = winsorise_by_facility(long, "y")

    wbgt_shift = long[WBGT_VAR].mean() if CENTER else 0.0
    year_shift = long["year"].mean() if CENTER else 0.0
    long["wbgt_c"] = long[WBGT_VAR] - wbgt_shift
    long["year_c"] = long["year"] - year_shift
    long["covid"]  = long["date"].between(COVID_START, COVID_END).astype(int)
    for lag in LAG_MONTHS:
        long[f"wbgt_c_lag{lag}"] = (
            long.groupby("facility")[WBGT_VAR].shift(lag) - wbgt_shift)

    lag_terms = [f"wbgt_c_lag{l}" for l in LAG_MONTHS]
    keep = ["y", "facility", "month", "year_c", "wbgt_c", "covid", CLUSTER_COL, WBGT_VAR] + lag_terms + PRECIP_TERMS
    nb = long.dropna(subset=keep).copy()
    nb["y_int"] = nb["y"].round().clip(lower=0).astype(int)

    # drop all-zero facilities (separation)
    allz = nb.groupby("facility")["y_int"].max() == 0
    nb = nb[~nb["facility"].isin(allz[allz].index)].copy()
    if nb["facility"].nunique() < 2 or nb[CLUSTER_COL].nunique() < 2:
        print(f"  [{indicator}] too few groups — skip"); return None

    return nb.reset_index(drop=True), wbgt_shift, year_shift, lag_terms


# ===========================================================================
# Per-indicator run
# ===========================================================================
def deficit_from_fit(nb, wbgt_shift, lag_terms, want_model=False):
    """Fit one fepois model and return the predict-twice deficit pieces.

    Returns (mu_obs, mu_ref, coef, spline_cols, design_info) or, if
    want_model=True, additionally (r_model). want_model=False in the bootstrap
    (no vcov/curve needed) keeps replicates lean."""
    basis = patsy.dmatrix(f"cr(x, df={SPLINE_DF}) - 1",
                          {"x": nb["wbgt_c"].values}, return_type="dataframe")
    design_info = basis.design_info
    spline_cols = [f"wbgt_s{i+1}" for i in range(basis.shape[1])]

    for c, b in zip(spline_cols, basis.columns):
        nb[c] = basis[b].values

    rhs = spline_cols + lag_terms + PRECIP_TERMS + ["covid", "year_c"]
    r_model, coef = fit_fepois(nb, rhs, FE_SPEC, FE_COLS, CLUSTER_COL)

    ref = _resolve_ref(nb).values
    # predict-twice using the row-wise reference
    mu_obs = np.asarray(stats_r.fitted(r_model), dtype=float)
    beta_s = np.array([coef.get(c, 0.0) for c in spline_cols], dtype=float)
    B_obs  = spline_basis_at(nb[WBGT_VAR].values, wbgt_shift, design_info)
    B_ref  = spline_basis_at(ref, wbgt_shift, design_info)
    d_eta  = (B_ref - B_obs) @ beta_s
    for lag_col in lag_terms:
        if lag_col in coef:
            d_eta += coef[lag_col] * ((ref - wbgt_shift) - nb[lag_col].values)
    mu_ref = mu_obs * np.exp(d_eta)
    if want_model:
        return mu_obs, mu_ref, coef, spline_cols, design_info, r_model
    return mu_obs, mu_ref, coef, spline_cols, design_info


def _pct(mu_obs, mu_ref, mask=None):
    if mask is not None:
        mu_obs, mu_ref = mu_obs[mask], mu_ref[mask]
    so, sr = float(mu_obs.sum()), float(mu_ref.sum())
    return 100.0 * (sr - so) / sr if sr > 0 else np.nan   # +ve = loss


def _bootstrap(nb, wbgt_shift, lag_terms, hot_thr):
    dist_ids = nb[CLUSTER_COL].unique()
    idx_by_d = {d: np.asarray(g) for d, g in
                nb.groupby(CLUSTER_COL, sort=False).indices.items()}
    seeds = np.random.SeedSequence(BOOT_SEED).spawn(N_BOOTSTRAP)
    agg, hot, fails = [], [], Counter()
    for s in seeds:
        rng = np.random.default_rng(s)
        picks = rng.choice(len(dist_ids), size=len(dist_ids), replace=True)
        parts = [idx_by_d[dist_ids[p]] for p in picks]
        take  = np.concatenate(parts)
        tags  = np.repeat(np.arange(len(picks)), [len(p) for p in parts]).astype(str)
        bdf = nb.take(take).reset_index(drop=True)
        bdf["facility"] = bdf["facility"].astype(str) + "__b" + tags
        bdf[CLUSTER_COL] = bdf[CLUSTER_COL].astype(str) + "__b" + tags
        allz = bdf.groupby("facility")["y_int"].max() == 0
        bdf = bdf[~bdf["facility"].isin(allz[allz].index)].copy()
        if bdf["facility"].nunique() < 2 or bdf[CLUSTER_COL].nunique() < 2:
            fails["too_few_groups"] += 1; continue
        try:
            mo, mr, *_ = deficit_from_fit(bdf, wbgt_shift, lag_terms)
            a = _pct(mo, mr)
            h = _pct(mo, mr, bdf[WBGT_VAR].values > hot_thr)
            if np.isfinite(a): agg.append(a)
            if np.isfinite(h): hot.append(h)
        except Exception as e:
            fails[f"{type(e).__name__}"] += 1
    return agg, hot, fails


def run_indicator(indicator):
    prep = prepare_indicator(indicator)
    if prep is None:
        return None
    nb, wbgt_shift, year_shift, lag_terms = prep
    print(f"  [{indicator}] n={len(nb)}, fac={nb['facility'].nunique()}, "
          f"clust={nb[CLUSTER_COL].nunique()}")

    try:
        mu_obs, mu_ref, coef, spline_cols, design_info, r_model = deficit_from_fit(
            nb, wbgt_shift, lag_terms, want_model=True)
        print(f"  [{indicator}] POINT FIT OK")
    except Exception as e:
        print(f"  [{indicator}] POINT FIT FAILED: {type(e).__name__}: {e}")
        return None

    hot_thr = np.percentile(nb[WBGT_VAR].values, REFERENCE_WBGT_PERCENTILE)
    hot = nb[WBGT_VAR].values > hot_thr

    res = {
        "indicator": indicator,
        "label": INDICATOR_LABELS.get(indicator, indicator),
        "n_obs": len(nb), "n_fac": nb["facility"].nunique(),
        "deficit_pct":     _pct(mu_obs, mu_ref),
        "hot_deficit_pct": _pct(mu_obs, mu_ref, hot),
        "ref_mode": REF_MODE,
        "reference_wbgt": hot_thr,
    }

    # ---- (a) Exposure-response curve + delta-method band -----------------
    names_a = vcov_a = None
    try:
        names_a, _beta_a, vcov_a = get_beta_vcov(r_model)
    except Exception as e:
        print(f"  [{indicator}] vcov extraction failed: {type(e).__name__}: {e}")
    try:
        curve_df = make_exposure_response_curve(
            coef, spline_cols, wbgt_shift, nb[WBGT_VAR], indicator,
            design_info, names_a=names_a, vcov_a=vcov_a)
        curve_df.to_csv(
            f"{OUT_DIR}exposure_response_curve_{indicator}_{WBGT_VAR}.csv",
            index=False)
    except AssertionError as e:
        # Non-finite spline vcov: emit the line with a NaN band rather than
        # dropping the curve entirely, but SAY SO loudly.
        print(f"  [{indicator}] curve band unavailable: {e}")
        curve_df = make_exposure_response_curve(
            coef, spline_cols, wbgt_shift, nb[WBGT_VAR], indicator,
            design_info, names_a=None, vcov_a=None)
        curve_df.to_csv(
            f"{OUT_DIR}exposure_response_curve_{indicator}_{WBGT_VAR}.csv",
            index=False)
    except Exception as e:
        print(f"  [{indicator}] curve export failed: {type(e).__name__}: {e}")

    # bootstrap CIs (deficit only)
    ci_lo = ci_hi = hot_lo = hot_hi = p_boot = np.nan
    if N_BOOTSTRAP > 0:
        agg, hotb, fails = _bootstrap(nb, wbgt_shift, lag_terms, hot_thr)
        if fails:
            print(f"  [{indicator}] bootstrap failures: {dict(fails)}")
        if len(agg) >= BOOT_MIN_SUCCESS * N_BOOTSTRAP:
            a = 1 - BOOT_CI_LEVEL
            arr = np.asarray(agg)
            ci_lo = float(np.percentile(arr, 100 * a / 2))
            ci_hi = float(np.percentile(arr, 100 * (1 - a / 2)))
            le, ge = np.mean(arr <= 0), np.mean(arr >= 0)
            p_boot = float(min(1.0, max(2 * min(le, ge), 1.0 / (len(arr) + 1))))
            pd.DataFrame({"deficit_pct": agg}).to_csv(
                f"{OUT_DIR}bootstrap_{indicator}_{WBGT_VAR}.csv", index=False)
        else:
            print(f"  [{indicator}] bootstrap UNRELIABLE "
                  f"({len(agg)}/{N_BOOTSTRAP}) — CI = NaN")
        if len(hotb) >= BOOT_MIN_SUCCESS * N_BOOTSTRAP:
            a = 1 - BOOT_CI_LEVEL
            harr = np.asarray(hotb)
            hot_lo = float(np.percentile(harr, 100 * a / 2))
            hot_hi = float(np.percentile(harr, 100 * (1 - a / 2)))

    res.update(ci_lo=ci_lo, ci_hi=ci_hi,
               hot_ci_lo=hot_lo, hot_ci_hi=hot_hi, p_boot=p_boot)

    # per-facility predictions out
    preds = nb[["facility", "date", WBGT_VAR]].copy()
    preds["y_obs"]  = nb["y_int"].values
    preds["mu_obs"] = mu_obs
    preds["mu_ref"] = mu_ref
    preds["deficit"] = mu_ref - mu_obs           # +ve = loss
    if PANEL_DIST_COL in nb.columns:
        preds["Dist"] = nb[PANEL_DIST_COL].values
    preds.to_csv(f"{OUT_DIR}predicttwice_predictions_{indicator}_{WBGT_VAR}.csv",
                 index=False)
    print(f"  [{indicator}] deficit={res['deficit_pct']:+.2f}%  "
          f"hot={res['hot_deficit_pct']:+.2f}%")

    # stash the pieces the projection block needs (private keys, not saved)
    res["_nb"]          = nb
    res["_coef"]        = coef
    res["_spline_cols"] = spline_cols
    res["_design_info"] = design_info
    res["_wbgt_shift"]  = wbgt_shift
    res["_year_shift"]  = year_shift
    res["_lag_terms"]   = lag_terms
    return res


# ===========================================================================
# (b) Forward projections — PREDICT-TWICE on future CMIP6 WBGT
# ===========================================================================
def _load_precip_wide(path, value_name):
    """Wide file (index='YYYY-M', cols=facility names) -> long
    [facility, date, value_name]. None if the file is missing."""
    if not os.path.exists(path):
        return None
    wide = pd.read_csv(path, index_col=0)
    wide.index = pd.to_datetime(
        wide.index.astype(str).str.strip(),
        format="%Y-%m", errors="coerce").to_period("M").to_timestamp()
    n_bad = wide.index.isna().sum()
    if n_bad:
        raise ValueError(f"{path}: {n_bad} unparseable date rows in index")
    wide.index.name = "date"
    wide.columns = wide.columns.astype(str).str.strip()
    return (wide.stack(future_stack=True)
                .rename(value_name)
                .rename_axis(index=["date", "facility"])
                .reset_index())


def _load_wbgt_proj(path):
    """Long file: facility, date, wbgt_day, wbgt_night. None if missing."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["facility"] = df["facility"].astype(str).str.strip()
    df["date"]     = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def _load_future_climate(ssp, tier):
    """Load future WBGT (+ precip if present) for one SSP/tier, restrict to the
    projection window, add WBGT lag columns per facility. Returns (df, None) or
    (None, missing_paths). Precip is loaded when available but NOT required —
    the predict-twice deficit only needs WBGT + its lags (precip cancels in the
    obs-vs-ref contrast since it's held fixed)."""
    wbgt_path = os.path.join(PROJECTION_DIR,
                             WBGT_PROJ_FILE_TPL.format(tier=tier, ssp=ssp))
    pm_path = os.path.join(PROJECTION_DIR,
                           PRECIP_MONTH_PROJ_FILE_TPL.format(ssp=ssp, tier=tier))
    p5_path = os.path.join(PROJECTION_DIR,
                           PRECIP_5DAY_PROJ_FILE_TPL.format(ssp=ssp, tier=tier))

    wbgt_df = _load_wbgt_proj(wbgt_path)
    if wbgt_df is None:
        return None, [wbgt_path]

    pm_df = _load_precip_wide(pm_path, "precip_month")
    p5_df = _load_precip_wide(p5_path, "precip_5day")

    clim = wbgt_df
    if p5_df is not None:
        clim = clim.merge(p5_df, on=["facility", "date"], how="left")
    if pm_df is not None:
        clim = clim.merge(pm_df, on=["facility", "date"], how="left")
    clim = clim.sort_values(["facility", "date"]).reset_index(drop=True)

    for k in LAG_MONTHS:
        clim[f"wbgt_c_lag{k}_raw"] = clim.groupby("facility")[WBGT_VAR].shift(k)
    clim = clim[(clim["date"].dt.year >= PROJ_PERIOD_START) &
                (clim["date"].dt.year <= PROJ_PERIOD_END)].copy()
    clim["year"]  = clim["date"].dt.year
    clim["month"] = clim["date"].dt.month
    return clim, None


def project_indicator(res, clim, ssp, tier):
    """Predict-twice deficit on future WBGT for one indicator/SSP/tier.

    mu_obs_future is NOT observed (no future counts), so we anchor on a
    facility+month+year BASELINE intensity implied by the fitted model at the
    reference WBGT, then scale by the spline/lag contrast. Concretely:
        mu_ref_future = baseline(facility, month, year_anchor)          [ref WBGT]
        mu_obs_future = mu_ref_future * exp( (B(obs) - B(ref)) @ beta
                                             + Σ_lag beta_lag (obs_lag - ref) )
    Only the CONTRAST is used downstream (deficit = mu_ref - mu_obs), so the
    facility/month/year baseline cancels exactly and the projected deficit
    depends only on the future WBGT distribution vs the reference — the same
    invariance that makes the historical predict-twice valid. We therefore set
    the baseline to 1.0 per row and report the deficit as a PERCENTAGE, which is
    baseline-free."""
    ind         = res["indicator"]
    wbgt_shift  = res["_wbgt_shift"]
    design_info = res["_design_info"]
    spline_cols = res["_spline_cols"]
    coef        = res["_coef"]
    lag_terms   = res["_lag_terms"]
    nb          = res["_nb"]

    train_facs = set(nb["facility"].unique())
    df = clim[clim["facility"].isin(train_facs)].copy()

    # centred future WBGT + lags
    df["wbgt_c"] = df[WBGT_VAR] - wbgt_shift
    for k in LAG_MONTHS:
        df[f"wbgt_c_lag{k}"] = df[f"wbgt_c_lag{k}_raw"] - wbgt_shift

    need = [WBGT_VAR] + [f"wbgt_c_lag{k}" for k in LAG_MONTHS]
    n_before = len(df)
    df = df.dropna(subset=need).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if df.empty:
        print(f"    {ind}/{ssp}/{tier}: no rows after covariate build — skip")
        return None

    ref = _resolve_ref(df).values          # same REF_MODE as the historical fit
    beta_s = np.array([coef.get(c, 0.0) for c in spline_cols], dtype=float)
    B_obs  = spline_basis_at(df[WBGT_VAR].values, wbgt_shift, design_info)
    B_ref  = spline_basis_at(ref, wbgt_shift, design_info)
    d_eta  = (B_ref - B_obs) @ beta_s
    for lag_col in lag_terms:
        if lag_col in coef:
            d_eta += coef[lag_col] * ((ref - wbgt_shift) - df[lag_col].values)

    # baseline-free: set mu_ref = 1 per row; deficit % is invariant to it.
    mu_ref = np.ones(len(df), dtype=float)
    mu_obs = mu_ref * np.exp(d_eta)

    df["mu_obs"]      = mu_obs
    df["mu_ref"]      = mu_ref
    df["Disruption"]  = mu_ref - mu_obs                       # +ve = loss
    df["Deficit_Pct"] = np.where(mu_ref > 0,
                                 100.0 * (mu_ref - mu_obs) / mu_ref, np.nan)

    fac_dist = nb[["facility", CLUSTER_COL]].drop_duplicates("facility")
    df = df.merge(fac_dist, on="facility", how="left")
    df["indicator"], df["ssp"], df["tier"] = ind, ssp, tier

    # ---- Per-facility per-month CSV ----
    keep_cols = ["indicator", "ssp", "tier", "facility", CLUSTER_COL,
                 "year", "month", "date", WBGT_VAR,
                 "mu_obs", "mu_ref", "Disruption", "Deficit_Pct"]
    df[keep_cols].to_csv(
        f"{OUT_DIR}projection_facility_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
        index=False)

    # ---- District × year × month roll-up ----
    dist_agg = df.groupby([CLUSTER_COL, "year", "month"]).agg(
        Total_mu_obs =("mu_obs", "sum"),
        Total_mu_ref =("mu_ref", "sum"),
        Mean_WBGT    =(WBGT_VAR, "mean"),
        N_Facilities =("facility", "nunique"),
    ).reset_index()
    dist_agg["Total_Disruption"] = dist_agg["Total_mu_ref"] - dist_agg["Total_mu_obs"]
    dist_agg["Deficit_Pct"] = np.where(
        dist_agg["Total_mu_ref"] > 0,
        100.0 * dist_agg["Total_Disruption"] / dist_agg["Total_mu_ref"], np.nan)
    dist_agg["indicator"], dist_agg["ssp"], dist_agg["tier"] = ind, ssp, tier
    dist_agg.to_csv(
        f"{OUT_DIR}projection_district_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
        index=False)

    # ---- Monthly time series pooled across facilities ----
    mon_agg = df.groupby(["year", "month"]).agg(
        Total_mu_obs =("mu_obs", "sum"),
        Total_mu_ref =("mu_ref", "sum"),
        Mean_WBGT    =(WBGT_VAR, "mean"),
        N_Facilities =("facility", "nunique"),
    ).reset_index()
    mon_agg["Total_Disruption"] = mon_agg["Total_mu_ref"] - mon_agg["Total_mu_obs"]
    mon_agg["Deficit_Pct"] = np.where(
        mon_agg["Total_mu_ref"] > 0,
        100.0 * mon_agg["Total_Disruption"] / mon_agg["Total_mu_ref"], np.nan)
    mon_agg["Year_Month"] = (mon_agg["year"].astype(str)
                             + "-" + mon_agg["month"].astype(str))
    mon_agg["indicator"], mon_agg["ssp"], mon_agg["tier"] = ind, ssp, tier
    mon_agg.to_csv(
        f"{OUT_DIR}projection_monthly_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
        index=False)

    deficit_proj = _pct(mu_obs, mu_ref)
    print(f"    {ind}/{ssp}/{tier}: deficit_proj={deficit_proj:+.2f}% "
          f"(n={len(df):,}, dropped {n_dropped:,})")
    return {
        "indicator": ind, "ssp": ssp, "tier": tier,
        "period_start": PROJ_PERIOD_START, "period_end": PROJ_PERIOD_END,
        "n_facility_months": len(df), "n_input_dropped": n_dropped,
        "mean_wbgt": float(df[WBGT_VAR].mean()),
        "deficit_pct": deficit_proj,
    }


def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p); q = np.full_like(p, np.nan)
    pv = p[ok]; n = len(pv)
    if n:
        order = pv.argsort(); ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
        qv = pv * n / ranks
        qs = np.minimum.accumulate((qv[order])[::-1])[::-1]
        out = np.empty(n); out[order] = np.clip(qs, 0, 1)
        q[ok] = out
    return q


if __name__ == "__main__":
    print(f"Poisson predict-twice | REF_MODE={REF_MODE} | "
          f"bootstrap={N_BOOTSTRAP}\n" + "=" * 60)
    all_results = []
    for ind in COUNT_INDICATORS:
        print(f"-> {ind}")
        r = run_indicator(ind)
        if r:
            all_results.append(r)
    if not all_results:
        raise SystemExit("No indicators fitted.")

    # -- summary table (public columns only) --
    save_cols = [c for c in all_results[0] if not c.startswith("_")]
    df = pd.DataFrame([{k: r.get(k) for k in save_cols} for r in all_results])
    df["q_boot"] = bh_fdr(df["p_boot"].values, FDR_ALPHA)
    df["sig_bh"] = df["q_boot"] <= FDR_ALPHA
    df = df.sort_values("deficit_pct", ascending=False)
    df.to_csv(f"{OUT_DIR}deficit_summary_predicttwice_{WBGT_VAR}.csv", index=False)
    print("\n" + "=" * 60)
    print(df[["indicator", "n_obs", "n_fac", "deficit_pct", "hot_deficit_pct",
              "ci_lo", "ci_hi", "p_boot", "q_boot", "sig_bh"]].to_string(index=False))
    print(f"\nSummary -> {OUT_DIR}deficit_summary_predicttwice_{WBGT_VAR}.csv")

    # -- (b) forward projections (predict-twice on future WBGT) --
    if PROJECT:
        print("\n" + "=" * 60)
        print(f"FORWARD PROJECTIONS ({PROJ_PERIOD_START}-{PROJ_PERIOD_END}, "
              "predict-twice)")
        print("=" * 60)
        proj_summary = []
        for ssp in SSP_SCENARIOS:
            for tier in MODEL_TIERS:
                clim, missing = _load_future_climate(ssp, tier)
                if clim is None:
                    print(f"  SKIP {ssp}/{tier}: missing {missing}")
                    continue
                print(f"  {ssp}/{tier}: {len(clim):,} facility-months "
                      f"({clim['facility'].nunique()} facilities)")
                for res in all_results:
                    row = project_indicator(res, clim, ssp, tier)
                    if row:
                        proj_summary.append(row)
        if proj_summary:
            pd.DataFrame(proj_summary).to_csv(
                f"{OUT_DIR}projection_summary_{WBGT_VAR}.csv", index=False)
            print(f"\nProjection summary -> "
                  f"{OUT_DIR}projection_summary_{WBGT_VAR}.csv")
        else:
            print("\nNo projections produced — check WBGT_PROJ_FILE_TPL "
                  "against your actual filenames.")
