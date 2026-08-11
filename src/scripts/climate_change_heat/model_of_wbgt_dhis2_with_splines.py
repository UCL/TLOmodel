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
    "fp_total_clients", "opd_attendance", "anc4_coverage", "bcg_under1",
    "fully_immunised_under1", "ipd_total_admissions", "measles_under1",
    "opd_attendance", "penta3_under1", "pnc_mother_checked_48h",
]

WBGT_VAR   = "wbgt_day"
SPLINE_DF  = 3                # fixed a priori
LAG_MONTHS = []#[1, 2, 3]
CENTER     = True
MIN_OBS    = 72 #int(0.8 * 12 * 10)
WINSOR_K   = 5.0
apply_cap  = True

REFERENCE_WBGT_PERCENTILE = 95     # hot-month threshold (for the hot deficit)

# --- counterfactual reference ---
REF_MODE       = "fixed"   # "facility_mean" | "fixed"
REF_WBGT_FIXED = 23.0

min_year_historical = 2015
max_year_historical = 2025
COVID_START = "2020-04-01"
COVID_END   = "2021-06-01"

CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre",  "2023-03-01", "2024-03-01"),
]

CLUSTER_COL = "Dist"
FE_SPEC     = ["facility", "month"]     # real FE terms in the formula
FE_COLS     = ["facility", "month", "Dist"]   # columns to factor-convert

N_BOOTSTRAP      = 1000
BOOT_SEED        = 42
BOOT_CI_LEVEL    = 0.95
BOOT_MIN_SUCCESS = 0.80
FDR_ALPHA        = 0.05

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs_poisson/"
PANEL_TMPL = (f"{DATA_DIR}/All_predictors_processed/"
              "regression_panel_{indicator}.csv")
PANEL_DIST_COL = "Dist"
os.makedirs(OUT_DIR, exist_ok=True)

PRECIP_LONG_PATH = ("/Users/rachelmurray-watson/Documents/Heat_data/"
                    "Thermofeel_WBGT/Indices/precip_long.csv")
PRECIP_TERMS = ["precip_month"]

# ===========================================================================
# Fit + predict-twice core
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


def spline_basis_at(wbgt_values, wbgt_shift, design_info):
    """Build the SAME spline basis (from the fitted design_info) at arbitrary
    WBGT values, centred with the training shift. Returns an (n, df) array."""
    xc = np.asarray(wbgt_values, dtype=float) - wbgt_shift
    return np.asarray(
        patsy.build_design_matrices([design_info], {"x": xc})[0], dtype=float)


def predict_twice_deficit(df, coef, spline_cols, design_info, wbgt_shift,
                          lag_terms, ref_wbgt):
    """Heat-attributable deficit via one-model predict-twice.

    The linear predictor is eta = FE + delta*year + covid + f(WBGT) + Σ lags.
    Everything except the WBGT spline + lag terms is IDENTICAL between the
    observed and reference predictions, so it cancels in the ratio. We therefore
    only need the spline/lag part to form the deficit RATIO per row:

        mu_obs / mu_ref = exp( [basis(obs) - basis(ref)] · beta_spline
                               + Σ_l [WBGTlag_obs - WBGTlag_ref] · beta_lag )

    and deficit_row = mu_obs * (mu_ref/mu_obs - 1) applied to the fitted mu.
    We use the model's fitted mu as mu_obs (exact), then scale to mu_ref.
    """
    mu_obs = np.asarray(stats_r.fitted(_CURRENT_MODEL[0]), dtype=float)

    # spline contrast
    beta_s = np.array([coef.get(c, 0.0) for c in spline_cols], dtype=float)
    B_obs  = spline_basis_at(df[WBGT_VAR].values, wbgt_shift, design_info)
    B_ref  = spline_basis_at(np.full(len(df), ref_wbgt), wbgt_shift, design_info)
    d_eta  = (B_ref - B_obs) @ beta_s

    # distributed-lag contrast: at reference, the lagged exposures are also ref
    for lag in lag_terms:
        # lag columns are named wbgt_c_lag{lag}; coef keyed the same
        lname = f"wbgt_c_lag{lag}"
        if lname in coef:
            obs_lag = df[lname].values                    # already centred
            ref_lag = (ref_wbgt - wbgt_shift)             # ref, centred
            d_eta  += coef[lname] * (ref_lag - obs_lag)

    mu_ref = mu_obs * np.exp(d_eta)
    return mu_obs, mu_ref


_CURRENT_MODEL = [None]   # tiny holder so predict_twice can see the fitted model


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
    precip = pd.read_csv(PRECIP_LONG_PATH, parse_dates=["date"])
    precip["facility"] = precip["facility"].astype(str).str.strip()
    precip["date"] = precip["date"].dt.to_period("M").dt.to_timestamp()
    precip = precip[(precip["date"] >= "2015-02-01") & (precip["date"] <= "2024-12-01")]
    n_before = len(long)
    long = long.merge(precip, on=["facility", "date"], how="left")
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
    long["wbgt_c"] = long[WBGT_VAR] - wbgt_shift
    long["year_c"] = long["year"] - (long["year"].mean() if CENTER else 0.0)
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

    return nb.reset_index(drop=True), wbgt_shift, lag_terms


# ===========================================================================
# Per-indicator run
# ===========================================================================
def deficit_from_fit(nb, wbgt_shift, lag_terms):
    """Fit one fepois model and return (mu_obs, mu_ref, coef, spline_cols,
    design_info)."""
    basis = patsy.dmatrix(f"cr(x, df={SPLINE_DF}) - 1",
                          {"x": nb["wbgt_c"].values}, return_type="dataframe")
    design_info = basis.design_info
    spline_cols = [f"wbgt_s{i+1}" for i in range(basis.shape[1])]
    for c, b in zip(spline_cols, basis.columns):
        nb[c] = basis[b].values

    rhs = spline_cols + lag_terms + PRECIP_TERMS + ["covid", "year_c"]
    r_model, coef = fit_fepois(nb, rhs, FE_SPEC, FE_COLS, CLUSTER_COL)
    _CURRENT_MODEL[0] = r_model

    ref = _resolve_ref(nb).values
    # predict-twice using the row-wise reference
    mu_obs = np.asarray(stats_r.fitted(r_model), dtype=float)
    beta_s = np.array([coef.get(c, 0.0) for c in spline_cols], dtype=float)
    B_obs  = spline_basis_at(nb[WBGT_VAR].values, wbgt_shift, design_info)
    B_ref  = spline_basis_at(ref, wbgt_shift, design_info)
    d_eta  = (B_ref - B_obs) @ beta_s
    for lag_col in lag_terms:  # was: `for lag in lag_terms`
        if lag_col in coef:
            d_eta += coef[lag_col] * ((ref - wbgt_shift) - nb[lag_col].values)
    mu_ref = mu_obs * np.exp(d_eta)
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
    nb, wbgt_shift, lag_terms = prep
    print(f"  [{indicator}] n={len(nb)}, fac={nb['facility'].nunique()}, "
          f"clust={nb[CLUSTER_COL].nunique()}")

    try:
        mu_obs, mu_ref, coef, spline_cols, design_info = deficit_from_fit(
            nb, wbgt_shift, lag_terms)
        print(f"  [{indicator}] POINT FIT OK")
    except Exception as e:
        print(f"  [{indicator}] POINT FIT FAILED: {type(e).__name__}: {e}")
        return None

    hot_thr = np.percentile(nb[WBGT_VAR].values, REFERENCE_WBGT_PERCENTILE)
    hot = nb[WBGT_VAR].values > hot_thr

    res = {
        "indicator": indicator,
        "n_obs": len(nb), "n_fac": nb["facility"].nunique(),
        "deficit_pct":     _pct(mu_obs, mu_ref),
        "hot_deficit_pct": _pct(mu_obs, mu_ref, hot),
        "ref_mode": REF_MODE,
    }

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
    return res


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
    rows = []
    for ind in COUNT_INDICATORS:
        print(f"-> {ind}")
        r = run_indicator(ind)
        if r:
            rows.append(r)
    if not rows:
        raise SystemExit("No indicators fitted.")
    df = pd.DataFrame(rows)
    df["q_boot"] = bh_fdr(df["p_boot"].values, FDR_ALPHA)
    df["sig_bh"] = df["q_boot"] < FDR_ALPHA
    df = df.sort_values("deficit_pct", ascending=False)
    df.to_csv(f"{OUT_DIR}deficit_summary_predicttwice_{WBGT_VAR}.csv", index=False)
    print("\n" + "=" * 60)
    print(df[["indicator", "n_obs", "n_fac", "deficit_pct", "hot_deficit_pct",
              "ci_lo", "ci_hi", "p_boot", "q_boot", "sig_bh"]].to_string(index=False))
    print(f"\nSummary -> {OUT_DIR}deficit_summary_predicttwice_{WBGT_VAR}.csv")
