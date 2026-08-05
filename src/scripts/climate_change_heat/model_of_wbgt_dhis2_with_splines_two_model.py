"""
loop_all_indicators_two_model_NB.py

Two-model approach for WBGT–health-service disruption:
  Model A (exposure):       y ~ cr(WBGT, df) + WBGT_lags + precip + covid + year | facility + month
  Model B (counterfactual): y ~                             precip + covid + year | facility + month

Negative Binomial FE (fixest::fenegbin via rpy2) with district-clustered SEs.

Precip enters as a CONFOUNDER: same terms in both models, so the deficit is
the WBGT-attributable disruption holding precip fixed.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
from collections import Counter
from contextvars import ContextVar
import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
    raise RuntimeError(
        "R package 'fixest' not installed. In R run: install.packages('fixest')"
    ) from e

ro.conversion.converter_ctx = ContextVar("converter", default=ro.default_converter)

warnings.filterwarnings(
    "once",
    message=r".*variables dropped due to multicollinearity.*",
    category=UserWarning,
)

SHAPEFILE_PATH    = "/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
DISTRICT_NAME_COL = "ADM2_EN"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
COUNT_INDICATORS = [
    "fp_total_clients",
    "opd_attendance",
     "ipd_total_admissions",
    "vmmc_first_visits",
    "pnc_mother_checked_48h",
     "anc_new_attendees",
    "anc_first_trimester_starts",
    "bcg_under1",
    "penta3_under1",
    "measles1_under1",
    "fully_immunised_under1",
    "pnc_within_2wks",
    "pnc_first_visit_2wks",
    "live_births_total",
    "skilled_deliveries",
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

WBGT_VAR      = "wbgt_day"
SPLINE_DF     = 3
LAG_MONTHS    = [1, 2, 3, 4]
CENTER        = True
MIN_OBS       = int(0.7 * 12 * 12)
REFERENCE_WBGT_PERCENTILE = 95
min_year_historical = 2015
max_year_historical = 2025
apply_cap           = True
WINSOR_K            = 5.0

N_CURVE_POINTS = 200
CURVE_REF      = "mean"

COVID_START = "2020-04-01"
COVID_END   = "2021-04-01"

DF_MODE       = "per_indicator"   # "fixed" | "per_indicator"
DF_CANDIDATES = (3, 4, 5)

CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre",  "2023-03-01", "2024-03-01"),
]

SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
MODEL_TIERS   = ["lowest", "median", "highest"]
CLUSTER_COL = "Dist"

N_BOOTSTRAP     = 50      # >0 turns on the district block bootstrap for the
                           # DEFICIT CIs only (aggregate + hot-month). IRR and
                           # exposure-response curve stay on the delta method.
BOOT_SEED       = 42
BOOT_CI_LEVEL   = 0.95
BOOT_MIN_SUCCESS = 0.80
N_JOBS          = 1

FDR_ALPHA = 0.05

WBGT_GRID      = np.arange(20.0, 37.0, 0.5)

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

THERMOFEEL_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices")
PROJECTION_DIR = str(THERMOFEEL_DIR)

PANEL_DIST_COL_IN_PANEL = "Dist"

# ---- Precip confounder config ----
# precip_long.csv is produced by preprocess_precip.py — (facility, date,
# precip_month, precip_5day). Enters Model A AND Model B as a linear
# confounder so the WBGT deficit is estimated net of precip. Note that
# `month` in fe_spec already absorbs seasonal precip, so precip is
# identified off anomalous (within-month, across-year) variation.
PRECIP_LONG_PATH = ("/Users/rachelmurray-watson/Documents/Heat_data/"
                    "Thermofeel_WBGT/Indices/precip_long.csv")
PRECIP_TERMS = ["precip_5day"]#["precip_month", "precip_5day"]

# ---- Projection file layout ------------------------------------------------
# Long: rows are (facility, date), cols include wbgt_day, wbgt_night
WBGT_PROJ_FILE_TPL         = "wbgt_monthly_mean_facility_{tier}_{ssp}.csv"
# Wide: rows are 'YYYY-M' strings, columns are facility names, values = precip
PRECIP_5DAY_PROJ_FILE_TPL  = ("ResourceFile_Precipitation_Disruptions_{ssp}_{tier}_"
                              "window_prediction_weather_by_facility.csv")
PRECIP_MONTH_PROJ_FILE_TPL = ("ResourceFile_Precipitation_Disruptions_{ssp}_{tier}_"
                              "monthly_total_weather_by_facility.csv")
PROJ_PERIOD_START = 2025
PROJ_PERIOD_END   = 2040


def _norm_sf(x):
    """P(Z > x); avoids scipy so we don't clash with R's BLAS on macOS."""
    return 0.5 * math.erfc(x / math.sqrt(2))


# ---------------------------------------------------------------------------
# R helpers
# ---------------------------------------------------------------------------
def district_deficit_analytical(mu_a, mu_b, X_data_a, X_data_b,
                                 names_a, vcov_a, names_b, vcov_b,
                                 group_ids):
    """
    Per-district delta-method CI on the deficit.
    Aggregates gradient by group; SE is delta on the linear combination
    of coefficients that produce that group's summed deficit.
    """
    mu_a = np.where(np.isnan(mu_a), 0.0, mu_a)
    mu_b = np.where(np.isnan(mu_b), 0.0, mu_b)
    vcov_a = np.where(np.isnan(vcov_a), 0.0, vcov_a)
    vcov_b = np.where(np.isnan(vcov_b), 0.0, vcov_b)

    results = []
    for grp in np.unique(group_ids):
        mask = group_ids == grp
        sum_a, sum_b = float(mu_a[mask].sum()), float(mu_b[mask].sum())
        if sum_b <= 0:
            results.append({
                "district": grp, "deficit_pct": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan,
                "n_facility_months": int(mask.sum()),
            })
            continue

        delta = 100.0 * (sum_a - sum_b) / sum_b

        g_a = np.zeros(len(names_a))
        for j, nm in enumerate(names_a):
            if nm in X_data_a.columns:
                g_a[j] = (100.0 / sum_b) * float(
                    (mu_a[mask] * X_data_a[nm].values[mask]).sum())
        g_b = np.zeros(len(names_b))
        for j, nm in enumerate(names_b):
            if nm in X_data_b.columns:
                g_b[j] = (-100.0 * sum_a / sum_b**2) * float(
                    (mu_b[mask] * X_data_b[nm].values[mask]).sum())

        var = float(g_a @ vcov_a @ g_a) + float(g_b @ vcov_b @ g_b)
        se = float(np.sqrt(max(var, 0.0)))
        results.append({
            "district": grp, "deficit_pct": delta,
            "ci_lo": delta - 1.96 * se, "ci_hi": delta + 1.96 * se,
            "n_facility_months": int(mask.sum()),
        })
    return pd.DataFrame(results)


def fit_nb_fixest(df, rhs_terms, fe_spec, fe_cols, cluster_col, y_col="y_int"):
    rhs = " + ".join(rhs_terms) if rhs_terms else "1"
    fe  = " + ".join(fe_spec)                      # <-- string may contain ^
    fml = ro.Formula(f"{y_col} ~ {rhs} | {fe}")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    r_df = base.as_data_frame(r_df)

    for col in set(fe_cols + [cluster_col]):       # <-- only real columns
        r_df.rx2[col] = base.as_factor(r_df.rx2(col))

    suppressWarnings = ro.r("suppressWarnings")
    r_model = suppressWarnings(
        fixest.fenegbin(
            fml=fml,
            data=r_df,
            cluster=ro.StrVector([cluster_col]),
        )
    )
    print(f"    [fit_nb] fenegbin done, rhs={len(rhs_terms)}", flush=True)

    mu_r = stats_r.fitted(r_model)
    with localconverter(ro.default_converter + pandas2ri.converter):
        mu = np.asarray(ro.conversion.rpy2py(mu_r), dtype=float)

    if len(mu) != len(df):
        raise ValueError(
            f"fenegbin returned {len(mu)} fitted values for {len(df)} rows."
        )
    return r_model, mu


def predict_fixest(r_model, py_df, factor_cols=("facility", "month")):
    """Direct predict from a fitted fixest model on new data. Rows with
    factor levels the model wasn't fit on come back as NaN — filter
    facilities to the training set before calling to avoid that."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(py_df)
    r_df = base.as_data_frame(r_df)
    for col in factor_cols:
        if col in py_df.columns:
            r_df.rx2[col] = base.as_factor(r_df.rx2(col))
    predict_r = ro.r("predict")
    mu_r = predict_r(r_model, newdata=r_df, type="response")
    with localconverter(ro.default_converter + pandas2ri.converter):
        return np.asarray(ro.conversion.rpy2py(mu_r), dtype=float)


def nb_coef_table(r_model) -> pd.DataFrame:
    coeftab = fixest.coeftable(r_model)
    with localconverter(ro.default_converter + pandas2ri.converter):
        tab = ro.conversion.rpy2py(coeftab)
    rownames = list(base.rownames(coeftab))
    tab = pd.DataFrame(np.asarray(tab))
    tab.index = rownames
    tab.index.name = "term"
    tab = tab.reset_index()
    if tab.shape[1] >= 5:
        tab = tab.iloc[:, :5].copy()
        tab.columns = ["term", "estimate", "se", "z", "p"]
    else:
        raise ValueError(f"Unexpected coeftable shape: {tab.shape}.")
    return tab


def nb_aic(r_model) -> float:
    """AIC from a fenegbin model."""
    try:
        return float(ro.r("AIC")(r_model)[0])
    except Exception:
        pass
    try:
        ll_obj = ro.r("logLik")(r_model)
        ll     = float(ll_obj[0])
        ro.globalenv["._tmp_model"] = r_model
        npar = int(ro.r('attr(logLik(._tmp_model), "df")')[0])
        del ro.globalenv["._tmp_model"]
        return -2 * ll + 2 * npar
    except Exception:
        return np.nan


def nb_theta(r_model) -> float:
    for attr in ["theta", "family.theta"]:
        try:
            val = r_model.rx2(attr)
            return float(val[0])
        except Exception:
            continue
    try:
        summ = fixest.summary_fixest(r_model)
        return float(summ.rx2("theta")[0])
    except Exception:
        return np.nan


def get_beta_vcov(r_model):
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


def nb_coefficient_lookup(r_model) -> dict[str, float]:
    tab = nb_coef_table(r_model)
    return dict(zip(tab["term"].astype(str), tab["estimate"].astype(float)))


# ---------------------------------------------------------------------------
# spline_irr_at — delta-method IRR contrast
# ---------------------------------------------------------------------------
def spline_irr_at(wbgt_values, design_info, spline_cols, names_a,
                  beta_a, vcov_a, wbgt_ref, wbgt_shift):
    wbgt_all = np.concatenate([[wbgt_ref], np.asarray(wbgt_values, dtype=float)])
    wbgt_all_c = wbgt_all - wbgt_shift
    dmat = np.asarray(
        patsy.build_design_matrices([design_info], {"x": wbgt_all_c})[0])
    dmat     = dmat[:, -len(spline_cols):]
    contrast = dmat[1:] - dmat[0][None, :]

    idx  = [names_a.index(nm) for nm in spline_cols if nm in names_a]
    beta = beta_a[idx]
    V    = vcov_a[np.ix_(idx, idx)]
    V    = np.where(np.isnan(V), 0.0, V)

    log_irr = contrast @ beta
    var     = np.einsum("ij,jk,ik->i", contrast, V, contrast)
    se      = np.sqrt(np.maximum(var, 0.0))
    return (np.exp(log_irr),
            np.exp(log_irr - 1.96 * se),
            np.exp(log_irr + 1.96 * se))


# ---------------------------------------------------------------------------
# Deficit helpers
# ---------------------------------------------------------------------------
def two_model_deficit_analytical(mu_a, mu_b, names_a, vcov_a,
                                  names_b, vcov_b, X_data):
    """Delta-method SE for aggregate deficit. Used for main forest plot."""
    mu_a  = np.where(np.isnan(mu_a), 0.0, mu_a)
    mu_b  = np.where(np.isnan(mu_b), 0.0, mu_b)
    sum_a = float(mu_a.sum())
    sum_b = float(mu_b.sum())
    if sum_b <= 0:
        return dict(deficit_pct=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    se_pct=np.nan, p_analytical=np.nan)
    deficit_pct = 100.0 * (sum_b - sum_a) / sum_b

    def _grad(mu_vec, names, scale):
        g = np.zeros(len(names))
        for j, name in enumerate(names):
            if name in X_data.columns:
                g[j] = scale * float((mu_vec * X_data[name].values).sum())
        return g

    g_a = _grad(mu_a, names_a,  100.0 / sum_b)
    g_b = _grad(mu_b, names_b, -100.0 * sum_a / sum_b**2)
    vcov_a_clean = np.where(np.isnan(vcov_a), 0.0, vcov_a)
    vcov_b_clean = np.where(np.isnan(vcov_b), 0.0, vcov_b)
    var_a  = float(g_a @ vcov_a_clean @ g_a)
    var_b  = float(g_b @ vcov_b_clean @ g_b)
    se_pct = float(np.sqrt(max(var_a + var_b, 0.0)))
    z      = deficit_pct / se_pct if se_pct > 0 else np.nan
    p      = float(2 * _norm_sf(abs(z))) if np.isfinite(z) else np.nan
    return dict(deficit_pct=deficit_pct,
                ci_lo=deficit_pct - 1.96 * se_pct,
                ci_hi=deficit_pct + 1.96 * se_pct,
                se_pct=se_pct, p_analytical=p)


def monthly_deficit_with_jackknife_ci(
    mu_a: np.ndarray,
    mu_b: np.ndarray,
    facility_ids: np.ndarray,
) -> dict:
    """
    Point estimate + 95% jackknife CI for (sum_B - sum_A)/sum_B * 100.
    Leave-one-facility-out. Used for the monthly descriptive panel only.
    """
    sum_a = float(mu_a.sum())
    sum_b = float(mu_b.sum())
    if sum_b <= 0:
        return dict(deficit_pct=np.nan, ci_lo=np.nan, ci_hi=np.nan)

    deficit_pct = 100.0 * (sum_b - sum_a) / sum_b
    facs = np.unique(facility_ids)
    n    = len(facs)
    if n < 3:
        return dict(deficit_pct=deficit_pct, ci_lo=np.nan, ci_hi=np.nan)

    jack_vals = []
    for fac in facs:
        keep = facility_ids != fac
        sa_j = float(mu_a[keep].sum())
        sb_j = float(mu_b[keep].sum())
        if sb_j <= 0:
            continue
        jack_vals.append(100.0 * (sa_j - sb_j) / sb_j)

    if len(jack_vals) < 3:
        return dict(deficit_pct=deficit_pct, ci_lo=np.nan, ci_hi=np.nan)

    jack      = np.asarray(jack_vals)
    jack_mean = jack.mean()
    jack_se   = np.sqrt((n - 1) / n * np.sum((jack - jack_mean) ** 2))
    return dict(
        deficit_pct=deficit_pct,
        ci_lo=deficit_pct - 1.96 * jack_se,
        ci_hi=deficit_pct + 1.96 * jack_se,
    )


# ---------------------------------------------------------------------------
# Winsorisation
# ---------------------------------------------------------------------------
def winsorise_by_facility(
    df: pd.DataFrame,
    y_col: str = "y",
    k: float = 5.0,
    indicator: str = "",
    out_dir: str = OUT_DIR,
) -> pd.DataFrame:
    df      = df.copy()
    flagged = []
    n_replaced = 0

    for fac, grp in df.groupby("facility"):
        vals = grp[y_col].dropna()
        q1, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        upper = q3 + k * iqr
        mask  = (df["facility"] == fac) & (df[y_col] > upper)
        n     = int(mask.sum())
        if n:
            for _, row in df[mask].iterrows():
                flagged.append({
                    "facility":    fac,
                    "date":        row.get("date", np.nan),
                    "y_orig":      row[y_col],
                    "upper_fence": upper,
                    "q3": q3, "iqr": iqr,
                })
            df.loc[mask, y_col] = upper
            n_replaced += n
            print(
                f"  [{indicator}] {fac}: {n} value(s) > {upper:.0f} "
                f"(Q3={q3:.0f}, IQR={iqr:.0f}) set to NaN"
            )

    if flagged:
        pd.DataFrame(flagged).to_csv(
            f"{out_dir}outliers_removed_{indicator}_{WBGT_VAR}.csv", index=False)
    print(f"  [{indicator}] Winsorisation total: {n_replaced} values replaced")
    return df


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def prepare_data(indicator: str) -> pd.DataFrame | None:
    panel_path = (
        f"{DATA_DIR}/All_predictors_processed/"
        f"regression_panel_{indicator}.csv"
    )
    if not os.path.exists(panel_path):
        print(f"  [{indicator}] Panel file not found — skipping.")
        return None

    long = pd.read_csv(panel_path, parse_dates=["date"])
    long = long.rename(columns={indicator: "y"})

    if WBGT_VAR not in long.columns:
        print(f"  [{indicator}] Missing {WBGT_VAR} — skipping.")
        return None

    if CLUSTER_COL not in long.columns:
        raise KeyError(
            f"[{indicator}] Cluster column '{CLUSTER_COL}' not present. "
            f"Available: {sorted(long.columns)}"
        )

    long["covid"] = long["date"].between(COVID_START, COVID_END).astype(int)

    for fac, start, end in CLOSURES:
        mask     = (long["date"].between(start, end)) & (long["facility"] == fac)
        n_masked = int(mask.sum())
        long.loc[mask, "y"] = 0
        if n_masked:
            print(f"  [{indicator}] Masked {n_masked} closure months for {fac}.")

    long["year"]  = long["date"].dt.year
    long["month"] = long["date"].dt.month
    long = long[long["year"].between(min_year_historical, max_year_historical - 1)]

    # ---- Precip merge (confounder) ------------------------------------
    # Normalise the join key on both sides so a day-of-month or whitespace
    # mismatch can't silently drop rows. Then audit coverage: NAs here mean
    # facility-name mismatch or an out-of-precip-window date, and either
    # needs to be understood before the fit.
    long["date"] = long["date"].dt.to_period("M").dt.to_timestamp()
    precip = pd.read_csv(PRECIP_LONG_PATH, parse_dates=["date"])
    precip["facility"] = precip["facility"].astype(str).str.strip()
    precip["date"]     = precip["date"].dt.to_period("M").dt.to_timestamp()
    precip = precip[precip["date"] <= "2024-12-01"]
    precip = precip[precip["date"] >= "2015-02-01"]
    long = long.merge(precip, on=["facility", "date"], how="left")

    # -------------------------------------------------------------------

    if apply_cap:
        long = winsorise_by_facility(
            long, y_col="y", k=WINSOR_K, indicator=indicator)

    obs_per_fac = long.dropna(subset=["y", WBGT_VAR]).groupby("facility").size()
    keep_facs   = obs_per_fac[obs_per_fac >= MIN_OBS].index
    long        = long[long["facility"].isin(keep_facs)].copy()

    if long.empty or long["facility"].nunique() < 2:
        print(f"  [{indicator}] Too few facilities after filter — skipping.")
        return None

    dist_per_fac = long.groupby("facility")[CLUSTER_COL].nunique()
    bad = dist_per_fac[dist_per_fac != 1]
    if len(bad):
        raise ValueError(
            f"[{indicator}] {len(bad)} facilities map to multiple districts: "
            f"{list(bad.index[:10])}"
        )
    if long[CLUSTER_COL].isna().any():
        n_bad = int(long[CLUSTER_COL].isna().sum())
        raise ValueError(
            f"[{indicator}] {n_bad} rows have missing '{CLUSTER_COL}'.")

    return long


def enforce_complete_monthly_grid(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    df       = df.sort_values(["facility", "date"]).reset_index(drop=True)
    n_before = len(df)
    frames   = []
    for fac, g in df.groupby("facility", sort=False):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="MS")
        g    = g.set_index("date").reindex(full)
        g.index.name = "date"
        g["facility"] = fac
        frames.append(g.reset_index())
    out = pd.concat(frames, ignore_index=True)

    n_inserted = len(out) - n_before
    if n_inserted:
        print(
            f"  [{indicator}] Inserted {n_inserted:,} placeholder rows "
            f"({100 * n_inserted / len(out):.2f}% of grid)."
        )

    out["year"]  = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["covid"] = out["date"].between(COVID_START, COVID_END).astype(int)
    out[CLUSTER_COL] = out.groupby("facility")[CLUSTER_COL].transform(
        lambda s: s.ffill().bfill()
    )

    diffs       = out.groupby("facility")["date"].diff().dropna()
    bad_spacing = diffs[~((diffs.dt.days >= 28) & (diffs.dt.days <= 31))]
    if len(bad_spacing):
        raise ValueError(
            f"[{indicator}] Non-monthly spacing remains after reindexing.")
    return out


def add_columns(
    df: pd.DataFrame, indicator: str
) -> tuple[pd.DataFrame, list[str], float, float]:
    df = enforce_complete_monthly_grid(df, indicator)

    year_shift = df["year"].mean()    if CENTER else 0.0
    wbgt_shift = df[WBGT_VAR].mean() if CENTER else 0.0

    df["year_c"] = df["year"]    - year_shift
    df["wbgt_c"] = df[WBGT_VAR] - wbgt_shift

    lag_terms = []
    for lag in LAG_MONTHS:
        col = f"wbgt_lag{lag}_c"
        df[col] = df.groupby("facility")[WBGT_VAR].shift(lag) - wbgt_shift
        lag_terms.append(col)

    return df, lag_terms, year_shift, wbgt_shift


def add_spline_basis(
    df: pd.DataFrame, df_spline: int
) -> tuple[pd.DataFrame, list[str], object]:
    basis = patsy.dmatrix(
        f"cr(x, df={df_spline}) - 1",
        {"x": df["wbgt_c"].values},
        return_type="dataframe",
    )
    design_info = basis.design_info
    cols = [f"wbgt_s{i+1}" for i in range(basis.shape[1])]
    for c, b in zip(cols, basis.columns):
        df[c] = basis[b].values
    return df, cols, design_info


def select_df_for_indicator(nb_data, lag_terms, wbgt_var, cluster_col,
                             fe_spec, fe_cols, candidates):
    """Fit Model A at each candidate df; return AIC table and best df.
    Must match the final Model A specification exactly, so precip terms
    are included here too."""
    rows = []
    for df_try in candidates:
        try:
            nb_try, splines_try, _ = add_spline_basis(nb_data.copy(), df_spline=df_try)
            rhs_try = splines_try + lag_terms + PRECIP_TERMS + ["covid", "year_c"]
            m_try, _ = fit_nb_fixest(
                nb_try, rhs_terms=rhs_try,
                fe_spec=fe_spec, fe_cols = fe_cols, cluster_col=cluster_col)
            rows.append({"df": df_try, "aic": nb_aic(m_try),
                         "theta": nb_theta(m_try), "note": ""})
        except Exception as e:
            rows.append({"df": df_try, "aic": np.nan, "theta": np.nan,
                         "note": str(e)[:60]})
    tab   = pd.DataFrame(rows)
    valid = tab.dropna(subset=["aic"])
    best_df = (int(valid.loc[valid["aic"].idxmin(), "df"])
               if not valid.empty else candidates[0])
    return tab, best_df


def drop_separated_facilities(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    all_zero = df.groupby("facility")["y_int"].max() == 0
    sep_facs = all_zero[all_zero].index
    if len(sep_facs):
        n_rows = int(df["facility"].isin(sep_facs).sum())
        print(
            f"  [{indicator}] SEPARATION: {len(sep_facs)} facilities "
            f"all-zero ({n_rows:,} rows) — dropped."
        )
        df = df[~df["facility"].isin(sep_facs)].copy()
    return df


# ---------------------------------------------------------------------------
# Exposure-response curve
# ---------------------------------------------------------------------------
def make_exposure_response_curve(
    r_model, spline_cols, wbgt_shift, observed_wbgt, indicator,
    design_info, names_a=None, vcov_a=None,
) -> pd.DataFrame:
    coefs    = nb_coefficient_lookup(r_model)
    x_min    = float(observed_wbgt.min())
    x_max    = float(observed_wbgt.max())
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
    basis_ref = np.asarray(
        patsy.build_design_matrices([design_info], {"x": np.array([x_ref_c])})[0],
        dtype=float)
    ref_row = basis_ref[0]

    beta     = np.array([coefs.get(c, 0.0) for c in spline_cols], dtype=float)
    contrast = basis_grid - ref_row[None, :]
    eta_grid = contrast @ beta
    rr       = np.exp(eta_grid)

    if vcov_a is not None and names_a is not None:
        idx = [names_a.index(c) for c in spline_cols if c in names_a]
        if len(idx) == len(spline_cols):
            V = vcov_a[np.ix_(idx, idx)]
            V = np.where(np.isnan(V), 0.0, V)
            var_grid = np.einsum("ij,jk,ik->i", contrast, V, contrast)
            se_grid  = np.sqrt(np.maximum(var_grid, 0.0))
            rr_lo = np.exp(eta_grid - 1.96 * se_grid)
            rr_hi = np.exp(eta_grid + 1.96 * se_grid)
        else:
            rr_lo = np.full_like(rr, np.nan)
            rr_hi = np.full_like(rr, np.nan)
    else:
        rr_lo = np.full_like(rr, np.nan)
        rr_hi = np.full_like(rr, np.nan)

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


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------
def bh_fdr(pvals, alpha=0.05):
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


# ---------------------------------------------------------------------------
# Bootstrap replicate
# ---------------------------------------------------------------------------
def _boot_replicate(seed_seq, nb_data, dist_index, dist_ids,
                    rhs_a, rhs_b, fe_spec, fe_cols, cluster_col,
                    hot_threshold):
    ro.conversion.converter_ctx.set(ro.default_converter)
    rng   = np.random.default_rng(seed_seq)
    picks = rng.choice(len(dist_ids), size=len(dist_ids), replace=True)
    idx_parts = [dist_index[dist_ids[p]] for p in picks]
    idx  = np.concatenate(idx_parts)
    tags = np.repeat(
        np.arange(len(picks)), [len(p) for p in idx_parts]).astype(str)
    boot_df = nb_data.take(idx).reset_index(drop=True)
    tags_s  = pd.Series(tags, index=boot_df.index).astype(str)
    boot_df["facility"]  = boot_df["facility"].astype(str)  + "__b" + tags_s
    boot_df[cluster_col] = boot_df[cluster_col].astype(str) + "__b" + tags_s
    fac_max  = boot_df.groupby("facility")["y_int"].max()
    sep_facs = fac_max[fac_max == 0].index
    if len(sep_facs):
        boot_df = boot_df[~boot_df["facility"].isin(sep_facs)].copy()
    if boot_df["facility"].nunique() < 2 or boot_df[cluster_col].nunique() < 2:
        return None, "too_few_groups"
    try:
        _, mu_a = fit_nb_fixest(boot_df, rhs_a, fe_spec, fe_cols, cluster_col)
        _, mu_b = fit_nb_fixest(boot_df, rhs_b, fe_spec, fe_cols, cluster_col)
        total_a, total_b = float(mu_a.sum()), float(mu_b.sum())
        if total_b == 0:
            return None, "zero_counterfactual"
        pct = 100.0 * (total_b - total_a) / total_b
        if not np.isfinite(pct):
            return None, "non-finite"
        # hot-month deficit on the SAME resample (this is what the forest plot
        # shows, so it needs a bootstrap CI too, not the analytical one).
        hot = boot_df[WBGT_VAR].values > hot_threshold
        pct_hot = np.nan
        if hot.any():
            ta_h, tb_h = float(mu_a[hot].sum()), float(mu_b[hot].sum())
            if tb_h > 0:
                pct_hot = 100.0 * (tb_h - ta_h) / tb_h
        return (pct, pct_hot), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:80]}"


# ---------------------------------------------------------------------------
# Per-indicator runner
# ---------------------------------------------------------------------------
def run_indicator(indicator: str) -> dict | None:

    long = prepare_data(indicator)
    if long is None:
        return None

    long, lag_terms, year_shift, wbgt_shift = add_columns(long, indicator)

    # Precip in nb_cols so placeholder-grid rows and unmatched-facility rows
    # are dropped here, not silently kept into fenegbin.
    nb_cols = (["y", "facility",  "month","year_c", "wbgt_c", "covid", CLUSTER_COL]
               + lag_terms + PRECIP_TERMS)
    nb_data = long.dropna(subset=nb_cols).copy()
    nb_data["y_int"] = nb_data["y"].round().clip(lower=0).astype(int)

    obs_nb  = nb_data.groupby("facility").size()
    nb_data = nb_data[
        nb_data["facility"].isin(obs_nb[obs_nb >= MIN_OBS].index)].copy()

    nb_data = drop_separated_facilities(nb_data, indicator)

    if nb_data.empty or nb_data["facility"].nunique() < 2:
        print(f"  [{indicator}] Sample too small — skipping.")
        return None
    if nb_data[CLUSTER_COL].nunique() < 2:
        raise ValueError(
            f"[{indicator}] Only {nb_data[CLUSTER_COL].nunique()} cluster(s).")

    nb_data = nb_data.reset_index(drop=True)

    fe_cols = ["facility", "Dist", "month"]
    fe_spec = ["facility", "month"]
    # ------------------------------------------------------------------
    # Per-indicator df selection via AIC
    # ------------------------------------------------------------------
    deficit: dict = {}

    if DF_MODE == "per_indicator":
        print(f"  [{indicator}] selecting df via AIC over {list(DF_CANDIDATES)}...")
        df_table, chosen_df = select_df_for_indicator(
            nb_data,
            lag_terms,
            WBGT_VAR,
            CLUSTER_COL,
            fe_spec=fe_spec,
            fe_cols=fe_cols,
            candidates=DF_CANDIDATES,
        )
        df_table.insert(0, "indicator", indicator)
        for _, r in df_table.iterrows():
            note   = f"  {r['note']}" if r["note"] else ""
            marker = " <-- PRIMARY" if int(r["df"]) == chosen_df else ""
            if pd.isna(r["aic"]):
                print(f"    df={int(r['df'])}: FAILED{note}")
            else:
                print(f"    df={int(r['df'])}: AIC={r['aic']:.1f}, "
                      f"theta={r['theta']:.2f}{marker}{note}")
        deficit["_df_table"] = df_table
    else:
        chosen_df = SPLINE_DF
        deficit["_df_table"] = pd.DataFrame([{"df": chosen_df, "aic": np.nan}])

    deficit["chosen_df"] = chosen_df
    deficit["reference_wbgt"] = np.nanpercentile(nb_data[WBGT_VAR], REFERENCE_WBGT_PERCENTILE)

    # ------------------------------------------------------------------
    # Fit Model A and Model B at chosen df
    # Precip is a CONFOUNDER: enters both models. Deficit is then
    # WBGT-attributable disruption holding precip fixed.
    # ------------------------------------------------------------------
    nb_data, spline_cols, design_info = add_spline_basis(nb_data, chosen_df)
    nb_data = nb_data.reset_index(drop=True)

    rhs_a = spline_cols + lag_terms + PRECIP_TERMS + ["covid", "year_c"]
    rhs_b = PRECIP_TERMS + ["covid", "year_c"]

    try:
        model_a, mu_a = fit_nb_fixest(
            nb_data, rhs_terms=rhs_a,
            fe_spec=fe_spec, fe_cols = fe_cols, cluster_col=CLUSTER_COL)
        model_b, mu_b = fit_nb_fixest(
            nb_data, rhs_terms=rhs_b,
            fe_spec=fe_spec, fe_cols = fe_cols, cluster_col=CLUSTER_COL)
    except Exception as e:
        print(f"  [{indicator}] fenegbin failed: {e} — skipping.")
        return None

    total_a = float(mu_a.sum())
    total_b = float(mu_b.sum())
    total_y = float(nb_data["y_int"].sum())
    print(
        f"  [{indicator}] sum(y)={total_y:,.0f}, "
        f"sum(mu_A)={total_a:,.0f}, sum(mu_B)={total_b:,.0f}"
    )

    deficit.update({
        "total_pred_exposure":       total_a,
        "total_pred_counterfactual": total_b,
        "deficit_abs":               total_b - total_a,
        "deficit_pct":               100.0 * (total_b - total_a) / total_b,
        "theta_a":                   nb_theta(model_a),
        "theta_b":                   nb_theta(model_b),
    })

    # ------------------------------------------------------------------
    # CIs
    # ------------------------------------------------------------------
    for k in ("ci_lo", "ci_hi", "deficit_se", "p_analytical",
              "p_hot_analytical", "hot_deficit_pct",
              "hot_deficit_ci_lo", "hot_deficit_ci_hi"):
        deficit[k] = np.nan
    deficit["p_boot"]    = np.nan
    deficit["n_boot_ok"] = 0

    names_a = names_b = beta_a = beta_b = vcov_a = vcov_b = None

    try:
        names_a, beta_a, vcov_a = get_beta_vcov(model_a)
        names_b, beta_b, vcov_b = get_beta_vcov(model_b)

        delta = two_model_deficit_analytical(
            mu_a, mu_b, names_a, vcov_a, names_b, vcov_b, nb_data)
        deficit["ci_lo"]        = delta["ci_lo"]
        deficit["ci_hi"]        = delta["ci_hi"]
        deficit["deficit_se"]   = delta["se_pct"]
        deficit["p_analytical"] = delta["p_analytical"]
        deficit["p_boot"]       = delta["p_analytical"]

        hot = nb_data[WBGT_VAR].values > deficit["reference_wbgt"]
        if hot.any():
            dh = two_model_deficit_analytical(
                mu_a[hot], mu_b[hot],
                names_a, vcov_a, names_b, vcov_b,
                nb_data.loc[hot].reset_index(drop=True))
            deficit["hot_deficit_pct"]    = dh["deficit_pct"]
            deficit["hot_deficit_ci_lo"]  = dh["ci_lo"]
            deficit["hot_deficit_ci_hi"]  = dh["ci_hi"]
            deficit["p_hot_analytical"]   = dh["p_analytical"]

        deficit["_names_a"], deficit["_beta_a"], deficit["_vcov_a"] = (
            names_a, beta_a, vcov_a)
        deficit["_names_b"], deficit["_vcov_b"] = names_b, vcov_b

    except Exception as e:
        print(f"  [{indicator}] CI extraction failed: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
    try:
        curve_df = make_exposure_response_curve(
            model_a, spline_cols, wbgt_shift,
            nb_data[WBGT_VAR], indicator, design_info,
            names_a=names_a, vcov_a=vcov_a,
        )
        curve_df.to_csv(
            f"{OUT_DIR}exposure_response_curve_{indicator}_{WBGT_VAR}.csv", index=False)
    except Exception as e:
        print(f"  [{indicator}] Curve export failed: {e}")

    # ------------------------------------------------------------------
    # Predictions CSV
    # ------------------------------------------------------------------
    preds = nb_data[["facility", "date"]].copy()
    preds["y_obs"]       = nb_data["y_int"].values
    preds["y_pred_wx"]   = mu_a
    preds["y_pred_base"] = mu_b
    preds["difference"]  = preds["y_pred_base"] - preds["y_pred_wx"]
    if PANEL_DIST_COL_IN_PANEL in nb_data.columns:
        preds["Dist"] = nb_data[PANEL_DIST_COL_IN_PANEL].values
    preds.to_csv(f"{OUT_DIR}two_model_predictions_{indicator}_{WBGT_VAR}.csv", index=False)

    try:
        coef_tab = nb_coef_table(model_a)
        coef_tab["indicator"] = indicator
        coef_tab.to_csv(
            f"{OUT_DIR}coef_table_model_a_{indicator}_{WBGT_VAR}.csv", index=False)
    except Exception as e:
        print(f"  [{indicator}] Coef table export failed: {e}")

    # ------------------------------------------------------------------
    # Bootstrap (only if N_BOOTSTRAP > 0)
    # ------------------------------------------------------------------
    if N_BOOTSTRAP > 0:
        failures: Counter = Counter()
        dist_ids   = nb_data[CLUSTER_COL].unique()
        dist_index = {
            d: np.asarray(g, dtype=np.int64)
            for d, g in nb_data.groupby(
                CLUSTER_COL, sort=False).indices.items()
        }
        seeds = np.random.SeedSequence(BOOT_SEED).spawn(N_BOOTSTRAP)
        hot_thr = deficit["reference_wbgt"]
        out   = [
            _boot_replicate(s, nb_data, dist_index, dist_ids,
                            rhs_a, rhs_b, fe_spec, fe_cols, CLUSTER_COL,
                            hot_thr)
            for s in seeds
        ]
        boot_pcts     = [v[0] for v, err in out if err is None]
        boot_pcts_hot = [v[1] for v, err in out
                         if err is None and np.isfinite(v[1])]
        for _, err in out:
            if err is not None:
                failures[err] += 1
        n_ok = len(boot_pcts)
        if failures:
            print(f"  [{indicator}] Bootstrap failures: {dict(failures)}")
        if n_ok / N_BOOTSTRAP < BOOT_MIN_SUCCESS:
            raise RuntimeError(
                f"[{indicator}] Only {n_ok}/{N_BOOTSTRAP} replicates converged.")
        alpha     = 1 - BOOT_CI_LEVEL
        boot_arr  = np.asarray(boot_pcts)
        # --- aggregate deficit CI (bootstrap) ---
        deficit["ci_lo"]     = float(np.percentile(boot_arr, 100 * alpha / 2))
        deficit["ci_hi"]     = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
        deficit["n_boot_ok"] = n_ok
        frac_le = float(np.mean(boot_arr <= 0))
        frac_ge = float(np.mean(boot_arr >= 0))
        deficit["p_boot"] = float(
            min(1.0, max(2 * min(frac_le, frac_ge), 1.0 / (n_ok + 1))))
        # --- hot-month deficit CI (bootstrap) — THIS is what the forest plot
        #     shows, so overwrite the analytical hot CI with the bootstrap one ---
        if len(boot_pcts_hot) >= BOOT_MIN_SUCCESS * N_BOOTSTRAP:
            hot_arr = np.asarray(boot_pcts_hot)
            deficit["hot_deficit_ci_lo"] = float(
                np.percentile(hot_arr, 100 * alpha / 2))
            deficit["hot_deficit_ci_hi"] = float(
                np.percentile(hot_arr, 100 * (1 - alpha / 2)))
            hle = float(np.mean(hot_arr <= 0)); hge = float(np.mean(hot_arr >= 0))
            deficit["p_hot_boot"] = float(
                min(1.0, max(2 * min(hle, hge), 1.0 / (len(hot_arr) + 1))))
        else:
            print(f"  [{indicator}] hot-month bootstrap: only "
                  f"{len(boot_pcts_hot)} usable replicates — hot CI left as NaN.")
        pd.DataFrame({"deficit_pct": boot_pcts}).to_csv(
            f"{OUT_DIR}bootstrap_distribution_{indicator}_{WBGT_VAR}.csv", index=False)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    deficit["indicator"] = indicator
    deficit["label"]     = INDICATOR_LABELS.get(indicator, indicator)
    deficit["n_obs"]     = int(len(nb_data))
    deficit["n_fac"]     = nb_data["facility"].nunique()
    deficit["n_clust"]   = nb_data[CLUSTER_COL].nunique()
    deficit["spline_df"] = chosen_df
    deficit["curve_ref"] = CURVE_REF

    deficit["_nb_data"]     = nb_data
    deficit["_mu_a"]        = mu_a
    deficit["_mu_b"]        = mu_b
    deficit["_model_a"]     = model_a
    deficit["_model_b"]     = model_b
    deficit["_spline_cols"] = spline_cols
    deficit["_design_info"] = design_info
    deficit["_wbgt_shift"]  = wbgt_shift
    deficit["_year_shift"]  = year_shift

    print(
        f"  [{indicator}] OK  n={len(nb_data):,}, "
        f"fac={nb_data['facility'].nunique()}, "
        f"clust={nb_data[CLUSTER_COL].nunique()}, "
        f"θ_a={deficit['theta_a']:.2f}, θ_b={deficit['theta_b']:.2f}, "
        f"deficit={deficit['deficit_pct']:+.2f}%"
    )
    return deficit


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Two-model NB analysis: exposure vs counterfactual")
    print(f"Estimator  = NB FE (fixest::fenegbin), CRV1 on {CLUSTER_COL}")
    print(f"Spline df  = {SPLINE_DF} (mode={DF_MODE}),  lags = {LAG_MONTHS}")
    print(f"Exposure   = {WBGT_VAR}")
    print(f"Confounder = {PRECIP_TERMS} (in both Model A and Model B)")
    print(f"MIN_OBS    = {MIN_OBS},  WINSOR_K = {WINSOR_K}")
    print(f"Bootstrap  = {N_BOOTSTRAP}")
    print("=" * 60)

    all_results: list[dict] = []
    for ind in COUNT_INDICATORS:
        print(f"\n-> {ind}")
        result = run_indicator(ind)
        if result is not None:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No indicators fitted — check panel paths.")

    results_df = pd.DataFrame(all_results)
    fitted     = [r["indicator"] for r in all_results]

    q, rej = bh_fdr(results_df["p_boot"].values, alpha=FDR_ALPHA)
    results_df["q_bh"]   = q
    results_df["sig_bh"] = rej

    save_cols = [c for c in results_df.columns if not c.startswith("_")]
    results_df[save_cols].to_csv(
        f"{OUT_DIR}two_model_deficit_results_NB_{WBGT_VAR}.csv", index=False)
    print(f"\nResults -> {OUT_DIR}two_model_deficit_results_NB_{WBGT_VAR}.csv")
    print(
        f"BH-FDR alpha={FDR_ALPHA}: "
        f"{int(results_df['sig_bh'].sum())}/{len(results_df)} significant.")

    # -----------------------------------------------------------------------
    # KEY STASH DIAGNOSTIC
    # -----------------------------------------------------------------------
    print("\n=== KEY STASH DIAGNOSTIC ===")
    for res in all_results:
        has = all(k in res for k in
                  ("_design_info", "_spline_cols",
                   "_names_a", "_beta_a", "_vcov_a"))
        print(f"  {res['indicator']}: irr_keys={has}")
    print("============================\n")

    # -----------------------------------------------------------------------
    # MAIN FOREST PLOT (delta-method CI)
    # -----------------------------------------------------------------------
    plot_df = (results_df.sort_values("deficit_pct")
               .reset_index(drop=True))
    y_pos  = np.arange(len(plot_df))
    has_ci = plot_df["ci_lo"].notna().any()
    colors = [
        "#823038" if bool(r["sig_bh"]) else "#888888"
        if has_ci else "#4a7298"
        for _, r in plot_df.iterrows()
    ]

    # -----------------------------------------------------------------------
    # HOT-MONTH FOREST PLOT (delta-method CI)
    # -----------------------------------------------------------------------
    q_hot, rej_hot = bh_fdr(
        results_df["p_hot_analytical"].values, alpha=FDR_ALPHA)
    results_df["q_hot_bh"]   = q_hot
    results_df["sig_hot_bh"] = rej_hot

    print("\n=== HOT-MONTH CI DIAGNOSTIC ===")
    diag = results_df[[
        "indicator", "hot_deficit_pct",
        "hot_deficit_ci_lo", "hot_deficit_ci_hi",
        "p_hot_analytical", "sig_hot_bh",
    ]].copy()
    diag["ci_width"] = diag["hot_deficit_ci_hi"] - diag["hot_deficit_ci_lo"]
    print(diag.to_string())
    print(f"CIs present: {diag['hot_deficit_ci_lo'].notna().sum()} / {len(diag)}")
    print(f"BH-significant: {int(diag['sig_hot_bh'].sum())} / {len(diag)}")
    print("================================\n")

    ph = (results_df.dropna(subset=["hot_deficit_pct"])
          .sort_values("hot_deficit_pct").reset_index(drop=True))
    y_ph       = np.arange(len(ph))
    hot_colors = ["#823038" if s else "#888888" for s in ph["sig_hot_bh"]]


    # -----------------------------------------------------------------------
    # SPLINE IRR FOREST PLOT
    # -----------------------------------------------------------------------
    IRR_LOW  = results_df["reference_wbgt"].iloc[0]
    IRR_HIGH = 32.0
    irr_rows = []
    for res in all_results:
        if not all(k in res for k in ("_design_info", "_spline_cols",
                                      "_names_a", "_beta_a", "_vcov_a")):
            print(f"  [{res['indicator']}] missing IRR keys — skipping")
            continue
        try:
            irr, lo, hi = spline_irr_at(
                [IRR_HIGH],
                res["_design_info"],
                res["_spline_cols"],
                res["_names_a"],
                res["_beta_a"],
                res["_vcov_a"],
                wbgt_ref=IRR_LOW,
                wbgt_shift=res["_wbgt_shift"],
            )
            irr_rows.append({
                "indicator": res["indicator"],
                "label":     INDICATOR_LABELS.get(
                    res["indicator"], res["indicator"]),
                "irr":    float(irr[0]),
                "irr_lo": float(lo[0]),
                "irr_hi": float(hi[0]),
            })
        except Exception as e:
            print(f"  [{res['indicator']}] IRR failed: {e}")

    if irr_rows:
        irr_df = pd.DataFrame(irr_rows).sort_values("irr").reset_index(drop=True)
    else:
        irr_df = pd.DataFrame(columns=["indicator", "label", "irr", "irr_lo", "irr_hi"])
    irr_df.to_csv(
        f"{OUT_DIR}irr_contrast_{WBGT_VAR}.csv",
        index=False,
    )
    if not irr_df.empty:
        irr_colors = [
            "#823038" if (r["irr_hi"] < 1 or r["irr_lo"] > 1) else "#888888"
            for _, r in irr_df.iterrows()
        ]

    # -----------------------------------------------------------------------
    # HISTORICAL BURDEN + DISTRICT AGGREGATION
    # -----------------------------------------------------------------------
    print("\nHistorical burden...")
    burden_rows = []
    all_dist    = []

    for res in all_results:
        ind     = res["indicator"]
        nb_data = res["_nb_data"]
        mu_a    = res["_mu_a"]
        mu_b    = res["_mu_b"]

        bdf = nb_data[["facility", "date", "year", "month",
                        WBGT_VAR, "y_int", CLUSTER_COL]].copy()
        bdf["mu_a"]       = mu_a
        bdf["mu_b"]       = mu_b
        bdf["difference"] = mu_b - mu_a
        bdf["hot_month"]  = nb_data[WBGT_VAR].values > results_df["reference_wbgt"].iloc[0]
        bdf.to_csv(f"{OUT_DIR}historical_burden_{ind}_{WBGT_VAR}.csv", index=False)

        hot_mask = nb_data[WBGT_VAR].values > results_df["reference_wbgt"].iloc[0]
        burden_rows.append({
            "indicator":       ind,
            "label":           INDICATOR_LABELS.get(ind, ind),
            "total_mu_a":      float(np.nansum(mu_a)),
            "total_mu_b":      float(np.nansum(mu_b)),
            "deficit_pct":     res["deficit_pct"],
            "hot_deficit_pct": float(
                100.0 * (mu_b[hot_mask].sum() - mu_a[hot_mask].sum())
                / mu_b[hot_mask].sum()
            ) if hot_mask.any() else np.nan,
        })

        # ---- District point-estimate aggregation ----
        d = bdf.groupby(CLUSTER_COL).agg(
            obs=("y_int", "sum"),
            mu_a=("mu_a", "sum"),
            mu_b=("mu_b", "sum"),
        ).reset_index()

        d["deficit_pct"] = (d["mu_b"] - d["mu_a"]) / d["mu_b"] * 100
        d["indicator"]   = ind
        all_dist.append(d)

        d.to_csv(f"{OUT_DIR}district_burden_{ind}_{WBGT_VAR}.csv", index=False)

        group_ids = res["_nb_data"][CLUSTER_COL].values
        district_ci = district_deficit_analytical(
            mu_a=res["_mu_a"],
            mu_b=res["_mu_b"],
            X_data_a=res["_nb_data"],
            X_data_b=res["_nb_data"],
            names_a=res["_names_a"],
            vcov_a=res["_vcov_a"],
            names_b=res["_names_b"],
            vcov_b=res["_vcov_b"],
            group_ids=group_ids,
        )
        district_ci["sig"] = (
            (district_ci["ci_lo"] > 0) | (district_ci["ci_hi"] < 0)
        ) & district_ci["deficit_pct"].notna()
        # Save CIs to a separate file so they don't get overwritten
        district_ci.to_csv(
            f"{OUT_DIR}district_burden_ci_{ind}_{WBGT_VAR}.csv", index=False)

        print(f"  {ind}: {len(d)} districts, "
              f"mean deficit {d['deficit_pct'].mean():.2f}%")

        hot_mask = res["_nb_data"][WBGT_VAR].values > results_df["reference_wbgt"].iloc[0]
        if hot_mask.any():
            district_ci_hot = district_deficit_analytical(
                mu_a=res["_mu_a"][hot_mask],
                mu_b=res["_mu_b"][hot_mask],
                X_data_a=res["_nb_data"][hot_mask].reset_index(drop=True),
                X_data_b=res["_nb_data"][hot_mask].reset_index(drop=True),
                names_a=res["_names_a"],
                vcov_a=res["_vcov_a"],
                names_b=res["_names_b"],
                vcov_b=res["_vcov_b"],
                group_ids=res["_nb_data"][hot_mask][CLUSTER_COL].values,
            )
            district_ci_hot["sig"] = (
                (district_ci_hot["ci_lo"] > 0) | (district_ci_hot["ci_hi"] < 0)
            ) & district_ci_hot["deficit_pct"].notna()
            district_ci_hot.to_csv(
                f"{OUT_DIR}district_burden_hot_{ind}_{WBGT_VAR}.csv", index=False)

    pd.DataFrame(burden_rows).to_csv(
        f"{OUT_DIR}historical_burden_summary_{WBGT_VAR}.csv", index=False)

    if all_dist:
        pd.concat(all_dist, ignore_index=True).to_csv(
            f"{OUT_DIR}district_burden_all_{WBGT_VAR}.csv", index=False)



    # -----------------------------------------------------------------------
    # FORWARD PROJECTIONS  (direct predict — no scaling, no anchor)
    # For each (facility, year, month) in 2025-2040:
    #   build newdata with the same columns as the fit (wbgt_c, wbgt_lag*_c,
    #   precip_month, precip_5day, covid=0, year_c, facility, month, plus
    #   the spline basis columns wbgt_s1..), then call fixest::predict on
    #   Model A and Model B. Difference = Model B - Model A (positive = loss).
    # -----------------------------------------------------------------------
    print(f"\nForward projections ({PROJ_PERIOD_START}-{PROJ_PERIOD_END}, "
          "direct predict)...")

    def _load_precip_wide(path, value_name):
        """Wide file (index='YYYY-M', cols=facility names) -> long
        [facility, date, value_name]. None if the file is missing."""
        if not os.path.exists(path):
            return None
        wide = pd.read_csv(path, index_col=0)
        wide.index = pd.to_datetime(wide.index.astype(str).str.strip(),
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
        """Load WBGT + both precip files, merge to a per-(facility, date)
        frame, restrict to the projection window, add lag columns per
        facility. Returns (df, None) or (None, missing_paths)."""
        wbgt_path = os.path.join(PROJECTION_DIR,
            WBGT_PROJ_FILE_TPL.format(tier=tier, ssp=ssp))
        p5_path   = os.path.join(PROJECTION_DIR,
            PRECIP_5DAY_PROJ_FILE_TPL.format(ssp=ssp, tier=tier))
        pm_path   = os.path.join(PROJECTION_DIR,
            PRECIP_MONTH_PROJ_FILE_TPL.format(ssp=ssp, tier=tier))

        wbgt_df = _load_wbgt_proj(wbgt_path)
        p5_df   = _load_precip_wide(p5_path, "precip_5day")
        pm_df   = _load_precip_wide(pm_path, "precip_month")

        missing = []
        if wbgt_df is None: missing.append(wbgt_path)
        if p5_df   is None: missing.append(p5_path)
        if pm_df   is None: missing.append(pm_path)
        if missing:
            return None, missing

        clim = (wbgt_df
                .merge(p5_df, on=["facility", "date"], how="outer")
                .merge(pm_df, on=["facility", "date"], how="outer"))
        clim = clim.sort_values(["facility", "date"]).reset_index(drop=True)

        # Future WBGT lags — first max(LAG_MONTHS) rows per facility are NaN
        for k in LAG_MONTHS:
            clim[f"wbgt_lag{k}"] = clim.groupby("facility")[WBGT_VAR].shift(k)
        clim = clim[(clim["date"].dt.year >= PROJ_PERIOD_START) &
                    (clim["date"].dt.year <= PROJ_PERIOD_END)].copy()
        clim["year"]  = clim["date"].dt.year
        clim["month"] = clim["date"].dt.month
        return clim, None

    all_proj_summary = []

    for ssp in SSP_SCENARIOS:
        for tier in MODEL_TIERS:
            clim, missing = _load_future_climate(ssp, tier)
            if clim is None:
                print(f"  SKIP {ssp}/{tier}: missing {missing}")
                continue
            print(f"  {ssp}/{tier}: {len(clim):,} facility-months loaded "
                  f"({clim['facility'].nunique()} facilities)")

            for res in all_results:
                ind         = res["indicator"]
                wbgt_shift  = res["_wbgt_shift"]
                year_shift  = res["_year_shift"]
                design_info = res["_design_info"]
                spline_cols = res["_spline_cols"]

                # fixest::predict returns NA for unseen factor levels; keep
                # only facilities the model actually saw at fit time.
                train_facs = set(res["_nb_data"]["facility"].unique())
                df = clim[clim["facility"].isin(train_facs)].copy()

                # Build every column the RHS references, matching the fit
                # exactly: RHS = spline_cols + lag_terms + PRECIP_TERMS
                #                + ["covid", "year_c"]  (facility, month absorbed)
                df["covid"]  = 0                          # no covid in future
                df["year_c"] = (max_year_historical - 1) - year_shift     # anchor to the last fitted year
                df["wbgt_c"] = df[WBGT_VAR] - wbgt_shift
                for k in LAG_MONTHS:
                    df[f"wbgt_lag{k}_c"] = df[f"wbgt_lag{k}"] - wbgt_shift

                # Spline basis columns (wbgt_s1, wbgt_s2, ...) — same names
                # as fit because we reuse the stashed design_info.
                basis = np.asarray(
                    patsy.build_design_matrices(
                        [design_info], {"x": df["wbgt_c"].values})[0],
                    dtype=float)
                for i, c in enumerate(spline_cols):
                    df[c] = basis[:, i]

                # Drop rows that would go into predict() as NaN
                need = ([WBGT_VAR, "precip_month", "precip_5day", "year_c"]
                        + [f"wbgt_lag{k}_c" for k in LAG_MONTHS]
                        + list(spline_cols))
                n_before = len(df)
                df = df.dropna(subset=need).reset_index(drop=True)
                n_dropped = n_before - len(df)
                if df.empty:
                    print(f"    {ind}: no rows after covariate build - skipping")
                    continue

                # ---- Direct predict from each fitted model ----
                mu_a = predict_fixest(res["_model_a"], df)
                mu_b = predict_fixest(res["_model_b"], df)

                # Any residual NaN (e.g. facility-month combo the model didn't
                # actually fit, even though the facility was in training).
                ok = np.isfinite(mu_a) & np.isfinite(mu_b)
                n_pred_nan = int((~ok).sum())
                df   = df.loc[ok].reset_index(drop=True)
                mu_a = mu_a[ok]
                mu_b = mu_b[ok]

                # Sign convention: same as elsewhere in the script — positive
                # = visits lost to WBGT under this scenario.
                df["Predicted_Weather_Model"]    = mu_a
                df["Predicted_No_Weather_Model"] = mu_b
                df["Disruption"]                 = mu_b - mu_a
                df["Deficit_Pct"] = np.where(
                    mu_b > 0, 100.0 * (mu_b - mu_a) / mu_b, np.nan)

                # attach district for the district roll-up
                fac_dist = (res["_nb_data"][["facility", CLUSTER_COL]]
                            .drop_duplicates("facility"))
                df = df.merge(fac_dist, on="facility", how="left")

                df["indicator"] = ind
                df["ssp"]       = ssp
                df["tier"]      = tier

                # ---- Per-facility per-month CSV ----
                full_cols = [
                    "indicator", "ssp", "tier", "facility", CLUSTER_COL,
                    "year", "month", "date",
                    WBGT_VAR, "precip_month", "precip_5day",
                    "Predicted_Weather_Model", "Predicted_No_Weather_Model",
                    "Disruption", "Deficit_Pct",
                ]
                df[full_cols].to_csv(
                    f"{OUT_DIR}projection_facility_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
                    index=False)

                # ---- District × year × month roll-up ----
                dist_agg = df.groupby([CLUSTER_COL, "year", "month"]).agg(
                    Total_Predicted_Weather_Model    =("Predicted_Weather_Model",    "sum"),
                    Total_Predicted_No_Weather_Model =("Predicted_No_Weather_Model", "sum"),
                    Total_Disruption                 =("Disruption",                 "sum"),
                    Mean_WBGT                        =(WBGT_VAR,                     "mean"),
                    Mean_Precip_Month                =("precip_month",               "mean"),
                    Mean_Precip_5day                 =("precip_5day",                "mean"),
                    N_Facilities                     =("facility",                   "nunique"),
                ).reset_index()
                dist_agg["Deficit_Pct"] = np.where(
                    dist_agg["Total_Predicted_No_Weather_Model"] > 0,
                    100.0 * dist_agg["Total_Disruption"]
                        / dist_agg["Total_Predicted_No_Weather_Model"], np.nan)
                dist_agg["indicator"], dist_agg["ssp"], dist_agg["tier"] = ind, ssp, tier
                dist_agg.to_csv(
                    f"{OUT_DIR}projection_district_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
                    index=False)

                # ---- Monthly time series pooled across facilities ----
                mon_agg = df.groupby(["year", "month"]).agg(
                    Total_Predicted_Weather_Model    =("Predicted_Weather_Model",    "sum"),
                    Total_Predicted_No_Weather_Model =("Predicted_No_Weather_Model", "sum"),
                    Mean_WBGT                        =(WBGT_VAR,                     "mean"),
                    Mean_Precip_Month                =("precip_month",               "mean"),
                    Mean_Precip_5day                 =("precip_5day",                "mean"),
                    N_Facilities                     =("facility",                   "nunique"),
                ).reset_index()
                mon_agg["Total_Disruption"] = (
                    mon_agg["Total_Predicted_No_Weather_Model"]
                    - mon_agg["Total_Predicted_Weather_Model"])
                mon_agg["Deficit_Pct"] = np.where(
                    mon_agg["Total_Predicted_No_Weather_Model"] > 0,
                    100.0 * mon_agg["Total_Disruption"]
                        / mon_agg["Total_Predicted_No_Weather_Model"], np.nan)
                mon_agg["Year_Month"] = (mon_agg["year"].astype(str)
                                         + "-" + mon_agg["month"].astype(str))
                mon_agg["indicator"], mon_agg["ssp"], mon_agg["tier"] = ind, ssp, tier
                mon_agg.to_csv(
                    f"{OUT_DIR}projection_monthly_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv",
                    index=False)

                # ---- Summary row ----
                tot_a = float(mu_a.sum())
                tot_b = float(mu_b.sum())
                deficit_proj = ((tot_b - tot_a) / tot_b * 100
                                if tot_b > 0 else np.nan)
                all_proj_summary.append({
                    "indicator":         ind,
                    "ssp":               ssp,
                    "tier":              tier,
                    "period_start":      PROJ_PERIOD_START,
                    "period_end":        PROJ_PERIOD_END,
                    "n_facility_months": len(df),
                    "n_input_dropped":   n_dropped,
                    "n_pred_nan":        n_pred_nan,
                    "mean_wbgt":         float(df[WBGT_VAR].mean()),
                    "mean_precip_month": float(df["precip_month"].mean()),
                    "mean_precip_5day":  float(df["precip_5day"].mean()),
                    "total_A_projected": tot_a,
                    "total_B_projected": tot_b,
                    "deficit_pct":       deficit_proj,
                })
                print(f"    {ind}: deficit_proj={deficit_proj:+.2f}% "
                      f"(n={len(df):,}, dropped {n_dropped:,} inputs, "
                      f"{n_pred_nan:,} pred NaN)")

    if all_proj_summary:
        pd.DataFrame(all_proj_summary).to_csv(
            f"{OUT_DIR}projection_summary_{WBGT_VAR}.csv", index=False)
        print(f"\nProjection summary -> {OUT_DIR}projection_summary_{WBGT_VAR}.csv")
    else:
        print("\nNo projections produced - check WBGT_PROJ_FILE_TPL / "
              "PRECIP_*_PROJ_FILE_TPL against your actual filenames.")

    # -----------------------------------------------------------------------
    # TLO LOOKUP TABLE
    # -----------------------------------------------------------------------
    print("\nTLO lookup tables...")
    tlo_rows = []

    for res in all_results:
        ind         = res["indicator"]
        design_info = res["_design_info"]
        wbgt_shift  = res["_wbgt_shift"]
        spline_cols = res["_spline_cols"]
        coefs_a     = nb_coefficient_lookup(res["_model_a"])
        spline_beta = np.array(
            [coefs_a.get(c, 0.0) for c in spline_cols], dtype=float)

        ref_c     = results_df["reference_wbgt"].iloc[0] - wbgt_shift
        all_pts_c = np.append(WBGT_GRID - wbgt_shift, ref_c)
        basis     = np.asarray(
            patsy.build_design_matrices(
                [design_info], {"x": all_pts_c})[0], dtype=float)
        ref_row = basis[-1]

        for i, w in enumerate(WBGT_GRID):
            eta = float((basis[i] - ref_row) @ spline_beta)
            irr = np.exp(eta)
            tlo_rows.append({
                "indicator":              ind,
                "label":                  INDICATOR_LABELS.get(ind, ind),
                "wbgt":                   w,
                "irr":                    irr,
                "disruption_probability": max(0.0, 1.0 - irr),
                "demand_multiplier":      max(1.0, irr),
            })
        print(f"  {ind}: {len(WBGT_GRID)} WBGT values")


    print(f"\nAll outputs in {OUT_DIR}")
    print("Done.")
