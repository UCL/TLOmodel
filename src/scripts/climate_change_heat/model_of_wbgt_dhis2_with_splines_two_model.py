"""
loop_all_indicators_two_model_NB.py

Two-model approach for WBGT–health-service disruption:
  Model A (exposure):       y ~ cr(WBGT, df) + WBGT_lags + covid + year | facility + month
  Model B (counterfactual): y ~                             covid + year | facility + month

Negative Binomial FE (fixest::fenegbin via rpy2) with district-clustered SEs.
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

WBGT_VAR      = "wbgt5x_day"
SPLINE_DF     = 4
LAG_MONTHS    = [1, 2, 3, 4]
CENTER        = True
MIN_OBS       = int(0.3 * 12 * 12)

min_year_historical = 2016
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

N_BOOTSTRAP     = 0
BOOT_SEED       = 42
BOOT_CI_LEVEL   = 0.95
BOOT_MIN_SUCCESS = 0.80
N_JOBS          = 1

FDR_ALPHA = 0.05

REFERENCE_WBGT = 28.0
WBGT_GRID      = np.arange(20.0, 37.0, 0.5)

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- FIX 1: Define PROJECTION_DIR (was missing → NameError) ----
THERMOFEEL_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices")
PROJECTION_DIR = str(THERMOFEEL_DIR)

PANEL_DIST_COL_IN_PANEL = "Dist"


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
            df.loc[mask, y_col] = np.nan
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
    """Fit Model A at each candidate df; return AIC table and best df."""
    rows = []
    for df_try in candidates:
        try:
            nb_try, splines_try, _ = add_spline_basis(nb_data.copy(), df_spline=df_try)
            rhs_try = splines_try + lag_terms + ["covid", "year_c"]
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


def save_exposure_response_plot(curve_df: pd.DataFrame, indicator: str):
    label = INDICATOR_LABELS.get(indicator, indicator)
    x_ref = float(curve_df["wbgt_ref"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    if curve_df["rr_lo"].notna().any():
        ax.fill_between(
            curve_df["wbgt"], curve_df["rr_lo"], curve_df["rr_hi"],
            color="#2f5d80", alpha=0.2, linewidth=0,
        )

    ax.plot(curve_df["wbgt"], curve_df["rr_vs_ref"], color="#2f5d80", lw=2)
    ax.axhline(1.0, color="black", ls="--", lw=0.9)
    ax.axvline(x_ref, color="#888888", ls=":", lw=1.0)
    ax.set_xlabel("WBGT (°C)")
    ax.set_ylabel("Relative rate vs reference WBGT")
    ax.set_title(
        f"Exposure-response curve: {label}\n"
        f"Contemporaneous WBGT spline (reference = {x_ref:.2f}°C)",
        fontsize=11, fontweight="bold")
    ax.grid(axis="both", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}exposure_response_curve_{indicator}_{WBGT_VAR}.png",
        dpi=180, bbox_inches="tight")
    plt.close()


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
                    rhs_a, rhs_b, fe_spec, fe_cols, cluster_col):
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
        return pct, None
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

    nb_cols = (["y", "facility",  "month","year_c", "wbgt_c", "covid", CLUSTER_COL]
               + lag_terms)
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

    # ------------------------------------------------------------------
    # Fit Model A and Model B at chosen df
    # ------------------------------------------------------------------
    nb_data, spline_cols, design_info = add_spline_basis(nb_data, chosen_df)
    nb_data = nb_data.reset_index(drop=True)

    rhs_a = spline_cols + lag_terms + ["covid", "year_c"]
    rhs_b = ["covid", "year_c"]

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

        hot = nb_data[WBGT_VAR].values > REFERENCE_WBGT
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

    # ------------------------------------------------------------------
    # Exposure-response curve
    # ---- FIX 4: pass names_a and vcov_a so CIs render ----
    # ------------------------------------------------------------------
    try:
        curve_df = make_exposure_response_curve(
            model_a, spline_cols, wbgt_shift,
            nb_data[WBGT_VAR], indicator, design_info,
            names_a=names_a, vcov_a=vcov_a,
        )
        curve_df.to_csv(
            f"{OUT_DIR}exposure_response_curve_{indicator}_{WBGT_VAR}.csv", index=False)
        save_exposure_response_plot(curve_df, indicator)
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
        out   = [
            _boot_replicate(s, nb_data, dist_index, dist_ids,
                            rhs_a, rhs_b, fe_spec, fe_cols, CLUSTER_COL)
            for s in seeds
        ]
        boot_pcts = [v for v, err in out if err is None]
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
        deficit["ci_lo"]     = float(np.percentile(boot_arr, 100 * alpha / 2))
        deficit["ci_hi"]     = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
        deficit["n_boot_ok"] = n_ok
        frac_le = float(np.mean(boot_arr <= 0))
        frac_ge = float(np.mean(boot_arr >= 0))
        deficit["p_boot"] = float(
            min(1.0, max(2 * min(frac_le, frac_ge), 1.0 / (n_ok + 1))))
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
    deficit["_spline_cols"] = spline_cols
    deficit["_design_info"] = design_info
    deficit["_wbgt_shift"]  = wbgt_shift

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
    ax.set_title(
        f"WBGT-associated deficit (NB FE: Model A vs B)\n"
        f"cr({WBGT_VAR}) + lags, facility+month FE, 95% delta-method CI",
        fontsize=11, fontweight="bold")
    if has_ci:
        ax.legend(handles=[
            mpatches.Patch(color="#823038", label=f"BH-FDR q≤{FDR_ALPHA}"),
            mpatches.Patch(color="#888888", label="not significant"),
        ], loc="lower right", fontsize=9, frameon=False)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"θ={r['theta_a']:.1f}" for _, r in plot_df.iterrows()],
        fontsize=7, color="#666666")
    ax2.tick_params(axis="y", length=0)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}forest_plot_two_model_deficit_NB_{WBGT_VAR}.png",
                dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Forest plot -> {OUT_DIR}forest_plot_two_model_deficit_NB_{WBGT_VAR}.png")

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

    fig, ax = plt.subplots(figsize=(7, max(4, len(ph) * 0.55 + 1.5)))
    for i, row in ph.iterrows():
        lo, hi, pt = (row["hot_deficit_ci_lo"],
                      row["hot_deficit_ci_hi"],
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
        f"% change in appointments (WBGT > {REFERENCE_WBGT}°C)", fontsize=10)
    ax.set_title(
        f"Hot-month deficit (WBGT > {REFERENCE_WBGT}°C)\n"
        f"NB FE, 95% delta-method CI, red = BH-FDR q≤{FDR_ALPHA}",
        fontsize=11, fontweight="bold")
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color="#823038", label=f"BH-FDR q≤{FDR_ALPHA}"),
        mpatches.Patch(color="#888888", label="not significant"),
    ], loc="lower right", fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}forest_plot_hot_deficit_NB_{WBGT_VAR}.png",
                dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Hot-month forest plot -> {OUT_DIR}forest_plot_hot_deficit_NB_{WBGT_VAR}.png")

    # -----------------------------------------------------------------------
    # SPLINE IRR FOREST PLOT
    # -----------------------------------------------------------------------
    IRR_LOW  = REFERENCE_WBGT
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
        f"{OUT_DIR}irr_contrast_{IRR_LOW:.0f}_{IRR_HIGH:.0f}_{WBGT_VAR}.csv", index=False)

    if not irr_df.empty:
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
                elinewidth=1.4, color=irr_colors[i], zorder=2, ecolor="black",)
        ax.axvline(1.0, color="black", ls="--", lw=0.9)
        ax.set_yticks(range(len(irr_df)))
        ax.set_yticklabels(irr_df["label"], fontsize=9)
        ax.set_xlabel(
            f"IRR: WBGT {IRR_HIGH:.0f}°C vs {IRR_LOW:.0f}°C", fontsize=10)
        ax.set_title(
            f"WBGT contrast IRR ({IRR_LOW:.0f}→{IRR_HIGH:.0f}°C)\n"
            "NB FE, 95% delta-method CI, red = CI excludes 1",
            fontsize=11, fontweight="bold")
        ax.grid(axis="x", ls=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(
            f"{OUT_DIR}forest_plot_IRR_{IRR_LOW:.0f}_{IRR_HIGH:.0f}_NB_{WBGT_VAR}.png",
            dpi=180, bbox_inches="tight")
        plt.close()
        print(f"IRR forest plot -> "
              f"{OUT_DIR}forest_plot_IRR_{IRR_LOW:.0f}_{IRR_HIGH:.0f}_NB_{WBGT_VAR}.png")
    else:
        print("  WARNING: irr_df empty — check KEY STASH DIAGNOSTIC above.")

    # -----------------------------------------------------------------------
    # EXPOSURE-RESPONSE PANEL
    # -----------------------------------------------------------------------
    wbgt_grid  = np.arange(20.0, 37.5, 0.5)
    curve_rows = []
    for res in all_results:
        if not all(k in res for k in ("_design_info", "_spline_cols",
                                      "_names_a", "_beta_a", "_vcov_a")):
            continue
        ref_wbgt = float(res["_nb_data"][WBGT_VAR].mean())
        try:
            irr, lo, hi = spline_irr_at(
                wbgt_grid,
                res["_design_info"],
                res["_spline_cols"],
                res["_names_a"],
                res["_beta_a"],
                res["_vcov_a"],
                wbgt_ref=IRR_LOW,
                wbgt_shift=res["_wbgt_shift"],
            )
            for w, i_, l_, h_ in zip(wbgt_grid, irr, lo, hi):
                curve_rows.append({
                    "indicator": res["indicator"],
                    "wbgt": w, "irr": i_,
                    "irr_lo": l_, "irr_hi": h_,
                    "ref_wbgt": ref_wbgt,
                })
        except Exception as e:
            print(f"  [{res['indicator']}] curve failed: {e}")

    # NOTE: irr_df was already built above from irr_rows; no need to rebuild here
    if curve_rows:
        curves_df = pd.DataFrame(curve_rows)
        curves_df.to_csv(
            f"{OUT_DIR}exposure_response_curves_{WBGT_VAR}.csv", index=False)

        inds   = [r["indicator"] for r in all_results
                  if r["indicator"] in curves_df["indicator"].unique()]
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
            ref_w = float(sub["ref_wbgt"].iloc[0])
            ax.fill_between(sub["wbgt"], sub["irr_lo"], sub["irr_hi"],
                            color="#4a7298", alpha=0.25, linewidth=0)
            ax.plot(sub["wbgt"], sub["irr"], color="#2a4d70", lw=1.5)
            ax.axhline(1.0, color="black", lw=0.5, ls="--")
            ax.axvline(ref_w, color="grey", lw=0.5, ls=":")
            ax.set_title(INDICATOR_LABELS.get(ind, ind),
                         fontsize=9, fontweight="bold")
            if idx % n_cols == 0:
                ax.set_ylabel("IRR", fontsize=8)
            if idx // n_cols == n_rows - 1:
                ax.set_xlabel("WBGT (°C)", fontsize=8)
            ax.tick_params(labelsize=7)

        for idx in range(n_ind, len(af)):
            af[idx].set_visible(False)

        fig.suptitle(
            "Exposure-response curves (spline Model A, IRR vs facility-mean WBGT)\n"
            "Shaded: 95% delta-method CI;  dotted: reference WBGT",
            fontsize=10, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}exposure_response_panel_{WBGT_VAR}.png",
                    dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Exposure-response panel -> {OUT_DIR}exposure_response_panel.png")
    else:
        print("  WARNING: no curve rows — check KEY STASH DIAGNOSTIC above.")

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
        bdf["hot_month"]  = nb_data[WBGT_VAR].values > REFERENCE_WBGT
        bdf.to_csv(f"{OUT_DIR}historical_burden_{ind}_{WBGT_VAR}.csv", index=False)

        hot_mask = nb_data[WBGT_VAR].values > REFERENCE_WBGT
        burden_rows.append({
            "indicator":       ind,
            "label":           INDICATOR_LABELS.get(ind, ind),
            "total_mu_a":      float(np.nansum(mu_a)),
            "total_mu_b":      float(np.nansum(mu_b)),
            "deficit_pct":     res["deficit_pct"],
            "hot_deficit_pct": float(
                100.0 * (mu_a[hot_mask].sum() - mu_b[hot_mask].sum())
                / mu_b[hot_mask].sum()
            ) if hot_mask.any() else np.nan,
        })

        # ---- District point-estimate aggregation ----
        d = bdf.groupby(CLUSTER_COL).agg(
            obs=("y_int", "sum"),
            mu_a=("mu_a", "sum"),
            mu_b=("mu_b", "sum"),
        ).reset_index()
        d["deficit_pct"] = (d["mu_a"] - d["mu_b"]) / d["mu_b"] * 100
        d["indicator"]   = ind
        all_dist.append(d)

        # ---- FIX 3: Save point-estimate and CI to SEPARATE files ----
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

        hot_mask = res["_nb_data"][WBGT_VAR].values > REFERENCE_WBGT
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

    # Monthly deficit panel — jackknife CI across facilities
    n_ind  = len(fitted)
    n_cols = 3
    n_rows = int(np.ceil(n_ind / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 3 * n_rows),
                              sharex=True)
    af = axes.flatten() if n_ind > 1 else [axes]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    for idx, res in enumerate(all_results):
        ax      = af[idx]
        ind     = res["indicator"]
        nb      = res["_nb_data"]
        mu_a_m  = res["_mu_a"]
        mu_b_m  = res["_mu_b"]
        fac_ids = nb["facility"].values
        months  = nb["month"].values

        pcts, los, his = [], [], []
        for m in range(1, 13):
            mask = months == m
            if not mask.any():
                pcts.append(0); los.append(np.nan); his.append(np.nan)
                continue
            d = monthly_deficit_with_jackknife_ci(
                mu_a_m[mask], mu_b_m[mask], fac_ids[mask])
            pcts.append(d["deficit_pct"])
            los.append(d["ci_lo"])
            his.append(d["ci_hi"])

        pcts_a = np.asarray(pcts, dtype=float)
        los_a  = np.asarray(los,  dtype=float)
        his_a  = np.asarray(his,  dtype=float)
        bar_c  = ["#823038" if p < 0 else "#2a78d6" for p in pcts_a]
        yerr   = np.array([
            np.nan_to_num(pcts_a - los_a, nan=0.0),
            np.nan_to_num(his_a  - pcts_a, nan=0.0),
        ])
        ax.bar(range(12), pcts_a, color=bar_c, alpha=0.8, yerr=yerr,
               error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": "#333"})
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_names, fontsize=6, rotation=45)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(INDICATOR_LABELS.get(ind, ind),
                     fontsize=9, fontweight="bold")
        if idx % n_cols == 0:
            ax.set_ylabel("% deficit", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n_ind, len(af)):
        af[idx].set_visible(False)
    fig.suptitle(
        "Two-model deficit by calendar month "
        "(95% jackknife CI across facilities)\n"
        "(mu_B − mu_A)/mu_B × 100;  red = heat reduced services",
        fontsize=10, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}deficit_by_month_{WBGT_VAR}.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Time-series panel
    nc = 3
    nr = int(np.ceil(n_ind / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5.5 * nc, 3.5 * nr), sharex=True)
    af = axes.flatten() if n_ind > 1 else [axes]

    for idx, ind in enumerate(fitted):
        ax  = af[idx]
        csv = f"{OUT_DIR}historical_burden_{ind}.csv"
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv, parse_dates=["date"])
        m  = (df.groupby("date")
              .agg(obs=("y_int","sum"), mu_a=("mu_a","sum"), mu_b=("mu_b","sum"))
              .sort_index())
        ax.plot(m.index, m.mu_b, color="#2a78d6", lw=1.0, ls="--",
                alpha=0.8, label="Model B (no weather)")
        ax.plot(m.index, m.obs,  color="#333",    lw=1.0, label="Observed")
        ax.fill_between(m.index, m.mu_a, m.mu_b,
                        where=m.mu_a < m.mu_b,
                        color="#823038", alpha=0.25, label="Heat deficit")
        ax.set_title(INDICATOR_LABELS.get(ind, ind),
                     fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6)

    for idx in range(n_ind, len(af)):
        af[idx].set_visible(False)
    fig.suptitle(
        "Observed vs Model B counterfactual\nShaded = two-model heat deficit",
        fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}timeseries_burden_{WBGT_VAR}.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {OUT_DIR}historical_burden_summary.csv")

    # -----------------------------------------------------------------------
    # DISTRICT CHOROPLETH MAPS
    # -----------------------------------------------------------------------
    try:
        import geopandas as gpd
        _have_gpd = True
    except ImportError:
        print("geopandas not installed — skipping maps.")
        _have_gpd = False

    if _have_gpd and os.path.exists(SHAPEFILE_PATH) and all_dist:
        print("\nDistrict choropleth maps...")
        try:
            shp = gpd.read_file(SHAPEFILE_PATH)
            shp[DISTRICT_NAME_COL] = (
                shp[DISTRICT_NAME_COL].astype(str).str.strip().str.title())
        except Exception as e:
            print(f"  Shapefile load failed: {e}")
            shp = None

        if shp is not None:
            dist_all = pd.concat(all_dist, ignore_index=True)
            dist_all[CLUSTER_COL] = (
                dist_all[CLUSTER_COL].astype(str).str.strip().str.title())

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
                ax.set_title(
                    f"{INDICATOR_LABELS.get(ind, ind)}\n"
                    "District-level heat deficit (%)",
                    fontsize=11, fontweight="bold")
                plt.tight_layout()
                plt.savefig(f"{OUT_DIR}map_district_deficit_{ind}_{WBGT_VAR}.png",
                            dpi=180, bbox_inches="tight")
                plt.close()
                print(f"  -> {OUT_DIR}map_district_deficit_{ind}_{WBGT_VAR}.png")

            # Summary panel
            global_vmax = max(
                dist_all["deficit_pct"].abs().quantile(0.95), 1.0)
            n_ind  = len(fitted)
            n_cols = 3
            n_rows = int(np.ceil(n_ind / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols,
                                      figsize=(5 * n_cols, 6 * n_rows))
            af = axes.flatten() if n_ind > 1 else [axes]

            for idx, ind in enumerate(fitted):
                ax  = af[idx]
                sub = dist_all[dist_all["indicator"] == ind].copy()
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
                ax.set_title(INDICATOR_LABELS.get(ind, ind),
                             fontsize=8, fontweight="bold")

            for idx in range(n_ind, len(af)):
                af[idx].set_visible(False)

            sm = plt.cm.ScalarMappable(
                cmap="RdBu",
                norm=plt.Normalize(vmin=-global_vmax, vmax=global_vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=af, orientation="horizontal",
                                fraction=0.02, pad=0.02, shrink=0.6)
            cbar.set_label("% deficit (Model A vs B)", fontsize=9)
            fig.suptitle(
                "District-level heat deficit — all indicators\n"
                "Red = heat reduced, Blue = heat increased",
                fontsize=11, fontweight="bold", y=1.01)
            plt.savefig(f"{OUT_DIR}map_district_deficit_panel_{WBGT_VAR}.png",
                        dpi=180, bbox_inches="tight")
            plt.close()
            print(f"  -> {OUT_DIR}map_district_deficit_panel_{WBGT_VAR}.png")

    elif not os.path.exists(SHAPEFILE_PATH):
        print(f"\nShapefile not found at {SHAPEFILE_PATH} — skipping maps.")

    # -----------------------------------------------------------------------
    # FORWARD PROJECTIONS
    # -----------------------------------------------------------------------
    print("\nForward projections...")
    print("\nForward projections...")

    all_proj = []

    for res in all_results:
        ind = res["indicator"]
        nb_data = res["_nb_data"]
        mu_b = res["_mu_b"]
        spline_cols = res["_spline_cols"]
        design_info = res["_design_info"]
        wbgt_shift = res["_wbgt_shift"]
        coefs_a = nb_coefficient_lookup(res["_model_a"])
        spline_beta = np.array([coefs_a.get(c, 0.0) for c in spline_cols], dtype=float)

        # historical two-model deficit (for reference / delta)
        sum_a_hist = float(res["_mu_a"].sum())
        sum_b_hist = float(mu_b.sum())
        deficit_hist = (sum_a_hist - sum_b_hist) / sum_b_hist * 100 if sum_b_hist > 0 else np.nan

        for ssp in SSP_SCENARIOS:
            for tier in MODEL_TIERS:
                pf_path = f"{PROJECTION_DIR}/{ssp}/{tier}.csv"
                if not os.path.exists(pf_path):
                    print(f"  SKIP {ind} {ssp}/{tier}: file not found")
                    continue
                proj = pd.read_csv(pf_path)
                proj["facility"] = proj["facility"].astype(str)
                if WBGT_VAR not in proj.columns:
                    continue
                proj = proj.rename(columns={WBGT_VAR: "wbgt_proj"})
                if proj.empty:
                    continue

                # --- Model A projection: scale mu_a_hist by IRR from WBGT change ---
                n_rows_p = len(proj)
                all_pts_c = np.concatenate(
                    [
                        proj["wbgt_hist"].values - wbgt_shift,
                        proj["wbgt_proj"].values - wbgt_shift,
                    ]
                )
                basis_all = np.asarray(patsy.build_design_matrices([design_info], {"x": all_pts_c})[0], dtype=float)
                basis_h = basis_all[:n_rows_p]
                basis_p = basis_all[n_rows_p:]
                irrs = np.exp((basis_p - basis_h) @ spline_beta)

                proj["mu_a_proj"] = proj["mu_a_hist"] * irrs  # Model A scaled
                proj["mu_b_proj"] = proj["mu_b_hist"]  # Model B unchanged

                # --- two-model deficit under projected WBGT ---
                sum_a_proj = float(proj["mu_a_proj"].sum())
                sum_b_proj = float(proj["mu_b_proj"].sum())
                deficit_proj = (sum_a_proj - sum_b_proj) / sum_b_proj * 100 if sum_b_proj > 0 else np.nan
                delta_deficit = deficit_proj - deficit_hist
                wd = float((proj["wbgt_proj"] - proj["wbgt_hist"]).mean())

                print(
                    f"  {ind} {ssp}/{tier}: dWBGT={wd:+.2f}  "
                    f"deficit_hist={deficit_hist:+.2f}%  "
                    f"deficit_proj={deficit_proj:+.2f}%  "
                    f"Δdeficit={delta_deficit:+.2f}%"
                )

                proj["indicator"] = ind
                proj["ssp"] = ssp
                proj["tier"] = tier
                proj["deficit_hist"] = deficit_hist
                proj["deficit_proj"] = deficit_proj
                proj["delta_deficit"] = delta_deficit
                proj.to_csv(f"{OUT_DIR}projection_{ind}_{ssp}_{tier}_{WBGT_VAR}.csv", index=False)
                all_proj.append(
                    {
                        "indicator": ind,
                        "ssp": ssp,
                        "tier": tier,
                        "mean_wbgt_diff": wd,
                        "deficit_hist": deficit_hist,
                        "deficit_proj": deficit_proj,
                        "delta_deficit": delta_deficit,
                    }
                )

    if all_proj:
        pd.DataFrame(all_proj).to_csv(
            f"{OUT_DIR}projection_summary_{WBGT_VAR}.csv", index=False)
        for ind in fitted:
            ind_proj = [p for p in all_proj if p["indicator"] == ind]
            if not ind_proj:
                continue
            grid = pd.DataFrame(
                index=SSP_SCENARIOS, columns=MODEL_TIERS, dtype=float)
            for p in ind_proj:
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
            ax.set_title(
                f"{INDICATOR_LABELS.get(ind, ind)}\n"
                "Change in two-model deficit under CMIP6",
                fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}projection_heatmap_{ind}_{WBGT_VAR}.png",
                        dpi=180, bbox_inches="tight")
            plt.close()
        print(f"  -> {OUT_DIR}projection_summary.csv")

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

        ref_c     = REFERENCE_WBGT - wbgt_shift
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

    if tlo_rows:
        tlo = pd.DataFrame(tlo_rows)
        tlo.to_csv(f"{OUT_DIR}tlo_wbgt_lookup.csv", index=False)
        for col, nm in [("disruption_probability", "disruption"),
                        ("demand_multiplier",       "demand")]:
            tlo.pivot(index="wbgt", columns="indicator",
                      values=col).to_csv(f"{OUT_DIR}tlo_{nm}_wide_{WBGT_VAR}.csv")

        fig, ax = plt.subplots(figsize=(8, 5))
        for ind in tlo["indicator"].unique():
            sub = tlo[tlo["indicator"] == ind]
            ax.plot(sub.wbgt, sub.disruption_probability, lw=1.3,
                    label=INDICATOR_LABELS.get(ind, ind))
        ax.set_xlabel("WBGT (°C)")
        ax.set_ylabel(f"Disruption probability (vs {REFERENCE_WBGT}°C)")
        ax.set_title("Heat disruption for TLO model (NB FE)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(ls=":", alpha=0.4)
        ax.set_ylim(bottom=-0.01)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}tlo_disruption_curves_{WBGT_VAR}.png",
                    dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  -> {OUT_DIR}tlo_wbgt_lookup_{WBGT_VAR}.csv")

    print(f"\nAll outputs in {OUT_DIR}")
    print("Done.")
