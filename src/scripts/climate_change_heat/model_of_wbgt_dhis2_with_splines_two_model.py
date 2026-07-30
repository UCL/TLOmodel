"""
loop_all_indicators_two_model_NB.py

Two-model approach for WBGT–health-service disruption:
  Model A (exposure):      y ~ cr(WBGT, df) + WBGT_lags + covid + year | facility + month
  Model B (counterfactual): y ~                            covid + year | facility + month

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
from joblib import Parallel, delayed

os.environ.setdefault("R_HOME", "/Library/Frameworks/R.framework/Resources")

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

base = importr("base")
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
SHAPEFILE_PATH    =  "/Users/rachelmurray-watson/Documents/Malawi_shapefiles/mwi_admbnda_adm2.shp"
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
    "fp_total_clients": "FP Total Clients",
    "opd_attendance": "OPD Attendance",
    "ipd_total_admissions": "IPD Total Admissions",
    "vmmc_first_visits": "VMMC First Visits",
    "pnc_mother_checked_48h": "PNC Mother <48h",
    "anc_new_attendees": "ANC New Attendees",
    "anc_first_trimester_starts": "ANC 1st Trimester Starts",
    "bcg_under1": "BCG Under-1",
    "penta3_under1": "Penta3 Under-1",
    "measles1_under1": "Measles 1st Dose Under-1",
    "fully_immunised_under1": "Fully Immunised Under-1",
    "pnc_within_2wks": "PNC Within 2 Weeks",
    "pnc_first_visit_2wks": "PNC First Visit <2 Weeks",
    "live_births_total": "Live Births Total",
    "skilled_deliveries": "Skilled Deliveries",
}

WBGT_VAR = "wbgt5x_day"
SPLINE_DF = 3
LAG_MONTHS = [1, 2, 3, 9]
CENTER = True
MIN_OBS = int(0.5 * 12 * 12)   # 72

min_year_historical = 2015
max_year_historical = 2025
apply_cap = False

N_CURVE_POINTS = 200
CURVE_REF = "mean"

COVID_START = "2020-04-01"
COVID_END = "2021-12-01"

CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre", "2023-03-01", "2024-03-01"),
]

CLUSTER_COL = "facility"

N_BOOTSTRAP = 0
BOOT_SEED = 42
BOOT_CI_LEVEL = 0.95
BOOT_MIN_SUCCESS = 0.80
N_JOBS = 1

FDR_ALPHA = 0.05

# New constants for added sections
REFERENCE_WBGT = 25.0
WBGT_GRID      = np.arange(20.0, 37.0, 0.5)

PROJECTION_DIR = (
    "/Users/rachelmurray-watson/Documents/Heat_data"
    "/CMIP6_facility_projections"
)
SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
MODEL_TIERS   = ["lowest", "median", "highest"]

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)
PANEL_DIST_COL_IN_PANEL = "Dist"


def _norm_sf(x):
    """P(Z > x); avoids scipy so we don't clash with R's BLAS on macOS."""
    return 0.5 * math.erfc(x / math.sqrt(2))
# ---------------------------------------------------------------------------
# R helpers  — identical to working script
# ---------------------------------------------------------------------------
def fit_nb_fixest(
    df: pd.DataFrame,
    rhs_terms: list[str],
    fe_terms: list[str],
    cluster_col: str,
    y_col: str = "y_int",
):
    print(f"    [fit_nb] entering, n={len(df)}, rhs={len(rhs_terms)}", flush=True)

    rhs = " + ".join(rhs_terms) if rhs_terms else "1"
    fe = " + ".join(fe_terms)
    fml = ro.Formula(f"{y_col} ~ {rhs} | {fe}")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    r_df = base.as_data_frame(r_df)

    for col in set(fe_terms + [cluster_col]):
        r_df.rx2[col] = base.as_factor(r_df.rx2(col))

    suppressWarnings = ro.r("suppressWarnings")
    r_model = suppressWarnings(
        fixest.fenegbin(
            fml=fml,
            data=r_df,
            cluster=ro.StrVector(['facility']),
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
        raise ValueError(
            f"Unexpected coeftable shape: {tab.shape}."
        )
    return tab


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

def two_model_deficit_analytical(mu_a, mu_b, names_a, vcov_a,
                                  names_b, vcov_b, X_data):
    """
    δ = 100 × (Σμ_A − Σμ_B) / Σμ_B.
    Delta-method SE. Treats A and B as independent (approximation:
    they share y and rows, so this understates uncertainty a bit).
    For NB2 with log-link: ∂μ_i/∂β_j = μ_i · X_ij, so
    ∂S/∂β_j = Σᵢ μ_i · X_ij over non-absorbed coefficients only.
    """
    mu_a = np.where(np.isnan(mu_a), 0.0, mu_a)
    mu_b = np.where(np.isnan(mu_b), 0.0, mu_b)
    sum_a, sum_b = float(mu_a.sum()), float(mu_b.sum())
    if sum_b <= 0:
        return dict(deficit_pct=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    se_pct=np.nan, p_analytical=np.nan)
    deficit_pct = 100.0 * (sum_a - sum_b) / sum_b

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
    var_a = float(g_a @ vcov_a_clean @ g_a)
    var_b = float(g_b @ vcov_b_clean @ g_b)
    var = var_a + var_b
    se_pct = float(np.sqrt(max(var, 0.0)))
    z = deficit_pct / se_pct if se_pct > 0 else np.nan
    p = float(2 * _norm_sf(abs(z))) if np.isfinite(z) else np.nan
    return dict(deficit_pct=deficit_pct,
                ci_lo=deficit_pct - 1.96 * se_pct,
                ci_hi=deficit_pct + 1.96 * se_pct,
                se_pct=se_pct, p_analytical=p)

def nb_coefficient_lookup(r_model) -> dict[str, float]:
    tab = nb_coef_table(r_model)
    return dict(zip(tab["term"].astype(str), tab["estimate"].astype(float)))


# ---------------------------------------------------------------------------
# Data helpers  — identical to working script
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
            f"[{indicator}] Cluster column '{CLUSTER_COL}' not present in "
            f"{panel_path}. Available: {sorted(long.columns)}"
        )

    long["covid"] = long["date"].between(COVID_START, COVID_END).astype(int)

    for fac, start, end in CLOSURES:
        mask = (long["date"].between(start, end)) & (long["facility"] == fac)
        n_masked = int(mask.sum())
        long.loc[mask, "y"] = np.nan
        if n_masked:
            print(f"  [{indicator}] Masked {n_masked} closure months for {fac}.")

    long["year"] = long["date"].dt.year
    long["month"] = long["date"].dt.month
    long = long[long["year"].between(min_year_historical, max_year_historical - 1)]

    if apply_cap:
        long.loc[long["y"] > 4e3, "y"] = np.nan

    obs_per_fac = long.dropna(subset=["y", WBGT_VAR]).groupby("facility").size()
    keep_facs = obs_per_fac[obs_per_fac >= MIN_OBS].index
    long = long[long["facility"].isin(keep_facs)].copy()

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
        raise ValueError(f"[{indicator}] {n_bad} rows have missing '{CLUSTER_COL}'.")

    return long


def enforce_complete_monthly_grid(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    df = df.sort_values(["facility", "date"]).reset_index(drop=True)
    n_before = len(df)
    frames = []
    for fac, g in df.groupby("facility", sort=False):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="MS")
        g = g.set_index("date").reindex(full)
        g.index.name = "date"
        g["facility"] = fac
        frames.append(g.reset_index())
    out = pd.concat(frames, ignore_index=True)

    n_inserted = len(out) - n_before
    if n_inserted:
        print(
            f"  [{indicator}] Inserted {n_inserted:,} placeholder rows to close "
            f"calendar gaps ({100 * n_inserted / len(out):.2f}% of grid)."
        )

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["covid"] = out["date"].between(COVID_START, COVID_END).astype(int)
    out[CLUSTER_COL] = out.groupby("facility")[CLUSTER_COL].transform(
        lambda s: s.ffill().bfill()
    )

    diffs = out.groupby("facility")["date"].diff().dropna()
    bad_spacing = diffs[~((diffs.dt.days >= 28) & (diffs.dt.days <= 31))]
    if len(bad_spacing):
        raise ValueError(
            f"[{indicator}] Non-monthly spacing remains after reindexing "
            f"({len(bad_spacing)} rows)."
        )
    return out


def add_columns(
    df: pd.DataFrame, indicator: str
) -> tuple[pd.DataFrame, list[str], float, float]:
    df = enforce_complete_monthly_grid(df, indicator)

    year_shift = df["year"].mean() if CENTER else 0.0
    wbgt_shift = df[WBGT_VAR].mean() if CENTER else 0.0

    df["year_c"] = df["year"] - year_shift
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


def drop_separated_facilities(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    all_zero = df.groupby("facility")["y_int"].max() == 0
    sep_facs = all_zero[all_zero].index
    if len(sep_facs):
        n_rows = int(df["facility"].isin(sep_facs).sum())
        print(
            f"  [{indicator}] SEPARATION: {len(sep_facs)} facilities are all-zero "
            f"({n_rows:,} rows) — dropped."
        )
        df = df[~df["facility"].isin(sep_facs)].copy()
    return df


# ---------------------------------------------------------------------------
# Exposure-response curve  — identical to working script
# ---------------------------------------------------------------------------
def make_exposure_response_curve(
    r_model,
    spline_cols: list[str],
    wbgt_shift: float,
    observed_wbgt: pd.Series,
    indicator: str,
    design_info,
) -> pd.DataFrame:
    coefs = nb_coefficient_lookup(r_model)

    x_min = float(observed_wbgt.min())
    x_max = float(observed_wbgt.max())
    x_grid = np.linspace(x_min, x_max, N_CURVE_POINTS)
    x_grid_c = x_grid - wbgt_shift

    basis_grid = np.asarray(
        patsy.build_design_matrices([design_info], {"x": x_grid_c})[0],
        dtype=float,
    )

    if CURVE_REF == "mean":
        x_ref = float(observed_wbgt.mean())
    elif CURVE_REF == "median":
        x_ref = float(observed_wbgt.median())
    elif CURVE_REF == "min":
        x_ref = float(observed_wbgt.min())
    else:
        raise ValueError(f"Unknown CURVE_REF='{CURVE_REF}'.")

    x_ref_c = x_ref - wbgt_shift
    basis_ref = np.asarray(
        patsy.build_design_matrices([design_info], {"x": np.array([x_ref_c])})[0],
        dtype=float,
    )
    ref_row = basis_ref[0]

    beta = np.array([coefs.get(col, 0.0) for col in spline_cols], dtype=float)
    eta_grid = basis_grid @ beta
    eta_ref = float(ref_row @ beta)
    rr = np.exp(eta_grid - eta_ref)
    pct_change = 100.0 * (rr - 1.0)

    return pd.DataFrame(
        {
            "indicator": indicator,
            "label": INDICATOR_LABELS.get(indicator, indicator),
            "wbgt": x_grid,
            "wbgt_c": x_grid_c,
            "rr_vs_ref": rr,
            "pct_change_vs_ref": pct_change,
            "wbgt_ref": x_ref,
            "curve_ref": CURVE_REF,
            "note": "Contemporaneous spline component only; lagged WBGT held fixed.",
        }
    )


def save_exposure_response_plot(curve_df: pd.DataFrame, indicator: str):
    label = INDICATOR_LABELS.get(indicator, indicator)
    x_ref = float(curve_df["wbgt_ref"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(curve_df["wbgt"], curve_df["rr_vs_ref"], color="#2f5d80", linewidth=2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(x_ref, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xlabel("WBGT")
    ax.set_ylabel("Relative rate vs reference WBGT")
    ax.set_title(
        f"Exposure-response curve: {label}\n"
        f"Contemporaneous WBGT spline (reference = {x_ref:.2f})",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}exposure_response_curve_{indicator}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------------
# BH-FDR  — identical to working script
# ---------------------------------------------------------------------------
def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Bootstrap replicate  — identical to working script
# ---------------------------------------------------------------------------
def _boot_replicate(
    seed_seq,
    nb_data,
    dist_index,
    dist_ids,
    rhs_a,
    rhs_b,
    fe_terms,
    cluster_col,
):
    ro.conversion.converter_ctx.set(ro.default_converter)

    rng = np.random.default_rng(seed_seq)
    picks = rng.choice(len(dist_ids), size=len(dist_ids), replace=True)

    idx_parts = [dist_index[dist_ids[p]] for p in picks]
    idx = np.concatenate(idx_parts)
    tags = np.repeat(
        np.arange(len(picks)), [len(part) for part in idx_parts]
    ).astype(str)

    boot_df = nb_data.take(idx).reset_index(drop=True)
    tags_s = pd.Series(tags, index=boot_df.index).astype(str)
    boot_df["facility"] = boot_df["facility"].astype(str) + "__b" + tags_s
    boot_df[cluster_col] = boot_df[cluster_col].astype(str) + "__b" + tags_s

    fac_max = boot_df.groupby("facility")["y_int"].max()
    sep_facs = fac_max[fac_max == 0].index
    if len(sep_facs):
        boot_df = boot_df[~boot_df["facility"].isin(sep_facs)].copy()

    if boot_df["facility"].nunique() < 2 or boot_df[cluster_col].nunique() < 2:
        return None, "too_few_groups"

    try:
        _, mu_a = fit_nb_fixest(boot_df, rhs_a, fe_terms, cluster_col)
        _, mu_b = fit_nb_fixest(boot_df, rhs_b, fe_terms, cluster_col)
        total_a, total_b = float(mu_a.sum()), float(mu_b.sum())
        if total_b == 0:
            return None, "zero_counterfactual"
        pct = 100.0 * (total_a - total_b) / total_b
        if not np.isfinite(pct):
            return None, "non-finite"
        return pct, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:80]}"


# ---------------------------------------------------------------------------
# Per-indicator runner  — identical to working script except _nb_data etc
#                         stored for post-processing sections
# ---------------------------------------------------------------------------
def run_indicator(indicator: str) -> dict | None:

    long = prepare_data(indicator)
    if long is None:
        return None

    long, lag_terms, year_shift, wbgt_shift = add_columns(long, indicator)

    nb_cols = ["y", "facility", "year_c", "wbgt_c", "covid", CLUSTER_COL] + lag_terms
    nb_data = long.dropna(subset=nb_cols).copy()
    nb_data["y_int"] = nb_data["y"].round().clip(lower=0).astype(int)

    obs_nb = nb_data.groupby("facility").size()
    nb_data = nb_data[nb_data["facility"].isin(obs_nb[obs_nb >= MIN_OBS].index)].copy()

    nb_data = drop_separated_facilities(nb_data, indicator)

    if nb_data.empty or nb_data["facility"].nunique() < 2:
        print(f"  [{indicator}] Sample too small — skipping.")
        return None
    if nb_data[CLUSTER_COL].nunique() < 2:
        raise ValueError(
            f"[{indicator}] Only {nb_data[CLUSTER_COL].nunique()} cluster(s) — "
            f"cluster-robust SEs are not defined."
        )

    nb_data, spline_cols, design_info = add_spline_basis(nb_data, SPLINE_DF)
    nb_data = nb_data.reset_index(drop=True)

    fe_terms = ["facility", "month"]
    rhs_a = spline_cols + lag_terms + ["covid", "year_c"]
    rhs_b = ["covid", "year_c"]

    try:
        model_a, mu_a = fit_nb_fixest(nb_data, rhs_a, fe_terms, CLUSTER_COL)
        model_b, mu_b = fit_nb_fixest(nb_data, rhs_b, fe_terms, CLUSTER_COL)
    except Exception as e:
        print(f"  [{indicator}] fenegbin failed: {type(e).__name__}: {e} — skipping.")
        return None

    total_a = float(mu_a.sum())
    total_b = float(mu_b.sum())
    total_y = float(nb_data["y_int"].sum())

    print(
        f"  [{indicator}] sum(y)={total_y:,.0f}, "
        f"sum(mu_A)={total_a:,.0f}, sum(mu_B)={total_b:,.0f}"
    )

    deficit = {
        "total_pred_exposure": total_a,
        "total_pred_counterfactual": total_b,
        "deficit_abs": total_a - total_b,
        "deficit_pct": 100.0 * (total_a - total_b) / total_b,
        "theta_a": nb_theta(model_a),
        "theta_b": nb_theta(model_b),
    }

    try:
        names_a, _, vcov_a = get_beta_vcov(model_a)
        names_b, _, vcov_b = get_beta_vcov(model_b)
        delta = two_model_deficit_analytical(mu_a, mu_b, names_a, vcov_a, names_b, vcov_b, nb_data)
        deficit["ci_lo"] = delta["ci_lo"]
        deficit["ci_hi"] = delta["ci_hi"]
        deficit["deficit_se"] = delta["se_pct"]
        deficit["p_analytical"] = delta["p_analytical"]
        deficit["p_boot"] = delta["p_analytical"]
        deficit["n_boot_ok"] = 0

        hot = nb_data[WBGT_VAR].values > REFERENCE_WBGT
        if hot.any():
            dh = two_model_deficit_analytical(
                mu_a[hot], mu_b[hot], names_a, vcov_a, names_b, vcov_b, nb_data.loc[hot].reset_index(drop=True)
            )
            deficit["hot_deficit_pct"] = dh["deficit_pct"]
            deficit["hot_deficit_ci_lo"] = dh["ci_lo"]
            deficit["hot_deficit_ci_hi"] = dh["ci_hi"]
            deficit["p_hot_analytical"] = dh["p_analytical"]
        else:
            for k in ("hot_deficit_pct", "hot_deficit_ci_lo", "hot_deficit_ci_hi", "p_hot_analytical"):
                deficit[k] = np.nan

        deficit["_names_a"], deficit["_vcov_a"] = names_a, vcov_a
        deficit["_names_b"], deficit["_vcov_b"] = names_b, vcov_b
    except Exception as e:
        print(f"  [{indicator}] Analytical CI failed: {type(e).__name__}: {e}")
        for k in (
            "ci_lo",
            "ci_hi",
            "deficit_se",
            "p_analytical",
            "hot_deficit_pct",
            "hot_deficit_ci_lo",
            "hot_deficit_ci_hi",
            "p_hot_analytical",
        ):
            deficit[k] = np.nan
        deficit["p_boot"] = np.nan
        deficit["n_boot_ok"] = 0
    try:
        curve_df = make_exposure_response_curve(
            r_model=model_a,
            spline_cols=spline_cols,
            wbgt_shift=wbgt_shift,
            observed_wbgt=nb_data[WBGT_VAR],
            indicator=indicator,
            design_info=design_info,
        )
        curve_df.to_csv(
            f"{OUT_DIR}exposure_response_curve_{indicator}.csv", index=False
        )
        save_exposure_response_plot(curve_df, indicator)
    except Exception as e:
        print(f"  [{indicator}] Curve export failed: {type(e).__name__}: {e}")

    preds = nb_data[["facility", "date"]].copy()
    preds["y_obs"] = nb_data["y_int"].values
    preds["y_pred_wx"] = mu_a
    preds["y_pred_base"] = mu_b
    preds["difference"] = preds["y_pred_base"] - preds["y_pred_wx"]
    if PANEL_DIST_COL_IN_PANEL in nb_data.columns:
        preds["Dist"] = nb_data[PANEL_DIST_COL_IN_PANEL].values
    preds.to_csv(f"{OUT_DIR}two_model_predictions_{indicator}.csv", index=False)

    try:
        coef_tab = nb_coef_table(model_a)
        coef_tab["indicator"] = indicator
        coef_tab.to_csv(
            f"{OUT_DIR}coef_table_model_a_{indicator}.csv", index=False
        )
    except Exception as e:
        print(f"  [{indicator}] Coef table export failed: {e}")

    failures: Counter = Counter()

    if N_BOOTSTRAP > 0:
        dist_ids = nb_data[CLUSTER_COL].unique()
        dist_index = {
            d: np.asarray(g, dtype=np.int64)
            for d, g in nb_data.groupby(CLUSTER_COL, sort=False).indices.items()
        }
        seeds = np.random.SeedSequence(BOOT_SEED).spawn(N_BOOTSTRAP)
        out = [
            _boot_replicate(
                s, nb_data, dist_index, dist_ids,
                rhs_a, rhs_b, fe_terms, CLUSTER_COL)
            for s in seeds
        ]

        boot_pcts = [v for v, err in out if err is None]
        for _, err in out:
            if err is not None:
                failures[err] += 1

        n_ok = len(boot_pcts)
        success_rate = n_ok / N_BOOTSTRAP
        if failures:
            print(f"  [{indicator}] Bootstrap failures: {dict(failures)}")
        if success_rate < BOOT_MIN_SUCCESS:
            raise RuntimeError(
                f"[{indicator}] Only {n_ok}/{N_BOOTSTRAP} bootstrap replicates "
                f"converged ({success_rate:.0%} < {BOOT_MIN_SUCCESS:.0%}). "
                f"Failure modes: {dict(failures)}."
            )

        alpha = 1 - BOOT_CI_LEVEL
        boot_arr = np.asarray(boot_pcts)
        deficit["ci_lo"] = float(np.percentile(boot_arr, 100 * alpha / 2))
        deficit["ci_hi"] = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
        deficit["n_boot_ok"] = n_ok
        frac_le = float(np.mean(boot_arr <= 0))
        frac_ge = float(np.mean(boot_arr >= 0))
        deficit["p_boot"] = float(
            min(1.0, max(2 * min(frac_le, frac_ge), 1.0 / (n_ok + 1)))
        )
        pd.DataFrame({"deficit_pct": boot_pcts}).to_csv(
            f"{OUT_DIR}bootstrap_distribution_{indicator}.csv", index=False
        )

    deficit["indicator"] = indicator
    deficit["label"] = INDICATOR_LABELS.get(indicator, indicator)
    deficit["n_obs"] = int(len(nb_data))
    deficit["n_fac"] = nb_data["facility"].nunique()
    deficit["n_clust"] = nb_data[CLUSTER_COL].nunique()
    deficit["spline_df"] = SPLINE_DF
    deficit["curve_ref"] = CURVE_REF

    # Store for post-processing — only addition to run_indicator
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
    print(f"Estimator = NB FE (fixest::fenegbin via rpy2), CRV1 on {CLUSTER_COL}")
    print(f"Spline df = {SPLINE_DF}, lags = {LAG_MONTHS}")
    print(f"Exposure = {WBGT_VAR}")
    print(f"MIN_OBS = {MIN_OBS}")
    print(f"Bootstrap replicates = {N_BOOTSTRAP}")
    print(f"Curve reference = {CURVE_REF}")
    print("=" * 60)

    all_results: list[dict] = []
    for ind in COUNT_INDICATORS:
        print(f"\n-> {ind}")
        result = run_indicator(ind)
        if result is not None:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No indicators fitted successfully — check panel paths.")

    results_df = pd.DataFrame(all_results)
    fitted     = [r["indicator"] for r in all_results]

    q, rej = bh_fdr(results_df["p_boot"].values, alpha=FDR_ALPHA)
    results_df["q_bh"]   = q
    results_df["sig_bh"] = rej

    save_cols = [c for c in results_df.columns if not c.startswith("_")]
    results_df[save_cols].to_csv(
        f"{OUT_DIR}two_model_deficit_results_NB.csv", index=False)
    print(f"\nResults saved -> {OUT_DIR}two_model_deficit_results_NB.csv")
    print(
        f"BH-FDR at alpha={FDR_ALPHA}: "
        f"{int(results_df['sig_bh'].sum())}/{len(results_df)} indicators significant."
    )

    # -----------------------------------------------------------------------
    # FOREST PLOT  — identical to working script
    # -----------------------------------------------------------------------
    plot_df = results_df.sort_values("deficit_pct", ascending=True).reset_index(drop=True)
    y_pos   = np.arange(len(plot_df))
    has_ci = plot_df["ci_lo"].notna().any()
    colors = []
    for _, row in plot_df.iterrows():
        if not has_ci:
            colors.append("#4a7298")
        elif bool(row["sig_bh"]):
            colors.append("#823038")
        else:
            colors.append("#888888")

    fig, ax = plt.subplots(figsize=(7, max(4, len(plot_df) * 0.55 + 1.5)))

    for i, row in plot_df.iterrows():
        if pd.notna(row["ci_lo"]):
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=colors[i], linewidth=1.4, zorder=1)

    ax.scatter(plot_df["deficit_pct"], y_pos, color=colors, s=55, zorder=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=9)
    ax.set_xlabel("% change in appointments associated with WBGT", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    ci_note = f", {int(BOOT_CI_LEVEL * 100)}% bootstrap CI" if has_ci else ""
    ax.set_title(
        f"WBGT-associated deficit (NB FE: Model A vs counterfactual)\n"
        f"fenegbin, cr({WBGT_VAR}, df={SPLINE_DF}) + lags, "
        f"facility + month FE{ci_note}",
        fontsize=11, fontweight="bold",
    )

    if has_ci:
        ax.legend(
            handles=[
                mpatches.Patch(color="#823038", label=f"BH-FDR q <= {FDR_ALPHA}"),
                mpatches.Patch(color="#888888", label="not significant"),
            ],
            loc="lower right", fontsize=9, frameon=False,
        )

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"θ={row['theta_a']:.1f}" for _, row in plot_df.iterrows()],
        fontsize=7, color="#666666",
    )
    ax2.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}forest_plot_two_model_deficit_NB.png",
        dpi=180, bbox_inches="tight",
    )
    plt.close()
    print(f"Forest plot saved -> {OUT_DIR}forest_plot_two_model_deficit_NB.png")

    # -----------------------------------------------------------------------
    # HOT-MONTH FOREST PLOT — with diagnostic and visible whiskers
    # -----------------------------------------------------------------------
    q_hot, rej_hot = bh_fdr(results_df["p_hot_analytical"].values, alpha=FDR_ALPHA)
    results_df["q_hot_bh"] = q_hot
    results_df["sig_hot_bh"] = rej_hot

    print("\n=== HOT-MONTH CI DIAGNOSTIC ===")
    diag = results_df[
        ["indicator", "hot_deficit_pct", "hot_deficit_ci_lo", "hot_deficit_ci_hi", "p_hot_analytical", "sig_hot_bh"]
    ].copy()
    diag["ci_width"] = diag["hot_deficit_ci_hi"] - diag["hot_deficit_ci_lo"]
    print(diag.to_string())
    print(f"CIs present: {diag['hot_deficit_ci_lo'].notna().sum()} / {len(diag)}")
    print(f"BH-significant: {int(diag['sig_hot_bh'].sum())} / {len(diag)}")
    print("================================\n")

    ph = results_df.dropna(subset=["hot_deficit_pct"]).copy()
    ph = ph.sort_values("hot_deficit_pct").reset_index(drop=True)
    y_ph = np.arange(len(ph))
    hot_colors = ["#823038" if s else "#888888" for s in ph["sig_hot_bh"]]

    fig, ax = plt.subplots(figsize=(7, max(4, len(ph) * 0.55 + 1.5)))
    for i, row in ph.iterrows():
        lo, hi, pt = row["hot_deficit_ci_lo"], row["hot_deficit_ci_hi"], row["hot_deficit_pct"]
        if pd.notna(lo) and pd.notna(hi):
            ax.errorbar(
                pt,
                i,
                xerr=[[pt - lo], [hi - pt]],
                fmt="o",
                markersize=7,
                capsize=4,
                capthick=1.4,
                elinewidth=1.4,
                color=hot_colors[i],
                zorder=2,
            )
        else:
            ax.scatter(pt, i, color=hot_colors[i], s=55, zorder=2)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_yticks(y_ph)
    ax.set_yticklabels(ph["label"], fontsize=9)
    ax.set_xlabel(f"% change in appointments in months with WBGT > {REFERENCE_WBGT}°C", fontsize=10)
    ax.set_title(
        f"Hot-month deficit (WBGT > {REFERENCE_WBGT}°C only)\n"
        f"NB FE two-model, 95% analytical CI, red = BH-FDR q ≤ {FDR_ALPHA}",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(
        handles=[
            mpatches.Patch(color="#823038", label=f"BH-FDR q ≤ {FDR_ALPHA}"),
            mpatches.Patch(color="#888888", label="not significant"),
        ],
        loc="lower right",
        fontsize=9,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}forest_plot_hot_deficit_NB.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Hot-month forest plot -> {OUT_DIR}forest_plot_hot_deficit_NB.png")
    # -----------------------------------------------------------------------
    # HISTORICAL BURDEN
    # -----------------------------------------------------------------------
    print("\nHistorical burden...")
    burden_rows = []

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
        bdf.to_csv(f"{OUT_DIR}historical_burden_{ind}.csv", index=False)

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

    pd.DataFrame(burden_rows).to_csv(
        f"{OUT_DIR}historical_burden_summary.csv", index=False)

    n_ind = len(fitted)
    n_cols = 3
    n_rows = int(np.ceil(n_ind / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows), sharex=True)
    af = axes.flatten() if n_ind > 1 else [axes]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for idx, res in enumerate(all_results):
        ax = af[idx]
        ind = res["indicator"]
        nb = res["_nb_data"]
        mu_a, mu_b = res["_mu_a"], res["_mu_b"]
        na, va = res.get("_names_a"), res.get("_vcov_a")
        nb_names, vb = res.get("_names_b"), res.get("_vcov_b")
        have_ci = va is not None and vb is not None

        months = nb["month"].values
        pcts, los, his = [], [], []
        for m in range(1, 13):
            mask = months == m
            if not mask.any():
                pcts.append(0)
                los.append(np.nan)
                his.append(np.nan)
                continue
            if have_ci:
                d = two_model_deficit_analytical(
                    mu_a[mask], mu_b[mask], na, va, nb_names, vb, nb.loc[mask].reset_index(drop=True)
                )
                pcts.append(d["deficit_pct"])
                los.append(d["ci_lo"])
                his.append(d["ci_hi"])
            else:
                sa, sb = float(mu_a[mask].sum()), float(mu_b[mask].sum())
                pcts.append(100 * (sa - sb) / sb if sb > 0 else 0)
                los.append(np.nan)
                his.append(np.nan)

        pcts_a = np.asarray(pcts, dtype=float)
        los_a = np.asarray(los, dtype=float)
        his_a = np.asarray(his, dtype=float)
        bar_c = ["#823038" if p < 0 else "#2a78d6" for p in pcts_a]

        yerr = np.array(
            [
                np.nan_to_num(pcts_a - los_a, nan=0.0),
                np.nan_to_num(his_a - pcts_a, nan=0.0),
            ]
        )
        ax.bar(
            range(12), pcts_a, color=bar_c, alpha=0.8, yerr=yerr, error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": "#333"}
        )
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_names, fontsize=6, rotation=45)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(INDICATOR_LABELS.get(ind, ind), fontsize=9, fontweight="bold")
        if idx % n_cols == 0:
            ax.set_ylabel("% deficit", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n_ind, len(af)):
        af[idx].set_visible(False)
    fig.suptitle(
        "Two-model deficit by calendar month (95% analytical CI)\n"
        "(mu_A − mu_B) / mu_B × 100;  red = heat reduced services",
        fontsize=10,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}deficit_by_month.png", dpi=180, bbox_inches="tight")
    plt.close()

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
              .agg(obs=("y_int", "sum"), mu_a=("mu_a", "sum"), mu_b=("mu_b", "sum"))
              .sort_index())
        ax.plot(m.index, m.mu_b, color="#2a78d6", lw=1.0, ls="--",
                alpha=0.8, label="Model B (no weather)")
        ax.plot(m.index, m.obs, color="#333", lw=1.0, label="Observed")
        ax.fill_between(m.index, m.mu_a, m.mu_b,
                        where=m.mu_a < m.mu_b,
                        color="#823038", alpha=0.25, label="Heat deficit")
        ax.set_title(INDICATOR_LABELS.get(ind, ind), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6)

    for idx in range(n_ind, len(af)):
        af[idx].set_visible(False)
    fig.suptitle(
        "Observed vs Model B counterfactual\nShaded = two-model heat deficit",
        fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}timeseries_burden.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {OUT_DIR}historical_burden_summary.csv")

    # -----------------------------------------------------------------------
    # FORWARD PROJECTIONS
    # -----------------------------------------------------------------------
    print("\nForward projections...")
    all_proj = []

    for res in all_results:
        ind         = res["indicator"]
        nb_data     = res["_nb_data"]
        mu_b        = res["_mu_b"]
        spline_cols = res["_spline_cols"]
        design_info = res["_design_info"]
        wbgt_shift  = res["_wbgt_shift"]
        coefs_a     = nb_coefficient_lookup(res["_model_a"])
        spline_beta = np.array(
            [coefs_a.get(c, 0.0) for c in spline_cols], dtype=float)

        sum_a_hist   = float(np.nansum(res["_mu_a"]))
        sum_b_hist   = float(np.nansum(mu_b))
        deficit_hist = (
            (sum_a_hist - sum_b_hist) / sum_b_hist * 100
            if sum_b_hist > 0 else np.nan)

        hist_clim = (nb_data.groupby(["facility", "month"])[WBGT_VAR]
                     .mean().reset_index()
                     .rename(columns={WBGT_VAR: "wbgt_hist"}))
        hist_counts = (nb_data.groupby(["facility", "month"])["y_int"]
                       .mean().reset_index()
                       .rename(columns={"y_int": "baseline"}))
        hist_mu_b = (nb_data.assign(mu_b=mu_b)
                     .groupby(["facility", "month"])["mu_b"]
                     .mean().reset_index()
                     .rename(columns={"mu_b": "mu_b_hist"}))

        for ssp in SSP_SCENARIOS:
            for tier in MODEL_TIERS:
                pf_path = f"{PROJECTION_DIR}/{ssp}/{tier}.csv"
                if not os.path.exists(pf_path):
                    continue
                proj = pd.read_csv(pf_path)
                proj["facility"] = proj["facility"].astype(str)
                if WBGT_VAR not in proj.columns:
                    continue
                proj = proj.rename(columns={WBGT_VAR: "wbgt_proj"})
                proj = (proj
                        .merge(hist_clim,   on=["facility", "month"], how="inner")
                        .merge(hist_counts, on=["facility", "month"], how="inner")
                        .merge(hist_mu_b,   on=["facility", "month"], how="inner"))
                if proj.empty:
                    continue

                n_rows    = len(proj)
                all_pts_c = np.concatenate([
                    proj["wbgt_hist"].values - wbgt_shift,
                    proj["wbgt_proj"].values - wbgt_shift,
                ])
                basis_all = np.asarray(
                    patsy.build_design_matrices(
                        [design_info], {"x": all_pts_c})[0],
                    dtype=float)
                basis_h = basis_all[:n_rows]
                basis_p = basis_all[n_rows:]
                irrs    = np.exp((basis_p - basis_h) @ spline_beta)

                proj["mu_a_proj"] = proj["baseline"] * irrs
                proj["mu_b_proj"] = proj["mu_b_hist"]

                sum_a_proj    = float(proj["mu_a_proj"].sum())
                sum_b_proj    = float(proj["mu_b_proj"].sum())
                deficit_proj  = (
                    (sum_a_proj - sum_b_proj) / sum_b_proj * 100
                    if sum_b_proj > 0 else np.nan)
                delta_deficit = deficit_proj - deficit_hist
                wd = float((proj["wbgt_proj"] - proj["wbgt_hist"]).mean())

                print(f"  {ind} {ssp}/{tier}: "
                      f"dWBGT={wd:+.2f}  "
                      f"deficit_hist={deficit_hist:+.2f}%  "
                      f"deficit_proj={deficit_proj:+.2f}%  "
                      f"Δdeficit={delta_deficit:+.2f}%")

                proj["indicator"]     = ind
                proj["ssp"]           = ssp
                proj["tier"]          = tier
                proj["deficit_hist"]  = deficit_hist
                proj["deficit_proj"]  = deficit_proj
                proj["delta_deficit"] = delta_deficit
                proj.to_csv(
                    f"{OUT_DIR}projection_{ind}_{ssp}_{tier}.csv", index=False)
                all_proj.append({
                    "indicator":      ind,
                    "ssp":            ssp,
                    "tier":           tier,
                    "mean_wbgt_diff": wd,
                    "deficit_hist":   deficit_hist,
                    "deficit_proj":   deficit_proj,
                    "delta_deficit":  delta_deficit,
                })

    if all_proj:
        pd.DataFrame(all_proj).to_csv(
            f"{OUT_DIR}projection_summary.csv", index=False)

        for ind in fitted:
            ind_proj = [p for p in all_proj if p["indicator"] == ind]
            if not ind_proj:
                continue
            grid = pd.DataFrame(
                index=SSP_SCENARIOS, columns=MODEL_TIERS, dtype=float)
            for p in ind_proj:
                grid.loc[p["ssp"], p["tier"]] = p["delta_deficit"]

            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(
                grid.values.astype(float), cmap="RdBu_r", aspect="auto")
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
                "Δ deficit (proj − hist, %)", fontsize=9)
            ax.set_title(
                f"{INDICATOR_LABELS.get(ind, ind)}\n"
                "Change in two-model deficit under CMIP6",
                fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(
                f"{OUT_DIR}projection_heatmap_{ind}.png",
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
                [design_info], {"x": all_pts_c})[0],
            dtype=float)
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
            tlo.pivot(
                index="wbgt", columns="indicator", values=col
            ).to_csv(f"{OUT_DIR}tlo_{nm}_wide.csv")

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
        plt.savefig(
            f"{OUT_DIR}tlo_disruption_curves.png", dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  -> {OUT_DIR}tlo_wbgt_lookup.csv")

    # -----------------------------------------------------------------------
    # DISTRICT AGGREGATION
    # -----------------------------------------------------------------------
    print("\nDistrict aggregation...")
    all_dist = []

    for res in all_results:
        ind = res["indicator"]
        csv = f"{OUT_DIR}historical_burden_{ind}.csv"
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        if CLUSTER_COL not in df.columns:
            continue
        d = df.groupby(CLUSTER_COL).agg(
            obs=("y_int",  "sum"),
            mu_a=("mu_a",  "sum"),
            mu_b=("mu_b",  "sum"),
        ).reset_index()
        d["deficit_pct"] = (d["mu_a"] - d["mu_b"]) / d["mu_b"] * 100
        d["indicator"]   = ind
        all_dist.append(d)
        d.to_csv(f"{OUT_DIR}district_burden_{ind}.csv", index=False)
        print(f"  {ind}: {len(d)} districts, "
              f"mean deficit {d['deficit_pct'].mean():.2f}%")

    if all_dist:
        pd.concat(all_dist, ignore_index=True).to_csv(
            f"{OUT_DIR}district_burden_all.csv", index=False)
        print(f"  -> {OUT_DIR}district_burden_all.csv")

    print(f"\nAll outputs in {OUT_DIR}")
    print("Done.")
