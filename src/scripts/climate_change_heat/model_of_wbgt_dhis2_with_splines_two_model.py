"""
loop_all_indicators_two_model_NB.py

Two-model approach for WBGT–health-service disruption:
  Model A (exposure):      y ~ cr(WBGT, df) + WBGT_lags + covid + year | facility + month
  Model B (counterfactual): y ~                            covid + year | facility + month

Negative Binomial FE (fixest::fenegbin via rpy2) with district-clustered SEs.

NB2 is used instead of Poisson FE because the Poisson pseudo-MLE score
equations force sum(mu) = sum(y) within each FE stratum, making the
A-minus-B deficit identically zero on a shared sample. The NB2 score
weights residuals by 1/(1 + theta*mu), relaxing that identity and
permitting a meaningful two-model comparison.

Computes % deficit = 100 * (sum(pred_A) - sum(pred_B)) / sum(pred_B)
for each indicator, with a district-level block bootstrap, produces a
forest plot, and saves exposure-response curves for the contemporaneous
WBGT spline component.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
from collections import Counter
from contextvars import ContextVar

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# rpy2 wiring — R + fixest
# ---------------------------------------------------------------------------
# Adjust R_HOME if your R installation lives elsewhere
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

# Ensure conversion rules are available inside joblib threading workers.
# rpy2 stores converter state in a ContextVar, so thread-local workers may
# otherwise see the converter context as missing even when localconverter(...)
# is used. We seed each thread with the default converter up front.
ro.conversion.converter_ctx = ContextVar("converter", default=ro.default_converter)

# Suppress the pyfixest multicollinearity warnings (not used, but in case
# any pyfixest code is imported transitively)
warnings.filterwarnings(
    "once",
    message=r".*variables dropped due to multicollinearity.*",
    category=UserWarning,
)

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

# Weather / model settings
WBGT_VAR = "wbgt5x_day"
SPLINE_DF = 3
LAG_MONTHS = [1, 2, 3, 9]
CENTER = True
MIN_OBS = 10
min_year_historical = 2015
max_year_historical = 2025
apply_cap = False

# Exposure-response curve settings
N_CURVE_POINTS = 200
CURVE_REF = "mean"  # "mean", "median", or "min"

# COVID window — controlled via a dummy, NOT masked to NaN
COVID_START = "2020-04-01"
COVID_END = "2021-12-01"

# Known structural closures — masked to NaN (missing), NOT set to 0
CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre", "2023-03-01", "2024-03-01"),
]

# Clustering / bootstrap unit
CLUSTER_COL = "Dist"

# Bootstrap
N_BOOTSTRAP = 50
BOOT_SEED = 42
BOOT_CI_LEVEL = 0.95
BOOT_MIN_SUCCESS = 0.80
N_JOBS = 1  # threading workers (rpy2 objects are not picklable — must use threading)

# Multiplicity
FDR_ALPHA = 0.05

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)
PANEL_DIST_COL_IN_PANEL = "Dist"


# ---------------------------------------------------------------------------
# R helpers
# ---------------------------------------------------------------------------
def fit_nb_fixest(
    df: pd.DataFrame,
    rhs_terms: list[str],
    fe_terms: list[str],
    cluster_col: str,
    y_col: str = "y_int",
):
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
            cluster=ro.StrVector([cluster_col]),
        )
    )

    mu_r = stats_r.fitted(r_model)
    with localconverter(ro.default_converter + pandas2ri.converter):
        mu = np.asarray(ro.conversion.rpy2py(mu_r), dtype=float)

    if len(mu) != len(df):
        raise ValueError(
            f"fenegbin returned {len(mu)} fitted values for {len(df)} rows."
        )
    return r_model, mu


def nb_coef_table(r_model) -> pd.DataFrame:
    """Extract coefficient table from a fenegbin model with term names preserved."""
    coeftab = fixest.coeftable(r_model)

    with localconverter(ro.default_converter + pandas2ri.converter):
        tab = ro.conversion.rpy2py(coeftab)

    # If rpy2 returns a numpy array / dataframe-like object without rownames,
    # fetch rownames directly from R.
    rownames = list(base.rownames(coeftab))
    tab = pd.DataFrame(np.asarray(tab))
    tab.index = rownames
    tab.index.name = "term"
    tab = tab.reset_index()

    # fixest coeftable columns are typically:
    # Estimate, Std. Error, z value, Pr(>|z|)
    if tab.shape[1] >= 5:
        tab = tab.iloc[:, :5].copy()
        tab.columns = ["term", "estimate", "se", "z", "p"]
    else:
        raise ValueError(
            f"Unexpected coeftable shape: {tab.shape}. Could not parse coefficients."
        )

    return tab


def nb_theta(r_model) -> float:
    """Extract θ (NB dispersion) from a fenegbin model."""
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


def nb_coefficient_lookup(r_model) -> dict[str, float]:
    """Map coefficient names to estimates from a fenegbin model."""
    tab = nb_coef_table(r_model)
    return dict(zip(tab["term"].astype(str), tab["estimate"].astype(float)))


# ---------------------------------------------------------------------------
# Data helpers (unchanged from original)
# ---------------------------------------------------------------------------
def prepare_data(indicator: str) -> pd.DataFrame | None:
    """Load, clean, and return the panel for one indicator."""
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
    """
    Reindex each facility onto a gap-free monthly sequence so that .shift(k)
    is a true k-month lag.
    """
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
    """Add centred year, centred WBGT, and lag columns."""
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
    """
    Materialise the natural cubic spline basis as explicit columns.

    Returns (df, spline_col_names, design_info).
    design_info is retained so that the exposure-response curve can
    reconstruct the same basis on a prediction grid.
    """
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
    """
    Drop facilities whose outcome is zero in every retained month (perfect
    separation under log-link).
    """
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
# Exposure-response curve
# ---------------------------------------------------------------------------
def make_exposure_response_curve(
    r_model,
    spline_cols: list[str],
    wbgt_shift: float,
    observed_wbgt: pd.Series,
    indicator: str,
    design_info,
) -> pd.DataFrame:
    """
    Build the contemporaneous WBGT exposure-response curve on the RR scale.

    Uses the saved design_info from fitting so that knots are identical.
    """
    coefs = nb_coefficient_lookup(r_model)

    x_min = float(observed_wbgt.min())
    x_max = float(observed_wbgt.max())
    x_grid = np.linspace(x_min, x_max, N_CURVE_POINTS)
    x_grid_c = x_grid - wbgt_shift

    # Reconstruct basis with SAME knots as fitting
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
    """Save a simple exposure-response plot for one indicator."""
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
# BH-FDR
# ---------------------------------------------------------------------------
def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg. Returns (adjusted p-values, reject flags)."""
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
# Bootstrap replicate (module level so joblib can find it)
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
    # Re-seed rpy2 conversion context inside each thread.
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

    # Drop any facility that is all-zero in this resample (separation)
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
# Per-indicator runner
# ---------------------------------------------------------------------------
def run_indicator(indicator: str) -> dict | None:
    """Fit Model A and Model B (NB FE), return deficit summary."""

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

    # Spline basis materialised once on the analysis sample
    nb_data, spline_cols, design_info = add_spline_basis(nb_data, SPLINE_DF)
    nb_data = nb_data.reset_index(drop=True)

    fe_terms = ["facility", "month"]

    # --- Model A: with WBGT spline + lags ----------------------------------
    rhs_a = spline_cols + lag_terms + ["covid", "year_c"]

    # --- Model B: counterfactual (no WBGT at all) --------------------------
    rhs_b = ["covid", "year_c"]

    try:
        model_a, mu_a = fit_nb_fixest(nb_data, rhs_a, fe_terms, CLUSTER_COL)
        model_b, mu_b = fit_nb_fixest(nb_data, rhs_b, fe_terms, CLUSTER_COL)
    except Exception as e:
        print(f"  [{indicator}] fenegbin failed: {type(e).__name__}: {e} — skipping.")
        return None

    # --- Deficit -----------------------------------------------------------
    total_a = float(mu_a.sum())
    total_b = float(mu_b.sum())
    total_y = float(nb_data["y_int"].sum())

    # Diagnostic: confirm NB breaks the Poisson identity
    print(
        f"  [{indicator}] sum(y)={total_y:,.0f}, "
        f"sum(mu_A)={total_a:,.0f}, sum(mu_B)={total_b:,.0f}"
    )

    deficit = {
        "total_pred_exposure": total_a,
        "total_pred_counterfactual": total_b,
        "deficit_abs": total_a - total_b,
        "deficit_pct": 100.0 * (total_a - total_b) / total_b,
    }

    # θ diagnostic
    deficit["theta_a"] = nb_theta(model_a)
    deficit["theta_b"] = nb_theta(model_b)

    # --- Exposure-response curve -------------------------------------------
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

    # --- Per-facility-month predictions (for mapping) ----------------------
    preds = nb_data[["facility", "date"]].copy()
    preds["y_obs"] = nb_data["y_int"].values
    preds["y_pred_wx"] = mu_a
    preds["y_pred_base"] = mu_b
    preds["difference"] = preds["y_pred_base"] - preds["y_pred_wx"]

    if PANEL_DIST_COL_IN_PANEL in nb_data.columns:
        preds["Dist"] = nb_data[PANEL_DIST_COL_IN_PANEL].values

    preds.to_csv(f"{OUT_DIR}two_model_predictions_{indicator}.csv", index=False)

    # --- Coefficient table export ------------------------------------------
    try:
        coef_tab = nb_coef_table(model_a)
        coef_tab["indicator"] = indicator
        coef_tab.to_csv(
            f"{OUT_DIR}coef_table_model_a_{indicator}.csv", index=False
        )
    except Exception as e:
        print(f"  [{indicator}] Coef table export failed: {e}")

    # --- Bootstrap CIs (district-level block, threading backend) -----------
    failures: Counter = Counter()

    if N_BOOTSTRAP > 0:
        dist_ids = nb_data[CLUSTER_COL].unique()
        dist_index = {
            d: np.asarray(g, dtype=np.int64)
            for d, g in nb_data.groupby(CLUSTER_COL, sort=False).indices.items()
        }

        seeds = np.random.SeedSequence(BOOT_SEED).spawn(N_BOOTSTRAP)

        # IMPORTANT: backend="threading" because rpy2 objects cannot be
        # pickled across loky/multiprocessing workers.
        out = [_boot_replicate(s, nb_data, dist_index, dist_ids, rhs_a, rhs_b, fe_terms, CLUSTER_COL) for s in seeds]

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
        # Two-sided bootstrap p-value against H0: deficit = 0
        frac_le = float(np.mean(boot_arr <= 0))
        frac_ge = float(np.mean(boot_arr >= 0))
        deficit["p_boot"] = float(
            min(1.0, max(2 * min(frac_le, frac_ge), 1.0 / (n_ok + 1)))
        )

        # Save bootstrap distribution for diagnostics
        pd.DataFrame({"deficit_pct": boot_pcts}).to_csv(
            f"{OUT_DIR}bootstrap_distribution_{indicator}.csv", index=False
        )
    else:
        deficit["ci_lo"] = np.nan
        deficit["ci_hi"] = np.nan
        deficit["n_boot_ok"] = 0
        deficit["p_boot"] = np.nan

    deficit["indicator"] = indicator
    deficit["label"] = INDICATOR_LABELS.get(indicator, indicator)
    deficit["n_obs"] = int(len(nb_data))
    deficit["n_fac"] = nb_data["facility"].nunique()
    deficit["n_clust"] = nb_data[CLUSTER_COL].nunique()
    deficit["spline_df"] = SPLINE_DF
    deficit["curve_ref"] = CURVE_REF

    print(
        f"  [{indicator}] OK  n={len(nb_data):,}, "
        f"fac={nb_data['facility'].nunique()}, "
        f"clust={nb_data[CLUSTER_COL].nunique()}, "
        f"θ_a={deficit['theta_a']:.2f}, θ_b={deficit['theta_b']:.2f}, "
        f"deficit={deficit['deficit_pct']:+.2f}%"
    )
    return deficit


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Two-model NB analysis: exposure vs counterfactual")
    print(f"Estimator = NB FE (fixest::fenegbin via rpy2), CRV1 on {CLUSTER_COL}")
    print(f"Spline df = {SPLINE_DF}, lags = {LAG_MONTHS}")
    print(f"Exposure = {WBGT_VAR}")
    print(f"Bootstrap replicates = {N_BOOTSTRAP} (n_jobs={N_JOBS}, threading)")
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

    # --- Benjamini-Hochberg across indicators ------------------------------
    q, rej = bh_fdr(results_df["p_boot"].values, alpha=FDR_ALPHA)
    results_df["q_bh"] = q
    results_df["sig_bh"] = rej

    results_df.to_csv(f"{OUT_DIR}two_model_deficit_results_NB.csv", index=False)
    print(f"\nResults saved -> {OUT_DIR}two_model_deficit_results_NB.csv")
    print(
        f"BH-FDR at alpha={FDR_ALPHA}: "
        f"{int(results_df['sig_bh'].sum())}/{len(results_df)} indicators significant."
    )

    # -----------------------------------------------------------------------
    # FOREST PLOT — % deficit per indicator
    # -----------------------------------------------------------------------
    plot_df = results_df.sort_values("deficit_pct", ascending=True).reset_index(
        drop=True
    )
    y_pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(7, max(4, len(plot_df) * 0.55 + 1.5)))

    has_ci = plot_df["ci_lo"].notna().all()

    colors = []
    for _, row in plot_df.iterrows():
        if not has_ci:
            colors.append("#4a7298")
        elif bool(row["sig_bh"]):
            colors.append("#823038")
        else:
            colors.append("#888888")

    if has_ci:
        for i, row in plot_df.iterrows():
            ax.plot(
                [row["ci_lo"], row["ci_hi"]],
                [i, i],
                color=colors[i],
                linewidth=1.4,
                zorder=1,
            )

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
        fontsize=11,
        fontweight="bold",
    )

    if has_ci:
        sig_patch = mpatches.Patch(color="#823038", label=f"BH-FDR q <= {FDR_ALPHA}")
        ns_patch = mpatches.Patch(color="#888888", label="not significant")
        ax.legend(
            handles=[sig_patch, ns_patch],
            loc="lower right",
            fontsize=9,
            frameon=False,
        )

    # Add θ annotations on the right margin
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"θ={row['theta_a']:.1f}" for _, row in plot_df.iterrows()],
        fontsize=7,
        color="#666666",
    )
    ax2.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}forest_plot_two_model_deficit_NB.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Forest plot saved -> {OUT_DIR}forest_plot_two_model_deficit_NB.png")
    print("\nDone.")
