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


# ===========================================================================
# 1. CONFIG
# ===========================================================================

COUNT_INDICATORS = [
    "fp_total_clients", "opd_attendance", "ipd_total_admissions",
    "vmmc_first_visits", "pnc_mother_checked_48h", "anc_new_attendees",
    "anc_first_trimester_starts", "bcg_under1", "penta3_under1",
    "measles1_under1", "fully_immunised_under1", "pnc_within_2wks",
    "pnc_first_visit_2wks", "live_births_total", "skilled_deliveries",
]

INDICATOR_LABELS = {
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

WBGT_VAR            = "wbgt5x_day"
SPLINE_DF           = 3
LAG_MONTHS          = [1, 2, 3, 9]
CENTER              = True
MIN_OBS             = int(0.5 * 12 * 12)   # 72  ← changed from 10

min_year_historical = 2015
max_year_historical = 2025
apply_cap           = False

N_CURVE_POINTS      = 200
CURVE_REF           = "mean"

COVID_START         = "2020-04-01"
COVID_END           = "2021-12-01"

CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre",  "2023-03-01", "2024-03-01"),
]

CLUSTER_COL         = "Dist"

N_BOOTSTRAP         = 50
BOOT_SEED           = 42
BOOT_CI_LEVEL       = 0.95
BOOT_MIN_SUCCESS    = 0.80
N_JOBS              = 1

FDR_ALPHA           = 0.05

# Reference WBGT for burden / TLO tables
REFERENCE_WBGT      = 25.0
WBGT_GRID           = np.arange(20.0, 37.0, 0.5)

# Forward projections
PROJECTION_DIR = (
    "/Users/rachelmurray-watson/Documents/Heat_data"
    "/CMIP6_facility_projections"
)
SSP_SCENARIOS  = ["ssp126", "ssp245", "ssp585"]
MODEL_TIERS    = ["lowest", "median", "highest"]

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)
PANEL_DIST_COL_IN_PANEL = "Dist"


# ===========================================================================
# 2. R HELPERS  — unchanged from working script
# ===========================================================================

def fit_nb_fixest(
    df: pd.DataFrame,
    rhs_terms: list[str],
    fe_terms: list[str],
    cluster_col: str,
    y_col: str = "y_int",
):
    rhs = " + ".join(rhs_terms) if rhs_terms else "1"
    fe  = " + ".join(fe_terms)
    fml = ro.Formula(f"{y_col} ~ {rhs} | {fe}")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    r_df = base.as_data_frame(r_df)

    for col in set(fe_terms + [cluster_col]):
        r_df.rx2[col] = base.as_factor(r_df.rx2(col))

    suppressWarnings = ro.r("suppressWarnings")
    r_model = suppressWarnings(
        fixest.fenegbin(
            fml     = fml,
            data    = r_df,
            cluster = ro.StrVector([cluster_col]),
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
    coeftab  = fixest.coeftable(r_model)
    with localconverter(ro.default_converter + pandas2ri.converter):
        tab = ro.conversion.rpy2py(coeftab)
    rownames = list(base.rownames(coeftab))
    tab = pd.DataFrame(np.asarray(tab))
    tab.index      = rownames
    tab.index.name = "term"
    tab = tab.reset_index()
    if tab.shape[1] >= 5:
        tab = tab.iloc[:, :5].copy()
        tab.columns = ["term", "estimate", "se", "z", "p"]
    else:
        raise ValueError(
            f"Unexpected coeftable shape: {tab.shape}.")
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


def nb_coefficient_lookup(r_model) -> dict[str, float]:
    tab = nb_coef_table(r_model)
    return dict(zip(tab["term"].astype(str), tab["estimate"].astype(float)))


# ===========================================================================
# 3. DATA HELPERS  — unchanged from working script
# ===========================================================================

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
            f"[{indicator}] Cluster column '{CLUSTER_COL}' not present.")

    long["covid"] = long["date"].between(COVID_START, COVID_END).astype(int)

    for fac, start, end in CLOSURES:
        mask    = (long["date"].between(start, end)) & (long["facility"] == fac)
        n_masked = int(mask.sum())
        long.loc[mask, "y"] = np.nan
        if n_masked:
            print(f"  [{indicator}] Masked {n_masked} closure months for {fac}.")

    long["year"]  = long["date"].dt.year
    long["month"] = long["date"].dt.month
    long = long[long["year"].between(
        min_year_historical, max_year_historical - 1)]

    if apply_cap:
        long.loc[long["y"] > 4e3, "y"] = np.nan

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
            f"[{indicator}] {len(bad)} facilities map to multiple districts.")
    if long[CLUSTER_COL].isna().any():
        n_bad = int(long[CLUSTER_COL].isna().sum())
        raise ValueError(
            f"[{indicator}] {n_bad} rows have missing '{CLUSTER_COL}'.")
    return long


def enforce_complete_monthly_grid(
        df: pd.DataFrame, indicator: str) -> pd.DataFrame:
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
            f"({100 * n_inserted / len(out):.2f}% of grid).")

    out["year"]  = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["covid"] = out["date"].between(COVID_START, COVID_END).astype(int)
    out[CLUSTER_COL] = out.groupby("facility")[CLUSTER_COL].transform(
        lambda s: s.ffill().bfill())

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

    year_shift = df["year"].mean()  if CENTER else 0.0
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


def drop_separated_facilities(
        df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    all_zero = df.groupby("facility")["y_int"].max() == 0
    sep_facs = all_zero[all_zero].index
    if len(sep_facs):
        n_rows = int(df["facility"].isin(sep_facs).sum())
        print(
            f"  [{indicator}] SEPARATION: {len(sep_facs)} facilities "
            f"all-zero ({n_rows:,} rows) — dropped.")
        df = df[~df["facility"].isin(sep_facs)].copy()
    return df


# ===========================================================================
# 4. EXPOSURE-RESPONSE CURVE  — unchanged from working script
# ===========================================================================

def make_exposure_response_curve(
    r_model,
    spline_cols: list[str],
    wbgt_shift: float,
    observed_wbgt: pd.Series,
    indicator: str,
    design_info,
) -> pd.DataFrame:
    coefs    = nb_coefficient_lookup(r_model)
    x_grid   = np.linspace(float(observed_wbgt.min()),
                            float(observed_wbgt.max()), N_CURVE_POINTS)
    x_grid_c = x_grid - wbgt_shift

    basis_grid = np.asarray(
        patsy.build_design_matrices([design_info], {"x": x_grid_c})[0],
        dtype=float)

    if CURVE_REF == "mean":
        x_ref = float(observed_wbgt.mean())
    elif CURVE_REF == "median":
        x_ref = float(observed_wbgt.median())
    else:
        x_ref = float(observed_wbgt.min())

    x_ref_c   = x_ref - wbgt_shift
    basis_ref = np.asarray(
        patsy.build_design_matrices(
            [design_info], {"x": np.array([x_ref_c])})[0],
        dtype=float)
    ref_row = basis_ref[0]

    beta     = np.array([coefs.get(col, 0.0) for col in spline_cols],
                         dtype=float)
    eta_grid = basis_grid @ beta
    eta_ref  = float(ref_row @ beta)
    rr       = np.exp(eta_grid - eta_ref)

    return pd.DataFrame({
        "indicator":          indicator,
        "label":              INDICATOR_LABELS.get(indicator, indicator),
        "wbgt":               x_grid,
        "wbgt_c":             x_grid_c,
        "rr_vs_ref":          rr,
        "pct_change_vs_ref":  100.0 * (rr - 1.0),
        "wbgt_ref":           x_ref,
        "curve_ref":          CURVE_REF,
        "note": "Contemporaneous spline component only; lagged WBGT held fixed.",
    })


def save_exposure_response_plot(curve_df: pd.DataFrame, indicator: str):
    label = INDICATOR_LABELS.get(indicator, indicator)
    x_ref = float(curve_df["wbgt_ref"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(curve_df["wbgt"], curve_df["rr_vs_ref"],
            color="#2f5d80", linewidth=2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(x_ref, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xlabel("WBGT")
    ax.set_ylabel("Relative rate vs reference WBGT")
    ax.set_title(
        f"Exposure-response curve: {label}\n"
        f"Contemporaneous WBGT spline (reference = {x_ref:.2f})",
        fontsize=11, fontweight="bold")
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}exposure_response_curve_{indicator}.png",
        dpi=180, bbox_inches="tight")
    plt.close()


# ===========================================================================
# 5. BH-FDR  — unchanged
# ===========================================================================

def bh_fdr(
        pvals: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    p  = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q  = np.full_like(p, np.nan)
    rej = np.zeros(p.shape, dtype=bool)
    if ok.sum() == 0:
        return q, rej
    p_ok  = p[ok]
    n     = len(p_ok)
    order = np.argsort(p_ok)
    adj   = p_ok[order] * n / np.arange(1, n + 1)
    adj   = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0, 1)
    q_ok  = np.empty(n)
    q_ok[order] = adj
    q[ok]   = q_ok
    rej[ok] = q_ok <= alpha
    return q, rej


# ===========================================================================
# 6. BOOTSTRAP REPLICATE  — unchanged from working script
# ===========================================================================

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

    rng   = np.random.default_rng(seed_seq)
    picks = rng.choice(len(dist_ids), size=len(dist_ids), replace=True)

    idx_parts = [dist_index[dist_ids[p]] for p in picks]
    idx  = np.concatenate(idx_parts)
    tags = np.repeat(
        np.arange(len(picks)), [len(part) for part in idx_parts]
    ).astype(str)

    boot_df = nb_data.take(idx).reset_index(drop=True)
    tags_s  = pd.Series(tags, index=boot_df.index).astype(str)
    boot_df["facility"]   = boot_df["facility"].astype(str) + "__b" + tags_s
    boot_df[cluster_col]  = boot_df[cluster_col].astype(str) + "__b" + tags_s

    fac_max  = boot_df.groupby("facility")["y_int"].max()
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


# ===========================================================================
# 7. PER-INDICATOR RUNNER
# ===========================================================================

def run_indicator(indicator: str) -> dict | None:

    long = prepare_data(indicator)
    if long is None:
        return None

    long, lag_terms, year_shift, wbgt_shift = add_columns(long, indicator)

    nb_cols = (["y", "facility", "year_c", "wbgt_c", "covid", CLUSTER_COL]
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

    nb_data, spline_cols, design_info = add_spline_basis(nb_data, SPLINE_DF)
    nb_data = nb_data.reset_index(drop=True)

    fe_terms = ["facility", "month"]
    rhs_a    = spline_cols + lag_terms + ["covid", "year_c"]
    rhs_b    = ["covid", "year_c"]

    try:
        model_a, mu_a = fit_nb_fixest(nb_data, rhs_a, fe_terms, CLUSTER_COL)
        model_b, mu_b = fit_nb_fixest(nb_data, rhs_b, fe_terms, CLUSTER_COL)
    except Exception as e:
        print(f"  [{indicator}] fenegbin failed: {e} — skipping.")
        return None

    total_a = float(mu_a.sum())
    total_b = float(mu_b.sum())
    total_y = float(nb_data["y_int"].sum())

    print(
        f"  [{indicator}] sum(y)={total_y:,.0f}, "
        f"sum(mu_A)={total_a:,.0f}, sum(mu_B)={total_b:,.0f}")

    deficit = {
        "total_pred_exposure":        total_a,
        "total_pred_counterfactual":  total_b,
        "deficit_abs":                total_a - total_b,
        "deficit_pct":                100.0 * (total_a - total_b) / total_b,
        "theta_a":                    nb_theta(model_a),
        "theta_b":                    nb_theta(model_b),
    }

    # --- Exposure-response curve -------------------------------------------
    try:
        curve_df = make_exposure_response_curve(
            r_model=model_a, spline_cols=spline_cols,
            wbgt_shift=wbgt_shift, observed_wbgt=nb_data[WBGT_VAR],
            indicator=indicator, design_info=design_info)
        curve_df.to_csv(
            f"{OUT_DIR}exposure_response_curve_{indicator}.csv", index=False)
        save_exposure_response_plot(curve_df, indicator)
    except Exception as e:
        print(f"  [{indicator}] Curve export failed: {e}")

    # --- Per-facility-month predictions ------------------------------------
    preds = nb_data[["facility", "date"]].copy()
    preds["y_obs"]       = nb_data["y_int"].values
    preds["y_pred_wx"]   = mu_a
    preds["y_pred_base"] = mu_b
    preds["difference"]  = preds["y_pred_base"] - preds["y_pred_wx"]
    if PANEL_DIST_COL_IN_PANEL in nb_data.columns:
        preds["Dist"] = nb_data[PANEL_DIST_COL_IN_PANEL].values
    preds.to_csv(
        f"{OUT_DIR}two_model_predictions_{indicator}.csv", index=False)

    # --- Coefficient table -------------------------------------------------
    try:
        coef_tab = nb_coef_table(model_a)
        coef_tab["indicator"] = indicator
        coef_tab.to_csv(
            f"{OUT_DIR}coef_table_model_a_{indicator}.csv", index=False)
    except Exception as e:
        print(f"  [{indicator}] Coef table export failed: {e}")

    # --- Bootstrap CIs — unchanged from working script --------------------
    failures: Counter = Counter()

    if N_BOOTSTRAP > 0:
        dist_ids   = nb_data[CLUSTER_COL].unique()
        dist_index = {
            d: np.asarray(g, dtype=np.int64)
            for d, g in nb_data.groupby(
                CLUSTER_COL, sort=False).indices.items()
        }
        seeds = np.random.SeedSequence(BOOT_SEED).spawn(N_BOOTSTRAP)
        out   = [
            _boot_replicate(
                s, nb_data, dist_index, dist_ids,
                rhs_a, rhs_b, fe_terms, CLUSTER_COL)
            for s in seeds
        ]

        boot_pcts = [v for v, err in out if err is None]
        for _, err in out:
            if err is not None:
                failures[err] += 1

        n_ok         = len(boot_pcts)
        success_rate = n_ok / N_BOOTSTRAP
        if failures:
            print(f"  [{indicator}] Bootstrap failures: {dict(failures)}")
        if success_rate < BOOT_MIN_SUCCESS:
            raise RuntimeError(
                f"[{indicator}] Only {n_ok}/{N_BOOTSTRAP} replicates "
                f"converged ({success_rate:.0%}). {dict(failures)}")

        alpha    = 1 - BOOT_CI_LEVEL
        boot_arr = np.asarray(boot_pcts)
        deficit["ci_lo"]     = float(np.percentile(boot_arr, 100 * alpha / 2))
        deficit["ci_hi"]     = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
        deficit["n_boot_ok"] = n_ok
        frac_le = float(np.mean(boot_arr <= 0))
        frac_ge = float(np.mean(boot_arr >= 0))
        deficit["p_boot"] = float(
            min(1.0, max(2 * min(frac_le, frac_ge), 1.0 / (n_ok + 1))))

        pd.DataFrame({"deficit_pct": boot_pcts}).to_csv(
            f"{OUT_DIR}bootstrap_distribution_{indicator}.csv", index=False)
    else:
        deficit["ci_lo"]     = np.nan
        deficit["ci_hi"]     = np.nan
        deficit["n_boot_ok"] = 0
        deficit["p_boot"]    = np.nan

    deficit.update({
        "indicator":  indicator,
        "label":      INDICATOR_LABELS.get(indicator, indicator),
        "n_obs":      int(len(nb_data)),
        "n_fac":      nb_data["facility"].nunique(),
        "n_clust":    nb_data[CLUSTER_COL].nunique(),
        "spline_df":  SPLINE_DF,
        "curve_ref":  CURVE_REF,
        # Store objects needed by post-processing sections
        "_nb_data":      nb_data,
        "_mu_a":         mu_a,
        "_mu_b":         mu_b,
        "_model_a":      model_a,
        "_spline_cols":  spline_cols,
        "_design_info":  design_info,
        "_wbgt_shift":   wbgt_shift,
    })

    print(
        f"  [{indicator}] OK  n={len(nb_data):,}, "
        f"fac={nb_data['facility'].nunique()}, "
        f"clust={nb_data[CLUSTER_COL].nunique()}, "
        f"θ_a={deficit['theta_a']:.2f}, θ_b={deficit['theta_b']:.2f}, "
        f"deficit={deficit['deficit_pct']:+.2f}%")
    return deficit


# ===========================================================================
# 8. MAIN
# ===========================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Two-model NB analysis: exposure vs counterfactual")
    print(f"MIN_OBS={MIN_OBS}, SPLINE_DF={SPLINE_DF}, CLUSTER={CLUSTER_COL}")
    print(f"Bootstrap replicates = {N_BOOTSTRAP}")
    print("=" * 60)

    all_results: list[dict] = []
    for ind in COUNT_INDICATORS:
        print(f"\n-> {ind}")
        result = run_indicator(ind)
        if result is not None:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No indicators fitted successfully.")

    results_df = pd.DataFrame(all_results)
    fitted     = [r["indicator"] for r in all_results]

    # BH-FDR
    q, rej = bh_fdr(results_df["p_boot"].values, alpha=FDR_ALPHA)
    results_df["q_bh"]   = q
    results_df["sig_bh"] = rej

    # Drop private columns before saving
    save_cols = [c for c in results_df.columns if not c.startswith("_")]
    results_df[save_cols].to_csv(
        f"{OUT_DIR}two_model_deficit_results_NB.csv", index=False)
    print(f"\nResults -> {OUT_DIR}two_model_deficit_results_NB.csv")
    print(
        f"BH-FDR alpha={FDR_ALPHA}: "
        f"{int(results_df['sig_bh'].sum())}/{len(results_df)} significant.")

    # -----------------------------------------------------------------------
    # FOREST PLOT  — unchanged from working script
    # -----------------------------------------------------------------------
    plot_df = results_df.sort_values(
        "deficit_pct", ascending=True).reset_index(drop=True)
    y_pos  = np.arange(len(plot_df))
    has_ci = plot_df["ci_lo"].notna().all()

    colors = []
    for _, row in plot_df.iterrows():
        if not has_ci:
            colors.append("#4a7298")
        elif bool(row["sig_bh"]):
            colors.append("#823038")
        else:
            colors.append("#888888")

    fig, ax = plt.subplots(
        figsize=(7, max(4, len(plot_df) * 0.55 + 1.5)))

    if has_ci:
        for i, row in plot_df.iterrows():
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                    color=colors[i], linewidth=1.4, zorder=1)

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
        fontsize=11, fontweight="bold")

    if has_ci:
        ax.legend(
            handles=[
                mpatches.Patch(color="#823038",
                               label=f"BH-FDR q <= {FDR_ALPHA}"),
                mpatches.Patch(color="#888888", label="not significant"),
            ],
            loc="lower right", fontsize=9, frameon=False)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"θ={row['theta_a']:.1f}" for _, row in plot_df.iterrows()],
        fontsize=7, color="#666666")
    ax2.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}forest_plot_two_model_deficit_NB.png",
        dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Forest plot -> {OUT_DIR}forest_plot_two_model_deficit_NB.png")

    # -----------------------------------------------------------------------
    # HISTORICAL BURDEN  ← new section
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
        bdf.to_csv(
            f"{OUT_DIR}historical_burden_{ind}.csv", index=False)

        burden_rows.append({
            "indicator":       ind,
            "label":           INDICATOR_LABELS.get(ind, ind),
            "total_mu_a":      float(np.nansum(mu_a)),
            "total_mu_b":      float(np.nansum(mu_b)),
            "deficit_pct":     res["deficit_pct"],
            "hot_deficit_pct": float(
                100.0 * (
                    mu_a[nb_data[WBGT_VAR].values > REFERENCE_WBGT].sum()
                    - mu_b[nb_data[WBGT_VAR].values > REFERENCE_WBGT].sum()
                ) / mu_b[nb_data[WBGT_VAR].values > REFERENCE_WBGT].sum()
                if (nb_data[WBGT_VAR].values > REFERENCE_WBGT).any()
                else np.nan
            ),
        })

    pd.DataFrame(burden_rows).to_csv(
        f"{OUT_DIR}historical_burden_summary.csv", index=False)

    # Per-month deficit bar charts
    n_ind  = len(fitted)
    n_cols = 3
    n_rows = int(np.ceil(n_ind / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows), sharex=True)
    af = axes.flatten() if n_ind > 1 else [axes]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    for idx, ind in enumerate(fitted):
        ax  = af[idx]
        csv = f"{OUT_DIR}historical_burden_{ind}.csv"
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        monthly = (df.groupby("month")
                   .agg(mu_a=("mu_a", "sum"), mu_b=("mu_b", "sum"))
                   .reindex(range(1, 13)))
        monthly["pct"] = (
            (monthly["mu_a"] - monthly["mu_b"])
            / monthly["mu_b"] * 100
        ).fillna(0)
        bar_c = ["#823038" if p < 0 else "#2a78d6"
                 for p in monthly["pct"]]
        ax.bar(range(12), monthly["pct"], color=bar_c, alpha=0.8)
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
        "Two-model deficit by calendar month\n"
        "(mu_A − mu_B) / mu_B × 100;  red = heat reduced services",
        fontsize=10, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}deficit_by_month.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Time-series panel
    nc = 3
    nr = int(np.ceil(n_ind / nc))
    fig, axes = plt.subplots(
        nr, nc, figsize=(5.5 * nc, 3.5 * nr), sharex=True)
    af = axes.flatten() if n_ind > 1 else [axes]

    for idx, ind in enumerate(fitted):
        ax  = af[idx]
        csv = f"{OUT_DIR}historical_burden_{ind}.csv"
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv, parse_dates=["date"])
        m  = (df.groupby("date")
              .agg(obs=("y_int", "sum"),
                   mu_a=("mu_a",  "sum"),
                   mu_b=("mu_b",  "sum"))
              .sort_index())
        ax.plot(m.index, m.mu_b, color="#2a78d6", lw=1.0, ls="--",
                alpha=0.8, label="Model B (no weather)")
        ax.plot(m.index, m.obs,  color="#333",    lw=1.0,
                label="Observed")
        ax.fill_between(m.index, m.mu_a, m.mu_b,
                        where=m.mu_a < m.mu_b,
                        color="#823038", alpha=0.25,
                        label="Heat deficit")
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
    plt.savefig(
        f"{OUT_DIR}timeseries_burden.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {OUT_DIR}historical_burden_summary.csv")

    # -----------------------------------------------------------------------
    # FORWARD PROJECTIONS  ← new section
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
        coefs       = nb_coefficient_lookup(res["_model_a"])

        spline_beta = np.array(
            [coefs.get(c, 0.0) for c in spline_cols], dtype=float)

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
                        .merge(hist_clim,   on=["facility", "month"],
                               how="inner")
                        .merge(hist_counts, on=["facility", "month"],
                               how="inner")
                        .merge(hist_mu_b,   on=["facility", "month"],
                               how="inner"))
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

                eta_h = basis_h @ spline_beta
                eta_p = basis_p @ spline_beta
                irrs  = np.exp(eta_p - eta_h)

                proj["mu_a_proj"] = proj["baseline"] * irrs
                proj["mu_b_proj"] = proj["mu_b_hist"]

                sum_a_proj   = float(proj["mu_a_proj"].sum())
                sum_b_proj   = float(proj["mu_b_proj"].sum())
                deficit_proj = (
                    (sum_a_proj - sum_b_proj) / sum_b_proj * 100
                    if sum_b_proj > 0 else np.nan)
                delta_deficit = deficit_proj - deficit_hist

                wd = (proj["wbgt_proj"] - proj["wbgt_hist"]).mean()
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
                    f"{OUT_DIR}projection_{ind}_{ssp}_{tier}.csv",
                    index=False)
                all_proj.append({
                    "indicator":      ind,
                    "ssp":            ssp,
                    "tier":           tier,
                    "mean_wbgt_diff": float(wd),
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
    # TLO LOOKUP TABLE  ← new section
    # -----------------------------------------------------------------------
    print("\nTLO lookup tables...")
    tlo_rows = []

    for res in all_results:
        ind         = res["indicator"]
        design_info = res["_design_info"]
        wbgt_shift  = res["_wbgt_shift"]
        spline_cols = res["_spline_cols"]
        coefs       = nb_coefficient_lookup(res["_model_a"])
        spline_beta = np.array(
            [coefs.get(c, 0.0) for c in spline_cols], dtype=float)

        ref_c     = REFERENCE_WBGT - wbgt_shift
        all_pts_c = np.append(WBGT_GRID - wbgt_shift, ref_c)
        basis     = np.asarray(
            patsy.build_design_matrices(
                [design_info], {"x": all_pts_c})[0],
            dtype=float)
        ref_row = basis[-1]

        for i, w in enumerate(WBGT_GRID):
            eta     = float((basis[i] - ref_row) @ spline_beta)
            irr     = np.exp(eta)
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
            f"{OUT_DIR}tlo_disruption_curves.png",
            dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  -> {OUT_DIR}tlo_wbgt_lookup.csv")

    # -----------------------------------------------------------------------
    # DISTRICT AGGREGATION  ← new section
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
            obs=("y_int", "sum"),
            mu_a=("mu_a",  "sum"),
            mu_b=("mu_b",  "sum"),
        ).reset_index()
        d["deficit_pct"] = (d["mu_a"] - d["mu_b"]) / d["mu_b"] * 100
        d["indicator"]   = ind
        all_dist.append(d)
        d.to_csv(
            f"{OUT_DIR}district_burden_{ind}.csv", index=False)
        print(f"  {ind}: {len(d)} districts, "
              f"mean deficit {d['deficit_pct'].mean():.2f}%")

    if all_dist:
        pd.concat(all_dist, ignore_index=True).to_csv(
            f"{OUT_DIR}district_burden_all.csv", index=False)
        print(f"  -> {OUT_DIR}district_burden_all.csv")

    print(f"\nAll outputs in {OUT_DIR}")
    print("Done.")
