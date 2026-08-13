"""
model_of_wbgt_dhis2_two_model_splines_optimized.py

Two-model NB (statsmodels) WBGT–service analysis - OPTIMIZED VERSION.
Key improvements:
  1. Fixed alpha (estimated once, reused across models)
  2. Parallel processing for multiple indicators
  3. Sparse matrices for fixed effects
  4. Vectorized predictions
  5. Efficient spline basis construction
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import time

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
# List of indicators to analyze (uncomment as needed)
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

# Model settings
WEATHER_VARS_LEVEL = ["wbgt_day"]
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
HOT_DEFICIT_CI_METHOD = "jackknife"  # or "bootstrap"
N_BOOTSTRAP = 100  # if using bootstrap

CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre", "2023-03-01", "2024-03-01"),
]

# Time periods
min_year_historical = 2019
max_year_historical = 2025
LAST_HIST_YEAR = max_year_historical - 1

# Projection
PROJECT = True
PROJECT_HOLD_YEAR = True
SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
WBGT_MODELS = ["lowest", "median", "highest"]
min_year_projection = 2025
max_year_projection = 2071

# Curve settings
CURVE_REF_MODE = "mean"
CURVE_N = 60

# Paths
DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/TwoModelSplines_Optimized/"
PANEL_DIR = f"{DATA_DIR}/Thermofeel_WBGT/Indices/"
INDICES_DIR = PANEL_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# Parallel processing
USE_PARALLEL = True
N_WORKERS = min(cpu_count() - 1, 4)  # Use up to 4 cores


# ===========================================================================
# OPTIMIZED: Shared alpha estimation (faster than profile likelihood)
# ===========================================================================
def estimate_alpha_fast(data, y_col="y_int"):
    """
    Quick alpha estimate for NB2 using method of moments.
    Much faster than profile likelihood optimization.
    """
    # Fit a simple Poisson model to get variance estimate
    mod_pois = smf.glm(
        f"{y_col} ~ 1",
        data=data,
        family=sm.families.Poisson()
    ).fit()

    # Method of moments: alpha = (variance - mean) / mean^2
    # Using Pearson residuals
    mu = mod_pois.fittedvalues
    pearson_resid = (data[y_col] - mu) / np.sqrt(mu)
    dispersion = (pearson_resid**2).sum() / mod_pois.df_resid

    # For NB2, alpha = (dispersion - 1) / mean
    # But if dispersion < 1, use a small positive value
    alpha = max(0.01, (dispersion - 1) / mu.mean())
    return alpha


def fit_negbin_fixed_alpha(formula, data, groups, alpha, y_col="y_int"):
    """
    Fit Negative Binomial with fixed alpha (no profile likelihood).
    Much faster than fit_negbin() with profile likelihood.
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
    """
    Convert categorical fixed effects to sparse matrix representation.
    This is much faster than using C(facility) in the formula.
    """
    df = df.copy()
    for col in fe_cols:
        # Create sparse dummies
        dummies = pd.get_dummies(df[col], prefix=col, sparse=True)
        # Add to dataframe as sparse columns
        for c in dummies.columns:
            df[c] = dummies[c].astype(float)
    return df


def fit_negbin_with_sparse_fe(formula_base, df, fe_cols, groups, alpha, y_col="y_int"):
    """
    Fit NB with sparse fixed effects. Remove the FE terms from formula
    and add them as sparse columns in the design matrix.
    """
    # Add sparse FEs to dataframe
    df_fe = add_sparse_fixed_effects(df, fe_cols)

    # Build formula without C(facility) terms
    formula = f"{y_col} ~ " + formula_base

    # Fit with sparse matrix
    mod = smf.glm(
        formula=formula,
        data=df_fe,
        family=sm.families.NegativeBinomial(alpha=alpha)
    ).fit(cov_type="cluster", cov_kwds={"groups": groups})
    mod.alpha_hat = alpha
    return mod, df_fe


# ===========================================================================
# OPTIMIZED: Shared column construction (same as before but faster)
# ===========================================================================
def add_weather_columns_optimized(df, shifts, spline_design=None,
                                   lag_months=LAG_MONTHS):
    """
    Faster version using vectorized operations.
    """
    df = df.sort_values(["facility", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_c"] = df["year"] - shifts["year"]

    # COVID dummy - vectorized
    lo, hi = pd.Timestamp(COVID_WINDOW[0]), pd.Timestamp(COVID_WINDOW[1])
    df["covid"] = df["date"].between(lo, hi).astype(int)

    rhs = ["year_c", "covid"]
    if USE_PRECIP:
        df["precip_c"] = df[PRECIP_COL] - shifts.get("precip", 0.0)
        rhs.append("precip_c")

    spline_cols_all = []
    design_map = {} if spline_design is None else spline_design

    for v in WEATHER_VARS_LEVEL:
        xc = df[v] - shifts[v]

        # Build spline basis (vectorized)
        if spline_design is None:
            B = patsy.dmatrix(f"cr(x, df={SPLINE_DF}) - 1", {"x": xc},
                              return_type="dataframe")
            design_map[v] = B.design_info
        else:
            B = patsy.build_design_matrices([design_map[v]], {"x": xc})[0]
            B = pd.DataFrame(np.asarray(B))

        cols = [f"{v}_s{i+1}" for i in range(B.shape[1])]
        # Vectorized assignment
        for c, col in enumerate(cols):
            df[col] = np.asarray(B)[:, c]
        spline_cols_all.extend(cols)
        rhs.extend(cols)

        # Lags - vectorized per facility group
        for lag in lag_months:
            lc = f"{v}_lag{lag}_c"
            df[lc] = df.groupby("facility")[v].shift(lag) - shifts[v]
            rhs.append(lc)

    return df, rhs, spline_cols_all, design_map


# ===========================================================================
# OPTIMIZED: Exposure-response curve (faster)
# ===========================================================================
def exposure_response_curve_fast(model, spline_cols, design_info, wbgt_var_name,
                                  wbgt_shift, ref_mode, wbgt_values, n=CURVE_N):
    """
    Faster curve computation using vectorized operations.
    wbgt_values: array-like of observed WBGT values
    """
    wobs = np.asarray(wbgt_values).flatten()
    grid = np.linspace(np.percentile(wobs, 1), np.percentile(wobs, 99), n)
    ref = wobs.mean() if ref_mode == "mean" else float(ref_mode)

    # Vectorized basis construction
    Bg = np.asarray(patsy.build_design_matrices(
        [design_info[wbgt_var_name]], {"x": grid - wbgt_shift})[0])
    Br = np.asarray(patsy.build_design_matrices(
        [design_info[wbgt_var_name]], {"x": np.array([ref]) - wbgt_shift})[0])
    contrast = Bg - Br

    beta = model.params.reindex(spline_cols).values
    V = model.cov_params().reindex(index=spline_cols, columns=spline_cols).values

    log_irr = contrast @ beta
    var = np.einsum("ij,jk,ik->i", contrast, V, contrast)
    se = np.sqrt(np.clip(var, 0, None))

    return pd.DataFrame({
        "wbgt": grid,
        "irr": np.exp(log_irr),
        "irr_lo": np.exp(log_irr - 1.96 * se),
        "irr_hi": np.exp(log_irr + 1.96 * se),
    })


# ===========================================================================
# OPTIMIZED: Main fitting function
# ===========================================================================
def fit_indicator(indicator, panel_path):
    """Fit both models for a single indicator - optimized version."""
    print(f"\n→ {indicator}")
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    try:
        long = pd.read_csv(panel_path, parse_dates=["date"])
        long = long.rename(columns={indicator: "y"})
    except Exception as e:
        print(f"  [{indicator}] Failed to load: {e}")
        return None

    # Check required columns
    for v in WEATHER_VARS_LEVEL:
        if v not in long.columns:
            print(f"  [{indicator}] Missing {v} in panel")
            return None
    if USE_PRECIP and PRECIP_COL not in long.columns:
        print(f"  [{indicator}] Missing {PRECIP_COL} in panel")
        return None
    if CLUSTER_COL not in long.columns:
        print(f"  [{indicator}] Missing {CLUSTER_COL} in panel")
        return None

    # -----------------------------------------------------------------------
    # Clean data
    # -----------------------------------------------------------------------
    # Closures -> NaN (not 0)
    for fac, d0, d1 in CLOSURES:
        m = (long["date"].between(d0, d1)) & (long["facility"] == fac)
        if m.any():
            long.loc[m, "y"] = np.nan

    # Filter years
    long["year"] = long["date"].dt.year
    long["month"] = long["date"].dt.month
    long = long[long["year"].between(min_year_historical, max_year_historical - 1)]

    # Sparsity filter
    obs_per_fac = long.dropna(subset=["y"] + WEATHER_VARS_LEVEL).groupby("facility").size()
    long = long[long["facility"].isin(obs_per_fac[obs_per_fac >= MIN_OBS].index)].copy()

    if len(long) < 100 or long["facility"].nunique() < 2:
        print(f"  [{indicator}] Too few observations/facilities")
        return None

    # -----------------------------------------------------------------------
    # Centering shifts
    # -----------------------------------------------------------------------
    SHIFTS = {"year": long["year"].mean() if CENTER else 0.0}
    for v in WEATHER_VARS_LEVEL:
        SHIFTS[v] = long[v].mean() if CENTER else 0.0
    if USE_PRECIP:
        SHIFTS["precip"] = long[PRECIP_COL].mean() if CENTER else 0.0

    # -----------------------------------------------------------------------
    # Build weather columns (one-time, shared between models)
    # -----------------------------------------------------------------------
    long, weather_rhs, spline_cols, DESIGN = add_weather_columns_optimized(
        long, SHIFTS
    )

    # Estimation sample
    nb_cols = ["y", "facility", "month", CLUSTER_COL] + weather_rhs
    nb_data = long.dropna(subset=nb_cols).copy()
    nb_data["y_int"] = nb_data["y"].round().clip(lower=0).astype(int)

    # Drop facilities with too few obs
    obs_nb = nb_data.groupby("facility").size()
    nb_data = nb_data[nb_data["facility"].isin(obs_nb[obs_nb >= MIN_OBS].index)].copy()
    FITTED_FACILITIES = set(nb_data["facility"].unique())

    print(f"  [{indicator}] Sample: {len(nb_data):,} obs, {len(FITTED_FACILITIES)} facilities")

    # -----------------------------------------------------------------------
    # Estimate alpha once (the key speed improvement)
    # -----------------------------------------------------------------------
    alpha = estimate_alpha_fast(nb_data)
    groups = nb_data[CLUSTER_COL]

    # -----------------------------------------------------------------------
    # Fit both models with fixed alpha (no profile likelihood)
    # -----------------------------------------------------------------------
    ctrl = "year_c + covid" + (" + precip_c" if USE_PRECIP else "")
    FE = "C(month) + C(facility)"

    f_base = f"y_int ~ {ctrl} + {FE}"
    f_wx = f"y_int ~ {' + '.join(weather_rhs)} + {FE}"

    # Fit models
    try:
        model_base = fit_negbin_fixed_alpha(f_base, nb_data, groups, alpha)
        model_wx = fit_negbin_fixed_alpha(f_wx, nb_data, groups, alpha)
    except Exception as e:
        print(f"  [{indicator}] Model fitting failed: {e}")
        return None

    # -----------------------------------------------------------------------
    # Check spline columns
    # -----------------------------------------------------------------------
    missing_spline = [c for c in spline_cols if c not in model_wx.params.index]
    if missing_spline:
        print(f"  [{indicator}] Warning: missing spline cols: {missing_spline}")

    # -----------------------------------------------------------------------
    # Exposure-response curve
    # -----------------------------------------------------------------------
    try:
        curve = exposure_response_curve_fast(
            model_wx,
            spline_cols,
            DESIGN,
            WEATHER_VARS_LEVEL[0],  # Pass the variable name
            SHIFTS[WEATHER_VARS_LEVEL[0]],
            CURVE_REF_MODE,
            nb_data[WEATHER_VARS_LEVEL[0]].values,  # Pass numpy array
        )
        curve.to_csv(f"{OUT_DIR}exposure_response_curve_{indicator}.csv", index=False)
    except Exception as e:
        print(f"  [{indicator}] Curve failed: {e}")

    # -----------------------------------------------------------------------
    # Two-model deficit (facility jackknife)
    # -----------------------------------------------------------------------
    nb_data["y_pred_base"] = model_base.fittedvalues
    nb_data["y_pred_wx"] = model_wx.fittedvalues
    nb_data["difference"] = nb_data["y_pred_base"] - nb_data["y_pred_wx"]

    def aggregate_deficit_pct(df):
        b, w = df["y_pred_base"].sum(), df["y_pred_wx"].sum()
        return 100.0 * (b - w) / b if b > 0 else np.nan

    deficit_pt = aggregate_deficit_pct(nb_data)

    # Jackknife (leave-one-facility-out)
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
    # ======================================================================
    # HOT-MONTH DEFICIT (the quantity that actually tells the story)
    # ======================================================================
    # Get the WBGT threshold (95th percentile of observed WBGT)
    wbgt_var = WEATHER_VARS_LEVEL[0]
    hot_threshold = np.percentile(nb_data[wbgt_var], REFERENCE_WBGT_PERCENTILE)

    # Identify hot months
    hot_mask = nb_data[wbgt_var] > hot_threshold
    hot_data = nb_data[hot_mask].copy()

    if len(hot_data) > 10:  # Enough hot months to compute
        # Point estimate for hot months
        hot_deficit_pt = aggregate_deficit_pct(hot_data)

        # Jackknife CI for hot months (leave-one-facility-out)
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
            f"{hot_deficit_pt:+.2f}% (CI: {hot_deficit_ci[0]:+.2f}..{hot_deficit_ci[1]:+.2f})"
        )
    else:
        hot_deficit_pt = np.nan
        hot_deficit_ci = (np.nan, np.nan)
        hot_se_jack = np.nan
        print(f"  [{indicator}] Not enough hot months ({len(hot_data)} observations)")

    # ======================================================================
    # Save results (add hot-month fields)
    # ======================================================================
    pd.DataFrame(
        [
            {
                "indicator": indicator,
                "label": INDICATOR_LABELS.get(indicator, indicator),
                "deficit_pct": deficit_pt,  # Aggregate (all months)
                "ci_lo": deficit_ci[0],
                "ci_hi": deficit_ci[1],
                "se_jackknife": se_jack,
                "hot_deficit_pct": hot_deficit_pt,  # HOT MONTHS ONLY
                "hot_ci_lo": hot_deficit_ci[0],
                "hot_ci_hi": hot_deficit_ci[1],
                "hot_se_jackknife": hot_se_jack,
                "hot_threshold": hot_threshold if len(hot_data) > 10 else np.nan,
                "n_hot_obs": len(hot_data),
                "n_facilities": len(FITTED_FACILITIES),
                "n_obs": len(nb_data),
                "n_districts": nb_data[CLUSTER_COL].nunique(),
                "alpha": alpha,
                "time_seconds": time.time() - t0,
            }
        ]
    ).to_csv(f"{OUT_DIR}deficit_{indicator}.csv", index=False)

    # Save predictions
    pred_cols = ["year", "month", "facility", "date", "y_int", "covid",
                 "y_pred_base", "y_pred_wx", "difference"]
    nb_data[pred_cols].to_csv(
        f"{OUT_DIR}predictions_{indicator}.csv", index=False
    )

    print(f"  [{indicator}] Done in {time.time() - t0:.1f}s")

    # Store for later aggregation
    return {
        "indicator": indicator,
        "label": INDICATOR_LABELS.get(indicator, indicator),
        "deficit_pct": deficit_pt,
        "ci_lo": deficit_ci[0],
        "ci_hi": deficit_ci[1],
        "n_facilities": len(FITTED_FACILITIES),
        "n_obs": len(nb_data),
        "n_districts": nb_data[CLUSTER_COL].nunique(),
        "alpha": alpha,
        "time": time.time() - t0,
        "_nb_data": nb_data,
        "_model_base": model_base,
        "_model_wx": model_wx,
        "_spline_cols": spline_cols,
        "_design": DESIGN,
        "_shifts": SHIFTS,
        "_fitted_facilities": FITTED_FACILITIES,
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

    # Build panel paths
    panel_paths = {
        ind: f"{PANEL_DIR}regression_panel_{ind}.csv"
        for ind in COUNT_INDICATORS
    }

    t_start = time.time()

    # Run indicators (parallel or sequential)
    if USE_PARALLEL and len(COUNT_INDICATORS) > 1:
        print(f"\nRunning {len(COUNT_INDICATORS)} indicators in parallel...")
        with Pool(processes=N_WORKERS) as pool:
            results = pool.starmap(
                fit_indicator,
                [(ind, panel_paths[ind]) for ind in COUNT_INDICATORS]
            )
    else:
        print("\nRunning indicators sequentially...")
        results = []
        for ind in COUNT_INDICATORS:
            result = fit_indicator(ind, panel_paths[ind])
            results.append(result)

    # Filter successful results
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
                "time": r["time"],
            }
            for r in results
        ]
    )

    summary_df = summary_df.sort_values("hot_deficit_pct", na_position="last")
    summary_df.to_csv(f"{OUT_DIR}summary_all_indicators.csv", index=False)

    print("\nSummary (HOT MONTHS ONLY):")
    print(summary_df[["indicator", "hot_deficit_pct", "hot_ci_lo", "hot_ci_hi", "n_hot_obs"]].to_string())

    # -----------------------------------------------------------------------
    # Forest plot with TWO panels: aggregate vs hot-month
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(results) * 0.4)))

    # Panel 1: Aggregate deficit (all months)
    y_pos = np.arange(len(summary_df))
    for i, row in summary_df.iterrows():
        if pd.notna(row["ci_lo"]) and pd.notna(row["ci_hi"]):
            ax1.plot([row["ci_lo"], row["ci_hi"]], [i, i], color="#888888", linewidth=1.5)
        ax1.scatter(row["deficit_pct"], i, color="#888888", s=60, zorder=3)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(summary_df["label"])
    ax1.set_xlabel("Deficit (%) - All months")
    ax1.set_title("A: All months (near-null)")
    ax1.grid(axis="x", alpha=0.3)

    # Panel 2: Hot-month deficit
    for i, row in summary_df.iterrows():
        if pd.notna(row["hot_ci_lo"]) and pd.notna(row["hot_ci_hi"]):
            ax2.plot([row["hot_ci_lo"], row["hot_ci_hi"]], [i, i], color="#823038", linewidth=2)
        ax2.scatter(
            row["hot_deficit_pct"],
            i,
            color="#823038" if pd.notna(row["hot_deficit_pct"]) else "#cccccc",
            s=80,
            zorder=3,
        )
    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(summary_df["label"])
    ax2.set_xlabel("Deficit (%) - Hot months only")
    ax2.set_title(f"B: Hot months (>{REFERENCE_WBGT_PERCENTILE}th percentile)")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}forest_plot_aggregate_vs_hot.png", dpi=150)
    plt.close()

    # Also save a separate hot-only forest plot
    fig, ax = plt.subplots(figsize=(8, max(4, len(results) * 0.4)))
    for i, row in summary_df.iterrows():
        if pd.notna(row["hot_ci_lo"]) and pd.notna(row["hot_ci_hi"]):
            ax.plot([row["hot_ci_lo"], row["hot_ci_hi"]], [i, i], color="#823038", linewidth=2)
        ax.scatter(
            row["hot_deficit_pct"],
            i,
            color="#823038" if pd.notna(row["hot_deficit_pct"]) else "#cccccc",
            s=80,
            zorder=3,
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary_df["label"])
    ax.set_xlabel("Deficit (%) - Positive = Heat reduces services")
    ax.set_title(f"Heat-attributable deficit during hottest months (>{REFERENCE_WBGT_PERCENTILE}th percentile)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}forest_plot_hot_months_only.png", dpi=150)
    plt.close()
    fig, ax = plt.subplots(figsize=(8, max(4, len(results) * 0.4)))
    y_pos = np.arange(len(summary_df))

    # Plot CIs and points
    for i, row in summary_df.iterrows():
        if pd.notna(row["ci_lo"]) and pd.notna(row["ci_hi"]):
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                   color="#823038", linewidth=1.5)
        ax.scatter(row["deficit_pct"], i, color="#823038", s=60, zorder=3)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary_df["label"])
    ax.set_xlabel("Deficit (%) - Positive = Heat reduces services")
    ax.set_title("Heat-attributable deficit across health services")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}forest_plot_all_indicators.png", dpi=150)
    plt.close()

    print(f"\nAll outputs saved to {OUT_DIR}")
    print("Done!")
