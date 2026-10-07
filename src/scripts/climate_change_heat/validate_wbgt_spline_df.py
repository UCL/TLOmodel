#!/usr/bin/env python3
"""
validate_wbgt_spline_df.py

Corrected forward (temporal out-of-sample) validation for choosing the
WBGT spline degrees of freedom in the current Poisson fixed-effects model.

Primary question
----------------
Does extra spline flexibility improve prediction of genuinely held-out years?

Candidate spline dfs:
    3, 4, 6, 8, 12, 20

Validation folds:
    train <= 2021 -> test 2022
    train <= 2022 -> test 2023
    train <= 2023 -> test 2024

Key safeguards
--------------
1. Facility eligibility is calculated separately inside each training fold.
   The coverage requirement is relative to the number of months actually
   available in that training period, NOT the full 2016-2024 period.
2. Spline knots/bounds are learned from training data only.
3. Validation WBGT is clipped to the TRAINING WBGT support before evaluating
   the spline, avoiding patsy out-of-bounds errors/extrapolation.
4. Centering values are learned from training only.
5. Lagged WBGT is constructed chronologically, so December training exposure
   can correctly feed January validation exposure.
6. Validation is restricted to facilities observed/retained in training.
7. Future year and zone×year fixed effects are frozen at the final training
   year because their validation-year levels do not exist in the fitted model.
8. If winsorisation is enabled in the production model, caps are estimated
   from TRAINING data only and then applied to train and validation.
9. All spline dfs are evaluated on the same rows within each indicator/fold.
10. Model selection is based primarily on held-out mean Poisson deviance.
    Lower is better. df=3 is the parsimonious reference.

This script imports the current production model but DOES NOT modify it.
"""

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIG
# ============================================================================

SOURCE_SCRIPT = Path(
    "/Users/rachelmurray-watson/PycharmProjects/TLOmodel/"
    "src/scripts/climate_change_heat/asborbed_fe_pois.py"
)

VALIDATION_YEARS = [2022, 2023, 2024]
SPLINE_DFS = [3, 4, 6, 8, 12, 20]

# Optional: evaluate performance specifically in the hottest validation rows,
# using a threshold estimated from TRAINING data only.
TAIL_PERCENTILE = 95

# A tiny numerical floor for Poisson predictions.
MU_FLOOR = 1e-12

# ============================================================================
# LOAD CURRENT PRODUCTION MODEL
# ============================================================================

if not SOURCE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Production script not found:\n{SOURCE_SCRIPT}\n"
        "Edit SOURCE_SCRIPT at the top of this validation script."
    )

spec = importlib.util.spec_from_file_location("wbgt_prod_cv", SOURCE_SCRIPT)
prod = importlib.util.module_from_spec(spec)
sys.modules["wbgt_prod_cv"] = prod
spec.loader.exec_module(prod)

CV_DIR = Path(prod.OUT_DIR) / "wbgt_df_validation"
CV_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# METRICS
# ============================================================================

def poisson_deviance(y, mu):
    """Total Poisson deviance."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    ok = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
    y = y[ok]
    mu = mu[ok]

    if len(y) == 0:
        return np.nan

    term = np.zeros_like(y)
    pos = y > 0
    term[pos] = y[pos] * np.log(y[pos] / mu[pos])

    return float(2.0 * np.sum(term - (y - mu)))


def score_predictions(y, mu):
    """Prediction metrics. Primary metric = mean Poisson deviance."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    ok = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
    y = y[ok]
    mu = mu[ok]

    if len(y) == 0:
        return {
            "n": 0,
            "poisson_deviance": np.nan,
            "mean_poisson_deviance": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "aggregate_error_pct": np.nan,
        }

    dev = poisson_deviance(y, mu)
    sy = float(y.sum())

    return {
        "n": int(len(y)),
        "poisson_deviance": dev,
        "mean_poisson_deviance": dev / len(y),
        "mae": float(np.mean(np.abs(y - mu))),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "aggregate_error_pct": (
            float(100.0 * (mu.sum() - sy) / sy) if sy > 0 else np.nan
        ),
    }


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_raw_indicator(indicator):
    """
    Load data and apply only rules that do not learn anything from future data.
    In particular: NO full-history facility coverage filtering here.
    """
    df = prod.load_indicator_panel(indicator, prod.PANEL_DIR)
    df = prod.apply_hard_ceilings(df, indicator)
    df = df.rename(columns={indicator: "y"})

    df["date"] = pd.to_datetime(df["date"])

    # Match production closure handling.
    for fac, d0, d1 in prod.CLOSURES:
        mask = (
            (df["facility"] == fac)
            & df["date"].between(pd.Timestamp(d0), pd.Timestamp(d1))
        )
        df.loc[mask, "y"] = 0

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    min_year = prod.MIN_YEAR_BY_INDICATOR.get(
        indicator, prod.min_year_historical
    )

    df = df[
        df["year"].between(min_year, prod.LAST_HIST_YEAR)
    ].copy()

    return df


def fold_min_obs(indicator, train):
    """
    Training-fold version of production get_min_obs().

    Coverage is relative to the months that could actually have been observed
    by the end of THIS training fold, rather than 2016-2024.
    """
    base_min = int(prod.MIN_OBS)
    coverage = float(getattr(prod, "MIN_OBS_COVERAGE", 0.0))

    start_year = prod.MIN_YEAR_BY_INDICATOR.get(
        indicator, prod.min_year_historical
    )

    if train.empty:
        return base_min

    train_end = int(train["year"].max())

    # Number of calendar months theoretically available in this fold.
    possible_months = max(0, (train_end - start_year + 1) * 12)

    dynamic_min = int(np.floor(possible_months * coverage))
    return max(base_min, dynamic_min)


def retain_training_facilities(indicator, train, val):
    """Apply facility coverage using TRAINING observations only."""
    min_obs = fold_min_obs(indicator, train)

    # Count rows that can potentially enter the WBGT model.
    count_cols = ["y", prod.WBGT_VAR]
    if prod.USE_PRECIP:
        count_cols.append(prod.PRECIP_COL)

    eligible_rows = train.dropna(subset=count_cols)
    n_by_fac = eligible_rows.groupby("facility").size()
    keep = set(n_by_fac[n_by_fac >= min_obs].index)

    train = train[train["facility"].isin(keep)].copy()
    val = val[val["facility"].isin(keep)].copy()

    return train, val, min_obs


def training_only_winsorise(indicator, train, val):
    """
    If production winsorisation is disabled, this is a no-op.

    If enabled, estimate each facility's upper cap using TRAINING y only,
    then apply that fixed cap to both training and validation.
    """
    if not getattr(prod, "WINSORIZE", False):
        return train, val

    q = prod.WINSORIZE_BY_INDICATOR.get(
        indicator, prod.WINSORIZE_DEFAULT
    )

    if q is None or q >= 1:
        return train, val

    train = train.copy()
    val = val.copy()

    caps = (
        train.groupby("facility")["y"]
        .quantile(q)
        .rename("_train_y_cap")
    )

    train = train.join(caps, on="facility")
    val = val.join(caps, on="facility")

    train["y"] = np.minimum(train["y"], train["_train_y_cap"])
    val["y"] = np.minimum(val["y"], val["_train_y_cap"])

    train = train.drop(columns="_train_y_cap")
    val = val.drop(columns="_train_y_cap")

    return train, val


# ============================================================================
# CONTROLS / FIXED-EFFECT LABELS
# ============================================================================

def training_shifts(train):
    shifts = {
        prod.WBGT_VAR: (
            float(train[prod.WBGT_VAR].mean())
            if prod.CENTER else 0.0
        )
    }

    if prod.USE_PRECIP:
        shifts["precip"] = (
            float(train[prod.PRECIP_COL].mean())
            if prod.CENTER else 0.0
        )

    return shifts


def add_lags_and_controls(sequence, shifts):
    """
    Construct controls on the continuous sequence through the validation year.
    This preserves the correct lag crossing from training -> validation.
    """
    out = sequence.sort_values(["facility", "date"]).copy()

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month

    lo = pd.Timestamp(prod.COVID_WINDOW[0])
    hi = pd.Timestamp(prod.COVID_WINDOW[1])
    out["covid"] = out["date"].between(lo, hi).astype(int)

    if prod.USE_PRECIP:
        out["precip_c"] = (
            out[prod.PRECIP_COL] - shifts["precip"]
        )

    lag_cols = []
    if prod.SA_LAG:
        for lag in prod.LAG_MONTHS:
            col = f"{prod.WBGT_VAR}_lag{lag}_cv"
            out[col] = (
                out.groupby("facility")[prod.WBGT_VAR].shift(lag)
                - shifts[prod.WBGT_VAR]
            )
            lag_cols.append(col)

    return out, lag_cols


def add_training_year_controls(train, val, train_end):
    """
    Explicit year controls learned only from training.

    The held-out year is assigned the reference level (last training year),
    analogous to holding the time effect fixed when forecasting.
    """
    train = train.copy()
    val = val.copy()

    train_years = sorted(int(y) for y in train["year"].dropna().unique())
    reference = int(train_end)

    cols = []
    for year in train_years:
        if year == reference:
            continue

        col = f"cv_year_fe_{year}"
        train[col] = (train["year"] == year).astype(int)
        val[col] = 0
        cols.append(col)

    return train, val, cols


def add_zone_year_fe(train, val, train_end):
    """
    Training zone×year FE. Validation zone×year is frozen at train_end.
    """
    train = train.copy()
    val = val.copy()

    if prod.USE_ZONE_YEAR_FE:
        train["cv_zone_year"] = (
            train[prod.ZONE_COL].astype(str)
            + "_"
            + train["year"].astype(int).astype(str)
        )

        val["cv_zone_year"] = (
            val[prod.ZONE_COL].astype(str)
            + "_"
            + str(int(train_end))
        )

    return train, val


# ============================================================================
# SPLINE DESIGN: TRAIN ONLY, VALIDATION CLIPPED TO TRAIN SUPPORT
# ============================================================================

def build_training_spline(train, spline_df, wbgt_shift):
    """
    Build spline on training data only and retain patsy design_info.
    """
    out = train.copy()

    x = out[prod.WBGT_VAR].to_numpy(dtype=float) - wbgt_shift

    lower = float(np.nanmin(x))
    upper = float(np.nanmax(x))

    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(
            f"Invalid training WBGT support: {lower} to {upper}"
        )

    formula = (
        f"bs(x, df={int(spline_df)}, "
        f"lower_bound={lower}, upper_bound={upper}) - 1"
    )

    Bdf = patsy.dmatrix(
        formula,
        {"x": x},
        return_type="dataframe",
    )

    B = np.asarray(Bdf)
    cols = [
        f"{prod.WBGT_VAR}_cv_s{i + 1}"
        for i in range(B.shape[1])
    ]

    for j, col in enumerate(cols):
        out[col] = B[:, j]

    support = {
        "lower_centered": lower,
        "upper_centered": upper,
        "lower_raw": lower + wbgt_shift,
        "upper_raw": upper + wbgt_shift,
    }

    return out, cols, Bdf.design_info, support


def build_validation_spline(
    val, design_info, spline_cols, wbgt_shift, support
):
    """
    Evaluate the TRAINING spline in validation.

    Values outside the training WBGT support are clipped to the nearest
    training boundary. We record how often this happens.
    """
    out = val.copy()

    x_raw = out[prod.WBGT_VAR].to_numpy(dtype=float)
    x = x_raw - wbgt_shift

    lower = support["lower_centered"]
    upper = support["upper_centered"]

    clipped_mask = np.isfinite(x) & ((x < lower) | (x > upper))
    x_clip = np.clip(x, lower, upper)

    B = np.asarray(
        patsy.build_design_matrices(
            [design_info],
            {"x": x_clip},
        )[0]
    )

    if B.shape[1] != len(spline_cols):
        raise RuntimeError(
            f"Spline design mismatch: got {B.shape[1]} columns; "
            f"expected {len(spline_cols)}."
        )

    for j, col in enumerate(spline_cols):
        out[col] = B[:, j]

    return out, clipped_mask


# ============================================================================
# MODEL FITTING
# ============================================================================

def fit_poisson(rhs, data):
    fe = "facility + month"

    if prod.USE_ZONE_YEAR_FE:
        fe += " + cv_zone_year"

    formula = f"y_int ~ {' + '.join(rhs)} | {fe}"

    return prod.pf.fepois(
        formula,
        data=data,
        vcov={"CRV1": prod.CLUSTER_COL},
    )


def evaluate_one_df(
    indicator,
    train_common,
    val_common,
    shifts,
    year_cols,
    lag_cols,
    spline_df,
    train_end,
):
    """
    Fit one spline df and score held-out predictions.
    """
    train, spline_cols, design_info, support = build_training_spline(
        train_common,
        spline_df=spline_df,
        wbgt_shift=shifts[prod.WBGT_VAR],
    )

    val, clipped_mask = build_validation_spline(
        val_common,
        design_info=design_info,
        spline_cols=spline_cols,
        wbgt_shift=shifts[prod.WBGT_VAR],
        support=support,
    )

    rhs = list(year_cols) + ["covid"]

    if prod.USE_PRECIP:
        rhs.append("precip_c")

    rhs += spline_cols
    rhs += lag_cols

    needed = (
        ["y", "facility", "month", prod.CLUSTER_COL]
        + rhs
        + (["cv_zone_year"] if prod.USE_ZONE_YEAR_FE else [])
    )

    # Because all WBGT values were clipped to valid support, this should be
    # the same common sample for every df.
    train = train.dropna(subset=list(dict.fromkeys(needed))).copy()
    val = val.dropna(subset=list(dict.fromkeys(needed))).copy()

    if len(train) < 100 or len(val) < 20:
        raise RuntimeError(
            f"Too little usable data: train={len(train)}, val={len(val)}"
        )

    train["y_int"] = train["y"].round().clip(lower=0).astype(int)
    val["y_int"] = val["y"].round().clip(lower=0).astype(int)

    model = fit_poisson(rhs, train)

    mu = np.asarray(
        model.predict(newdata=val, type="response"),
        dtype=float,
    )

    y = val["y_int"].to_numpy(dtype=float)

    ok = np.isfinite(mu) & (mu > MU_FLOOR) & np.isfinite(y)

    y_eval = y[ok]
    mu_eval = mu[ok]
    val_eval = val.loc[ok].copy()

    overall = score_predictions(y_eval, mu_eval)

    # Training-only hot threshold.
    hot_threshold = float(
        np.percentile(
            train[prod.WBGT_VAR].dropna(),
            TAIL_PERCENTILE,
        )
    )

    hot = (
        val_eval[prod.WBGT_VAR].to_numpy(dtype=float)
        >= hot_threshold
    )

    hot_score = score_predictions(
        y_eval[hot],
        mu_eval[hot],
    )

    retained = [
        c for c in spline_cols
        if c in model.coef().index
    ]

    row = {
        "indicator": indicator,
        "validation_year": int(train_end + 1),
        "train_end_year": int(train_end),
        "spline_df": int(spline_df),

        "n_train": int(len(train)),
        "n_validation": int(len(val_eval)),
        "n_train_facilities": int(train["facility"].nunique()),
        "n_validation_facilities": int(val_eval["facility"].nunique()),

        "n_spline_terms_requested": int(len(spline_cols)),
        "n_spline_terms_retained": int(len(retained)),

        "train_wbgt_min": support["lower_raw"],
        "train_wbgt_max": support["upper_raw"],
        "fraction_validation_wbgt_clipped": (
            float(np.mean(clipped_mask))
            if len(clipped_mask) else np.nan
        ),

        "mean_poisson_deviance": overall["mean_poisson_deviance"],
        "poisson_deviance": overall["poisson_deviance"],
        "mae": overall["mae"],
        "rmse": overall["rmse"],
        "aggregate_error_pct": overall["aggregate_error_pct"],

        "hot_threshold_train_p95": hot_threshold,
        "n_hot_validation": hot_score["n"],
        "hot_mean_poisson_deviance": hot_score["mean_poisson_deviance"],
        "hot_mae": hot_score["mae"],
        "hot_rmse": hot_score["rmse"],
    }

    pred = val_eval[
        [
            "facility",
            "date",
            "year",
            "month",
            prod.WBGT_VAR,
            prod.CLUSTER_COL,
        ]
    ].copy()

    pred["indicator"] = indicator
    pred["validation_year"] = int(train_end + 1)
    pred["spline_df"] = int(spline_df)
    pred["y"] = y_eval
    pred["mu"] = mu_eval
    pred["hot"] = hot

    return row, pred


# ============================================================================
# ONE TEMPORAL FOLD
# ============================================================================

def prepare_fold(indicator, raw, validation_year):
    train_end = validation_year - 1

    seq = raw[raw["year"] <= validation_year].copy()
    train0 = seq[seq["year"] <= train_end].copy()
    val0 = seq[seq["year"] == validation_year].copy()

    if train0.empty or val0.empty:
        raise RuntimeError(
            f"No data for train <= {train_end} / validation {validation_year}"
        )

    # Facility inclusion is learned from training only.
    train0, val0, min_obs = retain_training_facilities(
        indicator, train0, val0
    )

    # If winsorisation is ever switched on, avoid future leakage.
    train0, val0 = training_only_winsorise(
        indicator, train0, val0
    )

    keep_facs = set(train0["facility"].unique())

    # Rebuild continuous sequence using only training-retained facilities.
    seq = seq[seq["facility"].isin(keep_facs)].copy()

    # Carry the training-only winsorised outcome values back into seq.
    # (Currently production WINSORIZE=False, so this is normally a no-op.)
    if getattr(prod, "WINSORIZE", False):
        replacements = pd.concat([
            train0[["facility", "date", "y"]],
            val0[["facility", "date", "y"]],
        ]).drop_duplicates(["facility", "date"])
        seq = seq.drop(columns=["y"]).merge(
            replacements,
            on=["facility", "date"],
            how="left",
        )

    train_for_shift = seq[seq["year"] <= train_end].copy()
    shifts = training_shifts(train_for_shift)

    seq, lag_cols = add_lags_and_controls(seq, shifts)

    train = seq[seq["year"] <= train_end].copy()
    val = seq[seq["year"] == validation_year].copy()

    # Explicit year controls and zone×year labels.
    train, val, year_cols = add_training_year_controls(
        train, val, train_end
    )
    train, val = add_zone_year_fe(
        train, val, train_end
    )

    # Construct a common non-spline sample shared by every df.
    common_needed = [
        "y",
        "facility",
        "month",
        prod.CLUSTER_COL,
        prod.WBGT_VAR,
        "covid",
    ]

    if prod.USE_PRECIP:
        common_needed += [prod.PRECIP_COL, "precip_c"]

    common_needed += year_cols + lag_cols

    if prod.USE_ZONE_YEAR_FE:
        common_needed.append("cv_zone_year")

    train = train.dropna(
        subset=list(dict.fromkeys(common_needed))
    ).copy()

    val = val.dropna(
        subset=list(dict.fromkeys(common_needed))
    ).copy()

    # Validation must contain only FE levels estimable from training.
    train_facs = set(train["facility"].unique())
    train_months = set(train["month"].unique())

    val = val[
        val["facility"].isin(train_facs)
        & val["month"].isin(train_months)
    ].copy()

    if prod.USE_ZONE_YEAR_FE:
        train_zy = set(train["cv_zone_year"].unique())
        val = val[val["cv_zone_year"].isin(train_zy)].copy()

    return train, val, shifts, year_cols, lag_cols, min_obs


# ============================================================================
# SUMMARIES / MODEL-SELECTION TABLES
# ============================================================================

def add_df3_comparisons(fold):
    """
    Within each indicator-year fold, compare every df directly with df=3.
    Negative delta / negative percent = LOWER deviance = better than df3.
    """
    fold = fold.copy()

    ref = (
        fold[fold["spline_df"] == 3][
            [
                "indicator",
                "validation_year",
                "mean_poisson_deviance",
                "hot_mean_poisson_deviance",
            ]
        ]
        .rename(
            columns={
                "mean_poisson_deviance": "df3_mean_poisson_deviance",
                "hot_mean_poisson_deviance": "df3_hot_mean_poisson_deviance",
            }
        )
    )

    fold = fold.merge(
        ref,
        on=["indicator", "validation_year"],
        how="left",
    )

    fold["delta_mean_deviance_vs_df3"] = (
        fold["mean_poisson_deviance"]
        - fold["df3_mean_poisson_deviance"]
    )

    fold["pct_mean_deviance_vs_df3"] = (
        100.0
        * fold["delta_mean_deviance_vs_df3"]
        / fold["df3_mean_poisson_deviance"]
    )

    fold["delta_hot_mean_deviance_vs_df3"] = (
        fold["hot_mean_poisson_deviance"]
        - fold["df3_hot_mean_poisson_deviance"]
    )

    fold["pct_hot_mean_deviance_vs_df3"] = (
        100.0
        * fold["delta_hot_mean_deviance_vs_df3"]
        / fold["df3_hot_mean_poisson_deviance"]
    )

    return fold


def make_summary(fold):
    metrics = [
        "mean_poisson_deviance",
        "mae",
        "rmse",
        "aggregate_error_pct",
        "hot_mean_poisson_deviance",
        "hot_mae",
        "hot_rmse",
        "fraction_validation_wbgt_clipped",
        "delta_mean_deviance_vs_df3",
        "pct_mean_deviance_vs_df3",
        "delta_hot_mean_deviance_vs_df3",
        "pct_hot_mean_deviance_vs_df3",
    ]

    summary = (
        fold.groupby(["indicator", "spline_df"], as_index=False)[metrics]
        .mean()
    )

    counts = (
        fold.groupby(["indicator", "spline_df"])
        .size()
        .rename("n_folds")
        .reset_index()
    )

    summary = summary.merge(
        counts,
        on=["indicator", "spline_df"],
        how="left",
    )

    return summary


def make_overall_summary(fold):
    """
    Equal weight to each completed indicator × validation-year fold.
    This prevents high-volume outcomes from automatically dominating selection.
    """
    metrics = [
        "mean_poisson_deviance",
        "delta_mean_deviance_vs_df3",
        "pct_mean_deviance_vs_df3",
        "hot_mean_poisson_deviance",
        "delta_hot_mean_deviance_vs_df3",
        "pct_hot_mean_deviance_vs_df3",
    ]

    overall = (
        fold.groupby("spline_df", as_index=False)[metrics]
        .agg(["mean", "median"])
    )

    # Flatten MultiIndex columns.
    overall.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in overall.columns.to_flat_index()
    ]

    return overall


# ============================================================================
# PLOTS
# ============================================================================

def plot_validation(summary):
    # One figure: mean held-out Poisson deviance by df, one line per indicator.
    pivot = summary.pivot(
        index="spline_df",
        columns="indicator",
        values="mean_poisson_deviance",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for indicator in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[indicator],
            marker="o",
            label=indicator,
        )

    ax.set_xlabel("Spline degrees of freedom")
    ax.set_ylabel("Mean held-out Poisson deviance (lower is better)")
    ax.set_title("Temporal out-of-sample validation of WBGT spline df")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(
        CV_DIR / "cv_mean_poisson_deviance_by_df.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Percent change vs df3 is easier to compare across indicators.
    pivot_pct = summary.pivot(
        index="spline_df",
        columns="indicator",
        values="pct_mean_deviance_vs_df3",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for indicator in pivot_pct.columns:
        ax.plot(
            pivot_pct.index,
            pivot_pct[indicator],
            marker="o",
            label=indicator,
        )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xlabel("Spline degrees of freedom")
    ax.set_ylabel("% change in held-out deviance vs df=3\n(negative = better)")
    ax.set_title("Out-of-sample performance relative to df=3")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(
        CV_DIR / "cv_percent_deviance_change_vs_df3.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("WBGT SPLINE DF — TEMPORAL OUT-OF-SAMPLE VALIDATION")
    print("=" * 80)
    print(f"Production source: {SOURCE_SCRIPT}")
    print(f"Validation years: {VALIDATION_YEARS}")
    print(f"Spline dfs: {SPLINE_DFS}")
    print(f"WBGT variable: {prod.WBGT_VAR}")
    print(f"Lags: {prod.SA_LAG} {prod.LAG_MONTHS}")
    print(f"Precipitation: {prod.USE_PRECIP}")
    print(f"Zone×year FE: {prod.USE_ZONE_YEAR_FE}")
    print(f"Facility coverage: {getattr(prod, 'MIN_OBS_COVERAGE', 'NA')}")
    print()

    rows = []
    predictions = []

    for indicator in prod.COUNT_INDICATORS:
        print("\n" + "=" * 80)
        print(indicator)
        print("=" * 80)

        try:
            raw = load_raw_indicator(indicator)
        except Exception as exc:
            print(f"PREPARATION FAILED: {exc}")
            continue

        for validation_year in VALIDATION_YEARS:
            min_year = prod.MIN_YEAR_BY_INDICATOR.get(
                indicator, prod.min_year_historical
            )

            if validation_year <= min_year:
                continue

            try:
                (
                    train_common,
                    val_common,
                    shifts,
                    year_cols,
                    lag_cols,
                    min_obs,
                ) = prepare_fold(
                    indicator,
                    raw,
                    validation_year,
                )
            except Exception as exc:
                print(
                    f"  {validation_year}: fold preparation FAILED: {exc}"
                )
                continue

            print(
                f"  {validation_year}: "
                f"train={len(train_common):,}, "
                f"val={len(val_common):,}, "
                f"facilities={train_common['facility'].nunique()}, "
                f"fold min_obs={min_obs}"
            )

            for spline_df in SPLINE_DFS:
                print(f"      df={spline_df:<2}", end=" ")

                try:
                    row, pred = evaluate_one_df(
                        indicator=indicator,
                        train_common=train_common,
                        val_common=val_common,
                        shifts=shifts,
                        year_cols=year_cols,
                        lag_cols=lag_cols,
                        spline_df=spline_df,
                        train_end=validation_year - 1,
                    )

                    rows.append(row)
                    predictions.append(pred)

                    print(
                        f"mean dev={row['mean_poisson_deviance']:.4f}; "
                        f"MAE={row['mae']:.2f}; "
                        f"hot n={row['n_hot_validation']}; "
                        f"WBGT clipped="
                        f"{100*row['fraction_validation_wbgt_clipped']:.2f}%"
                    )

                except Exception as exc:
                    print(f"FAILED: {exc}")

    fold = pd.DataFrame(rows)

    if fold.empty:
        raise RuntimeError(
            "No validation models completed successfully."
        )

    fold = add_df3_comparisons(fold)

    fold.to_csv(
        CV_DIR / "cv_fold_metrics.csv",
        index=False,
    )

    if predictions:
        pd.concat(
            predictions,
            ignore_index=True,
        ).to_csv(
            CV_DIR / "cv_predictions.csv",
            index=False,
        )

    summary = make_summary(fold)
    summary.to_csv(
        CV_DIR / "cv_indicator_summary.csv",
        index=False,
    )

    overall = make_overall_summary(fold)
    overall.to_csv(
        CV_DIR / "cv_overall_summary.csv",
        index=False,
    )

    plot_validation(summary)

    print("\n" + "=" * 80)
    print("PRIMARY RESULT: HELD-OUT MEAN POISSON DEVIANCE")
    print("Lower is better.")
    print("=" * 80)

    print(
        summary.pivot(
            index="indicator",
            columns="spline_df",
            values="mean_poisson_deviance",
        ).round(4).to_string()
    )

    print("\n" + "=" * 80)
    print("% CHANGE IN HELD-OUT DEVIANCE VS df=3")
    print("Negative = better than df=3; positive = worse.")
    print("=" * 80)

    print(
        summary.pivot(
            index="indicator",
            columns="spline_df",
            values="pct_mean_deviance_vs_df3",
        ).round(3).to_string()
    )

    print("\n" + "=" * 80)
    print("OVERALL ACROSS INDICATOR × YEAR FOLDS")
    print("=" * 80)
    print(overall.round(4).to_string(index=False))

    print("\nInterpretation rule:")
    print(
        "  Prefer the LOWEST df unless a more flexible spline gives a "
        "meaningful and consistent reduction in held-out Poisson deviance."
    )
    print(
        "  Do not select df from significance, CI width, or in-sample AIC alone."
    )

    print("\nOutputs:")
    for name in [
        "cv_fold_metrics.csv",
        "cv_indicator_summary.csv",
        "cv_overall_summary.csv",
        "cv_predictions.csv",
        "cv_mean_poisson_deviance_by_df.png",
        "cv_percent_deviance_change_vs_df3.png",
    ]:
        print(f"  {CV_DIR / name}")


if __name__ == "__main__":
    main()
