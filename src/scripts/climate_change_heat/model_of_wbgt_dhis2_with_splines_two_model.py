"""
loop_all_indicators_two_model_with_curves.py

Two-model approach for WBGT–health-service disruption:
  Model A (exposure):      y ~ cr(WBGT, df) + WBGT_lags + covid + year | facility + month
  Model B (counterfactual): y ~                            covid + year | facility + month

Poisson fixed-effects (pyfixest.fepois) with district-clustered SEs.
Poisson QMLE is consistent for the coefficients under any variance structure
provided the conditional mean is correctly specified; the cluster-robust
sandwich handles overdispersion. Fixed effects are absorbed by demeaning
rather than entering as ~750 dummy columns.

Computes % deficit = 100 * (sum(pred_A) - sum(pred_B)) / sum(pred_B)
for each indicator, with a district-level block bootstrap, produces a
forest plot, and saves exposure-response curves for the contemporaneous
WBGT spline component.
"""

import os
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import pyfixest as pf
from joblib import Parallel, delayed

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
SPLINE_DF = 2  # df for natural cubic spline on contemporaneous WBGT
LAG_MONTHS = [1, 2, 3, 9]
CENTER = True
MIN_OBS = 10
min_year_historical = 2015
max_year_historical = 2025
apply_cap = False

# Exposure-response curve settings
N_CURVE_POINTS = 200
CURVE_REF = "mean"  # one of: "mean", "median", "min"
MAKE_CURVE_CI = False  # set True only if you later add uncertainty propagation

# COVID window — controlled via a dummy, NOT masked to NaN
COVID_START = "2020-04-01"
COVID_END = "2021-12-01"

# Known structural closures — masked to NaN (missing), NOT set to 0
CLOSURES = [
    ("Phalombe Health Centre", "2023-04-01", "2024-06-01"),
    ("Thumbwe Health Centre", "2023-03-01", "2024-03-01"),
]

# Clustering / bootstrap unit
CLUSTER_COL = "Dist"  # district-level clustering, consistent with pipeline

# Bootstrap
N_BOOTSTRAP = 50  # set to 0 to skip bootstrap CIs
BOOT_SEED = 42
BOOT_CI_LEVEL = 0.95
BOOT_MIN_SUCCESS = 0.80  # raise if fewer than this fraction of replicates converge
N_JOBS = -1  # joblib workers for the bootstrap; -1 = all cores

# Multiplicity
FDR_ALPHA = 0.05

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)
PANEL_DIST_COL_IN_PANEL = "Dist"


# ---------------------------------------------------------------------------
# Helpers
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

    # COVID: dummy, keeps the rows and the weather variation in them
    long["covid"] = long["date"].between(COVID_START, COVID_END).astype(int)

    # Structural closures → NaN (missing), not zero counts
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

    # Sparsity filter
    obs_per_fac = long.dropna(subset=["y", WBGT_VAR]).groupby("facility").size()
    keep_facs = obs_per_fac[obs_per_fac >= MIN_OBS].index
    long = long[long["facility"].isin(keep_facs)].copy()

    if long.empty or long["facility"].nunique() < 2:
        print(f"  [{indicator}] Too few facilities after filter — skipping.")
        return None

    # Facilities must map to exactly one district
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
    is a true k-month lag. Gaps in the raw panel would otherwise make shift()
    silently reach further back than intended.
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
            f"calendar gaps ({100 * n_inserted / len(out):.2f}% of grid). "
            f"These carry NaN weather and are dropped after lag construction."
        )

    # Recompute derived calendar columns on the completed grid
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["covid"] = out["date"].between(COVID_START, COVID_END).astype(int)
    out[CLUSTER_COL] = out.groupby("facility")[CLUSTER_COL].transform(
        lambda s: s.ffill().bfill()
    )

    # Hard check: every facility now has strictly monthly spacing
    diffs = out.groupby("facility")["date"].diff().dropna()
    bad_spacing = diffs[~((diffs.dt.days >= 28) & (diffs.dt.days <= 31))]
    if len(bad_spacing):
        raise ValueError(
            f"[{indicator}] Non-monthly spacing remains after reindexing "
            f"({len(bad_spacing)} rows)."
        )

    return out


def add_columns(df: pd.DataFrame, indicator: str) -> tuple[pd.DataFrame, list[str], float, float]:
    """
    Add centred year, centred WBGT, and lag columns.
    Returns (df, lag_rhs_terms, year_shift, wbgt_shift).
    """
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


def add_spline_basis(df: pd.DataFrame, df_spline: int) -> tuple[pd.DataFrame, list[str]]:
    """
    Materialise the natural cubic spline basis as explicit columns.

    pyfixest formulas do not carry patsy stateful transforms, so the basis is
    built ONCE on the analysis sample and then carried as ordinary columns.
    Bootstrap replicates are row subsets of that sample, so every replicate
    uses an identical basis (same knots) — which is what we want, since knots
    re-estimated per replicate would not be comparable across replicates.
    """
    basis = patsy.dmatrix(
        f"cr(x, df={df_spline}) - 1",
        {"x": df["wbgt_c"].values},
        return_type="dataframe",
    )
    cols = [f"wbgt_s{i+1}" for i in range(basis.shape[1])]
    for c, b in zip(cols, basis.columns):
        df[c] = basis[b].values
    return df, cols


def drop_separated_facilities(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """
    Facilities whose outcome is zero in every retained month are perfectly
    separated by their own fixed effect (intercept -> -inf). Drop them loudly
    rather than leaving them for the solver to discover.
    """
    all_zero = df.groupby("facility")["y_int"].max() == 0
    sep_facs = all_zero[all_zero].index
    if len(sep_facs):
        n_rows = int(df["facility"].isin(sep_facs).sum())
        print(
            f"  [{indicator}] SEPARATION: {len(sep_facs)} facilities are all-zero "
            f"({n_rows:,} rows) and are dropped. Disclose this in methods."
        )
        df = df[~df["facility"].isin(sep_facs)].copy()
    return df


def fit_pois(fml: str, data: pd.DataFrame, cluster_col: str):
    """Poisson fixed-effects fit with cluster-robust SEs at `cluster_col`."""
    return pf.fepois(fml=fml, data=data, vcov={"CRV1": cluster_col})


def response_pred(model) -> np.ndarray:
    """
    Fitted values on the response (count) scale.

    pyfixest's predict() signature has moved between versions; try the
    explicit response request first, then fall back to exponentiating the
    linear predictor, which is equivalent for a log link.
    """
    try:
        return np.asarray(model.predict(type="response"), dtype=float)
    except TypeError:
        return np.exp(np.asarray(model.predict(), dtype=float))


def get_coef_table(model) -> pd.DataFrame:
    """Return a coefficient table in a version-tolerant way."""
    for attr in ["tidy", "coeftable", "coef_table"]:
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        try:
            tab = obj() if callable(obj) else obj
            if isinstance(tab, pd.DataFrame):
                return tab.copy()
        except Exception:
            pass
    raise AttributeError(
        "Could not extract coefficient table from pyfixest model; "
        "inspect model methods in your installed version."
    )


def coefficient_lookup(model) -> dict[str, float]:
    """Map coefficient names to estimates."""
    tab = get_coef_table(model)

    name_col = None
    for cand in ["Coefficient", "coef", "term", "variable", "index"]:
        if cand in tab.columns:
            name_col = cand
            break
    if name_col is None:
        if tab.index.dtype == object or isinstance(tab.index, pd.Index):
            tab = tab.reset_index().rename(columns={tab.index.name or "index": "term"})
            name_col = "term"
        else:
            raise KeyError("Could not identify coefficient-name column in pyfixest table.")

    est_col = None
    for cand in ["Estimate", "estimate", "coef", "Coef."]:
        if cand in tab.columns:
            est_col = cand
            break
    if est_col is None:
        raise KeyError("Could not identify estimate column in pyfixest coefficient table.")

    return dict(zip(tab[name_col].astype(str), tab[est_col].astype(float)))


def make_exposure_response_curve(
    model,
    spline_cols: list[str],
    wbgt_shift: float,
    observed_wbgt: pd.Series,
    indicator: str,
) -> pd.DataFrame:
    """
    Build the contemporaneous WBGT exposure-response curve on the relative-risk scale.

    This uses only the spline component of the contemporaneous WBGT term, holding
    all other covariates fixed. The curve is expressed relative to a reference WBGT.
    """
    coefs = coefficient_lookup(model)

    x_min = float(observed_wbgt.min())
    x_max = float(observed_wbgt.max())
    x_grid = np.linspace(x_min, x_max, N_CURVE_POINTS)
    x_grid_c = x_grid - wbgt_shift

    basis_grid = patsy.dmatrix(
        f"cr(x, df={SPLINE_DF}) - 1",
        {"x": x_grid_c},
        return_type="dataframe",
    )
    basis_grid.columns = spline_cols

    if CURVE_REF == "mean":
        x_ref = float(observed_wbgt.mean())
    elif CURVE_REF == "median":
        x_ref = float(observed_wbgt.median())
    elif CURVE_REF == "min":
        x_ref = float(observed_wbgt.min())
    else:
        raise ValueError(f"Unknown CURVE_REF='{CURVE_REF}'.")

    x_ref_c = x_ref - wbgt_shift
    basis_ref = patsy.dmatrix(
        f"cr(x, df={SPLINE_DF}) - 1",
        {"x": [x_ref_c]},
        return_type="dataframe",
    )
    basis_ref.columns = spline_cols
    ref_row = basis_ref.iloc[0].to_numpy(dtype=float)

    beta = np.array([coefs.get(col, 0.0) for col in spline_cols], dtype=float)
    eta_grid = basis_grid.to_numpy(dtype=float) @ beta
    eta_ref = float(ref_row @ beta)

    rr = np.exp(eta_grid - eta_ref)
    pct_change = 100.0 * (rr - 1.0)

    curve_df = pd.DataFrame(
        {
            "indicator": indicator,
            "label": INDICATOR_LABELS.get(indicator, indicator),
            "wbgt": x_grid,
            "wbgt_c": x_grid_c,
            "rr_vs_ref": rr,
            "pct_change_vs_ref": pct_change,
            "wbgt_ref": x_ref,
            "curve_ref": CURVE_REF,
            "note": "Contemporaneous spline component only; lagged WBGT terms held fixed.",
        }
    )
    return curve_df


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
        f"Contemporaneous WBGT spline component (reference = {x_ref:.2f})",
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


def compute_deficit(model_a, model_b) -> dict:
    """
    % deficit = 100 * (sum(pred_A) - sum(pred_B)) / sum(pred_B)

    Both models are fitted to the same rows, so fitted values are used
    directly — no out-of-sample prediction is required.

    Negative values -> WBGT associated with lower service delivery.
    Positive values -> WBGT associated with higher attendance.
    """
    pred_a = response_pred(model_a)
    pred_b = response_pred(model_b)
    if len(pred_a) != len(pred_b):
        raise ValueError(
            f"Model A and Model B fitted to different row counts "
            f"({len(pred_a)} vs {len(pred_b)}) — separation has removed "
            f"different observations from each. Deficit is not comparable."
        )
    total_a = float(pred_a.sum())
    total_b = float(pred_b.sum())
    return {
        "total_pred_exposure": total_a,
        "total_pred_counterfactual": total_b,
        "deficit_abs": total_a - total_b,
        "deficit_pct": 100.0 * (total_a - total_b) / total_b,
    }


def dispersion(model) -> float:
    """Pearson dispersion diagnostic (reported, not used for inference)."""
    try:
        mu = response_pred(model)
        y = np.asarray(model._Y, dtype=float).ravel()
        resid2 = ((y - mu) ** 2) / np.clip(mu, 1e-8, None)
        return float(resid2.sum() / (len(y) - model._k))
    except Exception:
        return np.nan


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
# Bootstrap replicate (module level so joblib can pickle it)
# ---------------------------------------------------------------------------
def _boot_replicate(seed_seq, nb_data, dist_index, dist_ids, f_a, f_b, cluster_col):
    """
    One district-level block bootstrap replicate.

    Districts are drawn with replacement; whole districts move together, so
    within-district correlation (serial and cross-facility) is preserved by
    construction. Repeated draws get suffixed facility and district IDs so
    that duplicated blocks do not share fixed effects or cluster identity.
    """
    rng = np.random.default_rng(seed_seq)
    picks = rng.choice(len(dist_ids), size=len(dist_ids), replace=True)

    idx_parts = [dist_index[dist_ids[p]] for p in picks]
    idx = np.concatenate(idx_parts)
    tags = np.repeat(
        np.arange(len(picks)), [len(part) for part in idx_parts]
    ).astype(str)

    boot_df = nb_data.take(idx).reset_index(drop=True)
    tags = pd.Series(tags, index=boot_df.index).astype(str)
    boot_df["facility"] = boot_df["facility"].astype(str) + "__b" + tags
    boot_df[cluster_col] = boot_df[cluster_col].astype(str) + "__b" + tags

    try:
        ma = fit_pois(f_a, boot_df, cluster_col)
        mb = fit_pois(f_b, boot_df, cluster_col)
        d = compute_deficit(ma, mb)
        if not np.isfinite(d["deficit_pct"]):
            return None, "non-finite deficit"
        return d["deficit_pct"], None
    except Exception as e:
        return None, type(e).__name__


# ---------------------------------------------------------------------------
# Per-indicator runner
# ---------------------------------------------------------------------------
def run_indicator(indicator: str) -> dict | None:
    """Fit Model A and Model B, return deficit summary."""

    long = prepare_data(indicator)
    if long is None:
        return None

    long, lag_terms, year_shift, wbgt_shift = add_columns(long, indicator)

    # Analysis sample — drop rows with any missing lag or outcome
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
    nb_data, spline_cols = add_spline_basis(nb_data, SPLINE_DF)
    nb_data = nb_data.reset_index(drop=True)

    FE = "facility + month"

    # --- Model A: with WBGT spline + lags ----------------------------------
    rhs_a = " + ".join(spline_cols + lag_terms + ["covid", "year_c"])
    f_a = f"y_int ~ {rhs_a} | {FE}"

    # --- Model B: counterfactual (no WBGT at all) --------------------------
    f_b = f"y_int ~ covid + year_c | {FE}"

    try:
        model_a = fit_pois(f_a, nb_data, CLUSTER_COL)
        model_b = fit_pois(f_b, nb_data, CLUSTER_COL)
    except Exception as e:
        print(f"  [{indicator}] Model failed: {type(e).__name__}: {e} — skipping.")
        return None

    deficit = compute_deficit(model_a, model_b)

    # --- Save exposure-response curve for contemporaneous WBGT spline ------
    try:
        curve_df = make_exposure_response_curve(
            model=model_a,
            spline_cols=spline_cols,
            wbgt_shift=wbgt_shift,
            observed_wbgt=nb_data[WBGT_VAR],
            indicator=indicator,
        )
        curve_df.to_csv(f"{OUT_DIR}exposure_response_curve_{indicator}.csv", index=False)
        save_exposure_response_plot(curve_df, indicator)
    except Exception as e:
        print(f"  [{indicator}] Curve export failed: {type(e).__name__}: {e}")

    # --- Per-facility-month predictions (for mapping) ----------------------
    pred_a = response_pred(model_a)
    pred_b = response_pred(model_b)
    if len(pred_a) != len(nb_data):
        raise ValueError(
            f"[{indicator}] fepois returned {len(pred_a)} fitted values for "
            f"{len(nb_data)} rows — observations were dropped internally "
            f"(separation / perfect prediction). Row alignment for the "
            f"prediction export is not safe; investigate before proceeding."
        )

    preds = nb_data[["facility", "date"]].copy()
    preds["y_obs"] = nb_data["y_int"].values
    preds["y_pred_wx"] = pred_a
    preds["y_pred_base"] = pred_b
    # sign convention: deficit (heat suppressing appointments) is > 0
    preds["difference"] = preds["y_pred_base"] - preds["y_pred_wx"]

    if PANEL_DIST_COL_IN_PANEL in nb_data.columns:
        preds["Dist"] = nb_data[PANEL_DIST_COL_IN_PANEL].values

    preds.to_csv(f"{OUT_DIR}two_model_predictions_{indicator}.csv", index=False)

    # --- Bootstrap CIs (district-level block bootstrap, parallel) ----------
    failures: Counter = Counter()

    if N_BOOTSTRAP > 0:
        dist_ids = nb_data[CLUSTER_COL].unique()
        # positional row indices per district, computed once
        dist_index = {
            d: np.asarray(g, dtype=np.int64)
            for d, g in nb_data.groupby(CLUSTER_COL, sort=False).indices.items()
        }

        # independent, reproducible streams per replicate
        seeds = np.random.SeedSequence(BOOT_SEED).spawn(N_BOOTSTRAP)

        out = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
            delayed(_boot_replicate)(
                s, nb_data, dist_index, dist_ids, f_a, f_b, CLUSTER_COL
            )
            for s in seeds
        )

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
                f"Failure modes: {dict(failures)}. CIs from a non-random subset "
                f"of replicates are not trustworthy — fix the fit, do not report."
            )

        alpha = 1 - BOOT_CI_LEVEL
        boot_arr = np.asarray(boot_pcts)
        deficit["ci_lo"] = float(np.percentile(boot_arr, 100 * alpha / 2))
        deficit["ci_hi"] = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
        deficit["n_boot_ok"] = n_ok
        # two-sided bootstrap p-value against H0: deficit = 0
        frac_le = float(np.mean(boot_arr <= 0))
        frac_ge = float(np.mean(boot_arr >= 0))
        deficit["p_boot"] = float(
            min(1.0, max(2 * min(frac_le, frac_ge), 1.0 / (n_ok + 1)))
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
    deficit["dispersion_a"] = dispersion(model_a)
    deficit["dispersion_b"] = dispersion(model_b)
    deficit["spline_df"] = SPLINE_DF
    deficit["curve_ref"] = CURVE_REF

    print(
        f"  [{indicator}] OK  n={len(nb_data):,}, "
        f"fac={nb_data['facility'].nunique()}, "
        f"clust={nb_data[CLUSTER_COL].nunique()}, "
        f"phi={deficit['dispersion_a']:.1f}, "
        f"deficit={deficit['deficit_pct']:+.2f}%"
    )
    return deficit


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Two-model WBGT analysis: exposure vs counterfactual")
    print(f"Estimator = Poisson FE (pyfixest.fepois), CRV1 on {CLUSTER_COL}")
    print(f"Spline df = {SPLINE_DF}, lags = {LAG_MONTHS}")
    print(f"Exposure = {WBGT_VAR}")
    print(f"Bootstrap replicates = {N_BOOTSTRAP} (n_jobs={N_JOBS})")
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

    results_df.to_csv(f"{OUT_DIR}two_model_deficit_results.csv", index=False)
    print(f"\nResults saved -> {OUT_DIR}two_model_deficit_results.csv")
    print(
        f"BH-FDR at alpha={FDR_ALPHA}: "
        f"{int(results_df['sig_bh'].sum())}/{len(results_df)} indicators significant."
    )

    # -----------------------------------------------------------------------
    # FOREST PLOT — % deficit per indicator
    # -----------------------------------------------------------------------
    plot_df = results_df.sort_values("deficit_pct", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(7, max(4, len(plot_df) * 0.55 + 1.5)))

    has_ci = plot_df["ci_lo"].notna().all()

    colors = []
    for _, row in plot_df.iterrows():
        if not has_ci:
            colors.append("#4a7298")  # no CIs computed — neutral colour
        elif bool(row["sig_bh"]):
            colors.append("#823038")  # survives BH-FDR
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
        f"WBGT-associated deficit (Model A vs counterfactual)\n"
        f"Poisson FE, cr({WBGT_VAR}, df={SPLINE_DF}) + lags, "
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

    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}forest_plot_two_model_deficit.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Forest plot saved -> {OUT_DIR}forest_plot_two_model_deficit.png")
    print("\nDone.")
