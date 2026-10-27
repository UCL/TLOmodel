"""
Nested logit for healthcare seeking (IHS5) using Biogeme >= 3.3.

Current alternatives:
    none | gov | cham

The gov and cham alternatives form one free nest. ``none`` is the
reference alternative and is treated as a singleton nest by Biogeme.
Biogeme estimates mu = 1 / lambda, with mu >= 1.

Usage:
    python hsb_nested_logit_biogeme_corrected.py path/to/hsb_analysis.csv
    python hsb_nested_logit_biogeme_corrected.py --simulate

The CSV must contain:
    choice, ea_id
    cost_none, cost_gov, cost_cham

The choice column must contain exactly one of:
    none, gov, cham

Set N_BOOT to a positive integer to request EA-cluster bootstrap SEs.
"""

import logging
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

# Biogeme 3.3.2 can fail while writing its cosmetic TOML configuration file.
# This patch is harmless on later versions and avoids that failure.
import biogeme.parameters as _bp

_bp.Parameters.dump_file = lambda self, *args, **kwargs: None

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.models import loglogit, lognested
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit


# ---------------------------------------------------------------------------
# Configuration

ALTS = ["none", "gov_cham"]
NESTS = []
GENERIC = ["cost"]
CLUSTER = "ea_id"
USE_WEIGHTS = False
N_BOOT = 100
OUTNAME = "hsb_nested_logit_biogeme"
OUTPUT_DIR = "/Users/rachelmurray-watson/Documents/HSB/hsb_output"

INCLUDE_SEVERITY = True
EXTRA_NUMERIC = [
    "disability",
    "log_hhsize",
    "has_u5",
    "head_female",
    "bank",
    "insured",
    "climate_shock",
    "rainy_season",
    "rain_anom",
]

logging.getLogger("biogeme").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Data preparation


def prepare(df):
    """Create the individual-specific design matrix used in the model."""
    df = df.copy()

    required = ["choice"] + [f"{g}_{a}" for g in GENERIC for a in ALTS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cat = []
    num_cols = []

    if "wealth_quintile" in df.columns:
        df["wq"] = (
            "Q" + df["wealth_quintile"].astype("Int64").astype("string")
        )
        cat.append("wq")

    if "age_grp" in df.columns:
        df["agegrp"] = df["age_grp"].astype("string")
        cat.append("agegrp")

    for c in ["symptom", "region"]:
        if c in df.columns:
            cat.append(c)

    for c in ["female", "urban", "chronic"] + EXTRA_NUMERIC:
        if c in df.columns and df[c].notna().any():
            num_cols.append(c)

    if INCLUDE_SEVERITY and "days_stop" in df.columns:
        df["severity"] = pd.cut(
            df["days_stop"],
            bins=[-1, 0, 3, np.inf],
            labels=["0d", "1-3d", "4+d"],
        ).astype(object)
        df.loc[df["days_stop"].isna(), "severity"] = np.nan
        cat.append("severity")

    if "head_educ" in df.columns and df["head_educ"].notna().any():
        cat.append("head_educ")

    if not num_cols and not cat:
        raise ValueError("No usable covariates were found.")

    pieces = []
    if num_cols:
        pieces.append(df[num_cols].astype(float))
    if cat:
        pieces.append(pd.get_dummies(df[cat], dtype=float))

    X = pd.concat(pieces, axis=1)

    # Missing covariates cause the observation to be dropped from estimation.
    missing_covariate = df[num_cols + cat].isna().any(axis=1)
    X.loc[missing_covariate, :] = np.nan

    references = {
        "agegrp": "15-34",
        "symptom": "fever_malaria",
        "wq": "Q1",
        "region": "Northern",
        "severity": "0d",
        "head_educ": "none",
    }

    drop_cols = []
    for key, value in references.items():
        cols = [c for c in X.columns if c.startswith(f"{key}_")]
        if not cols:
            continue
        ref = f"{key}_{value}"
        if ref not in X.columns:
            raise ValueError(
                f"Reference {ref} not found; levels present: {cols}"
            )
        drop_cols.append(ref)

    X = X.drop(columns=drop_cols)

    # Drop constants, including all-zero dummy variables.
    X = X.loc[:, X.nunique(dropna=True) > 1]

    skipped = [
        c
        for c in EXTRA_NUMERIC + ["days_stop", "head_educ"]
        if c not in df.columns or df[c].isna().all()
    ]

    print(f"covariates in model ({X.shape[1]}): {list(X.columns)}")
    if skipped:
        print(f"not available in data, skipped: {skipped}")
    print(
        "observations with missing covariates: "
        f"{int(missing_covariate.sum())} of {len(df)}"
    )

    return df, X


def safe(name):
    """Make a covariate name safe for use as a Biogeme variable name."""
    return re.sub(
        r"[^0-9A-Za-z_]",
        "_",
        str(name).replace("+", "plus").replace("-", "to"),
    )


# ---------------------------------------------------------------------------
# Database and model construction


def build_database(df, X, alts=ALTS, generic=GENERIC, use_w=USE_WEIGHTS):
    available_generic = [
        g for g in generic
        if all(f"{g}_{a}" in df.columns for a in alts)
    ]
    if not available_generic:
        raise ValueError(
            "None of the requested generic variables have all "
            "alternative-specific columns."
        )

    xcols = list(X.columns)
    generic_cols = [
        f"{g}_{a}" for g in available_generic for a in alts
    ]

    ok = (
        X.notna().all(axis=1)
        & df[generic_cols].notna().all(axis=1)
        & df["choice"].notna()
    )

    choice_map = {a: i + 1 for i, a in enumerate(alts)}
    choices = df.loc[ok, "choice"].map(choice_map)
    if choices.isna().any():
        bad = df.loc[ok].loc[choices.isna(), "choice"].unique().tolist()
        raise ValueError(f"Unknown values in choice column: {bad}")

    if CLUSTER not in df.columns:
        raise ValueError(
            f"Cluster column {CLUSTER!r} is required for bootstrap SEs."
        )

    d = pd.DataFrame(
        {safe(c): X.loc[ok, c].to_numpy() for c in xcols}
    )

    sds = {}
    for g in available_generic:
        sd = float(
            df.loc[ok, [f"{g}_{a}" for a in alts]].to_numpy().std()
        )
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        sds[g] = sd
        for a in alts:
            d[f"{g}_{a}"] = (
                df.loc[ok, f"{g}_{a}"].to_numpy() / sd
            )

    d["CHOICE"] = choices.to_numpy(dtype=int)
    d["CLUSTER"] = df.loc[ok, CLUSTER].to_numpy()

    if use_w and "hh_weight" in df.columns:
        mean_weight = df.loc[ok, "hh_weight"].mean()
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            raise ValueError("hh_weight must have a positive finite mean.")
        d["WEIGHT"] = df.loc[ok, "hh_weight"].to_numpy() / mean_weight
    else:
        d["WEIGHT"] = 1.0

    # Standardize selected continuous covariates after constructing d.
    for c in ["log_hhsize", "rain_anom"]:
        c_safe = safe(c)
        if c_safe in d.columns:
            mean = d[c_safe].mean()
            sd = d[c_safe].std()
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            d[c_safe] = (d[c_safe] - mean) / sd

    return d, [safe(c) for c in xcols], available_generic, sds

def estimate(model, name):
    model.model_name = name
    return model.estimate()


def model_aic(result):
    k = len(result.get_beta_values())
    return -2.0 * result.final_loglikelihood + 2.0 * k


def model_bic(result, n_obs):
    k = len(result.get_beta_values())
    return -2.0 * result.final_loglikelihood + np.log(n_obs) * k


# ---------------------------------------------------------------------------
# Estimation output


def parameter_table(result, sds=None):
    """Return coefficients and robust sandwich standard errors."""
    beta = result.get_beta_values()
    names = list(beta)
    covariance = np.asarray(result.robust_variance_covariance_matrix)

    se = pd.Series(np.nan, index=names, dtype=float)
    for name in names:
        try:
            i = result.get_parameter_index(name)
            if i < covariance.shape[0]:
                value = covariance[i, i]
                se.loc[name] = np.sqrt(value) if value >= 0 else np.nan
        except Exception:
            pass

    out = pd.DataFrame(
        {
            "coef": pd.Series(beta, dtype=float),
            "se_robust": se,
        }
    )

    # Convert standardized generic coefficients back to original units.
    if sds is not None:
        for g, sd in sds.items():
            name = f"B_{g.upper()}"
            if name in out.index:
                out.loc[name, ["coef", "se_robust"]] /= sd

    out["z"] = out["coef"] / out["se_robust"]
    out["p"] = 2.0 * norm.sf(np.abs(out["z"]))
    return out

def cluster_bootstrap_logit(d, xcols, alts, generic, sds, result, out, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    clusters = d["CLUSTER"].drop_duplicates().to_numpy()
    draws = []
    for b in range(n_boot):
        mult = pd.Series(rng.choice(clusters, size=len(clusters), replace=True)).value_counts()
        d_b = d.copy()
        d_b["WEIGHT"] = d_b["WEIGHT"] * d_b["CLUSTER"].map(mult).fillna(0.0).astype(float)
        d_b = d_b.loc[d_b["WEIGHT"] > 0].reset_index(drop=True)
        if d_b["CHOICE"].nunique() < len(alts):
            continue
        try:
            r = fit_mnl(d_b, xcols, alts, generic)
            if not r.algorithm_has_converged:
                continue
            draw = pd.Series(r.get_beta_values(), dtype=float)
            for g in generic:
                draw[f"B_{g.upper()}"] /= sds[g]
            draws.append(draw)
        except Exception as exc:
            print(f"  bootstrap rep {b} failed: {exc}")
        if not draws:
            raise RuntimeError("No bootstrap replications succeeded.")
    D = pd.DataFrame(draws)
    out = out.copy()
    out["se_cluster_boot"] = D.std(ddof=1).reindex(out.index)
    out["z_cluster"] = out["coef"] / out["se_cluster_boot"]
    out["p_cluster"] = 2.0 * norm.sf(np.abs(out["z_cluster"]))
    print(f"cluster bootstrap: {len(D)} / {n_boot} replications succeeded")
    return out

# ---------------------------------------------------------------------------
# MNL model


def fit_mnl(d, xcols, alts, generic):
    """Estimate the unrestricted MNL model for comparison."""
    database = Database("hsb_mnl", d.drop(columns=["CLUSTER"]))

    V = {1: 0}
    for j, alt in enumerate(alts[1:], start=2):
        utility = Beta(f"ASC_{alt}", 0.0, None, None, 0)
        for c in xcols:
            utility += (
                Beta(f"B_{c}__{alt}", 0.0, None, None, 0)
                * Variable(c)
            )
        for g in generic:
            utility += (
                Beta(f"B_{g.upper()}", 0.0, None, None, 0)
                * Variable(f"{g}_{alt}")
            )
        V[j] = utility

    log_probability = loglogit(V, None, Variable("CHOICE"))
    if (d["WEIGHT"] != 1).any():
        formulas = {
            "log_like": log_probability,
            "weight": Variable("WEIGHT"),
        }
    else:
        formulas = log_probability

    model = BIOGEME(
        database,
        formulas,
        generate_html=False,
        generate_yaml=False,
    )
    return estimate(model, "mnl")


# ---------------------------------------------------------------------------
# Main estimation routine


def run(
    df,
    X,
    alts=ALTS,
    generic=GENERIC,
    use_w=USE_WEIGHTS,
    n_boot=N_BOOT,
):
    # Estimate MNL first and use its coefficients as nested-logit starts.
    d, xcols, used_generic, sds = build_database(df, X, alts=alts, generic=generic, use_w=use_w)
    print(f"N = {len(d)} | clusters = {d['CLUSTER'].nunique()}")

    mnl = fit_mnl(d, xcols, alts, used_generic)
    mnl_out = parameter_table(mnl, sds=sds)

    if n_boot:
        mnl_out = cluster_bootstrap_logit(d, xcols, alts, used_generic, sds, mnl, mnl_out, n_boot)

    print(mnl_out.to_string())
    print(f"logLik = {mnl.final_loglikelihood:.2f}, AIC = {model_aic(mnl):.1f}, BIC = {model_bic(mnl, len(d)):.1f}")
    return mnl_out, mnl, d


# ---------------------------------------------------------------------------
# Simulation

def simulate(N=20_000, seed=1, n_boot=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "female": rng.integers(0, 2, N),
        "urban": rng.integers(0, 2, N),
        "poor": rng.integers(0, 2, N),
        "age": rng.normal(0, 1, N),
    }).astype(float)

    df = pd.DataFrame({CLUSTER: rng.integers(0, 500, N)})
    df["cost_none"] = 0.0
    df["cost_gov_cham"] = rng.normal(0, 1, N)
    df[["female", "chronic", "bank", "climate_shock", "rainy_season"]].nunique()
    true_beta = np.array([-0.5, 0.2, 0.4, -0.3, 0.1])  # ASC, then X columns
    true_cost = -0.4
    v = true_beta[0] + X.to_numpy() @ true_beta[1:] + true_cost * df["cost_gov_cham"].to_numpy()
    df["choice"] = np.where(rng.random(N) < 1 / (1 + np.exp(-v)), "gov_cham", "none")

    out, _, _ = run(df, X, n_boot=n_boot)

    truth = {"ASC_gov_cham": true_beta[0], "B_COST": true_cost}
    for k, col in enumerate(X.columns):
        truth[f"B_{safe(col)}__gov_cham"] = true_beta[1 + k]
    out["true"] = pd.Series(truth)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(out.to_string())
    print("max |coef - true| / se:",
          (abs(out["coef"] - out["true"]) / out["se_robust"]).max().round(2))

def merge_facilities(df, cost_rule="min"):
    """Collapse gov + cham into a single 'care' alternative."""
    df = df.copy()
    df["choice"] = df["choice"].replace({"gov": "gov_cham", "cham": "gov_cham"})

    gc = df[["cost_gov", "cost_cham"]]
    if cost_rule == "min":
        df["cost_gov_cham"] = gc.min(axis=1, skipna=True)
    elif cost_rule == "mean":
        df["cost_gov_cham"] = gc.mean(axis=1, skipna=True)
    elif cost_rule == "chosen":
        # cost of the facility type actually used; fall back to mean if none chosen
        orig = df["choice_orig"]
        df["cost_gov_cham"] = np.where(
            orig == "gov", df["cost_gov"],
            np.where(orig == "cham", df["cost_cham"], gc.mean(axis=1)),
        )
    return df

# ---------------------------------------------------------------------------
# Command-line entry point


def main():
    if "--simulate" in sys.argv:
        simulate(n_boot=0)
        return

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python hsb_nested_logit_biogeme_corrected.py "
            "path/to/hsb_analysis.csv"
        )

    df = pd.read_csv(sys.argv[1])
    print(df[["female", "chronic", "bank", "climate_shock", "rainy_season"]].nunique())
    df["choice_orig"] = df["choice"]
    df = merge_facilities(df)
    df, X = prepare(df)
    mnl_out, mnl, d = run(df, X)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    coef_path = os.path.join(OUTPUT_DIR, f"{OUTNAME}_logit_coefs.csv")
    fit_path = os.path.join(OUTPUT_DIR, f"{OUTNAME}_logit_fit.csv")
    mnl_out.to_csv(coef_path)
    pd.DataFrame({
        "logLik": [mnl.final_loglikelihood],
        "n_parameters": [len(mnl.get_beta_values())],
        "AIC": [model_aic(mnl)],
        "BIC": [model_bic(mnl, len(d))],
        "N": [len(d)],
    }).to_csv(fit_path, index=False)
    print(f"\nSaved:\n  {coef_path}\n  {fit_path}")


if __name__ == "__main__":
    main()
