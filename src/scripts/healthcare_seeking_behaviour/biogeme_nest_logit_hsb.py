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

ALTS = ["none", "gov", "cham"]
NESTS = [["gov", "cham"]]
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


def build_model(
    d,
    xcols,
    alts=ALTS,
    nests=NESTS,
    generic=GENERIC,
    start=None,
    database_name="hsb",
):
    database = Database(database_name, d.drop(columns=["CLUSTER"]))
    start = {} if start is None else start

    def B(name, lower=None, upper=None, fixed=0):
        return Beta(name, start.get(name, 0.0), lower, upper, fixed)

    # Alternative 1 is the reference alternative with V = 0.
    V = {1: 0}
    for j, alt in enumerate(alts[1:], start=2):
        utility = B(f"ASC_{alt}")

        for c in xcols:
            utility += B(f"B_{c}__{alt}") * Variable(c)

        for g in generic:
            utility += B(f"B_{g.upper()}") * Variable(f"{g}_{alt}")

        V[j] = utility

    nest_objects = []
    for m, nest_alts in enumerate(nests, start=1):
        mu = Beta(
            f"MU_{m}",
            start.get(f"MU_{m}", 1.01),
            1.0,
            20.0,
            0,
        )
        indices = [alts.index(a) + 1 for a in nest_alts]
        nest_objects.append(
            OneNestForNestedLogit(
                mu,
                indices,
                name="nest_" + "_".join(nest_alts),
            )
        )

    tree = NestsForNestedLogit(
        choice_set=list(range(1, len(alts) + 1)),
        tuple_of_nests=tuple(nest_objects),
    )

    log_probability = lognested(V, None, tree, Variable("CHOICE"))
    if (d["WEIGHT"] != 1).any():
        formulas = {
            "log_like": log_probability,
            "weight": Variable("WEIGHT"),
        }
    else:
        formulas = log_probability

    return BIOGEME(
        database,
        formulas,
        generate_html=False,
        generate_yaml=False,
    )


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


def add_lambda_results(table, nests):
    """Add lambda = 1 / mu and its delta-method robust SE."""
    table = table.copy()

    for m in range(len(nests)):
        mu_name = f"MU_{m + 1}"
        lambda_name = f"lambda_{m + 1}"

        if mu_name not in table.index:
            raise ValueError(f"Missing {mu_name} in nested-logit output.")

        mu = table.loc[mu_name, "coef"]
        se_mu = table.loc[mu_name, "se_robust"]
        table.loc[lambda_name, "coef"] = 1.0 / mu
        table.loc[lambda_name, "se_robust"] = se_mu / mu**2

    table["z"] = table["coef"] / table["se_robust"]
    table["p"] = 2.0 * norm.sf(np.abs(table["z"]))
    return table


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
    nests=NESTS,
    generic=GENERIC,
    use_w=USE_WEIGHTS,
    n_boot=N_BOOT,
    outname=OUTNAME,
):
    d, xcols, used_generic, sds = build_database(
        df, X, alts=alts, generic=generic, use_w=use_w
    )

    print(f"N = {len(d)} | clusters = {d['CLUSTER'].nunique()}")

    # Estimate MNL first and use its coefficients as nested-logit starts.
    mnl = fit_mnl(d, xcols, alts, used_generic)
    start = dict(mnl.get_beta_values())
    for m in range(len(nests)):
        start[f"MU_{m + 1}"] = 1.05

    result = estimate(
        build_model(
            d,
            xcols,
            alts,
            nests,
            used_generic,
            start=start,
            database_name="hsb_nested",
        ),
        outname,
    )

    ll = result.final_loglikelihood
    ll_mnl = mnl.final_loglikelihood
    if ll < ll_mnl - 1e-3:
        raise RuntimeError(
            f"Nested logL {ll:.2f} < MNL logL {ll_mnl:.2f}: "
            "nested model did not converge to an acceptable solution."
        )

    nested_out = parameter_table(result, sds=sds)
    nested_out = add_lambda_results(nested_out, nests)
    mnl_out = parameter_table(mnl, sds=sds)

    k_nested = len(result.get_beta_values())
    k_mnl = len(mnl.get_beta_values())
    aic_nested = model_aic(result)
    aic_mnl = model_aic(mnl)
    bic_nested = model_bic(result, len(d))
    bic_mnl = model_bic(mnl, len(d))
    lr = max(0.0, 2.0 * (ll - ll_mnl))

    # The null value lambda = 1 is on the boundary. For one free nest,
    # use the usual 50:50 boundary-mixture approximation.
    if len(nests) == 1:
        lr_p = 0.5 * chi2.sf(lr, 1)
    else:
        lr_p = chi2.sf(lr, len(nests))
        print(
            "[warning] For multiple free nests, the LR p-value is "
            "only a simple chi-square approximation."
        )

    fit_stats = pd.DataFrame(
        {
            "model": ["nested_logit", "mnl"],
            "logLik": [ll, ll_mnl],
            "n_parameters": [k_nested, k_mnl],
            "AIC": [aic_nested, aic_mnl],
            "BIC": [bic_nested, bic_mnl],
        }
    )

    model_selection = pd.DataFrame(
        {
            "comparison": ["nested_logit_vs_mnl"],
            "LR_statistic": [lr],
            "LR_p_value": [lr_p],
            "preferred_by_AIC": [
                "nested_logit" if aic_nested < aic_mnl else "mnl"
            ],
            "preferred_by_BIC": [
                "nested_logit" if bic_nested < bic_mnl else "mnl"
            ],
        }
    )

    print("\nMNL coefficient estimates")
    print(mnl_out.to_string())
    print("\nNested-logit coefficient estimates")
    print(nested_out.to_string())
    print("\nModel fit statistics")
    print(fit_stats.to_string(index=False))
    print(
        f"\nLR test: nested logit versus MNL; "
        f"LR = {lr:.3f}, p = {lr_p:.4f}"
    )

    for m, nest in enumerate(nests):
        lambda_name = f"lambda_{m + 1}"
        lam = nested_out.loc[lambda_name, "coef"]
        se = nested_out.loc[lambda_name, "se_robust"]
        print(
            f"lambda[{'/'.join(nest)}] = {lam:.3f} "
            f"(robust SE {se:.3f})"
        )
        if lam > 0.95:
            print(
                "[warning] lambda is close to 1; MNL may be adequate."
            )

    nested_out.attrs["fit_stats"] = fit_stats
    nested_out.attrs["model_selection"] = model_selection

    if n_boot:
        nested_out = cluster_bootstrap(
            d,
            xcols,
            alts,
            nests,
            used_generic,
            sds,
            result,
            nested_out,
            n_boot,
        )
        nested_out.attrs["fit_stats"] = fit_stats
        nested_out.attrs["model_selection"] = model_selection

    return nested_out, mnl_out, result, mnl, d


# ---------------------------------------------------------------------------
# EA-cluster bootstrap


def cluster_bootstrap(
    d,
    xcols,
    alts,
    nests,
    generic,
    sds,
    result,
    out,
    n_boot,
    seed=0,
):
    rng = np.random.default_rng(seed)
    start = result.get_beta_values().copy()

    clusters = d["CLUSTER"].drop_duplicates().to_numpy()
    n_clusters = len(clusters)
    draws = []

    for b in range(n_boot):
        sampled_clusters = rng.choice(
            clusters,
            size=n_clusters,
            replace=True,
        )
        multiplicities = pd.Series(sampled_clusters).value_counts()

        d_boot = d.copy()
        d_boot["WEIGHT"] = d_boot["WEIGHT"] * (
            d_boot["CLUSTER"]
            .map(multiplicities)
            .fillna(0.0)
            .astype(float)
        )
        d_boot = d_boot.loc[d_boot["WEIGHT"] > 0].reset_index(drop=True)

        if d_boot["CHOICE"].nunique() < len(alts):
            print(
                f"  bootstrap rep {b} skipped: "
                "not all alternatives are represented"
            )
            continue

        if d_boot["CLUSTER"].nunique() < 5:
            print(f"  bootstrap rep {b} skipped: too few clusters")
            continue

        try:
            boot_model = build_model(
                d_boot,
                xcols,
                alts,
                nests,
                generic,
                start=start,
                database_name=f"hsb_boot_{b}",
            )
            boot_result = estimate(boot_model, f"boot_{b}")

            if not boot_result.algorithm_has_converged:
                print(f"  bootstrap rep {b} did not converge")
                continue

            draw = pd.Series(boot_result.get_beta_values(), dtype=float)

            for g in generic:
                name = f"B_{g.upper()}"
                if name in draw.index:
                    draw[name] /= sds[g]

            for m in range(len(nests)):
                mu_name = f"MU_{m + 1}"
                lambda_name = f"lambda_{m + 1}"
                if mu_name not in draw.index:
                    raise ValueError(f"Missing {mu_name} in bootstrap results")
                draw[lambda_name] = 1.0 / draw[mu_name]

            draws.append(draw)

        except Exception as exc:
            print(f"  bootstrap rep {b} failed: {exc}")

    if not draws:
        raise RuntimeError(
            "No bootstrap replications succeeded. "
            "The bootstrap samples may be too unstable."
        )

    D = pd.DataFrame(draws)
    out = out.copy()
    out["se_cluster_boot"] = D.std(ddof=1).reindex(out.index)
    out["z_cluster"] = out["coef"] / out["se_cluster_boot"]
    out["p_cluster"] = 2.0 * norm.sf(np.abs(out["z_cluster"]))

    print(
        f"cluster bootstrap: {len(D)} / "
        f"{n_boot} replications succeeded"
    )
    return out


# ---------------------------------------------------------------------------
# Simulation


def simulate(N=20_000, seed=1, n_boot=0):
    rng = np.random.default_rng(seed)
    alts = ALTS
    J = len(alts)

    X = pd.DataFrame(
        {
            "female": rng.integers(0, 2, N),
            "urban": rng.integers(0, 2, N),
            "poor": rng.integers(0, 2, N),
            "age": rng.normal(0, 1, N),
        }
    ).astype(float)

    df = pd.DataFrame({CLUSTER: rng.integers(0, 500, N)})
    cost = rng.normal(0, 1, (N, J))
    cost[:, 0] = 0.0

    for j, alt in enumerate(alts):
        df[f"cost_{alt}"] = cost[:, j]

    # Two non-reference alternatives: gov and cham.
    true_beta = np.array(
        [
            [-0.5, 0.2, 0.4, -0.3, 0.1],  # gov
            [-0.2, 0.3, 0.5, -0.5, 0.2],  # cham
        ]
    )
    true_generic = np.array([-0.4])
    true_lambda = 0.6

    V = np.zeros((N, J))
    V[:, 1:] = true_beta[:, 0] + X.to_numpy() @ true_beta[:, 1:].T
    V += true_generic[0] * cost

    # Facility nest consists of gov and cham, which are alternatives 1 and 2.
    inclusive = np.logaddexp.reduce(V[:, 1:] / true_lambda, axis=1)
    top = np.column_stack([V[:, 0], true_lambda * inclusive])
    top_prob = np.exp(top - top.max(axis=1, keepdims=True))
    top_prob /= top_prob.sum(axis=1, keepdims=True)

    top_choice = (
        rng.random(N)[:, None] > top_prob.cumsum(axis=1)
    ).sum(axis=1)
    facility_prob = np.exp(
        V[:, 1:] / true_lambda - inclusive[:, None]
    )
    facility_choice = 1 + (
        rng.random(N)[:, None] > facility_prob.cumsum(axis=1)
    ).sum(axis=1)
    choices = np.where(top_choice == 0, 0, facility_choice)
    df["choice"] = [alts[i] for i in choices]

    out, mnl_out, _, _, _ = run(df, X, n_boot=n_boot)

    truth = {}
    for j, alt in enumerate(alts[1:]):
        truth[f"ASC_{alt}"] = true_beta[j, 0]
        for k, col in enumerate(X.columns):
            truth[f"B_{safe(col)}__{alt}"] = true_beta[j, 1 + k]
    truth["B_COST"] = true_generic[0]
    truth["lambda_1"] = true_lambda

    out["true"] = pd.Series(truth)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)
    print("\nNested-logit output with simulation truth")
    print(out.to_string())
    print(
        "\nmax |coef - true| / se:",
        (abs(out["coef"] - out["true"]) / out["se_robust"])
        .max()
        .round(2),
    )
    print("\nMNL output")
    print(mnl_out.to_string())


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
    df, X = prepare(df)
    out, mnl_out, _, _, _ = run(df, X)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nested_path = os.path.join(
        OUTPUT_DIR, f"{OUTNAME}_nested_coefs.csv"
    )
    mnl_path = os.path.join(OUTPUT_DIR, f"{OUTNAME}_mnl_coefs.csv")
    fit_path = os.path.join(OUTPUT_DIR, f"{OUTNAME}_fit_stats.csv")
    selection_path = os.path.join(
        OUTPUT_DIR, f"{OUTNAME}_model_selection.csv"
    )

    out.to_csv(nested_path)
    mnl_out.to_csv(mnl_path)
    out.attrs["fit_stats"].to_csv(fit_path, index=False)
    out.attrs["model_selection"].to_csv(selection_path, index=False)

    print("\nSaved:")
    print(f"  {nested_path}")
    print(f"  {mnl_path}")
    print(f"  {fit_path}")
    print(f"  {selection_path}")


if __name__ == "__main__":
    main()
