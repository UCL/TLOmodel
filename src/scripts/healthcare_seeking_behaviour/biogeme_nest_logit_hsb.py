"""
Nested logit for healthcare seeking (IHS5) in Biogeme >= 3.3.

Alternatives:
    none | other | [gov | cham | private]

The facility alternatives form one free nest. ``none`` and ``other`` are
singleton nests. Biogeme estimates mu = 1 / lambda, with mu >= 1.

Usage:
    python hsb_nested_logit_biogeme_standalone.py path/to/hsb_analysis.csv
    python hsb_nested_logit_biogeme_standalone.py --simulate

Expected columns include:
    choice
    cost_none, cost_other, cost_gov, cost_cham, cost_private
    dist_none, dist_other, dist_gov, dist_cham, dist_private

The choice column must contain exactly one of:
    none, other, gov, cham, private

EA-cluster bootstrap SEs are disabled by default. Set N_BOOT to a positive
number for bootstrap SEs.
"""

import logging
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


#ALTS = ["none", "other", "gov", "cham", "private"]
ALTS = ["none", "care"] # i.e. not public care (incl. no care, faith healther, private) and public care (gov. or CHAM)
NESTS = [["gov", "cham"]]
GENERIC = ["cost", "dist"]
CLUSTER = "ea_id"
USE_WEIGHTS = False
N_BOOT = 0
OUTNAME = "hsb_nested_logit_biogeme"

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
        df["wq"] = "Q" + df["wealth_quintile"].astype("Int64").astype("string")
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

    # Missing covariates cause the corresponding observation to be dropped.
    # Dummy variables for missing categorical values are otherwise all zero.
    missing_covariate = df[num_cols + cat].isna().any(axis=1)
    X.loc[missing_covariate, :] = np.nan

    references = {
        "agegrp": "15-34",
        "symptom": "fever_malaria",
        "wq": "Q1",
        "region": "North",
        "severity": "0d",
        "head_educ": "none",
    }
    drop_cols = [
        f"{key}_{value}"
        for key, value in references.items()
        if f"{key}_{value}" in X.columns
    ]
    X = X.drop(columns=drop_cols)

    # Drop constant columns, including all-zero dummy columns.
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
        f"observations with missing covariates: "
        f"{int(missing_covariate.sum())} of {len(df)}"
    )

    return df, X


def safe(name):
    """Make a covariate name safe for use as a Biogeme variable name."""
    return re.sub(r"[^0-9A-Za-z_]", "_", str(name).replace("+", "plus").replace("-", "to"))


# ---------------------------------------------------------------------------
# Biogeme model construction

def build_database(df, X, alts=ALTS, generic=GENERIC, use_w=USE_WEIGHTS):
    available_generic = [
        g for g in generic if all(f"{g}_{a}" in df.columns for a in alts)
    ]
    if not available_generic:
        raise ValueError(
            "None of the requested generic variables have all alternative-specific columns."
        )

    xcols = list(X.columns)
    generic_cols = [f"{g}_{a}" for g in available_generic for a in alts]

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
            f"Cluster column {CLUSTER!r} is required for the sandwich/bootstrap SEs."
        )

    d = pd.DataFrame({safe(c): X.loc[ok, c].to_numpy() for c in xcols})

    sds = {}
    for g in available_generic:
        sd = float(df.loc[ok, [f"{g}_{a}" for a in alts]].to_numpy().std())
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        sds[g] = sd
        for a in alts:
            d[f"{g}_{a}"] = df.loc[ok, f"{g}_{a}"].to_numpy() / sd

    d["CHOICE"] = choices.to_numpy(dtype=int)
    d["CLUSTER"] = df.loc[ok, CLUSTER].to_numpy()

    if use_w and "hh_weight" in df.columns:
        mean_weight = df.loc[ok, "hh_weight"].mean()
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            raise ValueError("hh_weight must have a positive finite mean.")
        d["WEIGHT"] = df.loc[ok, "hh_weight"].to_numpy() / mean_weight
    else:
        d["WEIGHT"] = 1.0

    return d, [safe(c) for c in xcols], available_generic, sds


def build_model(d, xcols, alts=ALTS, nests=NESTS, generic=GENERIC, start=None):
    database = Database("hsb", d.drop(columns=["CLUSTER"]))
    start = {} if start is None else start

    def B(name, lower=None, upper=None, fixed=0):
        return Beta(name, start.get(name, 0.0), lower, upper, fixed)

    # Alternative 1 (none) is the reference alternative with V = 0.
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
            start.get(f"MU_{m}", 1.2),
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


# ---------------------------------------------------------------------------
# Estimation output

def parameter_table(result):
    """Return coefficients and Biogeme robust sandwich standard errors."""
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
            # A parameter without an index is treated as fixed/unavailable.
            pass

    return pd.DataFrame(
        {
            "coef": pd.Series(beta, dtype=float),
            "se_robust": se,
        }
    )


def fit_mnl(d, xcols, alts, generic):
    """Estimate the unrestricted MNL model for the nesting LR comparison."""
    database = Database("hsb_mnl", d.drop(columns=["CLUSTER"]))

    V = {1: 0}
    for j, alt in enumerate(alts[1:], start=2):
        utility = Beta(f"ASC_{alt}", 0.0, None, None, 0)
        for c in xcols:
            utility += Beta(f"B_{c}__{alt}", 0.0, None, None, 0) * Variable(c)
        for g in generic:
            utility += Beta(f"B_{g.upper()}", 0.0, None, None, 0) * Variable(f"{g}_{alt}")
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

    result = estimate(
        build_model(d, xcols, alts, nests, used_generic),
        outname,
    )
    out = parameter_table(result)

    # Undo standardisation for generic coefficients.
    for g in used_generic:
        name = f"B_{g.upper()}"
        if name in out.index:
            out.loc[name, "coef"] /= sds[g]
            out.loc[name, "se_robust"] /= sds[g]

    # Convert Biogeme's mu to lambda = 1 / mu using the delta method.
    for m in range(len(nests)):
        mu_name = f"MU_{m + 1}"
        lambda_name = f"lambda_{m + 1}"
        mu = out.loc[mu_name, "coef"]
        se_mu = out.loc[mu_name, "se_robust"]
        out.loc[lambda_name, "coef"] = 1.0 / mu
        out.loc[lambda_name, "se_robust"] = se_mu / mu**2

    out["z"] = out["coef"] / out["se_robust"]
    out["p"] = 2.0 * norm.sf(np.abs(out["z"]))

    # MNL comparison. For one free nest, the LR p-value is the usual
    # 50:50 boundary-mixture approximation.
    mnl = fit_mnl(d, xcols, alts, used_generic)
    ll = result.final_loglikelihood
    ll_mnl = mnl.final_loglikelihood
    lr = max(0.0, 2.0 * (ll - ll_mnl))

    print(
        f"converged = {result.algorithm_has_converged} | "
        f"logL = {ll:.2f} | MNL logL = {ll_mnl:.2f}"
    )

    if len(nests) == 1:
        lr_p = 0.5 * chi2.sf(lr, 1)
    else:
        lr_p = chi2.sf(lr, len(nests))
        print(
            "[warning] For multiple free nests, the reported LR p-value is "
            "only a simple chi-square approximation."
        )

    for m, nest in enumerate(nests):
        lambda_name = f"lambda_{m + 1}"
        lam = out.loc[lambda_name, "coef"]
        se = out.loc[lambda_name, "se_robust"]
        print(
            f"lambda[{'/'.join(nest)}] = {lam:.3f} "
            f"(robust se {se:.3f}) | LR vs MNL: {lr:.2f}, p = {lr_p:.4f}"
        )

    if any(out.loc[f"lambda_{m + 1}", "coef"] > 0.95 for m in range(len(nests))):
        print("[warning] lambda is close to 1; MNL may be adequate.")

    if n_boot:
        out = cluster_bootstrap(
            d,
            xcols,
            alts,
            nests,
            used_generic,
            sds,
            result,
            out,
            n_boot,
        )

    return out, result, d


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
    start = result.get_beta_values()

    groups = {
        cluster: group.index.to_numpy()
        for cluster, group in d.groupby("CLUSTER", sort=False)
    }
    keys = np.array(list(groups), dtype=object)
    draws = []

    for b in range(n_boot):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[key] for key in sampled_keys])
        d_boot = d.loc[indices].reset_index(drop=True)

        try:
            boot_result = estimate(
                build_model(d_boot, xcols, alts, nests, generic, start=start),
                f"boot_{b}",
            )
            if not boot_result.algorithm_has_converged:
                print(f"  bootstrap rep {b} did not converge")
                continue

            draw = pd.Series(boot_result.get_beta_values(), dtype=float)
            for g in generic:
                name = f"B_{g.upper()}"
                if name in draw.index:
                    draw[name] /= sds[g]

            for m in range(len(nests)):
                draw[f"lambda_{m + 1}"] = 1.0 / draw[f"MU_{m + 1}"]

            draws.append(draw)
        except Exception as exc:
            print(f"  bootstrap rep {b} failed: {exc}")

    if not draws:
        raise RuntimeError("No bootstrap replications succeeded.")

    D = pd.DataFrame(draws)
    out["se_cluster_boot"] = D.std(ddof=1).reindex(out.index)
    out["z_cluster"] = out["coef"] / out["se_cluster_boot"]
    out["p_cluster"] = 2.0 * norm.sf(np.abs(out["z_cluster"]))

    print(f"cluster bootstrap: {len(D)} / {n_boot} replications succeeded")
    return out


# ---------------------------------------------------------------------------
# Simulation and command-line entry point

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
    dist = rng.exponential(1, (N, J))
    cost[:, 0] = 0
    dist[:, :2] = 0

    for j, alt in enumerate(alts):
        df[f"cost_{alt}"] = cost[:, j]
        df[f"dist_{alt}"] = dist[:, j]

    true_beta = np.array(
        [
            [-0.5, 0.2, 0.4, -0.3, 0.1],
            [-0.2, 0.3, 0.5, -0.5, 0.2],
            [-0.4, 0.1, 0.6, -0.2, 0.3],
            [-0.9, 0.4, 0.3, -0.6, -0.1],
        ]
    )
    true_generic = np.array([-0.4, -0.5])
    true_lambda = 0.6

    V = np.zeros((N, J))
    V[:, 1:] = true_beta[:, 0] + X.to_numpy() @ true_beta[:, 1:].T
    V += np.einsum("njl,l->nj", np.stack([cost, dist], axis=2), true_generic)

    inclusive = np.logaddexp.reduce(V[:, 2:] / true_lambda, axis=1)
    top = np.column_stack([V[:, 0], V[:, 1], true_lambda * inclusive])
    top_prob = np.exp(top - top.max(axis=1, keepdims=True))
    top_prob /= top_prob.sum(axis=1, keepdims=True)

    top_choice = (rng.random(N)[:, None] > top_prob.cumsum(axis=1)).sum(axis=1)
    facility_prob = np.exp(V[:, 2:] / true_lambda - inclusive[:, None])
    facility_choice = 2 + (
        rng.random(N)[:, None] > facility_prob.cumsum(axis=1)
    ).sum(axis=1)
    choices = np.where(top_choice < 2, top_choice, facility_choice)
    df["choice"] = [alts[i] for i in choices]

    out, result, _ = run(df, X, n_boot=n_boot)

    truth = {}
    for j, alt in enumerate(alts[1:]):
        truth[f"ASC_{alt}"] = true_beta[j, 0]
        for k, col in enumerate(X.columns):
            truth[f"B_{safe(col)}__{alt}"] = true_beta[j, 1 + k]
    truth.update({"B_COST": true_generic[0], "B_DIST": true_generic[1]})
    truth["lambda_1"] = true_lambda

    out["true"] = pd.Series(truth)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)
    print("\n", out.to_string())
    print(
        "\nmax |coef - true| / se:",
        (abs(out["coef"] - out["true"]) / out["se_robust"]).max().round(2),
    )


def main():
    if "--simulate" in sys.argv:
        simulate()
        return

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python hsb_nested_logit_biogeme_standalone.py "
            "path/to/hsb_analysis.csv"
        )

    df = pd.read_csv(sys.argv[1])
    df, X = prepare(df)
    out, _, _ = run(df, X)

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)
    print("\n", out.to_string())
    out.to_csv(f"/Users/rachelmurray-watson/Documents/HSB/hsb_output/{OUTNAME}_coefs.csv")


if __name__ == "__main__":
    main()
