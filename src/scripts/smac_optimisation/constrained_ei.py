"""
Multi-surrogate constrained Bayesian optimization for SMAC3.

Fits a SEPARATE random-forest surrogate for the objective (DALYs) and for
each constraint, then combines the objective's expected improvement (EI)
with the constraints' predicted behaviour into one acquisition value. Two
combination rules are provided, selected by a single parameter
(`combination`, sourced from optimisation_parameters.CONSTRAINT_COMBINATION):

    "multiplicative":  acquisition(x) = EI(x) * prod_j P_j(x)
    "additive":        acquisition(x) = EI of the merit f_0(x) + alpha * sum_j pi_j(x)

where P_j(x) is the probability that constraint j is satisfied at x and
pi_j(x) is its expected budget overshoot. Both are defined below.

THE CONSTRAINT VALUE g_j(x)
---------------------------
Each constraint j is a period of a cost category (HIV-HRH or
HIV-consumable). Its value g_j(x) is the signed budget slack
(initialise.bucket_peak_budget_slack):

    g_j(x) = max over years y in the period of  cost_y(x) / budget_y  -  1.

The constraint is satisfied exactly when g_j(x) <= 0. A positive value is
the fractional overspend in the worst year; a negative value is the
headroom in the worst year. Because g_j is signed it is CONTINUOUS through
the feasibility boundary: it keeps varying on the feasible side (how much
headroom) as well as the infeasible side (how much overspend), instead of
being flat at 0 across the whole feasible region. That continuity is what
makes the standard Gaussian treatment of a regression surrogate below
well-behaved at the limit.

PREDICTIVE MEAN AND VARIANCE FROM THE FOREST (as in SMAC3)
----------------------------------------------------------
Each target (the objective f_0 and every g_j) has its own random forest of
B trees. For input x, tree b sends x to one leaf; write m_b(x) for the mean
and v_b(x) for the variance of the training targets in that leaf. As in
SMAC3's random-forest surrogate (its law-of-total-variance option), the
predictive distribution is N(mu(x), sigma(x)^2) with

    mu(x)       = (1/B) * sum_b m_b(x)                                    (1)
    sigma^2(x)  = Var_b[ m_b(x) ]  +  (1/B) * sum_b v_b(x)                (2)

The first term of (2) is the disagreement between trees (uncertainty about
the fit); the second is the scatter of the observed values inside the
leaves (noise, here mostly seed-to-seed simulation noise). sigma is floored
at a small positive value. The same procedure is applied to every
constraint, since each has its own forest. (SMAC3's own surrogate is NOT
used: it is trained on one scalar cost per trial, whereas this needs a
separate prediction for the objective and for each constraint, which exist
only in the pipeline's own `history` - see ConstrainedEI.) Like any forest,
it extrapolates flatly far from the data, so sigma can understate
uncertainty in unexplored regions.

EXPECTED IMPROVEMENT (objective, minimisation)
----------------------------------------------
With eta the best feasible objective value so far and xi >= 0 an offset,
(both eta and mu_0 come from the SAME objective emulator: eta is the lowest
predicted mean over the configs observed to be feasible by their median
slack, not a raw observed DALYs value, so one lucky seed cannot set the
target. SMAC3's own default is the same idea, over all evaluated configs.)

    EI(x) = (eta - mu_0 - xi) * Phi(z) + sigma_0 * phi(z),
    z = (eta - mu_0 - xi) / sigma_0,                                      (3)

Phi/phi the standard normal CDF/density. Before any feasible point has
been observed eta is undefined and EI is replaced by sigma_0 (pure
exploration on the objective).

COMBINATION 1: "multiplicative" (assumes uncorrelated constraints)
------------------------------------------------------------------
Under the Gaussian read, the probability that constraint j is satisfied is

    P_j(x) = Pr[ g_j(x) <= 0 ] = Phi( -mu_j(x) / sigma_j(x) ),            (4)

and the textbook constrained-EI weights EI by the probability that ALL
constraints hold. That equals the product of the P_j only if the
constraints are statistically independent given x, so this mode ASSUMES
uncorrelated constraints. Consequences worth knowing:
  - The product shrinks geometrically with the number of constraints: six
    constraints each 90% likely to hold multiply to about 53%, six at 70%
    to about 12%, so EI is heavily discounted even when no single
    constraint is a real concern.
  - If the constraints are in fact correlated, the product is biased
    relative to the true joint probability, in a direction that depends on
    the sign of the correlation (positively correlated constraints are
    over-penalised, negatively correlated ones under-penalised). Here the
    six constraints share one simulated trajectory and the sign of their
    correlations is not known in advance.
  - It needs no calibration beyond xi.

COMBINATION 2: "additive" (merit-improvement form)
--------------------------------------------------
Each constraint contributes its own expected budget overshoot, the expected
part of g_j that lies above a tolerance delta >= 0,

    pi_j(x) = E[ max(g_j(x) - delta, 0) ],   g_j(x) ~ N(mu_j, sigma_j^2)
            = (mu_j - delta) * Phi(z_j) + sigma_j * phi(z_j),
      z_j = (mu_j - delta) / sigma_j,                                     (5)

the same rectified-normal integral as EI (Eq. 3), mirrored. The overshoots
are added to the objective with an exchange rate alpha (DALYs per unit, i.e.
100%, of overshoot) to give a MERIT,

    M(x) = f_0(x) + alpha * sum_j max(g_j(x) - delta, 0),                 (6)

and the acquisition value is the expected improvement of M over the best
merit eta_M found so far,

    acquisition(x) = (eta_M - mu_M - xi) Phi(z) + sigma_M phi(z),         (7)
    mu_M = mu_0 + alpha * sum_j pi_j,
    sigma_M^2 = sigma_0^2 + alpha^2 * sum_j Var[max(g_j - delta, 0)],
    z = (eta_M - mu_M - xi) / sigma_M,

with Var[max(g - delta, 0)] = E[o^2] - E[o]^2 and
E[o^2] = (d^2 + sigma^2) Phi(z) + d sigma phi(z), d = mu - delta.
eta_M is the lowest predicted merit among the configs that are feasible by
their median slack (all observed configs if none is feasible yet).

Why this form: subtracting alpha * sum pi_j from EI (the previous form) did
not shrink as EI shrank, so late in a run, when EI is tiny everywhere, the
penalty dominated and the maximiser drifted to the safest, best-known
region instead of the most promising one. Here the penalty enters INSIDE
the improvement: a candidate scores only if its predicted DALYs, plus the
price of its expected overshoot, beat the best feasible configuration.
The value is non-negative and decays to zero as the achievable improvement
vanishes, exactly like ordinary EI.
Properties:
  - alpha keeps its meaning: DALYs per unit (100%) of expected overshoot, an
    exchange rate between objective and budget, derived in the pipeline from
    the baseline (optimisation_parameters.MERIT_DALYS_IMPROVEMENT_PCT).
  - The merit MEAN is exact whatever the correlations between constraints
    (expectations add). Only the merit VARIANCE treats the objective and the
    constraints as uncorrelated; that approximation affects exploration
    weight, not the exchange rate. No product of probabilities is formed, so
    the geometric discount of the multiplicative rule does not arise.
  - The penalty grows with the SIZE of the expected overshoot and linearly
    with the number of constraints; uncertain constraint estimates are
    penalised somewhat more than confident ones (pi_j increases with sigma_j).
  - delta (MERIT_OVERSHOOT_TOLERANCE) ignores expected overshoot below that
    fraction of the budget; 0 counts any. It only softens the search:
    feasibility of a reported answer is always judged on g_j <= 0 directly.
  - Before any config is feasible, eta_M comes from all observed configs, so
    the search minimises DALYs and overshoot together rather than exploring
    the objective alone.

In both modes only the choice of which configuration to evaluate NEXT is
affected. Which configuration is finally reported depends only on the
recorded g_j values (<= 0), never on this function.

NOTE ON VERSION SENSITIVITY
----------------------------
SMAC3's `AbstractAcquisitionFunction` internals (exact `_compute` array
shape, how `self.model` / `self.eta` get set via `update()`) have changed
across 2.x releases. This is written against the general v2 architecture.
If wiring this into your installed version raises an AttributeError or
shape mismatch, open `smac/acquisition/function/expected_improvement.py` in
your installed package and match this class's `_compute` signature/shape
to that file - the maths above will still be correct, only the plumbing
around it might need a one-line tweak.

HYPERPARAMETERS: every tunable knob in this file is marked inline with a
"HYPERPARAMETER" comment - grep for that tag across all files
(constrained_ei.py, smac_scenario.py, convergence_monitoring.py,
optimisation_pipeline.py, optimisation_parameters.py) to find the complete
list in one pass.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from ConfigSpace import Configuration, ConfigurationSpace
from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)

COMBINATION_MULTIPLICATIVE = "multiplicative"
COMBINATION_ADDITIVE = "additive"
VALID_COMBINATIONS = (COMBINATION_MULTIPLICATIVE, COMBINATION_ADDITIVE)


# --------------------------------------------------------------------------
# 1. Encoding configs into arrays sklearn can use
# --------------------------------------------------------------------------

def configs_to_array(configs: Sequence[Configuration]) -> np.ndarray:
    """
    Vectorize a list of ConfigSpace Configurations into a 2D float array.

    Uses Configuration.get_array(), which is the same normalized
    representation SMAC's own surrogate models train on. Inactive
    conditional hyperparameters come back as NaN, which sklearn's
    RandomForestRegressor cannot handle - we impute them with a fixed
    sentinel so "inactive" is still a learnable signal rather than dropped.
    """
    X = np.array([c.get_array() for c in configs], dtype=float)
    X = np.nan_to_num(X, nan=-1.0)
    return X


# --------------------------------------------------------------------------
# 2. Multi-surrogate manager: one RF per target, with ensemble-based std
# --------------------------------------------------------------------------

class MultiSurrogateModel:
    """
    Holds one RandomForestRegressor per target (the objective and each
    constraint), and exposes predictive mean + std for each, estimated by
    the law of total variance across trees (the SMAC3 approach, standing
    in for a GP's posterior variance) - Eqs. (1)-(2) in the module
    docstring.
    """

    def __init__(
        self,
        target_names: Sequence[str],
        n_estimators: int = 100,       # HYPERPARAMETER: more trees = smoother/
                                          # more stable mean+std, linear compute cost
        min_samples_leaf: int = 3,     # HYPERPARAMETER: noise-smoothing strength -
                                          # see the grouped-CV + leave-one-seed-out
                                          # validation approach discussed earlier.
                                          # In real pipeline use this value comes
                                          # from ConstrainedEI's own constructor
                                          # (which sources it from
                                          # optimisation_parameters.MIN_SAMPLES_LEAF),
                                          # not this class's own default here - this
                                          # default only matters if MultiSurrogateModel
                                          # is constructed directly/standalone.
        random_state: int = 0,         # reproducibility seed, not a tunable hyperparameter
    ):
        self.target_names = list(target_names)
        self._rf_kwargs = dict(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.models: dict[str, RandomForestRegressor] = {}
        self.n_fitted_points: int = 0

    def fit(self, X: np.ndarray, targets: dict[str, np.ndarray]) -> None:
        """targets: dict mapping target name -> 1D array, same length as X."""
        for name in self.target_names:
            y = targets[name]
            rf = RandomForestRegressor(**self._rf_kwargs)
            rf.fit(X, y)
            self.models[name] = rf
        self.n_fitted_points = X.shape[0]

    def predict(self, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Returns dict: target name -> (mean, std), each shape (n_points,).
        Variance is the law of total variance (module docstring Eq. 2):
        variance of the per-tree leaf means + mean of the per-tree
        within-leaf variances (a regression leaf's sklearn `impurity` is
        exactly the variance of the training targets it holds).
        """
        out = {}
        for name, rf in self.models.items():
            # shapes: (n_estimators, n_points)
            tree_means = np.stack([tree.predict(X) for tree in rf.estimators_], axis=0)
            leaf_vars = np.stack(
                [tree.tree_.impurity[tree.apply(X)] for tree in rf.estimators_], axis=0
            )
            mean = tree_means.mean(axis=0)
            var = tree_means.var(axis=0) + leaf_vars.mean(axis=0)
            std = np.sqrt(np.maximum(var, 0.0))
            std = np.maximum(std, 1e-6)  # avoid divide-by-zero in EI/CDF below
            out[name] = (mean, std)
        return out


# --------------------------------------------------------------------------
# 3. The building blocks of the acquisition value (pure functions of the
#    predictive mean/std, so they can be checked independently of SMAC)
# --------------------------------------------------------------------------

def expected_improvement(mean: np.ndarray, std: np.ndarray, eta: float, xi: float) -> np.ndarray:
    """Eq. (3): expected improvement over eta for a minimisation objective."""
    improvement = eta - mean - xi
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0.0)


def probability_feasible(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Eq. (4): Pr[g_j <= 0] for g_j ~ N(mean, std^2)."""
    return norm.cdf((0.0 - mean) / std)


def expected_overshoot(mean: np.ndarray, std: np.ndarray, tolerance: float) -> np.ndarray:
    """Eq. (5): E[max(g_j - tolerance, 0)] for g_j ~ N(mean, std^2)."""
    shifted = mean - tolerance
    z = shifted / std
    pi = shifted * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(pi, 0.0)


def overshoot_variance(mean: np.ndarray, std: np.ndarray, tolerance: float) -> np.ndarray:
    """Variance of o = max(g - tolerance, 0) for g ~ N(mean, std^2):
    E[o^2] - E[o]^2 with E[o^2] = (d^2 + std^2) Phi(z) + d std phi(z), d = mean - tolerance."""
    d = mean - tolerance
    z = d / std
    second = (d * d + std * std) * norm.cdf(z) + d * std * norm.pdf(z)
    first = d * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(second - first * first, 0.0)


def merit_moments(
    mean_obj: np.ndarray,
    std_obj: np.ndarray,
    constraint_preds: Sequence[tuple[np.ndarray, np.ndarray]],
    alpha: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (6): mean and std of the merit M = f_0 + alpha * sum_j max(g_j - tolerance, 0),
    moment-matched to a Gaussian. The mean is exact (expectations add whatever the
    correlations); the variance assumes the terms are uncorrelated."""
    mean = mean_obj.copy()
    var = std_obj ** 2
    for mean_c, std_c in constraint_preds:
        mean = mean + alpha * expected_overshoot(mean_c, std_c, tolerance)
        var = var + alpha ** 2 * overshoot_variance(mean_c, std_c, tolerance)
    return mean, np.maximum(np.sqrt(var), 1e-6)


# --------------------------------------------------------------------------
# 4. The constrained EI acquisition function
# --------------------------------------------------------------------------

class ConstrainedEI(AbstractAcquisitionFunction):
    """
    acquisition(x) = EI(x) * prod_j P_j(x)           (combination="multiplicative")
    acquisition(x) = EI of the merit M(x) = f_0 + alpha * sum_j pi_j  (combination="additive")

    See the module docstring for the definitions and the trade-off between
    the two rules.

    `history_provider` is a zero-arg callable returning your own log of
    dicts (the `history` list built up by record_result() in
    optimisation_pipeline.py), each with a "config_object" key plus one key
    per target_name (objective + constraints). Re-fitting only happens when
    new points have arrived since the last fit, so this is safe to call
    every iteration without wasted work.

    The constraint values in `history` must be the signed budget slack g_j
    (<= 0 feasible), NOT a clipped violation - the Gaussian read in the
    module docstring relies on the values continuing below 0.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        objective_name: str,
        constraint_names: Sequence[str],
        history_provider: Callable[[], list[dict]],
        xi: float = 0.0,  # HYPERPARAMETER: EI's exploration/exploitation trade-off -
                            # higher requires more expected improvement before a
                            # candidate is favored; not yet tuned, worth watching if
                            # the search wanders into marginal late-run configs
        retrain_every: int = 1,  # HYPERPARAMETER: how many new history entries
                                    # accumulate before the surrogate refits -
                                    # see the earlier discussion on retrain
                                    # cadence and its interaction with N_CONCURRENT
        min_samples_leaf: int = 3,  # HYPERPARAMETER: passed straight through to
                                       # MultiSurrogateModel's own RandomForestRegressors
                                       # (see that class's own docstring/comment) -
                                       # sourced from optimisation_parameters.py's
                                       # own MIN_SAMPLES_LEAF, matching every other
                                       # hyperparameter in this pipeline.
        combination: str = COMBINATION_MULTIPLICATIVE,  # HYPERPARAMETER: the single
                                       # switch between the two rules - "multiplicative"
                                       # (EI * prod P_j; assumes uncorrelated
                                       # constraints) or "additive" (EI of the merit f_0 + alpha *
                                       # sum pi_j; no such assumption). Sourced from
                                       # optimisation_parameters.CONSTRAINT_COMBINATION.
        alpha: float = 1.0,  # HYPERPARAMETER (additive only): derived from MERIT_DALYS_IMPROVEMENT_PCT/
                                # MERIT_BUDGET_VIOLATION_PCT x baseline DALYs -
                                # DALYs per unit (100%) of expected budget overshoot;
                                # see optimisation_parameters.py's own description for
                                # how to calibrate it. Ignored when multiplicative.
        tolerance: float = 0.0,  # HYPERPARAMETER (additive only):
                                    # MERIT_OVERSHOOT_TOLERANCE - fraction of the budget
                                    # by which expected overshoot is ignored (0 = count
                                    # any). Ignored when multiplicative.
    ):
        super().__init__()
        if combination not in VALID_COMBINATIONS:
            raise ValueError(
                f"combination must be one of {VALID_COMBINATIONS}, got {combination!r}"
            )
        self._configspace = configspace
        self._objective_name = objective_name
        self._constraint_names = list(constraint_names)
        self._history_provider = history_provider
        self._xi = xi
        self._retrain_every = retrain_every
        self._combination = combination
        self._alpha = alpha
        self._tolerance = tolerance
        self._last_fit_n = 0  # history length at last fit, distinct from
                                # self._surrogate.n_fitted_points

        self._surrogate = MultiSurrogateModel(
            target_names=[objective_name, *self._constraint_names],
            min_samples_leaf=min_samples_leaf,
        )
        self._eta: float | None = None  # best feasible objective value seen so far

    @property
    def name(self) -> str:
        return "ConstrainedEI"

    def _maybe_refit(self) -> None:
        history = self._history_provider()
        n_new = len(history) - self._last_fit_n
        if n_new < self._retrain_every:
            return  # not enough new realisations yet - reuse existing models

        configs = [h["config_object"] for h in history]
        X = configs_to_array(configs)

        targets = {
            name: np.array([h[name] for h in history], dtype=float)
            for name in self._surrogate.target_names
        }
        self._surrogate.fit(X, targets)
        self._last_fit_n = len(history)

        # NOISY-EI CORRECTION: eta is the best-so-far target that EI tries
        # to improve on. Using the raw observed DALYs at the best feasible
        # point is unsafe under noise - a single lucky low realisation can
        # set eta artificially low, making every subsequent candidate look
        # worse than it should. Instead, ask the just-fitted surrogate what
        # it PREDICTS at every feasible observed config, and take the min
        # of those predictions. A lucky outlier gets pulled back toward the
        # model's mean; a genuinely good config (especially one that's been
        # intensified across several seeds) stays low. This is the standard
        # fix for EI under observation noise, and is the only change needed
        # anywhere in this class to make it noise-aware - _compute() itself
        # is untouched, since it already works off surrogate predictions,
        # never raw observations, on the candidate side.
        # Feasibility is judged per CONFIGURATION on the median over its
        # seeds, the same rule final selection and convergence use, not per
        # single observation (one lucky seed must not define the incumbent).
        by_config: dict[tuple, list[dict]] = {}
        for h in history:
            by_config.setdefault(tuple(sorted(dict(h["config_object"]).items())), []).append(h)
        feasible_configs = [
            rows[0]["config_object"] for rows in by_config.values()
            if all(float(np.median([r[c] for r in rows])) <= 0 for c in self._constraint_names)
        ]
        if self._combination == COMBINATION_MULTIPLICATIVE:
            if feasible_configs:
                X_feasible = configs_to_array(feasible_configs)
                mean_pred, _ = self._surrogate.predict(X_feasible)[self._objective_name]
                self._eta = float(np.min(mean_pred))
            else:
                self._eta = None
        else:
            # Merit mode: eta is the best predicted MERIT (objective plus
            # penalised expected overshoot, Eq. 6) among the median-feasible
            # configs, so a candidate has to beat the best feasible
            # configuration after paying for its own expected overshoot. If
            # nothing is feasible yet, the best predicted merit over every
            # observed config is used, which steers the search towards low
            # DALYs and low overshoot together.
            pool = feasible_configs or [rows[0]["config_object"] for rows in by_config.values()]
            preds_pool = self._surrogate.predict(configs_to_array(pool))
            mean_m, _ = merit_moments(
                *preds_pool[self._objective_name],
                [preds_pool[c] for c in self._constraint_names],
                self._alpha, self._tolerance,
            )
            self._eta = float(np.min(mean_m))
        # If nothing feasible has been observed yet, self._eta stays None
        # and EI below falls back to pure exploration on the objective mean.

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """
        X: array of shape (n_configs, n_features) - candidate configs
           already vectorized by SMAC's acquisition maximizer.
        Returns: array of shape (n_configs, 1) - higher is better,
           the maximizer will pick the argmax.
        """
        self._maybe_refit()

        if not self._surrogate.models:
            # No data fitted yet (shouldn't normally happen post initial
            # design, but guards against being called too early).
            return np.zeros((X.shape[0], 1))

        preds = self._surrogate.predict(X)
        mean_obj, std_obj = preds[self._objective_name]

        if self._combination == COMBINATION_MULTIPLICATIVE:
            # --- Expected Improvement on the objective (minimization) ---
            if self._eta is None:
                # No feasible point observed yet: pure exploration signal,
                # so the search is pushed toward reducing predictive variance
                # rather than chasing an EI target we can't define yet.
                ei = std_obj
            else:
                ei = expected_improvement(mean_obj, std_obj, self._eta, self._xi)
            prob_feasible = np.ones(X.shape[0])
            for name in self._constraint_names:
                mean_c, std_c = preds[name]
                prob_feasible *= probability_feasible(mean_c, std_c)
            acquisition_value = ei * prob_feasible
        else:
            # --- Expected improvement of the merit (module docstring Eq. 6-7) ---
            mean_m, std_m = merit_moments(
                mean_obj, std_obj, [preds[c] for c in self._constraint_names],
                self._alpha, self._tolerance,
            )
            acquisition_value = expected_improvement(mean_m, std_m, self._eta, self._xi)

        return acquisition_value.reshape(-1, 1)


# --------------------------------------------------------------------------
# 5. Example wiring
#
# NOTE: this synchronous, single-process example (calling smac.optimize()
# with a blocking target_function) is kept here only to show ConstrainedEI
# in isolation - the acquisition function itself doesn't care how it's
# driven. For the real TLOmodel/Azure Batch integration, which uses the
# ask-tell interface instead of optimize() (since simulations run as
# async remote jobs, not local blocking calls), see optimisation_pipeline.py
# and smac_scenario.py - those are the current, up-to-date wiring; this
# function is a minimal standalone sanity-check only, e.g. for testing
# ConstrainedEI against a toy local objective.
# --------------------------------------------------------------------------

def example_usage():
    """
    Minimal synchronous sketch, for testing ConstrainedEI in isolation
    with a cheap local objective. NOT the pattern used for the real
    Azure-based simulation - see optimisation_pipeline.py for that.
    """
    from smac import HyperparameterOptimizationFacade, Scenario

    configspace = ConfigurationSpace()  # <-- define your real hyperparameters here

    history: list[dict] = []

    COST_LIMIT = 2_000_000
    HR_LIMIT = 500
    STOCK_LIMIT = 10_000

    def run_simulation(config):
        # <-- replace with your real simulator call -->
        raise NotImplementedError

    def target_function(config: Configuration, seed: int = 0) -> float:
        dalys, cost, hr_used, stock_used = run_simulation(config)

        # Signed budget slack: <= 0 feasible, > 0 the fractional overspend.
        cost_violation = cost / COST_LIMIT - 1
        hr_violation = hr_used / HR_LIMIT - 1
        stock_violation = stock_used / STOCK_LIMIT - 1

        # This log is what ConstrainedEI's history_provider reads from.
        # "config_object" keeps the real Configuration for re-vectorizing;
        # config_dict is just for your own inspection/debugging later.
        history.append({
            "config_object": config,
            "config_dict": dict(config),
            "dalys": dalys,
            "cost_violation": cost_violation,
            "hr_violation": hr_violation,
            "stock_violation": stock_violation,
        })

        # SMAC still needs *some* scalar returned here for its own
        # bookkeeping/incumbent tracking - a simple penalized sum is fine
        # since the real search intelligence now lives in the acquisition
        # function above, not in this number. Clipped at 0 so headroom on
        # one constraint never offsets overspend on another.
        K = 3 * dalys  # placeholder - tune per the process discussed earlier
        penalty = K * sum(max(0.0, v) for v in (cost_violation, hr_violation, stock_violation))
        return dalys + penalty

    acquisition_function = ConstrainedEI(
        configspace=configspace,
        objective_name="dalys",
        constraint_names=["cost_violation", "hr_violation", "stock_violation"],
        history_provider=lambda: history,
        combination="multiplicative",  # or "additive" (then set alpha=, tolerance=)
    )

    scenario = Scenario(configspace, n_trials=400, deterministic=False)
    smac = HyperparameterOptimizationFacade(
        scenario,
        target_function,
        acquisition_function=acquisition_function,
        overwrite=True,
    )
    smac.optimize()

    # Final answer: same principle as before - filter your own history,
    # never trust smac.incumbent directly, since the scalar it tracks
    # is still the penalized one.
    feasible = [
        h for h in history
        if h["cost_violation"] <= 0 and h["hr_violation"] <= 0 and h["stock_violation"] <= 0
    ]
    best = min(feasible, key=lambda h: h["dalys"])
    return best


if __name__ == "__main__":
    example_usage()
