"""
Multi-surrogate constrained Bayesian optimization for SMAC3.

Fits a SEPARATE random-forest surrogate for the objective (DALYs) and for
each constraint (a non-negative "violation" amount, 0 = feasible), then
combines them into a single acquisition value:

    acquisition(x) = EI(x; dalys model)  -  alpha * sum_j pi_j(x)

where pi_j(x) is a per-constraint "expected exceedance" merit term (defined
below) rather than a probability. This file explains what each piece is and
why it's built that way; MOTIVATION.md-style history is intentionally left
out - see the companion rationale document for the full derivation and the
alternatives that were considered.

WHY EACH CONSTRAINT'S PROBABILITY IS READ FROM TREE VOTES, NOT MEAN/STD
-------------------------------------------------------------------------
A RandomForestRegressor's per-tree predictions give an ensemble mean and
std at any point, and it's tempting to plug those into a Gaussian CDF to
get P(violation <= 0). That construction is degenerate wherever the
ensemble mean sits at or near the training floor (violation = 0, which is
common - most configs violate no constraint most years): mean ~= 0 and std
still > 0 pushes P(violation <= 0) toward a hard ceiling of 0.5, regardless
of how confidently "feasible" the forest's trees actually are. The read
used here instead treats each tree as a binary voter: tree b "votes"
infeasible at x iff its own prediction is > 0. The fraction of trees voting
infeasible,

    p_hat_j(x) = (1/B) * sum_b  1[ tree_b(x) > 0 ],                    (1)

is a direct, non-degenerate estimate of P(constraint j violated | x) - it
isn't capped at 0.5 by construction, and it reflects genuine tree-to-tree
disagreement rather than continuous-prediction noise around a floor.

Uncertainty in that vote fraction is estimated with the infinitesimal
jackknife (IJ) for random forests (Wager, Hastie & Efron, 2014), rather
than treated as a fixed or heuristic quantity. Every tree is grown on its
own bootstrap resample of the training rows; if a training point i happens
to be over- or under-represented in the bootstrap draws behind the trees
that vote "infeasible" at x, the IJ estimator picks that up as genuine
sampling variance of p_hat_j(x), not just noise. Writing N_{b,i} for how
many times row i appears in tree b's bootstrap sample and v_b(x) for tree
b's vote at x,

    Cov_i(x) = (1/B) * sum_b (N_{b,i} - 1) * (v_b(x) - p_hat_j(x)),      (2)

    Var_IJ_j(x) = sum_i Cov_i(x)^2  -  (n/B^2) * sum_b (v_b(x) - p_hat_j(x))^2,   (3)

with the second term the standard finite-B bias correction (Wager et al.
2014). This is exactly the "spread from consensus among retraining" signal:
points where the trees' bootstrap composition would have to look very
different to flip the vote get a small Var_IJ_j(x); points where a handful
of resampled rows would swing the vote get a large one - which is what
should steer the search to explore harder in genuinely uncertain regions,
not merely uncertain-looking ones.

WHY THE CONSTRAINTS ARE COMBINED ADDITIVELY, NOT MULTIPLIED
-------------------------------------------------------------------------
The most direct read of "P(all constraints satisfied)" is the product of
each P(constraint_j satisfied). Two things make that the wrong choice here.
First, the product assumes the constraints are independent, which there's
no reason to expect (the HIV-HRH and HIV-consumable constraints across
periods all derive from the same simulated trajectory, and the direction of
any correlation between them is genuinely mixed rather than reliably
positive - e.g. excess deaths can raise near-term palliative-care cost
while lowering later long-term-care cost). An independence-assuming product
of correlated probabilities is systematically biased relative to the true
joint probability, and the bias's sign depends on a correlation structure
that isn't known and would have to be estimated separately per constraint
pair. Second, with m constraints all held to a shared feasibility standard,
the product compounds geometrically: six constraints each independently
"90% probably fine" multiply to a joint feasibility read of about 53%, which
crushes the acquisition value even when no single constraint is a real
concern, and does so more aggressively as m grows regardless of whether
that reflects a genuine joint risk.

The additive merit term below avoids both problems. Each constraint
contributes its own bounded "expected exceedance beyond a tolerance
threshold tau" term,

    pi_j(x) = E[ max(P_j(x) - tau, 0) ],   P_j(x) ~ Normal(p_hat_j(x), Var_IJ_j(x)),   (4)

computed with the same rectified-normal integral used for EI itself:

    pi_j(x) = (p_hat_j(x) - tau) * Phi(z) + sigma_j(x) * phi(z),   z = (p_hat_j(x) - tau) / sigma_j(x).   (5)

tau (MERIT_VIOLATION_THRESHOLD) is a tolerance on the predicted violation
PROBABILITY axis (in [0, 1]) - below it, an elevated-but-small chance of
violating constraint j contributes nothing; above it, the merit term grows
smoothly. Summing pi_j(x) across constraints, rather than multiplying
probabilities, means each constraint's own risk is judged on its own terms
- a single genuinely risky constraint moves the total by roughly its own
pi_j(x), rather than every constraint's read being multiplicatively
entangled with every other's. The full acquisition value is

    acquisition(x) = EI(x; dalys) - alpha * sum_j pi_j(x),      alpha = MERIT_PENALTY_ALPHA.   (6)

alpha converts the merit term's probability-scale units into the
objective's own DALYs-scale units, so the two terms are comparable when
added; see optimisation_parameters.py's own description of
MERIT_PENALTY_ALPHA/MERIT_VIOLATION_THRESHOLD for how to calibrate it.

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

sklearn's `_generate_sample_indices`/`_get_n_samples_bootstrap` are private
APIs (`sklearn.ensemble._forest`) used here to reconstruct each tree's own
bootstrap composition for the IJ variance. They've been stable across
recent sklearn releases but aren't a public contract - if an upgrade moves
or renames them, `MultiSurrogateModel._bootstrap_counts_for_forest` is the
only place that needs updating.

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
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap

from ConfigSpace import Configuration, ConfigurationSpace
from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)


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
# 2. Multi-surrogate manager: one RF per target, with ensemble-based
#    mean/std for continuous targets, and vote-fraction + infinitesimal-
#    jackknife estimates of P(target > 0) for targets registered as
#    "IJ targets" (the constraints).
# --------------------------------------------------------------------------

class MultiSurrogateModel:
    """
    Holds one RandomForestRegressor per target (the objective and each
    constraint).

    For every target, `predict()` exposes the usual ensemble mean + std
    (std from the spread of per-tree predictions - what stands in for a
    GP's posterior variance when using random forests). For targets passed
    to `fit()` as `ij_targets` (the constraints), `predict_violation_
    probability()` additionally exposes the vote-fraction estimate of
    P(target > 0) and its infinitesimal-jackknife standard deviation - see
    the module docstring for the derivation.
    """

    def __init__(
        self,
        target_names: Sequence[str],
        n_estimators: int = 100,       # HYPERPARAMETER: more trees = smoother/
                                          # more stable mean+std and vote-fraction
                                          # estimates, linear compute cost
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

        # Populated only for targets passed as `ij_targets` to fit():
        # shape (n_estimators, n_fitted_points), bootstrap draw count of
        # each training row in each tree - the raw material for the IJ
        # variance (see predict_violation_probability()).
        self._bootstrap_counts: dict[str, np.ndarray] = {}

    def fit(
        self,
        X: np.ndarray,
        targets: dict[str, np.ndarray],
        ij_targets: Sequence[str] = (),
    ) -> None:
        """
        targets: dict mapping target name -> 1D array, same length as X.
        ij_targets: subset of target_names (typically the constraints) for
            which predict_violation_probability() will later be called -
            for these, each tree's bootstrap composition is reconstructed
            and cached here so it doesn't need recomputing on every predict
            call.
        """
        self._bootstrap_counts = {}
        for name in self.target_names:
            y = targets[name]
            rf = RandomForestRegressor(**self._rf_kwargs)
            rf.fit(X, y)
            self.models[name] = rf
            if name in ij_targets:
                self._bootstrap_counts[name] = self._bootstrap_counts_for_forest(rf, X.shape[0])
        self.n_fitted_points = X.shape[0]

    @staticmethod
    def _bootstrap_counts_for_forest(rf: RandomForestRegressor, n_samples: int) -> np.ndarray:
        """
        For every tree in `rf`, reconstructs how many times each of the
        `n_samples` training rows was drawn into that tree's own bootstrap
        sample, using the same private sklearn helpers the forest itself
        used at fit time (each tree's `random_state` was fixed by the
        forest, so this reconstruction is exact, not approximate).

        Returns an (n_estimators, n_samples) integer array: row b, column i
        is N_{b,i} in the module docstring's notation.
        """
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, rf.max_samples)
        counts = np.zeros((len(rf.estimators_), n_samples), dtype=float)
        for b, tree in enumerate(rf.estimators_):
            sample_indices = _generate_sample_indices(tree.random_state, n_samples, n_samples_bootstrap)
            counts[b] = np.bincount(sample_indices, minlength=n_samples)
        return counts

    def predict(self, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Returns dict: target name -> (mean, std), each shape (n_points,).
        Std is the std-dev across individual trees' predictions at each point.
        Used for the objective (EI needs a continuous mean/std), and
        available for any target even when predict_violation_probability()
        is also used for it.
        """
        out = {}
        for name, rf in self.models.items():
            # shape: (n_estimators, n_points)
            tree_preds = np.stack([tree.predict(X) for tree in rf.estimators_], axis=0)
            mean = tree_preds.mean(axis=0)
            std = tree_preds.std(axis=0)
            std = np.maximum(std, 1e-6)  # avoid divide-by-zero in EI/CDF below
            out[name] = (mean, std)
        return out

    def predict_violation_probability(self, X: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Vote-fraction estimate of P(target `name` > 0 | x) plus its
        infinitesimal-jackknife standard deviation, for every point in X.
        `name` must have been included in `ij_targets` at the last fit()
        call. Returns (p_hat, sigma), each shape (n_points,).
        """
        if name not in self._bootstrap_counts:
            raise ValueError(
                f"'{name}' has no cached bootstrap composition - pass it in "
                f"ij_targets= when calling fit() before requesting its "
                f"violation probability."
            )
        rf = self.models[name]
        counts = self._bootstrap_counts[name]  # (B, n_train)
        B, n_train = counts.shape

        # shape: (n_estimators, n_points)
        tree_preds = np.stack([tree.predict(X) for tree in rf.estimators_], axis=0)
        votes = (tree_preds > 0.0).astype(float)  # per-tree binary vote, Eq. (1)'s summand
        p_hat = votes.mean(axis=0)  # Eq. (1)

        # Eq. (2): covariance, per training row, between that row's
        # bootstrap draw count and the tree's vote, vectorized across every
        # candidate point at once.
        n_centered = counts - 1.0  # bootstrap draw count centered on its expectation of 1
        v_centered = votes - p_hat[np.newaxis, :]  # (B, n_points)
        cov = (n_centered.T @ v_centered) / B  # (n_train, n_points)

        # Eq. (3): sum of squared covariances, bias-corrected for finite B.
        var_ij = np.sum(cov ** 2, axis=0)
        bias_correction = (n_train / B ** 2) * np.sum(v_centered ** 2, axis=0)
        var_ij = var_ij - bias_correction
        var_ij = np.maximum(var_ij, 1e-8)  # the correction can push slightly negative; floor it

        return p_hat, np.sqrt(var_ij)


# --------------------------------------------------------------------------
# 3. The constrained EI acquisition function
# --------------------------------------------------------------------------

class ConstrainedEI(AbstractAcquisitionFunction):
    """
    acquisition(x) = EI(x; objective) - alpha * sum_j pi_j(x)

    See the module docstring for what pi_j(x) is and why the constraints
    are combined this way rather than as a probability product.

    `history_provider` is a zero-arg callable returning your own log of
    dicts (the `history` list built up by record_result() in
    optimisation_pipeline.py), each with a "config_object" key plus one key
    per target_name (objective + constraints). Re-fitting only happens when
    new points have arrived since the last fit, so this is safe to call
    every iteration without wasted work.
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
        alpha: float = 1.0,  # HYPERPARAMETER: MERIT_PENALTY_ALPHA - converts the
                                # probability-scale merit term sum_j pi_j(x) into
                                # the objective's own DALYs-scale units; see
                                # optimisation_parameters.py's own description of
                                # MERIT_PENALTY_ALPHA for how to calibrate it.
        tau: float = 0.5,  # HYPERPARAMETER: MERIT_VIOLATION_THRESHOLD - tolerance,
                             # on the predicted violation-probability axis [0, 1],
                             # below which an elevated-but-small violation
                             # probability contributes nothing to the merit term;
                             # see optimisation_parameters.py's own description of
                             # MERIT_VIOLATION_THRESHOLD.
    ):
        super().__init__()
        self._configspace = configspace
        self._objective_name = objective_name
        self._constraint_names = list(constraint_names)
        self._history_provider = history_provider
        self._xi = xi
        self._retrain_every = retrain_every
        self._alpha = alpha
        self._tau = tau
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
        self._surrogate.fit(X, targets, ij_targets=self._constraint_names)
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
        feasible_configs = [
            h["config_object"] for h in history
            if all(h[c] <= 0 for c in self._constraint_names)
        ]
        if feasible_configs:
            X_feasible = configs_to_array(feasible_configs)
            mean_pred, _ = self._surrogate.predict(X_feasible)[self._objective_name]
            self._eta = float(np.min(mean_pred))
        else:
            self._eta = None
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

        # --- Expected Improvement on the objective (minimization) ---
        if self._eta is None:
            # No feasible point observed yet: pure exploration signal,
            # so the search is pushed toward reducing predictive variance
            # rather than chasing an EI target we can't define yet.
            ei = std_obj
        else:
            improvement = self._eta - mean_obj - self._xi
            z = improvement / std_obj
            ei = improvement * norm.cdf(z) + std_obj * norm.pdf(z)
            ei = np.maximum(ei, 0.0)

        # --- Per-constraint merit term, vote-fraction + IJ, summed ---
        # (module docstring Eqs. 1-5) - replaces the multiplicative
        # prod_j P(feasible_j(x)) read.
        merit_penalty = np.zeros(X.shape[0])
        for name in self._constraint_names:
            p_hat, sigma = self._surrogate.predict_violation_probability(X, name)
            sigma = np.maximum(sigma, 1e-6)
            z = (p_hat - self._tau) / sigma
            pi_j = (p_hat - self._tau) * norm.cdf(z) + sigma * norm.pdf(z)
            pi_j = np.maximum(pi_j, 0.0)
            merit_penalty += pi_j

        acquisition_value = ei - self._alpha * merit_penalty
        return acquisition_value.reshape(-1, 1)


# --------------------------------------------------------------------------
# 4. Example wiring
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

        cost_violation = max(0.0, cost / COST_LIMIT - 1)
        hr_violation = max(0.0, hr_used / HR_LIMIT - 1)
        stock_violation = max(0.0, stock_used / STOCK_LIMIT - 1)

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
        # function above, not in this number.
        K = 3 * dalys  # placeholder - tune per the process discussed earlier
        penalty = K * (cost_violation + hr_violation + stock_violation)
        return dalys + penalty

    acquisition_function = ConstrainedEI(
        configspace=configspace,
        objective_name="dalys",
        constraint_names=["cost_violation", "hr_violation", "stock_violation"],
        history_provider=lambda: history,
        alpha=1.0,
        tau=0.5,
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
        if h["cost_violation"] == 0 and h["hr_violation"] == 0 and h["stock_violation"] == 0
    ]
    best = min(feasible, key=lambda h: h["dalys"])
    return best


if __name__ == "__main__":
    example_usage()
