"""
Multi-surrogate constrained Bayesian optimization for SMAC3.

Fits a SEPARATE random-forest surrogate for the objective (DALYs) and for
each constraint (returned as a non-negative "violation" amount, 0 = feasible),
then combines them into a single acquisition value:

    acquisition(x) = EI(x; dalys model)  *  prod_j P(feasible_j(x))

This lets the optimizer reason about each constraint's probability of being
satisfied independently, rather than folding everything into one penalized
scalar before the model ever sees it.

NOTE ON VERSION SENSITIVITY
----------------------------
SMAC3's `AbstractAcquisitionFunction` internals (exact `_compute` array shape,
how `self.model` / `self.eta` get set via `update()`) have changed across 2.x
releases. This is written against the general v2 architecture. If wiring this
into your installed version raises an AttributeError or shape mismatch, open
`smac/acquisition/function/expected_improvement.py` in your installed package
and match this class's `_compute` signature/shape to that file - the maths
below (predict -> EI * prod(P(feasible))) will still be correct, only the
plumbing around it might need a one-line tweak.

HYPERPARAMETERS: every tunable knob in this file is marked inline with a
"HYPERPARAMETER" comment - grep for that tag across all three files
(constrained_ei.py, smac_scenario.py, ask_tell_azure_example.py) to find
the complete list in one pass.
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
    constraint), and exposes predictive mean + std for each, estimated
    from the spread of per-tree predictions (this is what stands in for
    a GP's posterior variance when using random forests).
    """

    def __init__(
        self,
        target_names: Sequence[str],
        n_estimators: int = 100,       # HYPERPARAMETER: more trees = smoother/
                                          # more stable mean+std, linear compute cost
        min_samples_leaf: int = 3,     # HYPERPARAMETER: noise-smoothing strength -
                                          # see the grouped-CV + leave-one-seed-out
                                          # validation approach discussed earlier
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
        Std is the std-dev across individual trees' predictions at each point.
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


# --------------------------------------------------------------------------
# 3. The constrained EI acquisition function
# --------------------------------------------------------------------------

class ConstrainedEI(AbstractAcquisitionFunction):
    """
    acquisition(x) = EI(x; objective) * prod_j P(constraint_j(x) <= 0)

    `history_provider` is a zero-arg callable returning your own log of
    dicts (the `history` list built up in target_function), each with a
    "config" key plus one key per target_name (objective + constraints).
    Re-fitting only happens when new points have arrived since the last
    fit, so this is safe to call every iteration without wasted work.
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
    ):
        super().__init__()
        self._configspace = configspace
        self._objective_name = objective_name
        self._constraint_names = list(constraint_names)
        self._history_provider = history_provider
        self._xi = xi
        self._retrain_every = retrain_every
        self._last_fit_n = 0  # history length at last fit, distinct from
                                # self._surrogate.n_fitted_points

        self._surrogate = MultiSurrogateModel(
            target_names=[objective_name, *self._constraint_names]
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

        # --- Probability of feasibility, per constraint, multiplied ---
        prob_feasible = np.ones(X.shape[0])
        for name in self._constraint_names:
            mean_c, std_c = preds[name]
            # P(violation <= 0) under a Gaussian approx of the RF ensemble
            prob_feasible *= norm.cdf((0.0 - mean_c) / std_c)

        acquisition_value = ei * prob_feasible
        return acquisition_value.reshape(-1, 1)


# --------------------------------------------------------------------------
# 4. Example wiring
#
# NOTE: this synchronous, single-process example (calling smac.optimize()
# with a blocking target_function) is kept here only to show ConstrainedEI
# in isolation - the acquisition function itself doesn't care how it's
# driven. For the real TLOmodel/Azure Batch integration, which uses the
# ask-tell interface instead of optimize() (since simulations run as
# async remote jobs, not local blocking calls), see
# ask_tell_azure_example.py and smac_scenario.py - those are the current,
# up-to-date wiring; this function is a minimal standalone sanity-check
# only, e.g. for testing ConstrainedEI against a toy local objective.
# --------------------------------------------------------------------------

def example_usage():
    """
    Minimal synchronous sketch, for testing ConstrainedEI in isolation
    with a cheap local objective. NOT the pattern used for the real
    Azure-based simulation - see ask_tell_azure_example.py for that.
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
