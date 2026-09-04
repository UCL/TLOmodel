"""
Convergence tracking and history-to-disk logging, kept separate from
optimisation_pipeline.py's submission/polling/ask-tell orchestration -
same rationale as constrained_ei.py, smac_scenario.py, and
postprocess_output.py: this can be read, tested, or reused independently
of the orchestration logic that calls it.

Two responsibilities live here:
1. Mirroring every history entry to a local JSONL file as it's produced,
   so progress can be inspected live (tail -f history_log.jsonl) without
   touching the running process.
2. Deciding whether the search has converged - i.e. whether the best
   feasible DALYs found has stopped meaningfully improving over the last
   CONVERGENCE_WINDOW completed trials - so optimisation_pipeline.py can
   stop proposing new configs while still draining whatever's pending.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ConfigSpace import Configuration


# --------------------------------------------------------------------------
# 1. Live history logging
# --------------------------------------------------------------------------

HISTORY_LOG_FILE = Path("history_log.jsonl")  # append-only, one line per
                                                # result - tail -f this file
                                                # to inspect progress live


def json_safe_config(config) -> dict:
    """
    ConfigSpace hands back numpy scalar types (numpy.bool_, numpy.int64,
    numpy.float64, ...) for sampled hyperparameter values - these look
    identical to their Python equivalents but aren't JSON-serialisable,
    which surfaces as a `TypeError: Object of type bool_/int64/... is
    not JSON serializable` the first time any config touches
    json.dumps. Use this everywhere a config gets written to disk,
    instead of a bare dict(config). Lives here (not in
    optimisation_pipeline.py) so both files can use it without a
    circular import - optimisation_pipeline.py already imports several
    things from this module.
    """
    return {
        k: (v.item() if isinstance(v, np.generic) else v)
        for k, v in dict(config).items()
    }


def append_history_to_file(entry: dict) -> None:
    """
    Mirrors a single history entry to disk, immediately after it's
    appended to the in-memory `history` list in optimisation_pipeline.py.
    JSON Lines (one complete record per line) so the file is
    readable/tailable while the run is still in progress, and so a
    crash mid-write only corrupts the last line rather than the whole
    file - same convention as JOB_LOG_FILE. config_object (a ConfigSpace
    Configuration) isn't JSON-serialisable directly, so it's converted
    to a plain dict first via json_safe_config() - a bare dict(config)
    would leave numpy scalar values (numpy.bool_, numpy.int64, ...)
    behind, which json.dumps then can't serialise.
    """
    record = {**entry, "config_object": json_safe_config(entry["config_object"])}
    with open(HISTORY_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------
# 2. Convergence tracking
# --------------------------------------------------------------------------

CONVERGENCE_WINDOW = 15                    # HYPERPARAMETER: how many completed
                                             # trials back to compare against
CONVERGENCE_MIN_RELATIVE_IMPROVEMENT = 0.01 # HYPERPARAMETER: required fractional
                                             # improvement over that window to
                                             # keep going (0.01 = 1%)


def config_key(config: Configuration) -> tuple:
    """
    Hashable, order-independent identity for grouping history by config.
    Not underscore-prefixed since it's a shared utility used both here
    and in optimisation_pipeline.py's final-selection grouping.
    """
    return tuple(sorted(dict(config).items()))


def get_best_feasible_dalys(history: list[dict]) -> float | None:
    """
    Groups history by config, averages across whatever seeds each config
    has accumulated so far, and returns the lowest mean DALYs among
    configs that are feasible ON AVERAGE. Returns None if nothing
    feasible has been observed yet - check_convergence() below treats
    that as "not converged" rather than triggering on an undefined
    comparison.
    """
    grouped: dict[tuple, list[dict]] = {}
    for h in history:
        grouped.setdefault(config_key(h["config_object"]), []).append(h)

    best = None
    for entries in grouped.values():
        cost_v = np.mean([e["cost_violation"] for e in entries])
        hr_v = np.mean([e["hr_violation"] for e in entries])
        stock_v = np.mean([e["stock_violation"] for e in entries])
        if cost_v == 0 and hr_v == 0 and stock_v == 0:
            dalys = float(np.mean([e["dalys"] for e in entries]))
            if best is None or dalys < best:
                best = dalys
    return best


def check_convergence(best_dalys_over_time: list[float]) -> bool:
    """
    Returns True if convergence has been detected: less than
    CONVERGENCE_MIN_RELATIVE_IMPROVEMENT relative improvement over the
    last CONVERGENCE_WINDOW completed trials. Prints a message (with the
    actual before/after values) when it triggers. Call this once per
    completed trial, immediately after appending that trial's current
    best-feasible-DALYs value via get_best_feasible_dalys().

    Only reports the stall itself - optimisation_pipeline.py adds its
    own follow-up line about how many pending jobs are being drained,
    since that count isn't something this function has visibility into.
    """
    if len(best_dalys_over_time) <= CONVERGENCE_WINDOW:
        return False

    old_best = best_dalys_over_time[-1 - CONVERGENCE_WINDOW]
    new_best = best_dalys_over_time[-1]
    relative_improvement = (old_best - new_best) / abs(old_best)

    if relative_improvement < CONVERGENCE_MIN_RELATIVE_IMPROVEMENT:
        print(
            f"[convergence] no improvement >= {CONVERGENCE_MIN_RELATIVE_IMPROVEMENT:.1%} "
            f"over the last {CONVERGENCE_WINDOW} completed trials "
            f"({old_best:.4f} -> {new_best:.4f})."
        )
        return True
    return False
