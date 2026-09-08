"""
Maps a SMAC-issued seed (info.seed) to the run number its corresponding
pre-resume checkpoint was generated and stored under - for use with
TLOmodel's suspend/resume feature
(https://github.com/UCL/TLOmodel/wiki/Suspend-and-resume-simulations).

Kept separate from optimisation_pipeline.py's submission/polling/ask-tell
orchestration, same rationale as the other standalone modules in this
project - this can be read, tested, or reused independently.

BACKGROUND, for why this mapping exists at all:
SMAC's Intensifier compares a challenger against the incumbent using
runhistory.average_cost() over the INTERSECTION of seeds each has been
evaluated on (confirmed directly from smac/intensifier/abstract_intensifier.py's
_compare_configs) - meaning genuine seed REUSE across different configs
is required for comparisons to ever be possible at all, not incidental.
Empirically (see submitted_jobs.jsonl from a real run), only a small,
bounded number of distinct seed values appeared across 26 distinct
configs - consistent with SMAC drawing from a small internal pool via
self._rng.randint(...), where self._rng = np.random.RandomState(seed)
and seed defaults to scenario.seed (confirmed = 0, since
Scenario(configspace, n_trials=100, deterministic=False) in
optimisation_pipeline.py never passes seed= explicitly) - confirmed
directly from AbstractIntensifier.__init__ in SMAC3's own source, and
verified empirically: a fresh run's first two real seeds matched
RandomState(0).randint(...)'s first two outputs exactly.

Given that, this file independently REPLICATES that same
RandomState(0) draw (rather than reading it back from SMAC, which isn't
exposed) to precompute the small set of seed values SMAC is expected to
reuse - so a small pool of pre-resume checkpoints (one per seed) can be
generated ONCE, ahead of time, and referenced by every trial that later
gets one of those same seeds - rather than recomputing the expensive
pre-resume portion of the simulation on every single trial.

CONFIDENCE CAVEAT, worth keeping in mind: this replication is verified
against only the first TWO real seeds observed so far, not against a
full max_config_calls-length run. It also depends on scenario.seed
staying 0 (unset) going forward - if that Scenario(...) call is ever
changed to pass an explicit seed=, this whole mapping silently
desynchronizes from what SMAC actually issues. run_number_for_seed()
raises loudly (KeyError) rather than guessing, specifically to surface
that kind of desync immediately if it ever happens, rather than
silently referencing the wrong (or no) checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_CONFIG_CALLS = 5  # must match max_config_calls used by the real SMAC intensifier
                       # (see get_intensifier(scenario, max_config_calls=...) in
                       # optimisation_pipeline.py) - this is the assumed upper
                       # bound on how many distinct seeds SMAC's pool will ever
                       # contain; not yet fully confirmed, see module docstring.

# The exact, ordered sequence of seed values SMAC is expected to draw
# from RandomState(scenario.seed). Computed ONCE, independently -
# RandomState(0) is fully deterministic, so run_number is simply this
# list's own index, not something read back from SMAC itself.
CHECKPOINT_SEEDS = [
    int(s) for s in np.random.RandomState(0).randint(low=0, high=2**31 - 1, size=MAX_CONFIG_CALLS)
]

# seed -> run_number. A plain lookup table, not a true inverse of
# RandomState.randint (which has no closed form) - this works because
# the forward set is small and fully known in advance (MAX_CONFIG_CALLS
# values), so a table is both correct and trivially cheap to build.
_RUN_NUMBER_BY_SEED = {seed: i for i, seed in enumerate(CHECKPOINT_SEEDS)}


def run_number_for_seed(seed: int) -> int:
    """
    Maps a SMAC-issued seed (info.seed) to the run number its
    corresponding pre-resume checkpoint was generated and stored under.

    Raises KeyError loudly (not silently) if `seed` isn't one of the
    MAX_CONFIG_CALLS precomputed values - this means either (a)
    MAX_CONFIG_CALLS is set smaller than the intensifier's actual seed
    pool size, or (b) the RandomState(0) replication has desynchronized
    from SMAC's actual internal sequence (e.g. scenario.seed no longer
    being 0). Both are worth surfacing immediately rather than silently
    referencing a missing or mismatched checkpoint.
    """
    if seed not in _RUN_NUMBER_BY_SEED:
        raise KeyError(
            f"seed {seed} is not among the {MAX_CONFIG_CALLS} precomputed checkpoint "
            f"seeds {CHECKPOINT_SEEDS}. Either MAX_CONFIG_CALLS needs increasing, or "
            f"the RandomState(0) replication has desynchronized from SMAC's actual "
            f"seed sequence for this run."
        )
    return _RUN_NUMBER_BY_SEED[seed]


def checkpoint_path_for_seed(checkpoint_root: str | Path, seed: int) -> Path:
    """
    Local/relative path to the pre-resume checkpoint for a given
    SMAC-issued seed, following the <checkpoint_root>/0/<run_number>
    convention (draw=0, matching this pipeline's number_of_draws=1).
    """
    run_number = run_number_for_seed(seed)
    return Path(checkpoint_root) / "0" / str(run_number)
