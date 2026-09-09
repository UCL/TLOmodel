"""
Deterministic, seed-derived naming for pre-resume checkpoints - for use
with TLOmodel's suspend/resume feature
(https://github.com/UCL/TLOmodel/wiki/Suspend-and-resume-simulations).

Kept separate from optimisation_pipeline.py's submission/polling/ask-tell
orchestration, same rationale as the other standalone modules in this
project - this can be read, tested, or reused independently.

BACKGROUND, for why this file exists at all:
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

WHY SEED-DERIVED NAMING, NOT job_id+run_number:
TLO's resume feature does NOT accept a seed at load time
(Simulation.load_from_pickle(pickle_path, log_config=None) - no seed
parameter at all) - the pickled Simulation's RNG state is already
whatever it was at suspend time. So the ONLY way a resumed trial's
post-resume randomness can depend on its own seed is by resuming from
the CHECKPOINT generated under that exact seed. And matching TLO's own
low_bias_32(scenario_seed + sample_number) formula requires each
checkpoint be generated as its OWN single-run submission
(sample_number=0, scenario.seed set DIRECTLY to the target seed) - NOT
batched as multiple runs under one shared scenario, which would instead
vary sample_number under a fixed scenario_seed and produce entirely
different (and wrong) low_bias_32 outputs.

Since each checkpoint is therefore its own separate Azure job, Azure's
usual per-job naming (filename+timestamp+uuid, see submit_azure_job) is
deliberately UNPREDICTABLE, to avoid collisions - which means it can't
be reconstructed later purely from a seed value without keeping a
separate mapping around. Using a DETERMINISTIC, seed-derived job id
instead sidesteps that: given a seed, the checkpoint's location can be
computed directly, with no lookup table needed for path construction at
all - CHECKPOINT_SEEDS is only needed to know WHICH seeds to generate
checkpoints for in the first place, not to find them again afterwards.

CONFIDENCE CAVEAT, worth keeping in mind: this replication is verified
against only the first TWO real seeds observed so far, not against a
full max_config_calls-length run. It also depends on scenario.seed
staying 0 (unset) going forward - if that Scenario(...) call is ever
changed to pass an explicit seed=, this whole mapping silently
desynchronizes from what SMAC actually issues. is_known_checkpoint_seed()
exists specifically to surface that kind of desync loudly, rather than
silently referencing the wrong (or no) checkpoint.
"""

from __future__ import annotations

import numpy as np

from optimisation_parameters import MAX_CONFIG_CALLS  # SHARED with
    # optimisation_pipeline.py's get_intensifier(max_config_calls=...) - both
    # import this SAME value now, rather than being set independently (as
    # this file used to do) and relying on a comment to keep them in sync.

# The exact, ordered sequence of seed values SMAC is expected to draw
# from RandomState(scenario.seed). Computed ONCE, independently -
# RandomState(0) is fully deterministic, so this doesn't need to be read
# back from SMAC itself, only replicated ahead of time. Used by the
# (separate) checkpoint-generation step to know which seeds to generate
# checkpoints for - NOT used for path construction, which is derived
# directly from a given seed instead (see checkpoint_job_id() below).
CHECKPOINT_SEEDS = [
    int(s) for s in np.random.RandomState(0).randint(low=0, high=2**31 - 1, size=MAX_CONFIG_CALLS)
]
_CHECKPOINT_SEEDS_SET = set(CHECKPOINT_SEEDS)  # O(1) membership check, see is_known_checkpoint_seed()


def is_known_checkpoint_seed(seed: int) -> bool:
    """True if `seed` is one of the MAX_CONFIG_CALLS precomputed values a
    checkpoint should exist for."""
    return seed in _CHECKPOINT_SEEDS_SET


def checkpoint_job_id(seed: int, commit: str) -> str:
    """
    Deterministic, seed+commit-derived Azure job id for this seed's
    pre-resume checkpoint - deliberately NOT the usual
    filename+timestamp+uuid scheme submit_azure_job() uses for real
    trials (that scheme is intentionally unpredictable, to avoid
    collisions across many concurrent submissions). This one needs to
    be predictable instead, so a resuming trial can compute it directly
    from its own seed with no lookup table involved.

    COMMIT IS PART OF THE ID, not just a label. An earlier version of
    this function used seed alone - which meant regenerating checkpoints
    under a NEW commit collided with the OLD commit's already-existing
    job of the same name (Azure jobs can't be recreated/overwritten
    under an id already in use). Including the commit means different
    commits naturally get different, non-colliding checkpoint jobs -
    and lets optimisation_pipeline.py's checkpoint-selection logic
    search across several ACCEPTED commits
    (see VALID_CHECKPOINT_COMMITS in optimisation_parameters.py) to find
    and reuse an existing checkpoint from an older, still-trusted
    commit, rather than being forced to regenerate one every time the
    commit changes.

    Truncated to the first 12 hex characters (matching this project's
    existing convention for displaying commit hashes elsewhere) - short
    enough to keep the job id a reasonable length; a collision between
    two genuinely different commits' first 12 hex characters isn't a
    realistic concern.

    Raises ValueError if `seed` isn't a known checkpoint seed - same
    "fail loudly on desync" reasoning as elsewhere in this file, applied
    at the point a caller is about to construct a path/job id from an
    unexpected seed value.
    """
    if not is_known_checkpoint_seed(seed):
        raise ValueError(
            f"seed {seed} is not among the {MAX_CONFIG_CALLS} precomputed checkpoint "
            f"seeds {CHECKPOINT_SEEDS}. Either MAX_CONFIG_CALLS needs increasing, or "
            f"the RandomState(0) replication has desynchronized from SMAC's actual "
            f"seed sequence for this run."
        )
    return f"checkpoint-seed{seed}-{commit[:12]}"
