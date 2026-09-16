# --------------------------------------------------------------------------
# TLO / hyperparameters
# --------------------------------------------------------------------------
YEAR_START_DATE = 2010 # Year in which overall (from suspend) tlo sim starts
YEAR_END_DATE = 2013 # Year in which overall (including resume) tlo sim ends.
# Note: this is the last *full* year considered (i.e. TLO sim completes on YEAR_END_DATE+1/01/01
CONFIG_YEAR_START_DATE = 2011 # Year in which configuration changes are enforced
POP_SIZE = 1000 # Population size simulated
START_FIRST_BOUNDARY = CONFIG_YEAR_START_DATE
START_SECOND_BOUNDARY = 2012
START_THIRD_BOUNDARY = 2013
END_THIRD_BOUNDARY = YEAR_END_DATE

# --------------------------------------------------------------------------
# SMAC / search hyperparameters
# --------------------------------------------------------------------------
N_TRIALS = 5                 # total trial budget - not the same as "number of
                             # distinct configs explored", given MAX_CONFIG_CALLS
MAX_CONFIG_CALLS = 3         # caps how many seeds the intensifier will use to
                             # confirm any single config. SHARED with
                             # checkpoint_seeds.py's CHECKPOINT_SEEDS - both
                             # import this SAME value, so the size of the
                             # pre-resume checkpoint pool always matches what
                             # the real intensifier can actually request; these
                             # must never be set independently of each other.
RETRAIN_EVERY = 1            # how many new results accumulate before
                             # ConstrainedEI's surrogate refits - refitting is
                             # cheap relative to simulation cost, so there's no
                             # reason to tolerate staleness (see earlier design
                             # discussion)
EI_XI = 0.0                  # ConstrainedEI's exploration/exploitation
                             # trade-off - higher requires more expected
                             # improvement before a candidate is favoured; not
                             # yet tuned
PENALTY_COEFFICIENT_MULTIPLIER = 3  # K = PENALTY_COEFFICIENT_MULTIPLIER * dalys,
                             # the penalty coefficient in record_result()'s
                             # TrialValue - rough, not load-bearing for search
                             # quality (ConstrainedEI does the real steering),
                             # just keeps smac.incumbent/logging sane

# --------------------------------------------------------------------------
# Operational (Azure submission / polling) hyperparameters
# --------------------------------------------------------------------------
N_CONCURRENT = 3              # concurrent Azure jobs in flight - interacts
                             # with RETRAIN_EVERY; several jobs finishing in
                             # the same polling pass can mean refitting more
                             # often than intended
POLL_INTERVAL_SECONDS = 10   # trades API call frequency against latency
                             # between job completion and SMAC seeing it

# --------------------------------------------------------------------------
# Convergence-check hyperparameters (see convergence_monitoring.py)
# --------------------------------------------------------------------------
CONVERGENCE_WINDOW = 15                     # how many completed trials back
                                             # to compare against
CONVERGENCE_MIN_RELATIVE_IMPROVEMENT = 0.01 # required fractional improvement
                                             # over that window to keep going
                                             # (0.01 = 1%)

# --------------------------------------------------------------------------
# Suspend/resume toggles
# (https://github.com/UCL/TLOmodel/wiki/Suspend-and-resume-simulations)
# --------------------------------------------------------------------------
USE_SUSPEND_RESUME = True   # whether REAL trials resume from a pre-resume
                             # checkpoint (see submit_azure_job()) - if False,
                             # every trial runs a full, fresh simulation,
                             # completely independent of SUBMIT_SUSPEND_PART
                             # below
SUBMIT_SUSPEND_PART = True  # whether to submit the checkpoint-generation
                             # jobs (the "first part" of suspend/resume) THIS
                             # run - a plain user-controlled toggle, not
                             # derived from checking what's already present on
                             # Azure: set True when you actually want
                             # checkpoints (re)generated, False otherwise. Set
                             # this and USE_SUSPEND_RESUME independently -
                             # e.g. True/False to generate checkpoints without
                             # yet using them for real trials, or False/True
                             # to use already-generated checkpoints without
                             # regenerating them.
VALID_CHECKPOINT_COMMITS: list[str] = [
    # Full (or 12+ char) commit hashes whose ALREADY-GENERATED checkpoints
    # are still considered acceptable to reuse, even when the pipeline is
    # currently running under a DIFFERENT (e.g. newer) commit - e.g. a
    # later commit only changed something that doesn't affect the
    # pre-resume portion of the simulation at all. The CURRENT commit is
    # always checked FIRST, automatically - only list PAST commits here,
    # in the order you'd prefer them tried if the current commit has no
    # checkpoint of its own yet. See checkpoint_seeds.checkpoint_job_id()
    # and optimisation_pipeline.find_checkpoint_commit_for_seed().
]

# --------------------------------------------------------------------------
# Baseline run: a standard (non-suspend/resume), 10-differently-seeded-run
# submission of smac_scenario_baseline.py, used ONLY to derive the budget
# constraint values themselves (the upper 95% CI, across those 10 runs,
# of HIV-HRH and HIV-consumable cost) - written into COST_LIMITS_FILE
# before the real optimisation loop starts. See
# postprocess_output.compute_and_save_baseline_budgets() and
# optimisation_pipeline.submit_baseline_job().
# --------------------------------------------------------------------------
SUBMIT_BASELINE_RUN = True   # plain user toggle, same philosophy as
                             # SUBMIT_SUSPEND_PART - NOT derived from
                             # checking what's already in COST_LIMITS_FILE.
                             # Defaults to False (unlike SUBMIT_SUSPEND_PART)
                             # since this is a genuinely costly step (10
                             # full, non-resumed runs) you don't want
                             # repeated every time you just want to re-run
                             # the optimisation loop with an already-good
                             # budget file already in place - set True only
                             # when you actually want the budgets
                             # (re)derived from a fresh baseline run.

# --------------------------------------------------------------------------
# Budgets file: one row per year, columns year,hiv_dalys,hiv_hrh_budget,
# hiv_consumable_budget - read by initialise.py, WRITTEN by
# postprocess_output.compute_and_save_baseline_budgets() when
# SUBMIT_BASELINE_RUN is True. Lives here (not in initialise.py, where it
# used to be defined) so postprocess_output.py can import it too, without
# creating a circular import (initialise.py already imports FROM
# postprocess_output.py).
# --------------------------------------------------------------------------
COST_LIMITS_FILE = "cost_limits_by_year.csv"

# Small JSON summary of the baseline run's own results, written
# alongside COST_LIMITS_FILE by compute_and_save_baseline_budgets() -
# specifically the baseline's median HIV DALYs (same TARGET_PERIOD metric
# every real trial's own "dalys" field already is), since that value
# has no other persistent home otherwise - the baseline job's own
# Azure output isn't kept around once postprocessed. Used by
# evaluate_pipeline_run.py's plotting to draw the baseline comparison
# line - read from disk, independently, matching that file's own
# "never touches the live process" design.
BASELINE_SUMMARY_FILE = "baseline_summary.json"



