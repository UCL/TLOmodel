# --------------------------------------------------------------------------
# TLO / hyperparameters
# --------------------------------------------------------------------------
YEAR_START_DATE = 2010 # Year in which overall (from suspend) tlo sim starts
YEAR_END_DATE = 2014 # Year in which overall (including resume) tlo sim ends
CONFIG_YEAR_START_DATE = 2011 # Year in which configuration changes are enforced
POP_SIZE = 1000 # Population size simulated
START_FIRST_BOUNDARY = CONFIG_YEAR_START_DATE
START_SECOND_BOUNDARY = YEAR_END_DATE
START_THIRD_BOUNDARY = YEAR_END_DATE + 2
END_THIRD_BOUNDARY = YEAR_END_DATE + 3

# --------------------------------------------------------------------------
# SMAC / search hyperparameters
# --------------------------------------------------------------------------
N_TRIALS = 10                # total trial budget - not the same as "number of
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
MIN_SAMPLES_LEAF = 3         # ConstrainedEI's underlying RandomForestRegressors'
                             # (one per objective/constraint) own noise-smoothing
                             # strength - higher requires more samples per leaf
                             # before the surrogate will split further, trading
                             # sensitivity to real signal for robustness against
                             # per-seed simulation stochasticity (relevant
                             # directly to how noisy DALYs/cost are at small
                             # POP_SIZE - a smaller pop_size means noisier
                             # per-seed results, which argues for a HIGHER
                             # min_samples_leaf to compensate, not a lower one).
PENALTY_COEFFICIENT_MULTIPLIER = 3  # K = PENALTY_COEFFICIENT_MULTIPLIER * dalys,
                             # the penalty coefficient in record_result()'s
                             # TrialValue - rough, not load-bearing for search
                             # quality (ConstrainedEI does the real steering),
                             # just keeps smac.incumbent/logging sane
INFEASIBILITY_FLOOR_MULTIPLIER = 100  # ANY infeasibility (total violation > 0,
                             # however small) adds a FLAT
                             # INFEASIBILITY_FLOOR_MULTIPLIER * dalys on top of
                             # the existing proportional K * total_violation
                             # penalty above - confirmed necessary directly
                             # from a real case: with a small enough total
                             # violation, K * total_violation barely moves
                             # cost above the trial's own raw dalys, meaning a
                             # trivially-infeasible config can still look
                             # "better" (lower cost) than a genuinely feasible
                             # config with higher dalys - and SMAC's OWN
                             # intensifier (deciding whether to confirm an
                             # already-proposed challenger with another seed,
                             # and whether it beats the incumbent) has NO
                             # separate awareness of feasibility at all,
                             # unlike ConstrainedEI (used only for choosing
                             # NEW candidates) - it relies purely on this
                             # scalar, so without this floor it can end up
                             # spending extra seeds confirming an infeasible
                             # config. Self-scaling (multiplies THIS trial's
                             # own dalys, not a fixed constant) so it stays
                             # proportionate regardless of the objective's
                             # actual scale. Deliberately NOT np.inf (used
                             # elsewhere for genuinely CRASHED trials, which
                             # SMAC's own runhistory encoder explicitly
                             # excludes from surrogate training via
                             # considered_states) - a SUCCESSFUL-but-infeasible
                             # trial gets no such exclusion, so a literally
                             # infinite cost here would flow straight into
                             # SMAC's own RF surrogate, risking the exact
                             # "Input y contains NaN" crash seen earlier in
                             # this project.

# --------------------------------------------------------------------------
# Operational (Azure submission / polling) hyperparameters
# --------------------------------------------------------------------------
N_CONCURRENT = 3              # concurrent Azure jobs in flight - interacts
                             # with RETRAIN_EVERY; several jobs finishing in
                             # the same polling pass can mean refitting more
                             # often than intended
POLL_INTERVAL_SECONDS = 20   # trades API call frequency against latency
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

VALID_PRIOR_RUN_COMMITS: list[str] = [
    # Full (or 12+ char) commit hashes whose ALREADY-SUBMITTED real trial
    # jobs (in submitted_jobs.jsonl) are still considered safe to recover
    # into history, even when the pipeline is currently running under a
    # DIFFERENT (e.g. newer) commit - see
    # optimisation_pipeline.recover_from_job_log(). The CURRENT commit is
    # always checked first, automatically - only list PAST commits here.
    #
    # DELIBERATELY SEPARATE from VALID_CHECKPOINT_COMMITS above, despite
    # the identical structure/usage pattern - "safe to reuse a checkpoint
    # generated under this commit" (only the pre-resume portion of the
    # simulation needs to be unchanged) and "safe to recover a completed
    # real trial's result from this commit" (the ENTIRE simulation logic,
    # including draw_parameters()'s own config->module mapping, needs to
    # be unchanged) are genuinely different claims about a commit - one
    # holding doesn't imply the other. Point this at the same list as
    # VALID_CHECKPOINT_COMMITS if you're confident both always hold
    # together for your own commit history; kept separate here since
    # that's not true in general.
]

# --------------------------------------------------------------------------
# Baseline run: a standard (non-suspend/resume), 10-differently-seeded-run
# submission of smac_scenario_baseline.py, used ONLY to derive the budget
# constraint values themselves (the upper 95% CI, across those 10 runs,
# of HIV-HRH and HIV-consumable cost) - written into COST_LIMITS_FILE
# before the real optimisation loop starts. See
# postprocess_output.compute_and_save_baseline_budgets() and
# optimisation_pipeline.submit_baseline_job().
#
# TODO / KNOWN AWKWARDNESS, worth consolidating later: this is one of TWO
# genuinely separate "baseline" mechanisms this project currently has,
# which don't share any code or data with each other despite both
# representing the same underlying status-quo scenario - see the OTHER
# one, SUBMIT_INITIAL_DESIGN below, and submit_baseline_job()'s own
# docstring in optimisation_pipeline.py for the full picture. Kept
# deliberately separate for now.
# --------------------------------------------------------------------------
SUBMIT_BASELINE_RUN = False  # plain user toggle, same philosophy as
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
# Initial design: N_INIT jobs (N_INIT-1 randomly-sampled configs, plus the
# baseline itself, ONE seed each) submitted and waited on BEFORE the main
# ask-tell loop starts, when SUBMIT_INITIAL_DESIGN is True - giving
# ConstrainedEI's surrogates and SMAC's own runhistory real, diverse data
# to work with from trial one, rather than starting from nothing. See
# optimisation_pipeline.submit_initial_design_jobs() - these jobs are
# submitted, waited on, then picked up by the EXISTING
# recover_from_job_log() mechanism, exactly like any other previously-
# submitted job (same commit-matching, same restart-deduplication) -
# no separate loading path.
#
# TODO / KNOWN AWKWARDNESS, worth consolidating later: this function's own
# baseline point (BASELINE_CONFIG_VALUES, initialise.py) is submitted
# through the STANDARD smac_scenario.py path (suspend/resume, single
# seed) - a SEPARATE mechanism from SUBMIT_BASELINE_RUN above (10-run,
# non-suspend/resume, used only to derive the budget). Both represent
# the same underlying status-quo scenario, but currently share no code
# or data. Kept deliberately separate for now.
# --------------------------------------------------------------------------
SUBMIT_INITIAL_DESIGN = True  # plain user toggle, same philosophy as
                             # SUBMIT_SUSPEND_PART/SUBMIT_BASELINE_RUN -
                             # set True when you want this step to actually
                             # run, False to skip it (e.g. once it's
                             # already been done and its results are
                             # already recoverable from JOB_LOG_FILE under
                             # an acceptable commit).
N_INIT = 5  # total initial-design jobs, INCLUDING the baseline - so
                             # N_INIT-1 are randomly sampled from
                             # configspace. Must be <= MAX_CONFIG_CALLS + 1:
                             # seeds are drawn from CHECKPOINT_SEEDS (so
                             # each job can genuinely use suspend/resume,
                             # and stays comparable to anything SMAC's own
                             # intensifier later draws the same seed for),
                             # a pool of exactly MAX_CONFIG_CALLS distinct
                             # values - N_INIT random configs need N_INIT-1
                             # of those seeds (one each, the baseline takes
                             # the last slot).

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



