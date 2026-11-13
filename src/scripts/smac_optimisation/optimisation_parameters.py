from pathlib import Path

# --------------------------------------------------------------------------
# TLO / hyperparameters
# --------------------------------------------------------------------------
YEAR_START_DATE = 2010
CONFIG_YEAR_START_DATE = 2020
YEAR_END_DATE = 2025
POP_SIZE = 8000
START_FIRST_BOUNDARY = CONFIG_YEAR_START_DATE
START_SECOND_BOUNDARY = None
START_THIRD_BOUNDARY = None

# --------------------------------------------------------------------------
# SMAC / search hyperparameters
# --------------------------------------------------------------------------
N_TOTAL_RUNS = 50
MAX_CONFIG_RUNS = 5
RETRAIN_EVERY = 1
EI_XI = 0.0
MIN_SAMPLES_LEAF = "auto"
MIN_SAMPLES_LEAF_FLOOR = 3
INTENSIFIER_PROPORTIONAL_PENALTY_MULTIPLIER = 10
INTENSIFIER_FLOOR_PENALTY_MULTIPLIER = 5
CONSTRAINT_COMBINATION = "multiplicative"  # "multiplicative" or "additive"
MERIT_DALYS_IMPROVEMENT_PCT = 10.0         # additive mode only: X in "X% DALYs ...
MERIT_BUDGET_VIOLATION_PCT = 2.0           # ... is worth Y% budget violation" (Y)
MERIT_OVERSHOOT_TOLERANCE = 0.0            # additive mode only

# --------------------------------------------------------------------------
# Operational (Azure submission / polling) hyperparameters
# --------------------------------------------------------------------------
N_CONCURRENT = 20
POLL_INTERVAL_SECONDS = 10

# --------------------------------------------------------------------------
# Convergence-check hyperparameters (see convergence_monitoring.py)
# --------------------------------------------------------------------------
CONVERGENCE_WINDOW = 30
CONVERGENCE_MIN_RELATIVE_IMPROVEMENT = 0.01

# --------------------------------------------------------------------------
# Suspend/resume toggles
# (https://github.com/UCL/TLOmodel/wiki/Suspend-and-resume-simulations)
# --------------------------------------------------------------------------
USE_SUSPEND_RESUME = True
VALID_CHECKPOINT_COMMITS: list[str] = []

# --------------------------------------------------------------------------
# Baseline runs (see ensure_baseline_runs(), compute_and_save_baseline_budgets())
# --------------------------------------------------------------------------
N_BASELINE_RUNS = 10

# --------------------------------------------------------------------------
# Initial design (see submit_initial_design_jobs())
# --------------------------------------------------------------------------
VALID_PRIOR_RUN_COMMITS: list[str] = []
SUBMIT_INITIAL_DESIGN = True
N_INIT = 8
INITIAL_DESIGN_NEAR_BASELINE_FRACTION = 0.2
INITIAL_DESIGN_PERTURBATION_STD = 0.1

# --------------------------------------------------------------------------
# Paths (rarely changed)
# --------------------------------------------------------------------------
# Everything below is anchored to the directory holding this file (expected
# to be the same directory as optimisation_pipeline.py), NOT to the directory
# the pipeline is launched from - so launching from a parent directory
# (e.g. the repository root) still finds and writes the same folders.
#   inputs/   precomputed, baseline-derived files (budgets etc.). OVERWRITTEN
#             at every start, from the baseline runs.
#   logs/     history_log.jsonl, submitted_jobs.jsonl. APPENDED to, never
#             overwritten.
#   outputs/  run outputs downloaded from Azure.
# Each folder is created here if it does not exist.
CODE_DIR = Path(__file__).resolve().parent
LOG_DIR = CODE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_LOG_FILE = LOG_DIR / "history_log.jsonl"
JOB_LOG_FILE = LOG_DIR / "submitted_jobs.jsonl"

INPUTS_DIR = CODE_DIR / "inputs"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR = CODE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
COST_LIMITS_FILE = INPUTS_DIR / "cost_limits_by_year.csv"
BASELINE_SUMMARY_FILE = INPUTS_DIR / "baseline_summary.json"
BASELINE_BUDGET_COMPARISON_FILE = INPUTS_DIR / "baseline_budget_comparison.csv"

# ============================================================================
# PARAMETER DESCRIPTIONS
# Full explanation for every parameter above, in the same order - moved down
# here so the top of the file stays a quick, scannable overview. Search for
# a parameter's name to jump straight to its own description.
# ============================================================================

# --- TLO / hyperparameters -------------------------------------------------
#
# YEAR_START_DATE
#     Year in which overall (from suspend) tlo sim starts
#
# YEAR_END_DATE
#     Year in which overall (including resume) tlo sim ends
#
# CONFIG_YEAR_START_DATE
#     Year in which configuration changes are enforced
#
# POP_SIZE
#     Population size simulated
#
# Constraint periods (absolute calendar years, inclusive; built and checked in
# initialise.py as PERIOD_BOUNDARIES). Each period runs from its START to the
# year before the next START; the last period in use always ends at
# YEAR_END_DATE. Set START_SECOND_BOUNDARY (and START_THIRD_BOUNDARY) to None
# to use fewer periods. Checked on import of initialise.py:
# every start must satisfy CONFIG_YEAR_START_DATE <= start < YEAR_END_DATE,
# starts must increase strictly, and START_THIRD_BOUNDARY may only be set if
# START_SECOND_BOUNDARY is.
#   three periods: START_SECOND_BOUNDARY = 2012, START_THIRD_BOUNDARY = 2013
#   two periods:   START_SECOND_BOUNDARY = 2012, START_THIRD_BOUNDARY = None
#   one period:    START_SECOND_BOUNDARY = None, START_THIRD_BOUNDARY = None

# --- SMAC / search hyperparameters ------------------------------------------
#
# N_TOTAL_RUNS
#     Total trial budget.
#
# MAX_CONFIG_RUNS
#     Caps how many seeds the intensifier will use to confirm any single
#     config.
#
# RETRAIN_EVERY
#     How many new results accumulate before ConstrainedEI's surrogate
#     refits.
#
# EI_XI
#     (xi, in DALYs.) A candidate only earns expected improvement for
#     beating the best feasible predicted DALYs by MORE than xi. 0 is plain
#     EI (SMAC's default) and a sensible start; a positive value moves the
#     search from exploiting towards exploring. It can be changed on any
#     resume: postprocess_output.report_ei_xi_diagnostic() (printed at the
#     end of a run) compares the recent improvement with the seed noise and
#     says whether to raise it.
#
# MIN_SAMPLES_LEAF
#     Minimum number of (config, seed) results per leaf in every forest
#     (one per objective/constraint). Higher = smoother predictions and a
#     wider, more honest sigma, but a coarser model; lower = follows noisy
#     individual results and makes sigma overconfident. Either a positive
#     int, or "auto" (the default), which recomputes it at every refit as
#         round(n_results / (2 * n_parameters)),
#     clipped to [MIN_SAMPLES_LEAF_FLOOR, MAX_CONFIG_RUNS]: about two leaves
#     per tuned parameter per tree. With few results the floor applies;
#     the cap stops a leaf being filled by one config's seeds. A heuristic
#     of this pipeline, not a published rule. The value in use is printed
#     whenever it changes.
#
# MIN_SAMPLES_LEAF_FLOOR
#     Lower bound for "auto"; keeps every leaf able to give a scatter
#     estimate for sigma.
#
# CONSTRAINT_COMBINATION
#     The single switch for how ConstrainedEI combines the objective's
#     expected improvement (EI) with the six constraints. Each constraint
#     value is the signed slack g_j = (highest yearly cost/budget ratio in
#     the period) - 1, feasible when g_j <= 0 (see initialise.py).
#       "multiplicative": acquisition = EI * prod_j P(g_j <= 0). ASSUMES the
#           constraints are statistically independent (uncorrelated); the
#           product shrinks geometrically with the number of constraints.
#           No extra parameters.
#       "additive": acquisition = expected improvement of the MERIT
#           f_0 + alpha * sum_j max(g_j - MERIT_OVERSHOOT_TOLERANCE, 0)
#           over the best feasible merit so far. The penalty enters inside
#           the improvement, so the value stays >= 0 and shrinks with EI as
#           the search converges. Penalises the expected SIZE of the
#           overshoot, grows only linearly with the number of constraints,
#           and the merit mean needs no independence assumption. Needs the
#           exchange rate (MERIT_DALYS_IMPROVEMENT_PCT /
#           MERIT_BUDGET_VIOLATION_PCT) chosen.
#     Only steers which config is evaluated next; the reported answer is
#     always chosen on the recorded g_j <= 0.
#
# MERIT_DALYS_IMPROVEMENT_PCT, MERIT_BUDGET_VIOLATION_PCT
#     ADDITIVE MODE ONLY. A fixed exchange rate for the whole search: an
#     improvement of MERIT_DALYS_IMPROVEMENT_PCT (X) percent of the BASELINE
#     DALYs is worth MERIT_BUDGET_VIOLATION_PCT (Y) percent of budget
#     overshoot (in the worst year of a period). Budgets are the highest
#     yearly cost across the baseline runs, so every baseline run has
#     slack <= 0 (the baseline's median sits below 0). At startup the pipeline converts
#     this into ConstrainedEI's alpha, in DALYs per unit (100%) of expected
#     overshoot:
#         alpha = (X / Y) * baseline median DALYs
#     (baseline median DALYs is read from BASELINE_SUMMARY_FILE, written by
#     the baseline run). Example: X = 10, Y = 2 gives alpha = 5 * baseline
#     DALYs, i.e. a candidate with 1% of expected overshoot must be 5% of
#     baseline DALYs better than the best feasible config to be worth trying. The same rate applies to all six constraints.
#     Only X / Y matters. Larger X / Y = stricter on violations. Because the penalty
#     sits inside the improvement, it no longer overwhelms a small late-run
#     EI, but a rate that is too strict still keeps the search away from the
#     boundary.
#     Not used in multiplicative mode.
#
# MERIT_OVERSHOOT_TOLERANCE
#     (delta in ConstrainedEI.) ADDITIVE MODE ONLY. Fraction of the budget
#     (in cost/budget-ratio units, 0.02 = 2%) by which a constraint's
#     expected overshoot is ignored in the search penalty; 0.0 counts any
#     expected overshoot. Does not change feasibility, which is always
#     g_j <= 0.
#
# INTENSIFIER_PROPORTIONAL_PENALTY_MULTIPLIER, INTENSIFIER_FLOOR_PENALTY_MULTIPLIER
#     Shape the scalar cost SMAC's INTENSIFIER sees for each trial:
#         cost = dalys + K * v + [v > 0] * FLOOR * dalys,
#     K = INTENSIFIER_PROPORTIONAL_PENALTY_MULTIPLIER * dalys, v = total clipped slack.
#     The intensifier has no notion of feasibility beyond this number. The
#     scalar is NOT used by ConstrainedEI (which reads the raw history) nor
#     by the reported answer.
#       - FLOOR: any violation, however small, adds FLOOR * dalys. Without
#         it a barely infeasible config with lower DALYs looks better than
#         a feasible one (the real failure that motivated it), and the
#         intensifier spends seeds confirming it. A jump, not a slope.
#       - K: ranks infeasible trials by overspend (a 10% overspend adds
#         K*0.1), mainly before a feasible incumbent exists.
#     Defaults 5 and 10: 5 makes a config violating on ~20% of its seeds
#     lose to a feasible incumbent (cost ~2x its DALYs) yet does not let one
#     violating seed sink an otherwise good config, so the intensifier is
#     closer to the MEDIAN rule of the final answer; 10 makes a 50% overspend
#     (K*0.5 = 5*dalys) cost about as much as the floor, so the two terms are
#     comparable over realistic violations. Comparing both configs on the
#     same seeds cancels shocks common to both. Tuned on resume with
#     postprocess_output.report_false_rejection_diagnostic(): if many
#     median-feasible, better-DALYs configs stopped before MAX_CONFIG_RUNS
#     seeds, lower the floor (e.g. 3).
#     Why this strictness is safe: the intensifier is the search's
#     gatekeeper, never the source of the reported answer. A config wrongly
#     rejected is a lost candidate; ConstrainedEI proposes from ALL
#     observations and can propose nearby configs again. The final answer
#     needs at least MAX_CONFIG_RUNS seeds, the MEDIAN over seeds and every
#     median slack <= 0, so neither over-rejection nor over-promotion can
#     change what is reported. The EI's eta is independent of both terms.
#     Deliberately NOT np.inf (used elsewhere for genuinely CRASHED
#     trials, which SMAC's own runhistory encoder explicitly excludes
#     from surrogate training via considered_states) - a
#     SUCCESSFUL-but-infeasible trial gets no such exclusion, so a
#     literally infinite cost here would flow straight into SMAC's own RF
#     surrogate, risking the exact "Input y contains NaN" crash seen
#     earlier in this project.

# --- Operational (Azure submission / polling) hyperparameters --------------
#
# N_CONCURRENT
#     Concurrent Azure jobs in flight - interacts with RETRAIN_EVERY;
#     several jobs finishing in the same polling pass can mean refitting
#     more often than intended.
#
# POLL_INTERVAL_SECONDS
#     Trades API call frequency against latency between job completion and
#     SMAC seeing it.

# --- Convergence-check hyperparameters (see convergence_monitoring.py) -----
#
# CONVERGENCE_WINDOW
#     How many completed trials back to compare against.
#
# CONVERGENCE_MIN_RELATIVE_IMPROVEMENT
#     Required fractional improvement over that window to keep going
#     (0.01 = 1%).

# --- Suspend/resume toggles --------------------------------------------------
# (https://github.com/UCL/TLOmodel/wiki/Suspend-and-resume-simulations)
#
# USE_SUSPEND_RESUME
#     Whether EVERY run (baseline runs and trials alike, see
#     submit_azure_job()) resumes from a pre-resume checkpoint. If False,
#     every run is a full, fresh simulation and no checkpoints are made.
#     Checkpoints are never submitted by hand: at startup, any checkpoint
#     that does not exist under the current commit or a commit in
#     VALID_CHECKPOINT_COMMITS is submitted automatically (one per seed in
#     CHECKPOINT_SEEDS), and the pipeline waits for all of them to finish.
#
# VALID_CHECKPOINT_COMMITS
#     Full (or 12+ char) commit hashes whose ALREADY-GENERATED checkpoints
#     are still considered acceptable to reuse, even when the pipeline is
#     currently running under a DIFFERENT (e.g. newer) commit - e.g. a
#     later commit only changed something that doesn't affect the
#     pre-resume portion of the simulation at all. The CURRENT commit is
#     always checked FIRST, automatically - only list PAST commits here,
#     in the order you'd prefer them tried if the current commit has no
#     checkpoint of its own yet. See checkpoint_seeds.checkpoint_job_id()
#     and optimisation_pipeline.find_checkpoint_commit_for_seed(). See
#     VALID_PRIOR_RUN_COMMITS, under "Initial design" below, for the
#     related but DELIBERATELY SEPARATE commit-list used when recovering
#     completed real trials instead of checkpoints.

# --- Baseline runs -------------------------------------------------------------
# The baseline config (BASELINE_CONFIG_VALUES, baseline_config.py) is run
# once per seed on N_BASELINE_RUNS seeds, through the SAME path as every
# trial (smac_scenario.py, checkpoints included). That single set of runs
# gives both:
#   - the BUDGETS: per year and cost category, the HIGHEST cost across the
#     baseline runs, written to COST_LIMITS_FILE (see
#     postprocess_output.compute_and_save_baseline_budgets()); and
#   - the baseline config's own results, recorded in the history exactly
#     like any trial, one entry per seed, flagged role = "baseline" in
#     submitted_jobs.jsonl and history_log.jsonl.
# Nothing is toggled: at every start the pipeline looks for baseline runs
# under the current commit or VALID_PRIOR_RUN_COMMITS, submits only the
# seeds that have none (or whose run failed), waits for all of them, and
# re-derives the budgets from them. A resume therefore never resubmits
# finished baseline runs; a commit not listed in VALID_PRIOR_RUN_COMMITS
# does resubmit them (and rebuilds the budgets).
#
# N_BASELINE_RUNS
#     Number of seeds the baseline config is run on: CHECKPOINT_SEEDS[:N].
#     Must be >= MAX_CONFIG_RUNS (checked in checkpoint_seeds.py). The
#     first MAX_CONFIG_RUNS seeds are the ones SMAC's intensifier can draw,
#     so the baseline is told to SMAC on exactly those seeds (the
#     incumbent starts with the full comparison set). Any further seeds
#     feed the budget and the history only, NOT SMAC: a challenger can
#     never be run on a seed outside the intensifier's pool, so telling
#     SMAC about them would stop any challenger being promoted past the
#     baseline. The budget is the per-year MAXIMUM over all N runs, so it
#     grows with N (10 keeps the earlier definition). Because the budget is
#     the maximum over the same seeds the baseline is judged on, the
#     baseline is feasible on every one of its seeds by construction.
#     Costs N baseline runs, plus checkpoints for any seed beyond
#     MAX_CONFIG_RUNS.
#
# --- Initial design ----------------------------------------------------------
# N_INIT sampled configs (ONE seed each, in addition to the baseline runs
# above) submitted and waited on BEFORE the main ask-tell
# loop starts, when SUBMIT_INITIAL_DESIGN is True - giving ConstrainedEI's
# surrogates and SMAC's own runhistory real, diverse data to work with
# from trial one, rather than starting from nothing. See
# optimisation_pipeline.submit_initial_design_jobs() - these jobs are
# submitted, waited on, then picked up by the EXISTING
# recover_from_job_log() mechanism, exactly like any other
# previously-submitted job (same commit-matching, same
# restart-deduplication) - no separate loading path.
#
# VALID_PRIOR_RUN_COMMITS
#     Full (or 12+ char) commit hashes whose ALREADY-SUBMITTED real trial
#     jobs (in submitted_jobs.jsonl) are still considered safe to recover
#     into history, even when the pipeline is currently running under a
#     DIFFERENT (e.g. newer) commit - see
#     optimisation_pipeline.recover_from_job_log(). The CURRENT commit is
#     always checked first, automatically - only list PAST commits here.
#     Baseline runs follow the same rule: baseline runs from an unlisted
#     commit are not reused, so they are resubmitted and the budgets rebuilt.
#
#     DELIBERATELY SEPARATE from VALID_CHECKPOINT_COMMITS ("Suspend/
#     resume toggles" above), despite the identical structure/usage
#     pattern - "safe to reuse a checkpoint generated under this commit"
#     (only the pre-resume portion of the simulation needs to be
#     unchanged) and "safe to recover a completed real trial's result
#     from this commit" (the ENTIRE simulation logic, including
#     draw_parameters()'s own config->module mapping, needs to be
#     unchanged) are genuinely different claims about a commit - one
#     holding doesn't imply the other. Point this at the same list as
#     VALID_CHECKPOINT_COMMITS if you're confident both always hold
#     together for your own commit history; kept separate here since
#     that's not true in general.
#
# SUBMIT_INITIAL_DESIGN
#     Plain user toggle - set True when you want this step to run. Even
#     when True, the step is SKIPPED automatically on a resume: if
#     JOB_LOG_FILE already holds at least N_INIT distinct NON-baseline
#     configs (current commit or VALID_PRIOR_RUN_COMMITS), the warm start
#     exists. If it holds fewer (e.g. a crash part-way through), a full
#     initial design is submitted again.
#
# N_INIT
#     Number of initial-design configs, all sampled (the baseline is NOT
#     among them: it is run separately on N_BASELINE_RUNS seeds). Each is
#     run on ONE seed (CHECKPOINT_SEEDS[0]). Total initial jobs = N_INIT.
#
# INITIAL_DESIGN_NEAR_BASELINE_FRACTION
#     Of the N_INIT sampled configs, this fraction (0.0-1.0)
#     are sampled as LOCAL PERTURBATIONS of the baseline's own values
#     (BASELINE_CONFIG_VALUES) rather than uniformly across the full
#     search space - see sample_near_baseline_configs() in
#     optimisation_pipeline.py. Motivation: uniform sampling across a
#     13-dimensional space rarely lands anywhere near a specific
#     reference point by chance, but the baseline is presumably already
#     a reasonable, near-feasible operating point (real current-programme
#     values) - and the whole point of the search is to find something
#     that BEATS it, so giving the initial design a real chance of
#     finding feasible, competitive neighbours of the baseline early
#     (rather than only wildly different, often-infeasible global
#     samples) is a more targeted use of these first few trials. The
#     remaining (1 - this fraction) are still sampled globally, uniformly
#     - so this never fully replaces broad exploration, just supplements
#     it. Currently set to 0.2 above (i.e. a fifth of the random configs
#     are near-baseline perturbations, the rest globally uniform).
#
# INITIAL_DESIGN_PERTURBATION_STD
#     Standard deviation of the Gaussian perturbation applied to each
#     FLOAT parameter for a "near-baseline" sample, as a FRACTION of that
#     parameter's own (upper - lower) range - e.g. 0.1 means a perturbed
#     value is drawn from Normal(baseline_value, 0.1 * range), then
#     clipped back into [lower, upper]. Scaled per-parameter by its own
#     range (not an absolute value) so the same setting means the same
#     relative "closeness" regardless of a parameter's own scale.
#     CATEGORICAL parameters (tdf_test_replace_vl_test,
#     targeted_adherence_monitoring) aren't perturbed at all by this -
#     there's no natural notion of a "small" change to a binary switch,
#     so near-baseline samples keep the baseline's own value for both,
#     unchanged, every time.

# --- Paths (rarely changed) --------------------------------------------------
# COST_LIMITS_FILE
#     One row per year, columns year,hiv_dalys,hiv_hrh_budget,
#     hiv_consumable_budget - read by initialise.py, WRITTEN by
#     postprocess_output.compute_and_save_baseline_budgets() when
#     the baseline runs finish. Lives here (not in initialise.py, where
#     it used to be defined) so postprocess_output.py can import it too,
#     without creating a circular import (initialise.py already imports
#     FROM postprocess_output.py).
#
# BASELINE_SUMMARY_FILE
#     Small JSON summary of the baseline run's own results, written
#     alongside COST_LIMITS_FILE by compute_and_save_baseline_budgets() -
#     specifically the baseline's median HIV DALYs (same TARGET_PERIOD
#     metric every real trial's own "dalys" field already is), since that
#     value has no other persistent home otherwise - the baseline job's
#     own Azure output isn't kept around once postprocessed. Used by
#     evaluate_pipeline_run.py's plotting to draw the baseline comparison
#     line - read from disk, independently, matching that file's own
#     "never touches the live process" design.
#
# BASELINE_BUDGET_COMPARISON_FILE
#     CSV written by compute_and_save_baseline_budgets(): per year and
#     category, the baseline mean, the adopted budget (max across runs) and
#     the upper 95% CI of the mean, plus the gap between the two. For
#     inspection only - nothing reads it back. The same function prints,
#     per category, the summed yearly budget against the most expensive
#     single baseline run, and how many baseline runs would have exceeded
#     a CI-based budget.


# INPUTS_DIR, OUTPUTS_DIR, COST_LIMITS_FILE, BASELINE_SUMMARY_FILE, BASELINE_BUDGET_COMPARISON_FILE
#     Precomputed, baseline-derived files all live in INPUTS_DIR (an "inputs"
#     folder next to this file, whatever directory the pipeline is launched
#     from). OUTPUTS_DIR ("outputs", same place) holds run outputs downloaded
#     from Azure. The folder is created automatically
#     when optimisation_parameters is imported (and again before each write),
#     so it never needs creating by hand. Logs (history_log.jsonl,
#     submitted_jobs.jsonl) live in LOG_DIR ("logs", same place). An existing cost_limits_by_year.csv or
#     baseline_summary.json must be moved into inputs/ once.
