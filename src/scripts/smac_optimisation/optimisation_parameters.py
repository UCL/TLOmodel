# --------------------------------------------------------------------------
# TLO / hyperparameters
# --------------------------------------------------------------------------
YEAR_START_DATE = 2010 # Year in which overall (from suspend) tlo sim starts
YEAR_END_DATE = 2012 # Year in which overall (including resume) tlo sim ends
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
MAX_CONFIG_CALLS = 5         # caps how many seeds the intensifier will use to
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
SUBMIT_SUSPEND_PART = False  # whether to submit the checkpoint-generation
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

