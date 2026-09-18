"""
Constraint setup and pre-completed prior runs, kept separate from
optimisation_pipeline.py's submission/polling/ask-tell orchestration -
same rationale as constrained_ei.py, smac_scenario.py,
postprocess_output.py, and convergence_monitoring.py: this can be read
or edited independently of the orchestration logic that imports it.

Two responsibilities live here:
1. Defining PERIOD_BOUNDARIES (user-definable), loading the year-specific
   BUDGET for each cost category from a limits file, and the bucketing
   logic that turns a run's per-year cost into a period-level violation.
2. PRIOR_RUNS - already-completed runs to warm-start the search with.

TERMINOLOGY, since this has been a source of confusion before: "cost"
always means the actual $ a simulated run spent in a given year
(computed in postprocess_output.py). "Budget" always means the $
allowed for that same category in that year (loaded here, from a file).
Both are the same unit ($), for the same category (HIV-HRH or
HIV-consumable) - "cost" and "budget" are usage vs. limit, not two
different kinds of measurement. There is no separate physical
hours/units dimension - only cost/budget, for exactly two categories:
HIV-HRH and HIV-consumable.
"""

from __future__ import annotations

import pandas as pd

from postprocess_output import TARGET_PERIOD as POSTPROCESS_TARGET_PERIOD
from optimisation_parameters import (
    START_FIRST_BOUNDARY, START_SECOND_BOUNDARY, START_THIRD_BOUNDARY, END_THIRD_BOUNDARY,
    COST_LIMITS_FILE,
)

# NOTE: POSTPROCESS_TARGET_PERIOD is used EXACTLY as imported above, from
# postprocess_output.py's own TARGET_PERIOD - it must NOT be recomputed
# locally here (an earlier version of this file did, redundantly, from
# the same underlying CONFIG_YEAR_START_DATE/YEAR_END_DATE constants -
# this silently defeated the consistency check just below, since it
# would then always compare PERIOD_BOUNDARIES against a value that no
# longer reflected whatever postprocess_output.py's TARGET_PERIOD
# actually was, e.g. if that file's TARGET_PERIOD were ever changed
# independently. Always use the genuinely-imported value.

# --------------------------------------------------------------------------
# 1. Period-bucketed HIV-HRH and HIV-consumable budget constraints
#
# Rather than a year-by-year constraint (prohibitive across a long
# horizon - would need one separate RF surrogate per year, with the
# multiplied-probability problem getting worse with every added
# constraint), each cost category is bucketed into a small,
# USER-DEFINABLE number of periods. Within each period, the CUMULATIVE
# (mean-across-years, not max) violation is computed:
#
#     period_violation = mean( max(0, cost_year/budget_year - 1) for year in period )
#
# Mean-across-years (not max-across-years, and NOT an average restricted
# to only the years that violate) so that an ADDITIONAL bad year can
# only raise or maintain a period's score, never lower it - averaging
# only the positive values would let a config with MORE violating years
# score BETTER than one with fewer, which is the wrong direction.
#
# PERIOD_BOUNDARIES is user-definable (absolute calendar years,
# inclusive on both ends) and MUST be the same list for HIV-HRH and
# HIV-consumable budgets - both are bucketed identically, not with
# independently-chosen periods per category.
# --------------------------------------------------------------------------

PERIOD_BOUNDARIES = [
    (START_FIRST_BOUNDARY, START_SECOND_BOUNDARY-1),  # period 1 -  edit these
    (START_SECOND_BOUNDARY, START_THIRD_BOUNDARY-1),  # period 2   three (start_year, end_year) tuples to
    (START_THIRD_BOUNDARY, END_THIRD_BOUNDARY),  # period 3   whatever periods you actually want enforced.
]

# Consistency check against postprocess_output.py: every YEAR ACTUALLY
# SIMULATED (postprocess_output.py's TARGET_PERIOD) must fall into AT
# LEAST ONE defined period - otherwise that year's real cost data would
# be silently dropped from every constraint (bucket_cumulative_violation
# excludes years with no matching period, which is indistinguishable
# from "0 violation", i.e. always "feasible" - the wrong kind of failure
# to have hidden).
#
# The REVERSE is explicitly fine and NOT checked: PERIOD_BOUNDARIES may
# span MORE years than are actually simulated (e.g. the real long-term
# periods like 2025-2049 kept as-is while TARGET_PERIOD is temporarily
# shortened to 2010-2011 for a quick test) - those extra period-years
# simply never accumulate any data, which is harmless. This is what
# makes short test runs work without needing to touch PERIOD_BOUNDARIES
# at all, provided the simulated years land inside some period.
_postprocess_years = set(range(POSTPROCESS_TARGET_PERIOD[0].year, POSTPROCESS_TARGET_PERIOD[1].year + 1))
_period_years = set()
for _start, _end in PERIOD_BOUNDARIES:
    _period_years.update(range(_start, _end + 1))
if not _postprocess_years.issubset(_period_years):
    _uncovered = sorted(_postprocess_years - _period_years)
    raise ValueError(
        f"postprocess_output.py's TARGET_PERIOD "
        f"({POSTPROCESS_TARGET_PERIOD[0].year}-{POSTPROCESS_TARGET_PERIOD[1].year}) "
        f"includes years {_uncovered} that PERIOD_BOUNDARIES doesn't cover with ANY "
        f"period - data for those years would be silently dropped from every "
        f"constraint rather than counted. Either extend/add a period in "
        f"PERIOD_BOUNDARIES to cover {_uncovered}, or shorten TARGET_PERIOD in "
        f"postprocess_output.py so it stays within the periods already defined."
    )

# Budgets file: one row per year, columns year,hiv_dalys,hiv_hrh_budget,
# hiv_consumable_budget. hiv_dalys is loaded but not currently used as a
# constraint - DALYs remains the optimisation OBJECTIVE (see
# ConstrainedEI construction in optimisation_pipeline.py), not a
# constraint; flagging this explicitly in case a year-specific DALYs
# constraint is wanted later, which would need separate wiring.
# COST_LIMITS_FILE itself now lives in optimisation_parameters.py (not
# defined here) - see that file's own comment for why, and
# postprocess_output.compute_and_save_baseline_budgets() for how this
# file gets WRITTEN, if SUBMIT_BASELINE_RUN is True.
_limits_df = pd.read_csv(COST_LIMITS_FILE)
HIV_HRH_BUDGET_BY_YEAR = dict(zip(_limits_df["year"], _limits_df["hiv_hrh_budget"]))
HIV_CONSUMABLE_BUDGET_BY_YEAR = dict(zip(_limits_df["year"], _limits_df["hiv_consumable_budget"]))


_warned_zero_budget_years: set[int] = set()  # tracks which years have
    # already triggered bucket_cumulative_violation()'s zero-budget
    # warning below, so it prints once per year, not once per trial


def _year_to_period_index(year: int) -> int | None:
    """Returns the 0-based index into PERIOD_BOUNDARIES that `year` falls
    into, or None if it falls into none of the defined periods (in which
    case it's silently excluded from every period's violation - not an
    error, since PERIOD_BOUNDARIES may legitimately not need to span
    every single simulated year)."""
    for i, (start, end) in enumerate(PERIOD_BOUNDARIES):
        if start <= year <= end:
            return i
    return None


def bucket_cumulative_violation(cost_by_year: dict, budget_by_year: dict) -> list[float]:
    """
    For each period in PERIOD_BOUNDARIES, computes the mean of
    max(0, cost_year/budget_year - 1) across every year in that period -
    see the module-level comment above for why mean-across-all-years
    (not max, not mean-of-positives-only). Years with no defined budget
    ARE SKIPPED (excluded from that period's denominator too, not
    treated as a violation or as feasible) - a budget of exactly 0 (or
    negative) is treated the SAME as no budget at all: it almost always
    means the baseline run never actually recorded any cost for this
    year/category (compute_and_save_baseline_budgets() defaults a
    year's budget to 0.0 if the baseline's own runs never produced data
    for it), not that the true limit is genuinely zero. Dividing by a
    real zero would raise ZeroDivisionError outright - caught by
    optimisation_pipeline.py's own [postprocessing failed] handler, but
    silently CRASHING EVERY trial that has any cost in that year, which
    can empty out `history` entirely for the whole run (surfacing much
    later as "min() arg is an empty sequence" at final selection,
    genuinely confusing to trace back to this). Prints a warning the
    first time a zero/negative budget is actually skipped, so a
    systematically-broken baseline (e.g. most years ending up at 0.0)
    doesn't go unnoticed.
    """
    period_sums = [0.0] * len(PERIOD_BOUNDARIES)
    period_counts = [0] * len(PERIOD_BOUNDARIES)

    for year, cost in cost_by_year.items():
        idx = _year_to_period_index(year)
        if idx is None:
            continue
        budget = budget_by_year.get(year)
        if budget is None:
            continue
        if budget <= 0:
            if year not in _warned_zero_budget_years:
                print(
                    f"[warning] budget for year {year} is {budget} (<= 0) - treating as "
                    f"undefined and skipping, not dividing by it. Check cost_limits_by_year.csv - "
                    f"this usually means the baseline run never recorded any cost for this "
                    f"year/category (see compute_and_save_baseline_budgets())."
                )
                _warned_zero_budget_years.add(year)
            continue
        period_sums[idx] += max(0.0, cost / budget - 1)
        period_counts[idx] += 1

    return [
        (period_sums[i] / period_counts[i]) if period_counts[i] > 0 else 0.0
        for i in range(len(PERIOD_BOUNDARIES))
    ]


# Two constraint families (3 periods x 2 cost/budget categories: HIV-HRH
# and HIV-consumable - NOT four; a category's cost and its budget are
# the same $ measure, usage vs. limit, not two different constraints).
# Names generated from PERIOD_BOUNDARIES's actual length rather than
# hardcoded, so this stays correct if the number of periods is ever
# changed, not just their boundaries.
HIV_HRH_CONSTRAINT_NAMES = [f"hiv_hrh_violation_p{i + 1}" for i in range(len(PERIOD_BOUNDARIES))]
HIV_CONSUMABLE_CONSTRAINT_NAMES = [f"hiv_consumable_violation_p{i + 1}" for i in range(len(PERIOD_BOUNDARIES))]
CONSTRAINT_NAMES = HIV_HRH_CONSTRAINT_NAMES + HIV_CONSUMABLE_CONSTRAINT_NAMES


# --------------------------------------------------------------------------
# 2. Warm-start with runs you've already completed, THEN start the loop.
#    Replace this with however you're currently loading your existing
#    completed runs (CSV, dataframe, pickle, whatever they're sitting in).
# --------------------------------------------------------------------------

PRIOR_RUNS = [
    #{
    #    "config": {"config_annual_testing_rate_adults": 0.8, "annual_rate_selftest": 0.3, ...},
    #    "dalys": 45.3, "hiv_hrh_cost_by_year": {2025: 12000.0, 2026: 12500.0, ...},
    #    "hiv_consumable_cost_by_year": {2025: 8000.0, 2026: 8100.0, ...},
    #},
    # ... your other already-completed runs ...
]


# --------------------------------------------------------------------------
# 3. The baseline's own, genuine values for the 13 parameters
#    smac_scenario.py's configspace actually tunes - NOT set explicitly in
#    smac_scenario_baseline.py's own draw_parameters() (which only ever
#    sets 'type_of_scaleup': 'none', leaving these 13 to whatever the Hiv
#    module's own resource files default to), but a real, well-defined
#    point nonetheless - status-quo/current-programme values, supplied
#    directly rather than guessed at or fabricated.
#
#    Lives HERE, alongside PRIOR_RUNS (not in smac_scenario_baseline.py,
#    which defines TLO scenario/simulation behaviour, not SMAC/pipeline
#    bookkeeping) - conceptually the baseline is itself just another
#    config worth SMAC knowing about early, same spirit as PRIOR_RUNS'
#    own entries. Used by optimisation_pipeline.submit_initial_design_jobs()
#    (SUBMIT_INITIAL_DESIGN toggle) - submitted through the STANDARD
#    smac_scenario.py path (suspend/resume, single seed), NOT through
#    smac_scenario_baseline.py's own separate, 10-run submission
#    (submit_baseline_job(), SUBMIT_BASELINE_RUN toggle, used only to
#    derive the budget) - see that function's own docstring for the
#    known, deliberately-not-yet-unified overlap between the two.
# --------------------------------------------------------------------------

BASELINE_CONFIG_VALUES = {
    "config_annual_testing_rate_adults": 0.248765,
    "annual_rate_selftest": 0.07,
    "prob_hiv_test_at_anc_or_delivery": 0.95,
    "prob_hiv_test_for_newborn_infant": 0.95,
    "prob_prep_for_fsw_after_hiv_test": 0.11,
    "prob_prep_for_agyw": 0.01,
    "prob_injectable_prep_vs_oral": 0.7,
    "prob_circ_after_hiv_test": 0.05,
    "linked_to_care_after_selftest": 0.563,
    "prob_receive_viral_load_test_result": 0.6,
    "config_coverage_plhiv": 0.30,
    "tdf_test_replace_vl_test": False,
    "targeted_adherence_monitoring": False,
}

