"""
The baseline's own, genuine values for the 13 parameters smac_scenario.py's
configspace tunes: the status-quo / current-programme point. Supplied
directly rather than guessed at.

Lives in its own module (not initialise.py) because the baseline runs must
be SUBMITTED before the budgets exist, and initialise.py reads the budget
file at import time - so nothing needed to submit the baseline can live
there. initialise.py re-exports it for existing importers.

The baseline is submitted through the standard smac_scenario.py path (the
same one every trial uses, suspend/resume included) on MAX_CONFIG_RUNS
seeds; the same runs give both the budgets and the baseline config's
results. See optimisation_pipeline.ensure_baseline_runs().
"""

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

