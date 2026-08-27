"""
Committed once at src/scripts/smac_optimisation/smac_scenario.py

This is the ONE scenario file the ask-tell loop in
ask_tell_azure_example.py submits repeatedly - never regenerated or
re-committed per SMAC trial. Configuration values are set directly as
Python attributes on the instantiated object by the submitting code
(see submit_azure_job() in ask_tell_azure_example.py) - no CLI/argparse
round-trip, since that submitting code is itself Python, not a shell.

number_of_draws=1 because each submission represents exactly one
SMAC-proposed configuration. runs_per_draw=1 because each submission is
a single (config, seed) physical realisation - see SEEDING below for
why repeated evaluation of a config is handled by SMAC's intensifier
rather than by TLO's own multi-seed averaging.

SEEDING - leveraging SMAC's own intensification
--------------------------------------------------
scenario.seed is set directly from SMAC's info.seed at submission time
(see submit_azure_job() in ask_tell_azure_example.py) - NOT fixed here.
runs_per_draw is 1: each submission is a single (config, seed) physical
realisation, not a pre-averaged bundle.

This is deliberate: with the SMAC Scenario set to deterministic=False,
SMAC's own intensifier decides whether a given config's result is
promising enough to warrant re-evaluating with an additional seed
before trusting it as a challenger to the incumbent - compute is spent
proportionally to how competitive a config looks, rather than averaging
every proposed config over a fixed number of seeds regardless of merit.
Fixing runs_per_draw>1 here would double up on that job, paying a fixed
per-config cost SMAC's intensifier is already designed to avoid.

Because history now holds individual noisy realisations rather than
pre-averaged results, final config selection in
ask_tell_azure_example.py groups by config and averages across whatever
seeds SMAC ended up requesting for it, rather than trusting any single
realisation's DALYs value directly.

You can still sanity-check this scenario class works standalone via
`tlo scenario-run src/scripts/smac_optimisation/smac_scenario.py` using
its defaults below - useful for catching module/parameter wiring bugs
locally before submitting to batch - but that CLI path is now purely
optional and not what the SMAC loop itself uses.

HYPERPARAMETERS: every tunable knob in this file is marked inline with a
"HYPERPARAMETER" comment - grep for that tag across all three files
(constrained_ei.py, smac_scenario.py, ask_tell_azure_example.py) to find
the complete list in one pass.
"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.methods import demography, enhanced_lifestyle, healthburden, healthsystem
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class TloOptimisationScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0  # placeholder - overwritten with SMAC's info.seed
                       # by submit_azure_job() before every real submission
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1)
        self.pop_size = 1_000  # HYPERPARAMETER: simulation fidelity, not a BO
                                   # hyperparameter, but trades off noise level
                                   # against per-run cost - indirectly relevant
                                   # to min_samples_leaf and max_config_calls in
                                   # the other two files, since a smaller
                                   # population means noisier DALYs/cost per seed
        self.number_of_draws = 1     # one draw == one SMAC configuration
        self.runs_per_draw = 1       # one seed per submission - SMAC's own
                                       # intensifier decides if/when this
                                       # config gets re-evaluated with a
                                       # different seed, not a fixed sweep

        # Place holders
        self.tclose_days_offset_overwrite = 7
        self.year_mode_switch = 2019
        
        """
        self.hiv_testing_rates = None
        self.annual_rate_selftest = None
        self.annual_testing_rate_adults = None
        self.prob_hiv_test_at_anc_or_delivery = None
        self.prob_hiv_test_for_newborn_infant = None
        self.selftest_available = None
        self.switch_vl_test_to_tdf = None
        self.prob_prep_for_fsw_after_hiv_test = None
        self.prob_prep_for_agyw = None
        self.prob_injectable_prep_vs_oral = None
        self.prob_circ_after_hiv_test = None
        self.prob_circ_for_child_from_2020 = None
        self.beta = None
        self.reduction_in_hiv_beta = None
        self.probability_of_being_retained_on_prep_every_3_months = None
        self.probability_of_being_retained_on_art_every_3_months = None
        self.prob_start_art_or_vs = None
        self.tb_ipt_coverage = None
        self.virally_suppressed_on_art = None
        self.consumable_availability_HIV_test = None
        self.consumable_availability_VL_test = None
        """

    def log_configuration(self):
        return {
            "filename": "smac_optimisation",
            "directory": "./outputs",
            "custom_levels": {"*": logging.INFO},
        }

    def modules(self):
        return fullmodel()

    def draw_parameters(self, draw_number, rng):
        # draw_number is ignored - number_of_draws=1, so this always
        # returns the single SMAC config currently set on self.
        return {
            "HealthSystem": {
                "tclose_days_offset_overwrite": self.tclose_days_offset_overwrite,
                "year_mode_switch" : self.year_mode_switch,
            },
            """
            "Hiv": {
                "hiv_testing_rates": self.hiv_testing_rates,
                "annual_rate_selftest": self.annual_rate_selftest,
                "annual_testing_rate_adults": self.annual_testing_rate_adults,
                "prob_hiv_test_at_anc_or_delivery": self.prob_hiv_test_at_anc_or_delivery,
                "prob_hiv_test_for_newborn_infant": self.prob_hiv_test_for_newborn_infant,
                "selftest_available": self.selftest_available,
                "switch_vl_test_to_tdf": self.switch_vl_test_to_tdf,
                "prob_prep_for_fsw_after_hiv_test": self.prob_prep_for_fsw_after_hiv_test,
                "prob_prep_for_agyw": self.prob_prep_for_agyw,
                "prob_injectable_prep_vs_oral": self.prob_injectable_prep_vs_oral,
                "prob_circ_after_hiv_test": self.prob_circ_after_hiv_test,
                "prob_circ_for_child_from_2020": self.prob_circ_for_child_from_2020,
                "beta": self.beta,
                "reduction_in_hiv_beta": self.reduction_in_hiv_beta,
                "probability_of_being_retained_on_prep_every_3_months": self.probability_of_being_retained_on_prep_every_3_months,
                "probability_of_being_retained_on_art_every_3_months": self.probability_of_being_retained_on_art_every_3_months,
                "prob_start_art_or_vs": self.prob_start_art_or_vs,
                "tb_ipt_coverage": self.tb_ipt_coverage,
                "virally_suppressed_on_art": self.virally_suppressed_on_art,
                "consumable_availability_HIV_test": self.consumable_availability_HIV_test,
                "consumable_availability_VL_test": self.consumable_availability_VL_test,
            }
            """
            # map intervention_coverage / other config values onto
            # whichever module parameters they actually control in
            # your real model
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
