"""
This file defines a batch run through which the hiv and tb modules are run across a grid of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/hiv_tb_projections/scenario2.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/hiv_tb_projections/scenario2.py

or execute a single run:
tlo scenario-run src/scripts/hiv/hiv_tb_projections/scenario2.py --draw 0 0

Run on the batch system using:
tlo batch-submit src/scripts/hiv/hiv_tb_projections/scenario2.py

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

21st Feb
Job ID: baseline_runs-2022-02-21T171625Z

Download result files for a completed job:
tlo batch-download baseline_runs-2022-02-08T164813Z

"""

from random import randint

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = randint(0, 5000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2035, 1, 1)
        self.pop_size = 760000
        self.number_of_draws = 1
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            "filename": "scenario2",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthsystem": logging.INFO,
            },
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                service_availability=["*"],  # all treatment allowed
                mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
                cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
                ignore_priority=True,  # do not use the priority information in HSI event to schedule
                capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
                disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
                disable_and_reject_all=False,  # disable healthsystem and no HSI runs
                store_hsi_events_that_have_run=False,  # convenience function for debugging
            ),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
        ]

    # todo need to limit consumables availability
    def draw_parameters(self, draw_number, rng):

        return {
            'Tb': {
                'scenario': 2,
            },
        }


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
