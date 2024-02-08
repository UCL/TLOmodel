"""
This file defines a batch run through which the ALRI module is run across a set of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/Alri_analyses/alri_azure_run_scenarios/baseline_alri_scenario.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/Alri_analyses/alri_azure_run_scenarios/baseline_alri_scenario.py

or execute a single run:
tlo scenario-run src/scripts/Alri_analyses/alri_azure_run_scenarios/baseline_alri_scenario.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/Alri_analyses/alri_azure_run_scenarios/baseline_alri_scenario.py

weds 16th feb runs:
Job ID: baseline_alri_scenario-2022-03-18T142355Z

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download baseline_alri_scenario-2022-03-18T142355Z
"""

import random

from tlo import Date, logging
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    malaria,
    simplified_births,
    symptommanager,
    wasting,
)
from tlo.scenario import BaseScenario

number_of_draws = 1
runs_per_draw = 10
pop_size = 760000  # 1:25 representative sample


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically
    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2031, 1, 1)
        self.pop_size = pop_size
        self.number_of_draws = number_of_draws
        self.runs_per_draw = runs_per_draw

    def log_configuration(self):
        return {
            "filename": "baseline_scenario_alri_50k_pop_1drawx10runs",
            "directory": "./outputs",
            "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.alri": logging.INFO,
            },
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                service_availability=["*"],  # all treatment allowed
                mode_appt_constraints=0,
                # mode of constraints to do with officer numbers and time
                cons_availability="all",
                # mode for consumable constraints (if ignored, all consumables available)
                ignore_priority=True,
                # do not use the priority information in HSI event to schedule
                capabilities_coefficient=1.0,
                # multiplier for the capabilities of health officers
                disable=True,
                # disables the healthsystem (no constraints and no logging) and every HSI runs
                disable_and_reject_all=False,  # disable healthsystem and no HSI runs
            ),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(
                resourcefilepath=self.resources
            ),
            malaria.Malaria(resourcefilepath=self.resources),
            alri.Alri(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            wasting.Wasting(resourcefilepath=self.resources),
            alri.AlriPropertiesOfOtherModules(),
        ]

    def draw_parameters(self, draw_number, rng):
        return {}


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
