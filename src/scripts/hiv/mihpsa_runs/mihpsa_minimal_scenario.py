"""
This file defines a batch run through which the hiv and tb modules are run across a set of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/mihpsa_runs/mihpsa_minimal_scenario.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/deviance_for_calibration/calibration_script.py

or execute a single run:
tlo scenario-run src/scripts/hiv/deviance_for_calibration/calibration_script.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/hiv/deviance_for_calibration/calibration_script.py

weds 16th feb runs:
Job ID: calibration_script-2022-02-16T202042Z

this ^^ is the large calibration run (100 jobs, pop=760k)


Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download calibration_script-2022-04-12T190518Z



"""

import os
import random

import pandas as pd

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    hiv_tb_calibration,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.scenario import BaseScenario

number_of_draws = 1
runs_per_draw = 3


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2051, 1, 1)
        self.pop_size = 50_000
        self.number_of_draws = number_of_draws
        self.runs_per_draw = runs_per_draw

    def log_configuration(self):
        return {
            'filename': 'mihpsa_runs',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.tb": logging.INFO,
            }
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
                ignore_priority=False,  # do not use the priority information in HSI event to schedule
                capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
            ),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        return


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
