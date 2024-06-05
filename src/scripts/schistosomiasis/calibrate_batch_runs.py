"""
This file defines a batch run through which the hiv and tb modules are run across a set of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/schistosomiasis/calibrate_batch_runs.py

Run locally:
tlo scenario-run src/scripts/schistosomiasis/calibrate_batch_runs.py

or execute a single run:
tlo scenario-run src/scripts/schistosomiasis/calibrate_batch_runs.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/schistosomiasis/calibrate_batch_runs.py

"""

import os
import random

import pandas as pd

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthsystem,
    schisto,
    really_simplified_births,
    symptommanager,
)
from tlo.scenario import BaseScenario

number_of_draws = 1  # todo reset
runs_per_draw = 1


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2050, 12, 31)  # todo reset
        self.pop_size = 5_000  # todo reset
        self.number_of_draws = number_of_draws
        self.runs_per_draw = runs_per_draw

    def log_configuration(self):
        return {
            'filename': 'schisto_calibration',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.schisto": logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            really_simplified_births.ReallySimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                disable_and_reject_all=True,  # disable healthsystem and no HSI runs
            ),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            schisto.Schisto(resourcefilepath=self.resources, mda_execute=False),

        ]

    def draw_parameters(self, draw_number, rng):

        return {
            'Schisto': {
                # 'calibration_scenario': [1.0, 2.0, 3.0, 4.0][draw_number],
                'calibration_scenario': 1.0,
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
