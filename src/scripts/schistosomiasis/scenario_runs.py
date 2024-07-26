"""
This file uses the calibrated parameters for prop_susceptible
and runs with no MDA running
outputs can be plotted against data to check calibration

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/schistosomiasis/run_calibrated_params.py

Run locally:
tlo scenario-run src/scripts/schistosomiasis/run_calibrated_params.py

or execute a single run:
tlo scenario-run src/scripts/schistosomiasis/run_calibrated_params.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/schistosomiasis/run_calibrated_params.py

"""

import random

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthsystem,
    healthseekingbehaviour,
    schisto,
    simplified_births,
    symptommanager,
)
from tlo.scenario import BaseScenario

number_of_draws = 1  # todo reset
runs_per_draw = 4


class TestScenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2025, 12, 31)  # todo reset
        self.pop_size = 64_000  # todo reset, 64,000=2k per district
        self.number_of_draws = number_of_draws
        self.runs_per_draw = runs_per_draw

    def log_configuration(self):
        return {
            'filename': 'schisto_calibration',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.schisto": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources, equal_allocation_by_district=True),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                disable_and_reject_all=False,  # if True, disable healthsystem and no HSI runs
            ),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            schisto.Schisto(resourcefilepath=self.resources, mda_execute=True),
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'Schisto': {
                'scaleup_WASH': [False, True][draw_number],
                'scaleup_WASH_start_year': 2015,
                'projection_scenario': [0, 1][draw_number],
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
