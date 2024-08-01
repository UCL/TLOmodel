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
    bladder_cancer,
    demography,
    diarrhoea,
    enhanced_lifestyle,
    healthsystem,
    healthseekingbehaviour,
    hiv,
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
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.bladder_cancer": logging.INFO,
                "tlo.methods.diarrhoea": logging.INFO,
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
            # diseases
            bladder_cancer.BladderCancer(resourcefilepath=self.resources),
            diarrhoea.Diarrhoea(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            schisto.Schisto(resourcefilepath=self.resources, mda_execute=True),
        ]

    def draw_parameters(self, draw_number, rng):
        coverage_options = [0.6, 0.7, 0.8]
        target_group_options = ['SAC', 'PSAC', 'All']
        wash_options = [True, False]

        # Calculate the total number of combinations
        total_combinations = len(coverage_options) * len(target_group_options) * len(wash_options)

        # Ensure the draw_number is within the range of total combinations
        if draw_number >= total_combinations:
            raise ValueError("draw_number exceeds the total number of combinations.")

        # Determine indices for each parameter
        coverage_index = draw_number % len(coverage_options)
        target_group_index = (draw_number // len(coverage_options)) % len(target_group_options)
        wash_index = (draw_number // (len(coverage_options) * len(target_group_options))) % len(wash_options)

        return {
            'Schisto': {
                'mda_coverage': coverage_options[coverage_index],
                'mda_target_group': target_group_options[target_group_index],
                'mda_frequency': 6,
                'scaleup_WASH': wash_options[wash_index],
                'scaleup_WASH_start_year': 2024,
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
