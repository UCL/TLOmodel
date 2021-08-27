"""
This file defines a batch run through which the hiv modules are run across a grid of parameter values

Check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/hiv/PrEP_analyses/hiv_prep_baseline_scenario.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/PrEP_analyses/hiv_prep_baseline_scenario.py

or execute a single run:
tlo scenario-run src/scripts/hiv/PrEP_analyses/hiv_prep_baseline_scenario.py --draw 1 0

Run on the batch system using:
tlo batch-submit  src/scripts/hiv/PrEP_analyses/hiv_prep_baseline_scenario.py

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download tlo_q1_demo-123

test run, pop 5000
Job ID: hiv_prep_baseline_scenario-2021-08-27T095045Z

"""

import numpy as np

from tlo import Date
from tlo import logging

from tlo.methods import (
    demography,
    enhanced_lifestyle,
    simplified_births,
    dx_algorithm_child,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    hiv
)

from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2030, 12, 31)
        self.pop_size = 100000
        self.number_of_draws = 12
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'hiv_prep_baseline_scenario',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                # 'tlo.scr.scripts.hiv.PrEP_analyses.default_run_with_plots': logging.INFO,
                'tlo.methods.demography': logging.INFO,
                'tlo.scenario': logging.INFO
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                disable=True,  # no event queueing, run all HSIs
                ignore_cons_constraints=True),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        #
        # grid = self.make_grid(
        #     {
        #         'prob_for_prep_selection': np.linspace(start=0, stop=0.6, num=4),
        #         'prob_prep_high_adherence': np.linspace(start=0.3, stop=0.5, num=3),
        #         'prob_prep_mid_adherence': np.linspace(start=0.1, stop=0.3, num=3),
        #     }
        # )
        tmp = [0, 0.2, 0.4, 0.6]
        prob_for_prep_selection_list = np.repeat(tmp, 3)
        prob_prep_high_adherence_list = [0.3, 0.4, 0.5] * 4
        prob_prep_mid_adherence_list = [0.1, 0.2, 0.3] * 4

        return {
            'Hiv': {
                'prob_for_prep_selection': prob_for_prep_selection_list[draw_number],
                'prob_prep_high_adherence': prob_prep_high_adherence_list[draw_number],
                'prob_prep_mid_adherence': prob_prep_mid_adherence_list[draw_number],
            }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
