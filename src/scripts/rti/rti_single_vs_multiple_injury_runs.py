import numpy as np

from tlo import Date, logging
from tlo.methods import (
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthburden,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1)
        self.pop_size = 1000
        self.number_of_draws = 1
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'rti_single_vs_mutliple_injury',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=self.resources),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True, service_availability=['*']),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            rti.RTI(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources)
        ]

# Here I want to run the model with two sets of parameters multiple times. Once where only singular injuries
# are given out and once where we allow multiple injuries
    def draw_parameters(self, draw_number, rng):
        if draw_number < self.runs_per_draw / 2:
            return {
                'RTI': {
                    'number_of_injured_body_regions_distribution': [[1, 2, 3, 4, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0, 0, 0]]
                },
            }
        else:
            return {
                'RTI': {
                    'number_of_injured_body_regions_distribution':
                        [[1, 2, 3, 4, 5, 6, 7, 8], [0.38, 0.25, 0.153, 0.094, 0.055, 0.031, 0.018, 0.019]]
                },
            }



if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
