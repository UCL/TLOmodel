"""
This file defines a batch run through which the HIV module is run
"""

import numpy as np

from tlo import Date, logging
from tlo.methods import (
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
    mockitis
)
from tlo.scenario import BaseScenario

class Mockitis_Batch(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1)
        self.pop_size = 5000
        self.number_of_draws = 10
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'mockitis_batch',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True, service_availability=['*']),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            contraception.Contraception(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            mockitis.Mockitis(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'Mockitis': {
                'p_infection': np.linspace(0, 0.1)[draw_number],
                'p_cure': 0.52,
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
