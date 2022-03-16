"""
This file defines a batch run through which the Mockitis module is run across a 2-dimensional grid of parameters

Run on the batch system using:
```tlo batch-submit  src/scripts/dev/th_testing/mockitis_2D_grid.py tlo.conf```

"""

import numpy as np

from tlo import Date, logging
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    labour,
    mockitis,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.scenario import BaseScenario


class MockitisBatch(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 500
        self.number_of_draws = 6
        self.runs_per_draw = 2

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
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            contraception.Contraception(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            mockitis.Mockitis(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        grid = self.make_grid(
            {'p_infection': np.linspace(0, 1.0, 3), 'p_cure': [0.25, 0.5]}
        )

        return {
            'Mockitis': {
                'p_infection': grid['p_infection'][draw_number],
                'p_cure': grid['p_cure'][draw_number]
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
