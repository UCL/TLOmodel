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
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 500
        self.number_of_draws = 5
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

        grid = self.make_grid({
            'p_infection': np.linspace(0, 1.0, 5),
            'p_cure': np.linspace(0, 0.5, 5)
        })
        self.number_of_draws = len(list(grid.values()[0]))

        return {
            'Mockitis': {
                'p_infection': grid['p_infection'][draw_number],
                'p_cure': grid['p_cure'][draw_number]
            },
        }

    def make_grid(self, ranges: dict) -> list:
        """utility function to flatten a 2-dimension grid of parameters for use in batch-run.
        ?? Move to baseclass??"""

        def is_iter(x):
            try:
                iter(x)
                return True
            except TypeError:
                return False

        # check that the ranges given is a dict with two entries and that each entry is itterable
        assert type(ranges) is dict
        assert 2 == len(ranges)
        assert all([is_iter(v) for v in ranges.values()])

        # get the values to go on the x and y values
        x = list(ranges.values())[0]
        y = list(ranges.values())[1]

        X, Y = np.meshgrid(x, y)

        return {
            list(ranges.keys())[0]: X.ravel(),
            list(ranges.keys())[1]: Y.ravel()
        }



if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])


