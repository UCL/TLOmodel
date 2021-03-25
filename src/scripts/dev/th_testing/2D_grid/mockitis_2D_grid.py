"""
This file defines a batch run through which the Mockitis module is run across a 2-dimensional grid of parameters

Run on the batch system using:
```tlo batch-submit  src/scripts/dev/th_testing/mockitis_2D_grid.py tlo.conf```

"""

import numpy as np
import pandas as pd

from tlo import Date, logging
from tlo.methods import (
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    labour,
    mockitis,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.scenario import BaseScenario


class Mockitis_Batch(BaseScenario):
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
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            contraception.Contraception(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            mockitis.Mockitis(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        grid = self.make_grid(
            {
                'p_infection': np.linspace(0, 1.0, 3),
                'p_cure': [0.25, 0.5]
            }
        )

        return {
            'Mockitis': {
                'p_infection': grid['p_infection'][draw_number],
                'p_cure': grid['p_cure'][draw_number]
            },
        }

    def make_grid(self, ranges: dict) -> pd.DataFrame:
        """Utility function to flatten an n-dimension grid of parameters for use in scenarios

        Typically used in draw_parameters determining a set of parameters for a draw. This function will check that the
        number of draws of the scenario is equal to the number of coordinates in the grid.

        Parameter 'ranges' is a dictionary of { string key: iterable }, where iterable can be, for example, an np.array
        or list. The function will return a DataFrame where each key is a column and each row represents a single
        coordinate in the grid.

        Usage (in draw_parameters):

            grid = self.make_grid({'p_one': np.linspace(0, 1.0, 5), 'p_two': np.linspace(3.0, 4.0, 2)})
            return {
                'Mockitis': {
                    grid['p_one'][draw_number],
                    grid['p_two'][draw_number]
                }
            }
        """
        grid = np.meshgrid(*ranges.values())
        flattened = [g.ravel() for g in grid]
        positions = np.stack(flattened, axis=1)
        grid_lookup = pd.DataFrame(positions, columns=ranges.keys())
        assert self.number_of_draws == len(grid_lookup), f"{len(grid_lookup)} coordinates in grid, " \
                                                         f"but number_of_draws is {self.number_of_draws}."
        return grid_lookup


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
