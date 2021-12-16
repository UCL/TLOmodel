
"""
This file defines a batch run through which the hiv and tb modules are run across a grid of parameter values
Run on the batch system using:
```tlo batch-submit  src/scripts/hiv/hiv_tb_analyses/baseline_scenario.py```
"""

import numpy as np

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.scenario import BaseScenario

# define size of parameter lists
hiv_param_length = 10
tb_param_length = 10
number_of_draws = hiv_param_length * tb_param_length


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = 32
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2019, 12, 31)
        self.pop_size = 1500
        self.number_of_draws = number_of_draws
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'baseline_scenario',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.demography': logging.INFO
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
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        grid = self.make_grid(
            {
                'beta': np.linspace(start=0.02, stop=0.06, num=hiv_param_length),
                'transmission_rate': np.linspace(start=0.005, stop=0.2, num=tb_param_length),
            }
        )

        return {
            'Hiv': {
                'beta': grid['beta'][draw_number],
            },
            'Tb': {
                'transmission_rate': grid['transmission_rate'][draw_number],
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
