
"""
This file defines a batch run through which the Mockitis module is run across a sweep of a single parameter.
Run on the batch system using:
```tlo batch-submit  src/scripts/hiv/hiv_tb_analyses/baseline_scenario.py tlo.conf```
"""


import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    simplified_births,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    epi,
    hiv,
    tb
)

from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2014, 12, 31)
        self.pop_size = 1000
        self.number_of_draws = 9
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'baseline_scenario',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
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
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        grid = self.make_grid(
            {
                'beta': np.linspace(start=0.02, stop=0.06, num=3),
                'transmission_rate': np.linspace(start=0.005, stop=0.2, num=3),
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
