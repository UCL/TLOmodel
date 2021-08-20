"""
This file defines a batch run through which the hiv modules are run across a grid of parameter values
Run on the batch system using:
```tlo batch-submit  src/scripts/hiv/PrEP_analyses/hiv_prep_baseline_scenario.py```
"""
import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tlo import Date
from tlo import logging
from tlo.analysis.utils import parse_log_file
from tlo.scenario import BaseScenario
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
            'filename': 'hiv_prep_baseline_scenario', 'directory': './outputs',
            'custom_levels': {
                '*': logging.WANRING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.demography': logging.INFO}
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
        ]

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
    `
    def draw_parameters(self, draw_number, rng):
        grid = self.make_grid(
            {
                'prob_for_prep_selection': np.linspace(start=0, stop=0.6, num=4),
                'prob_prep_adherence_level': [([0.3, 0.1, 0.6], [0.4, 0.2, 0.4], [0.5, 0.3, 0.2])],
            }
        )

        return {
            'hiv': {
                'prob_for_prep_selection': grid['prob_for_prep_selection'][draw_number],
            },
            'hiv': {
                'prob_prep_adherence_level': grid['prob_prep_adherence_level'][draw_number],
            },
        }
