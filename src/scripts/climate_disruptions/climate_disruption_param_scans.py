from typing import Dict
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import (
    ImprovedHealthSystemAndCareSeekingScenarioSwitcher,
)
from tlo.scenario import BaseScenario

YEAR_OF_CHANGE = 2025
n_samples_to_use = 200
LHS_file = "/Users/rem76/PycharmProjects/TLOmodel/src/scripts/climate_disruptions/lhs_parameter_draws.json"
start_index = 0
# Latin Hypercube parameters and generation done in src/scripts/climate_disruptions/generate_LHS_params_mode_1.py

with open(LHS_file, 'r') as f:
    LHS_grid_full = json.load(f)

parameter_grid  = LHS_grid_full[start_index:start_index + n_samples_to_use]

class ClimateDisruptionScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2041, 1, 12)
        self.pop_size = 100_000
        self.runs_per_draw = 5
        self.YEAR_OF_CHANGE = YEAR_OF_CHANGE
        self._parameter_grid = parameter_grid[0]
        self.number_of_draws = 1#len(self._parameter_grid)

    def log_configuration(self):
        return {
            "filename": "climate_scenario_runs_lhs_param_scan",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.WARNING,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.population": logging.INFO,
            },
        }

    def modules(self):
        return fullmodel()

    def draw_parameters(self, draw_number, rng):
        return self._parameter_grid[draw_number]


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
