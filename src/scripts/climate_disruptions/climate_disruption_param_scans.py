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
LHS_file = "src/scripts/climate_disruptions/lhs_parameter_draws.json"
start_index = 0
# Latin Hypercube parameters and generation done in src/scripts/climate_disruptions/generate_LHS_params_mode_1.py

with open(LHS_file, 'r') as f:
    LHS_grid_full = json.load(f)

parameter_grid  = LHS_grid_full[start_index:start_index + n_samples_to_use]
print(parameter_grid)
class ClimateDisruptionScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2041, 1, 12)
        self.pop_size = 100_000
        self.runs_per_draw = 1
        self.YEAR_OF_CHANGE = YEAR_OF_CHANGE
        self._parameter_grid = parameter_grid
        self.number_of_draws = len(self._parameter_grid)

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
                "tlo.methods.healthsystem": logging.INFO,
                "tlo.methods.population": logging.INFO,
            },
        }

    def modules(self):
        return fullmodel()

    def draw_parameters(self, draw_number, rng):
        params = self._parameter_grid[draw_number]
        return {
            "HealthSystem": {
                "services_affected_precip": params["services_affected_precip"],
                "climate_ssp": params["climate_ssp"],
                "climate_model_ensemble_model": params["climate_model_ensemble_model"],
                "year_effective_climate_disruptions": params["year_effective_climate_disruptions"],
                "scale_factor_prob_disruption": params["scale_factor_prob_disruption"],
                "delay_in_seeking_care_weather": params["delay_in_seeking_care_weather"],
                "scale_factor_reseeking_healthcare_post_disruption": params[
                    "scale_factor_reseeking_healthcare_post_disruption"],
                "scale_factor_appointment_urgency": params["scale_factor_appointment_urgency"],
                "scale_factor_severity_disruption_and_delay": params["scale_factor_severity_disruption_and_delay"],
                "prop_supply_side_disruptions": float(params["prop_supply_side_disruptions"]),
                "mode_appt_constraints": params["mode_appt_constraints"],
                "mode_appt_constraints_postSwitch": params["mode_appt_constraints_postSwitch"],
                "cons_availability": params["cons_availability"],
                "cons_availability_postSwitch": params["cons_availability_postSwitch"],
                "year_cons_availability_switch": params["year_cons_availability_switch"],
                "beds_availability": params["beds_availability"],
                "equip_availability": params["equip_availability"],
                "equip_availability_postSwitch": params["equip_availability_postSwitch"],
                "year_equip_availability_switch": params["year_equip_availability_switch"],
                "use_funded_or_actual_staffing": params["use_funded_or_actual_staffing"],
                "scale_to_effective_capabilities": params["scale_to_effective_capabilities"],
                "policy_name": params["policy_name"],
                "tclose_overwrite": params["tclose_overwrite"],
            },
            "SymptomManager": {
                "spurious_symptoms": True,
            },
        }

if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
