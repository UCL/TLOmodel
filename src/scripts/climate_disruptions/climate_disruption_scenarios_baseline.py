import json
import random
from typing import Dict

import numpy as np

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario, make_cartesian_parameter_grid

YEAR_OF_CHANGE = 2025

# Create grid with all combinations of SSP and ensemble model
full_grid = make_cartesian_parameter_grid(
    {
        "HealthSystem": {
            "scale_factor_reseeking_healthcare_post_disruption": [float(2)],
            "scale_factor_prob_disruption": [float(2)],
            "delay_in_seeking_care_weather": [float(60)],
            "scale_factor_appointment_urgency": [float(2)],
            "scale_factor_severity_disruption_and_delay": [float(2)],
            "mode_appt_constraints": [1],
            "mode_appt_constraints_postSwitch": [1],
            "cons_availability": ["default"],
            "cons_availability_postSwitch": ["default"],
            "year_cons_availability_switch": [YEAR_OF_CHANGE],
            "beds_availability": ["default"],
            "equip_availability": ["default"],
            "equip_availability_postSwitch": ["default"],
            "year_equip_availability_switch": [YEAR_OF_CHANGE],
            "use_funded_or_actual_staffing": ["actual"],
            "scale_to_effective_capabilities": [False],
            "policy_name": ["Naive"],
            "climate_ssp": ["ssp245"],
            "climate_model_ensemble_model": ["mean"],
            "year_effective_climate_disruptions": [2025],
            "prop_supply_side_disruptions": [0.5], # moot in mode 1
            "services_affected_precip": ["none", "all"], # none nullifies all other climate impacts
            "tclose_overwrite": [1000],
        }
    }
)


class ClimateDisruptionScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2041, 1, 12)
        self.pop_size = 100_000
        self.runs_per_draw = 5
        self.YEAR_OF_CHANGE = 2025
        self._parameter_grid = full_grid
        self.number_of_draws = 1

        with open("selected_parameter_combinations_baseline.json", "w") as f:
            json.dump(self._parameter_grid, f, indent=2)

    def log_configuration(self):
        return {
            "filename": "climate_suspend_resume_base",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.population": logging.INFO,
            },
        }

    def modules(self):
        return fullmodel()

    def draw_parameters(self, draw_number, rng):
        return self._parameter_grid

if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
