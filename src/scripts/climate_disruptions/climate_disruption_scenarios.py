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
            "scale_factor_delay_in_seeking_care_weather": [float(28)],
            "rescaling_prob_seeking_after_disruption": [float(1)],
            "rescaling_prob_disruption": [float(1)],
            "scale_factor_severity_disruption_and_delay": [float(1)],
            "mode_appt_constraints": [1],
            "mode_appt_constraints_postSwitch": [2],
            "cons_availability": ["default"],
            "cons_availability_postSwitch": ["default"],
            "year_cons_availability_switch": [YEAR_OF_CHANGE],
            "beds_availability": ["default"],
            "equip_availability": ["default"],
            "equip_availability_postSwitch": ["default"],
            "year_equip_availability_switch": [YEAR_OF_CHANGE],
            "use_funded_or_actual_staffing": ["actual"],
            "scale_to_effective_capabilities": [True],
            "policy_name": ["Naive"],
            "climate_ssp": ["ssp126", "ssp245", "ssp585"],
            "year_effective_climate_disruptions": [2025],
            "climate_model_ensemble_model": ["lowest", "mean", "highest"],
            "services_affected_precip": ["all"],
            "tclose_overwrite": [1000],
            "prop_supply_side_disruptions": [0.5],
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
        self.runs_per_draw = 10
        self.YEAR_OF_CHANGE = 2025
        self._scenarios = self._get_scenarios()
        self._parameter_grid = full_grid  # Use all 9 combinations (3 SSPs × 3 ensemble models)
        self.number_of_draws = len(self._parameter_grid)
        with open("selected_parameter_combinations_ssp_scenarios.json", "w") as f:
            json.dump(self._parameter_grid, f, indent=2)

    def log_configuration(self):
        return {
            "filename": "climate_scenario_runs_all_ssp_ensemble",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.population": logging.INFO,
                "tlo.methods.enhanced_lifestyle": logging.INFO,
            },
        }

    def modules(self):
        return fullmodel() + [
            ImprovedHealthSystemAndCareSeekingScenarioSwitcher()
        ]

    def draw_parameters(self, draw_number, rng):
        return self._parameter_grid[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        # Single scenario definition that will be used with all parameter combinations
        return {
            "All SSP and Ensemble Combinations": self._scenario_all_climate(),
        }

    def _scenario_all_climate(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the scenario."""
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "ImprovedHealthSystemAndCareSeekingScenarioSwitcher": {
                    "max_healthsystem_function": [False, False],
                    "max_healthcare_seeking": [False, False],
                    "year_of_switch": self.YEAR_OF_CHANGE,
                },
                "Malaria": {
                    "type_of_scaleup": "max",
                    "scaleup_start_year": self.YEAR_OF_CHANGE,
                },
                "Tb": {
                    "type_of_scaleup": "max",
                    "scaleup_start_year": self.YEAR_OF_CHANGE,
                },
                "Hiv": {
                    "type_of_scaleup": "max",
                    "scaleup_start_year": self.YEAR_OF_CHANGE,
                },
            },
        )


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
