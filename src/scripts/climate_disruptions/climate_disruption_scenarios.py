from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario, make_cartesian_parameter_grid
import numpy as np
import random
import json

YEAR_OF_CHANGE = 2020
full_grid = make_cartesian_parameter_grid({
    "HealthSystem": {
        "scale_factor_delay_in_seeking_care_weather": [1], #[float(x) for x in [0, 1, 2, 10, 20, 60]],
        "rescaling_prob_seeking_after_disruption": [1], #np.arange(0.01, 1.51, 0.5),
        "rescaling_prob_disruption": [1], #np.arange(0.0, 2.01, 0.5),
        "scale_factor_severity_disruption_and_delay": [1], #[float(x) for x in np.linspace(0.11, 1.0, 4)],
        "mode_appt_constraints": [1],
        "cons_availability": ["default"],
        "cons_availability_postSwitch": ["all"],
        "year_cons_availability_switch": [YEAR_OF_CHANGE],
        "beds_availability": ["all"],
        "equip_availability": ["default"],
        "equip_availability_postSwitch": ["all"],
        "year_equip_availability_switch": [YEAR_OF_CHANGE],
        "use_funded_or_actual_staffing": ["funded_plus"],
        "climate_ssp": ['ssp245'],
        "climate_model_ensemble_model": ['mean'],
        "services_affected_precip": ['all'],
        "tclose_overwrite": [1000]
    }
})

class ClimateDisruptionScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2014, 1, 12)
        self.pop_size = 100_000
        self.runs_per_draw = 1
        self.YEAR_OF_CHANGE = 2020
        self._scenarios = self._get_scenarios()
        self._parameter_grid = full_grid #random.sample(full_grid, 1)
        print(self._parameter_grid)
        self.number_of_draws = 1#len(self._parameter_grid)

        with open("selected_parameter_combinations.json", "w") as f:
            json.dump(self._parameter_grid, f, indent=2)

    def log_configuration(self):
        return {
            'filename': 'climate_scenario_runs',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.population': logging.INFO,
                "tlo.methods.enhanced_lifestyle": logging.INFO
            }
        }

    def modules(self):
        return (
            fullmodel(resourcefilepath=self.resources)
            + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)]
        )

    def draw_parameters(self, draw_number, rng):
        return self._parameter_grid#[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        return {
                'SSP 2.45 Mean': self._ssp245_mean(),
                }

    def _ssp245_mean(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, False],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                "climate_ssp":'ssp245',
                "climate_model_ensemble_model":'mean',
                "services_affected_precip":'all',
                 "tclose_overwrite": 1000 # days of t close overwrite
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
