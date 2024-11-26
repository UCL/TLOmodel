'''
Run on the batch system using:
```tlo batch-submit src/scripts/costing/example_costing_scenario.py```

or locally using:
    ```tlo scenario-run src/scripts/costing/example_costing_scenario.py```

'''

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario

class CostingScenarios(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2030, 1, 1)
        self.pop_size = 1_000  # <- recommended population size for the runs
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 2 # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'cost_scenarios',
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return (fullmodel(resourcefilepath=self.resources) +
                [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)])

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""

        self.YEAR_OF_SYSTEM_CHANGE = 2020
        self.mode_appt_constraints_postSwitch = [1,2]
        self.cons_availability = ['default', 'all']
        self.healthsystem_function = [[False, False], [False, True]]
        self.healthcare_seeking = [[False, False], [False, True]]

        return {
            "Real world": self._common_baseline(),

            "Perfect health system":
                mix_scenarios(
                    self._common_baseline(),
                    {
                    'HealthSystem': {
                    # Human Resources
                    'mode_appt_constraints_postSwitch': self.mode_appt_constraints_postSwitch[1], # <-- Mode 2 post-change to show effects of HRH
                    "scale_to_effective_capabilities": True,  # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                    "year_mode_switch": self.YEAR_OF_SYSTEM_CHANGE,

                    # Consumables
                    'cons_availability_postSwitch': self.cons_availability[1],
                    'year_cons_availability_switch': self.YEAR_OF_SYSTEM_CHANGE,
                },
                'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                    'max_healthcare_seeking': self.healthcare_seeking[1],
                    'max_healthsystem_function': self.healthsystem_function[1],
                    'year_of_switch': self.YEAR_OF_SYSTEM_CHANGE,
                }
                    }
                ),
        }

    def _common_baseline(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(), # <-- Parameters that have been the calibration targets
            # Set up the HealthSystem to transition from Mode 1 -> Mode 2, with rescaling when there are HSS changes
            {
                'HealthSystem': {
                    # Human resources
                    'mode_appt_constraints': 1, # <-- Mode 1 prior to change to preserve calibration
                    'mode_appt_constraints_postSwitch': self.mode_appt_constraints_postSwitch[0], # <-- Mode 2 post-change to show effects of HRH
                    "scale_to_effective_capabilities": True,  # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                    # This happens in the year before mode change, as the model calibration is done by that year
                    "year_mode_switch": self.YEAR_OF_SYSTEM_CHANGE,
                    'yearly_HR_scaling_mode': 'historical_scaling',  # for 5 years of 2020-2024; source data year 2019

                    # Consumables
                    'cons_availability': 'default',
                    'cons_availability_postSwitch': self.cons_availability[0],
                    'year_cons_availability_switch': self.YEAR_OF_SYSTEM_CHANGE,

                    # Normalize the behaviour of Mode 2
                    "policy_name": 'Naive',
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                },
                'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                    'max_healthcare_seeking': self.healthcare_seeking[0],
                    'max_healthsystem_function': self.healthsystem_function[0],
                    'year_of_switch': self.YEAR_OF_SYSTEM_CHANGE,
                }
            },
        )


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
