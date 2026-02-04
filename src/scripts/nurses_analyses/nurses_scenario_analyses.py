"""
This scenario file sets up the scenarios for simulating the effects of nursing staffing levels
The scenarios are:
0- Baseline
1- Baseline Perfect Healthcare Seeking
2- Baseline Perfect Clinical Practice
3- Improved Staffing
4- Improved Perfect Healthcare Seeking
5- Improved Perfect Clinical Practice
6- Worst Case
7- Worst Perfect Healthcare Seeking
8- Worst Perfect Healthcare Seeking
"""
from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, get_root_path, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class StaffingScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.resources = get_root_path() / "resources"
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2035, 1, 1)
        self.pop_size = 200
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'nurses_scenario_outputs',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel() + [
            ImprovedHealthSystemAndCareSeekingScenarioSwitcher()]

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    def _default_of_all_scenarios(self) -> Dict:
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                'HealthSystem': {
                    'mode_appt_constraints': 1,
                    'mode_appt_constraints_postSwitch': 2,
                    "scale_to_effective_capabilities": True,
                    "year_HR_scaling_by_level_and_officer_type": 2027,
                    # This happens in the year before mode change, as the model calibration is done by that year
                    "year_mode_switch": 2020,
                    'cons_availability': 'default',
                    'cons_availability_postSwitch': "all",
                    # 'year_cons_availability_switch': 2027,
                    'yearly_HR_scaling_mode': 'historical_scaling',  # for 5 years of 2020-2024; source data year 2019
                    "policy_name": 'Naive',
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                },
                'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                    'max_healthcare_seeking': [False, False],
                    'max_healthsystem_function': [False, False],
                    'year_of_switch': 2027,
                }
            },
        )

    def _default_of_all_max_healthsystem_scenarios(self) -> Dict:
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                'HealthSystem': {
                    'mode_appt_constraints': 1,
                    'mode_appt_constraints_postSwitch': 2,
                    "scale_to_effective_capabilities": True,
                    "year_HR_scaling_by_level_and_officer_type": 2027,
                    # This happens in the year before mode change, as the model calibration is done by that year
                    "year_mode_switch": 2020,
                    'cons_availability': 'default',
                    'cons_availability_postSwitch': "all",
                    # 'year_cons_availability_switch': 2027,
                    'yearly_HR_scaling_mode': 'historical_scaling',  # for 5 years of 2020-2024; source data year 2019
                    "policy_name": 'Naive',
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                },
            },
        )

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        """
        return {
            "Baseline":
                mix_scenarios(
                    self._default_of_all_scenarios(),
                    {
                        "HealthSystem": {
                            'HR_scaling_by_level_and_officer_type_mode': "default",
                        },
                    }
                ),

            "Baseline Perfect Healthcare Seeking":
                mix_scenarios(
                    self._default_of_all_max_healthsystem_scenarios(),
                    {"HealthSystem": {
                        'HR_scaling_by_level_and_officer_type_mode': "default",
                    }},
                    {'ScenarioSwitcher': {
                        'max_healthsystem_function': [False] * 2,
                        'max_healthcare_seeking': [True] * 2,
                        'year_of_switch': 2027,
                    },
                    }
                ),

            "Baseline Perfect Clinical Practice":
                mix_scenarios(
                    self._default_of_all_max_healthsystem_scenarios(),
                    {"HealthSystem": {
                        'HR_scaling_by_level_and_officer_type_mode': "default",
                    }},
                    {'ScenarioSwitcher': {
                        'max_healthsystem_function': [True] * 2,
                        'max_healthcare_seeking': [True] * 2,
                        'year_of_switch': 2027,
                    },
                    }
                ),

            "Improved Staffing":
                mix_scenarios(
                    self._default_of_all_scenarios(),
                    {
                        "HealthSystem": {
                            'HR_scaling_by_level_and_officer_type_mode': "improved_staffing",
                        },
                    }
                ),

            "Improved Perfect Healthcare Seeking":
                mix_scenarios(
                    self._default_of_all_max_healthsystem_scenarios(),
                    {"HealthSystem": {
                        'HR_scaling_by_level_and_officer_type_mode': "improved_staffing",
                    }},
                    {'ScenarioSwitcher': {
                        'max_healthsystem_function': [False] * 2,
                        'max_healthcare_seeking': [True] * 2,
                        'year_of_switch': 2027,
                    },
                    }
                ),

            "Improved Perfect Clinical Practice":
                mix_scenarios(
                    self._default_of_all_max_healthsystem_scenarios(),
                    {"HealthSystem": {
                        'HR_scaling_by_level_and_officer_type_mode': "improved_staffing",
                    }},
                    {'ScenarioSwitcher': {
                        'max_healthsystem_function': [True] * 2,
                        'max_healthcare_seeking': [True] * 2,
                        'year_of_switch': 2027,
                    },
                    }
                ),

            "Worst Case":
                mix_scenarios(
                    self._default_of_all_scenarios(),
                    {
                        "HealthSystem": {
                            'HR_scaling_by_level_and_officer_type_mode': "custom_worse",
                        },
                    }
                ),

            "Worst Perfect Healthcare Seeking":
                mix_scenarios(
                    self._default_of_all_max_healthsystem_scenarios(),
                    {"HealthSystem": {
                        'HR_scaling_by_level_and_officer_type_mode': "custom_worse",
                    }},
                    {'ScenarioSwitcher': {
                        'max_healthsystem_function': [False] * 2,
                        'max_healthcare_seeking': [True] * 2,
                        'year_of_switch': 2027,
                    },
                    }
                ),

            "Worst Perfect Clinical Practice":
                mix_scenarios(
                    self._default_of_all_max_healthsystem_scenarios(),
                    {"HealthSystem": {
                        'HR_scaling_by_level_and_officer_type_mode': "custom_worse",
                    }},
                    {'ScenarioSwitcher': {
                        'max_healthsystem_function': [True] * 2,
                        'max_healthcare_seeking': [True] * 2,
                        'year_of_switch': 2027,
                    },
                    }
                ),
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
