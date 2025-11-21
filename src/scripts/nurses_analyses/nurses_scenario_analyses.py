"""
This scenario file sets up the scenarios for simulating the effects of nursing staffing levels
The scenario
0- Baseline scenario
1-
2-


"""
from pathlib import Path
from typing import Dict

import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class StaffingScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2030, 1, 1)
        self.initial_population_size = 200
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.number_of_draws = 2
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
        return fullmodel(resourcefilepath=self.resources) + [
            ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)]

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
                    # This happens in the year before mode change, as the model calibration is done by that year
                    "year_mode_switch": 2020,
                    'cons_availability': 'default',
                    'cons_availability_postSwitch': "all",
                    # 'year_cons_availability_switch': 2025,
                    'HR_budget_growth_rate': self.hr_budget[0],
                    'yearly_HR_scaling_mode': 'historical_scaling',  # for 5 years of 2020-2024; source data year 2019
                    'start_year_HR_expansion_by_officer_type': self.YEAR_OF_HRH_EXPANSION,
                    'end_year_HR_expansion_by_officer_type': self.end_date.year,
                    "policy_name": 'Naive',
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                },
                'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                    'max_healthcare_seeking': [False, False],
                    'max_healthsystem_function': self.hs_function[0],
                    'year_of_switch': self.YEAR_OF_HRH_EXPANSION,
                }
            },
        )

    # def _baseline_scenario(self) -> Dict:
    #     return mix_scenarios(
    #         self._default_of_all_scenarios(),
    #         {
    #             'HealthSystem': {
    #                 'ResourceFile_HR_scaling_by_level_and_officer_type': "historical_scaling",
    #                 'mode_appt_constraints_postSwitch': 2,
    #                 "use_funded_or_actual_staffing": "actual",
    #             },
    #         },
    #     )
    #
    # def _improved_staffing_scenario(self) -> Dict:
    #     return mix_scenarios(
    #         self._default_of_all_scenarios(),
    #         {
    #             'HealthSystem': {
    #                 'ResourceFile_HR_scaling_by_level_and_officer_type': "historical_scaling",
    #                 'mode_appt_constraints_postSwitch': 2,
    #                 "use_funded_or_actual_staffing": "funded_plus",
    #             },
    #         },
    #     )
    #
    # def _worst_case_scenario(self) -> Dict:
    #     return mix_scenarios(
    #         self._default_of_all_scenarios(),
    #         {
    #             'HealthSystem': {
    #                 'ResourceFile_HR_scaling_by_level_and_officer_type': "historical_scaling",
    #                 'mode_appt_constraints_postSwitch': 2,
    #                 "use_funded_or_actual_staffing": "actual",
    #             },
    #         },
    #     )
    ####################################################################
    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        """
        return {
            "Baseline":
                mix_scenarios(
                    self._default_of_all_scenarios(),
                    {
                        "HealthSystem": {
                            'ResourceFile_HR_scaling_by_level_and_officer_type': "historical_scaling",
                            'mode_appt_constraints_postSwitch': 2,
                            "use_funded_or_actual_staffing": "actual",
                        },
                    }
                ),

            "Improved Staffing":
                mix_scenarios(
                    self._default_of_all_scenarios(),
                    {
                        "HealthSystem": {
                            'ResourceFile_HR_scaling_by_level_and_officer_type': "historical_scaling",
                            'mode_appt_constraints_postSwitch': 2,
                            "use_funded_or_actual_staffing": "funded_plus",
                        },
                    }
                ),

            "Worse Case":
                mix_scenarios(
                    self._default_of_all_scenarios(),
                    {
                        "HealthSystem": {
                            'ResourceFile_HR_scaling_by_level_and_officer_type': "historical_scaling",
                            'mode_appt_constraints_postSwitch': 2,
                            "use_funded_or_actual_staffing": "actual",
                        },
                    }
                ),
        }




    # To be sensitivity analysis
    # def _baseline_scenario(self) -> Dict:
    #     return mix_scenarios(
    #         self._default_of_all_scenarios(),
    #         {
    #             'HealthSystem': {
    #                 'ResourceFile_HR_scaling_by_level_and_officer_type': "historical_scaling",
    #                 'year_mode_switch': 2020,
    #                 'mode_appt_constraints_postSwitch': 2,
    #                 'scale_to_effective_capabilities': True,
    #                 "use_funded_or_actual_staffing": "actual",
    #                 "year_cons_availability_switch": 2025,
    #                 "cons_availability_postSwitch": "all",
    #             },
    #         },
    #     )


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
