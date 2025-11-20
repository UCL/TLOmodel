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
        self.seed=12
        self.start_date=Date(2010, 1, 1)
        self.end_date=Date(2030, 1, 1)
        self.initial_population_size=200
        self.number_of_draws=2
        self.runs_per_draw=2

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
        return fullmodel(resourcefilepath=self.resources) + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

        def _get_scenarios(self) -> Dict[str, Dict]:
            """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
            """
            return {
                "Baseline":
                    mix_scenarios(
                        get_parameters_for_status_quo(),
                        {
                            "HealthSystem": {
                                "ResourceFile_HR_scaling_by_level_and_officer_type": "default",
                                "year_mode_switch": 2020,
                                "mode_appt_constraints_postSwitch": 2,
                                "scale_to_effective_capabilities": True,
                                "policy_name": "Naive",
                                "tclose_overwrite": 1,
                                "tclose_days_offset_overwrite": 7,
                                "use_funded_or_actual_staffing": "actual",
                                "year_cons_availability_switch": 2025,
                                "cons_availability_postSwitch": "all",
                            },
                        }
                    ),

                "Improved Staffing":
                    mix_scenarios(
                        get_parameters_for_status_quo(),
                        {
                            "HealthSystem": {
                                "ResourceFile_HR_scaling_by_level_and_officer_type": "default",
                                "year_mode_switch": 2020,
                                "mode_appt_constraints_postSwitch": 2,
                                "scale_to_effective_capabilities": True,
                                "policy_name": "Naive",
                                "tclose_overwrite": 1,
                                "tclose_days_offset_overwrite": 7,
                                "use_funded_or_actual_staffing": "funded_plus",
                                "year_cons_availability_switch": 2025,
                                "cons_availability_postSwitch": "all",
                            },
                        }
                    ),

                "Worst-case Scenario":
                    mix_scenarios(
                        get_parameters_for_status_quo(),
                        {
                            "HealthSystem": {
                                "yearly_HR_scaling_mode": "historical_scaling",
                                "year_mode_switch": 2020,
                                "mode_appt_constraints_postSwitch": 2,
                                "scale_to_effective_capabilities": True,
                                "policy_name": "Naive",
                                "tclose_overwrite": 1,
                                "tclose_days_offset_overwrite": 7,
                                "use_funded_or_actual_staffing": "actual",
                                "year_cons_availability_switch": 2019,
                                "cons_availability_postSwitch": "all",
                            },
                        }
                    ),

                "Demand Sensitivity":
                    mix_scenarios(
                        get_parameters_for_status_quo(),
                        {
                            "HealthSystem": {
                                "yearly_HR_scaling_mode": "historical_scaling",
                                "year_mode_switch": 2020,
                                "mode_appt_constraints_postSwitch": 2,
                                "scale_to_effective_capabilities": True,
                                "policy_name": "Naive",
                                "tclose_overwrite": 1,
                                "tclose_days_offset_overwrite": 7,
                                "use_funded_or_actual_staffing": "actual",
                                "year_cons_availability_switch": 2019,
                                "cons_availability_postSwitch": "all",
                            },
                        }
                    ),

#Look into doing sensitivity analyses in the model
            "Time appointments Sensitivity":
            mix_scenarios(
                get_parameters_for_status_quo(),
                {
                    "HealthSystem": {
                        "yearly_HR_scaling_mode": "historical_scaling",
                        "year_mode_switch": 2020,
                        "mode_appt_constraints_postSwitch": 2,
                        "scale_to_effective_capabilities": True,
                        "policy_name": "Naive",
                        "tclose_overwrite": 1,
                        "tclose_days_offset_overwrite": 7,
                        "use_funded_or_actual_staffing": "actual",
                        "year_cons_availability_switch": 2019,
                        "cons_availability_postSwitch": "all",
                    },
                }
            ),
            }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
