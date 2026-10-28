"""This Scenario file runs the model under different assumptions for the historical changes in Human Resources for Health

Run on the batch system using:
```
tlo batch-submit src/scripts/impact_of_historical_changes_in_hr/scenario_historical_changes_in_hr_extended.py
```

Or locally using:
```

tlo scenario-run src/scripts/impact_of_historical_changes_in_hr/scenario_historical_changes_in_hr_extended.py

```

"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class HistoricalChangesInHRH(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2031, 1, 1)  # <-- End at the end of year 2030
        self.pop_size = 20_000  # <-- Local small run: 30; previous study run: 20_000; for publication: 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 5  # for publication: 10

    def log_configuration(self):
        return {
            'filename': 'historical_changes_in_hr_extended',
            'directory': Path('./outputs'),
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.WARNING,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return (
            fullmodel() + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher()]
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]

    def draw_name(self, draw_number) -> str:
        """Store scenario name.
        (This name can be retrieved by the plotting scripts to make the graphs be labelled nicely).
        """
        if draw_number < self.number_of_draws:
            return list(self._scenarios.keys())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""

        return {
            # "Main Counterfactual/No growth + Lower bound settings":
            # self._common_baseline(),
            #
            # "Main Actual":
            # self._hrh_growth_baseline(),
            #
            # "Historical growth (cadre-mix)":
            # mix_scenarios(
            #     self._common_baseline(),
            #     {
            #         "HealthSystem": {
            #             "HR_scaling_by_year_and_officer_type_mode": "historical_cadre_mix",
            #         }
            #     }
            # ),
            #
            # "Historical growth + LCOA policy":
            # mix_scenarios(
            #     self._hrh_growth_baseline(),
            #     {
            #         "HealthSystem": {
            #             "policy_name": "LCOA_EHP",
            #         }
            #     }
            # ),
            #
            # "Historical growth + Consumables (better)":
            # mix_scenarios(
            #     self._hrh_growth_baseline(),
            #     {
            #         "HealthSystem": {
            #             "cons_availability_postSwitch": "scenario6",
            #         }
            #     }
            # ),
            #
            # "Historical growth + Consumables (perfect)":
            #     mix_scenarios(
            #         self._hrh_growth_baseline(),
            #         {
            #             "HealthSystem": {
            #                 "cons_availability_postSwitch": "all",
            #             }
            #         }
            #     ),
            #
            # "Historical growth + Low absorption rate/Historical growth + Lower bound settings":
            # mix_scenarios(
            #     self._hrh_growth_baseline(),
            #     {
            #         "HealthSystem": {
            #             "HR_expansion_absorption_rate": 0.5,
            #         }
            #     }
            # ),
            #
            # "Historical growth + Max HS performance":
            # mix_scenarios(
            #     self._hrh_growth_baseline(),
            #     {
            #         'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
            #             'max_healthsystem_function': [False, True],
            #         }
            #     }
            # ),
            #
            # "No growth + LCOA policy":
            #     mix_scenarios(
            #         self._common_baseline(),
            #         {
            #             "HealthSystem": {
            #                 "policy_name": "LCOA_EHP",
            #             }
            #         }
            #     ),
            #
            # "No growth + Consumables (better)":
            #     mix_scenarios(
            #         self._common_baseline(),
            #         {
            #             "HealthSystem": {
            #                 "cons_availability_postSwitch": "scenario6",
            #             }
            #         }
            #     ),
            #
            # "No growth + Consumables (perfect)":
            #     mix_scenarios(
            #         self._common_baseline(),
            #         {
            #             "HealthSystem": {
            #                 "cons_availability_postSwitch": "all",
            #             }
            #         }
            #     ),
            #
            # "No growth + Max HS performance":
            #     mix_scenarios(
            #         self._common_baseline(),
            #         {
            #             'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
            #                 'max_healthsystem_function': [False, True],
            #             }
            #         }
            #     ),
            #
            # "No growth + Upper bound settings":
            #     mix_scenarios(
            #         self._common_baseline(),
            #         {
            #             'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
            #                 'max_healthsystem_function': [False, False],
            #             },
            #             "HealthSystem": {
            #                 "policy_name": "LCOA_EHP",
            #                 "cons_availability_postSwitch": "all",
            #             }
            #         }
            #     ),
            #
            # "Historical growth + Upper bound settings":
            #     mix_scenarios(
            #         self._common_baseline(),
            #         {
            #             'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
            #                 'max_healthsystem_function': [False, False],
            #             },
            #             "HealthSystem": {
            #                 "policy_name": "LCOA_EHP",
            #                 "cons_availability_postSwitch": "all",
            #                 "HR_scaling_by_year_and_officer_type_mode": "historical_cadre_mix",
            #             }
            #         }
            #     ),

            "No growth + Upper bound settings++":
                mix_scenarios(
                    self._common_baseline(),
                    {
                        'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                            'max_healthsystem_function': [False, True],
                        },
                        "HealthSystem": {
                            "policy_name": "LCOA_EHP",
                            "cons_availability_postSwitch": "all",
                        }
                    }
                ),

            "Historical growth + Upper bound settings++":
                mix_scenarios(
                    self._common_baseline(),
                    {
                        'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                            'max_healthsystem_function': [False, True],
                        },
                        "HealthSystem": {
                            "policy_name": "LCOA_EHP",
                            "cons_availability_postSwitch": "all",
                            "HR_scaling_by_year_and_officer_type_mode": "historical_cadre_mix",
                        }
                    }
                ),

        }

    def _hrh_growth_baseline(self) -> Dict:
        return mix_scenarios(
            self._common_baseline(),
            {
                "HealthSystem": {
                    "HR_scaling_by_year_and_officer_type_mode": "historical_uniform",
                }
            },
        )

    def _common_baseline(self) -> Dict:
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "HealthSystem": {
                    "mode_appt_constraints": 1,                 # <-- Mode 1 prior to change to preserve calibration
                    "mode_appt_constraints_postSwitch": 2,      # <-- Mode 2 post-change to show effects of HRH
                    "scale_to_effective_capabilities": True,    # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                    "year_mode_switch": 2020,    # <-- transition happens at start of 2020 when HRH starts to grow

                    # Normalize the behaviour of Mode 2
                    "policy_name": "Naive",   # -- *For the alternative scenario of efficient implementation of EHP, otherwise use 'naive'* --
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,

                    # Clarify the consumable availability
                    "cons_availability": "default",
                    "cons_availability_postSwitch": "default",
                    "year_cons_availability_switch": 2020,

                    # Clarify the historical HRH growth mode between 2020-2024
                    "yearly_HR_scaling_mode": 'no_scaling',
                    "HR_scaling_by_year_and_officer_type_mode": 'no_historical_growth',

                    # Clarify the HRH expansion absorption rate
                    "HR_expansion_absorption_rate": 1.0,

                },
                # -- *For the alternative scenario of increased demand and improved clinician performance* --
                'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                    'max_healthcare_seeking': [False, False],  # <-- switch from False to True mid-way
                    'max_healthsystem_function': [False, False],
                    'year_of_switch': 2020,
                }
            },
        )


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
