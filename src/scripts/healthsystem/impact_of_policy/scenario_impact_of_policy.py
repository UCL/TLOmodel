"""This Scenario file run the model under different assumptions for the HealthSystem Mode in order to estimate the
impact that is achieved under each (relative to there being no health system).

Run on the batch system using:
```
tlo batch-submit src/scripts/healthsystem/impact_of_policy/scenario_impact_of_policy.py
```

or locally using:
```
tlo scenario-run src/scripts/healthsystem/impact_of_policy/scenario_impact_of_policy.py
```

"""
from pathlib import Path
from typing import Dict

import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ScenarioSwitcher
from tlo.scenario import BaseScenario


class ImpactOfHealthSystemMode(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(years=20)
        self.pop_size = 75_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 4

    def log_configuration(self):
        return {
            'filename': 'effect_of_each_treatment',
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
        return fullmodel(resourcefilepath=self.resources) + [ScenarioSwitcher(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        """

        return {
            "Unlimited Efficiency Idealised Performance":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        'use_funded_or_actual_staffing': "actual",
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "Naive Idealised Performance":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "Naive",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "RMNCH Idealised Performance":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "RMNCH",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "Clinically Vulnerable Idealised Performance":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "ClinicallyVulnerable",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "Vertical Programmes Idealised Performance":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "VerticalProgrammes",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "EHP III Idealised Performance":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "EHP_III",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "LCOA EHP Idealised Performance":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "LCOA_EHP",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "Naive Idealised Performance Longer tclose":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "Naive",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "RMNCH Idealised Performance Longer tclose":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "RMNCH",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "Clinically Vulnerable Idealised Performance Longer tclose":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "ClinicallyVulnerable",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "Vertical Programmes Idealised Performance Longer tclose":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "VerticalProgrammes",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "EHP III Idealised Performance Longer tclose":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "EHP_III",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "LCOA_EHP Idealised Performance Longer tclose":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "all",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "LCOA_EHP",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                     'ScenarioSwitcher': {'max_healthsystem_function': True, 'max_healthcare_seeking': True}},
                ),

            "Unlimited Efficiency Status Quo cons":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "default",
                        "use_funded_or_actual_staffing": "actual",
                     },
                    }
                ),

            "Naive Status Quo cons":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "default",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "Naive",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                    }
                ),

            "RMNCH Status Quo cons":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "default",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "RMNCH",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                      },
                    }
                ),

            "Clinically Vulnerable Status Quo cons":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "default",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "ClinicallyVulnerable",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                    }),

            "Vertical Programmes Status Quo cons":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "default",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "VerticalProgrammes",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                    }),

            "EHP III Status Quo cons":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "default",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "EHP_III",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                    }),

            "LCOA EHP Status Quo cons":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'cons_availability': "default",
                        "use_funded_or_actual_staffing": "actual",
                        'year_policy_switch': 2011,
                        'mode_appt_constraints_postSwitch': 2,
                        "Policy_Name_postSwitch": "LCOA_EHP",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 1,
                     },
                    }
                )
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
