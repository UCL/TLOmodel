"""This Scenario file runs the model under different prioritisation policies.

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
        self.end_date = self.start_date + pd.DateOffset(years=33)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'effect_of_policy',
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
            "No Healthcare System":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'Service_Availability': []
                      },
                    }
                ),

            "Naive status quo":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'use_funded_or_actual_staffing': "actual",
                        'year_mode_switch': 2023,
                        'mode_appt_constraints_postSwitch': 2,
                        'policy_name': "Naive",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                      },
                    }
                ),

            "RMNCH status quo":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'use_funded_or_actual_staffing': "actual",
                        'year_mode_switch': 2023,
                        'mode_appt_constraints_postSwitch': 2,
                        'policy_name': "RMNCH",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                      },
                    }
                ),

            "Clinically Vulnerable status quo":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'use_funded_or_actual_staffing': "actual",
                        'year_mode_switch': 2023,
                        'mode_appt_constraints_postSwitch': 2,
                        'policy_name': "ClinicallyVulnerable",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                    }),

            "Vertical Programmes status quo":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'use_funded_or_actual_staffing': "actual",
                        'year_mode_switch': 2023,
                        'mode_appt_constraints_postSwitch': 2,
                        'policy_name': "VerticalProgrammes",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                    }),
            
            "CVD status quo":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'use_funded_or_actual_staffing': "actual",
                        'year_mode_switch': 2023,
                        'mode_appt_constraints_postSwitch': 2,
                        'policy_name': "CVD",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                    }),

            "EHP III status quo":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'use_funded_or_actual_staffing': "actual",
                        'year_mode_switch': 2023,
                        'mode_appt_constraints_postSwitch': 2,
                        'policy_name': "EHP_III",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                    }),

            "LCOA EHP status quo":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'use_funded_or_actual_staffing': "actual",
                        'year_mode_switch': 2023,
                        'mode_appt_constraints_postSwitch': 2,
                        'policy_name': "LCOA_EHP",
                        'tclose_overwrite': 1,
                        'tclose_days_offset_overwrite': 7,
                     },
                    }),
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
