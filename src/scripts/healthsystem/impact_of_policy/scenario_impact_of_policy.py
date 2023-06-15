"""This Scenario file run the model under different assumptions for the HealthSystem Mode in order to estimate the
impact that is achieved under each (relative to there being no health system).

Run on the batch system using:
```
tlo batch-submit src/scripts/healthsystem/impact_of_mode/scenario_impact_of_mode.py
```

or locally using:
```
tlo scenario-run src/scripts/healthsystem/impact_of_mode/scenario_impact_of_mode.py
```

"""
from pathlib import Path
from typing import Dict

import pandas as pd

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class ImpactOfHealthSystemMode(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(months=5)
        self.pop_size = 100
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 1

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
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        """

        return {
            "No Healthcare System": {
                'HealthSystem': {
                    'cons_availability': "all",
                    'Service_Availability': []
                },
            },

            "Unlimited Resources all cons": {
                'HealthSystem': {
                    'cons_availability': "all",
                    'mode_appt_constraints': 0,
                    "use_funded_or_actual_staffing": "actual",
                 },
            },

            "Unlimited Efficiency all cons": {
                'HealthSystem': {
                    'cons_availability': "all",
                    'mode_appt_constraints': 1,
                    "use_funded_or_actual_staffing": "actual",
                 },
            },

            "Random all cons": {
                'HealthSystem': {
                    'cons_availability': "all",
                    'mode_appt_constraints': 2,
                    "use_funded_or_actual_staffing": "actual",
                    "Policy_Name": "Random"
                 },
            },

            "Naive all cons": {
                'HealthSystem': {
                    'cons_availability': "all",
                    'mode_appt_constraints': 2,
                    "use_funded_or_actual_staffing": "actual",
                    "Policy_Name": "Naive"
                 },
            },

            "RMNCH all cons": {
                'HealthSystem': {
                    'cons_availability': "all",
                    'mode_appt_constraints': 2,
                    "use_funded_or_actual_staffing": "actual",
                    "Policy_Name": "RMNCH"
                 },
            },

            "Clinically Vulnerable all cons": {
                'HealthSystem': {
                    'cons_availability': "all",
                    'mode_appt_constraints': 2,
                    "use_funded_or_actual_staffing": "actual",
                    "Policy_Name": "ClinicallyVulnerable"
                 },
            },

            "Vertical Programmes all cons": {
                'HealthSystem': {
                    'cons_availability': "all",
                    'mode_appt_constraints': 2,
                    "use_funded_or_actual_staffing": "actual",
                    "Policy_Name": "VerticalProgrammes"
                 },
            },

            "EHP1_binary all cons": {
                'HealthSystem': {
                    'cons_availability': "all",
                    'mode_appt_constraints': 2,
                    "use_funded_or_actual_staffing": "actual",
                    "Policy_Name": "EHP1_binary"
                 },
            },

            "EHP3_LPP_binary all cons": {
                'HealthSystem': {
                    'cons_availability': "all",
                    'mode_appt_constraints': 2,
                    "use_funded_or_actual_staffing": "actual",
                    "Policy_Name": "EHP3_LPP_binary"
                 },
            },

            "Unlimited Resources default cons": {
                 'HealthSystem': {
                     'cons_availability': "default",
                     'mode_appt_constraints': 0,
                     "use_funded_or_actual_staffing": "actual",
                  },
            },

            "Unlimited Efficiency default cons": {
                 'HealthSystem': {
                     'cons_availability': "default",
                     'mode_appt_constraints': 1,
                     "use_funded_or_actual_staffing": "actual",
                  },
            },

            "Random default cons": {
                 'HealthSystem': {
                     'cons_availability': "default",
                     'mode_appt_constraints': 2,
                     "use_funded_or_actual_staffing": "actual",
                     "Policy_Name": "Random"
                  },
            },

            "Naive default cons": {
                  'HealthSystem': {
                     'cons_availability': "default",
                     'mode_appt_constraints': 2,
                     "use_funded_or_actual_staffing": "actual",
                     "Policy_Name": "Naive"
                  },
            },

            "RMNCH default cons": {
                 'HealthSystem': {
                     'cons_availability': "default",
                     'mode_appt_constraints': 2,
                     "use_funded_or_actual_staffing": "actual",
                     "Policy_Name": "RMNCH"
                  },
            },

            "Clinically Vulnerable default cons": {
                 'HealthSystem': {
                     'cons_availability': "default",
                     'mode_appt_constraints': 2,
                     "use_funded_or_actual_staffing": "actual",
                     "Policy_Name": "ClinicallyVulnerable"
                  },
            },

            "Vertical Programmes default cons": {
                 'HealthSystem': {
                     'cons_availability': "default",
                     'mode_appt_constraints': 2,
                     "use_funded_or_actual_staffing": "actual",
                     "Policy_Name": "VerticalProgrammes"
                  },
            },

            "EHP1_binary default cons": {
                 'HealthSystem': {
                     'cons_availability': "default",
                     'mode_appt_constraints': 2,
                     "use_funded_or_actual_staffing": "actual",
                     "Policy_Name": "EHP1_binary"
                  },
            },

            "EHP3_LPP_binary default cons": {
                 'HealthSystem': {
                     'cons_availability': "default",
                     'mode_appt_constraints': 2,
                     "use_funded_or_actual_staffing": "actual",
                     "Policy_Name": "EHP3_LPP_binary"
                 },
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
