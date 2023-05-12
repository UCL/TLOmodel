"""This Scenario file run the model under different assumptions for the HealthSystem in order to estimate the
impact that is achieved under each, relative to there being no health system.

Run on the batch system using:
```
tlo batch-submit src/scripts/healthsystem/impact_of_healthsystem_under_diff_scenarios/scenario_impact_of_healthsystem.py
```

or locally using:
```
tlo scenario-run src/scripts/healthsystem/impact_of_healthsystem_under_diff_scenarios/scenario_impact_of_healthsystem.py
```



"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class ImpactOfHealthSystemAssumptions(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 5

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
        return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        0. "No Healthcare System"

        1. "Defaults":
            Normal healthcare seeking,
            Default consumables,

        2. "Perfect Healthcare Seeking":
            Perfect healthcare seeking,
            Default consumables,

        3. "Perfect Consumables":
            Normal healthcare seeking,
            100% availability of consumables,

        4. All changes:
            Perfect healthcare seeking,
            100% availability of consumables,
        """

        return {
            "No Healthcare System": {
                'HealthSystem': {
                    'Service_Availability': []
                },
            },

            "Defaults": {
                'HealthSystem': {
                    'Service_Availability': ['*'],
                    'cons_availability': 'default',
                },
                'HealthSeekingBehaviour': {
                    'force_any_symptom_to_lead_to_healthcareseeking': False
                },
            },

            "Perfect Healthcare Seeking": {
                'HealthSystem': {
                    'Service_Availability': ['*'],
                    'cons_availability': 'default',
                },
                'HealthSeekingBehaviour': {
                    'force_any_symptom_to_lead_to_healthcareseeking': True
                },
            },

            "Perfect Consumables Availability": {
                'HealthSystem': {
                    'Service_Availability': ['*'],
                    'cons_availability': 'all',
                },
                'HealthSeekingBehaviour': {
                    'force_any_symptom_to_lead_to_healthcareseeking': False
                },
            },

            "All Changes": {
                'HealthSystem': {
                    'Service_Availability': ['*'],
                    'cons_availability': 'all',
                },
                'HealthSeekingBehaviour': {
                    'force_any_symptom_to_lead_to_healthcareseeking': True
                },
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
