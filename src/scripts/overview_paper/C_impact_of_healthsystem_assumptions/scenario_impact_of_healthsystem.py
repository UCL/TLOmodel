"""This Scenario file run the model under different assumptions for the HealthSystem in order to estimate the
impact that is achieved under each, relative to there being no health system.

Run on the batch system using:
```
tlo batch-submit src/scripts/overview_paper/C_impact_of_healthsystem_assumptions/scenario_impact_of_healthsystem.py
```

or locally using:
```
tlo scenario-run src/scripts/overview_paper/C_impact_of_healthsystem_assumptions/scenario_impact_of_healthsystem.py
 ```

"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class ImpactOfHealthSystemAssumptions(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 50_000  # Note smaller population size (should be 100_000 but runs on draws did not complete)
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'healthsystem_under_different_assumptions',
            'directory': Path('./outputs'),
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
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""

        return {
            "No Healthcare System":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                        'HealthSystem': {
                            'Service_Availability': []
                        }
                    },
                ),

            "With Hard Constraints":
                # N.B. This is for Mode 2 on continuously from the beginning of the simulation.
                # ... And with the "natural" (i.e., as coded in each disease module and not-overwritten) `tclose`
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     'HealthSystem': {
                        'mode_appt_constraints': 2,
                        "policy_name": "Naive",
                        }
                    },
                ),

            "Status Quo":
                mix_scenarios(
                    get_parameters_for_status_quo()
                ),

            "Perfect Healthcare Seeking":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'ScenarioSwitcher': {
                        'max_healthsystem_function': [False] * 2,
                        'max_healthcare_seeking': [True] * 2
                    }},
                    # (These changes start immediately and last for the full length of the simulation.)
                ),

            "+ Perfect Clinical Practice":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'ScenarioSwitcher': {
                        'max_healthsystem_function': [True] * 2,
                        'max_healthcare_seeking': [True] * 2
                    }},
                ),

            "+ Perfect Consumables Availability":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'ScenarioSwitcher': {
                        'max_healthsystem_function': [False] * 2,
                        'max_healthcare_seeking': [True] * 2}},
                    {'HealthSystem': {'cons_availability': 'all'}}
                ),
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
