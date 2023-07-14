"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for analysis of TLO implementation re. HCW and health services usage, for the paper on HCW.

Run on the batch system using:
```tlo batch-submit src/scripts/healthsystem/hsi_in_typical_run/10_year_scale_run.py```

or locally using:
    ```tlo scenario-run src/scripts/healthsystem/hsi_in_typical_run/10_year_scale_run.py```

"""
from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ScenarioSwitcher
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 20_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'scale_run_for_hcw_analysis',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources) + [ScenarioSwitcher(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):
        return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""

        return {
            "Status Quo":
                mix_scenarios(
                    get_parameters_for_status_quo()
                ),

            "Establishment HCW":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {'use_funded_or_actual_staffing': 'funded_plus'}}
                ),

            "Perfect Healthcare Seeking":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'ScenarioSwitcher': {'max_healthsystem_function': False, 'max_healthcare_seeking': True}},
                ),

            "Establishment HCW + Perfect Healthcare Seeking":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {'use_funded_or_actual_staffing': 'funded_plus'}},
                    {'ScenarioSwitcher': {'max_healthsystem_function': False, 'max_healthcare_seeking': True}},
                ),
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
