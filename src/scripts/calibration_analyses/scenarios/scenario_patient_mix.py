"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for calibrations (demographic patterns, health burdens and healthsystem usage)

Run on the batch system using:
```tlo batch-submit src/scripts/calibration_analyses/scenarios/scenario_patient_mix.py```

or locally using:
    ```tlo scenario-run src/scripts/calibration_analyses/scenarios/scenario_patient_mix.py```

"""
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2025, 1, 1)  # The simulation will stop before reaching this date.
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'scenario_patient_mix',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.healthsystem': logging.DEBUG,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel()

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        return {
            "Mode 1 with historical HRH growth":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                        'HealthSystem': {
                            'yearly_HR_scaling_mode': 'historical_scaling',  # for 5 years between 2020-2024
                        },
                    }
                )
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
