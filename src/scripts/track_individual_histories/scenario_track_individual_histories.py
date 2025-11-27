"""This Scenario file run the model to track individual histories

Run on the batch system using:
```
tlo batch-submit 
    src/scripts/analysis_data_generation/scenario_track_individual_histories.py
```

or locally using:
```
    tlo scenario-run src/scripts/analysis_data_generation/scenario_track_individual_histories.py
```

"""
from pathlib import Path
from typing import Dict

import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class GenerateEventChains(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 42
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(months=1)
        self.pop_size = 1000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 3
        self.generate_event_chains = True

    def log_configuration(self):
        return {
            'filename': 'track_individual_histories',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.events': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.individual_history_tracker': logging.INFO
            }
        }

    def modules(self):
        return (
            fullmodel()
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    def _get_scenarios(self) -> Dict[str, Dict]:

        return {
            "Baseline":
                mix_scenarios(
                    self._baseline(),
                    {
                    }
                ),

        }
        
    def _baseline(self) -> Dict:
        #Return the Dict with values for the parameter changes that define the baseline scenario.
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "HealthSystem": {
                    "mode_appt_constraints": 1,                 # <-- Mode 1 prior to change to preserve calibration
                }
            },
        )

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
