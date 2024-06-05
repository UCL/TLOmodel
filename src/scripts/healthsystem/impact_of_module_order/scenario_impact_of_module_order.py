"""This Scenario file run the model under different assumptions for HR capabilities expansion in order to estimate the
impact that is achieved under each.

Run on the batch system using:
```
tlo batch-submit src/scripts/healthsystem/impact_of_policy/scenario_impact_of_module_order.py
```

or locally using:
```
tlo scenario-run src/scripts/healthsystem/impact_of_policy/scenario_impact_of_module_order.py
```

"""
from pathlib import Path
from typing import Dict

import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class ImpactOfHealthSystemMode(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(years=6)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'effect_of_module_order',
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
        return (
            fullmodel(resourcefilepath=self.resources)
            + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)]
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        """
        
        self.YEAR_OF_CHANGE = 2011

        return {
   
            "Original order":
                mix_scenarios(
                    self._baseline(),
                ),
                
            #"Invert HIV and CMD":
            #    mix_scenarios(
            #        self._baseline(),
            #    ),

            #"RTI first and HTM last":
            #    mix_scenarios(
            #        self._baseline(),
            #    ),
                
            #"Random shuffle 1":
            #    mix_scenarios(
            #        self._baseline(),
            #    ),
                
            #"Random shuffle 2":
            #    mix_scenarios(
            #       self._baseline(),
            #    ),

            #"Random shuffle 3":
            #    mix_scenarios(
            #        self._baseline(),
            #    ),
        }
        
    def _baseline(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "HealthSystem": {
                    "mode_appt_constraints": 1,                 # <-- Mode 1 prior to change to preserve calibration
                    "mode_appt_constraints_postSwitch": 2,      # <-- Mode 2 post-change to show effects of HRH
                    "year_mode_switch": self.YEAR_OF_CHANGE,
                    "policy_name": "Naive",
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                    "use_funded_or_actual_staffing": "actual",
                    "cons_availability": "default",
                }
            },
        )

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
