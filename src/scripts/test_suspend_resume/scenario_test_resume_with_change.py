"""This Scenario file run the model under different assumptions for HR capabilities expansion in order to estimate the
impact that is achieved under each.

Run on the batch system using:
```
tlo batch-submit src/scripts/healthsystem/impact_of_policy/scenario_impact_of_const_capabilities_expansion.py
```

or locally using:
```
tlo scenario-run src/scripts/healthsystem/impact_of_policy/scenario_impact_of_const_capabilities_expansion.py
```

"""
from pathlib import Path
from typing import Dict

import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class ImpactOfHealthSystemMode(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(years=5)
        self.pop_size = 20_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'effect_of_capabilities_scaling',
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
            fullmodel()
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        """
        
        self.YEAR_OF_CHANGE = 2012

        return {
   
            # =========== STATUS QUO ============
            "Baseline":
                mix_scenarios(
                    self._baseline(),
                ),
        }
        """
            "Mode 2 no rescaling":
                mix_scenarios(
                    self._baseline(),
                    {
                     "HealthSystem": {
                        "mode_appt_constraints_postSwitch": 2,      # <-- Mode 2 post-change to show effects of HRH
                      },
                    }
                ),
 
             "Mode 2 with rescaling":
                mix_scenarios(
                    self._baseline(),
                    {
                     "HealthSystem": {
                        "mode_appt_constraints_postSwitch": 2,      # <-- Mode 2 post-change to show effects of HRH
                        "scale_to_effective_capabilities": True,
                      },
                    }
                ),
 
            "Mode 2 with rescaling and funded plus":
                mix_scenarios(
                    self._baseline(),
                    {
                     "HealthSystem": {
                        "mode_appt_constraints_postSwitch": 2,      # <-- Mode 2 post-change to show effects of HRH
                        "scale_to_effective_capabilities": True,
                        "use_funded_or_actual_staffing_postSwitch": "funded_plus",
                      },
                    }
                ),
 
            "Mode 1 perfect consumables":
                mix_scenarios(
                    self._baseline(),
                    {
                     "HealthSystem": {
                        "cons_availability_postSwitch":"perfect",
                      },
                    }
                ),

        }
        """
    def _baseline(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "HealthSystem": {
                    "year_mode_switch":self.YEAR_OF_CHANGE,
                    "year_cons_availability_switch":self.YEAR_OF_CHANGE,
                    "year_use_funded_or_actual_staffing_switch":self.YEAR_OF_CHANGE,
                    "mode_appt_constraints": 1,
                    "cons_availability": "default",
                    "use_funded_or_actual_staffing": "actual",
                    "mode_appt_constraints_postSwitch": 1,
                    "use_funded_or_actual_staffing_postSwitch":"actual",
                    "cons_availability_postSwitch":"default",
                    "policy_name": "Naive",
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,

                }
            },
        )
        

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
