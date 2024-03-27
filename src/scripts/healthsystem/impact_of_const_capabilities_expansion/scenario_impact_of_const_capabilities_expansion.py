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
from tlo.scenario import BaseScenario


class ImpactOfHealthSystemMode(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(years=31) #CHECK
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 2

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
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    # case 1: gfHE = -0.030, factor = 1.01074
    # case 2: gfHE = -0.020, factor = 1.02116
    # case 3: gfHE = -0.015, factor = 1.02637
    # case 4: gfHE =  0.015, factor = 1.05763
    # case 5: gfHE =  0.020, factor = 1.06284
    # case 6: gfHE =  0.030, factor = 1.07326

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        """
        return {
            "No growth":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                    # MUST ADD SWITCH TO PERFECT CONSUMABLE AVAILABILITY IN 2018
                     "HealthSystem": {
                        "yearly_HR_scaling_mode": "no_scaling",
                        "year_mode_switch": 2019,
                        "mode_appt_constraints_postSwitch": 2,
                        "policy_name": "Naive",
                        "tclose_overwrite": 1,
                        "tclose_days_offset_overwrite": 7,
                        "use_funded_or_actual_staffing": "actual",
                        "year_cons_availability_switch": 2019,
                        "cons_availability_postSwitch": "all",
                      },
                    }
                ),

            "GDP growth":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     "HealthSystem": {
                        "yearly_HR_scaling_mode": "GDP_growth",
                        "year_mode_switch": 2019,
                        "mode_appt_constraints_postSwitch": 2,
                        "policy_name": "Naive",
                        "tclose_overwrite": 1,
                        "tclose_days_offset_overwrite": 7,
                        "use_funded_or_actual_staffing": "actual",
                        "year_cons_availability_switch": 2019,
                        "cons_availability_postSwitch": "all",
                      },
                    }
                ),

            "GDP growth fHE growth case 1":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     "HealthSystem": {
                        "yearly_HR_scaling_mode": "GDP_growth_fHE_case1",
                        "year_mode_switch": 2019,
                        "mode_appt_constraints_postSwitch": 2,
                        "policy_name": "Naive",
                        "tclose_overwrite": 1,
                        "tclose_days_offset_overwrite": 7,
                        "use_funded_or_actual_staffing": "actual",
                        "year_cons_availability_switch": 2019,
                        "cons_availability_postSwitch": "all",
                      },
                    }
                ),

            "GDP growth fHE growth case 3":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     "HealthSystem": {
                        "yearly_HR_scaling_mode": "GDP_growth_fHE_case3",
                        "year_mode_switch": 2019,
                        "mode_appt_constraints_postSwitch": 2,
                        "policy_name": "Naive",
                        "tclose_overwrite": 1,
                        "tclose_days_offset_overwrite": 7,
                        "use_funded_or_actual_staffing": "actual",
                        "year_cons_availability_switch": 2019,
                        "cons_availability_postSwitch": "all",
                      },
                    }
                ),

            "GDP growth fHE growth case 4":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     "HealthSystem": {
                        "yearly_HR_scaling_mode": "GDP_growth_fHE_case4",
                        "year_mode_switch": 2019,
                        "mode_appt_constraints_postSwitch": 2,
                        "policy_name": "Naive",
                        "tclose_overwrite": 1,
                        "tclose_days_offset_overwrite": 7,
                        "use_funded_or_actual_staffing": "actual",
                        "year_cons_availability_switch": 2019,
                        "cons_availability_postSwitch": "all",
                      },
                    }
                ),

            "GDP growth fHE growth case 6":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {
                     "HealthSystem": {
                        "yearly_HR_scaling_mode": "GDP_growth_fHE_case6",
                        "year_mode_switch": 2019,
                        "mode_appt_constraints_postSwitch": 2,
                        "policy_name": "Naive",
                        "tclose_overwrite": 1,
                        "tclose_days_offset_overwrite": 7,
                        "use_funded_or_actual_staffing": "actual",
                        "year_cons_availability_switch": 2019,
                        "cons_availability_postSwitch": "all",
                      },
                    }
                ),
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
