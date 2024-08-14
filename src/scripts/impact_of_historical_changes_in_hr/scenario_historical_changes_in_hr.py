"""This Scenario file run the model under different assumptions for the historical changes in Human Resources for Health

Run on the batch system using:
```
tlo batch-submit src/scripts/impact_of_historical_changes_in_hr/scenario_historical_changes_in_hr.py
```

"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import mix_scenarios, get_parameters_for_status_quo
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class HistoricalChangesInHRH(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2031, 1, 1)  # <-- End at the end of year 2030
        self.pop_size = 10_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'historical_changes_in_hr',
            'directory': Path('./outputs'),
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.WARNING,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""

        return {
            "Actual (Scale-up)":
                mix_scenarios(
                    self._common_baseline(),
                    {
                        "HealthSystem": {
                            # SCALE-UP IN HRH
                            'yearly_HR_scaling_mode': 'historical_scaling',
                            # Scale-up pattern defined from examining the data
                        }
                    }
                ),

            "Counterfactual (No Scale-up)":
                mix_scenarios(
                    self._common_baseline(),
                    {
                        "HealthSystem": {
                            # NO CHANGE IN HRH EVER
                            'yearly_HR_scaling_mode': 'no_scaling',
                        }
                    }
                ),
        }

    def _common_baseline(self) -> Dict:
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "HealthSystem": {
                    "mode_appt_constraints": 1,                 # <-- Mode 1 prior to change to preserve calibration
                    "mode_appt_constraints_postSwitch": 2,      # <-- Mode 2 post-change to show effects of HRH
                    "scale_to_effective_capabilities": True,    # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                    "year_mode_switch": 2020,    # <-- transition happens at start of 2020 when HRH starts to grow

                    # Normalize the behaviour of Mode 2
                    "policy_name": "Naive",
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                }
            },
        )


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
