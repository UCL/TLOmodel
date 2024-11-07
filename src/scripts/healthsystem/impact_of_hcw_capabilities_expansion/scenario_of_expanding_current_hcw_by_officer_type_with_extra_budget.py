"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for analysis of impact of expanding funded hcw, assuming all other setting as default.

Run on the batch system using:
```
tlo batch-submit src/scripts/healthsystem/impact_of_hcw_capabilities_expansion/scenario_of_expanding_current_hcw_by_officer_type_with_extra_budget.py
```

or locally using:
```
tlo scenario-run src/scripts/healthsystem/impact_of_hcw_capabilities_expansion/scenario_of_expanding_current_hcw_by_officer_type_with_extra_budget.py
```
"""

from pathlib import Path
from typing import Dict

from scripts.healthsystem.impact_of_hcw_capabilities_expansion.prepare_minute_salary_and_extra_budget_frac_data import (
    extra_budget_fracs,
)
from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class HRHExpansionByCadreWithExtraBudget(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2035, 1, 1)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'scenario_run_for_hcw_expansion_analysis',
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
        return (fullmodel(resourcefilepath=self.resources) +
                [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)])

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""

        self.YEAR_OF_MODE_CHANGE = 2020  # HCW capabilities data are for year of 2019, before the Covid-19 pandemic

        self.scenarios = extra_budget_fracs['s_0'].to_frame()
        # run no extra budget allocation scenarios first to get the never ran services and 'gap' allocation strategies

        return {
            self.scenarios.columns[i]:
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_expansion_by_officer_type': self.scenarios.iloc[:, i].to_dict()
                    }
                    }
                ) for i in range(len(self.scenarios.columns))  # run 33 scenarios
        }

    def _baseline(self) -> Dict:
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'HealthSystem': {
                'mode_appt_constraints': 1,
                'mode_appt_constraints_postSwitch': 2,
                "scale_to_effective_capabilities": True,
                # This happens in the year before mode change, as the model calibration is done by that year
                "year_mode_switch": self.YEAR_OF_MODE_CHANGE,
                'cons_availability': 'default',
                'cons_availability_postSwitch': 'all',
                'year_cons_availability_switch': self.YEAR_OF_MODE_CHANGE,  # todo: or the HRH expansion start year?
                'yearly_HR_scaling_mode': 'historical_scaling',  # for 5 years of 2020-2024; source data year 2019
                'start_year_HR_expansion_by_officer_type': 2025,  # start expansion from 2025
                'end_year_HR_expansion_by_officer_type': self.end_date.year,
                "policy_name": "Naive",
                "tclose_overwrite": 1,
                "tclose_days_offset_overwrite": 7,
            }
            },
        )


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
