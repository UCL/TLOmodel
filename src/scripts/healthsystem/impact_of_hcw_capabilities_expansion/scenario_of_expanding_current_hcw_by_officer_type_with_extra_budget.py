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
        self.end_date = Date(2030, 1, 1)
        self.pop_size = 20_000  # todo: TBC
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 10  # todo: TBC

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
                [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)])  # todo: TBC

    def draw_parameters(self, draw_number, rng):
        return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""

        self.YEAR_OF_CHANGE = 2020  # This is the year to change run settings and to start hr expansion.

        self.scenarios = extra_budget_fracs

        return {
            self.scenarios.columns[i]:
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_expansion_by_officer_type': list(self.scenarios.iloc[:, i])
                    }
                    }
                ) for i in range(len(self.scenarios.columns))
        }

    def _baseline(self) -> Dict:
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'HealthSystem': {
                'mode_appt_constraints': 1,
                'mode_appt_constraints_postSwitch': 2,
                # "scale_to_effective_capabilities": True,  # todo: TBC; will this change the capabilities of 2019?
                "year_mode_switch": self.YEAR_OF_CHANGE,
                'cons_availability': 'default',
                'cons_availability_postSwitch': 'all',
                'year_cons_availability_switch': self.YEAR_OF_CHANGE,
                'yearly_HR_scaling_mode': 'no_scaling',
                'start_year_HR_expansion_by_officer_type': self.YEAR_OF_CHANGE,
                'end_year_HR_expansion_by_officer_type': self.end_date.year,
            }  # as to expand current hr and analyse the impact, we keep using 'actual' hr capabilities
            },
        )


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
