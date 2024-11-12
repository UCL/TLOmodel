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
        self.seed = 0  # change seed to 1 if to do another 5 runs per draw
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

        self.YEAR_OF_MODE_CHANGE = 2020
        # HCW capabilities from data source are for year 2019,
        # and we want to rescale to effective capabilities in the end of 2019 considering model calibration
        self.YEAR_OF_HRH_EXPANSION = 2025
        # The start year to expand HRH by cadre given the extra budget, which is after the historical HRH scaling

        self.scenarios = extra_budget_fracs.drop(columns='s_2')
        # Test historical scaling changes; do not run 'gap' scenario that's based on "no historical scaling"

        # self.scenarios = extra_budget_fracs['s_0'].to_frame()
        # Run no extra budget allocation scenarios first to get never ran services and 'gap' allocation strategies

        # Baseline settings for change
        self.cons_availability = ['all', 'default']
        self.hr_budget = [0.042, 0.058, 0.026]
        self.hs_function = [[False, False], [False, True]]

        self.baselines = {
            'baseline': self._baseline_of_baseline(),  # test historical scaling changes first
            # 'default_cons': self._baseline_default_cons(),
            # 'more_budget': self._baseline_more_budget(),  # turn off when run baseline scenarios with no expansion
            # 'less_budget': self._baseline_less_budget(),
            # 'max_hs_function': self._baseline_max_hs_function(),
        }

        return {
            b + ' ' + self.scenarios.columns[i]:
                mix_scenarios(
                    self.baselines[b],
                    {'HealthSystem': {
                        'HR_expansion_by_officer_type': self.scenarios.iloc[:, i].to_dict()
                    }
                    }
                ) for b in self.baselines.keys() for i in range(len(self.scenarios.columns))
        }

    def _baseline_of_baseline(self) -> Dict:
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                'HealthSystem': {
                    'mode_appt_constraints': 1,
                    'mode_appt_constraints_postSwitch': 2,
                    "scale_to_effective_capabilities": True,
                    # This happens in the year before mode change, as the model calibration is done by that year
                    "year_mode_switch": self.YEAR_OF_MODE_CHANGE,
                    'cons_availability': 'default',
                    'cons_availability_postSwitch': self.cons_availability[0],
                    'year_cons_availability_switch': self.YEAR_OF_HRH_EXPANSION,
                    'HR_budget_growth_rate': self.hr_budget[0],
                    'yearly_HR_scaling_mode': 'historical_scaling',  # for 5 years of 2020-2024; source data year 2019
                    'start_year_HR_expansion_by_officer_type': self.YEAR_OF_HRH_EXPANSION,
                    'end_year_HR_expansion_by_officer_type': self.end_date.year,
                    "policy_name": 'Naive',
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                },
                'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                    'max_healthcare_seeking': [False, False],
                    'max_healthsystem_function': self.hs_function[0],
                    'year_of_switch': self.YEAR_OF_HRH_EXPANSION,
                }
            },
        )

    def _baseline_default_cons(self) -> Dict:
        return mix_scenarios(
            self._baseline_of_baseline(),
            {
                'HealthSystem': {
                    'cons_availability_postSwitch': self.cons_availability[1],
                },
            },
        )

    def _baseline_more_budget(self) -> Dict:
        return mix_scenarios(
            self._baseline_of_baseline(),
            {
                'HealthSystem': {
                    'HR_budget_growth_rate': self.hr_budget[1],
                },
            },
        )

    def _baseline_less_budget(self) -> Dict:
        return mix_scenarios(
            self._baseline_of_baseline(),
            {
                'HealthSystem': {
                    'HR_budget_growth_rate': self.hr_budget[2],
                },
            },
        )

    def _baseline_max_hs_function(self) -> Dict:
        return mix_scenarios(
            self._baseline_of_baseline(),
            {
                'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                    'max_healthsystem_function': self.hs_function[1],
                }
            },
        )


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
