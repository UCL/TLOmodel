"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for analysis of impact of expanding funded hcw, assuming all other setting as default.

Run on the batch system using:
```
tlo batch-submit src/scripts/healthsystem/impact_of_hcw_capabilities_expansion/scenario_of_expanding_funed_plus_hcw.py
```

or locally using:
```
tlo scenario-run src/scripts/healthsystem/impact_of_hcw_capabilities_expansion/scenario_of_expanding_funed_plus_hcw.py
```
"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2030, 1, 1)
        self.pop_size = 20_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 10

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
        return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:  # todo: create many scenarios of expanding HCW (C, NM, P)
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""

        self.YEAR_OF_CHANGE = 2020  # This is the year to change HR scaling mode.
        # Year 2030 is when the Establishment HCW will be met as estimated by Berman 2022.
        # But it can be 2020, or 2019, to reduce running time (2010-2030 instead of 2010-2040).

        return {
            "Establishment HCW":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'default',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C1":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c1',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C2":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c2',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C3":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c3',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion P1":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_p1',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion P2":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_p2',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion P3":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_p3',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C1P1":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c1p1',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C2P1":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c2p1',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C3P1":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c3p1',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C1P2":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c1p2',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C2P2":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c2p2',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C3P2":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c3p2',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C1P3":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c1p3',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C2P3":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c2p3',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),

            "Establishment HCW Expansion C3P3":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c3p3',
                        'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE
                    }
                    }
                ),
        }

    def _baseline(self) -> Dict:
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'HealthSystem': {
                'use_funded_or_actual_staffing': 'actual',
                'use_funded_or_actual_staffing_postSwitch': 'funded_plus',
                'year_use_funded_or_actual_staffing_switch': self.YEAR_OF_CHANGE,
                'mode_appt_constraints': 1,
                'mode_appt_constraints_postSwitch': 2,
                "year_mode_switch": self.YEAR_OF_CHANGE,
                'cons_availability': 'default',
                'cons_availability_postSwitch': 'all',
                'year_cons_availability_switch': self.YEAR_OF_CHANGE,
                'yearly_HR_scaling_mode': 'no_scaling',

            }
            },
        )


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
