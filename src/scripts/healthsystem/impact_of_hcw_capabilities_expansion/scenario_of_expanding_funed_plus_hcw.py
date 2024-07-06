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
        self.end_date = Date(2020, 1, 1)
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

        return {
            "Establishment HCW":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'default'
                    }
                    }
                ),

            "Establishment HCW Expansion C1":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c1'
                    }
                    }
                ),

            "Establishment HCW Expansion C2":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c2'
                    }
                    }
                ),

            "Establishment HCW Expansion C3":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c3'
                    }
                    }
                ),

            "Establishment HCW Expansion P1":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_p1'
                    }
                    }
                ),

            "Establishment HCW Expansion P2":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_p2'
                    }
                    }
                ),

            "Establishment HCW Expansion P3":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_p3'
                    }
                    }
                ),

            "Establishment HCW Expansion C1P1":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c1p1'
                    }
                    }
                ),

            "Establishment HCW Expansion C2P1":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c2p1'
                    }
                    }
                ),

            "Establishment HCW Expansion C3P1":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c3p1'
                    }
                    }
                ),

            "Establishment HCW Expansion C1P2":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c1p2'
                    }
                    }
                ),

            "Establishment HCW Expansion C2P2":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c2p2'
                    }
                    }
                ),

            "Establishment HCW Expansion C3P2":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c3p2'
                    }
                    }
                ),

            "Establishment HCW Expansion C1P3":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c1p3'
                    }
                    }
                ),

            "Establishment HCW Expansion C2P3":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c2p3'
                    }
                    }
                ),

            "Establishment HCW Expansion C3P3":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {
                        'equip_availability': 'default',  # if not specify here, the value will be 'all'
                        'use_funded_or_actual_staffing': 'funded_plus',
                        'yearly_HR_scaling_mode': 'no_scaling',
                        'mode_appt_constraints': 2,
                        'HR_scaling_by_level_and_officer_type_mode': 'expand_funded_c3p3'
                    }
                    }
                ),
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
