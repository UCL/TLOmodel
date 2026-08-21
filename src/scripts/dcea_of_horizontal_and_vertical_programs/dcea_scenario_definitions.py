"""The file contains all the definitions of scenarios used the Horizontal and Vertical Program Impact Analyses"""
from typing import Dict

from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios


class ScenarioDefinitions:

    @property
    def YEAR_OF_CHANGE_FOR_HSS(self) -> int:
        """Year in which Health Systems Strengthening changes are made."""
        return 2019  # <-- baseline year of Human Resources for Health is 2018, and this is consistent with calibration
        #                  during 2015-2019 period.

    @property
    def YEAR_OF_CHANGE_FOR_HTM(self) -> int:
        """Year in which HIV, TB, Malaria scale-up changes are made."""
        return 2024

    def baseline(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),  # <-- Parameters that have been the calibration targets

            # Set up the HealthSystem to transition from Mode 1 -> Mode 2, with rescaling when there are HSS changes
            {
                "HealthSystem": {
                    "cons_availability": 'default',
                    "mode_appt_constraints": 1,  # <-- Mode 1 prior to change to preserve calibration
                    "mode_appt_constraints_postSwitch": 2,  # <-- Mode 2 post-change to show effects of HRH
                    "scale_to_effective_capabilities": True,
                    # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                    "year_mode_switch": self.YEAR_OF_CHANGE_FOR_HSS,

                    # Normalize the behaviour of Mode 2
                    "policy_name": "HTM",
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,

                    # allow historical HRH scaling to occur 2018-2024
                    # 'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE_FOR_HSS,
                    'yearly_HR_scaling_mode': 'historical_scaling',
                }
            },
        )

    def hrh_using_historical_scaling(self) -> Dict:
        return {
            'HealthSystem': {
                'yearly_HR_scaling_mode': 'historical_scaling_maintained',
                # This uses historical trends in HRH scale-up to 2023, then uses 2023 values fixed to 2030
            }
        }

    def cons_at_75th_percentile(self) -> Dict:
        return {
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE_FOR_HSS,
                'cons_availability_postSwitch': 'scenario6',
            }
        }

    def hss_package_realistic(self) -> Dict:
        """The parameters for the Realistic Health System Strengthening Package with historical HR scale and
        75th percentile cons"""
        return mix_scenarios(
            self.hrh_using_historical_scaling(),
            self.cons_at_75th_percentile(),
        )

    def hiv_scaleup(self) -> Dict:
        """The parameters for the scale-up of the HIV program"""
        return {
            "Hiv": {
                'type_of_scaleup': 'target', #Change to target from max
                'scaleup_start_year': self.YEAR_OF_CHANGE_FOR_HTM,
            }
        }

    def tb_scaleup(self) -> Dict:
        """The parameters for the scale-up of the TB program"""
        return {
            "Tb": {
                'type_of_scaleup': 'target', #Change to target from max
                'scaleup_start_year': self.YEAR_OF_CHANGE_FOR_HTM,
            }
        }

    def malaria_scaleup(self) -> Dict:
        """The parameters for the scale-up of the Malaria program"""
        return {
            'Malaria': {
                'type_of_scaleup': 'target',
                'scaleup_start_year': self.YEAR_OF_CHANGE_FOR_HTM,
            }
        }
