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
        return 2019  # todo <-- what is the natural year of scale-up? Should this be the same as the when the HSS
        #               changes happen?

    def _baseline(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "HealthSystem": {
                    "mode_appt_constraints": 1,  # <-- Mode 1 prior to change to preserve calibration
                    "mode_appt_constraints_postSwitch": 2,  # <-- Mode 2 post-change to show effects of HRH
                    "scale_to_effective_capabilities": True,
                    # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                    "year_mode_switch": self.YEAR_OF_CHANGE_FOR_HSS,

                    # Baseline scenario is with absence of HCW
                    'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE_FOR_HSS,
                    'HR_scaling_by_level_and_officer_type_mode': 'default',

                    # Normalize the behaviour of Mode 2
                    "policy_name": "Naive",
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                }
            },
        )

    def _hss_package(self) -> Dict:
        """The parameters for the Health System Strengthening Package"""
        return {
            'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],  # <-- switch from False to True mid-way
                'max_healthcare_seeking': [False, True],  # <-- switch from False to True mid-way
                'year_of_switch': self.YEAR_OF_CHANGE_FOR_HSS
            },
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE_FOR_HSS,
                'cons_availability_postSwitch': 'all',
                'yearly_HR_scaling_mode': 'GDP_growth_fHE_case5',
                'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE_FOR_HSS,
                'HR_scaling_by_level_and_officer_type_mode': 'x2_fac0&1',
            }
        }

    def _hiv_scaleup(self) -> Dict:
        """The parameters for the scale-up of the HIV program"""
        return {
            "Hiv": {
                'do_scaleup': True,
                'scaleup_start_year': self.YEAR_OF_CHANGE_FOR_HTM,
            }
        }

    def _tb_scaleup(self) -> Dict:
        """The parameters for the scale-up of the TB program"""
        return {
            "Tb": {
                'do_scaleup': True,
                'scaleup_start_year': self.YEAR_OF_CHANGE_FOR_HTM,
            }
        }

    def _malaria_scaleup(self) -> Dict:
        """The parameters for the scale-up of the Malaria program"""
        return {
            'Malaria': {
                'do_scaleup': True,
                'scaleup_start_year': self.YEAR_OF_CHANGE_FOR_HTM,
            }
        }
