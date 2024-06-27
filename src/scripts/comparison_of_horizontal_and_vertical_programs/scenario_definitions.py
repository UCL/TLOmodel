"""The file contains all the definitions of scenarios used the Horizontal and Vertical Program Impact Analyses"""
from typing import Dict

from tlo.analysis.utils import mix_scenarios, get_parameters_for_status_quo


class ScenarioDefinitions:

    @property
    def YEAR_OF_CHANGE(self) -> int:
        """Year at which all changes are made."""
        return 2019  # <-- baseline year of Human Resources for Health is 2018, and this is consistent with calibration
        #                  during 2015-2019 period.

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
                    "year_mode_switch": self.YEAR_OF_CHANGE,

                    # Baseline scenario is with absence of HCW
                    'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE,
                    'HR_scaling_by_level_and_officer_type_mode': 'with_absence',
                    # todo <-- Do we want the first part of the run be with_abscence too...? (Although that will mean
                    #          that there is actually greater capacity if we do the rescaling)

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
                'year_of_switch': self.YEAR_OF_CHANGE
            },
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE,
                'cons_availability_postSwitch': 'all',
                'yearly_HR_scaling_mode': 'GDP_growth_fHE_case5',
                'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE,
                'HR_scaling_by_level_and_officer_type_mode': 'no_absence_&_x2_fac0+1',
            }
        }

    def _hiv_scaleup(self) -> Dict:
        """The parameters for the scale-up of the HIV program"""
        return {
            "Hiv": {
                'do_scaleup': True,
                'scaleup_start_year': self.YEAR_OF_CHANGE,  # todo what is the natural year of scale-up? Should this
                #                                                   be the same as the when the HSS changes happen?
            }
        }

    def _tb_scaleup(self) -> Dict:
        """The parameters for the scale-up of the TB program"""
        return {
            "Tb": {
                'do_scaleup': True,
                'scaleup_start_year': self.YEAR_OF_CHANGE,  # todo what is the natural year of scale-up? Should this
                #                                                   be the same as the when the HSS changes happen?
            }
        }

    def _malaria_scaleup(self) -> Dict:
        """The parameters for the scale-up of the Malaria program"""
        return {
            'Malaria': {
                'do_scaleup': True,
                'scaleup_start_year': self.YEAR_OF_CHANGE,  # todo what is the natural year of scale-up? Should this
                #                                                   be the same as the when the HSS changes happen?
            }
        }
