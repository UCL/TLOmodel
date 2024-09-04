"""The file contains all the definitions of scenarios used the Horizontal and Vertical Program Impact Analyses"""
from typing import Dict

from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios


class ScenarioDefinitions:

    @property
    def YEAR_OF_CHANGE_FOR_HSS(self) -> int:
        """Year in which Health Systems Strengthening changes are made."""
        return 2024  # <-- baseline year of Human Resources for Health is 2018, and this is consistent with calibration
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
                    "mode_appt_constraints": 1,  # <-- Mode 1 prior to change to preserve calibration
                    "mode_appt_constraints_postSwitch": 2,  # <-- Mode 2 post-change to show effects of HRH
                    "scale_to_effective_capabilities": True,
                    # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                    "year_mode_switch": self.YEAR_OF_CHANGE_FOR_HSS,

                    # Normalize the behaviour of Mode 2
                    "policy_name": "HTM",  # use priority ranking 2 for HTM treatments only
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                }
            },
        )

    def baseline_mode1(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario when using Mode 1.
        """
        return get_parameters_for_status_quo()  # <-- Parameters that have been the calibration targets

    def double_capacity_at_primary_care(self) -> Dict:
        return {
            'HealthSystem': {
                'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE_FOR_HSS,
                'HR_scaling_by_level_and_officer_type_mode': 'x2_fac0&1',
            }
        }

    def hrh_at_pop_growth(self) -> Dict:
        return {
            'HealthSystem': {
                'yearly_HR_scaling_mode': 'scaling_by_population_growth',
                # This is in-line with population growth _after 2018_ (baseline year for HRH)
            }
        }

    def hrh_at_gdp_growth(self) -> Dict:
        return {
            'HealthSystem': {
                'yearly_HR_scaling_mode': 'GDP_growth',
                # This is GDP growth after 2018 (baseline year for HRH)
            }
        }

    def hrh_above_gdp_growth(self) -> Dict:
        return {
            'HealthSystem': {
                'yearly_HR_scaling_mode': 'GDP_growth_fHE_case5',
                # This is above-GDP growth after 2018 (baseline year for HRH)
            }
        }

    def perfect_clinical_practices(self) -> Dict:
        return {
            'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],  # <-- switch from False to True mid-way
                'year_of_switch': self.YEAR_OF_CHANGE_FOR_HSS,
            }
        }

    def perfect_healthcare_seeking(self) -> Dict:
        return {
            'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthcare_seeking': [False, True],  # <-- switch from False to True mid-way
                'year_of_switch': self.YEAR_OF_CHANGE_FOR_HSS,
            }
        }

    def vital_items_available(self) -> Dict:
        return {
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE_FOR_HSS,
                'cons_availability_postSwitch': 'all_vital_available',
            }
        }

    def medicines_available(self) -> Dict:
        return {
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE_FOR_HSS,
                'cons_availability_postSwitch': 'all_medicines_available',
            }
        }

    def all_consumables_available(self) -> Dict:
        return {
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE_FOR_HSS,
                'cons_availability_postSwitch': 'all',
            }
        }

    def hss_package(self) -> Dict:
        """The parameters for the Health System Strengthening Package"""
        return mix_scenarios(
            self.double_capacity_at_primary_care(),  #  }
            self.hrh_above_gdp_growth(),             #  } <-- confirmed that these two do build on one another under
            # mode 2 rescaling: see `test_scaling_up_HRH_using_yearly_scaling_and_scaling_by_level_together`.
            self.perfect_clinical_practices(),
            self.perfect_healthcare_seeking(),
            self.all_consumables_available(),
        )

    def hiv_scaleup(self) -> Dict:
        """The parameters for the scale-up of the HIV program"""
        return {
            "Hiv": {
                'type_of_scaleup': 'target',  # <--- todo: using MAXIMUM SCALE-UP as an experiment
                'scaleup_start_year': self.YEAR_OF_CHANGE_FOR_HTM,
            }
        }

    def tb_scaleup(self) -> Dict:
        """The parameters for the scale-up of the TB program"""
        return {
            "Tb": {
                'type_of_scaleup': 'target',  # <--- todo: using MAXIMUM SCALE-UP as an experiment
                'scaleup_start_year': self.YEAR_OF_CHANGE_FOR_HTM,
            }
        }

    def malaria_scaleup(self) -> Dict:
        """The parameters for the scale-up of the Malaria program"""
        return {
            'Malaria': {
                'type_of_scaleup': 'target',  # <--- todo: using MAXIMUM SCALE-UP as an experiment
                'scaleup_start_year': self.YEAR_OF_CHANGE_FOR_HTM,
            }
        }
