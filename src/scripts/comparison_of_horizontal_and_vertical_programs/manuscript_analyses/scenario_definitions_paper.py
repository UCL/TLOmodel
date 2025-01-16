"""The file contains all the definitions of scenarios used the Horizontal and Vertical Program Impact Analyses"""
from typing import Dict

from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios


class ScenarioDefinitions:

    @property
    def YEAR_OF_CHANGE_FOR_HSS(self) -> int:
        """Year in which Health Systems Strengthening changes are made."""
        return 2011 # todo 2019  # baseline year of Human Resources for Health is 2018, and this is consistent with calibration
        #                  during 2015-2019 period.

    @property
    def YEAR_OF_CHANGE_FOR_HTM(self) -> int:
        """Year in which HIV, TB, Malaria scale-up changes are made."""
        return 2011 # todo 2024

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

    # HRH scenarios
    def increase_capacity_at_primary_care(self) -> Dict:
        return {
            'HealthSystem': {
                'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE_FOR_HSS,
                'HR_scaling_by_level_and_officer_type_mode': 'x1.338_fac0&1',
                # increase all cadres at level 0 and 1 by ((1+0.06)^5)-1 = 0.338, 5yrs of 6% growth
            }
        }

    def increase_capacity_of_dcsa(self) -> Dict:
        return {
            'HealthSystem': {
                'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE_FOR_HSS,
                'HR_scaling_by_level_and_officer_type_mode': 'x1.338_dcsa',
                # increase DCSA at level 0 by ((1+0.06)^5)-1 = 0.338, 5yrs of 6% growth
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

    def hrh_using_historical_scaling(self) -> Dict:
        return {
            'HealthSystem': {
                'yearly_HR_scaling_mode': 'historical_scaling_maintained',
                # This uses historical trends in HRH scale-up to 2023, then uses 2023 values fixed to 2030
            }
        }

    def accelerated_hrh_expansion(self) -> Dict:
        return {
            'HealthSystem': {
                'yearly_HR_scaling_mode': 'historical_scaling_accelerated',
                # This uses historical trends in HRH scale-up to 2023, then 6% constant scaling to 2030
            }
        }

    def moderate_hrh_expansion(self) -> Dict:
        return {
            'HealthSystem': {
                'yearly_HR_scaling_mode': 'historical_scaling_moderate',
                # This uses historical trends in HRH scale-up to 2023, then 1% constant scaling to 2030
            }
        }

    # Behavioural scenarios
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

    # Supply chains scenarios
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

    def cons_at_75th_percentile(self) -> Dict:
        return {
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE_FOR_HSS,
                'cons_availability_postSwitch': 'scenario6',
            }
        }

    def cons_at_90th_percentile(self) -> Dict:
        return {
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE_FOR_HSS,
                'cons_availability_postSwitch': 'scenario7',
            }
        }

    def cons_at_HIV_availability(self) -> Dict:
        return {
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE_FOR_HSS,
                'cons_availability_postSwitch': 'scenario10',
            }
        }

    def cons_at_EPI_availability(self) -> Dict:
        return {
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE_FOR_HSS,
                'cons_availability_postSwitch': 'scenario11',
            }
        }

    def all_consumables_available(self) -> Dict:
        return {
            'HealthSystem': {
                'year_cons_availability_switch': self.YEAR_OF_CHANGE_FOR_HSS,
                'cons_availability_postSwitch': 'all',
            }
        }

    def full_hss_package(self) -> Dict:
        """The parameters for the Full Health System Strengthening Package"""
        return mix_scenarios(
            self.increase_capacity_at_primary_care(),
            self.accelerated_hrh_expansion(),
            self.perfect_healthcare_seeking(),
            self.all_consumables_available(),
        )

    def hss_package_default_HSB(self) -> Dict:
        """The parameters for the Health System Strengthening Package WITHOUT perfect HSB"""
        return mix_scenarios(
            self.increase_capacity_at_primary_care(),
            self.accelerated_hrh_expansion(),
            self.cons_at_75th_percentile(),
        )

    def hss_package_realistic(self) -> Dict:
        """The parameters for the Health System Strengthening Package with 75th percentile cons"""
        return mix_scenarios(
            self.increase_capacity_at_primary_care(),
            self.accelerated_hrh_expansion(),
            self.perfect_healthcare_seeking(),
            self.cons_at_75th_percentile(),
        )

    # HTM scenarios
    def hiv_scaleup(self) -> Dict:
        """The parameters for the scale-up of the HIV program"""
        return {
            "Hiv": {
                'type_of_scaleup': 'target',
                'scaleup_start_year': self.YEAR_OF_CHANGE_FOR_HTM,
            }
        }

    def tb_scaleup(self) -> Dict:
        """The parameters for the scale-up of the TB program"""
        return {
            "Tb": {
                'type_of_scaleup': 'target',
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
