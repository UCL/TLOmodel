"""The file contains all the definitions of scenarios for the TLO-LCOA project."""
from typing import Dict

from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios


class ScenarioDefinitions:

    @property
    def YEAR_OF_SERVICE_AVAILABILITY_SWITCH(self) -> int:
        return 2026


    def baseline(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),  # <-- Parameters that have been the calibration targets

            {
                "HealthSystem": {
                    "cons_availability": 'default',
                    'year_cons_availability_switch': self.YEAR_OF_SERVICE_AVAILABILITY_SWITCH,
                    'cons_availability_postSwitch': 'all',

                    "mode_appt_constraints": 1,
                    "year_service_availability_switch": self.YEAR_OF_SERVICE_AVAILABILITY_SWITCH,

                    # allow historical HRH scaling to occur 2018-2024
                    # 'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_SERVICE_AVAILABILITY_SWITCH,
                    'yearly_HR_scaling_mode': 'historical_scaling',
                },

                "ImprovedHealthSystemAndCareSeekingScenarioSwitcher": {
                    'max_healthsystem_function': [False, True],  # <-- switch from False to True mid-way
                    'year_of_switch': self.YEAR_OF_SERVICE_AVAILABILITY_SWITCH,
                },

                "ImprovedHealthSystemAndCareSeekingScenarioSwitcher": {
                    'max_healthcare_seeking': [False, True],  # <-- switch from False to True mid-way
                    'year_of_switch': self.YEAR_OF_SERVICE_AVAILABILITY_SWITCH,
                }


            },
        )
