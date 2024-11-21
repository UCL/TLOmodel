""" The file contains all the definitions of scenarios used for schisto analyses """

from typing import Dict

from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios


class ScenarioDefinitions:

    @property
    def YEAR_OF_CHANGE_FOR_WASH(self) -> int:
        """Year in which WASH-related changes are made."""
        return 2025

    def set_parameters(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),  # <-- Parameters that have been the calibration targets
            {
                "Schisto": {
                    "single_district": False
                },
                "Demography": {
                    "equal_allocation_by_district": True,
                }
            },
        )

    def baseline(self) -> Dict:
        """ Return the Dict with values for the parameter changes that define the baseline scenario.
        The default settings are mda_coverage=0.7, target_group=SAC, mda_frequency=6 months
        """
        return mix_scenarios(
            get_parameters_for_status_quo(),  # <-- Parameters that have been the calibration targets
            # stop all future MDA activities
            {
                "Schisto": {
                    "mda_coverage": 0.0,  # default is 0.7
                    "scaleup_WASH": 0,  # although this is BOOL, python changes type when reading in from Excel
                }
            },
        )

    def scaleup_WASH(self) -> Dict:
        return {
            'Schisto': {
                'scaleup_WASH': 1,
                "scaleup_WASH_start_year": self.YEAR_OF_CHANGE_FOR_WASH,
            }
        }

    def expand_MDA_to_PSAC(self) -> Dict:
        return {
            'Schisto': {
                "mda_target_group": 'PSAC_SAC',
            }
        }

    def expand_MDA_to_All(self) -> Dict:
        return {
            'Schisto': {
                "mda_target_group": 'ALL',
            }
        }

    def high_coverage_MDA(self) -> Dict:
        return {
            'Schisto': {
                'mda_coverage': 0.8,
                "mda_frequency_months": 12,  # default is 6
            }
        }


