"""This Scenario file run the model under different assumptions for the HealthSystem and Vertical Program Scale-up

Run on the batch system using:
```
tlo batch-submit
 src/scripts/dcea_of_horizontal_and_vertical_programs/dcea_scenario_vertical_programs_with_and_without_hss.py
```

"""

from pathlib import Path
from typing import Dict

from scripts.dcea_of_horizontal_and_vertical_programs.dcea_scenario_definitions import (
    ScenarioDefinitions,
)
from tlo import Date, logging
from tlo.analysis.utils import mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class HTMWithAndWithoutHSS(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2036, 1, 1)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'htm_with_and_without_hss',
            'directory': Path('./outputs'),
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.enhanced_lifestyle': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.WARNING,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.malaria': logging.INFO,
            }
        }

    def modules(self):
        return (
            fullmodel()
            + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher()]
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        # Load helper class containing the definitions of the elements of all the scenarios
        scenario_definitions = ScenarioDefinitions()

        return {
            "Baseline":
                scenario_definitions.baseline(),

            # # - - - FULL PACKAGE OF HEALTH SYSTEM STRENGTHENING - - -
            # "FULL HSS PACKAGE":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hss_package(),
            #     ),
            #
            # # - - - FULL PACKAGE OF HEALTH SYSTEM STRENGTHENING, NORMAL HEALTHCARE SEEKING - - -
            # "FULL HSS PACKAGE Norm HCS":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hss_package_normal_HCS(),
            #     ),

            # - - - Realistic HEALTH SYSTEM STRENGTHENING - - -
            "Realistic HSS":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.hss_package_realistic(),
                ),

            # **************************************************
            # VERTICAL PROGRAMS WITH AND WITHOUT THE HSS PACKAGE
            # **************************************************

            # - - - HIV SCALE-UP WITHOUT HSS PACKAGE - - -
            "HIV Programs Scale-up Without HSS":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.hiv_scaleup(),
                ),
            # # - - - HIV SCALE-UP *WITH* HSS PACKAGE - - -
            # "HIV Programs Scale-up WITH HSS PACKAGE":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.hss_package(),
            #     ),
            # # - - - HIV SCALE-UP *WITH* HSS PACKAGE *AND* NORMAL HEALTHCARE SEEKING - - -
            # "HIV Programs Scale-up WITH HSS PACKAGE Norm HCS":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.hss_package_normal_HCS(),
            #     ),
            # - - - HIV SCALE-UP *WITH* REALISTIC HSS PACKAGE - - -
            "HIV Programs Scale-up With Realistic HSS":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.hiv_scaleup(),
                    scenario_definitions.hss_package_realistic(),
                ),

            # - - - TB SCALE-UP WITHOUT HSS PACKAGE - - -
            "TB Programs Scale-up Without HSS":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.tb_scaleup(),
                ),
            # # - - - TB SCALE-UP *WITH* HSS PACKAGE - - -
            # "TB Programs Scale-up WITH HSS PACKAGE":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.hss_package(),
            #     ),
            # # - - - TB SCALE-UP *WITH* HSS PACKAGE *AND* NORMAL HEALTHCARE SEEKING - - -
            # "TB Programs Scale-up WITH HSS PACKAGE Norm HCS":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.hss_package_normal_HCS(),
            #     ),
            # - - - TB SCALE-UP *WITH* REALISTIC HSS PACKAGE - - -
            "TB Programs Scale-up With Realistic HSS":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.tb_scaleup(),
                    scenario_definitions.hss_package_realistic(),
                ),

            # - - - MALARIA SCALE-UP WITHOUT HSS PACKAGE - - -
            "Malaria Programs Scale-up Without HSS":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.malaria_scaleup(),
                ),
            # # - - - MALARIA SCALE-UP *WITH* HSS PACKAGE - - -
            # "Malaria Programs Scale-up WITH HSS PACKAGE":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hss_package(),
            #     ),
            # # - - - MALARIA SCALE-UP *WITH* HSS PACKAGE *AND* NORMAL HEALTHCARE SEEKING - - -
            # "Malaria Programs Scale-up WITH HSS PACKAGE Norm HCS":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hss_package_normal_HCS(),
            #     ),
            # - - - MALARIA SCALE-UP *WITH* REALISTIC HSS PACKAGE - - -
            "Malaria Programs Scale-up With Realistic HSS":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.malaria_scaleup(),
                    scenario_definitions.hss_package_realistic(),
                ),

            # - - - HIV & TB & MALARIA SCALE-UP WITHOUT HSS PACKAGE - - -
            "HIV/Tb/Malaria Programs Scale-up Without HSS":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.hiv_scaleup(),
                    scenario_definitions.tb_scaleup(),
                    scenario_definitions.malaria_scaleup(),
                ),
            # # - - - HIV & TB & MALARIA SCALE-UP *WITH* HSS PACKAGE - - -
            # "HIV/Tb/Malaria Programs Scale-up WITH HSS PACKAGE":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hss_package(),
            #     ),
            # # - - - HIV & TB & MALARIA SCALE-UP *WITH* HSS PACKAGE *AND* NORMAL HEALTHCARE SEEKING - - -
            # "HIV/Tb/Malaria Programs Scale-up WITH HSS PACKAGE Norm HCS":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hss_package_normal_HCS(),
            #     ),
            # - - - HIV & TB & MALARIA SCALE-UP *WITH* REALISTIC HSS PACKAGE - - -
            "HTM Programs Scale-up With Realistic HSS":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.hiv_scaleup(),
                    scenario_definitions.tb_scaleup(),
                    scenario_definitions.malaria_scaleup(),
                    scenario_definitions.hss_package_realistic(),
                ),
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
