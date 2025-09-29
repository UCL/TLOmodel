"""This Scenario file run the model under different assumptions for the HealthSystem and Vertical Program Scale-up

check scenarios are generated correctly:
tlo scenario-run --draw-only
 src/scripts/comparison_of_horizontal_and_vertical_programs/manuscript_analyses/scenario_hss_htm_paper.py

to create locally available log files, have to do this command run-by-run
tlo parse-log outputs/htm_and_hss_runs-2025-01-16T095335Z/0/0

Run on the batch system using:

tlo batch-submit --more-memory
 src/scripts/comparison_of_horizontal_and_vertical_programs/manuscript_analyses/scenario_hss_htm_paper.py


"""

from pathlib import Path
from typing import Dict

from scripts.comparison_of_horizontal_and_vertical_programs.manuscript_analyses.scenario_definitions_paper import (
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
        self.runs_per_draw = 1  # todo reset

    def log_configuration(self):
        return {
            'filename': 'htm_and_hss_runs',
            'directory': Path('./outputs'),
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
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
            fullmodel(resourcefilepath=self.resources)
            + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)]
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        # Load helper class containing the definitions of the elements of all the scenarios
        scenario_definitions = ScenarioDefinitions()

        return {
            # "Baseline":
            #     scenario_definitions.baseline(),

            # ***************************
            # HEALTH SYSTEM STRENGTHENING
            # ***************************

            # - - - Human Resources for Health - - -

            # "HRH Scale-up (1%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.moderate_hrh_expansion(),
            #     ),
            #
            # "HRH Scale-up (4%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hrh_using_historical_scaling(),
            #     ),
            #
            # "HRH Scale-up (6%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.accelerated_hrh_expansion(),
            #     ),

            # - - - Supply Chains - - -

            "Consumables Increased to 75th Percentile":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.cons_at_75th_percentile(),
                ),
            #
            # "Consumables Available at HIV levels":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.cons_at_HIV_availability(),
            #     ),
            #
            # "Consumables Available at EPI levels":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.cons_at_EPI_availability(),
            #     ),
            #
            # "HSS Expansion Package":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hss_package_realistic(),
            #     ),

            # **************************************************
            # VERTICAL PROGRAMS WITH AND WITHOUT THE HSS PACKAGE
            # **************************************************

            # # - - - HIV SCALE-UP WITHOUT HSS - - -
            # "HIV Program Scale-up Without HSS Expansion":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #     ),
            #
            # # - - - HIV SCALE-UP MODERATE HRH HSS PACKAGE- - -
            # "HIV Program Scale-up With HRH Scale-up (1%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.moderate_hrh_expansion(),
            #     ),
            #
            # # - - - HIV SCALE-UP HISTORICAL HRH HSS PACKAGE- - -
            # "HIV Program Scale-up With HRH Scale-up (4%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.hrh_using_historical_scaling(),
            #     ),
            #
            # # - - - HIV SCALE-UP ACCELERATED HRH HSS PACKAGE- - -
            # "HIV Program Scale-up With HRH Scale-up (6%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.accelerated_hrh_expansion(),
            #     ),
            #
            # # - - - HIV SCALE-UP CONS 75th PERCENTILE- - -
            # "HIV Program Scale-up With Consumables at 75th Percentile":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.cons_at_75th_percentile(),
            #     ),
            #
            # # - - - HIV SCALE-UP CONS HIV LEVEL- - -
            # "HIV Program Scale-up With Consumables at HIV levels":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.cons_at_HIV_availability(),
            #     ),
            #
            # # - - - HIV SCALE-UP CONS EPI LEVEL- - -
            # "HIV Program Scale-up With Consumables at EPI levels":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.cons_at_EPI_availability(),
            #     ),
            #
            # # - - - HIV SCALE-UP *WITH* HSS PACKAGE- - -
            # "HIV Programs Scale-up With HSS Expansion Package":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.hss_package_realistic(),
            #     ),

            # - - - TB - - -

            # # - - - TB SCALE-UP WITHOUT HSS - - -
            # "TB Program Scale-up Without HSS Expansion":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #     ),
            #
            # # - - - TB SCALE-UP MODERATE HRH HSS PACKAGE- - -
            # "TB Program Scale-up With HRH Scale-up (1%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.moderate_hrh_expansion(),
            #     ),
            #
            # # - - - TB SCALE-UP HISTORICAL HRH HSS PACKAGE- - -
            # "TB Program Scale-up With HRH Scale-up (4%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.hrh_using_historical_scaling(),
            #     ),
            #
            # # - - - TB SCALE-UP ACCELERATED HRH HSS PACKAGE- - -
            # "TB Program Scale-up With HRH Scale-up (6%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.accelerated_hrh_expansion(),
            #     ),

            # - - - TB SCALE-UP CONS 75th PERCENTILE- - -
            "TB Program Scale-up With Consumables at 75th Percentile":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.tb_scaleup(),
                    scenario_definitions.cons_at_75th_percentile(),
                ),

            # # - - - TB SCALE-UP CONS HIV LEVEL- - -
            # "TB Program Scale-up With Consumables at HIV levels":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.cons_at_HIV_availability(),
            #     ),
            #
            # # - - - TB SCALE-UP CONS EPI LEVEL- - -
            # "TB Program Scale-up With Consumables at EPI levels":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.cons_at_EPI_availability(),
            #     ),
            #
            # # - - - TB SCALE-UP *WITH* HSS PACKAGE- - -
            # "TB Programs Scale-up With HSS Expansion Package":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.hss_package_realistic(),
            #     ),
            #
            # # - - - MALARIA - - -
            #
            # # - - - Malaria SCALE-UP WITHOUT HSS - - -
            # "Malaria Program Scale-up Without HSS Expansion":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.malaria_scaleup(),
            #     ),
            #
            # # - - - Malaria SCALE-UP MODERATE HRH HSS PACKAGE- - -
            # "Malaria Program Scale-up With HRH Scale-up (1%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.moderate_hrh_expansion(),
            #     ),
            #
            # # - - - Malaria SCALE-UP HISTORICAL HRH HSS PACKAGE- - -
            # "Malaria Program Scale-up With HRH Scale-up (4%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hrh_using_historical_scaling(),
            #     ),
            #
            # # - - - Malaria SCALE-UP ACCELERATED HRH HSS PACKAGE- - -
            # "Malaria Program Scale-up With HRH Scale-up (6%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.accelerated_hrh_expansion(),
            #     ),
            #
            # # - - - Malaria SCALE-UP CONS 75th PERCENTILE- - -
            # "Malaria Program Scale-up With Consumables at 75th Percentile":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.cons_at_75th_percentile(),
            #     ),
            #
            # # - - - Malaria SCALE-UP CONS HIV LEVEL- - -
            # "Malaria Program Scale-up With Consumables at HIV levels":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.cons_at_HIV_availability(),
            #     ),
            #
            # # - - - Malaria SCALE-UP CONS EPI LEVEL- - -
            # "Malaria Program Scale-up With Consumables at EPI levels":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.cons_at_EPI_availability(),
            #     ),
            #
            # # - - - Malaria SCALE-UP *WITH* HSS PACKAGE- - -
            # "Malaria Programs Scale-up With HSS Expansion Package":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hss_package_realistic(),
            #     ),
            #
            # # - - - HTM - - -
            #
            # # - - - HTM SCALE-UP WITHOUT HSS - - -
            # "HTM Program Scale-up Without HSS Expansion":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #     ),
            #
            # # - - - HTM SCALE-UP MODERATE HRH HSS PACKAGE- - -
            # "HTM Program Scale-up With HRH Scale-up (1%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.moderate_hrh_expansion(),
            #     ),
            #
            # # - - - HTM SCALE-UP HISTORICAL HRH HSS PACKAGE- - -
            # "HTM Program Scale-up With HRH Scale-up (4%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hrh_using_historical_scaling(),
            #     ),
            #
            # # - - - HTM SCALE-UP ACCELERATED HRH HSS PACKAGE- - -
            # "HTM Program Scale-up With HRH Scale-up (6%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.accelerated_hrh_expansion(),
            #     ),
            #
            # # - - - HTM SCALE-UP CONS 75th PERCENTILE- - -
            # "HTM Program Scale-up With Consumables at 75th Percentile":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.cons_at_75th_percentile(),
            #     ),
            #
            # # - - - HTM SCALE-UP CONS HIV LEVEL- - -
            # "HTM Program Scale-up With Consumables at HIV levels":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.cons_at_HIV_availability(),
            #     ),
            #
            # # - - - HTM SCALE-UP CONS EPI LEVEL- - -
            # "HTM Program Scale-up With Consumables at EPI levels":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.cons_at_EPI_availability(),
            #     ),
            #
            # - - - HTM SCALE-UP *WITH* HSS PACKAGE- - -
            "HTM Programs Scale-up With HSS Expansion Package":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.hiv_scaleup(),
                    scenario_definitions.tb_scaleup(),
                    scenario_definitions.malaria_scaleup(),
                    scenario_definitions.hss_package_realistic(),
                ),

            # **************************************************
            # FRONTIER COMBINATIONS
            # **************************************************

            # # - - - HIV / TB SCALE-UP - - -
            # "HIV/TB Program Scale-up":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #     ),
            #
            # "HIV/TB Program Scale-up With HRH Scale-up (4%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.hrh_using_historical_scaling(),
            #     ),
            #
            # "HIV/TB Program Scale-up With Consumables at 75th Percentile":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.cons_at_75th_percentile(),
            #     ),
            #
            # "HIV/TB Program Scale-up With HSS Expansion Package":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.hss_package_realistic(),
            #     ),
            #
            # # - - - HIV / Malaria SCALE-UP - - -
            # "HIV/Malaria Program Scale-up":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #     ),
            #
            # "HIV/Malaria Program Scale-up With HRH Scale-up (4%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hrh_using_historical_scaling(),
            #     ),
            #
            # "HIV/Malaria Program Scale-up With Consumables at 75th Percentile":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.cons_at_75th_percentile(),
            #     ),
            #
            # "HIV/Malaria Program Scale-up With HSS Expansion Package":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.hiv_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hss_package_realistic(),
            #     ),
            #
            # # - - - TB / Malaria SCALE-UP - - -
            # "TB/Malaria Program Scale-up":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #     ),
            #
            # "TB/Malaria Program Scale-up With HRH Scale-up (4%)":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hrh_using_historical_scaling(),
            #     ),
            #
            # "TB/Malaria Program Scale-up With Consumables at 75th Percentile":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.cons_at_75th_percentile(),
            #     ),
            #
            # "TB/Malaria Program Scale-up With HSS Expansion Package":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.tb_scaleup(),
            #         scenario_definitions.malaria_scaleup(),
            #         scenario_definitions.hss_package_realistic(),
            #     ),

        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
