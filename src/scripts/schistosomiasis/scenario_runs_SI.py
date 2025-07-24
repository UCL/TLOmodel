"""
This file uses the calibrated parameters for prop_susceptible
and runs with no MDA running
outputs can be plotted against data to check calibration

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/schistosomiasis/scenario_runs_SI.py

Run locally:
tlo scenario-run src/scripts/schistosomiasis/scenario_runs_SI.py

or execute a single run:
tlo scenario-run src/scripts/schistosomiasis/scenario_runs_SI.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/schistosomiasis/scenario_runs_SI.py

latest runs:
Job ID: schisto_scenarios-2024-12-04T124318Z
schisto_scenarios-2025-01-03T164654Z


"""

import random
from typing import Dict

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel

from tlo.scenario import BaseScenario
from scripts.schistosomiasis.schisto_scenario_definitions import (
    ScenarioDefinitions,
)
from tlo.analysis.utils import mix_scenarios


class SchistoScenarios(BaseScenario):

    def __init__(self):
        super().__init__()
        self.seed = 10
        self.start_date = Date(2010, 1, 1)

        # todo reset
        self.end_date = Date(2041, 1, 1)  # todo 2041 = 15 years of projections
        self.pop_size = 96_000  # todo if equal_allocation_by_district, 64,000=2k per district
        self.runs_per_draw = 10

        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)

    def log_configuration(self):
        return {
            'filename': 'schisto_scenarios_SI',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.schisto": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                # "tlo.methods.hiv": logging.INFO,
                # "tlo.methods.alri": logging.INFO,
                # "tlo.methods.diarrhoea": logging.INFO,
                # "tlo.methods.bladder_cancer": logging.INFO,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.enhanced_lifestyle": logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(use_simplified_births=True,
                         module_kwargs={
                             "Schisto": {"single_district": False},
                             "Demography": {"equal_allocation_by_district": True},
                             "Alri": {"log_indivdual": None},
                         }
                         )

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        # Load helper class containing the definitions of the elements of all the scenarios
        scenario_definitions = ScenarioDefinitions()
        # default is continued WASH, MDA 0.7 every 12 months in SAC

        return {

            "Continue WASH, no MDA":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.no_MDA(),
                    scenario_definitions.no_DALYs_light_infection(),
                ),

            # - - - Modify future MDA schedules with/without WASH activities - - -
            "Continue WASH, MDA SAC":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.high_coverage_MDA(),
                    scenario_definitions.no_DALYs_light_infection(),
                ),

            # MDA PSAC
            "Continue WASH, MDA PSAC":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.high_coverage_MDA(),
                    scenario_definitions.expand_MDA_to_PSAC(),
                    scenario_definitions.no_DALYs_light_infection(),
                ),

            # MDA ALL
            "Continue WASH, MDA All":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.high_coverage_MDA(),
                    scenario_definitions.expand_MDA_to_All(),
                    scenario_definitions.no_DALYs_light_infection(),
                ),

            "Scale-up WASH, no MDA":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.no_MDA(),
                    scenario_definitions.scaleup_WASH(),
                    scenario_definitions.no_DALYs_light_infection(),
                ),

        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
