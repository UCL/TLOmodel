"""This Scenario file run the model under different assumptions for the HealthSystem and Vertical Program Scale-up

check scenarios are generated correctly:
tlo scenario-run --draw-only
src/scripts/comparison_of_horizontal_and_vertical_programs/global_fund_analyses/scenario_hss_elements_gf.py

run locally:
tlo scenario-run src/scripts/comparison_of_horizontal_and_vertical_programs/global_fund_analyses/scenario_hss_elements_gf.py


Run on the batch system using:

tlo batch-submit --more-memory
src/scripts/comparison_of_horizontal_and_vertical_programs/global_fund_analyses/scenario_hss_elements_gf.py

"""

from pathlib import Path
from typing import Dict

from scripts.comparison_of_horizontal_and_vertical_programs.global_fund_analyses.scenario_definitions_gf import (
    ScenarioDefinitions,
)
from tlo import Date, logging
from tlo.analysis.utils import mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class HSSElements(BaseScenario):
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
            'filename': 'hss_elements',
            'directory': Path('./outputs'),
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthsystem': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
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

        scenario_definitions = ScenarioDefinitions()

        return {
            "Baseline": scenario_definitions.baseline(),

            # ***************************
            # HEALTH SYSTEM STRENGTHENING
            # ***************************

            # - - - Human Resource for Health - - -

            "HRH Scale-up Following Historical Growth":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.hrh_using_historical_scaling(),
                ),

            "HRH Moderate Scale-up (1%)":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.moderate_hrh_using_historical_scaling(),
                ),

            "HRH Accelerated Scale-up (6%)":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.accelerated_hrh_using_historical_scaling(),
                ),

            "CHW Scale-up Following Historical Growth":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.scale_dcsa_with_historical_average(),
                ),

            "Increase Capacity at Primary Care Levels":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.increase_capacity_at_primary_care(),
                ),

            # - - - Supply Chains - - -
            "Consumables Increased to 75th Percentile":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.cons_at_75th_percentile(),
                ),

            "Consumables Available at HIV levels":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.cons_at_HIV_availability(),
                ),

            "Consumables Available at EPI levels":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.cons_at_EPI_availability(),
                ),

            # "Perfect Consumables Availability":
            #     mix_scenarios(
            #         scenario_definitions.baseline(),
            #         scenario_definitions.all_consumables_available(),
            #     ),

            # - - - FULL PACKAGE OF HEALTH SYSTEM STRENGTHENING - - -
            "FULL PACKAGE":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.hss_package(),
                ),
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
