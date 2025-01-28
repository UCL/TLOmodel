"""This Scenario file run the model under different assumptions for the HealthSystem and Vertical Program Scale-up

Run on the batch system using:
```
tlo batch-submit
 src/scripts/comparison_of_horizontal_and_vertical_programs/scenario_hss_elements.py
```

"""

from pathlib import Path
from typing import Dict

from scripts.comparison_of_horizontal_and_vertical_programs.scenario_definitions import (
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
        self.end_date = Date(2031, 1, 1)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 3  # <--- todo: N.B. Very small number of repeated run, to be efficient for now

    def log_configuration(self):
        return {
            'filename': 'hss_elements',
            'directory': Path('./outputs'),
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.WARNING,
                'tlo.methods.healthsystem.summary': logging.INFO,
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

            "Double Capacity at Primary Care":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.double_capacity_at_primary_care(),
                ),

            "HRH Keeps Pace with Population Growth":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions._hrh_at_pop_growth(),
                ),

            "HRH Increases at GDP Growth":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions._hrh_at_grp_growth(),
                ),

            "HRH Increases above GDP Growth":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.hrh_above_gdp_growth(),
                ),


            # - - - Quality of Care - - -
            "Perfect Clinical Practice":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions._perfect_clinical_practice(),
                ),

            "Perfect Healthcare Seeking":
               mix_scenarios(
                   scenario_definitions.baseline(),
                   scenario_definitions.perfect_healthcare_seeking(),
               ),

            # - - - Supply Chains - - -
            "Perfect Availability of Vital Items":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.vital_items_available(),
                ),

            "Perfect Availability of Medicines":
            mix_scenarios(
                scenario_definitions.baseline(),
                scenario_definitions.medicines_available(),

            ),

            "Perfect Availability of All Consumables":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.all_consumables_available(),
                ),

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
