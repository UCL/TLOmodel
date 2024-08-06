"""This Scenario file is intended to help with debugging the scale-up of HIV. Tb and Malaria services, per issue #1413.

Changes to the main analysis:

* We're running this in MODE 1 and we're only looking.
* We're capturing the logged output from HIV, Tb and malaria
* We're limiting it to few scenarios: baseline + the scale-up of all HTM programs (no HealthSystem scale-up)

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


class MiniRunHTMWithAndWithoutHSS(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2031, 1, 1)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'mini_htm_with_and_without_hss',
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
            "Baseline":
                scenario_definitions.baseline(),

            # - - - HIV & TB & MALARIA SCALE-UP WITHOUT HSS PACKAGE- - -
            "HIV/Tb/Malaria Programs Scale-up WITHOUT HSS PACKAGE":
                mix_scenarios(
                    scenario_definitions.baseline(),
                    scenario_definitions.hiv_scaleup(),
                    scenario_definitions.tb_scaleup(),
                    scenario_definitions.malaria_scaleup(),
                ),
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
