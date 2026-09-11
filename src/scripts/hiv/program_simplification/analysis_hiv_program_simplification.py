"""This Scenario file runs the model under different assumptions for the HIV program simplification


check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/program_simplification/analysis_hiv_program_simplification.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/program_simplification/analysis_hiv_program_simplification.py

or execute a single run:
tlo scenario-run src/scripts/hiv/program_simplification/analysis_hiv_program_simplification.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/hiv/program_simplification/analysis_hiv_program_simplification.py

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download calibration_script-2022-04-12T190518Z

if running locally need to parse each folder
tlo parse-log /Users/tmangal/PycharmProjects/TLOmodel/outputs/hiv_program_simplification-2025-07-24T160218Z/0/0

"""

from pathlib import Path
from typing import Dict
from scripts.hiv.program_simplification.scenario_definitions import ScenarioDefinitions

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class HIV_Progam_Elements(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 10
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2013, 1, 1)
        self.pop_size = 1000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'hiv_program_simplification',
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
            }
        }

    def modules(self):
        return (
            fullmodel(use_simplified_births=True,
                      module_kwargs={"HealthSystem": {"equip_availability": 'all'}})
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""

        scenario_definitions = ScenarioDefinitions()

        return {
            "Status Quo": scenario_definitions.status_quo(),

        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
