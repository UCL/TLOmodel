
"""
This file run scenarios for assesing unavailability of TB-related Development Assistamce for Health (DAH)

It can be submitted on Azure Batch by running:

 tlo batch-submit src/scripts/hiv/projections_jan2023/tb_DAH_scenarios.py
or locally using:

 tlo scenario-run src/scripts/hiv/projections_jan2023/tb_DAH_scenarios.py
  execute a single run:

 tlo scenario-run src/scripts/hiv/projections_jan2023/tb_DAH_scenarios.py --draw 1 0

 check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/projections_jan2023/tb_DAH_scenarios.py

 """
import warnings
from pathlib import Path
from typing import Dict
import random
from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

# Ignore warnings to avoid cluttering output from simulation - generally you do not
# need (and generally shouldn't) do this as warnings can contain useful information but
# we will do so here for the purposes of this example to keep things simple.
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))


class ImpactOfTbDaH(BaseScenario):
    def __init__(self):
        super().__init__()
        # self.seed = 0
        self.seed = random.randint(0, 50000),
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2013, 12, 31)
        self.pop_size = 1000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 4

    def log_configuration(self):
        return {
            'filename': 'Tb_DAH_scenarios_test_run01',
           # 'directory': Path('./outputs')
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.hiv': logging.INFO,

            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        """
        return {
            "Baseline": {
                'Tb': {
                    'scenario': 0,
                    'probability_community_chest_xray': 0.0,
                },
            },
            "No Xpert Available": {
                'Tb': {
                    'scenario': 1,
                    'probability_community_chest_xray': 0.0,
                },
            },
            "No CXR Available": {
                'Tb': {
                    'scenario': 2,
                    'probability_access_to_xray': 0.0,
                    'probability_community_chest_xray': 0.0,
                },
            },
            "CXR scaleup": {
                'Tb': {
                    'scenario': 0,
                    'probability_access_to_xray': 0.11,
                    'probability_community_chest_xray': 0.0,
                }
            },
            "Outreach": {
                'Tb': {
                    'scenario': 0,
                    'probability_community_chest_xray': 0.005,
                }
            }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])

