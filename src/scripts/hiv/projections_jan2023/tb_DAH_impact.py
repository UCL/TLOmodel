
"""
This file run scenarios for assesing unavailability of TB-related Development Assistamce for Health (DAH)

It can be submitted on Azure Batch by running:

or locally using:

 tlo scenario-run src/scripts/hiv/projections_jan2023/tb_DAH_impact.py
  execute a single run:

 tlo scenario-run src/scripts/hiv/projections_jan2023/tb_DAH_impact.py --draw 1 0

 check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/projections_jan2023/tb_DAH_impact.py

 """
import warnings
from pathlib import Path
from typing import Dict
import random
from tlo import Date, logging
from tlo.scenario import BaseScenario
from tlo.methods import(
    demography,
    tb,
    symptommanager,
    hiv,
    healthburden,
    simplified_births,
    healthsystem,
    epi,
    enhanced_lifestyle,
    healthseekingbehaviour,
)

# Ignore warnings to avoid cluttering output from simulation - generally you do not
# need (and generally shouldn't) do this as warnings can contain useful information but
# we will do so here for the purposes of this example to keep things simple.
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))

class ImpactOfTbDaH(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 2525
       # self.seed = random.randint(0, 50000),
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2012, 12, 31)
        self.pop_size = 10000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'Tb_DAH_impactx36',
            'directory': Path('./outputs/nic503@york.ac.uk'),
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
        return [
            demography.Demography(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
        ]
    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return
    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        return {
            # baseline scenario
            "Baseline": {
                'Tb': {
                    'scenario': 0,
                    'probability_community_chest_xray': 0.0,


                },
            },
            # overrides availability of Xpert to nil
            "No Xpert Available": {
                'Tb': {
                    'scenario': 1,
                    'probability_community_chest_xray': 0.0,

                },
            },
            # overrides availability of CXR to nil
            "No CXR Available": {
                'Tb': {
                    'scenario': 2,
                    'probability_community_chest_xray': 0.0,

                },
            },
            #increase CXR by 10 percentage points
             "CXR_scale_up10": {
                 'Tb': {
                     'scenario': 3,
                     'probability_community_chest_xray': 0.0,

                 }
             },
            # increase CXR by 20 percentage points
            "CXR scale_up20": {
                'Tb': {
                    'scenario': 4,
                    'probability_community_chest_xray': 0.0,
                }
            },
            # increase CXR by 30 percentage points
            "CXR scale_up30": {
                'Tb': {
                    'scenario': 5,
                    'probability_community_chest_xray': 0.0,
                }
            },
            #introduce outreach services
            "Outreach services": {
                'Tb': {
                    'scenario': 6,
                    'probability_community_chest_xray': 0.1,
                }
            },
        }
if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
