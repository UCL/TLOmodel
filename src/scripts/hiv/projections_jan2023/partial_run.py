import warnings
from typing import Dict
import random
from tlo.scenario import BaseScenario
import random
from pathlib import Path
from tlo import Date, Simulation, logging
from tlo.methods import (
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

# Ignore warnings to avoid cluttering output from simulation
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))

class ImpactOfTbDaH(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2012, 12, 31)
        self.pop_size = 5000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 4

    def log_configuration(self):
        return {
            'filename': 'Tb_DAH_scenarios_test_run03',
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
# The resource files
resourcefilepath = Path("./resources")
sim = Simulation(start_date= Date(2012, 12, 31), seed=2025, log_config=log_config, show_progress_bar=True)
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    tb.Tb(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=False),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
)

def draw_parameters(self, draw_number, rng):
    if draw_number < self.number_of_draws:
        return list(self._scenarios.values())[draw_number]
    else:
        return None

def _get_scenarios(self) -> Dict[str, Dict]:
    """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
    return {
        "Baseline": {
            'Tb': {
                'scenario': 0,
                'probability_community_chest_xray': 0.0,
                'scaling_factor_WHO': 99.9,
            },
        },
        "No Xpert Available": {
            'Tb': {
                'scenario': 1,
                'probability_community_chest_xray': 0.0,
                'scaling_factor_WHO': 99.9,
            },
        },
        "No CXR Available": {
            'Tb': {
                'scenario': 2,
                'probability_access_to_xray': 0.0,
                'probability_community_chest_xray': 0.0,
                'scaling_factor_WHO': 99.9,
            },
        },
        "CXR scaleup": {
            'Tb': {
                'scenario': 0,
                'probability_access_to_xray': 0.11,
                'probability_community_chest_xray': 0.0,
                'scaling_factor_WHO': 99.9,
            }
        },
        "Outreach services": {
            'Tb': {
                'scenario': 0,
                'probability_community_chest_xray': 0.01,
                'scaling_factor_WHO': 99.9,
            }
        }
    }

if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])

