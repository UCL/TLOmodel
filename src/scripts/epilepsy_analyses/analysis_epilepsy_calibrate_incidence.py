import numpy as np

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epilepsy,
    healthseekingbehaviour,
    healthsystem,
    healthburden,
    simplified_births,
    symptommanager,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 1000
        self.number_of_draws = 5
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'analysis_epilepsy_calibrate_incidence.py',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            epilepsy.Epilepsy(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        base_rate_min = 0.5 - 0.15
        base_rate_max = 0.5 + 0.15
        base_rate_linspace = np.linspace(base_rate_min, base_rate_max, self.number_of_draws)
        return {
            'Epilepsy': {
                'base_3m_prob_epilepsy': base_rate_linspace[draw_number]
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
