import numpy as np

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
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
        self.pop_size = 20000
        self.smaller_pop_size = 20000
        self.number_of_draws = 25
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'analysis_epilepsy_calibrate_incidence_grid.py',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
                'tlo.methods.healthsystem': logging.DEBUG

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
        # Create parameters to vary
        grid = self.make_grid(
            {'base_3m_prob_epilepsy': np.linspace(0.0001767 - 0.000015, 00.0001767 + 0.000015, 5),
             'base_prob_3m_epi_death': np.linspace(0.000737 - 0.000015, 0.000737 + 0.000015, 5)}
        )
        return {
            'Epilepsy': {
                'base_3m_prob_epilepsy': grid['base_3m_prob_epilepsy'][draw_number],
                'base_prob_3m_epi_death': grid['base_prob_3m_epi_death'][draw_number]
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
