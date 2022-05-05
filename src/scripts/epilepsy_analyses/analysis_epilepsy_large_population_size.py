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
        self.pop_size = 500000
        self.smaller_pop_size = 500000
        self.number_of_draws = 1
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'analysis_epilepsy_large_population_size.py',
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
        return {}


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
