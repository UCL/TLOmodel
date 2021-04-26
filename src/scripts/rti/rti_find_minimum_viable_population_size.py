from tlo import Date, logging
from tlo.methods import (
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2050, 1, 1)
        self.pop_size = 1500000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'rti_min_viable_population',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=self.resources),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True, service_availability=['*']),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            rti.RTI(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources)
        ]

# Here I want to run the model with two sets of parameters multiple times. Once where only singular injuries
# are given out and once where we allow multiple injuries

    def draw_parameters(self, draw_number, rng):
        return {}


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
