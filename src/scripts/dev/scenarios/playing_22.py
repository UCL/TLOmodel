from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
)
from tlo.scenario import BaseScenario


class Playing22(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2014, 1, 1)
        self.pop_size = 1000
        self.number_of_draws = 3
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'playing_22',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True, service_availability=['*']),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
        ]


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
