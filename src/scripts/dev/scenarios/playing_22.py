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
        self.seed = 655123742
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2012, 1, 1)
        self.pop_size = 200
        self.number_of_draws = 1
        self.runs_per_draw = 3

        # self.suspend_date = Date(2011, 1, 1)
        self.restore_simulation = "playing_22-2023-12-22T010640Z"

    def log_configuration(self):
        return {
            # 'filename': 'my-scenario',
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
