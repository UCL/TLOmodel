from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class Playing22(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 655123742
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 500
        self.number_of_draws = 1
        self.runs_per_draw = 1

        self.resume_simulation = "<dirname of the suspended run>"

    def log_configuration(self):
        return {
            # 'filename': 'my-scenario',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
