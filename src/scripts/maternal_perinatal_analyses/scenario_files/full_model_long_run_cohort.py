from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel

from tlo.scenario import BaseScenario


class FullModelRunForCohort(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 562661
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2026, 1, 1)
        self.pop_size = 200_000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'fullmodel_200k_cohort', 'directory': './outputs',
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.contraception": logging.DEBUG,
            }
        }

    def modules(self):
        return fullmodel()

    def draw_parameters(self, draw_number, rng):
        return {}


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
