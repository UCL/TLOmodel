from tlo import Date, logging
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 655123742
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1)
        self.pop_size = 2000
        self.number_of_draws = 5
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'directory': None,
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return []

    def add_arguments(self, parser):
        parser.add_argument('--pop-size', type=int)

    def draw_parameters(self, draw_number, rng):
        return {
            'Lifestyle': {
                'init_p_urban': rng.randint(10, 20) / 100.0,
                'init_p_high_sugar': 0.52,
            },
        }
