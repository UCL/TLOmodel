from tlo import Date, logging

from tlo.methods import service_integration
from tlo.methods.fullmodel import fullmodel

from tlo.scenario import BaseScenario


class ServiceIntegrationScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 661184
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2035, 1, 1)
        self.pop_size = 100_000
        self.number_of_draws = 1
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'service_integration_scenario', 'directory': './outputs',
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
            }
        }

    def modules(self):
        return [*fullmodel(resourcefilepath=self.resources),
                service_integration.ServiceIntegration(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):

        if draw_number == 0:
            return {}
        else:
            pass


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
