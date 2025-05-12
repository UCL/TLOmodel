from tlo import Date, logging

from tlo.methods import service_integration
from tlo.methods.fullmodel import fullmodel

from tlo.scenario import BaseScenario


class ServiceIntegrationScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 537184
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2015, 1, 1)
        self.pop_size = 75_000
        self.number_of_draws = 12
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'service_integration_scenario', 'directory': './outputs',
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
            }
        }

    def modules(self):
        return [*fullmodel(resourcefilepath=self.resources),
                service_integration.ServiceIntegration(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):

        params_all = {'ServiceIntegration':{'integration_date': Date(2011, 1,1)}}
        params_oth = {2: {'serv_int_chronic': True},
                      3: {'serv_int_screening': ['htn']},
                      4: {'serv_int_screening': ['dm']},
                      5: {'serv_int_screening': ['hiv']},
                      6: {'serv_int_screening': ['tb']},
                      7: {'serv_int_screening': ['fp']},
                      8: {'serv_int_screening': ['mal']},
                      9: {'serv_int_mch': ['pnc']},
                      10: {'serv_int_mch': ['fp']},
                      11: {'serv_int_mch': ['mal']},
                      12: {'serv_int_mch': ['epi']}}

        # todo: start at 0
        if draw_number == 1:
            return params_all
        else:
            params_all['ServiceIntegration'].update(params_oth[draw_number])
            return params_all

if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
