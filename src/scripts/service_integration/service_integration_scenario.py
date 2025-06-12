from tlo import Date, logging

from tlo.methods import service_integration
from tlo.methods.fullmodel import fullmodel

from tlo.scenario import BaseScenario


class ServiceIntegrationScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 537184
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2050, 1, 1)
        self.pop_size = 100_000
        self.number_of_draws = 13
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'service_integration_scenario', 'directory': './outputs',
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.cardio_metabolic_disorders": logging.INFO,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.depression": logging.INFO,
                "tlo.methods.epilepsy": logging.INFO,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.labour": logging.INFO,
                "tlo.methods.labour.detail": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.pregnancy_supervisor": logging.INFO,
                "tlo.methods.postnatal_supervisor": logging.INFO,
                "tlo.methods.stunting": logging.INFO,
            }
        }

    def modules(self):
        return [*fullmodel(resourcefilepath=self.resources),
                service_integration.ServiceIntegration(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):

        params_all = {'ServiceIntegration':{'integration_year': 2020}}
        params_oth = {1: {'serv_int_chronic': True},
                      2: {'serv_int_screening': ['htn']},
                      3: {'serv_int_screening': ['dm']},
                      4: {'serv_int_screening': ['hiv']},
                      5: {'serv_int_screening': ['tb']},
                      6: {'serv_int_screening': ['fp']},
                      7: {'serv_int_screening': ['mal']},
                      8: {'serv_int_screening': ['htn', 'dm', 'hiv', 'tb', 'fp', 'mal']},
                      9: {'serv_int_mch': ['pnc']},
                      10: {'serv_int_mch': ['fp']},
                      11: {'serv_int_mch': ['pnc', 'fp']},
                      12: {'serv_int_chronic': True,
                           'serv_int_screening': ['htn', 'dm', 'hiv', 'tb', 'fp', 'mal'],
                           'serv_int_mch': ['pnc', 'fp']}}

        if draw_number == 0:
            return params_all
        else:
            params_all['ServiceIntegration'].update(params_oth[draw_number])
            return params_all

if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
