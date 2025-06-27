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
        self.pop_size = 200_000
        self.number_of_draws = 28
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
        return [*fullmodel(),
                service_integration.ServiceIntegration()]

    def draw_parameters(self, draw_number, rng):

        params_all = {'ServiceIntegration':{'integration_year': 2025}}
        params_oth ={1: {'serv_integration': 'htn'},
                     2: {'serv_integration': 'htn_max'},
                     3: {'serv_integration': 'dm'},
                     4: {'serv_integration': 'dm_max'},
                     5: {'serv_integration': 'hiv'},
                     6: {'serv_integration': 'hiv_max'},
                     7: {'serv_integration': 'tb'},
                     8: {'serv_integration': 'tb_max'},
                     9: {'serv_integration': 'mal'},
                     10: {'serv_integration': 'mal_max'},
                     11: {'serv_integration': 'fp_scr'},
                     12: {'serv_integration': 'fp_scr_max'},
                     13: {'serv_integration': 'anc'},
                     14: {'serv_integration': 'anc_max'},
                     15: {'serv_integration': 'pnc'},
                     16: {'serv_integration': 'pnc_max'},
                     17: {'serv_integration': 'fp_pn'},
                     18: {'serv_integration': 'fp_pn_max'},
                     19: {'serv_integration': 'epi'},
                     20: {'serv_integration': 'chronic_care'},
                     21: {'serv_integration': 'chronic_care_max'},
                     22: {'serv_integration': 'all_screening'},
                     23: {'serv_integration': 'all_screening_max'},
                     24: {'serv_integration': 'all_mch'},
                     25: {'serv_integration': 'all_mch_max'},
                     26: {'serv_integration': 'all_int'},
                     27: {'serv_integration': 'all_int_max'},
                     }

        if draw_number == 0:
            return params_all
        else:
            params_all['ServiceIntegration'].update(params_oth[draw_number])
            return params_all

if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
