from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

class ImpactOfConsumablesAvailability(BaseScenario):
    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2015, 1, 1),
            initial_population_size=10_000,
            number_of_draws= 2,
            runs_per_draw=2,
        )
    def log_configuration(self):
        return {
            'filename': 'impact_of_consumables_availability',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
            }
        }
    def modules(self):
        return fullmodel()
    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {'cons_availability': ['default', 'all'][draw_number]}
        }
