from random import randint

from tlo import Date, logging

from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb
)

from tlo.scenario import BaseScenario


class TestTBIncidence(BaseScenario):

    def __init__(self):
        super().__init__()
        self.seed = randint(0, 5000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2035, 1, 1)
        self.pop_size = 250_000
        self.number_of_draws = 5
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'test_tb_incidence',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.demography': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=False, service_availability=['*']),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'Tb': {
                'scenario': 0,
                'transmission_rate': [16.71012, 17.71012, 18.71012, 19.71012, 20.71012][draw_number]
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
