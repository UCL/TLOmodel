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


class TestShorterTreatmentScenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.seed = 5
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2050, 1, 1)
        self.pop_size = 100_000
        self.number_of_draws = 2
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'test_shorter_treatment_scenario',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=False, service_availability=['*']),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'Tb': {'scenario': [0, 5][draw_number]}
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
