from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 10000
        self.smaller_pop_size = 10000
        self.upper_iss_value = 5
        self.number_of_draws = 4
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'rti_calibrate_mortality.py',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, service_availability=['*']),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            rti.RTI(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources)
        ]


    def draw_parameters(self, draw_number, rng):
        mais_max = self.upper_iss_value + 1
        mais_min = 2
        iss_cut_off_scores = range(mais_min, mais_max)

        return {
            'RTI': {'unavailable_treatment_mortality_iss_cutoff': iss_cut_off_scores[draw_number]}
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
