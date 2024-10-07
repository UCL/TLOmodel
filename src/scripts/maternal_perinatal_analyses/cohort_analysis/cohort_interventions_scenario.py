from tlo import Date, logging
from tlo.methods import mnh_cohort_module
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class BaselineScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 537184
        self.start_date = Date(2024, 1, 1)
        self.end_date = Date(2025, 1, 1)
        self.pop_size = 5000
        self.number_of_draws = 1
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'cohort_test', 'directory': './outputs',
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.INFO,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.labour": logging.INFO,
                "tlo.methods.labour.detail": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.pregnancy_supervisor": logging.INFO,
                "tlo.methods.postnatal_supervisor": logging.INFO,
            }
        }

    def modules(self):
        return [*fullmodel(resourcefilepath=self.resources),
                 mnh_cohort_module.MaternalNewbornHealthCohort(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):
        return {
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])