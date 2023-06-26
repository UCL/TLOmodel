"""
Run simulation for two year and save logfile - for checking that change to logging in HealthBurden
does not affect results.

Run on the batch system using:
```tlo batch-submit src/scripts/calibration_analyses/scenarios/test_short_run.py```
"""

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class testRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2012, 1, 1)
        self.pop_size = 2_000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'test_run',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                "tlo.methods.contraception": logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return get_parameters_for_status_quo()


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
