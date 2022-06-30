"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for calibrations (demographic patterns, health burdens and healthsystem usage)

Run on the batch system using:
```tlo batch-submit src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py```

or locally using:
    ```tlo scenario-run src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py```

"""

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2019, 12, 31)
        self.pop_size = 50_000  # <- recommended population size for the runs
        self.number_of_draws = 1  # <- one scenario
        self.runs_per_draw = 50  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'long_run',  # <- (specified only for local running)
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
        return  # Using default parameters in all cases


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
