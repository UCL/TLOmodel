"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for calibrations (demographic patterns, health burdens and healthsystem usage)

Run on the batch system using:
```tlo batch-submit src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py```

or locally using:
    ```tlo scenario-run src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py```

"""

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2016, 1, 1)  # The simulation will stop before reaching this date.
        self.pop_size = 10_000
        self.number_of_draws = 1
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'long_run_all_diseases',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,  # turn on demography log
                'tlo.methods.demography.detail': logging.WARNING,  # ensure detailed log is off
                'tlo.methods.healthsystem.summary': logging.INFO,  # turn on healthsystem summary log
                'tlo.methods.healthsystem': logging.INFO,  # keep the verbose log, just in case needed
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return get_parameters_for_status_quo()


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
