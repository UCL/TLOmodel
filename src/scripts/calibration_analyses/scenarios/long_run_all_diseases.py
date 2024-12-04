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
from tlo.util import str_to_pandas_date


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2031, 1, 1)  # The simulation will stop before reaching this date.
        self.pop_size = 20_000
        self.number_of_draws = 1
        self.runs_per_draw = 10
        self.healthsystem_mode = 1

    def log_configuration(self):
        return {
            'filename': 'long_run_all_diseases',  # <- (specified only for local running)
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
        params = get_parameters_for_status_quo()
        params['HealthSystem']['mode_appt_constraints'] = self.healthsystem_mode
        return params

    def add_arguments(self, parser):
        parser.add_argument('--healthsystem-mode', type=int)
        parser.add_argument('--end-date', type=str_to_pandas_date)

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
