"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for calibrations (demographic patterns, health burdens and healthsystem usage)

Run on the batch system using:
```tlo batch-submit src/scripts/hsi_calibration_scale_run/10_year_scale_run.py```

or locally using:
    ```tlo scenario-run src/scripts/hsi_calibration_scale_run/10_year_scale_run.py```

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
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 20_000
        self.number_of_draws = 2
        self.runs_per_draw = 10

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
        return {**get_parameters_for_status_quo(),
                **{'HealthSeekingBehaviour': {
                    'prob_non_emergency_care_seeking_by_level': [[1.0, 0.0, 0.0, 0.0],
                                                                 [0.14, 0.61, 0.10, 0.15]][draw_number]
                }
                }
                }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
