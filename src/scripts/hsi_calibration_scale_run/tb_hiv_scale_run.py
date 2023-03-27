"""
This temparary file (a copy of src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py) defines a batch run
of a large population for 10 years with all disease modules and full use of HSIs.
It's used for tb/hiv hsi calibrations.

Run on the batch system using:
```tlo batch-submit src/scripts/hsi_calibration_scale_run/tb_hiv_scale_run.py```

or locally using:
    ```tlo scenario-run src/scripts/hsi_calibration_scale_run/tb_hiv_scale_run.py```

"""
import pandas as pd

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(years=20)
        self.pop_size = 20_000
        self.number_of_draws = 1
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
                "tlo.methods.hiv": logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return  # Using default parameters in all cases


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
