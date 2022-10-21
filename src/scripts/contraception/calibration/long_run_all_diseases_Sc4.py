"""
  ALTERNATIVE PARAMETER VALUES

```tlo batch-submit src/scripts/contraception/calibration/long_run_all_diseases_Sc4.py```

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
        self.end_date = self.start_date + pd.DateOffset(years=10)
        self.pop_size = 5_000
        self.number_of_draws = 1
        self.runs_per_draw = 2

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
        return {
            'Contraception': {
                'scaling_factor_on_monthly_risk_of_pregnancy': [
                    1.227,
                    0.799 * 0.9,    # <---- edited in Sc3
                    0.829 * 0.8,    # <---- edited in Sc3
                    0.809 * 1.05,   # <---- edited in Sc4
                    0.749 * 1.15,   # <---- edited in Sc4
                    0.645,
                    0.941
                ]
                }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
