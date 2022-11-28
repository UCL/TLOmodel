"""
This file defines a batch run to get sims results to be used by the analysis_contraception_plot_table.
Run on the remote batch system using:
```tlo batch-submit src/scripts/contraception/scenarios/run_analysis_contraception_all_diseases.py```
or locally using:
```tlo scenario-run src/scripts/contraception/scenarios/run_analysis_contraception_all_diseases.py```
"""

import pandas as pd

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class RunAnalysisCo(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(years=2)
        self.pop_size = 20_000  # <- recommended population size for the runs
        self.number_of_draws = 1  # <- one scenario
        self.runs_per_draw = 1  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'run_analysis_contraception_all_diseases',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.contraception": logging.DEBUG,
                "tlo.methods.demography": logging.INFO
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return  # Using default parameters in all cases


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
