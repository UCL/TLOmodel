"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for calibrations (demographic patterns, health burdens and healthsystem usage)

Run on the batch system using:
```tlo batch-submit src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py```

or locally using:
    ```tlo scenario-run src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py```

"""

from tlo import Date, logging
from tlo.analysis.performance import PerformanceMonitor
from tlo.analysis.utils import get_parameters_for_status_quo
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2012, 1, 10)  # The simulation will stop before reaching this date.
        self.pop_size = 100_000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'long_run_all_diseases',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
            }
        }

    def modules(self):
        return (fullmodel(resourcefilepath=self.resources) +
                [
                    PerformanceMonitor(log_perf=True,
                           log_perf_freq=1,
                           log_pop_hash=False,
                           save_sim=False)])

    def draw_parameters(self, draw_number, rng):
        return get_parameters_for_status_quo()


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
