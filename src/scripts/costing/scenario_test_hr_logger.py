'''
Run on the batch system using:
```tlo batch-submit src/scripts/costing/scenario_test_hr_logger.py```
or locally using:
    ```tlo scenario-run src/scripts/costing/scenario_test_hr_logger.py```
'''

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

class SampleCostingScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2012, 1, 1)
        self.pop_size = 1_000
        self.number_of_draws = 2  # <- one scenario
        self.runs_per_draw = 1 # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'scenario_test_hr_logger',
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'cons_availability': ['default', 'all'][draw_number]
            }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
