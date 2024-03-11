'''


Run on the batch system using:
```tlo batch-submit src/scripts/costing/example_costing_scenario.py```

or locally using:
    ```tlo scenario-run src/scripts/costing/example_costing_scenario.py```

'''

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

class SampleCostingScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2013, 1, 1)
        self.pop_size = 20_000  # <- recommended population size for the runs
        self.number_of_draws = 1  # <- one scenario
        self.runs_per_draw = 1 # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'example_costing_scenario',
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
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
