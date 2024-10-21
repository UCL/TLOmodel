"""
This file defines a batch run to test whether consumable availability swtich by year works
Run on the batch system using:
```tlo batch-submit src/scripts/check_consumable_availability_scenarios/test_scenario.py```
or locally using:
    ```tlo scenario-run src/scripts/check_consumable_availability_scenarios/test_scenario.py```
"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 99
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2013, 12, 31)
        self.pop_size = 1000 # large population size for final simulation - 100,000
        self.number_of_draws = 3
        self.runs_per_draw = 1  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'test_scenario',
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'cons_availability': ['default',
                                      'default', 'default'][draw_number],
                'cons_availability_postSwitch': ['default',
                                      'scenario6', 'scenario11'][draw_number],
                'year_cons_availability_switch': [2011, 2011, 2011][draw_number]
               }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
