"""This Scenario file run the model under different assumptions for the consumable availability in order to estimate the
cost under each scenario for the HSSP-III duration

Run on the batch system using:
```
tlo batch-submit src/scripts/costing/platform_based_costing/scenario_consumable_costing.py
```

or locally using:
```
tlo scenario-run src/scripts/costing/platform_based_costing/scenario_consumable_costing.py
 ```

"""

import random

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class ConsumablesCosting(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2013, 1, 1)  # Date(2051, 1, 1)  # todo need to log to mid-year 2050
        self.pop_size = 5000 # 100_000
        self.scenarios = [0, 1] # add scenarios as necessary
        self.number_of_draws = len(self.scenarios)
        self.runs_per_draw = 1 #3

    def log_configuration(self):
        return {
            'filename': 'consumables_costing',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthsystem": logging.INFO
            }
        }

    def modules(self):
        return (
                fullmodel(module_kwargs={"HealthSystem": {"disable": False}})
                + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher()]
        )

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'cons_availability': ['default', 'default'][draw_number],
                'year_cons_availability_switch': 2011,
                'cons_availability_postSwitch': ['default', 'all'][draw_number],
                },
            "ImprovedHealthSystemAndCareSeekingScenarioSwitcher": {
                "max_healthsystem_function": [False, True],
                "max_healthcare_seeking": [False, True],
                "year_of_switch": 2011
            }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
