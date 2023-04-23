"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
for consumable availability scenarios:
1. All HIV, TB, Malaria consumables are always available
2. All facilities have consumables (this has a positive impact on availability across consumables)

Run on the batch system using:
```tlo batch-submit src/scripts/healthsystem/impact_of_cons_availability/consumable_availability_scenarios.py```

or locally using:
    ```tlo scenario-run src/scripts/healthsystem/impact_of_cons_availability/consumable_availability_scenarios.py```

"""

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

class ConsumablesAvailabilityScenarios(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1) # %% test short period for quick run
        self.pop_size = 20_000  # <- recommended population size for the runs
        self.number_of_draws = 3  # <- one scenario
        self.runs_per_draw = 3 # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'consumables_availability_scenarios',
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(
            resourcefilepath=self.resources,
            module_kwargs = {
                "HealthSystem": {"service_availability": ["*"],
                                 # "cons_availability": "default",
                }
            }
        )

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'cons_availability': ['default', 'none', 'all'][draw_number]
                }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
