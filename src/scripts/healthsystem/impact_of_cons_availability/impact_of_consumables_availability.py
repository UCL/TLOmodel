"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for calibrations (demographic patterns, health burdens and healthsytstem usage)

Run on the batch system using:
```tlo batch-submit src/scripts/healthsystem/impact_of_cons_availability/impact_of_consumables_availability.py```

or locally using:
    ```tlo scenario-run src/scripts/healthsystem/impact_of_cons_availability/impact_of_consumables_availability.py```

"""

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class ImpactOfConsumablesAvailability(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2019, 12, 31)
        self.pop_size = 20_000  # <- recommended population size for the runs
        self.number_of_draws = 3  # <- one scenario
        self.runs_per_draw = 3  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'impact_of_consumables_availability',
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'cons_availability': ['default', 'none', 'all'][draw_number]
                }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
