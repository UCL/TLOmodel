"""
This file defines a batch run to calculate the health effect of updated consumable availability estimates
as a result of a supply chain intervention. The following scenarios are currently considered:
1. Scenario 1: Provide computers to all level 1a and 1b facilities.

The batch runs are for a large population for a long time with all disease modules and full use of HSIs.

Run on the batch system using:
```tlo batch-submit src/scripts/healthsystem/impact_of_cons_availability/impact_of_cons_availability_intervention.py```

or locally using:
    ```tlo scenario-run src/scripts/healthsystem/impact_of_cons_availability/impact_of_cons_availability_intervention.py```

"""

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class ImpactOfConsumablesAvailabilityIntervention(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2025, 12, 31)
        self.pop_size = 100_000 # large population size for final simulation
        self.number_of_draws = 2  # <- one scenario
        self.runs_per_draw = 3  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'impact_of_consumables_availability_intervention',
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
                "tlo.methods.healthsystem": logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'change_cons_availability_to': ['alternate_scenario1', 'NO_CHANGE'][draw_number]
               }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
