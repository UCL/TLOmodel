"""
This file defines a batch run to calculate the health effect of updated consumable availability estimates
as a result of a supply chain intervention. The following scenarios are considered:
1. 'default' : this is the benchmark scenario with 2018 levels of consumable availability
2. 'scenario1' : [Level 1a + 1b] All items perform as well as consumables other than drugs/diagnostic tests
3. 'scenario2' : [Level 1a + 1b] 1 + All items perform as well as consumables classified as 'Vital' in the Essential Medicines List
4. 'scenario3' : [Level 1a + 1b] 2 + All facilities perform as well as those in which consumables stock is managed by pharmacists
5. 'scenario6' : [Level 1a + 1b] All facilities have the same probability of consumable availability as the 75th percentile best performing facility for each individual item
6. 'scenario7' : [Level 1a + 1b] All facilities have the same probability of consumable availability as the 90th percentile best performing facility for each individual item
7. 'scenario8' : [Level 1a + 1b] All facilities have the same probability of consumable availability as the 99th percentile best performing facility for each individual item
8. 'scenario9' : [Level 1a + 1b + 2] All facilities have the same probability of consumable availability as the 99th percentile best performing facility for each individual item
9. 'scenario10' : [Level 1a + 1b] All programs perform as well as HIV
10. 'scenario11' : [Level 1a + 1b] All programs perform as well as EPI
12. 'all': all consumable are always available - provides the theoretical maximum health gains which can be made through improving consumable supply

The batch runs are for a large population for a long time with all disease modules and full use of HSIs.
Run on the batch system using:
```tlo batch-submit src/scripts/consumables_analyses/scenario_impact_of_consumable_scenarios.py```
or locally using:
    ```tlo scenario-run src/scripts/consumables_analyses/scenario_impact_of_consumable_scenarios.py```
"""


from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class ImpactOfConsumablesScenarios(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 99
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2019, 12, 31)
        self.pop_size = 100_000 # large population size for final simulation - 100,000
        self.number_of_draws = 11  # <- 11 scenarios
        self.runs_per_draw = 10  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'impact_of_consumables_scenarios',
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
                                      'scenario1', 'scenario2', 'scenario3',
                                      'scenario6', 'scenario7', 'scenario8',
                                      'scenario9', 'scenario10', 'scenario11',
                                      'all'][draw_number]
               }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
