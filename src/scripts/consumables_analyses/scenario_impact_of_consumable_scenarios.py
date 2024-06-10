"""
This file defines a batch run to calculate the health effect of updated consumable availability estimates
as a result of a supply chain intervention. The following scenarios are considered:
1. 'scenario1' - all facilities set to 1b,
2. 'scenario2' - all facility ownership set to CHAM,
3. 'scenario3' - all facilities have functional computers,
4. 'scenario4' - all facility drug stocks are managed by pharmacists or pharmacist technicians,
5. 'scenario5' - all facilities have a functional emergency vehicle,
6. 'scenario6' - all facilities provide diagnostic services,
7. 'scenario7' - all facilities are within 10 kms from the relevant DHO,
8. 'scenario8' - all facilities are within 10 kms from the relevant Regional medical Store (Warehouse),
The batch runs are for a large population for a long time with all disease modules and full use of HSIs.
Run on the batch system using:
```tlo batch-submit src/scripts/consumables_analyses/scenario_impact_of_consumable_scenarios.py```
or locally using:
    ```tlo scenario-run src/scripts/consumables_analyses/scenario_impact_of_consumable_scenarios.py```
"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

class ImpactOfConsumablesScenarios(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 99
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2019, 12, 31)
        self.pop_size = 50 # large population size for final simulation
        self.number_of_draws = 2  # <- 10 scenarios (10)
        self.runs_per_draw = 2  # <- repeated this many times

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
                'cons_availability': ['default', 'scenario1'][draw_number] # , 'scenario2', 'scenario3', 'scenario4', 'scenario5', 'scenario6', 'scenario7', 'scenario8', 'all'
               }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
