"""
This file defines a batch run to calculate the health effect of updated consumable availability estimates
as a result of a supply chain intervention. The following scenarios are currently considered:
1. 'scenario_fac_type' - all facilities set to 1b,
2. 'scenario_fac_owner' - all facility ownership set to CHAM,
3. 'scenario_functional_computer' - all facilities have functional computers,
4. 'scenario_incharge_drug_orders' - all facility drug stocks are managed by pharmacists or pharmacist technicians,
5. 'scenario_functional_emergency_vehicle' - all facilities have a functional emergency vehicle,
6. 'scenario_service_diagnostic' - all facilities provide diagnostic services,
7. 'scenario_dist_todh' - all facilities are within 10 kms from the relevant DHO,
8. 'scenario_dist_torms' - all facilities are within 10 kms from the relevant Regional medical Store (Warehouse),
9. 'scenario_drug_order_fulfilment_freq_last_3mts' - all facilities had 3 drug order fulfilments in the last three months,
10. 'scenario_all_features' - all of the above features

The batch runs are for a large population for a long time with all disease modules and full use of HSIs.

Run on the batch system using:
```tlo batch-submit src/scripts/healthsystem/impact_of_cons_availability/impact_of_cons_regression_scenarios.py```

or locally using:
    ```tlo scenario-run src/scripts/healthsystem/impact_of_cons_availability/impact_of_cons_regression_scenarios.py```

"""

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class ImpactOfConsumablesAvailabilityIntervention(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2025, 1, 1)
        self.pop_size = 20_000 # large population size for final simulation
        self.number_of_draws = 11  # <- 11 scenarios
        self.runs_per_draw = 3  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'impact_of_consumables_availability_intervention',
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
                'change_cons_availability_to': ['NO_CHANGE', 'scenario_fac_type', 'scenario_fac_owner',
                                                'scenario_functional_computer', 'scenario_incharge_drug_orders',
                                                'scenario_functional_emergency_vehicle', 'scenario_service_diagnostic',
                                                'scenario_dist_todh', 'scenario_dist_torms',
                                                'scenario_drug_order_fulfilment_freq_last_3mts',
                                                'scenario_all_features'][draw_number]
               }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
