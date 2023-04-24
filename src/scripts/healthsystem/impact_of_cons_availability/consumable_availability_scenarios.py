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
from pathlib import Path
from typing import Dict, List

from tlo import Date, Module, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

class ConsumablesAvailabilityScenarios(BaseScenario, Module):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1) # %% test short period for quick run
        self.pop_size = 20_000  # <- recommended population size for the runs
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)  # <- one scenario
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
                'override_availability_of_consumables': list(self._scenarios.values())[draw_number]
                }
        }

    def _get_scenarios(self) -> Dict[str, List[str]]:
        """ Return the Dict with values for the parameter `override_availability_of_consumables`
        keyed by a name for the scenario which represents the disease module for which consumable availability
        is assumed to be perfect """

        # HIV consumables
        tmp = list(self.sim.modules["Hiv"].item_codes_for_consumables_required.values())
        hiv_cons = [None] * len(tmp)
        for cons in range(len(tmp)):
            if isinstance(tmp[cons], dict):
                # extract item code
                item_code = list(tmp[cons].keys())[0]
                hiv_cons[cons] = item_code
            else:
                hiv_cons[cons] = tmp[cons]

        # TB consumables
        tmp2 = list(self.sim.modules["tb"].item_codes_for_consumables_required.values())
        tb_cons = [None] * len(tmp2)
        for cons in range(len(tmp2)):
            if isinstance(tmp2[cons], dict):
                # extract item code
                item_code = list(tmp2[cons].keys())[0]
                tb_cons[cons] = item_code
            else:
                tb_cons[cons] = tmp2[cons]

        override_availability_of_consumables = dict({"HIV": hiv_cons, "TB": tb_cons})
        return override_availability_of_consumables

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
