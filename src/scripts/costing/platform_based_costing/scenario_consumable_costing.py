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

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario
from tlo.analysis.utils import get_root_path

root = get_root_path()
resourcefilepath = root / "resources"


class ConsumablesCosting(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 12, 31)
        self.pop_size = 100  # <- recommended population size for the runs
        self.number_of_draws = 2  # <- one scenario
        self.runs_per_draw = 1  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'impact_of_consumables_availability',
            'directory': root / 'outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
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
