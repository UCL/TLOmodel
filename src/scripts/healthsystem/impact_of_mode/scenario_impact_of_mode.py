"""This Scenario file run the model under different assumptions for the HealthSystem Mode in order to estimate the
impact that is achieved under each (relative to there being no health system).

Run on the batch system using:
```
tlo batch-submit src/scripts/healthsystem/impact_of_mode/scenario_impact_of_mode.py
```

or locally using:
```
tlo scenario-run src/scripts/healthsystem/impact_of_mode/scenario_impact_of_mode.py
```

"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class ImpactOfHealthSystemMode(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2014, 12, 31)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'effect_of_each_treatment',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return list(self._scenarios.values())[0]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        0. "No Healthcare System"

        1. "Mode 0":
            Mode 0 & Actual HR funding

        2. "Mode 1":
            Mode 1 & Actual HR funding
            Default consumables,

        3. "Mode 2":
            Mode 2 & Actual HR funding
            100% availability of consumables,

        """

        return {
            "Mode 2": {
                'HealthSystem': {
                    'mode_appt_constraints': 2,
                    "use_funded_or_actual_staffing": "actual",
                 },
             },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
