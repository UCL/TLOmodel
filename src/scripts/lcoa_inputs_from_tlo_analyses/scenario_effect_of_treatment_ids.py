"""
This files runs the full model under a set of scenario in which only a single TREATMENT_ID is included.

To check scenarios are generated correctly:
```
tlo scenario-run --draw-only src/scripts/lcoa_inputs_from_tlo_analyses/scenario_effect_of_treatment_ids.py
```

Run on the batch system using:

```
tlo batch-submit src/scripts/lcoa_inputs_from_tlo_analyses/scenario_effect_of_treatment_ids.py
```

or locally using:
```
tlo scenario-run src/scripts/lcoa_inputs_from_tlo_analyses/scenario_effect_of_treatment_ids.py
```

"""
from pathlib import Path
from typing import Dict, List

from scripts.lcoa_inputs_from_tlo_analyses.scenarios_definitions import (
    ScenarioDefinitions,
)

from tlo import Date, logging
from tlo.analysis.utils import (
    get_filtered_treatment_ids,
    mix_scenarios,
)
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EffectOfEachTreatment(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2040, 1, 1)
        self.pop_size = 100
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'effect_of_each_treatment_id',
            'directory': Path('./outputs'),
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return (
            fullmodel(resourcefilepath=self.resources)
            + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)]
        )

    def draw_parameters(self, draw_number, rng):
        scenario_definitions = ScenarioDefinitions()
        return mix_scenarios(
            scenario_definitions.baseline(),
            {
                'HealthSystem': {
                    'service_availability_postSwitch': list(self._scenarios.values())[draw_number],
                },
            }
        )

    def _get_scenarios(self) -> Dict[str, List[str]]:
        """Return the Dict with values for the parameter `Service_Availability` keyed by a name for the scenario.
        The sequences of scenarios systematically omits all but one TREATMENT_ID that is defined in the model."""

        # Generate list of TREATMENT_IDs and filter to the resolution needed
        treatments = get_filtered_treatment_ids(depth=None)
        # Return 'Service_Availability' values, with scenarios for nothing, and ones for which all but one
        # treatment is omitted
        service_availability = dict({"Nothing": []})
        # For each treatment group, create scenarios keeping only one treatment from that group
        service_availability.update(
            {f"Only {treatment}": [treatment] for treatment in treatments}
        )

        return service_availability


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
