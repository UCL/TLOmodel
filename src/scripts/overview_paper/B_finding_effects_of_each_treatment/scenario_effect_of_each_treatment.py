"""
This scenario runs the full model under a set of scenario in which each one TREATMENT_ID is excluded.

This version of the scenario represents _actual_ healthcare capacity/performance and normal healthcare seeking.

Run on the batch system using:

```
tlo batch-submit
 src/scripts/overview_paper/B_finding_effects_of_each_treatment/scenario_effect_of_each_treatment.py
```

or locally using:
```
tlo scenario-run
 src/scripts/overview_paper/B_finding_effects_of_each_treatment/scenario_effect_of_each_treatment.py
```

"""
from pathlib import Path
from typing import Dict, List

from tlo import Date, logging
from tlo.analysis.utils import (
    get_filtered_treatment_ids,
    get_parameters_for_status_quo,
    mix_scenarios,
)
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EffectOfEachTreatment(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'effect_of_each_treatment_status_quo',
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
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                'HealthSystem': {
                    'Service_Availability': list(self._scenarios.values())[draw_number],
                },
            }
        )

    def _get_scenarios(self) -> Dict[str, List[str]]:
        """Return the Dict with values for the parameter `Service_Availability` keyed by a name for the scenario.
        The sequences of scenarios systematically omits one of the TREATMENT_ID's that is defined in the model."""

        # Generate list of TREATMENT_IDs and filter to the resolution needed
        treatments = get_filtered_treatment_ids(depth=1)

        # Return 'Service_Availability' values, with scenarios for everything, nothing, and ones for which each
        # treatment is omitted
        service_availability = dict({"Everything": ["*"], "Nothing": []})
        service_availability.update(
            {f"No {t.replace('_*', '*')}": [x for x in treatments if x != t] for t in treatments}
        )

        return service_availability


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
