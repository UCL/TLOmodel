"""
This scenario runs the full model under a set of scenario in which each one TREATMENT_ID is excluded.

Run on the batch system using:
```tlo batch-submit src/scripts/healthsystem/finding_effects_of_each_treatment/scenario_effect_of_each_treatment.py```

or locally using:
    ```tlo scenario-run src/scripts/healthsystem/finding_effects_of_each_treatment/scenario_effect_of_each_treatment.py
    ```

"""
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EffectOfEachTreatment(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2019, 12, 31)
        self.pop_size = 20_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 3  # <- repeated this many times (per draw)

    def log_configuration(self):
        return {
            'filename': 'effect_of_each_treatment',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
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
                'Service_Availability': list(self._scenarios.values())[draw_number]
                }
        }

    def _get_scenarios(self) -> Dict[str, List[str]]:
        """Return the Dict with values for the parameter `Service_Availability` keyed by a name for the scenario.
        The sequences of scenarios systematically omits one of the TREATMENT_ID's that is defined in the model. The
        complete list of TREATMENT_ID's is found by running `tlo_hsi_events.py`."""

        # Generate table of defined HSI
        tempfile_output_location = self.log_configuration()['directory'] / 'defined_hsi.csv'
        os.system(f'python docs/tlo_hsi_events.py --output-file {tempfile_output_location} --output-format csv')
        # todo - Could do some refactoring on `tlo_hsi_events.py` to enable this to be returned directly.
        defined_hsi = pd.read_csv(tempfile_output_location)

        # Generate list of TREATMENT_IDs
        treatments = sorted(list(set(defined_hsi['Treatment'])))

        # [OPTIONALLY] Filter/aggregate the TREATMENT_IDs to provide the resolution needed.

        # Add trailing '*'
        treatments = [t + '*' for t in treatments]

        # Return 'Service_Availability' values, in which one treatment is omitted, plus a scenario with everything
        service_availability = dict({"Everything": ["*"]})
        service_availability.update(
            {f"No {t}": [x for x in treatments if x != t] for t in treatments}
        )

        return service_availability


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
