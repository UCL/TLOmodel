"""
This scenario runs the full model under a set of scenario in which each one TREATMENT_ID is excluded.

Run on the batch system using:
```tlo batch-submit src/scripts/healthsystem/impact_of_cons_availability/scenario_effect_of_each_treatment.py```

or locally using:
    ```tlo scenario-run src/scripts/healthsystem/impact_of_cons_availability/scenario_effect_of_each_treatment.py```

"""
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EffectOfEachTreatment(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2010, 1, 31)
        self.pop_size = 1_000
        self.number_of_draws = len(self._scenarios())
        self.runs_per_draw = 3  # <- repeated this many times (per draw)

    def log_configuration(self):
        return {
            'filename': 'effect_of_each_treatment',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        _, service_availability = list(self._scenarios(draw_number=draw_number))[draw_number]

        return {
            'HealthSystem': {
                'Service_Availability': service_availability
                }
        }

    @property
    def _num_scenarios(self):
        return len(self.__scenarios())

    def _scenarios(self) -> Dict[str, List[str]]:
        """Return the Dict with values for the parameter `Service_Availability` that define the scenarios, keyed
        by a name for the scenario.
        The sequences of scenarios systematically omits one of the TREATMENT_ID's that is defined in the model."""

        # Generate table of defined HSI
        tempfile_output_location = self.log_configuration()['directory'] / 'defined_hsi.csv'
        os.system(f'python docs/tlo_hsi_events.py --output-file {tempfile_output_location} --output-format csv')
        defined_hsi = pd.read_csv(tempfile_output_location)

        # Generate list of TREATMENT_IDs
        treatments = list(set(defined_hsi['Treatment']))

        # Filter list to find the TREATMENT_IDs defined up to the first '_' (and replacing with '*')
        # todo...

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
