"""
This scenario runs the full model under a set of scenario in which each one TREATMENT_ID is excluded.

* No spurious symptoms
* Appts Contraints: Mode 0 (No Constraints)
* Consumables Availability: All

Run on the batch system using:
```tlo batch-submit src/scripts/healthsystem/finding_effects_of_each_treatment/scenario_effect_of_each_treatment.py```

or locally using:
    ```tlo scenario-run src/scripts/healthsystem/finding_effects_of_each_treatment/scenario_effect_of_each_treatment.py
    ```

"""
import os
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EffectOfEachTreatment(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2014, 12, 31)
        self.pop_size = 50_000
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
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources, healthsystem_mode_appt_constraints=0)

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'Service_Availability': list(self._scenarios.values())[draw_number],
                'cons_availability': 'all',
                }
        }

    def _get_scenarios(self) -> Dict[str, List[str]]:
        """Return the Dict with values for the parameter `Service_Availability` keyed by a name for the scenario.
        The sequences of scenarios systematically omits one of the TREATMENT_ID's that is defined in the model. The
        complete list of TREATMENT_ID's is found by running `tlo_hsi_events.py`."""

        def filter_treatments(_treatments: Iterable[str], depth: int = 1) -> List[str]:
            """Reduce an iterable of `TREATMENT_IDs` by ignoring difference beyond a certain depth of specification.
            The TREATMENT_ID is defined with each increasing level of specification separated by a `_`. """
            return sorted(list(set(
                [
                    "".join(f"{x}_" for i, x in enumerate(t.split('_')) if i < depth).rstrip('_') + '*'
                    for t in set(_treatments)
                ]
            )))

        # Generate table of defined HSI
        tempfile_output_location = self.log_configuration()['directory'] / 'defined_hsi.csv'
        os.system(f'python docs/tlo_hsi_events.py --output-file {tempfile_output_location} --output-format csv')
        defined_hsi = pd.read_csv(tempfile_output_location)
        # todo - Could do some refactoring on `tlo_hsi_events.py` to enable this to be returned directly without saving
        #  to a file.

        # Generate list of TREATMENT_IDs and filter to the resolution needed
        treatments = filter_treatments(defined_hsi['Treatment'], depth=1)

        # Return 'Service_Availability' values, with scenarios for everything, nothing, and ones for which each
        # treatment is omitted
        service_availability = dict({"Everything": ["*"], "Nothing": []})
        service_availability.update(
            {f"No {t}": [x for x in treatments if x != t] for t in treatments}
        )

        return service_availability


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
