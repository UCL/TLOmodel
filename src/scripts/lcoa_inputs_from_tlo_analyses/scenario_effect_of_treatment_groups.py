############################################################################
# 1. allow historical HRH scaling to occur 2018-2024 by setting 'yearly_HR_scaling_mode': 'historical_scaling'
# We don't care about future HRH scaling because constraints are derived from the state of play in 2025.
# We will run everything in mode 1; HR will be available regardless;
# We need the productivity factor; rescaling is generally used in mode 2 BUT
# Rescaling factors are perhaps calculated at the time of mode switch; so we might need to detach that logic.

# 2. Modify 'Service_Availability' to exclude pre-defined groups of TREATMENT_ID at a time.
# 3. HealthSystem should update 'Service_Availability' in 2025.
# Use HealthSystemChangeParameters for this? So define a new parameter for year of change
# and then use the same logic as for switching mode, or  consumables availability switch?
# Note that this requires is to update HealthSystemChangeParameters as well.
# Note that at the time of switching service availability, HSI Event Queue needs
# to be emptied of events that require services no longer available.

# 4. Turn on the following loggers: healthsystem.summary, demography, healthburden

## Questions:
# 1. is it ok to reset rescaling factors
############################################################################






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


class EffectOfEachTreatmentGroup(BaseScenario):
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
            'filename': 'effect_of_each_treatment_group_status_quo',
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
        treatment_groups = {
          'MATERNAL_CARE': ['AntenatalCare_FollowUp', 'AntenatalCare_Inpatient'],
          'CARDIAC_TREATMENTS': ['CardioMetabolicDisorders_Investigation', 'CardioMetabolicDisorders_Treatment'],
        }

        # Return 'Service_Availability' values, with scenarios for everything, nothing, and ones for which each
        # treatment is omitted
        service_availability = dict({"Everything": ["*"], "Nothing": []})
        service_availability.update(
           {f"No {group_name}": [x for x in treatments if x not in group_treatments]
           for group_name, group_treatments in treatment_groups.items()}
        )

        return service_availability


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
