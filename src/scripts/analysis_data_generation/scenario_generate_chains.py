"""This Scenario file run the model to generate event chans

Run on the batch system using:
```
tlo batch-submit 
    src/scripts/analysis_data_generation/scenario_generate_chains.py
```

or locally using:
```
    tlo scenario-run src/scripts/analysis_data_generation/scenario_generate_chains.py
```

"""
from pathlib import Path
from typing import Dict

import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios, get_filtered_treatment_ids
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario
from tlo.methods import (
    alri,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    rti,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    stunting,
    symptommanager,
    tb,
    wasting,
)

class GenerateDataChains(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(months=13)
        self.pop_size = 1000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 50
        self.generate_event_chains = True

    def log_configuration(self):
        return {
            'filename': 'generate_event_chains',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.events': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        # MODIFY
        # Here instead of running full module
        return [demography.Demography(resourcefilepath=self.resources),
                enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
                healthburden.HealthBurden(resourcefilepath=self.resources),
                symptommanager.SymptomManager(resourcefilepath=self.resources, spurious_symptoms=False),
                rti.RTI(resourcefilepath=self.resources),
                healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
                #simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                healthsystem.HealthSystem(resourcefilepath=self.resources,
                                          mode_appt_constraints=1,
                                          cons_availability='all')]
                                          
       # return (
       #     fullmodel(resourcefilepath=self.resources)
       #     + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)]
       # )
    """
    def draw_parameters(self, draw_number, rng):
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                'HealthSystem': {
                    'Service_Availability': list(self._scenarios.values())[draw_number],
                },
            }
        )

    def _get_scenarios(self) -> Dict[str, list[str]]:
        Return the Dict with values for the parameter `Service_Availability` keyed by a name for the scenario.
        The sequences of scenarios systematically omits one of the TREATMENT_ID's that is defined in the model.

        # Generate list of TREATMENT_IDs and filter to the resolution needed
        treatments = get_filtered_treatment_ids(depth=2)
        treatments_RTI = [item for item in treatments if 'Rti' in item]
        
        # Return 'Service_Availability' values, with scenarios for everything, nothing, and ones for which each
        # treatment is omitted
        service_availability = dict({"Everything": ["*", "Nothing": []})
        #service_availability.update(
        #    {f"No {t.replace('_*', '*')}": [x for x in treatments if x != t] for t in treatments_RTI}
        #)
        
        return service_availability

    """
    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    # case 1: gfHE = -0.030, factor = 1.01074
    # case 2: gfHE = -0.020, factor = 1.02116
    # case 3: gfHE = -0.015, factor = 1.02637
    # case 4: gfHE =  0.015, factor = 1.05763
    # case 5: gfHE =  0.020, factor = 1.06284
    # case 6: gfHE =  0.030, factor = 1.07326

    def _get_scenarios(self) -> Dict[str, Dict]:
        #Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        
        treatments = get_filtered_treatment_ids(depth=2)
        treatments_RTI = [item for item in treatments if 'Rti' in item]
        
        # Return 'Service_Availability' values, with scenarios for everything, nothing, and ones for which each
        # treatment is omitted
        service_availability = dict({"Everything": ["*"], "Nothing": []})
        service_availability.update(
            {f"No {t.replace('_*', '*')}": [x for x in treatments if x != t] for t in treatments_RTI}
        )
        print(service_availability.keys())

        return {
            # =========== STATUS QUO ============
            "Baseline":
                mix_scenarios(
                    self._baseline(),
                    {
                     "HealthSystem": {
                            "Service_Availability": service_availability["No Rti_BurnManagement*"],
                      },
                    }
                ),

        }
        
    def _baseline(self) -> Dict:
        #Return the Dict with values for the parameter changes that define the baseline scenario.
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "HealthSystem": {
                    "mode_appt_constraints": 1,                 # <-- Mode 1 prior to change to preserve calibration
                    "cons_availability": "all",
                }
            },
        )

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
