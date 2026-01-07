"""This Scenario file generates individual histories for the purpose of generating datasets for emulator training

Run on the batch system using:
```
tlo batch-submit
    src/scripts/analysis_data_generation/scenario_data_generation_for_emulator.py
```

or locally using:
```
    tlo scenario-run src/scripts/analysis_data_generation/scenario_data_generation_for_emulator.py
```

"""
from pathlib import Path
from typing import Dict

import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods import individual_history_tracker
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario, make_cartesian_parameter_grid


module_of_interest ='CervicalCancer'
N_service_scenarios = 10

def sample_service_availability:
        module_name = module_of_interest
        treatments = get_filtered_treatment_ids(depth=2)
        treatments_in_module = [item for item in treatments if module_name in item]
        
        service_availability = {}
        for i in [0,N_service_scenarios]:
            selected_treatments = [x for x in treatments_in_module if random.random() < 0.5]
            service_availability = {i:selected_treatments}
            
        return service_availability

# TO DO: need to create custom make_cartesian_parameter_grid that combines parameter combos with treatments and consumables
full_grid = make_cartesian_parameter_grid(
    {
        "module_of_interest": {
            #1) # Should iterate over all parameters labelled as "free", as span them
            # over prior
            #2) Should combine with Service configuration, i.e. include N random combinations of services being included/excluded
            #3) Should combine with different levels of relevant consumable availability
            
        
            """
            "scale_factor_delay_in_seeking_care_weather": [float(28)],
            "rescaling_prob_seeking_after_disruption": [float(1)],
            "rescaling_prob_disruption": [float(1)],
            "scale_factor_severity_disruption_and_delay": [float(1)],
            "mode_appt_constraints": [1],
            "mode_appt_constraints_postSwitch": [2],
            "cons_availability": ["default"],
            "cons_availability_postSwitch": ["default"],
            "year_cons_availability_switch": [YEAR_OF_CHANGE],
            "beds_availability": ["default"],
            "equip_availability": ["default"],
            "equip_availability_postSwitch": ["default"],
            "year_equip_availability_switch": [YEAR_OF_CHANGE],
            "use_funded_or_actual_staffing": ["actual"],
            "scale_to_effective_capabilities": [True],
            "policy_name": ["Naive"],
            "climate_ssp": ["ssp126", "ssp245", "ssp585"],
            "year_effective_climate_disruptions": [2025],
            "climate_model_ensemble_model": ["lowest", "mean", "highest"],
            "services_affected_precip": ["all"],
            "tclose_overwrite": [1000],
            "prop_supply_side_disruptions": [0.5],
            """
        }
    }
)



class TrackIndividualHistories(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 42
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(months=5)
        self.pop_size = 100
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 1
        self.generate_event_chains = True

    def log_configuration(self):
        return {
            'filename': 'data_generation_for_emulator',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.events': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.individual_history': logging.INFO
            }
        }

    def modules(self):
        return (
            fullmodel() + [individual_history_tracker.IndividualHistoryTracker()]
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    def _get_scenarios(self) -> Dict[str, Dict]:

        return {
            "Baseline":
                mix_scenarios(
                    self._baseline(),
                    {
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
                }
            },
        )

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
