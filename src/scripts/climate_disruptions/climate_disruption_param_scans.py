from typing import Dict
import json

import numpy as np
import pandas as pd
from scipy.stats import qmc

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import (
    ImprovedHealthSystemAndCareSeekingScenarioSwitcher,
)
from tlo.scenario import BaseScenario


YEAR_OF_CHANGE = 2025
n_samples = 200


# Latin Hypercube
param_ranges = {
    "scale_factor_prob_disruption": [0.0, 2.0],
    "delay_in_seeking_care_weather": [0, 60],
    "scale_factor_reseeking_healthcare_post_disruption": [0.0, 2.0],
    "scale_factor_appointment_urgency": [0.0, 2.0],
    "scale_factor_severity_disruption_and_delay": [0, 2.0],
    #"prop_supply_side_disruptions": [0.0, 1.0],
}

sampler = qmc.LatinHypercube(d=len(param_ranges), seed=0)
sample = sampler.random(n=n_samples)

param_names = list(param_ranges.keys())
l_bounds = [param_ranges[p][0] for p in param_names]
u_bounds = [param_ranges[p][1] for p in param_names]

scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
lhs_df = pd.DataFrame(scaled_sample, columns=param_names)

# Fixed parameters
lhs_df["mode_appt_constraints"] = 1
lhs_df["mode_appt_constraints_postSwitch"] = 1
lhs_df["cons_availability"] = "default"
lhs_df["cons_availability_postSwitch"] = "default"
lhs_df["year_cons_availability_switch"] = YEAR_OF_CHANGE
lhs_df["beds_availability"] = "default"
lhs_df["equip_availability"] = "default"
lhs_df["equip_availability_postSwitch"] = "default"
lhs_df["year_equip_availability_switch"] = YEAR_OF_CHANGE
lhs_df["use_funded_or_actual_staffing"] = "actual"
lhs_df["scale_to_effective_capabilities"] = False
lhs_df["policy_name"] = "Naive"
lhs_df["climate_ssp"] = "ssp245"
lhs_df["climate_model_ensemble_model"] = "mean"
lhs_df["services_affected_precip"] = "all"
lhs_df["year_effective_climate_disruptions"] = 2025
lhs_df["tclose_overwrite"] = 1000
# for mode 1
lhs_df["prop_supply_side_disruptions"] = 0 # does not matter for mode 1


parameter_grid = lhs_df.to_dict(orient="records")

# Save draws for reproducibility
with open("lhs_parameter_draws.json", "w") as f:
    json.dump(parameter_grid, f, indent=2)

class ClimateDisruptionScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2041, 1, 12)
        self.pop_size = 100_000
        self.runs_per_draw = 5
        self.YEAR_OF_CHANGE = YEAR_OF_CHANGE
        self._parameter_grid = parameter_grid
        self.number_of_draws = len(self._parameter_grid)

    def log_configuration(self):
        return {
            "filename": "climate_scenario_runs_lhs_param_scan",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.population": logging.INFO,
            },
        }

    def modules(self):
        return fullmodel()

    def draw_parameters(self, draw_number, rng):
        return self._parameter_grid[draw_number]



if __name__ == "__main__":
    from tlo.cli import scenario_run
    scenario_run([__file__])
