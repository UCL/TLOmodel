from typing import Dict
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc


YEAR_OF_CHANGE = 2025
n_samples_total = 1000
n_samples_to_use = 1
LHS_FILE = "/Users/rem76/PycharmProjects/TLOmodel/src/scripts/climate_disruptions/lhs_parameter_draws.json"
start_index = 0
generate = False #this generates the initial n_samples_total
# Latin Hypercube parameter ranges
param_ranges = {
    "scale_factor_prob_disruption": [0.0, 2.0],
    "delay_in_seeking_care_weather": [0, 60],
    "scale_factor_reseeking_healthcare_post_disruption": [0.0, 2.0],
    "scale_factor_appointment_urgency": [0.0, 2.0],
    "scale_factor_severity_disruption_and_delay": [0, 2.0],
}


def generate_lhs_samples(n_samples: int, param_ranges: dict, seed: int = 0) -> pd.DataFrame:
    """Generate Latin Hypercube samples and add fixed parameters."""
    sampler = qmc.LatinHypercube(d=len(param_ranges), seed=seed)
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
    lhs_df["prop_supply_side_disruptions"] = 0

    return lhs_df


def get_parameter_grid(
    lhs_file: str,
    n_samples_total: int,
    n_samples_to_use: int,
    param_ranges: dict,
    start_index: int,
    generate: bool = False
) -> list:
    """Load existing LHS samples or generate new ones, returning a slice."""

    if os.path.exists(lhs_file) and not generate:
        with open(lhs_file, "r") as f:
            all_samples = json.load(f)
    else:
        lhs_df = generate_lhs_samples(n_samples_total, param_ranges, seed=0)
        all_samples = lhs_df.to_dict(orient="records")

        with open(lhs_file, "w") as f:
            json.dump(all_samples, f, indent=2)

    # Return the requested slice
    return all_samples[start_index:start_index + n_samples_to_use]

# Get the parameter grid (will load from file if it exists)
parameter_grid = get_parameter_grid(
    LHS_FILE,
    n_samples_total,
    n_samples_to_use,
    param_ranges,
    start_index,
    generate
)

print(parameter_grid)
