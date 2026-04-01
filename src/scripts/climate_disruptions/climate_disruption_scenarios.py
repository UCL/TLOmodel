import json
from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

YEAR_OF_CHANGE = 2025

# ── BASE PARAMETER SETS ───────────────────────────────────────────────────────

no_disruption_params = {
    "HealthSystem": {
        "scale_factor_reseeking_healthcare_post_disruption": 1.0,
        "scale_factor_prob_disruption": 1.0,
        "delay_in_seeking_care_weather": 28.0,
        "scale_factor_appointment_urgency": 1.0,
        "scale_factor_severity_disruption_and_delay": 1.0,
        "mode_appt_constraints": 1,
        "mode_appt_constraints_postSwitch": 2,
        "year_mode_switch": YEAR_OF_CHANGE,
        "cons_availability": "default",
        "cons_availability_postSwitch": "default",
        "year_cons_availability_switch": YEAR_OF_CHANGE,
        "beds_availability": "default",
        "equip_availability": "default",
        "equip_availability_postSwitch": "default",
        "year_equip_availability_switch": YEAR_OF_CHANGE,
        "use_funded_or_actual_staffing": "actual",
        "scale_to_effective_capabilities": True,
        "policy_name": "Naive",
        "year_effective_climate_disruptions": 2025,
        "prop_supply_side_disruptions": 0.5,
        "services_affected_precip": "none",
        "tclose_overwrite": 1000,
    },
    "SymptomManager": {"spurious_symptoms": True},
}

baseline_params = no_disruption_params.copy()
baseline_params["HealthSystem"] = no_disruption_params["HealthSystem"].copy()
baseline_params["HealthSystem"]["services_affected_precip"] = "all"

worst_case_params = no_disruption_params.copy()
worst_case_params["HealthSystem"] = no_disruption_params["HealthSystem"].copy()
worst_case_params["HealthSystem"].update({
    "scale_factor_reseeking_healthcare_post_disruption": 0.5,
    "scale_factor_prob_disruption": 2.0,
    "delay_in_seeking_care_weather": 60.0,
    "scale_factor_appointment_urgency": 2.0,
    "scale_factor_severity_disruption_and_delay": 2.0,
    "services_affected_precip": "all",
})

# ── SSP126 LOWEST  ───────────────────────────────────────────

ssp126_lowest_baseline = baseline_params.copy()
ssp126_lowest_baseline["HealthSystem"] = baseline_params["HealthSystem"].copy()
ssp126_lowest_baseline["HealthSystem"]["climate_ssp"] = "ssp126"
ssp126_lowest_baseline["HealthSystem"]["climate_model_ensemble_model"] = "lowest"

ssp126_lowest_worst_case = worst_case_params.copy()
ssp126_lowest_worst_case["HealthSystem"] = worst_case_params["HealthSystem"].copy()
ssp126_lowest_worst_case["HealthSystem"]["climate_ssp"] = "ssp126"
ssp126_lowest_worst_case["HealthSystem"]["climate_model_ensemble_model"] = "lowest"

# ── SSP585 LOWEST  ───────────────────────────────────────────

ssp585_lowest_baseline = baseline_params.copy()
ssp585_lowest_baseline["HealthSystem"] = baseline_params["HealthSystem"].copy()
ssp585_lowest_baseline["HealthSystem"]["climate_ssp"] = "ssp585"
ssp585_lowest_baseline["HealthSystem"]["climate_model_ensemble_model"] = "lowest"

ssp585_lowest_worst_case = worst_case_params.copy()
ssp585_lowest_worst_case["HealthSystem"] = worst_case_params["HealthSystem"].copy()
ssp585_lowest_worst_case["HealthSystem"]["climate_ssp"] = "ssp585"
ssp585_lowest_worst_case["HealthSystem"]["climate_model_ensemble_model"] = "lowest"

# ── SSP245 — ALREADY HAVE ──────────────────────

# ── SSP585 HIGHEST  ─────────────────────────────

ssp585_highest_baseline = baseline_params.copy()
ssp585_highest_baseline["HealthSystem"] = baseline_params["HealthSystem"].copy()
ssp585_highest_baseline["HealthSystem"]["climate_ssp"] = "ssp585"
ssp585_highest_baseline["HealthSystem"]["climate_model_ensemble_model"] = "highest"

ssp585_highest_worst_case = worst_case_params.copy()
ssp585_highest_worst_case["HealthSystem"] = worst_case_params["HealthSystem"].copy()
ssp585_highest_worst_case["HealthSystem"]["climate_ssp"] = "ssp585"
ssp585_highest_worst_case["HealthSystem"]["climate_model_ensemble_model"] = "highest"

# ── SSP126 HIGHEST  ─────────────────────────────────────────

ssp126_highest_baseline = baseline_params.copy()
ssp126_highest_baseline["HealthSystem"] = baseline_params["HealthSystem"].copy()
ssp126_highest_baseline["HealthSystem"]["climate_ssp"] = "ssp126"
ssp126_highest_baseline["HealthSystem"]["climate_model_ensemble_model"] = "highest"

ssp126_highest_worst_case = worst_case_params.copy()
ssp126_highest_worst_case["HealthSystem"] = worst_case_params["HealthSystem"].copy()
ssp126_highest_worst_case["HealthSystem"]["climate_ssp"] = "ssp126"
ssp126_highest_worst_case["HealthSystem"]["climate_model_ensemble_model"] = "highest"

# ── DRAW ORDER ────────────────────────────────────────────────────────────────
# draw 0:  ssp126 lowest   — baseline
# draw 1:  ssp126 lowest   — worst case
# draw 2:  ssp585 lowest   — baseline
# draw 3:  ssp585 lowest   — worst case
# draw 4:  ssp585 highest  — baseline
# draw 5:  ssp585 highest  — worst case
# draw 6:  ssp126 highest  — baseline
# draw 7:  ssp126 highest  — worst case

full_grid = [
    ssp126_lowest_baseline,
    ssp126_lowest_worst_case,
    ssp585_lowest_baseline,
    ssp585_lowest_worst_case,
    ssp585_highest_baseline,
    ssp585_highest_worst_case,
    ssp126_highest_baseline,
    ssp126_highest_worst_case,
]


class ClimateDisruptionScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2041, 1, 1)
        self.pop_size = 100_000
        self.runs_per_draw = 5
        self._parameter_grid = full_grid
        self.number_of_draws = len(self._parameter_grid)

        with open("selected_parameter_combinations_climate_sa.json", "w") as f:
            json.dump(self._parameter_grid, f, indent=2)

    def log_configuration(self):
        return {
            "filename": "climate_sensitivity_analysis",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.WARNING,
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
