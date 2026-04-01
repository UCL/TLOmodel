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
        "climate_ssp": "ssp245",
        "climate_model_ensemble_model": "mean",
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

# ── BASELINE — prop_supply_side_disruptions scan ──────────────────────────────

baseline_supply_0_1 = baseline_params.copy()
baseline_supply_0_1["HealthSystem"] = baseline_params["HealthSystem"].copy()
baseline_supply_0_1["HealthSystem"]["prop_supply_side_disruptions"] = 0.1

baseline_supply_0_5 = baseline_params.copy()
baseline_supply_0_5["HealthSystem"] = baseline_params["HealthSystem"].copy()
baseline_supply_0_5["HealthSystem"]["prop_supply_side_disruptions"] = 0.5

baseline_supply_0_9 = baseline_params.copy()
baseline_supply_0_9["HealthSystem"] = baseline_params["HealthSystem"].copy()
baseline_supply_0_9["HealthSystem"]["prop_supply_side_disruptions"] = 0.9

# ── WORST CASE — prop_supply_side_disruptions scan ────────────────────────────

worst_case_supply_0_1 = worst_case_params.copy()
worst_case_supply_0_1["HealthSystem"] = worst_case_params["HealthSystem"].copy()
worst_case_supply_0_1["HealthSystem"]["prop_supply_side_disruptions"] = 0.1

worst_case_supply_0_5 = worst_case_params.copy()
worst_case_supply_0_5["HealthSystem"] = worst_case_params["HealthSystem"].copy()
worst_case_supply_0_5["HealthSystem"]["prop_supply_side_disruptions"] = 0.5

worst_case_supply_0_9 = worst_case_params.copy()
worst_case_supply_0_9["HealthSystem"] = worst_case_params["HealthSystem"].copy()
worst_case_supply_0_9["HealthSystem"]["prop_supply_side_disruptions"] = 0.9

# ── DRAW ORDER ────────────────────────────────────────────────────────────────
# draw 0:  baseline   — prop_supply_side_disruptions = 0.1
# draw 1:  baseline   — prop_supply_side_disruptions = 0.5
# draw 2:  baseline   — prop_supply_side_disruptions = 0.9
# draw 3:  worst case — prop_supply_side_disruptions = 0.1
# draw 4:  worst case — prop_supply_side_disruptions = 0.5
# draw 5:  worst case — prop_supply_side_disruptions = 0.9

full_grid = [
    baseline_supply_0_1,
    baseline_supply_0_5,
    baseline_supply_0_9,
    worst_case_supply_0_1,
    worst_case_supply_0_5,
    worst_case_supply_0_9,
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

        with open("selected_parameter_combinations_supply_side_scan.json", "w") as f:
            json.dump(self._parameter_grid, f, indent=2)

    def log_configuration(self):
        return {
            "filename": "supply_side_disruption_scan",
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
