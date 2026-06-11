from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

YEAR_OF_CHANGE = 2025

# Draw 0: STATUS QUO — realistic constrained health system, no climate disruption
status_quo_params = {
    "HealthSystem": {
        "mode_appt_constraints": 1,
        "mode_appt_constraints_postSwitch": 1,
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
        "services_affected_precip": "none",  # no climate disruption
    },
    "SymptomManager": {
        "spurious_symptoms": True,
    },
}

# Draw 1: PERFECT WORLD — no access constraints, no climate disruption.
# Every scheduled HSI runs (consumables/beds/equipment always available,
# capabilities unconstrained). This is the "demand" denominator.
perfect_world_params = status_quo_params.copy()
perfect_world_params["HealthSystem"] = status_quo_params["HealthSystem"].copy()
perfect_world_params["HealthSystem"].update({
    "mode_appt_constraints": 1,  # elastic: all HSIs run if officers have any capability
    "mode_appt_constraints_postSwitch": 1,
    "cons_availability": "all",
    "cons_availability_postSwitch": "all",
    "beds_availability": "all",
    "equip_availability": "all",
    "equip_availability_postSwitch": "all",
    "use_funded_or_actual_staffing": "funded_plus",  # max staffing distribution
    "services_affected_precip": "none",  # still no climate disruption
})

full_grid = [status_quo_params, perfect_world_params]


class PerfectWorldScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 1
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2041, 1, 1)
        self.pop_size = 100_000
        self.runs_per_draw = 5
        self._parameter_grid = full_grid
        self.number_of_draws = len(self._parameter_grid)

    def log_configuration(self):
        return {
            "filename": "epi_status_quo_vs_perfect_world",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.WARNING,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.population": logging.INFO,
                "tlo.methods.epi": logging.INFO,  # vaccine coverage
            },
        }

    def modules(self):
        return fullmodel()

    def draw_parameters(self, draw_number, rng):
        return self._parameter_grid[draw_number]


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
