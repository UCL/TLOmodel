from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario, make_cartesian_parameter_grid

YEAR_OF_CHANGE = 2025

no_disruption_params = {  # no disruptions at all (scale_factor_prob_disruption=0)
    "HealthSystem": {
        "mode_appt_constraints": 1,
        "mode_appt_constraints_postSwitch": 2,
        "cons_availability": "default",
        "cons_availability_postSwitch": "all",
        "year_cons_availability_switch": YEAR_OF_CHANGE,
        "beds_availability": "default",
        "equip_availability": "default",
        "equip_availability_postSwitch": "default",
        "year_equip_availability_switch": YEAR_OF_CHANGE,
        "use_funded_or_actual_staffing": "actual",
        "scale_to_effective_capabilities": False,
        "policy_name": "Naive",
        "tclose_overwrite": 1000,
    },
    "SymptomManager": {
        "spurious_symptoms": True,
    },
}

full_grid = [
    no_disruption_params,

]


class TestLoggingScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2028, 1, 1)
        self.pop_size = 10_000
        self.runs_per_draw = 1
        self._parameter_grid = full_grid
        self.number_of_draws = 1

    def log_configuration(self):
        return {
            "filename": "test_logging_mode_2",
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
        return self._parameter_grid[0]


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
