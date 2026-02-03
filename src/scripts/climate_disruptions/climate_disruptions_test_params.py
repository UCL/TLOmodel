
from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario, make_cartesian_parameter_grid

YEAR_OF_CHANGE = 2025

# Define the three scenarios explicitly
no_disruptional_params = { # no disruptions
    "HealthSystem": {
        "scale_factor_reseeking_healthcare_post_disruption": 1.0,
        "scale_factor_prob_disruption": 0.0,
        "delay_in_seeking_care_weather": 28.0,
        "scale_factor_appointment_urgency": 1.0,
        "scale_factor_severity_disruption_and_delay": 1.0,
        "mode_appt_constraints": 1,
        "mode_appt_constraints_postSwitch": 1,
        "cons_availability": "default",
        "cons_availability_postSwitch": "default",
        "year_cons_availability_switch": YEAR_OF_CHANGE,
        "beds_availability": "default",
        "equip_availability": "default",
        "equip_availability_postSwitch": "default",
        "year_equip_availability_switch": YEAR_OF_CHANGE,
        "use_funded_or_actual_staffing": "actual",
        "scale_to_effective_capabilities": False,
        "policy_name": "Naive",
        "climate_ssp": "ssp245",
        "climate_model_ensemble_model": "mean",
        "year_effective_climate_disruptions": 2025,
        "prop_supply_side_disruptions": 0.5,
        "services_affected_precip": "all",
        "tclose_overwrite": 1000,
    },
    "SymptomManager": {
        "spurious_symptoms": True,
    },
}

no_reseeking_param = no_disruptional_params.copy() # never reseek healthcare, lots of cancellations
no_reseeking_param["HealthSystem"] = no_disruptional_params["HealthSystem"].copy()
no_reseeking_param["HealthSystem"]["scale_factor_prob_disruption"] = 1.0
no_reseeking_param["HealthSystem"]["scale_factor_reseeking_healthcare_post_disruption"] = 0.0


always_disrupted_param = no_disruptional_params.copy() # expect to always see delays, as this will then go to 1
always_disrupted_param["HealthSystem"] = no_disruptional_params["HealthSystem"].copy()
always_disrupted_param["HealthSystem"]["scale_factor_prob_disruption"] = 10000.0

huge_delay_param = no_disruptional_params.copy() # expecting to see overall lower HSIs as they are delayed and never re-ran?
huge_delay_param["HealthSystem"] = huge_delay_param["HealthSystem"].copy()
huge_delay_param["HealthSystem"]["scale_factor_prob_disruption"] = 1
huge_delay_param["HealthSystem"]["delay_in_seeking_care_weather"] = 10000

full_grid = [no_disruptional_params, no_reseeking_param, always_disrupted_param, huge_delay_param]

class ClimateDisruptionScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2027, 1, 1)
        self.pop_size = 100_000
        self.runs_per_draw = 5
        self._parameter_grid = full_grid
        self.number_of_draws = len(self._parameter_grid)

        #with open("selected_parameter_combinations_baseline.json", "w") as f:
        #    json.dump(self._parameter_grid, f, indent=2)

    def log_configuration(self):
        return {
            "filename": "test_memory_issue_w_param_grid_2025",
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
