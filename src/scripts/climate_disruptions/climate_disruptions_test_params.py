from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario, make_cartesian_parameter_grid

YEAR_OF_CHANGE = 2025

no_disruption_params = {  # no disruptions at all (scale_factor_prob_disruption=0)
    "HealthSystem": {
        "scale_factor_reseeking_healthcare_post_disruption": 1.0,
        "scale_factor_prob_disruption": 0.0,
        "delay_in_seeking_care_weather": 28.0,
        "scale_factor_appointment_urgency": 1.0,
        "scale_factor_severity_disruption_and_delay": 1.0,
        "mode_appt_constraints": 1,
        "mode_appt_constraints_postSwitch": 2,
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

demand_side_disruption_params = no_disruption_params.copy()  # all disruptions are demand-side (prop_supply_side=0)
demand_side_disruption_params["HealthSystem"] = no_disruption_params["HealthSystem"].copy()
demand_side_disruption_params["HealthSystem"]["scale_factor_prob_disruption"] = 1.0
demand_side_disruption_params["HealthSystem"]["prop_supply_side_disruptions"] = 0.0

mixed_disruption_params = no_disruption_params.copy()  # disruptions split 50/50 supply- and demand-side (prop_supply_side=0.5)
mixed_disruption_params["HealthSystem"] = no_disruption_params["HealthSystem"].copy()
mixed_disruption_params["HealthSystem"]["scale_factor_prob_disruption"] = 1.0
mixed_disruption_params["HealthSystem"]["prop_supply_side_disruptions"] = 0.5

supply_side_disruption_params = no_disruption_params.copy()  # all disruptions are supply-side (prop_supply_side=1)
supply_side_disruption_params["HealthSystem"] = no_disruption_params["HealthSystem"].copy()
supply_side_disruption_params["HealthSystem"]["scale_factor_prob_disruption"] = 1.0
supply_side_disruption_params["HealthSystem"]["prop_supply_side_disruptions"] = 1.0

full_grid = [
    no_disruption_params,
    demand_side_disruption_params,
    mixed_disruption_params,
    supply_side_disruption_params,
]

class ClimateDisruptionScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2027, 1, 1)
        self.pop_size = 10_000
        self.runs_per_draw = 1
        self._parameter_grid = full_grid
        self.number_of_draws = len(self._parameter_grid)

    def log_configuration(self):
        return {
            "filename": "test_disruption_parameters_mode_2",
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
