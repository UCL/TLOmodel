from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.methods.weather_disruptions import WeatherDisruptions
from tlo.scenario import BaseScenario

YEAR_OF_CHANGE = 2025

baseline_params = {
    "WeatherDisruptions": {
        "climate_ssp": "ssp245",
        "climate_model_ensemble_model": "mean",
        "year_effective_climate_disruptions": 2025,
        "services_affected_precip": "none",
        "delay_in_seeking_care_weather": 28.0,
        "scale_factor_reseeking_healthcare_post_disruption": 1.0,
        "scale_factor_prob_disruption": 1.0,
        "scale_factor_severity_disruption_and_delay": 1.0,
        "prop_supply_side_disruptions": 0.3,
    },
    "HealthSystem": {
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
        "tclose_overwrite": 1000,
    },
    "SymptomManager": {
        "spurious_symptoms": True,
    },
}


class ClimateDisruptionScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2041, 1, 1)
        self.pop_size = 1000
        self.runs_per_draw = 1
        self._parameter_grid = baseline_params
        self.number_of_draws = 1  # len(self._parameter_grid)

    def log_configuration(self):
        return {
            "filename": "climate_disruption_scenario",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.WARNING,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.weather_disruptions.summary": logging.INFO,
                "tlo.methods.population": logging.INFO,
            },
        }

    def modules(self):
        return fullmodel()

    def draw_parameters(self, draw_number, rng):
        return self._parameter_grid  # [draw_number]


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
