import numpy as np

from tlo import Date, logging
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2010, 6, 1)
        self.pop_size = 100
        self.number_of_draws = 10
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            "filename": "test_scenario",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.INFO,
            },
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                disable=True,
                service_availability=["*"],
            ),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(
                resourcefilepath=self.resources
            ),
            contraception.Contraception(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            "Lifestyle": {
                "init_p_urban": rng.randint(10, 20) / 100.0,
                "init_p_high_sugar": 0.52,
            },
            "Labour": {
                "intercept_parity_lr2010": -10 * rng.exponential(0.1),
                "effect_age_parity_lr2010": np.arange(0.1, 1.1, 0.1)[draw_number],
            },
        }


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
