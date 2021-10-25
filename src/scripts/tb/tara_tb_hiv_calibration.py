"""
This file defines a batch run through which the hiv and tb modules are run across a grid of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/tb/tara_tb_hiv_calibration.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/tb/tara_tb_hiv_calibration.py

or execute a single run:
tlo scenario-run src/scripts/tb/tara_tb_hiv_calibration.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/tb/tara_tb_hiv_calibration.py


save Job ID: tara_tb_hiv_calibration-2021-10-06T094337Z


Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download tlo_q1_demo-123

This currently will run multiple simulations of the baseline scenario
with all parameters set to default

"""

from random import randint

from tlo import Date, logging
from tlo.methods import (
    demography,
    simplified_births,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    epi,
    hiv,
    tb,
)

from tlo.scenario import BaseScenario

# define size of parameter lists
# hiv_param_length = 1
# tb_param_length = 1
# number_of_draws = hiv_param_length * tb_param_length


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = randint(0, 5000)

        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 200000
        self.number_of_draws = 1
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            "filename": "baseline_scenario",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.demography": logging.INFO,
            },
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                service_availability=["*"],  # all treatment allowed
                mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
                ignore_cons_constraints=False,
                # mode for consumable constraints (if ignored, all consumables available)
                ignore_priority=False,  # do not use the priority information in HSI event to schedule
                capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
                disable=True,  # disables the healthsystem (no constraints and no logging) and every HSI runs
                disable_and_reject_all=False,  # disable healthsystem and no HSI runs
                store_hsi_events_that_have_run=False,
            ),  # convenience function for debugging
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(
                resourcefilepath=self.resources
            ),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        # grid = self.make_grid(
        #     {
        #         "beta": np.linspace(start=0.02, stop=0.06, num=hiv_param_length),
        #         "transmission_rate": np.linspace(
        #             start=0.005, stop=0.2, num=tb_param_length
        #         ),
        #     }
        # )
        #
        # return {
        #     "Hiv": {
        #         "beta": grid["beta"][draw_number],
        #     },
        #     "Tb": {
        #         "transmission_rate": grid["transmission_rate"][draw_number],
        #     },
        # }
        return


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
