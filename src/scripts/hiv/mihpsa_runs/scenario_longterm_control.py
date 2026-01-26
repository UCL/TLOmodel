"""
This file defines a batch run through which the hiv and tb modules are run across a set of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/mihpsa_runs/scenario_longterm_control.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/mihpsa_runs/scenario_longterm_control.py

or execute a single run:
tlo scenario-run src/scripts/hiv/mihpsa_runs/scenario_longterm_control.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/hiv/mihpsa_runs/scenario_longterm_control.py


Download result files for a completed job:
tlo batch-download calibration_script-2022-04-12T190518Z

"""

import random

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2075, 1, 1)  # todo need to log to mid-year 2050
        self.pop_size = 100_000
        self.scenarios = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.number_of_draws = len(self.scenarios)
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'longterm_mihpsa_runs',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.demography": logging.INFO,
                # "tlo.methods.tb": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO
            }
        }

    def modules(self):
        return [
            demography.Demography(),
            simplified_births.SimplifiedBirths(),
            enhanced_lifestyle.Lifestyle(),
            healthsystem.HealthSystem(service_availability=["*"],  # all treatment allowed
                mode_appt_constraints=1,  # mode of constraints to do with officer numbers and time
                cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
                ignore_priority=False,  # do not use the priority information in HSI event to schedule
                capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
            ),
            symptommanager.SymptomManager(),
            healthseekingbehaviour.HealthSeekingBehaviour(),
            healthburden.HealthBurden(),
            epi.Epi(),
            hiv.Hiv(),
            tb.Tb(),
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'Hiv': {
                'select_mihpsa_scenario': self.scenarios[draw_number],
                'scaleup_start_year': 2024
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
