"""
This file defines a batch run through which the hiv and tb modules are run across a set of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/deviance_for_calibration/calibration_script.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/deviance_for_calibration/calibration_script.py

or execute a single run:
tlo scenario-run src/scripts/hiv/deviance_for_calibration/calibration_script.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/hiv/deviance_for_calibration/calibration_script.py

weds 16th feb runs:
Job ID: calibration_script-2022-02-16T202042Z

this ^^ is the large calibration run (100 jobs, pop=760k)


Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download calibration_script-2022-04-12T190518Z

12th Apr 2022
Job ID: calibration_script-2022-04-12T190518Z


"""

import random
from pathlib import Path

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    hiv_tb_calibration,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.scenario import BaseScenario
from tlo.util import read_csv_files

number_of_draws = 1
runs_per_draw = 5


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 500000
        self.number_of_draws = number_of_draws
        self.runs_per_draw = runs_per_draw

        self.sampled_parameters = read_csv_files(
            Path("./resources")/"ResourceFile_HIV",
            files="LHC_samples",
        )

    def log_configuration(self):
        return {
            'filename': 'deviance_runs',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.deviance_measure": logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(),
            simplified_births.SimplifiedBirths(),
            enhanced_lifestyle.Lifestyle(),
            healthsystem.HealthSystem(service_availability=["*"],  # all treatment allowed
                mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
                cons_availability="all",  # mode for consumable constraints (if ignored, all consumables available)
                ignore_priority=True,  # do not use the priority information in HSI event to schedule
                capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
                disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
                disable_and_reject_all=False,  # disable healthsystem and no HSI runs
                store_hsi_events_that_have_run=False,  # convenience function for debugging
            ),
            symptommanager.SymptomManager(),
            healthseekingbehaviour.HealthSeekingBehaviour(),
            healthburden.HealthBurden(),
            epi.Epi(),
            hiv.Hiv(),
            tb.Tb(),
            hiv_tb_calibration.Deviance(),
        ]

    def draw_parameters(self, draw_number, rng):

        return {
            'Hiv': {
                'beta': self.sampled_parameters.hiv[draw_number],
            },
            'Tb': {
                'transmission_rate': self.sampled_parameters.tb[draw_number],
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
