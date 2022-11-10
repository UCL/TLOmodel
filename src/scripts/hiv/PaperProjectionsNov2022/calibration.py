"""
This file defines a batch run through which the hiv and tb modules are run across a set of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/PaperProjectionsNov2022/calibration.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/PaperProjectionsNov2022/calibration.py

or execute a single run:
tlo scenario-run src/scripts/hiv/deviance_for_calibration/calibration_script.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/hiv/PaperProjectionsNov2022/calibration.py

10th Nov 2022:
Job ID: calibration-2022-11-10T201529Z

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download calibration-2022-11-10T201529Z

12th Apr 2022
Job ID: calibration_script-2022-04-12T190518Z


"""

import os
import random

import pandas as pd
from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel

from tlo.scenario import BaseScenario

number_of_draws = 20
runs_per_draw = 3


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 100000
        self.number_of_draws = number_of_draws
        self.runs_per_draw = runs_per_draw

        self.sampled_parameters = pd.read_excel(
            os.path.join(self.resources, "ResourceFile_HIV.xlsx"),
            sheet_name="LHC_samples",
        )

    def log_configuration(self):
        return {
            'filename': 'calibration',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.deviance_measure": logging.INFO,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.demography": logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(
                resourcefilepath=self.resources,
                use_simplified_births=False,
                symptommanager_spurious_symptoms=True,
                healthsystem_disable=False,
                healthsystem_mode_appt_constraints=0,  # no constraints
                healthsystem_cons_availability="default",  # all cons always available
                healthsystem_beds_availability="all",  # all beds always available
                healthsystem_ignore_priority=False,  # ignore priority in HSI scheduling
                healthsystem_use_funded_or_actual_staffing="funded_plus",  # daily capabilities of staff
                healthsystem_capabilities_coefficient=1.0,  # if 'None' set to ratio of init 2010 pop
                healthsystem_record_hsi_event_details=False
            ),

    def draw_parameters(self, draw_number, rng):

        return {
            'Hiv': {
                'beta': self.sampled_parameters.hiv_Nov22[draw_number],
            },
            'Tb': {
                'transmission_rate': self.sampled_parameters.tb_Nov22[draw_number],
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
