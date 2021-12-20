"""
This file defines a batch run through which the hiv and tb modules are run across a set of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/hiv_tb_analyses/calibration_script.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/hiv_tb_analyses/calibration_script.py

or execute a single run:
tlo scenario-run src/scripts/hiv/hiv_tb_analyses/calibration_script.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/hiv/hiv_tb_analyses/calibration_script.py


save Job ID: tara_tb_hiv_calibration-2021-10-06T094337Z


Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download tlo_q1_demo-123

"""

import numpy as np
import pandas as pd
import random
import os

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
    deviance_measure,
)

from tlo.scenario import BaseScenario

number_of_draws = 20
runs_per_draw = 5


# todo: check all below with analysis_logged_deviance
class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 1500
        self.number_of_draws = number_of_draws
        self.runs_per_draw = runs_per_draw

        self.sampled_parameters = pd.read_excel(
            os.path.join(self.resources, "ResourceFile_HIV.xlsx"),
            sheet_name="LHC_samples",
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
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                service_availability=["*"],  # all treatment allowed
                mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
                cons_availability="all",  # mode for consumable constraints (if ignored, all consumables available)
                ignore_priority=True,  # do not use the priority information in HSI event to schedule
                capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
                disable=True,  # disables the healthsystem (no constraints and no logging) and every HSI runs
                disable_and_reject_all=False,  # disable healthsystem and no HSI runs
                store_hsi_events_that_have_run=False,  # convenience function for debugging
            ),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
            deviance_measure.Deviance(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        # todo bring in LHC samples here
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
