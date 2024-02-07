"""
This file defines a batch run through which the hiv and tb modules are run across a set of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/projections_jan2023/calibration_script.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/projections_jan2023/calibration_script.py

or execute a single run:
tlo scenario-run src/scripts/hiv/deviance_for_calibration/calibration_script.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/hiv/projections_jan2023/calibration_script.py

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download calibration_script-2022-04-12T190518Z

"""

import os
import warnings

import pandas as pd

from tlo import Date, logging
from tlo.methods import deviance_measure
# from tlo.methods import deviance_measure
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

# Ignore warnings to avoid cluttering output from simulation
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))

number_of_draws = 30
runs_per_draw = 3


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2023, 1, 1),
            initial_population_size=100000,
            number_of_draws=number_of_draws,
            runs_per_draw=runs_per_draw,
        )

        self.sampled_parameters = pd.read_excel(
            os.path.join(self.resources, "ResourceFile_HIV.xlsx"),
            sheet_name="LHC_samples",
        )

    def log_configuration(self):
        return {
            'filename': 'calibration_runs',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.deviance_measure": logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources) + [deviance_measure.Deviance(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):

        return {
            'Hiv': {
                'beta': self.sampled_parameters.hiv[draw_number],
            },
            'Tb': {
                'scaling_factor_WHO': self.sampled_parameters.tb[draw_number],
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
