"""
This file defines a batch run through which the hiv and tb modules are run across a set of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/projections_jan2023/batch_test_runs.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/projections_jan2023/batch_test_runs.py

or execute a single run:
tlo scenario-run src/scripts/hiv/deviance_for_calibration/calibration_script.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/hiv/projections_jan2023/batch_test_runs.py

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download calibration_script-2022-04-12T190518Z

"""

import warnings

from tlo import Date, logging
from tlo.methods import hiv_tb_calibration
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

# Ignore warnings to avoid cluttering output from simulation
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2046, 1, 1),
            initial_population_size=50000,
            number_of_draws=1,
            runs_per_draw=5,
        )

    def log_configuration(self):
        return {
            'filename': 'test_runs',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.demography": logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources) + [
            hiv_tb_calibration.Deviance(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):
        return


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
