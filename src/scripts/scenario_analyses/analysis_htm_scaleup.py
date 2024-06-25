
"""
This scenario file sets up the scenarios for simulating the effects of scaling up programs

The scenarios are:
*0 baseline mode 1
*1 scale-up HIV program
*2 scale-up TB program
*3 scale-up malaria program

scale-up occurs on the default scale-up start date (01/01/2025: in parameters list of resourcefiles)

For all scenarios, keep all default health system settings

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/scenario_analyses/analysis_htm_scaleup.py

Run on the batch system using:
tlo batch-submit src/scripts/scenario_analyses/analysis_htm_scaleup.py

or locally using:
tlo scenario-run src/scripts/scenario_analyses/analysis_htm_scaleup.py

or execute a single run:
tlo scenario-run src/scripts/scenario_analyses/analysis_htm_scaleup.py --draw 1 0

"""

from pathlib import Path

import pandas as pd

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EffectOfProgrammes(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 10_000
        self.number_of_draws = 4
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'scaleup_tests',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.malaria': logging.INFO,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.healthburden': logging.INFO
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        scaleup_start = 5

        return {
            'Hiv': {
                'do_scaleup': [False, True, False, False][draw_number],
                'scaleup_start': scaleup_start
            },
            'Tb': {
                'do_scaleup': [False, False, True, False][draw_number],
                'scaleup_start':  scaleup_start
            },
            'Malaria': {
                'do_scaleup': [False, False, False, True][draw_number],
                'scaleup_start':  scaleup_start
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
