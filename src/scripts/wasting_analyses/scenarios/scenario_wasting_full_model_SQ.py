"""
This file defines a scenario for wasting analysis.

It can be submitted on Azure Batch by running:

    tlo batch-submit src/scripts/wasting_analyses/scenarios/scenario_wasting_full_model_SQ.py

or locally using:

    tlo scenario-run src/scripts/wasting_analyses/scenarios/scenario_wasting_full_model_SQ.py
"""
# import itertools
import warnings

from tlo import Date, logging
from tlo.analysis.utils import (
    get_parameters_for_status_quo,
    mix_scenarios,
)
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

# capture warnings during simulation run
warnings.simplefilter('default', (UserWarning, RuntimeWarning))


class WastingAnalysis(BaseScenario):

    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(year=2010, month=1, day=1),
            end_date=Date(year=2031, month=1, day=1),
            initial_population_size=4_000,
            number_of_draws=1,
            runs_per_draw=10,
        )

    def log_configuration(self):
        return {
            'filename': 'wasting_analysis__full_model_SQ',
            'directory': './outputs/wasting_analysis',
            "custom_levels": {  # Customise the output of specific loggers
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.population": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.wasting": logging.DEBUG,
                '*': logging.WARNING
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    # Keeping Status Quo parameters
    def draw_parameters(self, draw_number, rng):
        return get_parameters_for_status_quo()

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
