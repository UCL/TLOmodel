"""
This file defines a scenario for wasting analysis.

It can be submitted on Azure Batch by running:

    tlo batch-submit src/scripts/wasting_analyses/scenarios/scenario_wasting_full_model.py

or locally using:

    tlo scenario-run src/scripts/wasting_analyses/scenarios/scenario_wasting_full_model.py
"""
import warnings

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

# capture warnings during simulation run
warnings.simplefilter('default', (UserWarning, RuntimeWarning))


class WastingAnalysis(BaseScenario):

    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(year=2010, month=1, day=1),
            end_date=Date(year=2030, month=1, day=1),
            initial_population_size=20_000,
            number_of_draws=1,
            runs_per_draw=1,
        )

    def log_configuration(self):
        return {
            'filename': 'wasting_analysis__full_model',
            'directory': './outputs/wasting_analysis',
            "custom_levels": {  # Customise the output of specific loggers
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.population": logging.INFO,
                "tlo.methods.wasting": logging.INFO,
                '*': logging.WARNING
            }
        }

    def modules(self):
        return fullmodel(
            resourcefilepath=self.resources
        )

    def draw_parameters(self, draw_number, rng):
        # Using default parameters in all cases
        return {}


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
