"""
This file defines an example scenario for analysing the impact of consumable availability.

It can be submitted on Azure Batch by running:

    tlo batch-submit src/scripts/analysis_example/scenario_impact_of_consumables_availability.py

or locally using:

    tlo scenario-run src/scripts/analysis_example/scenario_impact_of_consumables_availability.py
"""
import warnings

import pandas as pd

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


# Ignore warnings to avoid cluttering output from simulation - generally you do not
# need (and generally shouldn't) do this as warnings can contain useful information but
# we will do so here for the purposes of this example to keep things simple.
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))


class ImpactOfConsumablesAvailability(BaseScenario):

    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(years=5)
        self.pop_size = 10_000
        self.number_of_draws = 2
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'impact_of_consumables_availability',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'cons_availability': ['default', 'all'][draw_number]
            }
        }


if __name__ == '__main__':

    from tlo.cli import scenario_run

    scenario_run([__file__])
