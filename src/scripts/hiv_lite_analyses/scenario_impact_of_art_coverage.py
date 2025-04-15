"""
This file defines an example scenario for analysing the impact of ART coverage.

It can be submitted on Azure Batch by running:

    tlo batch-submit src/scripts/analysis_example/scenario_impact_of_art_coverage.py

or locally using:

    tlo scenario-run src/scripts/analysis_example/scenario_impact_of_art_coverage.py
"""
import warnings

from tlo import Date, logging
from tlo.methods import demography, hivlite
from tlo.scenario import BaseScenario


# Ignore warnings to avoid cluttering output from simulation - generally you do not
# need (and generally shouldn't) do this as warnings can contain useful information but
# we will do so here for the purposes of this example to keep things simple.
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))


class ImpactOfARTCoverage(BaseScenario):

    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2015, 1, 1),
            initial_population_size=5_000,
            number_of_draws=2,
            runs_per_draw=1,
        )

    def log_configuration(self):
        return {
            'filename': 'scenario_impact_of_art_coverage',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hivlite': logging.INFO
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            hivlite.HivLite(resourcefilepath=self.resources)
            ]

    def draw_parameters(self, draw_number, rng):
        return {
            'HivLite': {
                'art_coverage': [0.02, 0.8][draw_number]
            }
        }


if __name__ == '__main__':

    from tlo.cli import scenario_run

    scenario_run([__file__])
