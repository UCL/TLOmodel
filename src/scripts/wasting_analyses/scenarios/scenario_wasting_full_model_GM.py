"""
This file defines a scenario for wasting analysis.

It can be submitted on Azure Batch by running:

    tlo batch-submit src/scripts/wasting_analyses/scenarios/scenario_wasting_full_model_GM.py

or locally using:

    tlo scenario-run src/scripts/wasting_analyses/scenarios/scenario_wasting_full_model_GM.py
"""
# import itertools
import warnings

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo
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
            initial_population_size=30_000,
            number_of_draws=3,
            runs_per_draw=10,
        )

    def log_configuration(self):
        return {
            'filename': 'wasting_analysis__full_model_GM',
            'directory': './outputs/wasting_analysis',
            "custom_levels": {  # Customise the output of specific loggers
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.population": logging.INFO,
                "tlo.methods.wasting": logging.INFO,
                '*': logging.WARNING
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources,
                         module_kwargs=get_parameters_for_status_quo())

    # Scaling up growth monitoring (GM) attendance scenarios
    def draw_parameters(self, draw_number, rng):
        ### growth_monitoring_attendance_probs by age categories
        # < 1 year
        attendance_prob_below1y = 0.76
        # 1-2 years
        attendance_prob_1to2y = [0.20, 0.25, 1.00]
        # > 2 years
        attendance_prob_above2y = [0.55, 0.50, 1.00]

        return {
            'Wasting': {
                'interv_growth_monitoring_attendance_prob_agecat':
                    [attendance_prob_below1y, attendance_prob_1to2y[draw_number], attendance_prob_above2y[draw_number]]
            }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
