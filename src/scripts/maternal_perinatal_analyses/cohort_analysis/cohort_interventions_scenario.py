import numpy as np
import pandas as pd

from pathlib import Path

from tlo import Date, logging
from tlo.methods import mnh_cohort_module
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class BaselineScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 562661
        self.start_date = Date(2024, 1, 1)
        self.end_date = Date(2025, 1, 2)
        self.pop_size = 12_000
        self.number_of_draws = 9
        self.runs_per_draw = 30

    def log_configuration(self):
        return {
            'filename': 'block_intervention_test', 'directory': './outputs',
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.INFO,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.labour": logging.INFO,
                "tlo.methods.labour.detail": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.pregnancy_supervisor": logging.INFO,
                "tlo.methods.postnatal_supervisor": logging.INFO,
            }
        }

    def modules(self):
        return [*fullmodel(resourcefilepath=self.resources,
                           module_kwargs={'Schisto': {'mda_execute': False}}),
                 mnh_cohort_module.MaternalNewbornHealthCohort(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):
        if draw_number == 0:
            return {'PregnancySupervisor': {
                    'analysis_year': 2024}}
        else:
            interventions_for_analysis = ['oral_antihypertensives', 'oral_antihypertensives',
                                          'iv_antihypertensives', 'iv_antihypertensives',
                                          'mgso4', 'mgso4',
                                          'post_abortion_care_core', 'post_abortion_care_core']

            avail_for_draw = [0.0, 1.0,
                              0.0, 1.0,
                              0.0, 1.0,
                              0.0, 1.0]

            return {'PregnancySupervisor': {
                    'analysis_year': 2024,
                    'interventions_analysis': True,
                    'interventions_under_analysis':[interventions_for_analysis[draw_number-1]],
                    'intervention_analysis_availability': avail_for_draw[draw_number-1]}}


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
