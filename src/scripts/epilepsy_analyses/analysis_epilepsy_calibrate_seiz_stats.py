import numpy as np

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2030, 1, 1)
        self.pop_size = 20000
        self.smaller_pop_size = 20000
        self.number_of_draws = 25
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'analysis_epilepsy_calibrate_seiz_stats.py',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
                'tlo.methods.epilepsy': logging.INFO,
                'tlo.methods.healthsystem': logging.WARNING,
                'tlo.methods.healthburden': logging.WARNING,
                'tlo.methods.demography': logging.WARNING,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            epilepsy.Epilepsy(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        seiz_stat_0_1_min = 0.05
        seiz_stat_0_1_max = 1.15
        seiz_stat_1_2_min = 0.05
        seiz_stat_1_2_max = 1.15
        prob_seiz_stat_0_1 = np.linspace(seiz_stat_0_1_min, seiz_stat_0_1_max, int(np.sqrt(self.number_of_draws)))
        prob_seiz_stat_1_2 = np.linspace(seiz_stat_1_2_min, seiz_stat_1_2_max, int(np.sqrt(self.number_of_draws)))

        grid = self.make_grid(
            {'base_prob_3m_seiz_stat_none_infreq': prob_seiz_stat_0_1, 'base_prob_3m_seiz_stat_infreq_freq':
                [prob_seiz_stat_1_2]}
        )
        return {
            'Epilepsy': {
                'base_prob_3m_seiz_stat_none_infreq': grid['base_prob_3m_seiz_stat_none_infreq'][draw_number],
                'base_prob_3m_seiz_stat_infreq_freq': grid['base_prob_3m_seiz_stat_none_infreq'][draw_number],
            }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
