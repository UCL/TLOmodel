import numpy as np

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epilepsy,
    healthseekingbehaviour,
    healthsystem,
    healthburden,
    simplified_births,
    symptommanager,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 10000
        self.smaller_pop_size = 10000
        self.number_of_draws = 5
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'analysis_epilepsy_calibrate_incidence.py',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
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
        # Create parameters to vary
        base_rate_min = 0.003 - 0.0015
        base_rate_max = 0.5 + 0.0015
        base_rate_linspace = np.linspace(base_rate_min, base_rate_max, self.number_of_draws)
        # Reset rest of modules parameters to default stated in the document/master
        init_epil_seiz_status = [0.9875, 0.004, 0.008, 0.0005]
        init_prop_antiepileptic_seiz_stat_1 = 0.25
        init_prop_antiepileptic_seiz_stat_2 = 0.3
        init_prop_antiepileptic_seiz_stat_3 = 0.3
        rr_epilepsy_age_ge20 = 0.3
        prop_inc_epilepsy_seiz_freq = 0.1
        rr_effectiveness_antiepileptics = 5.0
        base_prob_3m_seiz_stat_freq_infreq = 0.005
        base_prob_3m_seiz_stat_infreq_freq = 0.05
        base_prob_3m_seiz_stat_none_freq = 0.05
        base_prob_3m_seiz_stat_none_infreq = 0.05
        base_prob_3m_seiz_stat_infreq_none = 0.005
        base_prob_3m_stop_antiepileptic = 0.1
        rr_stop_antiepileptic_seiz_infreq_or_freq = 0.5
        base_prob_3m_epi_death = 0.001

        return {
            'Epilepsy': {
                'base_3m_prob_epilepsy': base_rate_linspace[draw_number],
                'init_epil_seiz_status': init_epil_seiz_status,
                'init_prop_antiepileptic_seiz_stat_1': init_prop_antiepileptic_seiz_stat_1,
                'init_prop_antiepileptic_seiz_stat_2': init_prop_antiepileptic_seiz_stat_2,
                'init_prop_antiepileptic_seiz_stat_3': init_prop_antiepileptic_seiz_stat_3,
                'rr_epilepsy_age_ge20': rr_epilepsy_age_ge20,
                'prop_inc_epilepsy_seiz_freq': prop_inc_epilepsy_seiz_freq,
                'rr_effectiveness_antiepileptics': rr_effectiveness_antiepileptics,
                'base_prob_3m_seiz_stat_freq_infreq': base_prob_3m_seiz_stat_freq_infreq,
                'base_prob_3m_seiz_stat_infreq_freq': base_prob_3m_seiz_stat_infreq_freq,
                'base_prob_3m_seiz_stat_none_freq': base_prob_3m_seiz_stat_none_freq,
                'base_prob_3m_seiz_stat_none_infreq': base_prob_3m_seiz_stat_none_infreq,
                'base_prob_3m_seiz_stat_infreq_none': base_prob_3m_seiz_stat_infreq_none,
                'base_prob_3m_stop_antiepileptic': base_prob_3m_stop_antiepileptic,
                'rr_stop_antiepileptic_seiz_infreq_or_freq': rr_stop_antiepileptic_seiz_infreq_or_freq,
                'base_prob_3m_epi_death': base_prob_3m_epi_death
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
