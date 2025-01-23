"""
This file defines a scenario for wasting analysis.

It can be submitted on Azure Batch by running:

    tlo batch-submit src/scripts/wasting_analyses/scenarios/scenario_wasting_minimal_model.py

or locally using:

    tlo scenario-run src/scripts/wasting_analyses/scenarios/scenario_wasting_minimal_model.py
"""
# import itertools
import warnings

from tlo import Date, logging
from tlo.methods import (
    alri,
    demography,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    stunting,
    symptommanager,
    tb,
    wasting,
)
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
            runs_per_draw=1,
        )

    def log_configuration(self):
        return {
            'filename': 'wasting_analysis__minimal_model',
            'directory': './outputs/wasting_analysis',
            "custom_levels": {  # Customise the output of specific loggers
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.population": logging.INFO,
                "tlo.methods.wasting": logging.INFO,
                '*': logging.WARNING
            }
        }

    def modules(self):
        return [demography.Demography(resourcefilepath=self.resources),
                healthsystem.HealthSystem(resourcefilepath=self.resources,
                                          service_availability=['*'], use_funded_or_actual_staffing='actual',
                                          mode_appt_constraints=1,
                                          cons_availability='default', beds_availability='default',
                                          equip_availability='all'),
                healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
                healthburden.HealthBurden(resourcefilepath=self.resources),
                symptommanager.SymptomManager(resourcefilepath=self.resources, spurious_symptoms=True),
                enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
                simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
                hiv.Hiv(resourcefilepath=self.resources),
                tb.Tb(resourcefilepath=self.resources),
                epi.Epi(resourcefilepath=self.resources),
                alri.Alri(resourcefilepath=self.resources),
                diarrhoea.Diarrhoea(resourcefilepath=self.resources),
                stunting.Stunting(resourcefilepath=self.resources),
                wasting.Wasting(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):
        # Using default parameters in all cases
        return {}

    # def draw_parameters(self, draw_number, rng):
    #     base_death_rate_untreated_sam__draws = [0.01, 0.03, 0.05, 0.08, 0.1]
    #     mod_wast_incidence__coef = [1, 5, 10, 15, 20]
    #     base_inc_rate_wasting_props_by_agegp = [0.0023,0.0099,0.0189,0.0102,0.003, 0.002]
    #     progression_to_sev_wast__coef = [1, 5, 10, 15, 20]
    #     progression_severe_wasting_monthly_props_by_agegp =  [0.0027,0.0036,0.0079,0.0053,0.0025,0.002]
    #     prob_death_after_SAMcare__as_prop_of_death_rate_untreated_sam = [0.85, 0.7, 0.55, 0.4]
    #
    #     pars_combinations = list(itertools.product(
    #         base_death_rate_untreated_sam__draws,
    #         mod_wast_incidence__coef,
    #         progression_to_sev_wast__coef,
    #         prob_death_after_SAMcare__as_prop_of_death_rate_untreated_sam
    #     ))
    #     return {
    #         'Wasting': {
    #             'base_death_rate_untreated_SAM': pars_combinations[draw_number][0],
    #             'base_inc_rate_wasting_by_agegp': [s * pars_combinations[draw_number][1] for \
    #                                                s in base_inc_rate_wasting_props_by_agegp],
    #             'progression_severe_wasting_monthly_by_agegp': [s * pars_combinations[draw_number][2] for \
    #                                                s in progression_severe_wasting_monthly_props_by_agegp],
    #             'prob_death_after_SAMcare': ((pars_combinations[draw_number][0] * pars_combinations[draw_number][3]) /
    #                                          (1-0.738))
    #         }
    #     }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
