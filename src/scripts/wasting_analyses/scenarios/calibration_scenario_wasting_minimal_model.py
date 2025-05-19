"""
This file defines a scenario for wasting analysis.

It can be submitted on Azure Batch by running:

    tlo batch-submit src/scripts/wasting_analyses/scenarios/scenario_wasting_minimal_model.py

or locally using:

    tlo scenario-run src/scripts/wasting_analyses/scenarios/scenario_wasting_minimal_model.py


After several iterations of simulations, a range of values for the calibrated parameters was identified and tested.
From these, the best-calibrated values were selected for the module.
"""



import itertools
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
            initial_population_size=30_000,
            number_of_draws=81,
            runs_per_draw=1,
        )

    def log_configuration(self):
        return {
            'filename': 'wasting_analysis__minimal_model',
            'directory': './outputs/wasting_analysis',
            "custom_levels": {  # Customise the output of specific loggers
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.population": logging.INFO,
                "tlo.methods.wasting": logging.DEBUG,
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
        base_death_rate_untreated_sam__draws = [0.033, 0.030, 0.027]
        mod_wast_incidence__coef = [0.45, 0.40, 0.35]
        # base mod wast incidence rate calibrated with bathtub model
        base_overall_mod_wast_inc_rate_bathtub = 0.019
        # relative risks for age groups of mod wast incidence rates calibrated with bathtub model
        # rr_inc_rate_wasting_by_agegp = [1.00, 1.22, 1.71, 0.30, 0.40, 0.26] --- as in RFWast/parameters
        progression_to_sev_wast__coef = [0.88, 1.00, 1.12]
        # progression rates to severe wast calibrated with bathtub model
        progression_severe_wasting_monthly_props_by_agegp = [0.3082, 0.8614, 0.4229, 0.4337, 0.2508, 0.3321]
        prob_death_after_SAMcare__as_prop_of_death_rate_untreated_sam = [0.07, 0.10, 0.13]

        pars_combinations = list(itertools.product(
            base_death_rate_untreated_sam__draws,
            mod_wast_incidence__coef,
            progression_to_sev_wast__coef,
            prob_death_after_SAMcare__as_prop_of_death_rate_untreated_sam
        ))

        return {
            'Wasting': {
                'base_death_rate_untreated_SAM': pars_combinations[draw_number][0],
                'base_overall_inc_rate_wasting': base_overall_mod_wast_inc_rate_bathtub * pars_combinations[draw_number][1] ,
                'progression_severe_wasting_monthly_by_agegp': [s * pars_combinations[draw_number][2] for \
                                                   s in progression_severe_wasting_monthly_props_by_agegp],
                'prob_death_after_SAMcare': ((pars_combinations[draw_number][0] * pars_combinations[draw_number][3]) /
                                             (1-0.738))
            }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
