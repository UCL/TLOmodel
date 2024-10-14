"""
Azure copd analyses
 """
import warnings

from tlo import Date, logging
from tlo.methods import (
    copd,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.scenario import BaseScenario

# Ignore warnings to avoid cluttering output from simulation - generally you do not
# need (and generally shouldn't) do this as warnings can contain useful information but
# we will do so here for the purposes of this example to keep things simple.
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))


class CopdAnalyses(BaseScenario):

    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2030, 1, 1),
            initial_population_size=250_000,
            number_of_draws=1,
            runs_per_draw=1,
        )

    def log_configuration(self):
        return {
            'filename': 'copd_analyses_azure',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.copd': logging.INFO,
            }
        }

    def modules(self):
        return [demography.Demography(resourcefilepath=self.resources),
                simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
                enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
                healthsystem.HealthSystem(resourcefilepath=self.resources,
                                          disable=False,
                                          cons_availability='all'),
                symptommanager.SymptomManager(resourcefilepath=self.resources),
                healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
                healthburden.HealthBurden(resourcefilepath=self.resources),
                copd.Copd(resourcefilepath=self.resources),
                ]

    # def draw_parameters(self, draw_number, rng):
    #     return {
    #         'Copd': {
    #             'rel_risk_tob': [0.0, 10.0][draw_number],
    #             'rel_risk_wood_burn_stove': [0.0, 2.0][draw_number]
    #         }
    #     }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
