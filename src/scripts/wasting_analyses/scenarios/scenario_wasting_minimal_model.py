"""
This file defines a scenario for wasting analysis.

It can be submitted on Azure Batch by running:

    tlo batch-submit src/scripts/wasting_analyses/scenarios/scenario_wasting_minimal_model.py

or locally using:

    tlo scenario-run src/scripts/wasting_analyses/scenarios/scenario_wasting_minimal_model.py
"""
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
            number_of_draws=4,
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
        base_death_rate_untreated_SAM__draws = [0.05, 0.1, 0.15, 0.2]
        return {
            'Wasting': {
                'base_death_rate_untreated_SAM': base_death_rate_untreated_SAM__draws[draw_number]
            }
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
