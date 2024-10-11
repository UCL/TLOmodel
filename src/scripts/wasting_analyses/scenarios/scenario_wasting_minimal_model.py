"""
This file defines a scenario for wasting analysis.

It can be submitted on Azure Batch by running:

    tlo batch-submit src/scripts/wasting_analyses/scenario_wasting.py

or locally using:

    tlo scenario-run src/scripts/wasting_analyses/scenario_wasting.py
"""
import warnings

from tlo import Date, logging
from tlo.methods import (
    alri,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
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
            start_date=Date(2010, 1, 1),
            end_date=Date(2030, 1, 1),
            initial_population_size=20_000,
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
                                          service_availability=['*'], cons_availability='default'),
                healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
                healthburden.HealthBurden(resourcefilepath=self.resources),
                symptommanager.SymptomManager(resourcefilepath=self.resources),
                enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
                labour.Labour(resourcefilepath=self.resources),
                care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
                contraception.Contraception(resourcefilepath=self.resources),
                pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
                postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
                newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
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


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
