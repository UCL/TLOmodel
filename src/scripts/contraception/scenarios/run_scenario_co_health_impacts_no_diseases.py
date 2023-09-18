"""
This scenario file defines a batch run to get sims results to be used by an analysis script
( # TODO: create one and add its name here
).
Run on the remote batch system using:
```tlo batch-submit src/scripts/contraception/scenarios/run_scenario_co_health_impacts_no_diseases.py```
or locally using:
```tlo scenario-run src/scripts/contraception/scenarios/run_scenario_co_health_impacts_no_diseases.py```

# TODO: update
SCENARIO SETTINGS
-----------------
used modules:
...

# TODO: update
FP INTERVENTIONS' IMPLICATIONS FOR MNCH PAPER (Eva J et al. 2024):
------------------------------------------------------------------
...
"""

from tlo import Date, logging
from tlo.methods import (
    alri,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    stunting,
    symptommanager,
    tb,
    wasting,
)
from tlo.scenario import BaseScenario


class RunAnalysisCo(BaseScenario):
    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2016, 12, 31),
            initial_population_size=4_000,  # selected size for the Tim C at al. 2023 paper: 250K  #TODO: update
            number_of_draws=1,  # <- one scenario
            runs_per_draw=5,  # <- repeated this many times
        )

    def log_configuration(self):
        return {
            'filename': 'run_scenario_co_health_impacts_no_diseases',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.INFO,
                "tlo.methods.depression": logging.INFO,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.labour": logging.INFO,
                "tlo.methods.labour.detail": logging.INFO,
                "tlo.methods.malaria": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.pregnancy_supervisor": logging.INFO,
                "tlo.methods.postnatal_supervisor": logging.INFO,
                "tlo.methods.tb": logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            contraception.Contraception(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),

            # Register all the modules that are reference in the maternal perinatal health suite (including their
            # dependencies)
            alri.Alri(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),
            cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=self.resources),
            depression.Depression(resourcefilepath=self.resources),
            stunting.Stunting(resourcefilepath=self.resources),
            wasting.Wasting(resourcefilepath=self.resources),
            diarrhoea.Diarrhoea(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'PregnancySupervisor': {'analysis_year': 2023},
            'Labour': {'analysis_year': 2023},
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
