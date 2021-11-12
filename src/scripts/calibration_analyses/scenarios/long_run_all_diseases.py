"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for calibrations (demographic patterns, health burdens and healthsytstem usage)

Run on the batch system using:
```tlo batch-submit src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py```

or locally using:
    ```tlo scenario-run src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py```

"""

from tlo import Date, logging
from tlo.methods import (
    bladder_cancer,
    breast_cancer,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    oesophagealcancer,
    other_adult_cancers,
    postnatal_supervisor,
    pregnancy_supervisor,
    prostate_cancer,
    symptommanager,
)
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2029, 12, 31)
        self.pop_size = 20_000  # <- recommened population size for the runs
        self.number_of_draws = 1  # <- one scenario
        self.runs_per_draw = 10  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'long_run',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            # Core Modules
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources, spurious_symptoms=False),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),

            # Representations of the Healthcare System
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),

            # - Contraception, Pregnancy and Labour
            contraception.Contraception(resourcefilepath=self.resources, use_healthsystem=True),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),

            # - Conditions of Early Childhood
            diarrhoea.Diarrhoea(resourcefilepath=self.resources),

            # - Communicable Diseases
            hiv.Hiv(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),

            # - Non-Communicable Conditions
            # -- Cancers
            bladder_cancer.BladderCancer(resourcefilepath=self.resources),
            breast_cancer.BreastCancer(resourcefilepath=self.resources),
            oesophagealcancer.OesophagealCancer(resourcefilepath=self.resources),
            other_adult_cancers.OtherAdultCancer(resourcefilepath=self.resources),
            prostate_cancer.ProstateCancer(resourcefilepath=self.resources),

            # -- Caridometabolic Diorders
            cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=self.resources),

            # -- Injuries (Forthcoming)

            # -- Other Non-Communicable Conditions
            depression.Depression(resourcefilepath=self.resources),
            epilepsy.Epilepsy(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        # Using default parameters in all cases
        return


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
