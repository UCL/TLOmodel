"""
This file defines a batch run of a large population for a long time with all disease modules and full use of HSIs
It's used for calibrations (demographic patterns, health burdens and healthsystem usage)

Run on the batch system using:
```tlo batch-submit src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py```

or locally using:
    ```tlo scenario-run src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py```

"""

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    symptommanager,
    epi,
    healthsystem,
    simplified_births,
    contraception,
    pregnancy_supervisor,
    care_of_women_during_pregnancy,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    alri,
    diarrhoea,
    stunting,
    wasting,
    hiv,
    malaria,
    measles,
    schisto,
    tb,
    bladder_cancer,
    breast_cancer,
    cervical_cancer,
    oesophagealcancer,
    other_adult_cancers,
    prostate_cancer,
    cardio_metabolic_disorders,
    rti,
    copd,
    depression,
    epilepsy,
)

class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1)  # The simulation will stop before reaching this date.
        self.pop_size = 10_000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'long_run_all_diseases_2040_test_100k',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                "tlo.methods.contraception": logging.INFO,
            }
        }

    def modules(self):
        return [
    demography.Demography(),
    enhanced_lifestyle.Lifestyle(),
    healthburden.HealthBurden(),
    healthseekingbehaviour.HealthSeekingBehaviour(),
    symptommanager.SymptomManager(),
    epi.Epi(),
    healthsystem.HealthSystem(),
    alri.Alri(),
    diarrhoea.Diarrhoea(),
    stunting.Stunting(),
    wasting.Wasting(),
    hiv.Hiv(),
    malaria.Malaria(),
    measles.Measles(),
    schisto.Schisto(),
    tb.Tb(),
    bladder_cancer.BladderCancer(),
    breast_cancer.BreastCancer(),
    cervical_cancer.CervicalCancer(),
    oesophagealcancer.OesophagealCancer(),
    other_adult_cancers.OtherAdultCancer(),
    prostate_cancer.ProstateCancer(),
    cardio_metabolic_disorders.CardioMetabolicDisorders(),
    rti.RTI(),
    copd.Copd(),
    depression.Depression(),
    epilepsy.Epilepsy(),
            contraception.Contraception(),
            pregnancy_supervisor.PregnancySupervisor(),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(),
            labour.Labour(),
            newborn_outcomes.NewbornOutcomes(),
            postnatal_supervisor.PostnatalSupervisor(),
        ]
    def draw_parameters(self, draw_number, rng):
        return get_parameters_for_status_quo()


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
