"""
This file defines a batch run to get sims results to be used by the analysis_contraception_plot_table.
Run on the remote batch system using:
```tlo batch-submit src/scripts/contraception/scenarios/run_analysis_contraception_all_diseases.py```
or locally using:
```tlo scenario-run src/scripts/contraception/scenarios/run_analysis_contraception_all_diseases.py```
"""


from tlo import Date, logging
from tlo.methods import (
    alri,
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
    measles,
    newborn_outcomes,
    oesophagealcancer,
    other_adult_cancers,
    postnatal_supervisor,
    pregnancy_supervisor,
    prostate_cancer,
    rti,
    schisto,
    simplified_births,
    stunting,
    symptommanager,
    tb,
    wasting
)
from tlo.scenario import BaseScenario


class RunAnalysisCo(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2099, 12, 31)
        self.pop_size = 50_000  # <- recommended population size for the runs
        self.number_of_draws = 1  # <- one scenario
        self.runs_per_draw = 1  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'run_analysis_contraception_all_diseases',  # <- (specified only for local running)
            'directory': './outputs/run_on_laptop',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.contraception": logging.DEBUG,
                "tlo.methods.demography": logging.INFO
            }
        }

    def modules(self):
        all_modules = []

        # Standard modules:
        all_modules.extend([
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            symptommanager.SymptomManager(
                resourcefilepath=self.resources,
                spurious_symptoms=True),
        ])

        # HealthSystem and the Expanded Programme on Immunizations
        all_modules.extend([
            epi.Epi(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                cons_availability="all",
                disable=False,
                mode_appt_constraints=1,
                capabilities_coefficient=None,
                record_hsi_event_details=False),
        ])

        # Contraception, Pregnancy, Labour, etc. (or SimplifiedBirths)
        all_modules.extend([
            contraception.Contraception(resourcefilepath=self.resources, use_healthsystem=True,
                                        use_interventions=False,  # default: False
                                        # interventions_start_date=Date(2016, 1, 1),  # if needs to be changed
                                        # the default date is Date(2023, 1, 1)
                                        ),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources)
        ])

        # Conditions of Early Childhood
        all_modules.extend([
            alri.Alri(resourcefilepath=self.resources),
            diarrhoea.Diarrhoea(resourcefilepath=self.resources),
            stunting.Stunting(resourcefilepath=self.resources),
            wasting.Wasting(resourcefilepath=self.resources),
        ])

        # Communicable Diseases
        all_modules.extend([
            hiv.Hiv(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),
            measles.Measles(resourcefilepath=self.resources),
            schisto.Schisto(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources)
        ])

        # Non-Communicable Conditions
        #  - Cancers
        all_modules.extend([
            bladder_cancer.BladderCancer(resourcefilepath=self.resources),
            breast_cancer.BreastCancer(resourcefilepath=self.resources),
            oesophagealcancer.OesophagealCancer(resourcefilepath=self.resources),
            other_adult_cancers.OtherAdultCancer(resourcefilepath=self.resources),
            prostate_cancer.ProstateCancer(resourcefilepath=self.resources),
        ])

        #  - Cardio-metabolic Disorders
        all_modules.extend([
            cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=self.resources)
        ])

        #  - Injuries
        all_modules.extend([
            rti.RTI(resourcefilepath=self.resources)
        ])

        #  - Other Non-Communicable Conditions
        all_modules.extend([
            depression.Depression(resourcefilepath=self.resources),
            epilepsy.Epilepsy(resourcefilepath=self.resources)
        ])

        return all_modules

    def draw_parameters(self, draw_number, rng):
        return  # Using default parameters in all cases


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
