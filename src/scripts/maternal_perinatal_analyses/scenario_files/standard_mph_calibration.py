from tlo import Date, logging
from tlo.methods import (
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 666
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2030, 1, 2)
        self.pop_size = 30000
        self.number_of_draws = 5
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'normal_30k_pop',
            "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.labour": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.pregnancy_supervisor": logging.INFO,
                "tlo.methods.postnatal_supervisor": logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            contraception.Contraception(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources,
                                      service_availability=['*'],
                                      ignore_cons_constraints=True),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            depression.Depression(resourcefilepath=self.resources),
            cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        return {
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
