from tlo import Date
from tlo import logging
from tlo.scenario import BaseScenario
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    dx_algorithm_adult,
    dx_algorithm_child,
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
    cardio_metabolic_disorders
)


class MyTestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 888
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 2)
        self.pop_size = 15000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'trial_run_high_preg', 'directory': './outputs',
            'custom_levels': {'*': logging.INFO}
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
            dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=self.resources),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        return {}

    def intervene_in_initial_population(self, population):
        # Do something to population
        df = population.props

        women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]

        df.loc[women_repro.index, 'is_pregnant'] = True
        df.loc[women_repro.index, 'date_of_last_pregnancy'] = Date(2010, 1, 1)
        for person in women_repro.index:
            self.modules['Labour'].set_date_of_labour(person)

        #all_women = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14)]
        #df.loc[all_women.index, 'la_parity'] = 0
