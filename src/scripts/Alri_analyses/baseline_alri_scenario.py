from tlo import Date, logging
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
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
    alri,
    epi, wasting
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 456
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2026, 1, 2)
        self.pop_size = 10000
        self.number_of_draws = 5
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'baseline_scenario_alri', 'directory': './outputs',
            "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.alri": logging.INFO,
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
                                      cons_availability='all'),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
            alri.Alri(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            wasting.Wasting(resourcefilepath=self.resources),
            alri.AlriPropertiesOfOtherModules()
        ]

    def draw_parameters(self, draw_number, rng):
        return {
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
