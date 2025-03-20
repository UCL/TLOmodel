"""
Produce a run of the Stunting Module to assess the levels of Stunting induced with default parameters and
HealthSystem availability - including the effects of Diarrhoea and Alri and all the Labour modules.

Run on the batch system using:
```tlo batch-submit src/scripts/undernutrition_analyses/stunting/stunting_analysis_scenario.py```

Or locally using:
```tlo scenario-run src/scripts/undernutrition_analyses/stunting/stunting_analysis_scenario.py```
"""

from pathlib import Path

from tlo import Date, logging
from tlo.methods import (
    alri,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    stunting,
    symptommanager,
    wasting,
)
from tlo.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2029, 12, 31)
        self.pop_size = 20_000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'analysis_stunting',
            'directory': Path('./outputs'),
            'custom_levels': {
                "*": logging.WARNING,
                "tlo.methods.stunting": logging.INFO}
        }

    def modules(self):
        return [
            demography.Demography(),
            symptommanager.SymptomManager(),
            enhanced_lifestyle.Lifestyle(),
            healthsystem.HealthSystem(),
            epi.Epi(),
            hiv.Hiv(),
            contraception.Contraception(),
            labour.Labour(),
            pregnancy_supervisor.PregnancySupervisor(),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(),
            postnatal_supervisor.PostnatalSupervisor(),
            newborn_outcomes.NewbornOutcomes(),
            diarrhoea.Diarrhoea(),
            wasting.Wasting(),
            alri.Alri(),
            stunting.Stunting()
        ]

    def draw_parameters(self, draw_number, rng):
        # service_availability = [
        #     ['*'],  # draw 0: HealthSystem operational
        #     []      # draw 1: HealthSystem not operational
        # ]
        # return {
        #     'HealthSystem': {
        #         'Service_Availability': service_availability[draw_number],
        #     },
        # }
        # Awaiting fix to https://github.com/UCL/TLOmodel/issues/392
        pass


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
