"""
Produce a run of the Stunting Module to assess the levels of Stunting induced with default parameters and
HealthSystem availability - including the effects of Diarrhoea and Alri and all the labour modules.

Run on the batch system using:
```tlo batch-submit src/scripts/undernutrition_analyses/analysis_stunting.py```

Or locally using:
```tlo batch-job src/scripts/undernutrition_analyses/analysis_stunting.py```
"""

from pathlib import Path

from tlo import Date, logging
from tlo import Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    hiv,
    newborn_outcomes,
    postnatal_supervisor,
    stunting, diarrhoea, alri, wasting, epi,
)
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 12, 31)
        self.pop_size = 500
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'analysis_stunting',
            'directory': './outputs',
            'custom_levels': {
                "*": logging.WARNING,
                "tlo.methods.stunting": logging.INFO}
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            contraception.Contraception(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),

            diarrhoea.Diarrhoea(resourcefilepath=self.resources),
            wasting.Wasting(resourcefilepath=self.resources),
            alri.Alri(resourcefilepath=self.resources),
            stunting.Stunting(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        pass

if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])


# %% Analysis
# use outputs/analysis_stunting-2021-10-14T130317Z

