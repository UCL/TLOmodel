"""
This file uses the calibrated parameters for prop_susceptible
and runs with no MDA running
outputs can be plotted against data to check calibration

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/schistosomiasis/baseline_run.py

Run locally:
tlo scenario-run src/scripts/schistosomiasis/baseline_run.py

or execute a single run:
tlo scenario-run src/scripts/schistosomiasis/baseline_run.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/schistosomiasis/baseline_run.py

"""

import random

from tlo import Date, logging
from tlo.methods import (
    alri,
    bladder_cancer,
    demography,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    schisto,
    simplified_births,
    stunting,
    symptommanager,
    tb,
    wasting,
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.seed = random.randint(0, 50000)
        self.start_date = Date(2010, 1, 1)

        self.end_date = Date(2023, 12, 31)
        self.pop_size = 64_000  # if equal_allocation_by_district, 64,000=2k per district
        self.runs_per_draw = 3

        self.mda_execute = True
        self.single_district = False
        self.equal_allocation_by_district = True

        # Calculate the total number of combinations
        self.number_of_draws = 1

    def log_configuration(self):
        return {
            'filename': 'schisto_scenarios',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.enhanced_lifestyle": logging.INFO,
                "tlo.methods.schisto": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources,
                                  equal_allocation_by_district=self.equal_allocation_by_district),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            # diseases
            alri.Alri(resourcefilepath=self.resources),
            bladder_cancer.BladderCancer(resourcefilepath=self.resources),
            diarrhoea.Diarrhoea(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            schisto.Schisto(resourcefilepath=self.resources,
                            mda_execute=self.mda_execute,
                            single_district=self.single_district),
            stunting.Stunting(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
            wasting.Wasting(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):

        return


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
