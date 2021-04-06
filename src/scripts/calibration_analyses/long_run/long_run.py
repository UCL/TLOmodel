"""
This file defines a batch run through which the Mockitis module is run across a sweep of a single parameter.

Run on the batch system using:
```tlo batch-submit src/scripts/calibration_analyses/long_run/long_run.py```

"""

import numpy as np

from tlo import Date, logging
from tlo.methods import (
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    labour,
    mockitis,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2030, 12, 31)
        self.pop_size = 20_000
        self.number_of_draws = 1    # <- one scenario
        self.runs_per_draw = 10     # <- repeated ten times

    def log_configuration(self):
        return {
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            # Core Modules
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources),
            dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources, spurious_symptoms=False),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),

            # Pregnancy and Birth
            contraception.Contraception(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            antenatal_care.CareOfWomenDuringPregnancy(resourcefilepapth=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),

            # Disease modules considered complete:
            diarrhoea.Diarrhoea(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            depression.Depression(resourcefilepath=self.resources),
            oesophagealcancer.OesophagealCancer(resourcefilepath=self.resources),
            epilepsy.Epilepsy(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        # Using default parameters in all cases
        return


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
