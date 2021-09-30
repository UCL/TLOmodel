"""
This file defines a batch run of a large population for a long time with *NO* disease modules and no tracking of
HealthSystem usage.
It's used for calibrations of the demographic components of the model only.

Run on the batch system using:
```tlo batch-submit src/scripts/calibration_analyses/scenarios/long_run_no_diseases.py```

or locally using:
    ```tlo scenario-run src/scripts/calibration_analyses/scenarios/long_run_no_diseases.py```

"""

from tlo import Date, logging
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2039, 12, 31)
        self.pop_size = 20_000  # <- recommended population size for the runs
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

            # Representations of the Healthcare System
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True),

            # - Contraception, Pregnancy and Labour
            contraception.Contraception(resourcefilepath=self.resources, use_healthsystem=False),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),

            # - Supportiving Modules
            hiv.DummyHivModule(),
        ]

    def draw_parameters(self, draw_number, rng):
        # Using default parameters in all cases
        return


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
