"""
This file defines a batch run of a large population for a long time with *NO* disease modules and no HealthSystem.
It's used for calibrations of the demographic components of the model only.

Run on the remote batch system using:
```tlo batch-submit src/scripts/calibration_analyses/scenarios/long_run_no_diseases.py```

or locally using:
    ```tlo scenario-run src/scripts/calibration_analyses/scenarios/long_run_no_diseases.py```

"""

from tlo import Date, logging
from tlo.methods import contraception, demography, hiv
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 12, 31)
        self.pop_size = 2000  # <- recommended population size for the runs is 50k
        self.number_of_draws = 1  # <- one scenario
        self.runs_per_draw = 5  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'long_run_no_diseases',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.population': logging.INFO,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.contraception': logging.INFO
            }
        }

    def modules(self):
        return [
            # Core Modules
            demography.Demography(resourcefilepath=self.resources),

            # - Contraception and replacement for Labour etc.
            contraception.Contraception(resourcefilepath=self.resources,
                                        use_healthsystem=False),
            contraception.SimplifiedPregnancyAndLabour(),

            # - Supporting Modules required by Contraception
            hiv.DummyHivModule(),
        ]

    def draw_parameters(self, draw_number, rng):
        # Using default parameters in all cases
        return


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
