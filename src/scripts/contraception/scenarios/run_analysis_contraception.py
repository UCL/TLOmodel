"""
This file defines a batch run of the analysis_contraception
It's used to create figure(s) and table(s)
Run on the remote batch system using:
```tlo batch-submit src/scripts/contraception/scenarios/run_analysis_contraception.py```
or locally using:
```tlo scenario-run src/scripts/contraception/scenarios/run_analysis_contraception.py```
"""

from tlo import Date, logging
from tlo.methods import contraception, demography, enhanced_lifestyle,\
    healthseekingbehaviour, healthsystem, hiv, symptommanager
from tlo.scenario import BaseScenario


class RunAnalysisCo(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 2022
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2099, 12, 31)
        self.pop_size = 1000  # <- recommended population size for the runs is 50k
        self.number_of_draws = 1  # <- one scenario
        self.runs_per_draw = 1  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'run_analysis_contraception',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem": logging.INFO
            }
        }

    def modules(self):
        return [
            # Core Modules
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
            healthsystem.HealthSystem(resourcefilepath=resources,
                                      cons_availability="all",
                                      disable=False), # <-- HealthSystem functioning

            # - Contraception and replacement for Labour etc.
            contraception.Contraception(resourcefilepath=self.resources,
                                        use_healthsystem=True),  # <-- using HealthSystem
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
