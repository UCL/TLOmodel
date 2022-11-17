"""
This file defines a batch run to get sims results to be used by the analysis_contraception_plot_table.
Run on the remote batch system using:
```tlo batch-submit src/scripts/contraception/scenarios/run_analysis_contraception_no_diseases.py```
or locally using:
```tlo scenario-run src/scripts/contraception/scenarios/run_analysis_contraception_no_diseases.py```
"""

from tlo import Date, logging
from tlo.methods import contraception, demography, enhanced_lifestyle,\
    healthseekingbehaviour, healthsystem, hiv, symptommanager
from tlo.scenario import BaseScenario


class RunAnalysisCo(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2050, 12, 31)
        self.pop_size = 20000  # <- recommended population size for the runs is 50k
        self.number_of_draws = 1  # <- one scenario
        self.runs_per_draw = 1  # <- repeated this many times

    def log_configuration(self):
        return {
            'filename': 'run_analysis_contraception_no_diseases',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem": logging.INFO,
                "tlo.methods.demography": logging.INFO
            }
        }

    def modules(self):
        return [
            # Core Modules
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources,
                                      cons_availability="all",
                                      disable=False),  # <-- HealthSystem functioning

            # - Contraception and replacement for Labour etc.
            contraception.Contraception(resourcefilepath=self.resources,
                                        use_interventions=False,
                                        # interventions_start_date=Date(2016, 1, 1),  # if needs to be changed
                                        # the default date is Date(2023, 1, 1)
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
