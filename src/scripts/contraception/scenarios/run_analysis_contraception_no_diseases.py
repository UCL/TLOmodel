"""
This file defines a batch run to get sims results to be used by the analysis_contraception_plot_table.
Run on the remote batch system using:
```tlo batch-submit src/scripts/contraception/scenarios/run_analysis_contraception_no_diseases.py```
or locally using:
```tlo scenario-run src/scripts/contraception/scenarios/run_analysis_contraception_no_diseases.py```
"""

from tlo import Date, logging
from tlo.methods import (
    contraception,
    demography,
    healthsystem,
    hiv,
)
from tlo.scenario import BaseScenario


class RunAnalysisCo(BaseScenario):
    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2050, 12, 31),
            initial_population_size=1_000,  # selected size for the Tim C at al. paper: 250K
            number_of_draws=1,  # <- one scenario
            runs_per_draw=1,  # <- repeated this many times
        )

    def log_configuration(self):
        return {
            'filename': 'run_analysis_contraception_no_diseases',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.demography": logging.INFO
            }
        }

    def modules(self):
        return [
            # Core Modules
            demography.Demography(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources,
                                      cons_availability="all",
                                      disable=False),  # <-- HealthSystem functioning

            # - Contraception and replacement for Labour etc.
            contraception.Contraception(resourcefilepath=self.resources,
                                        use_interventions=True,  # default: False
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
