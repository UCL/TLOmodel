"""
This file defines a batch run to get sims results to be used by the analysis_contraception_plot_table.
Run on the remote batch system using:
```tlo batch-submit src/scripts/contraception/scenarios/run_analysis_contraception_no_diseases.py```
or locally using:
```tlo scenario-run src/scripts/contraception/scenarios/run_analysis_contraception_no_diseases.py```


SCENARIO SETTINGS
-----------------
used modules:
* Demography
* HealthSystem
    - cons_availability="all", i.e. all consumables are assumed to be always available,
* Contraception, for which SimplifiedPregnancyAndLabour is used
    - use_interventions=False/True according to what we need (False => without interventions,
    True => interventions since 2023; it needs to be set in the ResourceFile_ContraceptionParams.csv)
* DummyHivModule (a supporting module required by Contraception module)

logging above warning level:
* contraception:
    - INFO if only analysis_all_calibration or figs but not the table from analysis_contraception_plot_table required,
    - DEBUG if tabel from analysis_contraception_plot_table required.
* demography: INFO.
NB. For analysis_all_calibration this is enough only if analysis_hsi_descriptions are not required, and the analysis
needs to be changed accordingly to run properly. We use an adjusted analysis_all_calibration script, stored in the
EvaJ/contraception_2023-02_inclPR807/AnalysisAllCalib_Contraception branch.


CONTRACEPTION PAPER (Tim C et al. 2023):
---------------------------------------
- 1 draw & 1 run/per draw with 250K initial_population_size
- use_interventions=False/True for simulation without/with interventions,
- for analysis_all_calibration adjusted in the branch
EvaJ/contraception_2023-02_inclPR807/AnalysisAllCalib_Contraception with the analysis_hsi_descriptions excluded (Fig 2):
    2010-2099 simulated with contraception logging at the level INFO (a job to simulate that many years with DEBUG
    logging fails)
- for analysis_contraception_plot_table (Fig 3, Fig 4, Tab 4, and Fig A6.1):
    2010-2050 simulated with contraception logging at the level DEBUG
"""

from tlo import Date, logging
from tlo.methods import contraception, demography, healthsystem, hiv
from tlo.scenario import BaseScenario


class RunAnalysisCo(BaseScenario):
    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2099, 12, 31),
            initial_population_size=250_000,  # selected size for the Tim C at al. 2023 paper: 250K
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
                                      cons_availability="all"),

            # - Contraception and replacement for Labour etc.
            contraception.Contraception(resourcefilepath=self.resources,
                                        use_healthsystem=True  # default: True <-- using HealthSystem
                                        # if True initiation and switches to contraception require an HSI
                                        ),
            contraception.SimplifiedPregnancyAndLabour(),

            # - Supporting Module required by Contraception
            hiv.DummyHivModule(),
        ]

    def draw_parameters(self, draw_number, rng):
        # Using default parameters in all cases
        return


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
