"""
This file defines a batch run to get sims results to be used by the analysis_contraception_plot_table.
Run on the remote batch system using:
```tlo batch-submit src/scripts/contraception/scenarios/run_analysis_co_health_impacts_no_diseases.py```
or locally using:
```tlo scenario-run src/scripts/contraception/scenarios/run_analysis_co_health_impacts_no_diseases.py```

# TODO: update
SCENARIO SETTINGS
-----------------
used modules:
* Demography
* HealthSystem
    - cons_availability="all", i.e. all consumables are assumed to be always available,
    - disable=False, i.e. the health system is disabled (hence no constraints and no logging) and every HSI event runs
* Contraception, for which SimplifiedPregnancyAndLabour is used
    - use_interventions=False/True according to what we need (False => without interventions,
    True => interventions since 2023)
* DummyHivModule (a supporting module required by Contraception module)

logging above warning level:
* contraception:
    - INFO if only analysis_all_calibration or figs but not the table from analysis_contraception_plot_table required,
    - DEBUG if tabel from analysis_contraception_plot_table required.
* demography: INFO.
NB. For analysis_all_calibration this is enough only if analysis_hsi_descriptions are not required, and the analysis
needs to be changed accordingly to run properly. We use an adjusted analysis_all_calibration script, stored in the
EvaJ/contraception_2023-02_inclPR807/AnalysisAllCalib_Contraception branch.

# TODO: update
CONTRACEPTION PAPER (Eva J et al. 2023):
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
from tlo.methods import (
    alri,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    stunting,
    symptommanager,
    tb,
    wasting,
)
from tlo.scenario import BaseScenario


class RunAnalysisCo(BaseScenario):
    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2030, 12, 31),
            initial_population_size=2_000,  # selected size for the Tim C at al. 2023 paper: 250K
            number_of_draws=1,  # <- one scenario
            runs_per_draw=1,  # <- repeated this many times
        )

    def log_configuration(self):
        return {
            'filename': 'run_analysis_contraception_no_diseases',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.INFO,
                "tlo.methods.depression": logging.INFO,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.labour": logging.INFO,
                "tlo.methods.labour.detail": logging.INFO,
                "tlo.methods.malaria": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.pregnancy_supervisor": logging.INFO,
                "tlo.methods.postnatal_supervisor": logging.INFO,
                "tlo.methods.tb": logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            contraception.Contraception(resourcefilepath=self.resources,
                                        use_interventions=False,  # default: False
                                        # interventions_start_date=Date(2016, 1, 1),  # if needs to be changed
                                        # the default date is Date(2023, 1, 1)
                                        ),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources,
                                      mode_appt_constraints=1,
                                      cons_availability='all'),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),

            # Register all the modules that are reference in the maternal perinatal health suite (including their
            # dependencies)
            alri.Alri(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),
            cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=self.resources),
            depression.Depression(resourcefilepath=self.resources),
            stunting.Stunting(resourcefilepath=self.resources),
            wasting.Wasting(resourcefilepath=self.resources),
            diarrhoea.Diarrhoea(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'PregnancySupervisor': {'analysis_year': 2023},
            'Labour': {'analysis_year': 2023},
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
