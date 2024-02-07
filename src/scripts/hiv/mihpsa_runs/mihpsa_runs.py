
"""
This file defines a batch run through which the hiv and tb modules are run across a set of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/mihpsa_runs/mihpsa_runs.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/mihpsa_runs/mihpsa_runs.py

or execute a single run:
tlo scenario-run src/scripts/hiv/deviance_for_calibration/calibration_script.py --draw 1 0

Run on the batch system using:
tlo batch-submit src/scripts/hiv/mihpsa_runs/mihpsa_runs.py

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download calibration_script-2022-04-12T190518Z

"""

import warnings

from tlo import Date, logging
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
    tb,
)
from tlo.scenario import BaseScenario

# Ignore warnings to avoid cluttering output from simulation
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2046, 1, 1),
            initial_population_size=150_000,
            number_of_draws=1,
            runs_per_draw=5,
        )

    def log_configuration(self):
        return {
            'filename': 'mihpsa_runs',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.demography": logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            contraception.Contraception(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                service_availability=["*"],  # all treatment allowed
                mode_appt_constraints=1,  # mode of constraints to do with officer numbers and time
                cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
                ignore_priority=True,  # do not use the priority information in HSI event to schedule
                capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
                disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
                disable_and_reject_all=False,  # disable healthsystem and no HSI runs
            ),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
        ]
    def draw_parameters(self, draw_number, rng):
        return


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
