
"""
This scenario file sets up the scenarios for simulating the effects of scaling up programs

The scenarios are:
*0 baseline mode 1
*1 scale-up HIV program
*2 scale-up TB program
*3 scale-up malaria program
*4 scale-up HIV and Tb and malaria programs

scale-up occurs on the default scale-up start date (01/01/2025: in parameters list of resourcefiles)

For all scenarios, keep all default health system settings

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/htm_scenario_analyses/analysis_htm_scaleup.py

Run on the batch system using:
tlo batch-submit src/scripts/htm_scenario_analyses/analysis_htm_scaleup.py

or locally using:
tlo scenario-run src/scripts/htm_scenario_analyses/analysis_htm_scaleup.py

or execute a single run:
tlo scenario-run src/scripts/htm_scenario_analyses/analysis_htm_scaleup.py --draw 1 0

"""

from pathlib import Path

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    malaria,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.scenario import BaseScenario


class EffectOfProgrammes(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2030, 1, 1)
        self.pop_size = 15_000
        self.number_of_draws = 5
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'scaleup_tests',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.malaria': logging.INFO,
                'tlo.methods.demography': logging.INFO,
            }
        }

    def modules(self):

        return [
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        scaleup_start_year = 2019

        return {
            'Hiv': {
                'do_scaleup': [False, True, False, False, True][draw_number],
                'scaleup_start_year': scaleup_start_year
            },
            'Tb': {
                'do_scaleup': [False, False, True, False, True][draw_number],
                'scaleup_start_year':  scaleup_start_year
            },
            'Malaria': {
                'do_scaleup': [False, False, False, True, True][draw_number],
                'scaleup_start_year':  scaleup_start_year
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])


