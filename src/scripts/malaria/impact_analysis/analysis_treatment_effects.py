"""
This scenario file sets up the scenarios for simulating the effects of removing sets of services

The scenarios are:
*0 baseline mode 1
*1 remove HIV-related services
*2 remove TB-related services
*3 remove malaria-related services
*4 baseline mode 2
*4 remove all three sets of services: mode 2

For scenarios 0-3, keep all default health system settings

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/malaria/impact_analysis/analysis_treatment_effects.py

Run on the batch system using:
tlo batch-submit src/scripts/malaria/impact_analysis/analysis_treatment_effects.py

or locally using:
tlo scenario-run src/scripts/malaria/impact_analysis/analysis_treatment_effects.py

or execute a single run:
tlo scenario-run src/scripts/malaria/impact_analysis/analysis_treatment_effects.py --draw 1 0

"""

from pathlib import Path
from typing import Dict

import os
import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_filtered_treatment_ids, get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EffectOfProgrammes(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 100_000
        self.number_of_draws = 1  # todo reset
        self.runs_per_draw = 5

        self.treatment_effects = pd.read_excel(
            os.path.join(self.resources, "ResourceFile_HIV.xlsx"),
            sheet_name="treatment_effects",
        )

    def log_configuration(self):
        return {
            'filename': 'effect_of_treatment_packages',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.malaria': logging.INFO,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
                'tlo.methods.healthburden': logging.INFO
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        treatments = get_filtered_treatment_ids(depth=1)

        # get service availability with all select services removed
        services_to_remove = ['Hiv_*', 'Tb_*', 'Malaria_*']
        service_availability = dict({'Everything': ["*"]})

        # create service package with all three sets of interventions removed
        service_availability.update(
            {f'No_HTM': [v for v in treatments if v not in services_to_remove]}
        )
        # add in HIV/TB EOL care plus malaria_complicated treatment
        # run in scenario 5 so malaria treatment has no effect on mortality
        service_availability['No_HTM'].append('Hiv_PalliativeCare')
        service_availability['No_HTM'].append('Tb_PalliativeCare')
        service_availability['No_HTM'].append('Malaria_Treatment_Complicated')

        return {
            'HealthSystem': {
                'Service_Availability': service_availability[
                    'Everything', 'Everything', 'Everything', 'Everything', 'Everything', 'No_HTM'][draw_number],
                'use_funded_or_actual_staffing': 'funded',
                'mode_appt_constraints': 1,
                'policy_name': 'Naive',
            },
            'Hiv': {
                'scenario': [0, 1, 2, 3, 5, 5][draw_number],
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
