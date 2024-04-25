"""
This scenario file sets up the scenarios for simulating the effects of removing sets of services

The scenarios are:
*1 remove all three sets of services

keep all default health system settings

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/malaria/impact_analysis/analysis_remove_packages.py

Run on the batch system using:
tlo batch-submit src/scripts/malaria/impact_analysis/analysis_remove_packages.py

or locally using:
tlo scenario-run src/scripts/malaria/impact_analysis/analysis_remove_packages.py

or execute a single run:
tlo scenario-run src/scripts/malaria/impact_analysis/analysis_remove_packages.py --draw 1 0

"""

from pathlib import Path
from typing import Dict

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
        self.number_of_draws = 1
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'remove_treatment_packages',
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
        service_availability = dict({"Everything": ["*"]})

        # create service package with all three sets of interventions removed
        service_availability.update(
            {f"No_HTM": [v for v in treatments if v not in services_to_remove]}
        )
        # add in HIV/TB EOL care plus malaria_complicated treatment
        # run in scenario 5 so malaria treatment has no effect on mortality
        service_availability['No_HTM'].append('Hiv_PalliativeCare')
        service_availability['No_HTM'].append('Tb_PalliativeCare')
        service_availability['No_HTM'].append('Malaria_Treatment_Complicated')

        return {
            'HealthSystem': {
                'Service_Availability': service_availability['No_HTM'],
                'use_funded_or_actual_staffing': 'funded',
                'mode_appt_constraints': 1,
            },
            'Hiv': {
                'scenario': 5,  # remove treatment effects for malaria EOL care
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
