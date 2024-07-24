import datetime
import time
from pathlib import Path

from tlo import Date, logging
from tlo.scenario import BaseScenario
from tlo.methods import ( fullmodel
)

from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2099, 1, 12)  #Date(2099, 12, 31)
        self.pop_size = 100_000
        self.number_of_draws = 1
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'longterm_trends_all_diseases',
            'directory': './outputs',
            'custom_levels': {
                # '*': logging.WARNING,
                # "*": logging.DEBUG,
                # "*": logging.FATAL,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.contraception': logging.INFO,
            }
        }

    def modules(self):
        fullmodel_instance = fullmodel.fullmodel(resourcefilepath=self.resources, use_simplified_births=False,
                                                 module_kwargs={"HealthSystem": {
                                                     "mode_appt_constraints": 0,
                                                     "cons_availability": "all",
                                                     "ignore_priority": False,
                                                     "beds_availability": 'all',
                                                     "equip_availability": 'all',
                                                     "capabilities_coefficient": 1.0,
                                                     "use_funded_or_actual_staffing": "funded",
                                                     "disable": False,
                                                     "disable_and_reject_all": False}})

        switcher = ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)

        switcher.parameters = {
            "year_of_switch": 2010,
            "max_healthsystem_function": [True, True],
            "max_healthcare_seeking": [True, True]
        }
        fullmodel_instance = fullmodel_instance + [switcher]
        return fullmodel_instance

    def draw_parameters(self, draw_number, rng):
        return


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
