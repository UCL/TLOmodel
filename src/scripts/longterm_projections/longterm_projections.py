import datetime
import time
from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.scenario import BaseScenario
from tlo.methods import ( fullmodel
)
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios

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
        self.YEAR_OF_CHANGE = 2010
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)

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
        return (fullmodel.fullmodel(resourcefilepath=self.resources, use_simplified_births=False,)
        + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)])


    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]


    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        # Load helper class containing the definitions of the elements of all the scenarios
        return {
            "Baseline":
                self._baseline()}

    def _baseline(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {"HealthSystem":{
                    "mode_appt_constraints": 1,
                    "mode_appt_constraints_postSwitch": 1,
                    "year_mode_switch": self.YEAR_OF_CHANGE,
                    "cons_availability": "all",
                    "beds_availability": 'all',
                    "equip_availability": 'all',
                    "use_funded_or_actual_staffing": "funded_plus",
                    },
                    'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                            'max_healthsystem_function': [False, True],  # <-- switch from False to True mid-way
                            'max_healthcare_seeking': [False, True],  # <-- switch from False to True mid-way
                            'year_of_switch': self.YEAR_OF_CHANGE
                            }

            },
        )


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
