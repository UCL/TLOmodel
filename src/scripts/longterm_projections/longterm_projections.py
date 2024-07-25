from typing import Dict

from tlo import Date, logging
from tlo.scenario import BaseScenario
from tlo.methods.fullmodel import fullmodel
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios

from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 12)  #Date(2099, 12, 31)
        self.pop_size = 1000
        self.number_of_draws = 1
        self.runs_per_draw = 1
        self.YEAR_OF_CHANGE = 2010
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)

    def log_configuration(self):
        return {
            'filename': 'longterm_trends_all_diseases',
            'directory': './outputs',
            'custom_levels': {
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.contraception': logging.INFO,
            }
        }

    def modules(self):
        return (
            fullmodel(resourcefilepath=self.resources)
            + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)]
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            print(list(self._scenarios.values())[draw_number])
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        return {'Baseline': self._baseline()}

    def _baseline(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [True, True],
                'max_healthcare_seeking': [True, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "mode_appt_constraints_postSwitch": 1,
                "cons_availability": "all",
                "beds_availability": 'all',
                "equip_availability": 'all',
                "use_funded_or_actual_staffing": "funded_plus",
                },

             },
        )


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])