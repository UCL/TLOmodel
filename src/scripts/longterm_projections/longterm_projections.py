from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2045, 1, 12)
        self.pop_size = 1000
        self.number_of_draws = 1
        self.runs_per_draw = 10
        self.YEAR_OF_CHANGE = 2020
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)

    def log_configuration(self):
        return {
            'filename': 'longterm_trends_all_diseases',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.population': logging.INFO,
                "tlo.methods.enhanced_lifestyle": logging.INFO
            }
        }

    def modules(self):
        return (
            fullmodel(resourcefilepath=self.resources)
            + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)]
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):

            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        return {'Baseline': self._baseline(),
                'Perfect World': self._perfect_world(),
                'HTM Scale-up': self._htm_scaleup}

    def _baseline(self) -> Dict:
        return get_parameters_for_status_quo()

    def _perfect_world(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

    def _htm_scaleup(self) -> Dict:
            return mix_scenarios(
                get_parameters_for_status_quo(),
                {"Malaria": {
                'type_of_scaleup': 'target',
                'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                    'type_of_scaleup': 'target',
                     'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                    'type_of_scaleup': 'target',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
                }
            )

    def _lifestyle_factors(self) -> Dict:
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {"Lifestyle": {
                'r_urban': 1,
                'r_rural': 1,
                'r_higher_bmi': 1,
                'r_lower_bmi': 1,
                'r_high_salt_urban': 1,
                'r_high_sugar': 1,
                'r_low_ex':1,
                'r_not_low_ex':1,
                'r_tob':1,
                'r_not_tob':1,
                'r_ex_alc':1,
                'r_not_ex_alc' 1,
                'r_non_wood_burn_stove':1
            }}
        )
if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
