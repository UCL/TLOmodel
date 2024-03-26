"""This Scenario file run the model under different assumptions for the HealthSystem in order to estimate the
impact that is achieved under each, relative to there being no health system.

Run on the batch system using:
```
tlo batch-submit
 src/scripts/comparison_of_horizontal_and_vertical_programs/scenario_comparison_of_horizontal_and_vertical_programs.py
```

"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ScenarioSwitcher
from tlo.scenario import BaseScenario


class HorizontalAndVerticalPrograms(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'horizontal_and_vertical_programs',
            'directory': Path('./outputs'),
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources) + [ScenarioSwitcher(resourcefilepath=self.resources)]

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""

        YEAR_OF_CHANGE = 2025

        return {
            "Baseline": self._baseline(),

            # - - - Human Resource for Health - - -

            "Reduced Absense":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'year_HR_scaling_by_level_and_officer_type': YEAR_OF_CHANGE,
                            'HR_scaling_by_level_and_officer_type_mode': 'reduced_absence',
                            # todo - create the scenario in that spreadsheet
                        }
                    }
                ),

            "+ Double Capacity at Primary Care":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'yearly_HR_scaling_mode': 'double_capacity_at_primary_care',
                            # todo - create the scenario in that spreadsheet
                        }
                    }
                ),

            "+ Keep Pace with Population Growth":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'yearly_HR_scaling_mode': 'double_capacity_at_primary_care_and_pop_growth',
                            # todo - create the scenario in that spreadsheet
                        }
                    }
                ),

            # - - - Quality of Care - - -

            "Perfect Clinical Practice":
                mix_scenarios(
                    self._baseline(),
                    {'ScenarioSwitcher': {'max_healthsystem_function': True}},
                    # todo - make this change happen on a specific year
                ),

            "Perfect Healthcare Seeking":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'ScenarioSwitcher': {'max_healthsystem_function': False, 'max_healthcare_seeking': True}},
                    # todo - make this change happen on a specific year
                ),




            "+ Perfect Consumables Availability":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {'cons_availability': 'all'}}
                ),




        }

    def _baseline(self) -> Dict:
        """Return the Dict with values for the parameters changes that define the baseline scenario. """
        # todo Make this to be either with a shift into mode2 when other thigns chnage, or with the effective
        #  capacity applied throughout
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {"HealthSystem": {"mode_appt_constraints": 2, "policy_name": "Naive",}},
        ),

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
