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

        self.YEAR_OF_CHANGE = 2025

        return {
            "Baseline": self._baseline(),

            # ***************************
            # HEALTH SYSTEM STRENGTHENING
            # ***************************

            # - - - Human Resource for Health - - -
            "Reduced Absence":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE,
                            'HR_scaling_by_level_and_officer_type_mode': 'no_absence',
                        }
                    }
                ),

            "Reduced Absence + Double Capacity at Primary Care":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE,
                            'HR_scaling_by_level_and_officer_type_mode': 'no_absence_&_x2_fac0+1',
                        }
                    }
                ),

            "HRH Keeps Pace with Population Growth":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'yearly_HR_scaling_mode': f'pop_growth_from_{self.YEAR_OF_CHANGE}',
                            # todo - create the scenario in that spreadsheet
                        }
                    }
                ),

            "HRH Increases Above Population Growth":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'yearly_HR_scaling_mode': f'expansion_from_{self.YEAR_OF_CHANGE}',
                            # todo - create the scenario in that spreadsheet
                        }
                    }
                ),


            # - - - Quality of Care - - -

            "Perfect Clinical Practice":
                mix_scenarios(
                    self._baseline(),
                    {'ScenarioSwitcher': {'max_healthsystem_function': True}},
                    # todo - make this change happen on a specific year???
                ),

            "Perfect Healthcare Seeking":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'ScenarioSwitcher': {'max_healthcare_seeking': True}},
                    # todo - make this change happen on a specific year???
                ),

            "Perfect Availability of Diagnostics":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {'cons_availability': 'all'}}
                    # todo - Margherita currently developing functionality inside the HealthSystem by which this can happen
                    # todo - this could be inside health system: a new option none/default/all/all-diagnostics/all-medicines // ....would have to combin with issue about RF generation.
                ),

            "Perfect Availability of Medicines & other Consumables":
                mix_scenarios(
                    self._baseline(),
                    {'HealthSystem': {'cons_availability': 'all'}}
                    # todo - Margherita currently developing functionality inside the HealthSystem by which this can happen
                    # todo - this could be inside health system: a new option none/default/all/all-diagnostics/all-medicines // ....would have to combin with issue about RF generation.
                ),

            # ***************************
            # VERTICAL PROGRAMS
            # ***************************

            # - - - HIV - - -
            "HIV Programs Scale-up":
                mix_scenarios(
                    self._baseline(),

                    # todo HIV
                ),

            # - - - TB - - -
            "TB Programs Scale-up":
                mix_scenarios(
                    self._baseline(),

                    # todo TB
                ),

            # - - - MALARIA - - -
            "Malaria Programs Scale-up":
                mix_scenarios(
                    self._baseline(),

                    # todo MALARIA
                ),

        }

    def _baseline(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "HealthSystem": {
                    "mode_appt_constraints": 1,                 # <-- Mode 1 prior to change to preserve calibration
                    "mode_appt_constraints_postSwitch": 2,      # <-- Mode 2 post-change to show effects of HRH
                    "year_mode_switch": self.YEAR_OF_CHANGE,

                    # Baseline scenario is with absence of HCW
                    'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE,
                    'HR_scaling_by_level_and_officer_type_mode': 'with_absence',

                    # Normalize the behaviour of Mode 2
                    "policy_name": "Naive",
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                }
            },
        ),

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
