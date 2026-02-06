"""
This scenario file sets up the scenarios for simulating the effects of nursing staffing levels.

Run on the batch system using:
```
tlo batch-submit src/scripts/nurses_analyses/nurses_scenario_analyses.py
```

or locally using:
```
tlo scenario-run src/scripts/nurses_analyses/nurses_scenario_analyses.py
 ```



"""

from pathlib import Path
from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import (
    get_parameters_for_hrh_historical_scaling_and_rescaling_for_mode2,
    get_root_path,
    mix_scenarios,
)
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class StaffingScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.resources = get_root_path() / "resources"
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2035, 1, 1)
        self.pop_size = 200
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'nurses_scenario_outputs',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel() + [
            ImprovedHealthSystemAndCareSeekingScenarioSwitcher()]

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]

    def draw_name(self, draw_number) -> str:
        """Store scenario name.
        (This name can be retrieved by the plotting scripts to make the graphs be labelled nicely).
        """
        if draw_number < self.number_of_draws:
            return list(self._scenarios.keys())[draw_number]

    @property
    def _default_of_all_scenarios(self) -> Dict:
        """Base set of parameters is the standard historical scaling and transition into Mode 2."""
        return get_parameters_for_hrh_historical_scaling_and_rescaling_for_mode2()

    @property
    def _default_of_all_max_healthsystem_scenarios(self) -> Dict:
        """Improved Health System Performance: the same as the default for scenarios, but increases health system
        function and healthcare seeking behaviour in 2027"""
        return mix_scenarios(
            self._default_of_all_scenarios,  # <-- start with the same default set of parameters (to avoid repeating them)
            {
                'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                    'max_healthcare_seeking': [False, True],
                    'max_healthsystem_function': [False, True],
                    'year_of_switch': 2027,
                },
            },
        )

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario.
        """
        return {
            "Baseline Nurses / Default Healthsystem Function":
                mix_scenarios(
                    self._default_of_all_scenarios,
                    {
                        "HealthSystem": {
                            'HR_scaling_by_level_and_officer_type_mode': "default",
                            "year_HR_scaling_by_level_and_officer_type": 2027,
                        },
                    },
                ),

            "Fewer Nurses / Default Healthsystem Function":
                mix_scenarios(
                    self._default_of_all_scenarios,
                    {
                        "HealthSystem": {
                            'HR_scaling_by_level_and_officer_type_mode': "custom_worse",
                            "year_HR_scaling_by_level_and_officer_type": 2027,
                        },
                    },
                ),

            "More Nurses / Default Healthsystem Function":
                mix_scenarios(
                    self._default_of_all_scenarios,
                    {
                        "HealthSystem": {
                            'HR_scaling_by_level_and_officer_type_mode': "improved_staffing",
                            "year_HR_scaling_by_level_and_officer_type": 2027,
                        },
                    },
                ),

            "Baseline Nurses / Improved Healthsystem Function":
                mix_scenarios(
                    self._default_of_all_max_healthsystem_scenarios,
                    {
                        "HealthSystem": {
                            'HR_scaling_by_level_and_officer_type_mode': "default",
                            "year_HR_scaling_by_level_and_officer_type": 2027,
                        },
                    },
                ),

            "Fewer Nurses / Improved Healthsystem Function":
                mix_scenarios(
                    self._default_of_all_max_healthsystem_scenarios,
                    {
                        "HealthSystem": {
                            'HR_scaling_by_level_and_officer_type_mode': "custom_worse",
                            "year_HR_scaling_by_level_and_officer_type": 2027,
                        },
                    },
                ),

            "More Nurses / Improved Healthsystem Function":
                mix_scenarios(
                    self._default_of_all_max_healthsystem_scenarios,
                    {
                        "HealthSystem": {
                            'HR_scaling_by_level_and_officer_type_mode': "improved_staffing",
                            "year_HR_scaling_by_level_and_officer_type": 2027,
                        },
                    },
                ),
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
