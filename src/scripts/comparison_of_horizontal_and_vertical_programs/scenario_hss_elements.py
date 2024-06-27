"""This Scenario file run the model under different assumptions for the HealthSystem and Vertical Program Scale-up

Run on the batch system using:
```
tlo batch-submit
 src/scripts/comparison_of_horizontal_and_vertical_programs/scenario_hss_elements.py
```

"""

from pathlib import Path
from typing import Dict

from scripts.comparison_of_horizontal_and_vertical_programs.scenario_definitions import ScenarioDefinitions
from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class HorizontalAndVerticalPrograms(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2031, 1, 1)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 3  # <--- todo: N.B. Very small number of repeated run, to be efficient for now

    def log_configuration(self):
        return {
            'filename': 'horizontal_and_vertical_programs',
            'directory': Path('./outputs'),
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.WARNING,
                'tlo.methods.healthsystem.summary': logging.INFO,
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
        # todo - decide on final definition of scenarios and the scenario package
        # todo - refactorize to use the ScenariosDefinitions helperclass, which will make sure that this script and
        #  'scenario_vertical_programs)_with_and_without_hss.py' are synchronised (e.g. baseline and HSS pkg scenarios)

        self.YEAR_OF_CHANGE = 2019
        # <-- baseline year of Human Resources for Health is 2018, and this is consistent with calibration during
        # 2015-2019 period.

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
                            'yearly_HR_scaling_mode': 'scaling_by_population_growth',
                            # This is in-line with population growth _after 2018_ (baseline year for HRH)
                        }
                    }
                ),

            "HRH Increases at GDP Growth":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'yearly_HR_scaling_mode': 'GDP_growth',
                            # This is GDP growth after 2018 (baseline year for HRH)
                        }
                    }
                ),

            "HRH Increases above GDP Growth":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'yearly_HR_scaling_mode': 'GDP_growth_fHE_case5',
                            # This is above-GDP growth after 2018 (baseline year for HRH)
                        }
                    }
                ),


            # - - - Quality of Care - - -
            "Perfect Clinical Practice":
                mix_scenarios(
                    self._baseline(),
                    {
                        'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                            'max_healthsystem_function': [False, True],  # <-- switch from False to True mid-way
                            'year_of_switch': self.YEAR_OF_CHANGE,
                        }
                    },
                ),

            "Perfect Healthcare Seeking":
               mix_scenarios(
                   get_parameters_for_status_quo(),
                   {
                       'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                           'max_healthcare_seeking': [False, True],
                           'year_of_switch': self.YEAR_OF_CHANGE,
                       }
                   },
               ),

            # - - - Supply Chains - - -
            "Perfect Availability of Vital Items":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'year_cons_availability_switch': self.YEAR_OF_CHANGE,
                            'cons_availability_postSwitch': 'all_vital_available',
                        }
                    }
                ),

            "Perfect Availability of Medicines":
            mix_scenarios(
                self._baseline(),
                {
                    'HealthSystem': {
                        'year_cons_availability_switch': self.YEAR_OF_CHANGE,
                        'cons_availability_postSwitch': 'all_medicines_available',
                    }
                }
            ),

            "Perfect Availability of All Consumables":
                mix_scenarios(
                    self._baseline(),
                    {
                        'HealthSystem': {
                            'year_cons_availability_switch': self.YEAR_OF_CHANGE,
                            'cons_availability_postSwitch': 'all',
                        }
                    }
                ),

            # - - - FULL PACKAGE OF HEALTH SYSTEM STRENGTHENING - - -
            "FULL PACKAGE":
                mix_scenarios(
                    self._baseline(),
                    {
                        'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                            'max_healthsystem_function': [False, True],  # <-- switch from False to True mid-way
                            'max_healthcare_seeking': [False, True],  # <-- switch from False to True mid-way
                            'year_of_switch': self.YEAR_OF_CHANGE
                        },
                        'HealthSystem': {
                            'year_cons_availability_switch': self.YEAR_OF_CHANGE,
                            'cons_availability_postSwitch': 'all',
                            'yearly_HR_scaling_mode': 'GDP_growth_fHE_case5',
                            'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE,
                            'HR_scaling_by_level_and_officer_type_mode': 'no_absence_&_x2_fac0+1',
                        }
                    },
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
                    "scale_to_effective_capabilities": True,    # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                    "year_mode_switch": self.YEAR_OF_CHANGE,

                    # Baseline scenario is with absence of HCW
                    'year_HR_scaling_by_level_and_officer_type': self.YEAR_OF_CHANGE,
                    'HR_scaling_by_level_and_officer_type_mode': 'with_absence',
                    # todo <-- Do we want the first part of the run be with_abscence too...? (Although that will mean
                    #          that there is actually greater capacity if we do the rescaling)

                    # Normalize the behaviour of Mode 2
                    "policy_name": "Naive",
                    "tclose_overwrite": 1,
                    "tclose_days_offset_overwrite": 7,
                }
            },
        )

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
