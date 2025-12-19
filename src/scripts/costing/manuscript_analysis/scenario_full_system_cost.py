"""This Scenario file run the model under different assumptions of HR capacitu and consumable availability
 for the HSSP-III duration. The scenarios were conceptualised for the Horizontal versus Vertical analysis and
 adapted for the Costing manuscript

Run on the batch system using:
```
tlo batch-submit src/scripts/costing/manuscript_analysis/scenario_full_system_cost.py
```

or locally using:
```
tlo scenario-run src/scripts/costing/manuscript_analysis/scenario_full_system_cost.py
 ```

"""
from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario

class FullSystemCosting(BaseScenario):

    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2031, 1, 1)
        self.pop_size = 100_000
        self.scenarios = list(range(0,4)) # add scenarios as necessary
        self.number_of_draws = len(self.scenarios)
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'full_system_costing',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO, # detailed healthsystem log has been removed
                "tlo.methods.hiv": logging.INFO,
            }
        }

    def modules(self):
        return (
                fullmodel() #
                + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher()]
        )

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'cons_availability': 'default',
                'year_cons_availability_switch': 2020,
                'cons_availability_postSwitch': ['default', 'default', 'scenario6', 'scenario6'][draw_number],
                'mode_appt_constraints':1,
                'mode_appt_constraints_postSwitch':2,
                'year_mode_switch':2020,
                'policy_name': 'Naive',
                'use_funded_or_actual_staffing': 'actual',
                'scale_to_effective_capabilities':True,  # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                'yearly_HR_scaling_mode': ['historical_scaling', 'historical_scaling_maintained', 'historical_scaling', 'historical_scaling_maintained'][draw_number],
                'equip_availability':'all',
                'beds_availability':'all',
                },
            "ImprovedHealthSystemAndCareSeekingScenarioSwitcher": {
                "max_healthsystem_function": [False, False],
                "max_healthcare_seeking": [False, False],
                "year_of_switch": 2020,
            }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
