"""This Scenario file run the model under different assumptions for the consumable availability in order to estimate the
cost under each scenario for the HSSP-III duration

Run on the batch system using:
```
tlo batch-submit src/scripts/costing/platform_based_costing/scenario_consumable_costing.py
```

or locally using:
```
tlo scenario-run src/scripts/costing/platform_based_costing/scenario_consumable_costing.py
 ```

"""
from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario

class ConsumablesCosting(BaseScenario):

    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 4, 1)
        self.pop_size = 5000 # 100_000
        self.scenarios = [0, 1] # add scenarios as necessary
        self.number_of_draws = len(self.scenarios)
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'consumables_costing',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthsystem": logging.INFO,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.malaria": logging.INFO,
                "tlo.methods.epi": logging.INFO,
                "tlo.methods.cardio_metabolic_disorders": logging.INFO,
                "tlo.methods.wasting": logging.INFO,
            }
        }

    def modules(self):
        return (
                fullmodel(use_simplified_births = False,
                          module_kwargs={"HealthSystem": {"disable": False}}) #
                + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher()]
        )

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'cons_availability': 'default',
                'year_cons_availability_switch': 2011, # 2020
                'cons_availability_postSwitch': ['default', 'all'][draw_number], #TODO default not allowed?
                'mode_appt_constraints':1,
                'mode_appt_constraints_postSwitch':[2,1][draw_number],
                'year_mode_switch':2011,
                'policy_name': 'Default',
                'use_funded_or_actual_staffing': 'actual',
                'scale_to_effective_capabilities':True,  # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                'yearly_HR_scaling_mode': 'historical_scaling', # allow historical HRH scaling to occur 2018-2024
                'equip_availability':'all',
                'beds_availability':'all',
                },
            "ImprovedHealthSystemAndCareSeekingScenarioSwitcher": {
                "max_healthsystem_function": [[False, False], [False, True]][draw_number],
                "max_healthcare_seeking": [[False, False], [False, True]][draw_number],
                "year_of_switch": 2011,
            }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
