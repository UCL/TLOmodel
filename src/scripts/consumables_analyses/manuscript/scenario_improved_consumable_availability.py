

"""This Scenario file run the model under different assumptions for the consumable availability in order to estimate the
cost under each scenario for the HSSP-III duration

Run on the batch system using:
```
tlo batch-submit src/scripts/consumables_analyses/manuscript/scenario_improved_consumable_availability.py
```

or locally using:
```
tlo scenario-run src/scripts/consumables_analyses/manuscript/scenario_improved_consumable_availability.py

# Pending actions
# check if 7 days of persistence
# Scale-up in 2026
# Relaxing health worker capacity constraint
# Reduced persistence of care-seeking
# Private market substitution - derive percentage from TLM data
# Don't run sensitivity analyses yet (can be added later) - only run the HR one --> 20 scenarios
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
        self.end_date = Date(2031, 1, 1) # TODO change to 2041
        # Run until 2040 even though analysis maybe focused on years until 2030
        self.pop_size = 5_000 # TODO change to 100_000
        self.scenarios = list(range(0,13)) # add scenarios as necessary
        self.number_of_draws = len(self.scenarios)
        self.runs_per_draw = 1 #TODO change to 5

    def log_configuration(self):
        return {
            'filename': 'consumables_costing',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthsystem": logging.INFO, # TODO Confirm whether this needs to be logged
            }
        }

    def modules(self):
        return (
                fullmodel()
                + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher()]
        )

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'cons_availability': 'default',
                'year_cons_availability_switch': 2026,
                'cons_availability_postSwitch': ['default', # Actual
                                                 'scenario1', 'scenario2', 'scenario3', # Predictive factors scenarios
                                                 'scenario6', 'scenario7', 'scenario8', # Benchmark facility scenarios
                                                 # TODO add redistribution scenarios
                                                 'all', # Perfect
                                                 'default',
                                                 'scenario1', 'scenario2', 'scenario3',
                                                 'scenario6', 'scenario7', 'scenario8',
                                                 'all'][draw_number],
                'mode_appt_constraints':1,
                'mode_appt_constraints_postSwitch':[x for x in (1, 2) for _ in range(8)][draw_number], # once without HR constraints and once with HR constraints
                'year_mode_switch':2026,
                'policy_name': 'HSSP-III',
                'use_funded_or_actual_staffing': 'actual',
                'scale_to_effective_capabilities':True,  # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                'yearly_HR_scaling_mode': 'historical_scaling', # allow historical HRH scaling to occur 2018-2024
                'equip_availability':'all',
                'beds_availability':'all',
                },
            "ImprovedHealthSystemAndCareSeekingScenarioSwitcher": {
                "max_healthsystem_function": [x for x in ([False, False], [False, True]) for _ in range(8)][draw_number],
                "max_healthcare_seeking": [x for x in ([False, False], [False, True]) for _ in range(8)][draw_number],
                "year_of_switch": 2026,
            }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
