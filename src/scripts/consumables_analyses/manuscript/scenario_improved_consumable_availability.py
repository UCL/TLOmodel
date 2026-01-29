

"""This Scenario file run the model under different assumptions for the consumable availability in order to estimate the
cost under each scenario for the HSSP-III duration

Run on the batch system using:
```
tlo batch-submit src/scripts/consumables_analyses/manuscript/scenario_improved_consumable_availability.py
```

or locally using:
```
tlo scenario-run src/scripts/consumables_analyses/manuscript/scenario_improved_consumable_availability.py

# TODO Pending actions
# Private market substitution - derive percentage from TLM data
# Don't run sensitivity analyses yet (can be added later) - only run the HR one --> 24 scenarios
 ```

"""
from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario

class ConsumablesImpact(BaseScenario):
    # -----------------------------
    # 1) DEFINE SCENARIOS EXPLICITLY
    # -----------------------------
    CONSUMABLE_SCENARIOS = [
        'default',
        'scenario1', 'scenario2', 'scenario3',  # Predictive factors
        'scenario6', 'scenario7', 'scenario8',  # Benchmark facilities
        'scenario16', 'scenario17', 'scenario18', 'scenario19',  # Redistribution
        'all'  # Perfect
    ]

    SYSTEM_MODES = [
        {
            "mode_appt_constraints": 2,
            "max_healthsystem_function": [False, False],
            "max_healthcare_seeking": [False, False],
        },
        {
            "mode_appt_constraints": 1,
            "max_healthsystem_function": [False, True],
            "max_healthcare_seeking": [False, True],
        },
    ]

    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2028, 1, 1) # TODO change to 2041
        # Run until 2040 even though analysis maybe focused on years until 2030
        self.pop_size = 5_000 # TODO change to 100_000


        # Build cartesian product of scenarios
        self.SCENARIOS = [
            (cons, sys)
            for cons in self.CONSUMABLE_SCENARIOS
            for sys in self.SYSTEM_MODES
        ]

        self.number_of_draws = len(self.SCENARIOS)
        self.scenarios = list(range(self.number_of_draws))

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
            }
        }

    def modules(self):
        return (
                fullmodel()
                + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher()]
        )

    def draw_parameters(self, draw_number, rng):
        cons_scenario, sys = self.SCENARIOS[draw_number]

        return {
            'HealthSystem': {
                'cons_availability': 'default',
                'data_source_for_cons_availability_estimates': 'updated',
                'year_cons_availability_switch': 2026,
                'cons_availability_postSwitch': cons_scenario,
                'mode_appt_constraints':1,
                'mode_appt_constraints_postSwitch':sys["mode_appt_constraints"], # once without HR constraints and once with HR constraints
                'year_mode_switch':2026,
                'policy_name': 'EHP_III',
                'use_funded_or_actual_staffing': 'actual',
                'scale_to_effective_capabilities':True,  # <-- Transition into Mode2 with the effective capabilities in HRH 'revealed' in Mode 1
                'yearly_HR_scaling_mode': 'historical_scaling', # allow historical HRH scaling to occur 2018-2024
                'equip_availability':'all',
                'beds_availability':'all',
                },
            "ImprovedHealthSystemAndCareSeekingScenarioSwitcher": {
                "max_healthsystem_function": sys["max_healthsystem_function"],
                "max_healthcare_seeking": sys["max_healthcare_seeking"],
                "year_of_switch": 2026,
            }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
