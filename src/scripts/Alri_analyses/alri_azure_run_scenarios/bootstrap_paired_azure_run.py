from tlo import Date
from tlo import logging
from pathlib import Path
from tlo.scenario import BaseScenario
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.methods.alri import (
    AlriIncidentCase,
    AlriPollingEvent,
    AlriPropertiesOfOtherModules,
    HSI_Alri_Treatment,
    _make_hw_diagnosis_perfect,
    _make_perfect_conditions,
    _make_treatment_and_diagnosis_perfect,
    _reduce_hw_dx_sensitivity,
    _prioritise_oxygen_to_hospitals

)

class MyTestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 1
        self.start_date = Date(2024, 1, 1)
        self.number_of_draws = 1


def log_configuration(self):
    return {
        'filename': 'my_test_scenario', 'directory': './outputs',
        'custom_levels': {'*': logging.INFO}
    }

resourcefilepath = Path('./resources')


def modules(self):
    return [
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(
            resourcefilepath=resourcefilepath,
            force_any_symptom_to_lead_to_healthcareseeking=True,
        ),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  cons_availability='all',
                                  ),
        alri.Alri(resourcefilepath=resourcefilepath),
        AlriPropertiesOfOtherModules(),
    ]

def draw_parameters(self, draw_number, rng):
    return {
        'Lifestyle': {
            'init_p_urban': rng.randint(10, 20) / 100.0,
        }
    }
