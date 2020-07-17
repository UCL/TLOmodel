"""Test for HealthCareSeeking Module

* Disease modules can declare the effect that defined symptoms have on healthcare seeking
* The SymptomManager module declares generic symptoms and specifies the healthcare seeking effect
* The HealthCareSeekingModule constructs a LinearModel from all the specified symptoms
"""

import os
from pathlib import Path

from pandas import DateOffset
from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    labour,
    mockitis,
    pregnancy_supervisor,
    symptommanager, chronicsyndrome,
)
from tlo.methods.symptommanager import Symptom

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = './resources'


def test_make_a_symptom():
    symp = Symptom(name='weird_sense_of_deja_vu')

    assert isinstance(symp, Symptom)

    # check contents and defaults
    assert 'name' in dir(symp)
    assert 'emergency_in_adults' in dir(symp)
    assert 'emergency_in_children' in dir(symp)
    assert 'odds_ratio_health_seeking_in_adults' in dir(symp)
    assert 'odds_ratio_health_seeking_in_children' in dir(symp)

    assert symp.emergency_in_adults is False
    assert symp.emergency_in_children is False
    assert symp.odds_ratio_health_seeking_in_adults == 1.0
    assert symp.odds_ratio_health_seeking_in_children == 1.0



def test_disease_module_declare_the_effect_that_a_symptom_has_on_healthcareseking():
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date)

    # Register the core modules including Chronic Syndrome and Mockities
    # Mockitis symptoms are not associated with any health care seeking, Chronic Syndrom symptoms are
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation for one day

    end_date = start_date + DateOffset(days=1)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that the healthcareseeking LinearModel is as expected.




# Need to rule the use of
