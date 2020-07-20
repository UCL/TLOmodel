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
from tlo.methods.symptommanager import Symptom, DuplicateSymptomWithNonIdenticalPropertiesError

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = './resources'

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)
popsize = 200


def test_make_a_symptom():
    symp = Symptom(name='weird_sense_of_deja_vu')

    assert isinstance(symp, Symptom)

    # check contents and the values defaulted to.
    assert hasattr(symp, 'name')
    assert hasattr(symp, 'no_healthcareseeking_in_children')
    assert hasattr(symp, 'no_healthcareseeking_in_adults')
    assert hasattr(symp, 'emergency_in_children')
    assert hasattr(symp, 'emergency_in_adults')
    assert hasattr(symp, 'odds_ratio_health_seeking_in_children')
    assert hasattr(symp, 'odds_ratio_health_seeking_in_adults')

    assert symp.no_healthcareseeking_in_children is False
    assert symp.no_healthcareseeking_in_adults is False

    assert symp.emergency_in_children is False
    assert symp.emergency_in_adults is False

    assert symp.odds_ratio_health_seeking_in_children == 1.0
    assert symp.odds_ratio_health_seeking_in_adults == 1.0





def test_register_duplicate_symptoms():
    symp = Symptom(name='symptom')
    symp_with_different_properties = Symptom(name='symptom', emergency_in_children=True)

    sm = symptommanager.SymptomManager(resourcefilepath=resourcefilepath)

    # register original
    sm.register_symptom(symp)
    assert 1 == len(sm.all_registered_symptoms)
    assert 1 == len(sm.symptom_names)

    # register duplicate (same name and same properties): should give no error
    sm.register_symptom(symp)
    assert 1 == len(sm.all_registered_symptoms)
    assert 1 == len(sm.symptom_names)

    # register non-identical duplicate (same name but different properties): should error
    created_error = False
    try:
        sm.register_symptom(symp_with_different_properties)
    except DuplicateSymptomWithNonIdenticalPropertiesError:
        created_error = True

    assert created_error






def test_no_symptoms_if_no_diseases():
    sim = Simulation(start_date=start_date)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    generic_symptoms = list(sim.modules['SymptomManager'].parameters['generic_symptoms'])

    for symp in generic_symptoms:
        # No one should have any symptom currently (as no disease modules registered)
        assert list() == sim.modules['SymptomManager'].who_has(symp)

def test_adding_quering_and_removing_symptoms():
    sim = Simulation(start_date=start_date)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that symptoms are add to checked and removed correctly
    df = sim.population.props

    symp = 'backache'  # this is a generic symptoms that is not added by any disease module that is registered

    # No one should have any symptom currently
    assert list() == sim.modules['SymptomManager'].who_has(symp)

    # Add the symptom to a random selection of people
    ids = list(sim.rng.choice(list(df.index[df.is_alive]), 5))

    sim.modules['SymptomManager'].change_symptom(
        symptom_string=symp,
        person_id=ids,
        add_or_remove='+',
        disease_module=sim.modules['Mockitis']
    )

    # Check who_has() and has_what()
    has_symp = sim.modules['SymptomManager'].who_has(symp)
    assert set(has_symp) == set(ids)

    for person_id in ids:
        assert [symp] == sim.modules['SymptomManager'].has_what(person_id=person_id, disease_module=sim.modules['Mockitis'])

    # Check cause of the symptom:
    for person in ids:
        causes = sim.modules['SymptomManager'].causes_of(person, symp)
        assert 'Mockitis' in causes
        assert 1 == len(causes)

    # Remove the symptoms:
    for person in ids:
        sim.modules['SymptomManager'].clear_symptoms(person, disease_module=sim.modules['Mockitis'])

    assert list() == sim.modules['SymptomManager'].who_has(symp)

def test_spurious_symptoms():
    sim = Simulation(start_date=start_date)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + DateOffset(days=2))

    generic_symptoms = list(sim.modules['SymptomManager'].parameters['generic_symptoms'])

    # Someone should have any symptom currently (because spurious_symptoms are being generated)
    has_any_generic_symptom = []
    for symp in generic_symptoms:
        has_this_symptom = sim.modules['SymptomManager'].who_has(symp)
        if has_this_symptom:
            has_any_generic_symptom = has_any_generic_symptom + has_this_symptom

    assert len(has_any_generic_symptom) > 0
