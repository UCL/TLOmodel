import os
from pathlib import Path

import pytest
from pandas import DateOffset

from tlo import Date, Simulation
from tlo.methods import (
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    simplified_births,
    symptommanager,
)
from tlo.methods.healthseekingbehaviour import HIGH_ODDS_RATIO
from tlo.methods.symptommanager import (
    DuplicateSymptomWithNonIdenticalPropertiesError,
    Symptom,
    SymptomManager_AutoOnsetEvent,
    SymptomManager_AutoResolveEvent,
    SymptomManager_SpuriousSymptomOnset,
)

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
    assert hasattr(symp, 'prob_seeks_emergency_appt_in_children')
    assert hasattr(symp, 'prob_seeks_emergency_appt_in_adults')
    assert hasattr(symp, 'odds_ratio_health_seeking_in_children')
    assert hasattr(symp, 'odds_ratio_health_seeking_in_adults')

    assert symp.no_healthcareseeking_in_children is False
    assert symp.no_healthcareseeking_in_adults is False

    assert symp.prob_seeks_emergency_appt_in_children == 0.0
    assert symp.prob_seeks_emergency_appt_in_adults == 0.0

    assert symp.odds_ratio_health_seeking_in_children == 1.0
    assert symp.odds_ratio_health_seeking_in_adults == 1.0


def test_emergency_symptom_defined_through_static_method():
    """Check that can create an emergency symptom and that is has the expected properties by default"""
    # Emergency in adults and children
    symp = Symptom.emergency('emergency_symptom')
    assert isinstance(symp, Symptom)
    assert symp.no_healthcareseeking_in_children is False
    assert symp.no_healthcareseeking_in_adults is False
    assert symp.prob_seeks_emergency_appt_in_children == 1.0
    assert symp.prob_seeks_emergency_appt_in_adults == 1.0
    assert symp.odds_ratio_health_seeking_in_children >= HIGH_ODDS_RATIO
    assert symp.odds_ratio_health_seeking_in_adults >= HIGH_ODDS_RATIO

    # Emergency in adults only
    symp = Symptom.emergency(name='emergency_symptom', which='adults')
    assert isinstance(symp, Symptom)
    assert symp.no_healthcareseeking_in_children is False
    assert symp.no_healthcareseeking_in_adults is False
    assert symp.prob_seeks_emergency_appt_in_children == 0.0
    assert symp.prob_seeks_emergency_appt_in_adults == 1.0
    assert symp.odds_ratio_health_seeking_in_children == 0.0
    assert symp.odds_ratio_health_seeking_in_adults >= HIGH_ODDS_RATIO

    # Emergency in children only
    symp = Symptom.emergency(name='emergency_symptom', which='children')
    assert isinstance(symp, Symptom)
    assert symp.no_healthcareseeking_in_children is False
    assert symp.no_healthcareseeking_in_adults is False
    assert symp.prob_seeks_emergency_appt_in_children == 1.0
    assert symp.prob_seeks_emergency_appt_in_adults == 0.0
    assert symp.odds_ratio_health_seeking_in_children >= HIGH_ODDS_RATIO
    assert symp.odds_ratio_health_seeking_in_adults == 0.0


def test_register_duplicate_symptoms():
    symp = Symptom(name='symptom')
    symp_duplicate = Symptom(name='symptom')
    symp_with_different_properties = Symptom.emergency(name='symptom')
    symp_with_different_name = Symptom(name='symptom_a')

    sm = symptommanager.SymptomManager(resourcefilepath=resourcefilepath)

    # register original
    sm.register_symptom(symp)
    assert 1 == len(sm.all_registered_symptoms)
    assert 1 == len(sm.symptom_names)

    # register duplicate (same name and same properties but different instance): should give no error
    sm.register_symptom(symp_duplicate)
    assert 1 == len(sm.all_registered_symptoms)
    assert 1 == len(sm.symptom_names)

    # register non-identical duplicate (same name but different properties): should error
    created_error = False
    try:
        sm.register_symptom(symp_with_different_properties)
    except DuplicateSymptomWithNonIdenticalPropertiesError:
        created_error = True

    assert created_error

    # register a second, which is different: should accept it:
    sm.register_symptom(symp_with_different_name)
    assert 2 == len(sm.all_registered_symptoms)
    assert 2 == len(sm.symptom_names)


def test_no_symptoms_if_no_diseases(seed):
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    for symp in sim.modules['SymptomManager'].generic_symptoms:
        # No one should have any symptom currently (as no disease modules registered)
        assert list() == sim.modules['SymptomManager'].who_has(symp)


def test_adding_quering_and_removing_symptoms(seed):
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Make the population:
    sim.make_initial_population(n=popsize)

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
        assert symp in sim.modules['SymptomManager'].has_what(person_id=person_id,
                                                              disease_module=sim.modules['Mockitis'])

    # Check cause of the symptom:
    for person in ids:
        causes = sim.modules['SymptomManager'].causes_of(person, symp)
        assert 'Mockitis' in causes
        assert 1 == len(causes)

    # Remove the symptoms:
    for person in ids:
        sim.modules['SymptomManager'].clear_symptoms(person, disease_module=sim.modules['Mockitis'])

    assert list() == sim.modules['SymptomManager'].who_has(symp)


def test_baby_born_has_no_symptoms(seed):
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + DateOffset(days=1))

    # do a birth (from a randomly chosen mother)
    df = sim.population.props
    mother_id = df.loc[df.sex == 'F'].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date - DateOffset(months=9)

    person_id = sim.do_birth(mother_id)

    # check that the new person does not have symptoms:
    assert [] == sim.modules['SymptomManager'].has_what(person_id)


def test_auto_onset_symptom(seed):
    """Test to check that symptoms that are delayed in onset work as expected.
    """
    # Generate a simulation:
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
                 mockitis.Mockitis()
                 )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + DateOffset(days=0))
    sim.event_queue.queue = []

    sm = sim.modules['SymptomManager']

    # Select a person and make them alive and no symptoms
    person_id = 0
    sim.population.props.loc[person_id, 'is_alive'] = True
    for symptom in sm.symptom_names:
        sim.population.props.loc[person_id, sm.get_column_name_for_symptom(symptom)] = 0
    assert 0 == len(sm.has_what(person_id))

    def get_events_in_sim():
        return [ev for ev in sim.event_queue.queue if (person_id in ev[3].person_id)]

    assert 0 == len(get_events_in_sim())

    # The symptom:
    symptom_string = 'weird_sense_of_deja_vu'
    duration_in_days = 10
    date_of_onset = sim.date + DateOffset(days=5)

    # Mockitis to schedule the onset of a symptom for a date in the future
    sm.change_symptom(
        person_id=person_id,
        symptom_string=symptom_string,
        add_or_remove='+',
        duration_in_days=duration_in_days,
        date_of_onset=date_of_onset,
        disease_module=sim.modules['Mockitis']
    )

    # check that the symptom is not imposed
    assert 0 == len(sm.has_what(person_id))

    # get the future events for this person (should be just the auto-onset event)
    assert 1 == len(get_events_in_sim())
    onset = get_events_in_sim()[0]

    assert onset[0] == date_of_onset
    assert isinstance(onset[3], SymptomManager_AutoOnsetEvent)

    # run the events and check for the changing of symptoms
    sim.date = date_of_onset
    onset[3].apply(sim.population)
    assert symptom_string in sm.has_what(person_id)

    # get the future events for this person (should now include the auto-resolve event)
    assert 2 == len(get_events_in_sim())
    resolve = get_events_in_sim()[1]

    assert resolve[0] == date_of_onset + DateOffset(days=duration_in_days)
    assert isinstance(resolve[3], SymptomManager_AutoResolveEvent)

    resolve[3].apply(sim.population)
    assert 0 == len(sm.has_what(person_id))


def test_nonemergency_spurious_symptoms_during_simulation(seed):
    """Test on the functionality of a generic non-emergency spurious symptom"""
    the_generic_symptom = 'fever'

    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
                 )

    # Make the probability of onset of one of the generic symptoms be 1.0 and duration of one day
    generic_symptoms = sim.modules['SymptomManager'].parameters['generic_symptoms_spurious_occurrence']

    generic_symptoms.loc[
        (the_generic_symptom == generic_symptoms['name']),
        ['prob_spurious_occurrence_in_children_per_day',
         'prob_spurious_occurrence_in_adults_per_day']
    ] = (1.0, 1.0)

    generic_symptoms.loc[
        (the_generic_symptom == generic_symptoms['name']),
        ['duration_in_days_of_spurious_occurrence_in_children',
         'duration_in_days_of_spurious_occurrence_in_adults']
    ] = (1, 1)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + DateOffset(days=0))

    # Check that no one has symptoms
    assert [] == sim.modules['SymptomManager'].who_has(the_generic_symptom)

    # Run the onset event & check that all persons now have the generic symptom
    onset = SymptomManager_SpuriousSymptomOnset(module=sim.modules['SymptomManager'])
    onset.apply(sim.population)
    df = sim.population.props
    assert set(df.is_alive.index) == set(sim.modules['SymptomManager'].who_has(the_generic_symptom))

    # Update time, run resolve event and check that no one has symptom
    sim.date += DateOffset(days=1)
    sim.modules['SymptomManager'].spurious_symptom_resolve_event.apply(sim.population)
    assert [] == sim.modules['SymptomManager'].who_has(the_generic_symptom)


def test_emergency_spurious_symptom_during_simulation(seed):
    """Test on the functionality of the spurious emergency symptom"""
    emergency_spurious_symptom = 'spurious_emergency_symptom'

    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
                 )

    # Make the probability of onset of the spurious emergency symptom be 1.0 and duration of one day
    generic_symptoms = sim.modules['SymptomManager'].parameters['generic_symptoms_spurious_occurrence']
    generic_symptoms.loc[
        (emergency_spurious_symptom == generic_symptoms['name']),
        ['prob_spurious_occurrence_in_children_per_day',
         'prob_spurious_occurrence_in_adults_per_day']
    ] = (1.0, 1.0)

    generic_symptoms.loc[
        (emergency_spurious_symptom == generic_symptoms['name']),
        ['duration_in_days_of_spurious_occurrence_in_children',
         'duration_in_days_of_spurious_occurrence_in_adults']
    ] = (1, 1)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + DateOffset(days=0))

    # Check that no one has symptoms
    assert [] == sim.modules['SymptomManager'].who_has(emergency_spurious_symptom)

    # Run the onset event & check that all persons now have the generic symptom
    onset = SymptomManager_SpuriousSymptomOnset(module=sim.modules['SymptomManager'])
    onset.apply(sim.population)
    df = sim.population.props
    assert len(df.is_alive.index) > 0
    assert set(df.is_alive.index) == set(sim.modules['SymptomManager'].who_has(emergency_spurious_symptom))

    # Update time, run resolve event and check that no one has symptom
    sim.date += DateOffset(days=1)
    sim.modules['SymptomManager'].spurious_symptom_resolve_event.apply(sim.population)
    assert [] == sim.modules['SymptomManager'].who_has(emergency_spurious_symptom)


@pytest.fixture
def symptom_manager():
    return symptommanager.SymptomManager(resourcefilepath=resourcefilepath)


@pytest.fixture
def disease_module():
    return mockitis.Mockitis(resourcefilepath=resourcefilepath)


@pytest.fixture()
def disease_module_symptoms():
    return (
        'weird_sense_of_deja_vu', 'coughing_and_irritable', 'extreme_pain_in_the_nose'
    )


@pytest.fixture
def simulation(seed):
    return Simulation(start_date=start_date, seed=seed)


def register_modules_and_initialise(simulation, *modules):
    simulation.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        *modules
    )
    simulation.make_initial_population(n=popsize)


def test_disease_module_in_recognised_module_names(
    symptom_manager, disease_module, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    assert disease_module.name in symptom_manager.recognised_module_names


def test_disease_module_symptoms_registered(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    assert symptom_manager.symptom_names.issuperset(disease_module_symptoms)


def test_who_has_single_symptom(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        persons_with_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] > 0)
        ]
        # Check at least some people have symptom so that check is non-trivial
        assert len(persons_with_symptom) != 0, 'Initial population too small'
        assert set(symptom_manager.who_has(symptom)) == set(persons_with_symptom)


def test_who_has_multiple_symptoms(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    persons_with_symptom = df.index[
        df.is_alive
        & (
            df[symptom_manager.get_column_name_for_symptom(disease_module_symptoms[0])]
            > 0
        )
        & (
            df[symptom_manager.get_column_name_for_symptom(disease_module_symptoms[1])]
            > 0
        )
    ]
    assert set(
        symptom_manager.who_has(disease_module_symptoms[:2])
    ) == set(persons_with_symptom)


def test_who_not_have(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        persons_without_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] == 0)
        ]
        # Check at least some people do not have symptom so that check is non-trivial
        assert len(persons_without_symptom) != 0
        assert set(
            df.index[symptom_manager.who_not_have(symptom)]
        ) == set(persons_without_symptom)


def test_has_what(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        person_with_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] > 0)
        ][0]
        assert symptom in symptom_manager.has_what(person_with_symptom)
        person_without_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] == 0)
        ][0]
        assert symptom not in symptom_manager.has_what(person_without_symptom)


def test_has_what_disease_module(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        person_with_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] > 0)
        ][0]
        assert symptom in symptom_manager.has_what(person_with_symptom, disease_module)
        person_without_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] == 0)
        ][0]
        assert symptom not in symptom_manager.has_what(person_without_symptom, disease_module)


def test_have_what(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        persons_with_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] > 0)
        ]
        assert all(
            symptom in symptom_set
            for symptom_set in symptom_manager.have_what(persons_with_symptom)
        )
        persons_without_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] == 0)
        ]
        assert all(
            symptom not in symptom_set
            for symptom_set in symptom_manager.have_what(persons_without_symptom)
        )


def test_causes_of(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        person_with_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] > 0)
        ][0]
        assert disease_module.name in symptom_manager.causes_of(
            person_with_symptom, symptom
        )


def test_clear_symptoms_individual(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        person_with_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] > 0)
        ][0]
        symptom_manager.clear_symptoms(person_with_symptom, disease_module)
        assert all(
            disease_module.name not in symptom_manager.causes_of(person_with_symptom, s)
            for s in symptom_manager.symptom_names
        )


def test_clear_symptoms_multiple_persons(
        symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        persons_with_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] > 0)
        ]
        symptom_manager.clear_symptoms(persons_with_symptom, disease_module)
        assert all(
            disease_module.name not in symptom_manager.causes_of(p, s)
            for s in symptom_manager.symptom_names
            for p in persons_with_symptom
        )


def test_remove_symptom_individual_single_symptom(
        symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        person_with_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] > 0)
        ][0]
        symptom_manager.change_symptom(
            person_with_symptom, symptom, '-', disease_module
        )
        assert disease_module.name not in symptom_manager.causes_of(
            person_with_symptom, symptom
        )


def test_remove_symptom_multiple_persons_single_symptom(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        persons_with_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] > 0)
        ]
        symptom_manager.change_symptom(
            persons_with_symptom, symptom, '-', disease_module
        )
        assert all(
            disease_module.name not in symptom_manager.causes_of(p, symptom)
            for p in persons_with_symptom
        )


def test_add_symptom_individual_single_symptom(
        symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        person_without_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] == 0)
        ][0]
        symptom_manager.change_symptom(
            person_without_symptom, symptom, '+', disease_module
        )
        assert disease_module.name in symptom_manager.causes_of(
            person_without_symptom, symptom
        )


def test_add_symptom_multiple_persons_single_symptom(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    for symptom in disease_module_symptoms:
        persons_without_symptom = df.index[
            df.is_alive
            & (df[symptom_manager.get_column_name_for_symptom(symptom)] == 0)
        ]
        symptom_manager.change_symptom(
            persons_without_symptom, symptom, '+', disease_module
        )
        assert all(
            disease_module.name in symptom_manager.causes_of(p, symptom)
            for p in persons_without_symptom
        )


def test_change_symptom_individual_multiple_symptoms(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    alive_person = df.index[df.is_alive][0]
    symptom_manager.change_symptom(
        alive_person, disease_module_symptoms, '+', disease_module
    )
    assert all(
        disease_module.name in symptom_manager.causes_of(alive_person, s)
        for s in disease_module_symptoms
    )
    symptom_manager.change_symptom(
        alive_person, disease_module_symptoms, '-', disease_module
    )
    assert all(
        disease_module.name not in symptom_manager.causes_of(alive_person, s)
        for s in disease_module_symptoms
    )


def test_change_symptom_multiple_persons_multiple_symptoms(
    symptom_manager, disease_module, disease_module_symptoms, simulation
):
    register_modules_and_initialise(simulation, symptom_manager, disease_module)
    df = simulation.population.props
    alive_persons = df.index[df.is_alive]
    symptom_manager.change_symptom(
        alive_persons, disease_module_symptoms, '+', disease_module
    )
    assert all(
        disease_module.name in symptom_manager.causes_of(p, s)
        for s in disease_module_symptoms
        for p in alive_persons
    )
    symptom_manager.change_symptom(
        alive_persons, disease_module_symptoms, '-', disease_module
    )
    assert all(
        disease_module.name not in symptom_manager.causes_of(p, s)
        for s in disease_module_symptoms
        for p in alive_persons
    )
