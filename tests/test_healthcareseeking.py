"""Test for HealthCareSeeking Module"""
import os
from pathlib import Path

from pandas import DateOffset

from tlo import Date, Module, Simulation
from tlo.methods import (
    Metadata,
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    simplified_births,
    symptommanager,
)
from tlo.methods.hsi_generic_first_appts import (
    HSI_GenericEmergencyFirstApptAtFacilityLevel1,
    HSI_GenericFirstApptAtFacilityLevel0,
)
from tlo.methods.symptommanager import Symptom

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = './resources'


def test_healthcareseeking_does_occur_from_symptom_that_does_give_healthcareseeking_behaviour(seed):
    """test that a symptom that gives healthcare seeking results in generic HSI scheduled."""

    class DummyDisease(Module):
        METADATA = {Metadata.USES_SYMPTOMMANAGER}
        """Dummy Disease - it's only job is to create a symptom and impose it everyone"""
        def read_parameters(self, data_folder):
            self.sim.modules['SymptomManager'].register_symptom(
                Symptom(
                    name='Symptom_that_does_cause_healthcare_seeking',
                ),
            )

        def initialise_population(self, population):
            """Give everyone the symptom"""
            df = population.props
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=list(df.loc[df.is_alive].index),
                disease_module=self,
                symptom_string='Symptom_that_does_cause_healthcare_seeking',
                add_or_remove='+'
            )

        def initialise_simulation(self, sim):
            pass

        def on_birth(self, mother, child):
            pass

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 DummyDisease()
                 )

    # Run the simulation for zero days
    end_date = start_date + DateOffset(days=1)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that the symptom has been registered and is flagged as not causing healthcare seeking
    assert 'Symptom_that_does_cause_healthcare_seeking' in \
           sim.modules['SymptomManager'].symptom_names
    assert 'Symptom_that_does_cause_healthcare_seeking' not in \
           sim.modules['HealthSeekingBehaviour'].no_healthcareseeking_in_children
    assert 'Symptom_that_does_cause_healthcare_seeking' not in \
           sim.modules['HealthSeekingBehaviour'].no_healthcareseeking_in_adults

    # Check that everyone has the symptom
    df = sim.population.props
    assert set(df.loc[df.is_alive].index) == set(
        sim.modules['SymptomManager'].who_has('Symptom_that_does_cause_healthcare_seeking'))

    # Check that an Non-Emergency Generic HSI and no Emergency Generic HSI events are scheduled
    q = sim.modules['HealthSystem'].HSI_EVENT_QUEUE
    assert any([isinstance(e[4], HSI_GenericFirstApptAtFacilityLevel0) for e in q])
    assert not any([isinstance(e[4], HSI_GenericEmergencyFirstApptAtFacilityLevel1) for e in q])


def test_healthcareseeking_does_not_occurs_from_symptom_that_do_not_give_healthcareseeking_behaviour(seed):
    """test that a symptom that should not give healthseeeking does not give heaslth seeking."""

    class DummyDisease(Module):
        METADATA = {Metadata.USES_SYMPTOMMANAGER}
        """Dummy Disease - it's only job is to create a symptom and impose it everyone"""
        def read_parameters(self, data_folder):
            self.sim.modules['SymptomManager'].register_symptom(
                Symptom(
                    name='Symptom_that_does_not_cause_healthcare_seeking',
                    no_healthcareseeking_in_adults=True,
                    no_healthcareseeking_in_children=True,
                ),
            )

        def initialise_population(self, population):
            """Give everyone the symptom"""
            df = population.props
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=list(df.loc[df.is_alive].index),
                disease_module=self,
                symptom_string='Symptom_that_does_not_cause_healthcare_seeking',
                add_or_remove='+'
            )

        def initialise_simulation(self, sim):
            pass

        def on_birth(self, mother, child):
            pass

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 DummyDisease()
                 )

    # Run the simulation for zero days
    end_date = start_date + DateOffset(days=1)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that the symptom has been registered and is flagged as not causing healthcare seeking
    assert 'Symptom_that_does_not_cause_healthcare_seeking' in \
           sim.modules['SymptomManager'].symptom_names
    assert 'Symptom_that_does_not_cause_healthcare_seeking' in \
           sim.modules['HealthSeekingBehaviour'].no_healthcareseeking_in_children
    assert 'Symptom_that_does_not_cause_healthcare_seeking' in \
           sim.modules['HealthSeekingBehaviour'].no_healthcareseeking_in_adults

    # Check that everyone has the symptom
    df = sim.population.props
    assert set(df.loc[df.is_alive].index) == set(
        sim.modules['SymptomManager'].who_has('Symptom_that_does_not_cause_healthcare_seeking'))

    # Check that no Non-Emergency Generic HSI and no Emergency Generic HSI events are scheduled
    q = sim.modules['HealthSystem'].HSI_EVENT_QUEUE
    assert not any([isinstance(e[4], HSI_GenericFirstApptAtFacilityLevel0) for e in q])
    assert not any([isinstance(e[4], HSI_GenericEmergencyFirstApptAtFacilityLevel1) for e in q])


def test_healthcareseeking_does_occur_from_symptom_that_does_give_emergency_healthcareseeking_behaviour(seed):
    """test that a symptom that give emergency healthcare seeking results in emergency HSI scheduled."""

    class DummyDisease(Module):
        METADATA = {Metadata.USES_SYMPTOMMANAGER}
        """Dummy Disease - it's only job is to create a symptom and impose it everyone"""
        def read_parameters(self, data_folder):
            self.sim.modules['SymptomManager'].register_symptom(
                Symptom(
                    name='Symptom_that_does_cause_emergency_healthcare_seeking',
                    emergency_in_adults=True,
                    emergency_in_children=True
                ),
            )

        def initialise_population(self, population):
            """Give everyone the symptom"""
            df = population.props
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=list(df.loc[df.is_alive].index),
                disease_module=self,
                symptom_string='Symptom_that_does_cause_emergency_healthcare_seeking',
                add_or_remove='+'
            )

        def initialise_simulation(self, sim):
            pass

        def on_birth(self, mother, child):
            pass

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 DummyDisease()
                 )

    # Run the simulation for zero days
    end_date = start_date + DateOffset(days=1)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that the symptom has been registered and is flagged as not causing healthcare seeking
    assert 'Symptom_that_does_cause_emergency_healthcare_seeking' in \
           sim.modules['SymptomManager'].symptom_names
    assert 'Symptom_that_does_cause_emergency_healthcare_seeking' in \
           sim.modules['HealthSeekingBehaviour'].emergency_in_children
    assert 'Symptom_that_does_cause_emergency_healthcare_seeking' in \
           sim.modules['HealthSeekingBehaviour'].emergency_in_adults

    # Check that everyone has the symptom
    df = sim.population.props
    assert set(df.loc[df.is_alive].index) == set(
        sim.modules['SymptomManager'].who_has('Symptom_that_does_cause_emergency_healthcare_seeking'))

    # Check that no Non-Emergency Generic HSI and no Emergency Generic HSI events are scheduled or have happened
    q = sim.modules['HealthSystem'].HSI_EVENT_QUEUE
    assert not any([isinstance(e[4], HSI_GenericFirstApptAtFacilityLevel0) for e in q])
    assert any([isinstance(e[4], HSI_GenericEmergencyFirstApptAtFacilityLevel1) for e in q])


def test_no_healthcareseeking_when_no_spurious_symptoms_and_no_disease_modules(seed):
    """there should be no generic HSI if there are no spurious symptoms or disease module"""
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules including Chronic Syndrome and Mockitis -
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 )

    # Run the simulation for one day
    end_date = start_date + DateOffset(days=1)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that Generic HSI events are scheduled
    q = sim.modules['HealthSystem'].HSI_EVENT_QUEUE
    assert not any([isinstance(e[4], HSI_GenericFirstApptAtFacilityLevel0) for e in q])
    assert not any([isinstance(e[4], HSI_GenericEmergencyFirstApptAtFacilityLevel1) for e in q])


def test_healthcareseeking_occurs_with_spurious_symptoms_only(seed):
    """spurious symptoms should generate non-emergency HSI"""
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules including Chronic Syndrome and Mockitis -
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )

    # Make spurious symptoms certain to occur and cause non-emergency care-seeking:
    sim.modules['SymptomManager'].parameters['generic_symptoms_spurious_occurrence'][[
        'prob_spurious_occurrence_in_children_per_day',
        'prob_spurious_occurrence_in_adults_per_day'
    ]] = 1.0
    sim.modules['SymptomManager'].parameters['generic_symptoms_spurious_occurrence'][[
        'odds_ratio_for_health_seeking_in_children',
        'odds_ratio_for_health_seeking_in_adults'
    ]] = 10.0

    # Run the simulation for one day
    end_date = start_date + DateOffset(days=1)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that Generic Non-Emergency HSI events are scheduled but not Emergency HSI
    q = sim.modules['HealthSystem'].HSI_EVENT_QUEUE
    assert any([isinstance(e[4], HSI_GenericFirstApptAtFacilityLevel0) for e in q])
    assert not any([isinstance(e[4], HSI_GenericEmergencyFirstApptAtFacilityLevel1) for e in q])

    # And that the persons who have those HSI do have symptoms currently:
    person_ids = [i[4].target for i in q]
    for person in person_ids:
        assert 0 < len(sim.modules['SymptomManager'].has_what(person))


def test_healthcareseeking_occurs_with_spurious_symptoms_and_disease_modules(seed):
    """Mockitis and Chronic Syndrome should lead to there being emergency and non-emergency generic HSI"""
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules including Chronic Syndrome and Mockitis -
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation for one day
    end_date = start_date + DateOffset(days=1)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that Non-Emergency Generic HSI and Emergency Generic HSI events are scheduled
    q = sim.modules['HealthSystem'].HSI_EVENT_QUEUE
    assert any([isinstance(e[4], HSI_GenericFirstApptAtFacilityLevel0) for e in q])
    assert any([isinstance(e[4], HSI_GenericEmergencyFirstApptAtFacilityLevel1) for e in q])


def test_one_per_hsi_scheduled_per_day_when_emergency_and_non_emergency_symptoms_are_onset(seed):
    """When an individual is onset with a set of symptoms including emergency and non-emergency symptoms, there should
    be only scheduled the emergency appointment."""

    class DummyDisease(Module):
        METADATA = {Metadata.USES_SYMPTOMMANAGER}
        """Dummy Disease - it's only job is to create a symptom and impose it everyone"""
        def read_parameters(self, data_folder):
            self.sim.modules['SymptomManager'].register_symptom(
                Symptom(name='NonEmergencySymptom'),
                Symptom(name='EmergencySymptom', emergency_in_adults=True, emergency_in_children=True)
            )

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            """Give person_id=0 both symptoms"""
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=[0],
                disease_module=self,
                symptom_string=['NonEmergencySymptom', 'EmergencySymptom'],
                add_or_remove='+'
            )

        def on_birth(self, mother, child):
            pass

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                               force_any_symptom_to_lead_to_healthcareseeking=True),
                 DummyDisease()
                 )

    # Initialise the simulation (run the simulation for zero days)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)

    # Run the HealthSeeingBehaviourPoll
    sim.modules['HealthSeekingBehaviour'].theHealthSeekingBehaviourPoll.run()

    # See what HSI are scheduled to occur for the person
    evs = [x[1].TREATMENT_ID for x in sim.modules['HealthSystem'].sim.modules['HealthSystem'].find_events_for_person(0)]

    assert 'FirstAttendance_Emergency' in evs
    assert 'FirstAttendance_NonEmergency' not in evs
