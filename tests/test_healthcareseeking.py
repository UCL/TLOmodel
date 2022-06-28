"""Test for HealthCareSeeking Module"""
import os
from pathlib import Path
from typing import List

import pandas as pd
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
from tlo.methods.symptommanager import Symptom

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = './resources'


def get_events_run_and_scheduled(_sim) -> List:
    """Returns a list of HSI_Events that have been run already or are scheduled to run."""
    return [ev['HSI_Event'] for ev in _sim.modules['HealthSystem'].store_of_hsi_events_that_have_run] + \
           [e[4].__class__.__name__ for e in _sim.modules['HealthSystem'].HSI_EVENT_QUEUE]


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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, store_hsi_events_that_have_run=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 DummyDisease()
                 )

    # Run the simulation for zero days
    end_date = start_date + DateOffset(days=1)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that the symptom has been registered and is flagged as causing healthcare seeking
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

    # Check that `HSI_GenericFirstApptAtFacilityLevel0` are triggered (but not
    # `HSI_GenericEmergencyFirstApptAtFacilityLevel1`)
    events_run_and_scheduled = get_events_run_and_scheduled(sim)
    assert 'HSI_GenericFirstApptAtFacilityLevel0' in events_run_and_scheduled
    assert 'HSI_GenericEmergencyFirstApptAtFacilityLevel1' not in events_run_and_scheduled


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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, store_hsi_events_that_have_run=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 DummyDisease()
                 )

    # Run the simulation for zero days
    end_date = start_date + DateOffset(days=1)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that the symptom has been registered and is flagged as _not_ causing healthcare seeking
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

    # Check no GenericFirstAppts at all
    events_run_and_scheduled = get_events_run_and_scheduled(sim)
    assert 0 == len(events_run_and_scheduled)
    assert 'HSI_GenericFirstApptAtFacilityLevel0' not in events_run_and_scheduled
    assert 'HSI_GenericEmergencyFirstApptAtFacilityLevel1' not in events_run_and_scheduled


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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, store_hsi_events_that_have_run=True),
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

    # Check that `HSI_GenericEmergencyFirstApptAtFacilityLevel1` are triggered (but not
    # `HSI_GenericFirstApptAtFacilityLevel0`)
    events_run_and_scheduled = get_events_run_and_scheduled(sim)
    assert 'HSI_GenericFirstApptAtFacilityLevel0' not in events_run_and_scheduled
    assert 'HSI_GenericEmergencyFirstApptAtFacilityLevel1' in events_run_and_scheduled
    assert all(map(lambda x: x == 'HSI_GenericEmergencyFirstApptAtFacilityLevel1', events_run_and_scheduled))


def test_no_healthcareseeking_when_no_spurious_symptoms_and_no_disease_modules(seed):
    """there should be no generic HSI if there are no spurious symptoms or disease module"""
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules including Chronic Syndrome and Mockitis -
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, store_hsi_events_that_have_run=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 )

    # Run the simulation for one day
    end_date = start_date + DateOffset(days=1)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check no GenericFirstAppts at all
    events_run_and_scheduled = get_events_run_and_scheduled(sim)
    assert 0 == len(events_run_and_scheduled)
    assert 'HSI_GenericFirstApptAtFacilityLevel0' not in events_run_and_scheduled
    assert 'HSI_GenericEmergencyFirstApptAtFacilityLevel1' not in events_run_and_scheduled


def test_healthcareseeking_occurs_with_spurious_symptoms_only(seed):
    """spurious symptoms should generate non-emergency HSI"""
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules including Chronic Syndrome and Mockitis -
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, store_hsi_events_that_have_run=True),
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

    # Check that 'HSI_GenericFirstApptAtFacilityLevel0' are triggerd (but not
    # 'HSI_GenericEmergencyFirstApptAtFacilityLevel1')
    events_run_and_scheduled = get_events_run_and_scheduled(sim)
    assert 'HSI_GenericFirstApptAtFacilityLevel0' in events_run_and_scheduled
    assert 'HSI_GenericEmergencyFirstApptAtFacilityLevel1' not in events_run_and_scheduled

    # And that the persons who have those HSI do have symptoms currently:
    person_ids = [i[4].target for i in sim.modules['HealthSystem'].HSI_EVENT_QUEUE]
    for person in person_ids:
        assert 0 < len(sim.modules['SymptomManager'].has_what(person))


def test_healthcareseeking_occurs_with_spurious_symptoms_and_disease_modules(seed):
    """Mockitis and Chronic Syndrome should lead to there being emergency and non-emergency generic HSI"""
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules including Chronic Syndrome and Mockitis -
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, store_hsi_events_that_have_run=True),
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

    # Check that Emergency and Non-Emergency GenericFirstAppts are triggerd
    events_run_and_scheduled = get_events_run_and_scheduled(sim)
    assert 'HSI_GenericFirstApptAtFacilityLevel0' in events_run_and_scheduled
    assert 'HSI_GenericEmergencyFirstApptAtFacilityLevel1' in events_run_and_scheduled


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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, store_hsi_events_that_have_run=True),
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


def test_force_healthcare_seeking(seed):
    """Check that the parameter/argument 'force_any_symptom_to_lead_to_healthcare_seeking' causes any symptom onset to
    lead immediately to healthcare seeking."""

    def hsi_scheduled_following_symptom_onset(force_any_symptom_to_lead_to_healthcare_seeking):
        """Returns True if a FirstAttendance HSI has been scheduled for a person following onset of symptoms with low
        probability of causing healthcare seeking."""

        class DummyDisease(Module):
            METADATA = {Metadata.USES_SYMPTOMMANAGER}
            """Dummy Disease - it's only job is to create a symptom and impose it everyone"""

            def read_parameters(self, data_folder):
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name='NonEmergencySymptom',
                            odds_ratio_health_seeking_in_adults=0.0001,
                            odds_ratio_health_seeking_in_children=0.0001),
                )

            def initialise_population(self, population):
                pass

            def initialise_simulation(self, sim):
                """Give person_id=0 both symptoms"""
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=[0],
                    disease_module=self,
                    symptom_string='NonEmergencySymptom',
                    add_or_remove='+'
                )

            def on_birth(self, mother, child):
                pass

        start_date = Date(2010, 1, 1)
        sim = Simulation(start_date=start_date, seed=seed)

        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                     symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                     healthseekingbehaviour.HealthSeekingBehaviour(
                         resourcefilepath=resourcefilepath,
                         force_any_symptom_to_lead_to_healthcareseeking=force_any_symptom_to_lead_to_healthcare_seeking
                     ),
                     DummyDisease()
                     )

        # Initialise the simulation (run the simulation for zero days)
        popsize = 200
        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=start_date)

        # Run the HealthSeeingBehaviourPoll
        sim.modules['HealthSeekingBehaviour'].theHealthSeekingBehaviourPoll.run()

        # See what HSI are scheduled to occur for the person
        evs = [x[1].TREATMENT_ID for x in
               sim.modules['HealthSystem'].sim.modules['HealthSystem'].find_events_for_person(0)]

        return 'FirstAttendance_NonEmergency' in evs

    assert not hsi_scheduled_following_symptom_onset(force_any_symptom_to_lead_to_healthcare_seeking=False)
    assert hsi_scheduled_following_symptom_onset(force_any_symptom_to_lead_to_healthcare_seeking=True)


def test_force_healthcare_seeking_control_of_behaviour_through_parameters_and_arguements(seed):
    """Check that behaviour of 'forced healthcare seeking' can be controlled via parameters and arguments to the
    module."""
    start_date = Date(2010, 1, 1)

    value_in_resourcefile = bool(pd.read_csv(
        resourcefilepath / 'ResourceFile_HealthSeekingBehaviour.csv'
    ).set_index('parameter_name').at['force_any_symptom_to_lead_to_healthcareseeking', 'value'])

    # No specification with argument --> behaviour is as per the parameter value in the ResourceFile
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=0))
    assert value_in_resourcefile == sim.modules['HealthSeekingBehaviour'].force_any_symptom_to_lead_to_healthcareseeking
    assert False is sim.modules['HealthSeekingBehaviour'].force_any_symptom_to_lead_to_healthcareseeking,\
        "Default behaviour in resourcefile should be 'False'"

    # Editing parameters --> behaviour is as per the edited parameters
    value_as_edited = not value_in_resourcefile
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    )
    # edit the value:
    sim.modules['HealthSeekingBehaviour'].parameters['force_any_symptom_to_lead_to_healthcareseeking'] = value_as_edited

    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=0))
    assert value_as_edited == sim.modules['HealthSeekingBehaviour'].force_any_symptom_to_lead_to_healthcareseeking

    # Editing parameters *and* with an argument provided to module --> argument over-writes parameter edits
    value_in_argument = not value_in_resourcefile
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(
            resourcefilepath=resourcefilepath,
            force_any_symptom_to_lead_to_healthcareseeking=value_in_argument),
    )
    # edit the value (to nonsense)
    sim.modules['HealthSeekingBehaviour'].parameters['force_any_symptom_to_lead_to_healthcareseeking'] = 'XXXXXX'

    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=0))
    assert value_in_argument == sim.modules['HealthSeekingBehaviour'].force_any_symptom_to_lead_to_healthcareseeking
