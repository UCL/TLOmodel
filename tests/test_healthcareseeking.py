"""Test for HealthCareSeeking Module"""
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pandas import DateOffset

from tlo import Date, Module, Simulation
from tlo.events import Event, IndividualScopeEventMixin
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


def get_events_run_and_scheduled(sim) -> List:
    """Returns a list of HSI_Events that have been run already or are scheduled to run."""
    return [
        event_details.event_name
        for event_details in sim.modules['HealthSystem'].hsi_event_counts.keys()
    ] + [
        type(event_queue_item.hsi_event).__name__
        for event_queue_item in sim.modules['HealthSystem'].HSI_EVENT_QUEUE
    ]


def test_healthcareseeking_does_occur_from_symptom_that_does_give_healthcareseeking_behaviour(seed):
    """test that a symptom that gives healthcare seeking results in generic HSI scheduled."""

    class DummyDisease(Module):
        METADATA = {Metadata.USES_SYMPTOMMANAGER}
        """Dummy Disease - it's only job is to create a symptom and impose it everyone"""

        def read_parameters(self, data_folder):
            self.sim.modules['SymptomManager'].register_symptom(
                Symptom(
                    name='Symptom_that_does_cause_healthcare_seeking',
                    odds_ratio_health_seeking_in_adults=1000000.0,  # <--- very high odds of seeking care
                    odds_ratio_health_seeking_in_children=1000000.0  # <--- very high odds of seeking care
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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, hsi_event_count_log_period="simulation"),
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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, hsi_event_count_log_period="simulation"),
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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, hsi_event_count_log_period="simulation"),
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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, hsi_event_count_log_period="simulation"),
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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, hsi_event_count_log_period="simulation"),
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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, hsi_event_count_log_period="simulation"),
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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, hsi_event_count_log_period="simulation"),
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


def test_same_day_healthcare_seeking_for_emergency_symptoms(seed, tmpdir):
    """Check that emergency symptoms cause the FirstGenericEmergency HSI_Event to run on the same day. N.B. The
    fullmodel is used here because without it, the ordering of events can be correct by chance."""

    date_symptom_is_imposed = Date(2010, 1, 3)

    class EventToImposeSymptom(Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)

        def apply(self, person_id):
            """Give person 0 the symptom."""
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=0,
                disease_module=self.module,
                symptom_string='Symptom_that_does_cause_emergency_healthcare_seeking',
                add_or_remove='+',
            )

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
            pass

        def initialise_simulation(self, sim):
            """Schedule for the symptom to be imposed on `date_symptom_is_imposed`"""
            sim.schedule_event(EventToImposeSymptom(self, person_id=0), date_symptom_is_imposed)

        def on_birth(self, mother, child):
            pass

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            hsi_event_count_log_period="simulation",
        ),
        symptommanager.SymptomManager(
            resourcefilepath=resourcefilepath,
            spurious_symptoms=False,
        ),
        healthseekingbehaviour.HealthSeekingBehaviour(
            resourcefilepath=resourcefilepath,
        ),
        DummyDisease(),
    )

    # Run the simulation for ten days
    end_date = start_date + DateOffset(days=10)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that the 'HSI_GenericEmergencyFirstApptAtFacilityLevel1' was the only event to occur
    assert len(sim.modules['HealthSystem'].hsi_event_counts) == 1
    only_event_that_ran, count = sim.modules['HealthSystem'].hsi_event_counts.popitem()
    assert count == 1
    assert (
        only_event_that_ran.event_name
        == 'HSI_GenericEmergencyFirstApptAtFacilityLevel1'
    )


def test_same_day_healthcare_seeking_when_using_force_healthcare_seeking(seed, tmpdir):
    """Check that when using `force_healthcare_seeking` non-emergency symptoms cause the FirstGenericNonEmergency
    HSI_Event to run on the same day."""

    date_symptom_is_imposed = Date(2010, 1, 3)

    class EventToImposeSymptom(Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)

        def apply(self, person_id):
            """Give person 0 the symptom on the date that the symptom is imposed."""
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=0,
                disease_module=self.module,
                symptom_string='Symptom_that_does_not_cause_emergency_healthcare_seeking',
                add_or_remove='+',
            )

    class DummyDisease(Module):
        METADATA = {Metadata.USES_SYMPTOMMANAGER}
        """Dummy Disease - it's only job is to create a symptom and impose it everyone"""

        def read_parameters(self, data_folder):
            self.sim.modules['SymptomManager'].register_symptom(
                Symptom(
                    name='Symptom_that_does_not_cause_emergency_healthcare_seeking',
                    emergency_in_adults=False,
                    emergency_in_children=False
                ),
            )

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            """Schedule for the symptom to be imposed on `date_symptom_is_imposed`"""
            sim.schedule_event(EventToImposeSymptom(self, person_id=0), date_symptom_is_imposed)

        def on_birth(self, mother, child):
            pass

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            hsi_event_count_log_period="simulation",
        ),
        symptommanager.SymptomManager(
            resourcefilepath=resourcefilepath,
            spurious_symptoms=False,
        ),
        healthseekingbehaviour.HealthSeekingBehaviour(
            resourcefilepath=resourcefilepath,
            force_any_symptom_to_lead_to_healthcareseeking=True,
        ),
        DummyDisease()
    )

    # Run the simulation for ten days
    end_date = start_date + DateOffset(days=10)
    popsize = 200
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that the 'HSI_GenericFirstApptAtFacilityLevel0' was the only event to occur
    assert len(sim.modules['HealthSystem'].hsi_event_counts) == 1
    only_event_that_ran, count = sim.modules['HealthSystem'].hsi_event_counts.popitem()
    assert count == 1
    assert only_event_that_ran.event_name == 'HSI_GenericFirstApptAtFacilityLevel0'


def test_everyone_seeks_care_for_symptom_with_high_odds_ratio_of_seeking_care(seed):
    """Check that a non-emergency symptom with a VERY high odds of healthcare seeking will cause everyone who has that
    symptom to seek care (a non-emergency first appointment)."""

    class DummyDisease(Module):
        METADATA = {Metadata.USES_SYMPTOMMANAGER}
        """Dummy Disease - it's only job is to create a symptom and impose it on everyone"""

        def read_parameters(self, data_folder):
            self.sim.modules['SymptomManager'].register_symptom(
                Symptom(name='NonEmergencySymptom',
                        odds_ratio_health_seeking_in_adults=1000000.0,   # <--- very high odds of seeking care
                        odds_ratio_health_seeking_in_children=1000000.0  # <--- very high odds of seeking care
                        ),
            )

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            """Give all persons the symptom"""
            df = self.sim.population.props
            idx_all_alive_persons = df.loc[df.is_alive].index.to_list()
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=idx_all_alive_persons,
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
                     force_any_symptom_to_lead_to_healthcareseeking=False,
                 ),
                 DummyDisease()
                 )

    # Initialise the simulation (run the simulation for zero days)
    popsize = 1000
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)

    # Check that everyone has the symptom
    df = sim.population.props
    assert set(df.loc[df.is_alive].index) == set(sim.modules['SymptomManager'].who_has('NonEmergencySymptom'))

    # Check that the linear model of health-care seeking show that the prob of seeking care is ~1.0
    hsb = sim.modules['HealthSeekingBehaviour']
    assert np.allclose(
        hsb.hsb_linear_models['children'].predict(df.loc[df.is_alive & (df.age_years < 15)]),
        1.0,
        atol=0.001
    )
    assert np.allclose(
        hsb.hsb_linear_models['adults'].predict(df.loc[df.is_alive & (df.age_years >= 15)]),
        1.0,
        atol=0.001
    )

    # Check that all persons are scheduled for the generic HSI when the simulation runs:
    # - clear HealthSystem queue and run the HealthSeekingPoll
    sim.modules['HealthSystem'].reset_queue()
    sim.modules['HealthSeekingBehaviour'].theHealthSeekingBehaviourPoll.run()

    # - check that every person for whom the symptom was onset has been scheduled an HSI
    for _person_id in df.loc[df.is_alive].index:
        evs = [x[1].TREATMENT_ID for x in
               sim.modules['HealthSystem'].sim.modules['HealthSystem'].find_events_for_person(_person_id)]
        assert 'FirstAttendance_NonEmergency' in evs, f"No FirstAttendance_NonEmergency for {_person_id=}"
