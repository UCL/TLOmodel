"""Test for HealthCareSeeking Module"""
import os
from pathlib import Path
from typing import Iterable, List

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


def get_events_run_and_scheduled_for_person(_sim, person_ids: Iterable) -> List:
    """Returns a list of HSI_Events that have been run already or are scheduled to run, for a particular set of
    persons"""
    return [
               ev['HSI_Event'] for ev in _sim.modules['HealthSystem'].store_of_hsi_events_that_have_run
               if ev['Person_ID'] in person_ids
           ] + [
               e[4].__class__.__name__ for e in _sim.modules['HealthSystem'].HSI_EVENT_QUEUE
               if e[4].target in person_ids
           ]


def test_healthcareseeking_does_occur_from_symptom_that_does_give_healthcareseeking_behaviour(seed):
    """test that a symptom that gives healthcare seeking results in generic HSI scheduled (and that those without the
    symptom do not seek care)."""

    popsize = 200
    idx_gets_symptom = set(range(100))
    idx_no_symptom = set(range(popsize)) - idx_gets_symptom

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
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=list(idx_gets_symptom),
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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, hsi_event_count_log_period="simulation",
                                           store_hsi_events_that_have_run=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 DummyDisease()
                 )

    # Run the simulation
    end_date = start_date + DateOffset(days=10)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that the symptom has been registered and is flagged as causing healthcare seeking
    assert 'Symptom_that_does_cause_healthcare_seeking' in \
           sim.modules['SymptomManager'].symptom_names
    assert 'Symptom_that_does_cause_healthcare_seeking' in \
           sim.modules['HealthSeekingBehaviour'].odds_ratio_health_seeking_in_children
    assert 'Symptom_that_does_cause_healthcare_seeking' in \
           sim.modules['HealthSeekingBehaviour'].odds_ratio_health_seeking_in_adults

    # Check that everyone has the symptom
    assert idx_gets_symptom == set(
        sim.modules['SymptomManager'].who_has('Symptom_that_does_cause_healthcare_seeking'))

    # Check that `HSI_GenericFirstApptAtFacilityLevel0` (but not `HSI_GenericEmergencyFirstApptAtFacilityLevel1`) are
    # triggered for some of persons with the symptom `idx_gets_symptom`
    events_run_and_scheduled = get_events_run_and_scheduled(sim)
    assert 'HSI_GenericFirstApptAtFacilityLevel0' in events_run_and_scheduled
    assert 'HSI_GenericEmergencyFirstApptAtFacilityLevel1' not in events_run_and_scheduled

    # Check that there is no HSI for those with no symptom
    assert 0 == len(get_events_run_and_scheduled_for_person(sim, idx_no_symptom))


def test_healthcareseeking_does_not_occurs_from_symptom_that_do_not_give_healthcareseeking_behaviour(seed):
    """test that a symptom that should not give healthseeeking does not give health seeking."""

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
    assert 'Symptom_that_does_not_cause_healthcare_seeking' not in \
           sim.modules['HealthSeekingBehaviour'].odds_ratio_health_seeking_in_children
    assert 'Symptom_that_does_not_cause_healthcare_seeking' not in \
           sim.modules['HealthSeekingBehaviour'].odds_ratio_health_seeking_in_adults

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
                Symptom.emergency('Symptom_that_does_cause_emergency_healthcare_seeking')
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

    # Check that the symptom has been registered and is flagged as causing emergency healthcare seeking
    assert 'Symptom_that_does_cause_emergency_healthcare_seeking' in sim.modules['SymptomManager'].symptom_names
    assert sim.modules['HealthSeekingBehaviour'].prob_seeks_emergency_appt_in_children[
               'Symptom_that_does_cause_emergency_healthcare_seeking'] > 0
    assert sim.modules['HealthSeekingBehaviour'].prob_seeks_emergency_appt_in_adults[
               'Symptom_that_does_cause_emergency_healthcare_seeking'] > 0

    # Check that everyone has the symptom
    df = sim.population.props
    assert set(df.loc[df.is_alive].index) == set(
        sim.modules['SymptomManager'].who_has('Symptom_that_does_cause_emergency_healthcare_seeking'))

    # Check that `HSI_GenericEmergencyFirstApptAtFacilityLevel1` are triggered for everyone (but not
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
                Symptom.emergency(name='EmergencySymptom'),
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
    evs = [x[1].TREATMENT_ID for x in sim.modules['HealthSystem'].find_events_for_person(0)]

    assert 'FirstAttendance_Emergency' in evs
    assert 'FirstAttendance_NonEmergency' not in evs


def test_force_healthcare_seeking(seed):
    """Check that the parameter/argument 'force_any_symptom_to_lead_to_healthcare_seeking' causes any symptom onset to
    lead immediately to healthcare seeking, except those symptoms that are declared to not lead to any healthcare
    seeking with the flag `no_healthcareseeking_in_children` or `no_healthcareseeking_in_adults`."""

    def hsi_scheduled_following_symptom_onset(force_any_symptom_to_lead_to_healthcare_seeking,
                                              no_healthcareseeking_in_children=False,
                                              no_healthcareseeking_in_adults=False):
        """Returns True if a FirstAttendance HSI has been scheduled for a person on the same day as the onset of
        symptoms that would ordinarily have a low probability of causing healthcare seeking."""

        class DummyDisease(Module):
            METADATA = {Metadata.USES_SYMPTOMMANAGER}
            """Dummy Disease - it's only job is to create a symptom and impose it everyone"""

            def read_parameters(self, data_folder):
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name='NonEmergencySymptom',
                            odds_ratio_health_seeking_in_adults=(0.0001  # <--- low chance of care seeking usually
                                                                 if not no_healthcareseeking_in_adults else None),
                            odds_ratio_health_seeking_in_children=(0.0001  # <--- low chance of care seeking usually
                                                                   if not no_healthcareseeking_in_children else None),
                            no_healthcareseeking_in_adults=no_healthcareseeking_in_adults,
                            no_healthcareseeking_in_children=no_healthcareseeking_in_children,
                            ),
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

        # See what HSI are scheduled to occur for the person on the same day
        evs = [x[1].TREATMENT_ID for x in
               sim.modules['HealthSystem'].find_events_for_person(0) if x[0].date() == start_date]

        return 'FirstAttendance_NonEmergency' in evs

    # A symptom that is unlikely to cause healthcare seeking usually, does cause healthcare seeking when "forced"
    assert not hsi_scheduled_following_symptom_onset(force_any_symptom_to_lead_to_healthcare_seeking=False)
    assert hsi_scheduled_following_symptom_onset(force_any_symptom_to_lead_to_healthcare_seeking=True)

    # A symptom that declares that it never cause healthcare seeking is not affected by the "force" argument.
    assert not hsi_scheduled_following_symptom_onset(force_any_symptom_to_lead_to_healthcare_seeking=False,
                                                     no_healthcareseeking_in_children=True,
                                                     no_healthcareseeking_in_adults=True,
                                                     )
    assert not hsi_scheduled_following_symptom_onset(force_any_symptom_to_lead_to_healthcare_seeking=True,
                                                     no_healthcareseeking_in_children=True,
                                                     no_healthcareseeking_in_adults=True,
                                                     )


def test_force_healthcare_seeking_control_of_behaviour_through_parameters_and_arguments(seed):
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
    assert False is sim.modules['HealthSeekingBehaviour'].force_any_symptom_to_lead_to_healthcareseeking, \
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
                Symptom.emergency('Symptom_that_does_cause_emergency_healthcare_seeking')
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


def test_same_day_healthcare_seeking_when_using_force_healthcareseeking(seed, tmpdir):
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
                Symptom(name='Symptom_that_does_not_cause_emergency_healthcare_seeking',
                        odds_ratio_health_seeking_in_children=0.0001,  # <-- low chance of healthcare seeking
                        odds_ratio_health_seeking_in_adults=0.0001,  # <-- low chance of healthcare seeking
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
                        odds_ratio_health_seeking_in_adults=1000000.0,  # <--- very high odds of seeking care
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
               sim.modules['HealthSystem'].find_events_for_person(_person_id)]
        assert 'FirstAttendance_NonEmergency' in evs, f"No FirstAttendance_NonEmergency for {_person_id=}"


def test_care_seeking_from_symptoms_with_intermediate_level_of_care_seeking_and_emergency(seed):
    """Check that a symptom with an intermediate level of healthcare-seeking and intermediate probability of emergency
    care leads to some persons having no care, some having emergency care and some having non-emergency care."""

    class DummyDisease(Module):
        METADATA = {Metadata.USES_SYMPTOMMANAGER}
        """Dummy Disease - it's only job is to create a symptom and impose it on everyone"""

        def read_parameters(self, data_folder):
            self.sim.modules['SymptomManager'].register_symptom(
                Symptom(name='TestSymptom',
                        odds_ratio_health_seeking_in_adults=1.0,  # <--- intermediate odds of seeking care
                        odds_ratio_health_seeking_in_children=1.0,  # <--- intermediate odds of seeking care
                        prob_seeks_emergency_appt_in_adults=0.5,  # <--- intermediate prob of emergency care
                        prob_seeks_emergency_appt_in_children=0.5,  # <--- intermediate prob of emergency care
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
                symptom_string='TestSymptom',
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
    assert set(df.loc[df.is_alive].index) == set(sim.modules['SymptomManager'].who_has('TestSymptom'))

    # Check what HSI are created when the poll is run
    # - clear HealthSystem queue and run the HealthSeekingPoll
    sim.modules['HealthSystem'].reset_queue()
    sim.modules['HealthSeekingBehaviour'].theHealthSeekingBehaviourPoll.run()

    # - find what events have been scheduled for each person
    evs = pd.Series({
        _person_id: [
            x[1].TREATMENT_ID
            for x in sim.modules['HealthSystem'].find_events_for_person(_person_id)
        ]
        for _person_id in df.loc[df.is_alive].index
    }).explode()

    # - check that some people have no HSI
    assert evs.isnull().any()  # null values would be created by no appts being found for this person

    # - check that some have Non-Emergency
    assert 'FirstAttendance_NonEmergency' in evs.values

    # - check that some have Emergency HSI
    assert 'FirstAttendance_Emergency' in evs.values

    # - check that no person has more than one appointment scheduled for them
    assert not evs.index.has_duplicates  # index would have been duplicated by `pd.Series.explode` if a person had more
    #                                      than one appt.


def test_care_seeking_from_symptoms_with_different_levels_of_prob_emergency(seed):
    """Check that a symptom with a high degree of healthcare-seeking and probability of emergency care of p%, leads to
     all having some care, and p% having emergency care, and (1-p)% having non-emergency care."""

    def get_evs_generated_by_hcs_poll(prob_seeks_emergency_appt: float) -> pd.Series:
        """Returns a pd.Series describing the events scheduled for each person after running the HealthcareSeeking poll,
        in a simulation when a Symptom is defined with a high degree of healthcare-seeking and the specified probability
        of seeking emergency care."""
        class DummyDisease(Module):
            METADATA = {Metadata.USES_SYMPTOMMANAGER}
            """Dummy Disease - it's only job is to create a symptom and impose it on everyone"""

            def read_parameters(self, data_folder):
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name='TestSymptom',
                            odds_ratio_health_seeking_in_adults=10_000.0,  # <--- high odds of seeking care
                            odds_ratio_health_seeking_in_children=10_000.0,  # <--- high odds of seeking care
                            prob_seeks_emergency_appt_in_adults=prob_seeks_emergency_appt,
                            prob_seeks_emergency_appt_in_children=prob_seeks_emergency_appt,
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
                    symptom_string='TestSymptom',
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
        assert set(df.loc[df.is_alive].index) == set(sim.modules['SymptomManager'].who_has('TestSymptom'))

        # Check what HSI are created when the poll is run
        # - clear HealthSystem queue and run the HealthSeekingPoll
        sim.modules['HealthSystem'].reset_queue()
        sim.modules['HealthSeekingBehaviour'].theHealthSeekingBehaviourPoll.run()

        # - find what events have been scheduled for each person
        df = sim.population.props
        evs = pd.Series({
            _person_id: [
                x[1].TREATMENT_ID
                for x in sim.modules['HealthSystem'].find_events_for_person(_person_id)
            ]
            for _person_id in df.loc[df.is_alive].index
        }).explode()

        return evs

    for prob_seeks_emergency_appt in [0.0, 0.25, 0.5, 0.75, 1.0]:

        evs = get_evs_generated_by_hcs_poll(prob_seeks_emergency_appt=prob_seeks_emergency_appt)

        # - check that all people have exactly one HSI
        assert not evs.isnull().any()
        assert not evs.index.has_duplicates

        # - compare proportion of Emergency and Non-Emergency appointments to the expectation, with some degree of
        # tolerance for stochastic effects
        assert np.isclose((evs == 'FirstAttendance_Emergency').sum() / len(evs), prob_seeks_emergency_appt,
                          atol=0.02)
        assert np.isclose((evs == 'FirstAttendance_NonEmergency').sum() / len(evs), (1.0 - prob_seeks_emergency_appt),
                          atol=0.02)


def test_persons_have_maximum_of_one_hsi_scheduled(seed):
    """Check that when persons have a mixture of symptoms, some emergency and some not, there is only one
    FirstAttendance appointment for them."""

    class DummyDisease(Module):
        METADATA = {Metadata.USES_SYMPTOMMANAGER}
        """Dummy Disease - it's only job is to create a symptom and impose it on everyone"""

        def read_parameters(self, data_folder):
            self.sim.modules['SymptomManager'].register_symptom(
                Symptom(name='EmergencySymptom',
                        odds_ratio_health_seeking_in_adults=1.0,  # <--- intermediate degree of healthcare seeking
                        odds_ratio_health_seeking_in_children=1.0,  # <--- intermediate degree of healthcare seeking
                        prob_seeks_emergency_appt_in_adults=0.5,  # <--- possibility of seeking emergency care
                        prob_seeks_emergency_appt_in_children=0.5,  # <--- possibility of seeking emergency care
                        ),
                Symptom(name='NonEmergencySymptom',
                        odds_ratio_health_seeking_in_adults=1.0,  # <--- intermediate degree of healthcare seeking
                        odds_ratio_health_seeking_in_children=1.0,  # <--- intermediate degree of healthcare seeking
                        prob_seeks_emergency_appt_in_adults=0.0,  # <--- will not seek emergency care
                        prob_seeks_emergency_appt_in_children=0.0,  # <--- will not seek emergency care
                        ),
            )

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            """Give all persons both symptoms"""
            df = self.sim.population.props
            idx_all_alive_persons = df.loc[df.is_alive].index.to_list()
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=idx_all_alive_persons,
                disease_module=self,
                symptom_string=['EmergencySymptom', 'NonEmergencySymptom'],
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

    # Check that everyone has the symptoms
    df = sim.population.props
    assert set(df.loc[df.is_alive].index) == set(sim.modules['SymptomManager'].who_has('EmergencySymptom'))
    assert set(df.loc[df.is_alive].index) == set(sim.modules['SymptomManager'].who_has('NonEmergencySymptom'))

    # Check what HSI are created when the poll is run
    # - clear HealthSystem queue and run the HealthSeekingPoll
    sim.modules['HealthSystem'].reset_queue()
    sim.modules['HealthSeekingBehaviour'].theHealthSeekingBehaviourPoll.run()

    # - find what events have been scheduled for each person
    evs = pd.Series({
        _person_id: [
            x[1].TREATMENT_ID
            for x in sim.modules['HealthSystem'].find_events_for_person(_person_id)
        ]
        for _person_id in df.loc[df.is_alive].index
    }).explode()

    # - check that all persons have exactly one HSI scheduled
    assert not evs.index.has_duplicates  # index would have been duplicated by `pd.Series.explode` if a person had more
    #                                      than one appt.
