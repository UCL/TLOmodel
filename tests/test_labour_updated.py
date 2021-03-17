import os
from pathlib import Path

import pandas as pd

from tlo.lm import LinearModel, LinearModelType, Predictor

import pytest
from tlo import Date, Simulation, logging
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    symptommanager, postnatal_supervisor
)

seed = 567


# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def run_sim_for_0_days_get_mother_id(sim):
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]
    return mother_id


def find_and_return_hsi_events_list(sim, individual_id):
    """Returns HSI event list for an individual"""
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=individual_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    return hsi_events


def check_event_queue_for_event_and_return_scheduled_event_date(sim, queue_of_interest, individual_id,
                                                                event_of_interest):
    """Checks the hsi OR event queue for an event and returns scheduled date"""
    if queue_of_interest == 'event':
        date_event, event = [
            ev for ev in sim.find_events_for_person(person_id=individual_id) if
            isinstance(ev[1], event_of_interest)
        ][0]
        return date_event
    else:
        date_event, event = [
            ev for ev in sim.modules['HealthSystem'].find_events_for_person(individual_id) if
            isinstance(ev[1], event_of_interest)
        ][1]
        return date_event


def register_modules(ignore_cons_constraints):
    """Register all modules that are required for labour to run"""

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           ignore_cons_constraints=ignore_cons_constraints),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    return sim


def test_run_no_constraints():
    """This test runs a simulation with a functioning health system with full service availability and no set
    constraints"""

    sim = register_modules(ignore_cons_constraints=False)

    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 1, 1))

    check_dtypes(sim)


def test_event_scheduling_for_labour_onset_and_home_birth_no_care_seeking():
    """Test that the right events are scheduled during the labour module (and in the right order) for women who delivery
     at home. Spacing between events (in terms of days since labour onset) is enforced via assert functions within the
    labour module"""
    sim = register_modules(ignore_cons_constraints=False)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # Set pregnancy characteristics that will allow the labour events to run
    df = sim.population.props
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'la_due_date_current_pregnancy'] = sim.date
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(weeks=38)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 40
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    mni[mother_id]['test_run'] = True

    # force this woman to decide to deliver at home
    params = sim.modules['Labour'].parameters
    params['test_care_seeking_probs'] = [1, 0, 0]

    # define and run labour onset event
    labour_onset = labour.LabourOnsetEvent(individual_id=mother_id, module=sim.modules['Labour'])
    labour_onset.apply(mother_id)
    assert (mni[mother_id]['labour_state'] == 'term_labour')

    # Check that the correct events are scheduled for this woman whose labour has started
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]

    assert labour.BirthEvent in events
    assert labour.LabourDeathAndStillBirthEvent in events
    # todo: struggling to force care seeking in this event (way the regression is coded)
    # assert labour.LabourAtHomeEvent in events

    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert labour.HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour not in hsi_events

    # run birth event as this event manages scheduling of postpartum events (home birth and death events have their own
    # tests)
    mni[mother_id]['delivery_setting'] = 'home_birth'
    sim.date = sim.date + pd.DateOffset(days=5)
    sim.event_queue.queue.clear()

    birth_event = labour.BirthEvent(mother_id=mother_id, module=sim.modules['Labour'])
    birth_event.apply(mother_id)

    # Ensure that the postpartum home birth event is scheduled correctly
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert labour.PostpartumLabourAtHomeEvent in events

    # Set care seeking odds to 0 (as changes event sequence if women seek care- tested later)
    params['odds_careseeking_for_complication'] = 0

    # Define and run postpartum event
    pn_event = labour.PostpartumLabourAtHomeEvent(individual_id=mother_id, module=sim.modules['Labour'])
    pn_event.apply(mother_id)

    # And finally check the first event of the postnatal module is correctly scheduled
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert postnatal_supervisor.PostnatalWeekOneEvent in events


def test_event_scheduling_for_labour_onset_and_facility_delivery():
    pass

def test_event_scheduling_for_admissions_from_antenatal_inpatient_ward():
    pass


def test_event_scheduling_for_care_seeking_during_home_birth():
    pass


def test_run_health_system_high_squeeze():
    """This test runs a simulation in which the contents of scheduled HSIs will not be performed because the squeeze
    factor is too high. Therefore it tests the logic in the did_not_run functions of the Labour HSIs to ensure women
    who want to deliver in a facility, but cant, due to lacking capacity, have the correct events scheduled to continue
    their labour"""
    pass


@pytest.mark.group2
def test_run_health_system_events_wont_run():
    """This test runs a simulation in which no scheduled HSIs will run.. Therefore it tests the logic in the
    not_available functions of the Labour HSIs to ensure women who want to deliver in a facility, but cant, due to the
    service being unavailble, have the correct events scheduled to continue their labour"""
    pass

def test_custom_linear_models():
    pass
    """sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=0))

    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]

    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'la_due_date_current_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 37
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(months=9)

    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)

    labour_onset = labour.LabourOnsetEvent(module=sim.modules['Labour'], individual_id=mother_id)
    labour_onset.apply(mother_id)

    params = sim.modules['Labour'].parameters
    params['la_labour_equations']['predict_chorioamnionitis_ip'].predict(
        df.loc[[mother_id]])[mother_id] """


# todo: test event scheduling in all different methiods
