"""Test for for the HIV Module."""

import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.lm import LinearModel
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    pregnancy_supervisor,
    symptommanager, dx_algorithm_child, hsi_generic_first_appts,
)
from tlo.methods.healthseekingbehaviour import HealthSeekingBehaviourPoll
from tlo.methods.hiv import Hiv, HSI_Hiv_TestAndRefer, HivAidsOnsetEvent

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

def get_sim():
    """get sim with the checks for configuration of properties running in the HIV module"""
    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     ignore_cons_constraints=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=True)
                 )

    # Edit the efficacy of PrEP to be perfect (for the purpose of these tests)
    sim.modules['Hiv'].parameters['proportion_reduction_in_risk_of_hiv_aq_if_on_prep'] = 1.0
    # Let there be a 100% probability of TestAndRefer events being scheduled
    sim.modules['Hiv'].parameters['prob_spontaneous_test_12m'] = 1.0

    # Make the population
    sim.make_initial_population(n=popsize)
    return sim

def test_basic_run_with_default_parameters():
    """Run the HIV module with check and check dtypes consistency"""
    end_date = Date(2015, 12, 31)

    sim = get_sim()
    check_dtypes(sim)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    # confirm configuration of properties at the end of the simulation:
    sim.modules['Hiv'].check_config_of_properties()

def test_initialisation():
    """check that the natural history plays out as expected for those that are infected at the beginning of the sim"""

    # get simulation and initialise the simulation:
    sim = get_sim()
    sim.modules['Hiv'].initialise_simulation(sim)
    df = sim.population.props

    # check that everyone who is infected but not AIDS or ART, gets a future AIDS event (but no future AIDS death)
    inf = df.loc[df.is_alive & df.hv_inf].index.tolist()
    art = df.loc[df.is_alive & (df.hv_art != "not")].index.tolist()
    aids = sim.modules['SymptomManager'].who_has('aids_symptoms')
    before_aids_idx = set(inf) - set(art) - set(aids)

    for idx in before_aids_idx:
        events_for_this_person = sim.find_events_for_person(idx)
        assert 1 == len(events_for_this_person)
        next_event_date, next_event_obj = events_for_this_person[0]
        assert isinstance(next_event_obj, hiv.HivAidsOnsetEvent)
        assert next_event_date >= sim.date

    # check that everyone who is infected and has got AIDS event get a future AIDS death event but nothing else
    for idx in aids:
        events_for_this_person = sim.find_events_for_person(idx)
        assert 1 == len(events_for_this_person)
        next_event_date, next_event_obj = events_for_this_person[0]
        assert isinstance(next_event_obj, hiv.HivAidsDeathEvent)
        assert next_event_date >= sim.date

def test_generation_of_new_infection():
    """Check that the generation of new infections is as expected.
    This occurs in the Main Polling Event.
    """

    sim = get_sim()
    pollevent = hiv.HivRegularPollingEvent(module=sim.modules['Hiv'])
    df = sim.population.props

    def any_hiv_infection_event_in_queue():
        for date, counter, event in sim.event_queue.queue:
            if isinstance(event, hiv.HivInfectionEvent):
                return True

    # If no people living with HIV, no new infections
    df.hv_inf = False
    pollevent.apply(sim.population)
    assert not any_hiv_infection_event_in_queue()

    # If everyone living with HIV, no new infections
    df.hv_inf = True
    pollevent.apply(sim.population)
    assert not any_hiv_infection_event_in_queue()

    # If lots of people living with HIV but all VL suppressed, no new infections
    df.hv_inf = sim.rng.rand(len(df.hv_inf)) < 0.5
    df.hv_art.values[:] = 'on_VL_suppressed'
    pollevent.apply(sim.population)
    assert not any_hiv_infection_event_in_queue()

    # If lots of people living with HIV, but those uninfected are all on PrEP (efficacy of PrEP is assumed to be
    # perfect), ... no new infections
    df.hv_art.values[:] = 'not'
    df.hv_is_on_prep = True
    pollevent.apply(sim.population)
    assert not any_hiv_infection_event_in_queue()

    # If lots of people living with HIV, and people are not on PrEP, some infection.
    df.hv_is_on_prep = False
    pollevent.apply(sim.population)
    assert any_hiv_infection_event_in_queue()

def test_generation_of_natural_history_process_no_art():
    """Check that:
    * New infections leads to a scheduled AIDS event
    * AIDS events lead to a scheduled AIDS death when no ART
    * The AIDS death event results in an actual death when no ART
    """

    sim = get_sim()
    df = sim.population.props

    # select an adult who is alive and not currently infected
    person_id = df.loc[df.is_alive & ~df.hv_inf & df.age_years.between(15, 80)].index[0]

    # make an run infection event for an adult who is not currently infected
    infection_event = hiv.HivInfectionEvent(module=sim.modules['Hiv'], person_id=person_id)
    infection_event.apply(person_id)

    assert True is bool(df.at[person_id, 'hv_inf'])
    assert "not" == df.at[person_id, 'hv_art']
    assert sim.date == df.at[person_id, 'hv_date_inf']

    # find the AIDS onset event for this person
    date_aids_event, aids_event = [ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.HivAidsOnsetEvent)][0]
    assert date_aids_event > sim.date

    # run the AIDS onset event for this person:
    aids_event.apply(person_id)
    assert "aids_symptoms" in sim.modules['SymptomManager'].has_what(person_id)

    # find the AIDS death event for this person
    date_aids_death_event, aids_death_event = [ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.HivAidsDeathEvent)][0]
    assert date_aids_death_event > sim.date

    # run the AIDS death event for this person:
    aids_death_event.apply(person_id)

    # confirm the person is dead
    assert False is bool(df.at[person_id, "is_alive"])
    assert sim.date == df.at[person_id, "date_of_death"]
    assert "AIDS" == df.at[person_id, "cause_of_death"]

def test_generation_of_natural_history_process_with_art_before_aids():
    """Check that:
    * New infections leads to a scheduled AIDS event
    * If on ART before AIDS onset, the AIDS events does not do anything and does not lead to a scheduled AIDS death
    """

    sim = get_sim()
    df = sim.population.props

    # select an adult who is alive and not currently infected
    person_id = df.loc[df.is_alive & ~df.hv_inf & df.age_years.between(15, 80)].index[0]

    # make an run infection event for an adult who is not currently infected
    infection_event = hiv.HivInfectionEvent(module=sim.modules['Hiv'], person_id=person_id)
    infection_event.apply(person_id)

    assert True is bool(df.at[person_id, 'hv_inf'])
    assert sim.date == df.at[person_id, 'hv_date_inf']

    # find the AIDS onset event for this person
    date_aids_event, aids_event = \
    [ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.HivAidsOnsetEvent)][0]
    assert date_aids_event > sim.date

    # Put person on ART with VL suppression prior to AIDS onset
    df.at[person_id, 'hv_art'] = "on_VL_suppressed"

    # run the AIDS onset event for this person:
    aids_event.apply(person_id)

    # check no AIDS death event for this person
    assert [] == [ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.HivAidsDeathEvent)]

    # check no AIDS symptoms for this person
    assert "aids_symptoms" not in sim.modules['SymptomManager'].has_what(person_id)

def test_generation_of_natural_history_process_with_art_after_aids():
    """Check that:
    * New infections leads to a scheduled AIDS event
    * AIDS event leads to AIDS death scheduled
    * If on ART before AIDS death, the AIDS Death does not do anything and does not lead to an actual death
    """

    sim = get_sim()
    df = sim.population.props

    # select an adult who is alive and not currently infected
    person_id = df.loc[df.is_alive & ~df.hv_inf & df.age_years.between(15, 80)].index[0]

    # make an run infection event for an adult who is not currently infected
    infection_event = hiv.HivInfectionEvent(module=sim.modules['Hiv'], person_id=person_id)
    infection_event.apply(person_id)

    assert True is bool(df.at[person_id, 'hv_inf'])
    assert "not" == df.at[person_id, 'hv_art']
    assert sim.date == df.at[person_id, 'hv_date_inf']

    # find the AIDS onset event for this person
    date_aids_event, aids_event = \
    [ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.HivAidsOnsetEvent)][0]
    assert date_aids_event > sim.date

    # run the AIDS onset event for this person:
    aids_event.apply(person_id)

    # find the AIDS death  event for this person
    date_aids_death_event, aids_death_event = \
    [ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.HivAidsDeathEvent)][0]
    assert date_aids_death_event > sim.date
    assert "aids_symptoms" in sim.modules['SymptomManager'].has_what(person_id)

    # Put the person on ART with VL suppression prior to the AIDS death (but following AIDS onset)
    df.at[person_id, 'hv_art'] = "on_VL_suppressed"

    # run the AIDS death event for this person:
    aids_death_event.apply(person_id)

    # confirm the person has not dead
    assert True is bool(df.at[person_id, "is_alive"])
    assert pd.isnull(df.at[person_id, "date_of_death"])
    assert "" == df.at[person_id, "cause_of_death"]

def test_mtct_at_birth():
    """Check that:
    * HIV infection events are created when the mother during breastfeeding
    """

    sim = get_sim()

    # Manipulate MTCT rates so that transmission always occurs at/before birth
    sim.modules['Hiv'].parameters["prob_mtct_treated"] = 1.0
    sim.modules['Hiv'].parameters["prob_mtct_untreated"] = 1.0
    sim.modules['Hiv'].parameters["prob_mtct_incident_preg"] = 1.0

    # Do a birth from a mother that is HIV-positive:
    df = sim.population.props
    mother_id = df.loc[df.is_alive & (df.sex == "F")].index[0]
    df.at[mother_id, 'hv_inf'] = True
    df.at[mother_id, 'hv_date_inf'] = sim.date
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date

    child_id = sim.population.do_birth()
    sim.modules['Hiv'].on_birth(mother_id, child_id)

    # Check that child is now HIV-positive
    assert sim.population.props.at[child_id, "hv_inf"]

def test_mtct_during_breastfeeding():
    """Check that:
    * HIV infection events are created when the mother during breastfeeding
    """

    sim = get_sim()

    # Manipulate MTCT rates so that transmission always occurs following birth
    sim.modules['Hiv'].parameters["prob_mtct_treated"] = 0.0
    sim.modules['Hiv'].parameters["prob_mtct_untreated"] = 0.0
    sim.modules['Hiv'].parameters["prob_mtct_incident_preg"] = 0.0
    sim.modules['Hiv'].parameters["monthly_prob_mtct_bf_treated"] = 1.0
    sim.modules['Hiv'].parameters["monthly_prob_mtct_bf_untreated"] = 1.0

    # Do a birth from a mother that is HIV-positive:
    df = sim.population.props
    mother_id = df.loc[df.is_alive & (df.sex == "F")].index[0]
    df.at[mother_id, 'hv_inf'] = True
    df.at[mother_id, 'hv_date_inf'] = sim.date
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date

    child_id = sim.population.do_birth()
    sim.modules['Demography'].on_birth(mother_id, child_id)
    sim.modules['Hiv'].on_birth(mother_id, child_id)

    # Check child is not yet HIV-positive
    assert not sim.population.props.at[child_id, "hv_inf"]

    # Check that there is an infection event:
    date_inf_event, inf_event = [
        ev for ev in sim.find_events_for_person(child_id) if isinstance(ev[1], hiv.HivInfectionDuringBreastFeedingEvent)
    ][0]

    # Run the infection event
    inf_event.apply(child_id)

    # Check child is now HIV-positive
    assert sim.population.props.at[child_id, "hv_inf"]

def test_test_and_refer_event_scheduled_by_main_event_poll():
    """Check that the main event poll causes there to be event of the HSI_TestAndRefer"""

    sim = get_sim()

    # Simulate for 0 days so as to complete all the initialisation steps
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props

    # Control the number of people for whom there should be a TestAndReferEvent (parameter for prob of testing is 100%)
    num_not_diagnosed = sum(~df.hv_diagnosed & df.is_alive)

    # Run a polling event
    pollevent = hiv.HivRegularPollingEvent(module=sim.modules['Hiv'])
    pollevent.apply(sim.population)

    # Check number and dates of TestAndRefer events in the HSI Event Queue
    dates_of_tr_events = [
        ev[1] for ev in sim.modules['HealthSystem'].HSI_EVENT_QUEUE if isinstance(ev[4], hiv.HSI_Hiv_TestAndRefer)
    ]
    assert num_not_diagnosed == len(dates_of_tr_events)
    assert all([(sim.date <= d <= (sim.date + pd.DateOffset(months=12))) for d in dates_of_tr_events])

def test_aids_symptoms_lead_to_treatment_being_initiated():
    """Check that if aids-symptoms onset then treatment can be initiated (even without spontaneous testing)"""

    # Set up simulation object in custom way:
    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     ignore_cons_constraints=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                               # force symptoms to lead to health care seeking:
                                                               force_any_symptom_to_lead_to_healthcareseeking=True),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=True)
                 )

    # Let there be a 0% probability of TestAndRefer events being scheduled
    sim.modules['Hiv'].parameters['prob_spontaneous_test_12m'] = 0.0

    # Make the population and simulate for 0 days to get everything initialised:
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props

    # Make no-one have HIV and clear the event queues:
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE = []
    sim.event_queue.queue = []
    df.hv_inf = False
    df.hv_art = "not"
    df.hv_is_on_prep = False
    df.hv_behaviour_change = False
    df.hv_diagnosed = False
    df.hv_number_tests = 0

    # Let one person have HIV and let AIDS be onset for that one person
    person_id = 0
    df.at[person_id, 'hv_inf'] = True
    aids_event = HivAidsOnsetEvent(person_id=person_id, module=sim.modules['Hiv'])
    aids_event.apply(person_id)

    # Confirm that they have aids symptoms and an AIDS death schedule
    assert 'aids_symptoms' in sim.modules['SymptomManager'].has_what(person_id)
    assert 1 == len([ev[0] for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.HivAidsDeathEvent)])

    # Run the health-seeking poll and run the GenericFirstAppt That is Created
    hsp = HealthSeekingBehaviourPoll(module=sim.modules['HealthSeekingBehaviour'])
    hsp.apply(sim.population)
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if isinstance(ev[1], hsi_generic_first_appts.HSI_GenericFirstApptAtFacilityLevel1)][0]
    ge.apply(ge.target, squeeze_factor=0.0)

    # Check that the person has a TestAndReferEvent scheduled
    assert 1 == len([ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if isinstance(ev[1], hiv.HSI_Hiv_TestAndRefer)])

def test_hsi_testandrefer_and_circ():
    """Test that the HSI for testing and referral to circumcision works as intended"""
    sim = get_sim()

    # Make the chance of being referred 100%
    sim.modules['Hiv'].lm_circ = LinearModel.multiplicative()

    # Simulate for 0 days so as to complete all the initialisation steps
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props

    # Get target person and make them HIV-negative man and not ever having had a test and not already circumcised
    person_id = 0
    df.at[person_id, "sex"] = "M"
    df.at[person_id, "li_is_circ"] = False
    df.at[person_id, "hv_inf"] = False
    df.at[person_id, "hv_diagnosed"] = False
    df.at[person_id, "hv_number_tests"] = 0

    # Run the TestAndRefer event
    t = HSI_Hiv_TestAndRefer(module=sim.modules['Hiv'], person_id=person_id)
    t.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that there is an VMMC event scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if isinstance(ev[1], hiv.HSI_Hiv_Circ)
    ][0]

    # Run the event:
    event.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that the person is now circumcised
    assert df.at[person_id, "li_is_circ"]
    assert df.at[person_id, "hv_number_tests"] > 0

def test_hsi_testandrefer_and_behavchg():
    """Test that the HSI for testing and behaviour change works as intended"""
    sim = get_sim()

    # Make the chance of having behaviour change 100%
    sim.modules['Hiv'].lm_behavchg = LinearModel.multiplicative()

    # Simulate for 0 days so as to complete all the initialisation steps
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props

    # Get target person and make them HIV-negative woman who had not previously had behaviour change
    person_id = 0
    df.at[person_id, "sex"] = "F"
    df.at[person_id, "hv_inf"] = False
    df.at[person_id, "hv_diagnosed"] = False
    df.at[person_id, "hv_number_tests"] = 0
    df.at[person_id, "hv_behaviour_change"] = False

    # Run the TestAndRefer event
    t = HSI_Hiv_TestAndRefer(module=sim.modules['Hiv'], person_id=person_id)
    t.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that the person has now had behaviour change
    assert df.at[person_id, "hv_behaviour_change"]
    assert df.at[person_id, "hv_number_tests"] > 0

def test_hsi_testandrefer_and_prep():
    """Test that the HSI for testing and referral to PrEP works as intended"""
    sim = get_sim()

    # Make the chance of being referred 100%
    sim.modules['Hiv'].lm_circ = LinearModel.multiplicative()

    # Simulate for 0 days so as to complete all the initialisation steps
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props

    # Get target person and make them HIV-negative women FSW and not on prep currently
    person_id = 0
    df.at[person_id, "sex"] = "F"
    df.at[person_id, "hv_inf"] = False
    df.at[person_id, "hv_diagnosed"] = False
    df.at[person_id, "hv_number_tests"] = 0
    df.at[person_id, "li_is_sexworker"] = True
    df.at[person_id, "hv_is_on_prep"] = False

    # Run the TestAndRefer event
    t = HSI_Hiv_TestAndRefer(module=sim.modules['Hiv'], person_id=person_id)
    t.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that there is an PrEP event scheduled
    date_hsi_event, hsi_event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueOnPrep)
    ][0]

    # Run the event:
    hsi_event.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that the person is now on PrEP
    assert df.at[person_id, "hv_is_on_prep"]
    assert df.at[person_id, "hv_number_tests"] > 0

    # Check that there is a 'decision' event scheduled
    date_decision_event, decision_event = [
        ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.Hiv_DecisionToContinueOnPrEP)
    ][0]

    assert date_decision_event == date_hsi_event + pd.DateOffset(months=3)

    # Advance simulation date to when the decision_event would run
    sim.date = date_decision_event

    # Run the decision event when probability of continuation is 1.0, and check for a further HSI
    sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_3_months"] = 1.0
    decision_event.apply(person_id)
    assert df.at[person_id, "hv_is_on_prep"]
    date_next_hsi_event, next_hsi_event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueOnPrep) & (ev[0] >= date_decision_event))
    ][0]

    # Run the decision event when probability of continuation is 0, and check that PrEP is off and no further HSI
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()  # clear the queue to avoid being confused by results of the check done just above.
    sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_3_months"] = 0.0
    decision_event.apply(person_id)
    assert not df.at[person_id, "hv_is_on_prep"]
    assert [] == [
        ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueOnPrep) & (ev[0] >= date_decision_event))
    ]
    assert [] == [
        ev[0] for ev in sim.find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.Hiv_DecisionToContinueOnPrEP) & (ev[0] > date_decision_event))
    ]



    # todo- check no more decision events:

def test_hsi_testandrefer_and_art():
    """Test that the HSI for testing and referral to ART works as intended"""
    sim = get_sim()

    # Make the chance of being referred to ART following testing is 100%
    sim.modules['Hiv'].lm_art = LinearModel.multiplicative()

    # Simulate for 0 days so as to complete all the initialisation steps
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props

    # Get target person and make them HIV-positive but not previously diagnosed
    person_id = 0
    df.at[person_id, "sex"] = "F"
    df.at[person_id, "hv_inf"] = True
    df.at[person_id, "hv_diagnosed"] = False
    df.at[person_id, "hv_number_tests"] = 0

    # Run the TestAndRefer event
    t = HSI_Hiv_TestAndRefer(module=sim.modules['Hiv'], person_id=person_id)
    t.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that there is an ART HSI event scheduled
    date_hsi_event, hsi_event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment)
    ][0]

    # Run the event:
    hsi_event.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that the person is now on ART and diagnosed
    assert df.at[person_id, "hv_art"] in ["on_VL_suppressed", "on_not_VL_suppressed"]
    assert df.at[person_id, "hv_diagnosed"]
    assert df.at[person_id, "hv_number_tests"] > 0

    # Check that there is a 'decision' event scheduled
    date_decision_event, decision_event = [
        ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.Hiv_DecisionToContinueTreatment)
    ][0]

    assert date_decision_event == date_hsi_event + pd.DateOffset(months=6)

    # Advance simulation date to when the decision_event would run
    sim.date = date_decision_event

    # Run the decision event when probability of continuation is 1.0, and check for a further HSI
    sim.modules["Hiv"].parameters["probability_of_being_retained_on_art_every_6_months"] = 1.0
    decision_event.apply(person_id)
    assert df.at[person_id, "hv_art"] in ["on_VL_suppressed", "on_not_VL_suppressed"]
    date_next_hsi_event, next_hsi_event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment) & (ev[0] >= date_decision_event))
    ][0]

    # Run the decision event when probability of continuation is 0, and check that PrEP is off and no further HSI
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()  # clear the queue to avoid being confused by results of the check done just above.
    sim.modules["Hiv"].parameters["probability_of_being_retained_on_art_every_6_months"] = 0.0
    decision_event.apply(person_id)
    assert df.at[person_id, "hv_art"] not in ["on_VL_suppressed", "on_not_VL_suppressed"]
    assert [] == [
        ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment) & (ev[0] >= date_decision_event))
    ]
    assert [] == [
        ev[0] for ev in sim.find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.Hiv_DecisionToContinueOnPrEP) & (ev[0] > date_decision_event))
    ]

# todo - test_mtct_from_mother_to_child


