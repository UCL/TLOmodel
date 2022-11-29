"""Test for for the HIV Module"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.lm import LinearModel
from tlo.methods import (
    care_of_women_during_pregnancy,
    demography,
    enhanced_lifestyle,
    epi,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    hsi_generic_first_appts,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_helper_functions,
    pregnancy_supervisor,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.methods.healthseekingbehaviour import HealthSeekingBehaviourPoll
from tlo.methods.healthsystem import HealthSystemScheduler
from tlo.methods.hiv import (
    HivAidsOnsetEvent,
    HSI_Hiv_StartOrContinueTreatment,
    HSI_Hiv_TestAndRefer,
)

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


def get_sim(seed, use_simplified_birth=True, cons_availability='all'):
    """get sim with the checks for configuration of properties running in the HIV module"""
    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    if use_simplified_birth:
        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(resourcefilepath=resourcefilepath, cons_availability=cons_availability),
                     symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                     healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                     epi.Epi(resourcefilepath=resourcefilepath),
                     hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=True),
                     tb.Tb(resourcefilepath=resourcefilepath),
                     )
    else:
        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                     care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                     labour.Labour(resourcefilepath=resourcefilepath),
                     newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                     postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(resourcefilepath=resourcefilepath, cons_availability=cons_availability),
                     symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                     healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                     epi.Epi(resourcefilepath=resourcefilepath),
                     hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=True),
                     tb.Tb(resourcefilepath=resourcefilepath),
                     # Disable check to avoid error due to lack of Contraception module
                     check_all_dependencies=False,
                     )

    # Edit the efficacy of PrEP to be perfect (for the purpose of these tests)
    sim.modules["Hiv"].parameters[
        "proportion_reduction_in_risk_of_hiv_aq_if_on_prep"
    ] = 1.0
    # Let there be a 100% probability of TestAndRefer events being scheduled
    testing_rates = sim.modules["Hiv"].parameters["hiv_testing_rates"]
    testing_rates["annual_testing_rate_children"] = 1.0
    testing_rates["annual_testing_rate_adults"] = 1.0

    # Make the population
    sim.make_initial_population(n=popsize)
    return sim


def start_sim_and_clear_event_queues(sim):
    """Simulate for 0 days so as to complete all the initialisation steps, but then clear the event queues"""
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()
    return sim


@pytest.mark.slow
def test_basic_run_with_default_parameters(seed):
    """Run the HIV module with check and check dtypes consistency"""
    end_date = Date(2015, 12, 31)

    sim = get_sim(seed=seed)
    check_dtypes(sim)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    # confirm configuration of properties at the end of the simulation:
    sim.modules['Hiv'].check_config_of_properties()


def test_initialisation(seed):
    """check that the natural history plays out as expected for those that are infected at the beginning of the sim"""

    # get simulation and initialise the simulation:
    sim = get_sim(seed=seed)
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


def test_generation_of_new_infection(seed):
    """Check that the generation of new infections is as expected.
    This occurs in the Main Polling Event.
    """

    sim = get_sim(seed=seed)
    pollevent = hiv.HivRegularPollingEvent(module=sim.modules['Hiv'])
    df = sim.population.props

    def any_hiv_infection_event_in_queue():
        for date, _, event in sim.event_queue.queue:
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


def test_generation_of_natural_history_process_no_art(seed):
    """Check that:
    * New infections leads to a scheduled AIDS event
    * AIDS events lead to a scheduled AIDS death when no ART
    * The AIDS death event results in an actual death when no ART
    """

    sim = get_sim(seed=seed)
    sim.modules['Hiv'].parameters["prop_delayed_aids_onset"] = 0.0

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
    assert "aids_symptoms" in sim.modules['SymptomManager'].has_what(person_id)

    # find the AIDS death event for this person
    date_aids_death_event, aids_death_event = \
        [ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.HivAidsDeathEvent)][0]
    assert date_aids_death_event > sim.date

    # run the AIDS death event for this person:
    aids_death_event.apply(person_id)

    # confirm the person is dead
    assert False is bool(df.at[person_id, "is_alive"])
    assert sim.date == df.at[person_id, "date_of_death"]
    assert "AIDS_non_TB" == df.at[person_id, "cause_of_death"]


def test_generation_of_natural_history_process_with_art_before_aids(seed):
    """Check that:
    * New infections leads to a scheduled AIDS event
    * If on ART before AIDS onset, the AIDS events does not do anything and does not lead to a scheduled AIDS death
    """

    sim = get_sim(seed=seed)
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


def test_generation_of_natural_history_process_with_art_after_aids(seed):
    """Check that:
    * New infections leads to a scheduled AIDS event
    * AIDS event leads to AIDS death scheduled
    * If on ART before AIDS death, the AIDS Death does not do anything and does not lead to an actual death
    """

    sim = get_sim(seed=seed)
    sim.modules['Hiv'].parameters["prop_delayed_aids_onset"] = 0.0

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
    assert np.isnan(df.at[person_id, "cause_of_death"])


def test_mtct_at_birth(seed):
    """Check that:
    * HIV infection events are created when the mother during breastfeeding
    """

    sim = get_sim(seed=seed)

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
    sim.modules['Demography'].on_birth(mother_id, child_id)
    sim.modules['Hiv'].on_birth(mother_id, child_id)

    # Check that child is now HIV-positive
    assert sim.population.props.at[child_id, "hv_inf"]


def test_mtct_during_breastfeeding_if_mother_infected_already(seed):
    """Check that:
    * HIV infection events are created during breastfeeding if the mother is HIV-positive (prior to the birth)
    """

    sim = get_sim(seed=seed)

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


def test_mtct_during_breastfeeding_if_mother_infected_during_breastfeeding(seed):
    """Check that:
    * HIV infection events are created during breastfeeding if the mother is infected _whilst_ breastfeeding.
    """

    sim = get_sim(seed=seed)

    # Manipulate MTCT rates so that transmission always occurs during bf is the mother is HIV-pos
    sim.modules['Hiv'].parameters["prob_mtct_treated"] = 0.0
    sim.modules['Hiv'].parameters["prob_mtct_untreated"] = 0.0
    sim.modules['Hiv'].parameters["prob_mtct_incident_preg"] = 0.0
    sim.modules['Hiv'].parameters["monthly_prob_mtct_bf_treated"] = 1.0
    sim.modules['Hiv'].parameters["monthly_prob_mtct_bf_untreated"] = 1.0

    # Do a birth from a mother that is HIV-negative:
    df = sim.population.props
    mother_id = df.loc[df.is_alive & (df.sex == "F")].index[0]
    df.at[mother_id, 'hv_inf'] = False
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date

    child_id = sim.population.do_birth()
    sim.modules['Demography'].on_birth(mother_id, child_id)
    sim.modules['Hiv'].on_birth(mother_id, child_id)

    # Check child is not yet HIV-positive
    assert not sim.population.props.at[child_id, "hv_inf"]

    # Check that there is no infection event for the child:
    assert 0 == len([
        ev for ev in sim.find_events_for_person(child_id) if isinstance(ev[1], hiv.HivInfectionDuringBreastFeedingEvent)
    ])

    # Let the mother become infected
    inf_event = hiv.HivInfectionEvent(person_id=mother_id, module=sim.modules['Hiv'])
    inf_event.apply(mother_id)

    # Check that there is now an infection event scheduled for the child
    assert 1 == len([
        ev for ev in sim.find_events_for_person(child_id) if isinstance(ev[1], hiv.HivInfectionDuringBreastFeedingEvent)
    ])


def test_test_and_refer_event_scheduled_by_main_event_poll(seed):
    """Check that the main event poll causes there to be event of the HSI_TestAndRefer"""

    sim = get_sim(seed=seed)

    # set baseline testing probability to far exceed 1.0 to ensure everyone assigned a test after lm and scaling
    sim.modules['Hiv'].parameters["hiv_testing_rates"]["annual_testing_rate_children"] = 100
    sim.modules['Hiv'].parameters["hiv_testing_rates"]["annual_testing_rate_adults"] = 100

    # Simulate for 0 days so as to complete all the initialisation steps
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Run a polling event
    pollevent = hiv.HivRegularPollingEvent(module=sim.modules["Hiv"])
    pollevent.apply(sim.population)

    # Check number and dates of TestAndRefer events in the HSI Event Queue
    dates_of_tr_events = [
        ev[1] for ev in sim.modules['HealthSystem'].HSI_EVENT_QUEUE if isinstance(ev[4], hiv.HSI_Hiv_TestAndRefer)
    ]

    df = sim.population.props
    num_not_diagnosed = sum(~df.hv_diagnosed & df.is_alive)
    # diagnosed adults can re-test, so should have more tests than undiagnosed people
    assert num_not_diagnosed <= len(dates_of_tr_events)
    assert all([(sim.date <= d <= (sim.date + pd.DateOffset(months=12))) for d in dates_of_tr_events])


def test_aids_symptoms_lead_to_treatment_being_initiated(seed):
    """Check that if aids-symptoms onset then treatment can be initiated (even without spontaneous testing)"""

    # Set up simulation object in custom way:
    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                               # force symptoms to lead to health care seeking:
                                                               force_any_symptom_to_lead_to_healthcareseeking=True),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=True),
                 tb.Tb(resourcefilepath=resourcefilepath),
                 )

    # Let there be a 0% probability of TestAndRefer events being scheduled
    sim.modules['Hiv'].parameters['prob_spontaneous_test_12m'] = 0.0

    # Make the population and simulate for 0 days to get everything initialised:
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.date)

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
    # set this cause (TB) to make sure AIDS onset occurs
    aids_event = HivAidsOnsetEvent(person_id=person_id, module=sim.modules['Hiv'], cause='AIDS_TB')
    aids_event.apply(person_id)

    # Confirm that they have aids symptoms and an AIDS death schedule
    assert 'aids_symptoms' in sim.modules['SymptomManager'].has_what(person_id)
    assert 1 == len(
        [ev[0] for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.HivAidsTbDeathEvent)])

    # Run the health-seeking poll and run the GenericFirstApptLevel0 that is Created
    hsp = HealthSeekingBehaviourPoll(module=sim.modules['HealthSeekingBehaviour'])
    hsp.apply(sim.population)
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1], hsi_generic_first_appts.HSI_GenericFirstApptAtFacilityLevel0)][0]
    ge.apply(ge.target, squeeze_factor=0.0)

    # Check that the person has a TestAndReferEvent scheduled
    assert 1 == len([ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
                     isinstance(ev[1], hiv.HSI_Hiv_TestAndRefer)])


def test_art_is_initiated_for_infants(seed):
    """Check that infant infected at birth, and tested, diagnosed and start ART"""
    # This test ensures that HIVTestAndRefer is scheduled for all newborns who
    # pass through the newborn HSI (i.e. their mother gave birth in a health facility). We therefore now assume that no
    # newborns born at home get HIV testing unless they interact with the health system at a different point

    # Create simulation object that uses the Newborn module:
    sim = get_sim(seed=seed, use_simplified_birth=False)

    # Simulate for 0 days so as to complete all the initialisation steps
    sim = start_sim_and_clear_event_queues(sim)

    # Manipulate MTCT rates so that transmission always occurs at/before birth
    sim.modules['Hiv'].parameters["prob_mtct_treated"] = 1.0
    sim.modules['Hiv'].parameters["prob_mtct_untreated"] = 1.0
    sim.modules['Hiv'].parameters["prob_mtct_incident_preg"] = 1.0

    # change prob ART start after diagnosis
    sim.modules["Hiv"].parameters["prob_start_art_or_vs"]["prob_art_if_dx"] = 1.0

    # Manipulate CFR for deaths due to not breathing at birth
    sim.modules['NewbornOutcomes'].parameters['cfr_failed_to_transition'] = 0.0

    # Do a birth from a mother that is HIV-positive:
    df = sim.population.props
    mother_id = df.loc[df.is_alive & (df.sex == "F")].index[0]
    df.at[mother_id, 'hv_inf'] = True
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'hv_date_inf'] = sim.date
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date

    # Populate the mni
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'hospital'

    # Do birth
    child_id = sim.do_birth(mother_id)
    sim.modules['Demography'].on_birth(mother_id, child_id)
    sim.modules['Hiv'].on_birth(mother_id, child_id)

    # Check that child is now HIV-positive but not diagnosed
    assert sim.population.props.at[child_id, "hv_inf"]
    assert not sim.population.props.at[child_id, "hv_diagnosed"]

    # Define the newborn HSI and run the event
    newborn_pnc = newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck(
        module=sim.modules['NewbornOutcomes'], person_id=child_id)

    newborn_pnc.apply(person_id=child_id, squeeze_factor=0.0)

    # Check that the child has a TestAndRefer event scheduled via the newborn HSI
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(child_id) if
        isinstance(ev[1], hiv.HSI_Hiv_TestAndRefer)
    ][0]
    assert date_event == sim.date

    # Run the TestAndRefer event for the child
    rtn = event.apply(person_id=child_id, squeeze_factor=0.0)

    # check that the event returned a footprint for a VCTPositive
    assert rtn == event.make_appt_footprint({'VCTPositive': 1.0})

    # check that child is now diagnosed
    assert sim.population.props.at[child_id, "hv_diagnosed"]

    # Check that the child has an art initiation event scheduled
    assert 1 == len([
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(child_id) if
        isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment)
    ])


def test_hsi_testandrefer_and_circ(seed):
    """Test that the HSI for testing and referral to circumcision works as intended"""
    sim = get_sim(seed=seed)
    sim = start_sim_and_clear_event_queues(sim)

    # Make the chance of being referred 100%
    sim.modules['Hiv'].lm['lm_circ'] = LinearModel.multiplicative()
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
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], hiv.HSI_Hiv_Circ)
    ][0]

    # Run the event:
    event.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that the person is now circumcised
    assert df.at[person_id, "li_is_circ"]
    assert df.at[person_id, "hv_number_tests"] > 0


def test_hsi_testandrefer_and_behavchg(seed):
    """Test that the HSI for testing and behaviour change works as intended"""
    sim = get_sim(seed=seed)
    sim = start_sim_and_clear_event_queues(sim)

    # Make the chance of having behaviour change 100%
    sim.modules['Hiv'].lm['lm_behavchg'] = LinearModel.multiplicative()

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


def test_hsi_testandrefer_and_prep(seed):
    """Test that the HSI for testing and referral to PrEP works as intended"""
    sim = get_sim(seed=seed)
    sim = start_sim_and_clear_event_queues(sim)

    # Make the chance of being referred 100%
    sim.modules['Hiv'].lm['lm_prep'] = LinearModel.multiplicative()

    df = sim.population.props

    # Get target person and make them HIV-negative women FSW and not on prep currently
    person_id = 0
    df.at[person_id, "sex"] = "F"
    df.at[person_id, "hv_inf"] = False
    df.at[person_id, "hv_diagnosed"] = False
    df.at[person_id, "hv_number_tests"] = 0
    df.at[person_id, "li_is_sexworker"] = True
    df.at[person_id, "hv_is_on_prep"] = False

    # change PrEP start date so will occur from 01-01-2010
    sim.modules['Hiv'].parameters["prep_start_year"] = 2010

    # Run the TestAndRefer event
    t = HSI_Hiv_TestAndRefer(module=sim.modules['Hiv'], person_id=person_id)
    t.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that there is an PrEP event scheduled
    date_hsi_event, hsi_event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueOnPrep)
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
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueOnPrep) & (ev[0] >= date_decision_event))
    ][0]

    # Run the decision event when probability of continuation is 0, and check that PrEP is off and no further HSI or
    # "decision" events
    # - First, clear the queue to avoid being confused by results of the check done just above.
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_3_months"] = 0.0
    decision_event.apply(person_id)
    assert not df.at[person_id, "hv_is_on_prep"]
    assert 0 == len([
        ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueOnPrep) & (ev[0] >= date_decision_event))
    ])
    assert 0 == len([
        ev[0] for ev in sim.find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.Hiv_DecisionToContinueOnPrEP) & (ev[0] > date_decision_event))
    ])


def test_hsi_testandrefer_and_art(seed):
    """Test that the HSI for testing and referral to ART works as intended
    Check that ART is stopped (and AIDS event scheduled) if person decides not to continue ART"""
    sim = get_sim(seed=seed)

    sim = start_sim_and_clear_event_queues(sim)

    # Make the chance of being referred to ART following testing is 100%
    sim.modules['Hiv'].lm['lm_art'] = LinearModel.multiplicative()

    # change prob ART start after diagnosis
    sim.modules["Hiv"].parameters["prob_start_art_or_vs"]["prob_art_if_dx"] = 1.0

    # Make sure that the person will continue to seek care
    sim.modules['Hiv'].parameters["probability_of_seeking_further_art_appointment_if_drug_not_available"] = 1.0
    sim.modules['Hiv'].parameters["probability_of_seeking_further_art_appointment_if_appointment_not_available"] = 1.0

    # Get target person and make them HIV-positive adult but not previously diagnosed
    df = sim.population.props
    person_id = 0
    df.at[person_id, "sex"] = "F"
    df.at[person_id, "hv_inf"] = True
    df.at[person_id, "hv_diagnosed"] = False
    df.at[person_id, "hv_number_tests"] = 0
    df.at[person_id, "age_years"] = 40

    # Run the TestAndRefer event
    t = HSI_Hiv_TestAndRefer(module=sim.modules['Hiv'], person_id=person_id)
    rtn = t.apply(person_id=person_id, squeeze_factor=0.0)

    # check that the footprint is updated to be that of a positive person and that the person is diagnosed
    assert rtn == t.make_appt_footprint({'VCTPositive': 1.0})
    assert df.at[person_id, 'hv_diagnosed']

    # Check that there is an ART HSI event scheduled
    date_hsi_event, hsi_event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment)
    ][0]

    # Run the event:
    hsi_event.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that the person is now on ART and diagnosed and does not have symptoms of aids
    assert df.at[person_id, "hv_art"] in ["on_VL_suppressed", "on_not_VL_suppressed"]
    assert df.at[person_id, "hv_diagnosed"]
    assert df.at[person_id, "hv_number_tests"] > 0
    assert "aids_symptoms" not in sim.modules['SymptomManager'].has_what(person_id=person_id)

    # Check that there is a 'decision' event scheduled
    date_decision_event, decision_event = [
        ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.Hiv_DecisionToContinueTreatment)
    ][0]

    assert date_decision_event == date_hsi_event + pd.DateOffset(months=3)

    # Advance simulation date to when the decision_event would run
    sim.date = date_decision_event

    # Run the decision event when probability of continuation is 1.0, and check for a further HSI
    sim.modules["Hiv"].parameters["probability_of_being_retained_on_art_every_6_months"] = 1.0
    decision_event.apply(person_id)
    assert df.at[person_id, "hv_art"] in ["on_VL_suppressed", "on_not_VL_suppressed"]
    assert 1 == len([
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment) & (ev[0] >= date_decision_event))
    ])

    # Check stops being on ART if "decides" to stop ->
    # Run the decision event when probability of continuation is 0, and check that Treatment is off and
    # another treatment appt is scheduled
    # First, clear the queue to avoid being confused by results of the check done just above.
    assert df.at[person_id, "hv_art"] in ["on_VL_suppressed", "on_not_VL_suppressed"]
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.modules["Hiv"].parameters["probability_of_being_retained_on_art_every_3_months"] = 0.0
    decision_event.apply(person_id)
    assert df.at[person_id, "hv_art"] == "not"
    assert 1 == len([
        ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment) & (ev[0] >= date_decision_event))
    ])
    assert 0 == len([
        ev[0] for ev in sim.find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.Hiv_DecisionToContinueTreatment) & (ev[0] > date_decision_event))
    ])

    # check that there is at least one AIDS event scheduled following when the person stopped treatment:
    assert 0 < len([
        ev[0] for ev in sim.find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.HivAidsOnsetEvent) & (ev[0] >= date_decision_event))
    ])


def test_hsi_art_stopped_due_to_no_drug_available_and_no_restart(seed):
    """Check that if drug not available at HSI, person will default off ART.
    If set not to restart, will have no further HSI"""

    sim = get_sim(seed=seed, cons_availability='none')  # make sure consumables for art are *NOT* available:
    sim = start_sim_and_clear_event_queues(sim)

    # Get target person and make them HIV-positive adult, diagnosed and on ART
    df = sim.population.props
    person_id = 0
    df.at[person_id, "sex"] = "F"
    df.at[person_id, "hv_inf"] = True
    df.at[person_id, "hv_art"] = "on_VL_suppressed"
    df.at[person_id, "hv_diagnosed"] = True
    df.at[person_id, "hv_number_tests"] = 1
    df.at[person_id, "age_years"] = 40

    # Make and run the Treatment event (when consumables not available): and the person will not try to restart
    sim.modules['Hiv'].parameters["probability_of_seeking_further_art_appointment_if_drug_not_available"] = 0.0
    t = HSI_Hiv_StartOrContinueTreatment(module=sim.modules['Hiv'],
                                         person_id=person_id, facility_level_of_this_hsi="1a")
    t.apply(person_id=person_id, squeeze_factor=0.0)

    # confirm person is no longer on ART and has an AIDS event scheduled:
    assert df.at[person_id, "hv_art"] == "not"
    assert 1 == len([
        ev[0] for ev in sim.find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.HivAidsOnsetEvent) & (ev[0] >= sim.date))
    ])
    # confirm new treatment appt scheduled for this person
    assert 1 == len([
        ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment) & (ev[0] >= sim.date))
    ])


def test_hsi_art_stopped_due_to_no_drug_available_but_will_restart(seed):
    """Check that if drug not available at HSI, person will default off ART.
    If set not restart, will have a further HSI scheduled"""

    sim = get_sim(seed=seed, cons_availability='none')  # make sure consumables for art are *NOT* available:
    sim = start_sim_and_clear_event_queues(sim)

    # Get target person and make them HIV-positive adult, diagnosed and on ART
    df = sim.population.props
    person_id = 0
    df.at[person_id, "sex"] = "F"
    df.at[person_id, "hv_inf"] = True
    df.at[person_id, "hv_art"] = "on_VL_suppressed"
    df.at[person_id, "hv_diagnosed"] = True
    df.at[person_id, "hv_number_tests"] = 1
    df.at[person_id, "age_years"] = 40

    # Make and run the Treatment event (when consumables not available): and the person will try to restart
    sim.modules['Hiv'].parameters["probability_of_seeking_further_art_appointment_if_drug_not_available"] = 1.0
    t = HSI_Hiv_StartOrContinueTreatment(module=sim.modules['Hiv'], person_id=person_id,
                                         facility_level_of_this_hsi="1a")
    t.apply(person_id=person_id, squeeze_factor=0.0)

    # confirm person is no longer on ART and has an AIDS event scheduled:
    assert df.at[person_id, "hv_art"] == "not"
    assert 1 == len([
        ev[0] for ev in sim.find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.HivAidsOnsetEvent) & (ev[0] >= sim.date))
    ])
    # confirm HSI Treatment event has been scheduled for this person
    assert 1 == len([
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment) & (ev[0] >= sim.date))
    ])


def test_hsi_art_stopped_if_healthsystem_cannot_run_hsi_and_no_restart(seed):
    # Make the health-system unavailable to run any HSI event
    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2,
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=True),
                 tb.Tb(resourcefilepath=resourcefilepath),
                 )

    # Make the population
    sim.make_initial_population(n=popsize)

    # Get the simulation running and clear the event queues:
    sim = start_sim_and_clear_event_queues(sim)

    # make persons try to restart if HSI are not being run
    sim.modules['Hiv'].parameters["probability_of_seeking_further_art_appointment_if_appointment_not_available"] = 0.0

    # person_id=0:  HIV-positive adult, diagnosed and ready to start ART
    df = sim.population.props
    df.at[0, "sex"] = "F"
    df.at[0, "hv_inf"] = True
    df.at[0, "hv_art"] = "not"
    df.at[0, "hv_diagnosed"] = True
    df.at[0, "hv_number_tests"] = 1
    df.at[0, "age_years"] = 40

    # person-id=1: HIV-positive adult, diagnosed and already on ART
    df = sim.population.props
    df.at[1, "sex"] = "F"
    df.at[1, "hv_inf"] = True
    df.at[1, "hv_art"] = "on_VL_suppressed"
    df.at[1, "hv_diagnosed"] = True
    df.at[1, "hv_number_tests"] = 1
    df.at[1, "age_years"] = 40

    # schedule each person  a treatment
    sim.modules['HealthSystem'].schedule_hsi_event(
        HSI_Hiv_StartOrContinueTreatment(person_id=0, module=sim.modules['Hiv'], facility_level_of_this_hsi="1a"),
        topen=sim.date,
        tclose=sim.date + pd.DateOffset(days=1),
        priority=0
    )
    sim.modules['HealthSystem'].schedule_hsi_event(
        HSI_Hiv_StartOrContinueTreatment(person_id=1, module=sim.modules['Hiv'], facility_level_of_this_hsi="1a"),
        topen=sim.date,
        tclose=sim.date + pd.DateOffset(days=1),
        priority=0
    )

    # Run the HealthSystemScheduler for the days (the HSI should not be run and the never_run function should be called)
    hss = HealthSystemScheduler(module=sim.modules['HealthSystem'])
    for i in range(3):
        sim.date = sim.date + pd.DateOffset(days=i)
        hss.apply(sim.population)

    # check that neither person is not on ART
    assert df.at[0, "hv_art"] == "not"
    assert df.at[1, "hv_art"] == "not"

    # check that NO further HSI treatment event has been scheduled for the future for each person
    assert 0 == len([
        ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id=0) if
        (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment) & (ev[0] >= sim.date))
    ])
    assert 0 == len([
        ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id=1) if
        (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment) & (ev[0] >= sim.date))
    ])

    # check that no additional AIDS Onset Event has been scheduled for person 0 (who had not started ART)
    assert 0 == len([
        ev[0] for ev in sim.find_events_for_person(person_id=0) if
        (isinstance(ev[1], hiv.HivAidsOnsetEvent) & (ev[0] >= sim.date))
    ])

    # check that there is a new AIDS Onset Event for person 1 (who had started ART)
    assert 1 == len([
        ev[0] for ev in sim.find_events_for_person(person_id=1) if
        (isinstance(ev[1], hiv.HivAidsOnsetEvent) & (ev[0] >= sim.date))
    ])


def test_hsi_art_stopped_if_healthsystem_cannot_run_hsi_but_will_restart(seed):
    # Make the health-system unavailable to run any HSI event

    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2,
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=True),
                 tb.Tb(resourcefilepath=resourcefilepath),
                 )

    # Make the population
    sim.make_initial_population(n=popsize)

    # Get the simulation running and clear the event queues:
    sim = start_sim_and_clear_event_queues(sim)

    # make persons try to restart if HSI are not being run
    sim.modules['Hiv'].parameters["probability_of_seeking_further_art_appointment_if_appointment_not_available"] = 1.0

    # person_id=0:  HIV-positive adult, diagnosed and ready to start ART
    df = sim.population.props
    df.at[0, "sex"] = "F"
    df.at[0, "hv_inf"] = True
    df.at[0, "hv_art"] = "not"
    df.at[0, "hv_diagnosed"] = True
    df.at[0, "hv_number_tests"] = 1
    df.at[0, "age_years"] = 40

    # person-id=1: HIV-positive adult, diagnosed and already on ART
    df = sim.population.props
    df.at[1, "sex"] = "F"
    df.at[1, "hv_inf"] = True
    df.at[1, "hv_art"] = "on_VL_suppressed"
    df.at[1, "hv_diagnosed"] = True
    df.at[1, "hv_number_tests"] = 1
    df.at[1, "age_years"] = 40

    # schedule each person  a treatment
    sim.modules['HealthSystem'].schedule_hsi_event(
        HSI_Hiv_StartOrContinueTreatment(person_id=0, module=sim.modules['Hiv'], facility_level_of_this_hsi="1a"),
        topen=sim.date,
        tclose=sim.date + pd.DateOffset(days=1),
        priority=0
    )
    sim.modules['HealthSystem'].schedule_hsi_event(
        HSI_Hiv_StartOrContinueTreatment(person_id=1, module=sim.modules['Hiv'], facility_level_of_this_hsi="1a"),
        topen=sim.date,
        tclose=sim.date + pd.DateOffset(days=1),
        priority=0
    )

    # Run the HealthSystemScheduler for the days (the HSI should not be run and the never_run function should be called)
    hss = HealthSystemScheduler(module=sim.modules['HealthSystem'])
    for i in range(3):
        sim.date = sim.date + pd.DateOffset(days=i)
        hss.apply(sim.population)

    # check that neither person is not on ART
    assert df.at[0, "hv_art"] == "not"
    assert df.at[1, "hv_art"] == "not"

    # check that a HSI treatment event has been scheduled for the future for each person
    assert 1 == len([
        ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id=0) if
        (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment) & (ev[0] >= sim.date))
    ])
    assert 1 == len([
        ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id=1) if
        (isinstance(ev[1], hiv.HSI_Hiv_StartOrContinueTreatment) & (ev[0] >= sim.date))
    ])

    # check that no additional AIDS Onset Event has been scheduled for person 0 (who had not started ART)
    assert 0 == len([
        ev[0] for ev in sim.find_events_for_person(person_id=0) if
        (isinstance(ev[1], hiv.HivAidsOnsetEvent) & (ev[0] >= sim.date))
    ])

    # check that there is a new AIDS Onset Event for person 1 (who had started ART)
    assert 1 == len([
        ev[0] for ev in sim.find_events_for_person(person_id=1) if
        (isinstance(ev[1], hiv.HivAidsOnsetEvent) & (ev[0] >= sim.date))
    ])


def test_use_dummy_version(seed):
    """check that the dummy version of the HIV module works ok: provides the hv_inf property as a bool"""
    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 hiv.DummyHivModule(hiv_prev=1.0),
                 tb.DummyTbModule(active_tb_prev=0.01),
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=Date(2014, 12, 31))

    check_dtypes(sim)

    df = sim.population.props
    assert df.dtypes['hv_inf'].name == 'bool'
    assert df.loc[df.is_alive, 'hv_inf'].all()


@pytest.mark.slow
def test_baseline_hiv_prevalence(seed):
    """
    check baseline prevalence set correctly
    """

    # get data on 2010 prevalence
    # HIV resourcefile
    xls = pd.ExcelFile(resourcefilepath / "ResourceFile_HIV.xlsx")
    prev_data = pd.read_excel(xls, sheet_name="DHS_prevalence")

    adult_prev_1549_data = prev_data.loc[
        (prev_data.Year == 2010, "HIV prevalence among general population 15-49")].values[0] / 100
    female_prev_1549_data = prev_data.loc[
        (prev_data.Year == 2010, "HIV prevalence among women 15-49")].values[0] / 100
    male_prev_1549_data = prev_data.loc[
        (prev_data.Year == 2010, "HIV prevalence among men 15-49")].values[0] / 100

    start_date = Date(2010, 1, 1)
    popsize = 100000
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=False),
                 tb.Tb(resourcefilepath=resourcefilepath),
                 )

    # Make the population
    sim.make_initial_population(n=popsize)
    df = sim.population.props

    adult_prev_1549 = len(
        df[df.hv_inf & df.is_alive & df.age_years.between(15, 49)]
    ) / len(df[df.is_alive & df.age_years.between(15, 49)])
    assert np.isclose(adult_prev_1549, adult_prev_1549_data, rtol=0.05)

    female_prev_1549 = len(
        df[df.hv_inf & df.is_alive & df.age_years.between(15, 49) & (df.sex == "F")]
    ) / len(df[df.is_alive & df.age_years.between(15, 49) & (df.sex == "F")])
    assert np.isclose(female_prev_1549, female_prev_1549_data, rtol=0.05)

    male_prev_1549 = len(
        df[df.hv_inf & df.is_alive & df.age_years.between(15, 49) & (df.sex == "M")]
    ) / len(df[df.is_alive & df.age_years.between(15, 49) & (df.sex == "M")])
    assert np.isclose(male_prev_1549, male_prev_1549_data, rtol=0.05)
