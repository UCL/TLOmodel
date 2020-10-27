"""Test for for the HIV Module."""

import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.events import IndividualScopeEventMixin
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
    symptommanager, dx_algorithm_child,
)
from tlo.methods.hiv import Hiv, HSI_Hiv_TestAndRefer

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


start_date = Date(2010, 1, 1)
end_date = Date(2015, 12, 31)
popsize = 1000

def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def get_sim():
    """get sim with the checks for configuration of properties running in the HIV module"""
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=True)
                 )

    # Edit the efficacy of PrEP to be perfect (for the purpose of these tests)
    sim.modules['Hiv'].parameters['proportion_reduction_in_risk_of_hiv_aq_if_on_prep'] = 1.0

    # Make the population
    sim.make_initial_population(n=popsize)
    return sim

def test_basic_run_with_default_parameters():
    """Run the HIV module with check and check dtypes consistency"""

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

def test_hsi_spontaneoustest():
    """Test that the spontaneous test HSI works as intended"""
    # Get simulation and simulate for 0 days so as to complete all the initialisation steps
    sim = get_sim()
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props

    # Get target person and make them HIV-negative and not ever having had a test
    person_id = 0
    df.at[person_id, "hv_inf"] = False
    df.at[person_id, "hv_diagnosed"] = False
    df.at[person_id, "hv_number_tests"] = 0

    # Run the SpontaneousTest event
    t = HSI_Hiv_TestAndRefer(module=sim.modules['Hiv'], person_id=person_id)
    t.apply(person_id=person_id, squeeze_factor=0.0)


# todo - test that the event is run is aids symptoms occur
