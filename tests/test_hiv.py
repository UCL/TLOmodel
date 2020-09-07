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
    symptommanager,
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


start_date = Date(2010, 1, 1)
end_date = Date(2010, 12, 31)
popsize = 1000

def get_sim():

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
                 hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=True)
                 )
    sim.make_initial_population(n=popsize)
    return sim

def test_basic_run_with_default_parameters():
    """Run the HIV module with check and check dtypes consistency"""

    sim = get_sim()

    sim.modules['Hiv'].check_config_of_properties()
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    sim.modules['Hiv'].check_config_of_properties()


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

    # If lots of people living with HIV but all VL suppressed, no new infections
    df.hv_inf = sim.rng.rand(len(df.hv_inf)) < 0.5
    df.hv_art.values[:] = 'on_VL_suppressed'
    pollevent.apply(sim.population)
    assert not any_hiv_infection_event_in_queue()

    # If lots of people living with HIV, some new infections
    df.hv_art.values[:] = 'not'
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
    assert sim.date == df.at[person_id, 'hv_date_aids']
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
    assert pd.isnull(df.at[person_id, 'hv_date_aids'])

    # check no AIDS onset event for this person
    assert [] == [ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], hiv.HivAidsDeathEvent)]

    # check no AIDS symptoms for this person
    assert [] == sim.modules['SymptomManager'].has_what(person_id)

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
    assert sim.date == df.at[person_id, 'hv_date_aids']

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
