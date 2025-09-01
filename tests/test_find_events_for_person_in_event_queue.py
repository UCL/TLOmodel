import os
from operator import itemgetter
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    simplified_births,
    symptommanager,
)

start_date = Date(2010, 1, 1)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = './resources'


def test_can_look_at_future_events(seed):
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    sim.make_initial_population(n=10)

    # check can look at future events in the sim.event_queue and HSI_EVENT_QUEUE:
    person_id = 0  # select a random person

    # Schedule some events for this person
    # event queue
    dummy_event = mockitis.MockitisDeathEvent(sim.modules['Mockitis'], person_id)
    sim.schedule_event(dummy_event, sim.date)

    # hsi event queue
    dummy_hsi = chronicsyndrome.HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(
        sim.modules['ChronicSyndrome'], person_id=person_id)
    sim.modules['HealthSystem'].schedule_hsi_event(dummy_hsi, priority=0, topen=sim.date,
                                                   tclose=sim.date + pd.DateOffset(days=1))

    # Query the queue of events for this person:
    events = sim.find_events_for_person(person_id=person_id)
    hsi_events = sim.modules['HealthSystem'].find_events_for_person(person_id=person_id)

    all_events = events + hsi_events
    all_sorted_events = sorted(((i, j) for i, j in all_events), key=itemgetter(0))

    assert len(events) > 0
    assert events[0][1] is dummy_event

    assert len(hsi_events) > 0
    assert hsi_events[0][1] is dummy_hsi

    assert len(all_sorted_events) == len(events) + len(hsi_events)
