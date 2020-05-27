import os
from pathlib import Path
import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,

    mockitis, contraception, chronicsyndrome, healthsystem, symptommanager, healthburden, healthseekingbehaviour,
    dx_algorithm_child, labour, pregnancy_supervisor, newborn_outcomes)

start_date = Date(2010, 1, 1)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = './resources'


def test_can_look_at_future_events():
    sim = Simulation(start_date=start_date)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(),
                 dx_algorithm_child.DxAlgorithmChild(),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    sim.seed_rngs(0)
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
    events = sim.event_queue.find_events_for_person(person_id=person_id)
    hsi_events = sim.modules['HealthSystem'].find_events_for_person(person_id=person_id)

    assert len(events) > 0
    assert len(hsi_events) > 0

    # TODO: nice way to combine and sort these two lists.
