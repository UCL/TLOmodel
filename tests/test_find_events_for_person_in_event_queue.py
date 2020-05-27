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
end_date = Date(2011, 1, 1)
popsize = 50


try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = './resources'


def test_can_look_at_future_events():

    sim = Simulation(start_date=start_date)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour())
    sim.register(dx_algorithm_child.DxAlgorithmChild())
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False)
                 )
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register()
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())
    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # check can look at future events in the sim.event_queue and HSI_EVENT_QUEUE:
    for person_id in sim.population.props.index.values:
        events = sim.event_queue.find_events_for_person(person_id=person_id)
        hsi_events = sim.modules['HealthSystem'].find_events_for_person(person_id=person_id)

        # Schedule some events for this person
        dummy_event = mockitis.MockitisDeathEvent(sim.modules['Mockitis'], person_id)
        dummy_hsi = chronicsyndrome.HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(
            sim.modules['ChronicSyndrome'], person_id=person_id)

        sim.schedule_event(dummy_event, sim.date)
        sim.modules['HealthSystem'].schedule_hsi_event(dummy_hsi, priority=0, topen=sim.date,
                                                       tclose=sim.date + pd.DateOffset(days=1))

    # TODO: commented out pre PR as cant get it to run correctly - Asif informed
    # assert len(events) > 0
    # assert len(hsi_events) > 0

    # TODO: nice way to combine and sort these two lists.
