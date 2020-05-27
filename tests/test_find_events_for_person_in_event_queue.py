import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,

    mockitis, contraception, chronicsyndrome, healthsystem, symptommanager, healthburden, healthseekingbehaviour,
    dx_algorithm_child, labour, pregnancy_supervisor)

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 1000


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
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())
    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # check can look at future events in the sim.event_queue and HSI_EVENT_QUEUE:
    for person_id in sim.population.props.index.values:
        events = sim.event_queue.find_events_for_person(person_id=person_id)
        hsi_events = sim.modules['HealthSystem'].find_events_for_person(person_id=person_id)

    assert len(events) > 0
    assert len(hsi_events) > 0

    # TODO: nice way to combine and sort these two lists.
