import os
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
)

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10


@pytest.fixture(scope='module')
def simulation():
    sim = Simulation(start_date=start_date)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    # sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    # sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    # sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    # sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    # sim.register(antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    return sim


def test_mockitis_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_mockitis_simulation(simulation)
