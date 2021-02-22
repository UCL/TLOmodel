import os
from pathlib import Path

import pytest

from tlo import Simulation, Date
from tlo.methods import demography, simplified_births

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 1000


@pytest.fixture(scope='module')
def simulation():
    sim = Simulation(start_date=start_date)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))

    # Instantiate and add the simplified_births module to the simulation
    simplified_births_module = simplified_births.Simplifiedbirths()
    sim.register(simplified_births_module)

    return sim


def test_simplified_births_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_simplified_births_simulation(simulation)
