import os
from pathlib import Path

import pytest

from tlo import Simulation, Date
from tlo.methods import demography, contraception

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10


@pytest.fixture(scope='module')
def simulation():
    sim = Simulation(start_date=start_date)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    return sim


def test_mockitis_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_mockitis_simulation(simulation)
