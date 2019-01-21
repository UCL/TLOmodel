import logging

import pytest

from tlo import Simulation, Date
from tlo.methods import demography, lifestyle
from tlo.test import random_birth

path = 'C:/Users/Andrew Phillips/Dropbox/Thanzi la Onse/05 - Resources\Demographic data\Demography_WorkingFile_Complete.xlsx'
# Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 3000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    lifestyle_module = lifestyle.Lifestyle()
    sim.register(core_module)
    sim.register(lifestyle_module)

    logging.getLogger('tlo.methods.demography').setLevel(logging.WARNING)

    return sim


def test_lifestyle_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_lifestyle_simulation(simulation)

