import logging

import pytest

from tlo import Simulation, Date
from tlo.methods import demography, depression
from tlo.test import random_birth

path = 'C:/Users/Andrew Phillips/Dropbox/Thanzi la Onse/05 - Resources\Demographic data\Demography_WorkingFile_Complete.xlsx'
# Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
# if end_date = Date(2010, 1, 1) then population.props are the baseline values
end_date = Date(2015, 1, 1)
popsize = 10000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    depression_module = depression.Depression()
    sim.register(core_module)
    sim.register(depression_module)

    logging.getLogger('tlo.methods.demography').setLevel(logging.WARNING)

    return sim


def test_depression_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)



#TODO: Asif will probably recommned more tests to check about basic behaviours


if __name__ == '__main__':
    simulation = simulation()
    test_depression_simulation(simulation)


