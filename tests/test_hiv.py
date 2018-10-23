
import pytest  # this is the library for testing

from tlo import Date, DateOffset, Person, Simulation, Types
from tlo.test import hiv_mockitis_format
from tlo.methods import demography

path = '/Users/Tara/Desktop/TLO/Demography.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2014, 1, 1)
end_date = Date(2016, 1, 1)
popsize = 100000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    hiv_module = hiv_mockitis_format.hiv_mock()
    sim.register(core_module)
    sim.register(hiv_module)

    return sim


def test_tb_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_tb_simulation(simulation)



