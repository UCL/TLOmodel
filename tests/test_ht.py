import pytest

from tlo import Simulation, Date
from tlo.methods import demography

path = '/Users/mc1405/Desktop/Demography.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2060, 1, 1)
popsize = 1000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    sim.register(core_module)
    return sim


def test_HT_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_HT_simulation(simulation)
