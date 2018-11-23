import pytest

from tlo import Simulation, Date
from tlo.methods import demography, hypertension

path = '/Users/mc1405/Dropbox/Projects - ongoing/Malawi Project/Model/Demography.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2060, 1, 1)
popsize = 1000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    hypertension_module = hypertension.HT()               # This will load method for hypertension
    sim.register(core_module)
    sim.register(hypertension_module)                     # This will register method for hypertension
    return sim


def test_hypertension_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_hypertension_simulation(simulation)
