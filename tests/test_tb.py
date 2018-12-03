import pytest  # this is the library for testing

from tlo import Date, Simulation
from tlo.methods import demography, tb

# for desktop
path = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile.xlsx'  # Edit this path so it points to Demography.xlsx file

# for laptop
# path = '/Users/Tara/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile.xlsx'  # Edit this path so it points to Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 100000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    tb_module = tb.tb_baseline()
    sim.register(core_module)
    sim.register(tb_module)

    return sim


def test_tb_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_tb_simulation(simulation)
