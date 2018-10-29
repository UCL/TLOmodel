
import pytest  # this is the library for testing
import matplotlib.pyplot as plt

from tlo import Date, DateOffset, Person, Simulation, Types
from tlo.test import hiv_mockitis_format
from tlo.methods import demography

# for desktop
path = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile.xlsx'  # Edit this path so it points to Demography.xlsx file

# for laptop
# path = '/Users/Tara/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile.xlsx'  # Edit this path so it points to Demography.xlsx file

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 10000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    hiv_module = hiv_mockitis_format.hiv()
    sim.register(core_module)
    sim.register(hiv_module)

    return sim


def test_hiv_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_hiv_simulation(simulation)


# Make a nice plot;
stats = simulation.modules['hiv'].store['Total_HIV']
time = simulation.modules['hiv'].store['Time']

# plt.plot(np.arange(0, len(stats)), stats)

plt.plot(time, stats)

plt.show()

