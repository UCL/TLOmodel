
import pytest  # this is the library for testing
import matplotlib.pyplot as plt

from tlo import Date, DateOffset, Person, Simulation, Types
from tlo.test import hiv_infection, tb
from tlo.methods import demography

# for desktop
path = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Old versions/Demography_WorkingFile.xlsx'  # Edit this path so it points to Demography.xlsx file

# for laptop
# path = '/Users/Tara/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile.xlsx'  # Edit this path so it points to Demography.xlsx file

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    hiv_module = hiv_infection.hiv()
    tb_module = tb.tb_baseline()

    sim.register(core_module)
    sim.register(hiv_module)
    sim.register(tb_module)
    return sim


def test_hiv_tb_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_hiv_tb_simulation(simulation)



# Make a nice plot
hiv_output = simulation.modules['hiv'].store['Total_HIV']
time = simulation.modules['hiv'].store['Time']

tb_output = simulation.modules['tb_baseline'].store['Total_active_tb']
time2 = simulation.modules['tb_baseline'].store['Time']

plt.plot(time, hiv_output)
plt.plot(time2, tb_output)
plt.legend(['HIV', 'TB'], loc='upper left')
plt.xticks(rotation=45)
plt.ylabel('Number of cases')

plt.show()

