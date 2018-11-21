
import pytest  # this is the library for testing
import matplotlib.pyplot as plt
import numpy as np

from tlo import Date, DateOffset, Person, Simulation, Types
from tlo.test import hiv_infection, tb
from tlo.methods import demography

# for desktop
# path = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Old versions/Demography_WorkingFile.xlsx'  # Edit this path so it points to Demography.xlsx file
# path = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile_Complete.xlsx'  # Edit this path so it points to Demography.xlsx file
path = 'P:/Documents/TLO/Demography_WorkingFile.xlsx'  # York

art_path = 'P:/Documents/TLO/Method_ART.xlsx'  # York

# for laptop
# path = '/Users/Tara/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile.xlsx'  # Edit this path so it points to Demography.xlsx file

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 50000


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


# add plots for infections on birth and deaths when done

# Make a nice plot
hiv_output = simulation.modules['hiv'].store['Total_HIV']
time = simulation.modules['hiv'].store['Time']
hiv_deaths = simulation.modules['hiv'].store['HIV_deaths']

active_tb = simulation.modules['tb_baseline'].store['Total_active_tb']
coinfected = simulation.modules['tb_baseline'].store['Total_co-infected']
tb_deaths = simulation.modules['tb_baseline'].store['TB_deaths']
time_tb_death = simulation.modules['tb_baseline'].store['Time_death_TB']
time2 = simulation.modules['tb_baseline'].store['Time']


plt.figure(1)
plt.subplot(211)  # numrows, numcols, fignum
plt.plot(time, hiv_output)
plt.plot(time2, active_tb)
plt.plot(time2, coinfected)
plt.legend(['HIV', 'TB', 'HIV + TB'], loc='upper left')
plt.xticks(rotation=45)
plt.ylabel('Number of cases')

plt.subplot(212)
plt.plot(time, hiv_deaths)
plt.plot(time_tb_death, tb_deaths)
plt.ylim(bottom=0)
plt.legend(['HIV deaths', 'TB deaths'], loc='upper left')
plt.xticks(rotation=45)
plt.ylabel('Number of deaths')

plt.show()


