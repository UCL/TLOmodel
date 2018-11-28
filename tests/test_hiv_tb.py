
import pytest  # this is the library for testing
import matplotlib.pyplot as plt
import numpy as np

from tlo import Date, DateOffset, Person, Simulation, Types
from tlo.test import hiv_infection, tb, health_system, antiretroviral_therapy, health_system_tb
from tlo.methods import demography

# for desktop
# path = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Old versions/Demography_WorkingFile.xlsx'
# path = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile_Complete.xlsx'
# path = 'P:/Documents/TLO/Demography_WorkingFile.xlsx'  # York

# path_hs = 'P:/Documents/TLO/Method_ART.xlsx'  # York

# for laptop
path = '/Users/Tara/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Old versions/Demography_WorkingFile.xlsx'
path_hs = '/Users/Tara/Documents/TLO/Method_ART.xlsx'  # York

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 50000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    hiv_module = hiv_infection.hiv()
    tb_module = tb.tb_baseline()
    hs_module = health_system.health_system(workbook_path=path_hs)
    art_module = antiretroviral_therapy.art(workbook_path=path_hs)
    hs_tb_module = health_system_tb.health_system_tb()

    sim.register(core_module)
    sim.register(hiv_module)
    sim.register(tb_module)
    sim.register(hs_module)
    sim.register(art_module)
    sim.register(hs_tb_module)

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

number_tested = simulation.modules['health_system'].store['Number_tested']
number_treated = simulation.modules['health_system'].store['Number_treated']
testing_dates = simulation.modules['health_system'].store['Time']

deaths_art = simulation.modules['art'].store['Number_dead_art']
time_deaths_art = simulation.modules['art'].store['Time']

active_tb = simulation.modules['tb_baseline'].store['Total_active_tb']
coinfected = simulation.modules['tb_baseline'].store['Total_co-infected']
tb_deaths = simulation.modules['tb_baseline'].store['TB_deaths']
time_tb_death = simulation.modules['tb_baseline'].store['Time_death_TB']
time2 = simulation.modules['tb_baseline'].store['Time']

time_test_tb = simulation.modules['health_system_tb'].store['Time']
tb_tests = simulation.modules['health_system_tb'].store['Number_tested_tb']


plt.figure(1)
ax = plt.subplot(221)  # numrows, numcols, fignum
plt.plot(time, hiv_output)
plt.plot(time, hiv_deaths)
plt.plot(time_deaths_art, deaths_art)
plt.legend(['HIV', 'AIDS deaths', 'AIDS deaths on ART'], loc='upper right')
ax.set_xticklabels([])
plt.ylabel('Number of cases')

ax = plt.subplot(222)
plt.plot(time2, active_tb)
plt.plot(time_tb_death, tb_deaths)
plt.ylim(bottom=0)
plt.legend(['TB', 'TB deaths'], loc='upper right')
ax.set_xticklabels([])
plt.ylabel('Number of cases')

plt.subplot(223)
plt.plot(testing_dates, number_tested)
plt.plot(time_test_tb, tb_tests)
plt.ylim(bottom=0)
plt.legend(['HIV testing', 'TB testing'], loc='upper right')
plt.xticks(rotation=45)
plt.ylabel('Number of tests')

plt.subplot(224)
plt.plot(testing_dates, number_treated)
plt.ylim(bottom=0)
plt.legend(['on ART'], loc='upper right')
plt.xticks(rotation=45)
plt.ylabel('Number treated')

plt.show()


