
import pytest  # this is the library for testing
import matplotlib.pyplot as plt

from tlo import Date, Simulation
from tlo.methods import demography, antiretroviral_therapy, hiv_infection, health_system_hiv, health_system_tb, tb, \
    male_circumcision, hiv_behaviour_change



# for desktop
# path_dem = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Old versions/Demography_WorkingFile.xlsx'
# path_dem = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile_Complete.xlsx'
# path_tb = 'Q:/Thanzi la Onse/TB/Method_TB.xlsx'

# York
path_hiv = 'P:/Documents/TLO/Method_HIV.xlsx'
path_dem = 'P:/Documents/TLO/Demography_WorkingFile.xlsx'
path_hs = 'P:/Documents/TLO/Method_ART.xlsx'
path_tb = 'P:/Documents/TLO/Method_TB.xlsx'

# for laptop
# path_dem = '/Users/Tara/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Old versions/Demography_WorkingFile.xlsx'
# path_hs = '/Users/Tara/Documents/TLO/Method_ART.xlsx'
# path_hiv = '/Users/Tara/Documents/TLO/Method_HIV.xlsx'
# path_tb = '/Users/Tara/Documents/TLO/Method_TB.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 50000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)

    #  call modules
    core_module = demography.Demography(workbook_path=path_dem)
    hiv_module = hiv_infection.hiv(workbook_path=path_hiv)
    art_module = antiretroviral_therapy.art(workbook_path=path_hs)
    hs_module = health_system_hiv.health_system(workbook_path=path_hs)
    circumcision_module = male_circumcision.male_circumcision(workbook_path=path_hiv)
    behavioural_module = hiv_behaviour_change.BehaviourChange()
    tb_module = tb.tb_baseline(workbook_path=path_tb)
    hs_tb_module = health_system_tb.health_system_tb()

    #  register modules
    sim.register(core_module)
    sim.register(hiv_module)
    sim.register(art_module)
    sim.register(hs_module)
    sim.register(circumcision_module)
    sim.register(behavioural_module)
    sim.register(tb_module)
    sim.register(hs_tb_module)

    return sim


def test_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_simulation(simulation)


# add plots for infections on birth and deaths when done
#  TODO: use the store to calculate scheduled hiv deaths and deaths actually occurring
#  gap will be deaths averted due to ART

# Make a nice plot
hiv_output = simulation.modules['hiv'].store['Total_HIV']
time = simulation.modules['hiv'].store['Time']
hiv_deaths = simulation.modules['hiv'].store['HIV_scheduled_deaths']

number_tested = simulation.modules['health_system'].store['Number_tested']
number_treated = simulation.modules['health_system'].store['Number_treated']
testing_dates = simulation.modules['health_system'].store['Time']

deaths_art = simulation.modules['art'].store['Number_dead_art']
time_deaths_art = simulation.modules['art'].store['Time']

time_circum = simulation.modules['male_circumcision'].store['Time']
prop_circum = simulation.modules['male_circumcision'].store['proportion_circumcised']

time_behav = simulation.modules['BehaviourChange'].store['Time']
prop_counselled = simulation.modules['BehaviourChange'].store['Proportion_hiv_counselled']

active_tb = simulation.modules['tb_baseline'].store['Total_active_tb']
coinfected = simulation.modules['tb_baseline'].store['Total_co-infected']
tb_deaths = simulation.modules['tb_baseline'].store['TB_deaths']
time_tb_death = simulation.modules['tb_baseline'].store['Time_death_TB']
time2 = simulation.modules['tb_baseline'].store['Time']

time_test_tb = simulation.modules['health_system_tb'].store['Time']
tb_tests = simulation.modules['health_system_tb'].store['Number_tested_tb']


plt.figure(1)

# hiv cases
ax = plt.subplot(221)  # numrows, numcols, fignum
plt.plot(time, hiv_output)
plt.legend(['HIV'], loc='upper right')
ax.set_xticklabels([])
plt.ylabel('Number of cases')

# hiv deaths
ax = plt.subplot(222)  # numrows, numcols, fignum
plt.plot(time, hiv_deaths)
plt.plot(time_deaths_art, deaths_art)
plt.legend(['AIDS deaths', 'AIDS deaths on ART'], loc='upper right')
ax.set_xticklabels([])
plt.ylabel('Number of death')

# tb cases
# ax = plt.subplot(222)
# plt.plot(time2, active_tb)
# plt.plot(time_tb_death, tb_deaths)
# plt.ylim(bottom=0)
# plt.legend(['TB', 'TB deaths'], loc='upper right')
# ax.set_xticklabels([])
# plt.ylabel('Number of cases')

# testing hiv/tb
plt.subplot(223)
plt.plot(testing_dates, number_tested)
plt.plot(time_test_tb, tb_tests)
plt.plot(testing_dates, number_treated)
plt.ylim(bottom=0)
plt.legend(['HIV testing', 'TB testing', 'on ART'], loc='upper right')
plt.xticks(rotation=45)
plt.ylabel('Number of tests')

# counselling / circumcised
plt.subplot(224)
plt.plot(time_circum, prop_circum)
plt.plot(time_behav, prop_counselled)
plt.ylim(bottom=0)
plt.legend(['circumcised', 'counselled'], loc='upper right')
plt.xticks(rotation=45)
plt.ylabel('Proportion')

plt.show()

# print(simulation.modules['hiv'].store_DeathsLog['DeathEvent_Time'])
