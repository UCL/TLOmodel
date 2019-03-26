import logging
import os

import pytest  # this is the library for testing
import matplotlib.pyplot as plt

from tlo import Date, Simulation
from tlo.methods import demography, lifestyle, hiv, hiv_hs, tb_hs_engagement, tb, \
    male_circumcision, hiv_behaviour_change

# for desktop
# path_dem = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile_Complete.xlsx'
# path_tb = 'Q:/Thanzi la Onse/TB/Method_TB.xlsx'

# York
# path_hiv = 'Z:/Thanzi la Onse/HIV/Method_HIV.xlsx'
# path_dem = 'P:/Documents/TLO/Demography_WorkingFile_Complete.xlsx'  # update for new demog file
# path_hs = 'Z:/Thanzi la Onse/HIV/Method_ART.xlsx'
# path_tb = 'Z:/Thanzi la Onse/TB/Method_TB.xlsx'

# for laptop
path_dem = '/Users/Tara/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile_Complete.xlsx'
# path_hs = '/Users/Tara/Documents/TLO/Method_ART.xlsx'
# path_hiv = '/Users/Tara/Documents/TLO/Method_HIV.xlsx'
# path_tb = '/Users/Tara/Documents/TLO/Method_TB.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2013, 2, 1)
popsize = 10000

params = [0.2, 0.1, 0.05, 0.4, 0.5, 0.05]  # sample params for runs


@pytest.fixture(scope='module')
def simulation():
    resourceFile = os.path.join(os.path.dirname(__file__), 'resources')

    sim = Simulation(start_date=start_date)

    #  call modules
    core_module = demography.Demography(workbook_path=path_dem)
    lifestyle_module = lifestyle.Lifestyle()
    hiv_module = hiv.hiv(resourcefilepath=resourceFile, par_est=params[0])
    tb_module = tb.tb_baseline(resourcefilepath=resourceFile)

    hs_module = hiv_hs.health_system(resourcefilepath=resourceFile, par_est1=params[1], par_est2=params[2],
                                     par_est3=params[3], par_est4=params[4])

    circumcision_module = male_circumcision.male_circumcision(resourcefilepath=resourceFile, par_est5=params[5])
    behavioural_module = hiv_behaviour_change.BehaviourChange()
    hs_tb_module = tb_hs_engagement.health_system_tb()

    #  register modules
    sim.register(core_module)
    sim.register(lifestyle_module)
    sim.register(hiv_module)
    sim.register(hs_module)
    sim.register(circumcision_module)
    sim.register(behavioural_module)
    sim.register(tb_module)
    sim.register(hs_tb_module)

    logging.getLogger('tlo.methods.demography').setLevel(logging.WARNING)  # stops info logging outputs
    logging.getLogger('tlo.methods.lifestyle').setLevel(logging.WARNING)

    return sim


def test_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_simulation(simulation)

# # Make a nice plot
# hiv_output = simulation.modules['hiv'].store['Total_HIV']
# time = simulation.modules['hiv'].store['Time']
# hiv_deaths = simulation.modules['hiv'].store['HIV_scheduled_deaths']
#
# hiv_prev_adult = simulation.modules['hiv'].store['hiv_prev_adult']
# hiv_prev_child = simulation.modules['hiv'].store['hiv_prev_child']
#
# number_tested = simulation.modules['health_system'].store['Number_tested_adult']
# number_treated = simulation.modules['health_system'].store['Number_treated_adult']
# testing_dates = simulation.modules['health_system'].store['Time']
#
# time_circum = simulation.modules['male_circumcision'].store['Time']
# prop_circum = simulation.modules['male_circumcision'].store['proportion_circumcised']
#
# time_behav = simulation.modules['BehaviourChange'].store['Time']
# prop_counselled = simulation.modules['BehaviourChange'].store['Proportion_hiv_counselled']
#
# active_tb = simulation.modules['tb_baseline'].store['Total_active_tb']
# active_tb_mdr = simulation.modules['tb_baseline'].store['Total_active_tb_mdr']
# coinfected = simulation.modules['tb_baseline'].store['Total_co-infected']
# tb_deaths = simulation.modules['tb_baseline'].store['TB_deaths']
# time_tb_death = simulation.modules['tb_baseline'].store['Time_death_TB']
# time2 = simulation.modules['tb_baseline'].store['Time']
#
# time_test_tb = simulation.modules['health_system_tb'].store['Time']
# tb_tests = simulation.modules['health_system_tb'].store['Number_tested_tb']
#
#
# plt.figure(1)
#
# # hiv cases
# ax = plt.subplot(321)  # numrows, numcols, fignum
# plt.plot(time, hiv_prev_adult)
# plt.plot(time, hiv_prev_child)
# plt.legend(['hiv prev adult', 'hiv_prev_child'], loc='upper right')
# ax.set_xticklabels([])
# plt.ylabel('Prevalence')
#
# # # hiv deaths
# # ax = plt.subplot(322)  # numrows, numcols, fignum
# # plt.plot(time, hiv_deaths)
# # plt.legend(['scheduled AIDS deaths'], loc='upper right')
# # ax.set_xticklabels([])
# # plt.ylabel('Number of death')
#
# # tb cases
# ax = plt.subplot(322)
# plt.plot(time2, active_tb)
# plt.ylim(bottom=0)
# plt.legend(['TB'], loc='upper right')
# ax.set_xticklabels([])
# plt.ylabel('Number of cases')
#
# # testing hiv/tb
# plt.subplot(323)
# plt.plot(testing_dates, number_tested)
# plt.plot(time_test_tb, tb_tests)
# plt.ylim(bottom=0)
# plt.legend(['HIV testing', 'TB testing'], loc='upper right')
# plt.xticks(rotation=45)
# plt.ylabel('Number of tests')
#
# # treatment hiv/tb
# plt.subplot(324)
# plt.plot(testing_dates, number_treated)
# plt.ylim(bottom=0)
# plt.legend(['on ART'], loc='upper right')
# plt.xticks(rotation=45)
# plt.ylabel('Number on treatment')
#
# # counselling / circumcised
# plt.subplot(325)
# plt.plot(time_circum, prop_circum)
# plt.plot(time_behav, prop_counselled)
# plt.ylim(bottom=0)
# plt.legend(['circumcised', 'counselled'], loc='upper right')
# plt.xticks(rotation=45)
# plt.ylabel('Proportion')
#
# # mdr tb cases
# ax = plt.subplot(326)
# plt.plot(time2, active_tb_mdr)
# plt.ylim(bottom=0)
# plt.legend(['MDR-TB'], loc='upper right')
# ax.set_xticklabels([])
# plt.ylabel('Number of cases')
#
# plt.show()

# print(simulation.modules['hiv'].store_DeathsLog['DeathEvent_Time'])
