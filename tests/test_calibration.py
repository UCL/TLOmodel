import logging

import pytest  # this is the library for testing
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

from tlo import Date, Simulation
from tlo.methods import demography, antiretroviral_therapy, hiv_infection, health_system_hiv, health_system_tb, tb, \
    male_circumcision, hiv_behaviour_change

# for desktop
# path_dem = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Old versions/Demography_WorkingFile.xlsx'
# path_dem = '/Users/tmangal/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile_Complete.xlsx'
# path_tb = 'Q:/Thanzi la Onse/TB/Method_TB.xlsx'

# York
path_hiv = 'P:/Documents/TLO/Method_HIV.xlsx'
path_dem = 'P:/Documents/TLO/Demography_WorkingFile_Complete.xlsx' # update for new demog file
path_hs = 'P:/Documents/TLO/Method_ART.xlsx'
path_tb = 'P:/Documents/TLO/Method_TB.xlsx'

# for laptop
# path_dem = '/Users/Tara/Dropbox/Thanzi la Onse/05 - Resources/Demographic data/Demography_WorkingFile_Complete.xlsx'
# path_hs = '/Users/Tara/Documents/TLO/Method_ART.xlsx'
# path_hiv = '/Users/Tara/Documents/TLO/Method_HIV.xlsx'
# path_tb = '/Users/Tara/Documents/TLO/Method_TB.xlsx'

# read in data files for calibration
# number new infections
inc_data = pd.read_excel(path_hiv, sheet_name='incidence_calibration', header=0)
inc_data = inc_data[inc_data.year >= 2011]
# print(inc_data)

# new tests
test_data = pd.read_excel(path_hs, sheet_name='testing_calibration', header=0)

# number starting treatment
treat_data = pd.read_excel(path_hs, sheet_name='art_calibration', header=0)



start_date = Date(2010, 1, 1)
end_date = Date(2018, 2, 1)
popsize = 10000


def test_function(params):

    @pytest.fixture
    def simulation():
        sim = Simulation(start_date=start_date)

        #  call modules
        core_module = demography.Demography(workbook_path=path_dem)
        hiv_module = hiv_infection.hiv(workbook_path=path_hiv, par_est=params[0])
        art_module = antiretroviral_therapy.art(workbook_path=path_hs)
        hs_module = health_system_hiv.health_system(workbook_path=path_hs, par_est1=params[1], par_est2=params[2],
                                                    par_est3=params[3], par_est4=params[4])
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

        logging.getLogger('tlo.methods.demography').setLevel(logging.WARNING)

        return sim

    def test_simulation(simulation):
        simulation.make_initial_population(n=popsize)
        simulation.simulate(end_date=end_date)

    if __name__ == '__main__':
        simulation = simulation()
        test_simulation(simulation)

    # to calibrate: number infections (adult), number testing, number starting treatment
    print('new infections', simulation.modules['hiv'].store['HIV_new_infections_adult'])
    print('new tests', simulation.modules['health_system'].store['Number_tested_adult'])
    print('new treatment_adult', simulation.modules['health_system'].store['Number_treated_adult'])

    new_inf_ad = simulation.modules['hiv'].store['HIV_new_infections_adult']
    new_inf_child = simulation.modules['hiv'].store['HIV_new_infections_child']
    new_test_ad = simulation.modules['health_system'].store['Number_tested_adult']
    new_test_child = simulation.modules['health_system'].store['Number_tested_child']
    new_treatment_adult = simulation.modules['health_system'].store['Number_treated_adult']
    new_treatment_child = simulation.modules['health_system'].store['Number_treated_child']

    # calibrate using least squares
    # check years are matching - 2011-2018
    ss_inf_ad = sum((inc_data.new_cases_adults - new_inf_ad) ^ 2)
    ss_inf_child = sum((inc_data.new_cases_children - new_inf_child) ^ 2)

    ss_test_ad = sum((test_data.adult - new_test_ad) ^ 2)
    ss_test_child = sum((test_data.children - new_test_child) ^ 2)

    ss_treat_ad = sum((treat_data.adults - new_treatment_adult) ^ 2)
    ss_treat_child = sum((treat_data.children - new_treatment_child) ^ 2)

    total_ss = ss_inf_ad + ss_inf_child + ss_test_ad + ss_test_child + ss_treat_ad + ss_treat_child
    print('total_ss', total_ss)
    return total_ss


# test run with starting values
params = [0.5, 0.5, 0.5, 0.5, 0.5]
test_function(params)

# calibration
# res = optimize.minimize(test_function, 0.8, method="L-BFGS-B", bounds=[(0.3, 2)])
# print(res)


# add plots for infections on birth and deaths when done

# Make a nice plot
# hiv_output = simulation.modules['hiv'].store['Total_HIV']
# time = simulation.modules['hiv'].store['Time']
# hiv_deaths = simulation.modules['hiv'].store['HIV_scheduled_deaths']
#
# number_tested = simulation.modules['health_system'].store['Number_tested']
# number_treated = simulation.modules['health_system'].store['Number_treated']
# testing_dates = simulation.modules['health_system'].store['Time']
#
# deaths_art = simulation.modules['art'].store['Number_dead_art']
# time_deaths_art = simulation.modules['art'].store['Time']
#
# time_circum = simulation.modules['male_circumcision'].store['Time']
# prop_circum = simulation.modules['male_circumcision'].store['proportion_circumcised']
#
# time_behav = simulation.modules['BehaviourChange'].store['Time']
# prop_counselled = simulation.modules['BehaviourChange'].store['Proportion_hiv_counselled']
#
# plt.figure(1)
#
# # hiv cases
# ax = plt.subplot(221)  # numrows, numcols, fignum
# plt.plot(time, hiv_output)
# plt.legend(['HIV'], loc='upper right')
# ax.set_xticklabels([])
# plt.ylabel('Number of cases')
#
# # hiv deaths
# ax = plt.subplot(222)  # numrows, numcols, fignum
# plt.plot(time, hiv_deaths)
# plt.plot(time_deaths_art, deaths_art)
# plt.legend(['AIDS deaths', 'AIDS deaths on ART'], loc='upper right')
# ax.set_xticklabels([])
# plt.ylabel('Number of death')
#
# # testing hiv
# plt.subplot(223)
# plt.plot(testing_dates, number_tested)
# plt.plot(testing_dates, number_treated)
# plt.ylim(bottom=0)
# plt.legend(['HIV testing', 'on ART'], loc='upper right')
# plt.xticks(rotation=45)
# plt.ylabel('Number of tests')
#
# # counselling / circumcised
# plt.subplot(224)
# plt.plot(time_circum, prop_circum)
# plt.plot(time_behav, prop_counselled)
# plt.ylim(bottom=0)
# plt.legend(['circumcised', 'counselled'], loc='upper right')
# plt.xticks(rotation=45)
# plt.ylabel('Proportion')
#
# plt.show()

# print(simulation.modules['hiv'].store_DeathsLog['DeathEvent_Time'])
