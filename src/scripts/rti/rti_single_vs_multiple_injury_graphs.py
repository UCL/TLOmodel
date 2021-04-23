"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_grid,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path('./outputs')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('rti_single_vs_mutliple_injury.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)
# get the time the simulation ran from some of the logging output
sim_run_time = log['tlo.methods.healthsystem']['bed_tracker_general_bed']['item_2'].iloc[-1] - \
               log['tlo.methods.healthsystem']['bed_tracker_general_bed']['item_2'].iloc[0]
sim_run_time_years = int(sim_run_time.days / 365)
pop_size = log['tlo.methods.demography']['population']['total'].iloc[0]
# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract a series for all runs:
people_in_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                          column="incidence of rti per 100,000", index="date")
deaths_from_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                            column="incidence of rti death per 100,000", index="date")
incidence_of_injuries = extract_results(results_folder, module="tlo.methods.rti", key="Inj_category_incidence",
                                        column="tot_inc_injuries", index="date")
YLD = extract_results(results_folder, module="tlo.methods.healthburden", key="dalys", column="YLD_RTI_rt_disability",
                      index="date")
dalys = YLD.copy()
yll_names = ['YLL_RTI_RTI_imm_death', 'YLL_RTI_RTI_death_without_med', 'YLL_RTI_RTI_death_with_med',
             'YLL_RTI_RTI_unavailable_med']
for death_type in yll_names:
    try:
        YLL = extract_results(results_folder, module="tlo.methods.healthburden", key="dalys", column=death_type,
                              index="date")
        dalys = dalys + YLL
    except:
        KeyError

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
mean_incidence_single = np.mean(summarize(people_in_rti_incidence, only_mean=True)[0])
mean_incidence_multiple = np.mean(summarize(people_in_rti_incidence, only_mean=True)[1])
mean_incidence_of_death_single = np.mean(summarize(deaths_from_rti_incidence, only_mean=True)[0])
mean_incidence_of_death_multiple = np.mean(summarize(deaths_from_rti_incidence, only_mean=True)[1])
mean_incidence_of_injuries_single = np.mean(summarize(incidence_of_injuries, only_mean=True)[0])
mean_incidence_of_injuries_multiple = np.mean(summarize(incidence_of_injuries, only_mean=True)[1])
mean_dalys_single = np.mean(summarize(dalys, only_mean=True)[0])
mean_dalys_multiple = np.mean(summarize(dalys, only_mean=True)[1])
single_injury_data = [mean_incidence_single, mean_incidence_of_death_single, mean_incidence_of_injuries_single]
multiple_injury_data = [mean_incidence_multiple, mean_incidence_of_death_multiple, mean_incidence_of_injuries_multiple]

n = np.arange(len(single_injury_data))
# 4) plot a bar chart showing the base rate of injury vs the incidence of injury:
plt.bar(n, single_injury_data, width=0.4, color='lightsteelblue', label='Single')
plt.bar(n+ 0.4, multiple_injury_data, width=0.4, color='lightsalmon',
        label='Multiple')

xlabels = ['Incidence of people \n with RTIs', 'Incidence of RTI death', 'Incidence of RTIs']
plt.xticks(n + 0.2, xlabels)
plt.ylabel('Incidence per 100,000 person years')
plt.title(f"Incidence of people having road traffic injuries, \nRTI death and incidence of injuries for simulations \n"
          f"ran with single and multiple injuries, years ran: {sim_run_time_years}, \n"
          f"population size: {pop_size}, runs per scenario: {info['runs_per_draw']}")
for i in range(len(single_injury_data)):
    plt.annotate(str(np.round(single_injury_data[i], 2)), xy=(n[i], single_injury_data[i]),
                 ha='center', va='bottom')
for i in range(len(multiple_injury_data)):
    plt.annotate(str(np.round(multiple_injury_data[i], 2)), xy=(n[i] + 0.4, multiple_injury_data[i]),
                 ha='center', va='bottom')

plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/SingleVsMultiple/incidence_output.png",
            bbox_inches='tight')
plt.clf()
plt.bar(np.arange(2), [mean_dalys_single, mean_dalys_multiple], color='lightsteelblue')
plt.xticks(np.arange(2), ['Single injuries', 'Muliple injuries'])
plt.ylabel('DALYS')
plt.title(f"DALYS produced by the model when ran with single \n"
          f"and multiple injuries, years ran: {sim_run_time_years}, \n"
          f"population size: {pop_size}, runs per scenario: {info['runs_per_draw']}")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/SingleVsMultiple/DALYs_output.png",
            bbox_inches='tight')
