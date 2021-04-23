"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_grid,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('rti_grid_search_parameterisation.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
# get the time the simulation ran from some of the logging output
sim_run_time = log['tlo.methods.healthsystem']['bed_tracker_general_bed']['item_2'].iloc[-1] - \
               log['tlo.methods.healthsystem']['bed_tracker_general_bed']['item_2'].iloc[0]
sim_run_time_years = int(sim_run_time.days / 365)
pop_size = log['tlo.methods.demography']['population']['total'].iloc[0]

# 2) Extract a series for all runs:
people_in_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                          column="incidence of rti per 100,000", index="date")
deaths_from_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                            column="incidence of rti death per 100,000", index="date")
# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
incidence_results = summarize(people_in_rti_incidence, only_mean=True).mean(axis=0)
incidence_results.name = 'z'
death_incidence = summarize(deaths_from_rti_incidence, only_mean=True).mean(axis=0)
death_incidence.name = 'z'

# 4) Create a heatmap for incidence of RTI:
filtered_params = params.loc[params['module_param'] != 'RTI:number_of_injured_body_regions_distribution']
grid = get_grid(filtered_params, incidence_results)
fig, ax = plt.subplots()
c = ax.pcolormesh(
    grid['RTI:base_rate_injrti'],
    grid['RTI:imm_death_proportion_rti'],
    grid['z'],
    shading='nearest'
)
plt.xlabel('RTI:base_rate_injrti')
plt.ylabel('RTI:imm_death_proportion_rti')
plt.title(f"RTI incidence produced by the model when using single injuries only \n"
          f"years ran: {sim_run_time_years}, population size: {pop_size}, runs per scenario: {info['runs_per_draw']}")
fig.colorbar(c, ax=ax)
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/incidence_and_death_calibration/Incidence",
    bbox_inches='tight')

grid = get_grid(filtered_params, death_incidence)
fig, ax = plt.subplots()
c = ax.pcolormesh(
    grid['RTI:base_rate_injrti'],
    grid['RTI:imm_death_proportion_rti'],
    grid['z'],
    shading='nearest'
)
plt.title(f"RTI death incidence produced by the model when using single injuries only \n"
          f"years ran: {sim_run_time_years}, population size: {pop_size}, runs per scenario: {info['runs_per_draw']}")
plt.xlabel('RTI:base_rate_injrti')
plt.ylabel('RTI:imm_death_proportion_rti')
fig.colorbar(c, ax=ax)
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/incidence_and_death_calibration/IncidenceDeath",
    bbox_inches='tight')
