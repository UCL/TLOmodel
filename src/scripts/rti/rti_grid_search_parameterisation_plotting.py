"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from tlo.analysis.utils import (
    get_grid,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

# create a function that extracts results in the same way as the utils function, but allows failed
# runs to pass


def rti_extract_params(results_folder: Path) -> pd.DataFrame:
    """Utility function to get overridden parameters from scenario runs

    Returns dateframe summarizing parameters that change across the draws. It produces a dataframe with index of draw
    and columns of each parameters that is specified to be varied in the batch. NB. This does the extraction from run 0
    in each draw, under the assumption that the over-written parameters are the same in each run.
    """

    # Get the paths for the draws
    draws = [f for f in os.scandir(results_folder) if f.is_dir()]

    list_of_param_changes = list()

    for d in draws:
        p = load_pickled_dataframes(results_folder, d.name, 0, name="tlo.scenario")
        try:
            p = p["tlo.scenario"]["override_parameter"]
        except KeyError:
            json_file_path = "outputs/rmjlra2@ucl.ac.uk/rti_grid_search_parameterisation-2021-04-26T090157Z/" \
                             "rti_grid_search_parameterisation_draws.json"
            with open(json_file_path, 'r') as myfile:
                data = myfile.read()
            data = json.loads(data)
            data = data['draws'][int(d.name)]
            df_format = list_of_param_changes[-1].copy()
            for parameter in data['parameters']['RTI'].keys():
                if parameter == 'number_of_injured_body_regions_distribution':
                    pass
                else:
                    equivalent_df_row = 'RTI:' + parameter
                    df_format.loc[df_format['module_param'] == equivalent_df_row, 'new_value'] = \
                        data['parameters']['RTI'][parameter]
            p = df_format
            p.index = [int(d.name)] * len(p.index)
        if len(p.columns) > 2:
            p['module_param'] = p['module'] + ':' + p['name']
            p.index = [int(d.name)] * len(p.index)

        list_of_param_changes.append(p[['module_param', 'new_value']])

    params = pd.concat(list_of_param_changes)
    params.index.name = 'draw'
    params = params.rename(columns={'new_value': 'value'})
    params = params.sort_index()

    return params


def rti_extract_results(results_folder: Path, module: str, key: str, column: str, index: str = None) -> pd.DataFrame:
    """Utility function to unpack results

    Produces a dataframe that summaries one series from the log, with column multi-index for the draw/run. If an 'index'
    component of the log_element is provided, the dataframe uses that index (but note that this will only work if the
    index is the same in each run).
    """

    results_index = None
    if index is not None:
        # extract the index from the first log, and use this ensure that all other are exactly the same.
        filename = f"{module}.pickle"
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw=0, run=0, name=filename)[module][key]
        results_index = df[index]

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    results = pd.DataFrame(columns=pd.MultiIndex.from_product(
        [range(info['number_of_draws']), range(info['runs_per_draw'])],
        names=["draw", "run"]
    ))

    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):
            try:
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                results[draw, run] = df[column]

                if index is not None:
                    idx = df[index]
                    assert idx.equals(results_index), "Indexes are not the same between runs"

            except ValueError:
                results[draw, run] = np.nan
            except KeyError:
                results[draw, run] = np.nan

    # if 'index' is provided, set this to be the index of the results
    if index is not None:
        results.index = results_index

    return results


outputspath = Path('./outputs/')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('rti_grid_search_parameterisation.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = rti_extract_params(results_folder)
# get the time the simulation ran from some of the logging output
sim_run_time = log['tlo.methods.healthsystem']['bed_tracker_general_bed']['item_2'].iloc[-1] - \
               log['tlo.methods.healthsystem']['bed_tracker_general_bed']['item_2'].iloc[0]
sim_run_time_years = int(sim_run_time.days / 365)
pop_size = log['tlo.methods.demography']['population']['total'].iloc[0]

# 2) Extract a series for all runs:
people_in_rti_incidence = rti_extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                              column="incidence of rti per 100,000", index="date")
deaths_from_rti_incidence = rti_extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                                column="incidence of rti death per 100,000", index="date")
# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
incidence_results = summarize(people_in_rti_incidence, only_mean=True).mean(axis=0)
incidence_results.name = 'z'
death_incidence = summarize(deaths_from_rti_incidence, only_mean=True).mean(axis=0)
death_incidence.name = 'z'

# 4) Create a heatmap for incidence of RTI:
filtered_params = params.loc[params['module_param'] != 'RTI:number_of_injured_body_regions_distribution']
inc_grid = get_grid(filtered_params, incidence_results)

ax = plt.figure().add_subplot(projection='3d')
surf = ax.plot_surface(inc_grid['RTI:base_rate_injrti'], inc_grid['RTI:imm_death_proportion_rti'], inc_grid['z'],
                       linewidth=0)
ax.set_xlabel('RTI:base_rate_injrti')
ax.set_ylabel('RTI:imm_death_proportion_rti')
ax.set_zlabel('Incidence of RTI')
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/incidence_and_death_calibration/"
    "Incidence_unfiltered_surf",
    bbox_inches='tight')
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(inc_grid['RTI:base_rate_injrti'], inc_grid['RTI:imm_death_proportion_rti'], inc_grid['z'], c='r',
           marker='o')

ax.set_xlabel('RTI:base_rate_injrti')
ax.set_ylabel('RTI:imm_death_proportion_rti')
ax.set_zlabel('Incidence of RTI')
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/incidence_and_death_calibration/"
    "Incidence_unfiltered_scatter",
    bbox_inches='tight')
plt.clf()

fig, ax = plt.subplots()
c = ax.pcolormesh(
    inc_grid['RTI:base_rate_injrti'],
    inc_grid['RTI:imm_death_proportion_rti'],
    inc_grid['z'],
    cmap='Greys',
    shading='flat'
)
plt.xlabel('RTI:base_rate_injrti')
plt.ylabel('RTI:imm_death_proportion_rti')
plt.title(f"RTI incidence produced by the model when using single injuries only \n"
          f"years ran: {sim_run_time_years}, population size: {pop_size}, runs per scenario: {info['runs_per_draw']}")
fig.colorbar(c, ax=ax)
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/incidence_and_death_calibration/"
    "Incidence_unfiltered",
    bbox_inches='tight')
plt.clf()
# get values in the range of the GBD estimates, first filter out values below lower boundary
inc_grid['z'][inc_grid['z'] < 809.234] = 0
# filter out results above upper boundary
inc_grid['z'][inc_grid['z'] > 1130.626] = 0
fig, ax = plt.subplots()
c = ax.pcolormesh(
    inc_grid['RTI:base_rate_injrti'],
    inc_grid['RTI:imm_death_proportion_rti'],
    inc_grid['z'],
    cmap='Greys',
    shading='flat'
)
plt.xlabel('RTI:base_rate_injrti')
plt.ylabel('RTI:imm_death_proportion_rti')
plt.title(f"RTI incidence produced by the model when using single injuries only \n"
          f"years ran: {sim_run_time_years}, population size: {pop_size}, runs per scenario: {info['runs_per_draw']}")
fig.colorbar(c, ax=ax)
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/incidence_and_death_calibration/"
    "Incidence_filtered",
    bbox_inches='tight')
plt.clf()
# Incidence of death
death_inc_grid = get_grid(filtered_params, death_incidence)

ax = plt.figure().add_subplot(projection='3d')
surf = ax.plot_surface(death_inc_grid['RTI:base_rate_injrti'], death_inc_grid['RTI:imm_death_proportion_rti'],
                       death_inc_grid['z'], linewidth=0)
ax.set_xlabel('RTI:base_rate_injrti')
ax.set_ylabel('RTI:imm_death_proportion_rti')
ax.set_zlabel('Incidence of RTI death')
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/incidence_and_death_calibration/"
    "Death_incidence_unfiltered_surf",
    bbox_inches='tight')
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(death_inc_grid['RTI:base_rate_injrti'], death_inc_grid['RTI:imm_death_proportion_rti'], death_inc_grid['z'],
           c='r', marker='o')

ax.set_xlabel('RTI:base_rate_injrti')
ax.set_ylabel('RTI:imm_death_proportion_rti')
ax.set_zlabel('Incidence of RTI death')
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/incidence_and_death_calibration/"
    "Death_incidence_unfiltered_scatter",
    bbox_inches='tight')
plt.clf()

fig, ax = plt.subplots()
c = ax.pcolormesh(
    death_inc_grid['RTI:base_rate_injrti'],
    death_inc_grid['RTI:imm_death_proportion_rti'],
    death_inc_grid['z'],
    cmap='Greys',
    shading='nearest'
)
plt.title(f"RTI death incidence produced by the model when using single injuries only \n"
          f"years ran: {sim_run_time_years}, population size: {pop_size}, runs per scenario: {info['runs_per_draw']}")
plt.xlabel('RTI:base_rate_injrti')
plt.ylabel('RTI:imm_death_proportion_rti')
fig.colorbar(c, ax=ax)
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/incidence_and_death_calibration/"
    "IncidenceDeath_unfiltered",
    bbox_inches='tight')
plt.clf()
# get values in the range of the GBD estimates, first filter out values below lower boundary
death_inc_grid['z'][death_inc_grid['z'] < 9.606] = 0
# filter out results above upper boundary
death_inc_grid['z'][death_inc_grid['z'] > 15.13] = 0
fig, ax = plt.subplots()
c = ax.pcolormesh(
    death_inc_grid['RTI:base_rate_injrti'],
    death_inc_grid['RTI:imm_death_proportion_rti'],
    death_inc_grid['z'],
    cmap='Greys',
    shading='nearest'
)
plt.title(f"RTI death incidence produced by the model when using single injuries only \n"
          f"years ran: {sim_run_time_years}, population size: {pop_size}, runs per scenario: {info['runs_per_draw']}")
plt.xlabel('RTI:base_rate_injrti')
plt.ylabel('RTI:imm_death_proportion_rti')
fig.colorbar(c, ax=ax)
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Scenarios/incidence_and_death_calibration/"
    "IncidenceDeath_filtered",
    bbox_inches='tight')
# Print out the results
inc_grid
inc_results = np.where(inc_grid['z'] > 0)
inc_death_results = np.where(death_inc_grid['z'] > 0)
parameter_combinations = [np.intersect1d(inc_results[0], inc_death_results[0]),
                          np.intersect1d(inc_results[1], inc_death_results[1])]
if len(parameter_combinations[0]) > 0:
    accepted_params = {'base_rate_injrti': inc_grid['RTI:base_rate_injrti'][tuple(parameter_combinations)][0],
                       'imm_death_proportion_rti': inc_grid['RTI:imm_death_proportion_rti']
                       [tuple(parameter_combinations)][0]}
    resulting_incidences = {'incidence of rti': inc_grid['z'][tuple(parameter_combinations)][0],
                            'incidence of rti death': death_inc_grid['z'][tuple(parameter_combinations)][0]}
    print(accepted_params)
    print(resulting_incidences)
else:
    print('No parameter combinations found.')
