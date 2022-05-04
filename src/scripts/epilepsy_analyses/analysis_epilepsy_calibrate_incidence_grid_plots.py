"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_grid,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('analysis_epilepsy_calibrate_incidence_grid.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract a series for all runs:
inc = extract_results(results_folder,
                            module="tlo.methods.epilepsy",
                            key="inc_epilepsy",  # <-- the key used for the logging entry
                            column="incidence_epilepsy",  # <-- the column in the dataframe
                            index="date")
inc_death = extract_results(results_folder,
                            module="tlo.methods.epilepsy",
                            key="epilepsy_logging",  # <-- the key used for the logging entry
                            column="epi_death_rate",  # <-- the column in the dataframe
                            index="date")  # <-- optional index

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
inc_summary = summarize(inc, only_mean=True).mean()
inc_summary.name = 'z'
inc_death_summary = summarize(inc_death, only_mean=True).mean()
inc_death_summary.name = 'z'
# 4) Get the summary statistics from the GBD data
gbd_data = pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/gbddata/epilepsy/IHME-GBD_2019_DATA-4a82f00e-1.csv")
# get incidence estimates
gbd_inc_data = gbd_data.loc[gbd_data['measure'] == 'Incidence']
# mean incidence of epilepsy
mean_inc_gbd = gbd_inc_data.val.mean()
mean_inc_upper_gbd = gbd_inc_data.upper.mean()
mean_inc_lower_gbd = gbd_inc_data.lower.mean()
# get death estimate
gbd_inc_death_data = gbd_data.loc[gbd_data['measure'] == 'Deaths']
mean_inc_death_gbd = gbd_inc_death_data.val.mean()
mean_inc_death_upper_gbd = gbd_inc_death_data.upper.mean()
mean_inc_death_lower_gbd = gbd_inc_death_data.lower.mean()

# 5) Create a heatmap:

inc_grid = get_grid(params, inc_summary)
fig, ax = plt.subplots()
c = ax.pcolormesh(
    inc_grid['Epilepsy:base_3m_prob_epilepsy'],
    inc_grid['Epilepsy:base_prob_3m_epi_death'],
    inc_grid['z'],
    shading='nearest'
)
ax.set_title('Heat Map')
plt.xlabel('Epilepsy:base_3m_prob_epilepsy')
plt.ylabel('Epilepsy:base_prob_3m_epi_death')
fig.colorbar(c, ax=ax)
plt.show()
inc_death_grid = get_grid(params, inc_death_summary)
fig, ax = plt.subplots()
c = ax.pcolormesh(
    inc_death_grid['Epilepsy:base_3m_prob_epilepsy'],
    inc_death_grid['Epilepsy:base_prob_3m_epi_death'],
    inc_death_grid['z'],
    shading='nearest'
)
ax.set_title('Heat Map')
plt.xlabel('Epilepsy:base_3m_prob_epilepsy')
plt.ylabel('Epilepsy:base_prob_3m_epi_death')
fig.colorbar(c, ax=ax)
plt.show()
index_of_best_fit_inc = []
for r_idx, results in enumerate(inc_grid['z']):
    closest_est_found = min(results, key=lambda x: abs(x - mean_inc_gbd))
    best_fit_idx = np.where(results == closest_est_found)[0][0]
    index_of_best_fit_inc.append([r_idx, best_fit_idx, closest_est_found])

index_of_best_fit_inc_death = []
for r_idx, results in enumerate(inc_death_grid['z']):
    closest_est_found = min(results, key=lambda x: abs(x - mean_inc_death_gbd))
    best_fit_idx = np.where(results == closest_est_found)[0][0]
    index_of_best_fit_inc_death.append([r_idx, best_fit_idx, closest_est_found])

inc_grid_values = [i for sublist in inc_grid['z'] for i in sublist]
np.median(inc_grid_values)
