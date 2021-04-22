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

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract a series for all runs:
people_in_rti_incidence = extract_results(results_folder,
                                          module="tlo.methods.rti",
                                          key="summary_1m",  # <-- the key used for the logging entry
                                          column="incidence of rti per 100,000",  # <-- the column in the dataframe
                                          index="date")  # <-- optional index
deaths_from_rti_incidence = extract_results(results_folder,
                                          module="tlo.methods.rti",
                                          key="summary_1m",  # <-- the key used for the logging entry
                                          column="incidence of rti death per 100,000",  # <-- the column in the dataframe
                                          index="date")
incidence_of_injuries = extract_results(results_folder,
                                          module="tlo.methods.rti",
                                          key="Inj_category_incidence",  # <-- the key used for the logging entry
                                          column="tot_inc_injuries",  # <-- the column in the dataframe
                                          index="date")
# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
mean_incidence = summarize(people_in_rti_incidence, only_mean=True).iloc[-1]
mean_incidence_of_death = summarize(deaths_from_rti_incidence, only_mean=True).iloc[-1]
mean_incidence_of_injuries = summarize(incidence_of_injuries, only_mean=True).iloc[-1]

grid = get_grid(params, res)

# 4) plot a bar chart showing the base rate of injury vs the incidence of injury:
plt.bar(np.arange(len(grid['RTI:base_rate_injrti'][0])), res, color='lightsteelblue')
xlabels = [str(parameter) for parameter in grid['RTI:base_rate_injrti'][0].tolist()]
plt.xticks(np.arange(len(grid['RTI:base_rate_injrti'][0])), xlabels)
plt.xlabel('RTI:base_rate_injrti')
plt.ylabel('injury incidence per 100,000')
plt.title('Base rate of injury vs incidence of injuries per 100,000')
plt.show()
