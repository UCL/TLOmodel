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

outputspath = Path("./outputs")

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs("mockitis_2D_grid.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract a series for all runs:
extracted = extract_results(
    results_folder,
    module="tlo.methods.mockitis",
    key="summary",  # <-- the key used for the logging entry
    column="PropInf",  # <-- the column in the dataframe
    index="date",
)  # <-- optional index

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extracted, only_mean=True).iloc[-1]
res.name = "z"

# 4) Create a heatmap:

grid = get_grid(params, res)
fig, ax = plt.subplots()
c = ax.pcolormesh(
    grid["Mockitis:p_cure"], grid["Mockitis:p_infection"], grid["z"], shading="nearest"
)
ax.set_title("Heat Map")
plt.xlabel("Mockitis:p_cure")
plt.ylabel("Mockitis:p_infection")
fig.colorbar(c, ax=ax)
plt.show()
