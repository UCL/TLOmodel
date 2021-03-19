"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt

from scripts.dev.th_testing.th_utils import (
    extract_params,
    extract_results,
    get_folders,
    get_grid,
    get_info,
    getalog,
    summarize,
)

outputspath = Path('./outputs')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_folders('mockitis_2D_grid.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = getalog(results_folder)

# get basic information about the results
info = get_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Define the log-element to extract:
log_element = {
    "component": "tlo.methods.mockitis",  # <-- the dataframe that is output
    "series": "['summary'].PropInf",  # <-- series in the dateframe to be extracted
    "index": "['summary'].date",  # <-- (optional) index to use
}

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extract_results(results_folder, log_element), only_mean=True).iloc[-1]
res.name = 'z'

# 4) Create a heatmap:

grid = get_grid(params, res)
fig, ax = plt.subplots()
c = ax.pcolormesh(
    grid['Mockitis:p_cure'],
    grid['Mockitis:p_infection'],
    grid['z'],
    shading='nearest'
)
ax.set_title('Heat Map')
plt.xlabel('Mockitis:p_cure')
plt.ylabel('Mockitis:p_infection')
fig.colorbar(c, ax=ax)
plt.show()
