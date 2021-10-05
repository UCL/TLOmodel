"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputs' results_folder
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

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("baseline_scenario.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# ---------------------- EXTRACT HIV PREVALENCE ---------------------- #
# 2) Extract a specific logged output for all runs, e.g. prevalence:
extracted = extract_results(
    results_folder,
    module="tlo.methods.hiv",
    key="summary_inc_and_prev_for_adults_and_children_and_fsw",
    column="hiv_prev_adult_1549",
    index="date",
)

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extracted, only_mean=True).iloc[-1]
res.name = "z"

# 4) Create a heatmap:

grid = get_grid(params, res)
fig, ax = plt.subplots()
c = ax.pcolormesh(
    grid["Hiv:beta"], grid["Tb:transmission_rate"], grid["z"], shading="nearest"
)
ax.set_title("HIV prevalence 2019")
plt.xlabel("Hiv: transmission rate")
plt.ylabel("Tb: transmission rate")
fig.colorbar(c, ax=ax)
plt.show()

# ---------------------- EXTRACT TB PREVALENCE ---------------------- #
# Extract a specific logged output for all runs
extracted_tb = extract_results(
    results_folder,
    module="tlo.methods.tb",
    key="tb_prevalence",
    column="tbPrevActive",
    index="date",
)

# Get summary of the results for that log-element (only mean and the value at then of the simulation)
tb_prev = summarize(extracted_tb, only_mean=True).iloc[-1]
tb_prev.name = "z"

# 4) Create a heatmap:

grid = get_grid(params, res)
fig, ax = plt.subplots()
c = ax.pcolormesh(
    grid["Hiv:beta"], grid["Tb:transmission_rate"], grid["z"], shading="nearest"
)
ax.set_title("TB prevalence 2019")
plt.xlabel("Hiv: transmission rate")
plt.ylabel("Tb: transmission rate")
fig.colorbar(c, ax=ax)
plt.show()

# plot the output values for 2019 (last year for which data are available)
# grid plot with hiv and tb transmission rates with (i) hiv prevalence and (ii) tb prevalence

# grid plot with hiv and tb transmission rates with (i) hiv incidence and (ii) tb incidence


# for best-fitting parameter set:
# HIV
# prevalence in adults over time 2010-2030

# incidence in adults

# ART coverage
