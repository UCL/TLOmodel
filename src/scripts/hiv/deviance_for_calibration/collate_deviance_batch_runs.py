
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

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files run for calibration
# calibration_script-2021-12-20T144906Z

# run in terminal: tlo batch-download calibration_script-2021-12-20T144906Z

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("calibration_script.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)
# tlo.methods.deviance_measure[deviance_measure]

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract deviance measure for all runs
# uses draw/run as index by default
extracted = extract_results(
    results_folder,
    module="tlo.methods.deviance_measure",
    key="deviance_measure",
    column="deviance_measure",
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
ax.set_title("Deviance measure")
plt.xlabel("Hiv: transmission rate")
plt.ylabel("Tb: transmission rate")
fig.colorbar(c, ax=ax)
plt.show()

