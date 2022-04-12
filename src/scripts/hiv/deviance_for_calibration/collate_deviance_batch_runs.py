
from pathlib import Path

import matplotlib.pyplot as plt

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files run for calibration
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
extracted.to_csv(outputspath / ("full_LHC_outputs2" + ".csv"))

# 3) Get summary of the results for that log-element
res = summarize(extracted, only_mean=True).iloc[-1]
res.name = "z"

# combine the outputs and export results to csv
params['draw'] = params.index
combined_output = params.pivot(index="draw", columns="module_param", values="value")
combined_output["deviance"] = res.values
combined_output.to_csv(outputspath / ("LHC_outputs2" + ".csv"))


# plot the deviance against parameters
fig, ax = plt.subplots()
sc = ax.scatter(combined_output["Hiv:beta"],
                combined_output["Tb:transmission_rate"],
                c=combined_output["deviance"],
                cmap="GnBu", s=400)
ax.set_title("Deviance measure")
plt.xlabel("Hiv: transmission rate")
plt.ylabel("Tb: transmission rate")
fig.colorbar(sc, ax=ax)
plt.show()
