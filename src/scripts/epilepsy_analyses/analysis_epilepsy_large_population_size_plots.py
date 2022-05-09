"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

# NOTE THAT THIS FILE PATH IS UNIQUE EACH INDIVIDUAL AND WILL BE DIFFERENT FOR EACH USER
outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('analysis_epilepsy_large_population_size.py', outputspath)[-1]

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

inc_summary = summarize(inc, only_mean=True)
inc_mean = summarize(inc, only_mean=True).mean()

inc_death_summary = summarize(inc_death, only_mean=True)
inc_death_mean = summarize(inc_death, only_mean=True).mean()

gbd_inc = 47.26
gbd_inc_death = 1.88
plt.tight_layout()
plt.subplot(2, 1, 1)
plt.plot(inc_summary.index, inc_summary.values, color='lightsteelblue', label='Inc')
plt.plot(inc_death_summary.index, inc_death_summary.values, color='lightsalmon', label='Inc death')
plt.xlabel('Date')
plt.ylabel('Incidence per 100,000 p.y.')
plt.legend()
plt.subplot(2, 1, 2)
plt.bar(np.arange(2), [inc_mean.values[0], inc_death_mean.values[0]], color='rebeccapurple', width=0.4, label='Model')
plt.bar(np.arange(2) + 0.4, [gbd_inc, gbd_inc_death], color='royalblue', width=0.4, label='GBD')
plt.xticks(np.arange(2) + 0.2, ['Mean incidence', 'Mean incidence\nof death'])
plt.legend()
plt.ylabel('Incidence per 100,000 p.y.')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Epilepsy/large_pop_size_inc_summary.png",
            bbox_inches='tight')
