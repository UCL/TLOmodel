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
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

# NOTE THAT THIS FILE PATH IS UNIQUE EACH INDIVIDUAL AND WILL BE DIFFERENT FOR EACH USER
outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('analysis_epilepsy_calibrate_seiz_stats.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

prop_seiz_0 = extract_results(results_folder,
                              module="tlo.methods.epilepsy",
                              key="epilepsy_logging",  # <-- the key used for the logging entry
                              column="prop_seiz_stat_0",  # <-- the column in the dataframe
                              index="date")

prop_seiz_1 = extract_results(results_folder,
                              module="tlo.methods.epilepsy",
                              key="epilepsy_logging",  # <-- the key used for the logging entry
                              column="prop_seiz_stat_1",  # <-- the column in the dataframe
                              index="date")

prop_seiz_2 = extract_results(results_folder,
                              module="tlo.methods.epilepsy",
                              key="epilepsy_logging",  # <-- the key used for the logging entry
                              column="prop_seiz_stat_2",  # <-- the column in the dataframe
                              index="date")

prop_seiz_3 = extract_results(results_folder,
                              module="tlo.methods.epilepsy",
                              key="epilepsy_logging",  # <-- the key used for the logging entry
                              column="prop_seiz_stat_3",  # <-- the column in the dataframe
                              index="date")


seiz_status_over_time = pd.DataFrame(index=prop_seiz_0.index)
seiz_status_over_time['seiz_stat_0'] = summarize(prop_seiz_0, only_mean=True).values
seiz_status_over_time['seiz_stat_1'] = summarize(prop_seiz_1, only_mean=True).values
seiz_status_over_time['seiz_stat_2'] = summarize(prop_seiz_3, only_mean=True).values
seiz_status_over_time['seiz_stat_3'] = summarize(prop_seiz_3, only_mean=True).values
seiz_status_over_time['year'] = seiz_status_over_time.index.year
seiz_status_over_time = seiz_status_over_time.groupby('year').mean()
plt.tight_layout()
for status_number, col in enumerate(seiz_status_over_time.columns):
    plt.subplot(2, 2, status_number + 1)
    plt.plot(seiz_status_over_time.index, seiz_status_over_time[col], color='lightsteelblue', label='Model')
    plt.xticks(seiz_status_over_time.index, rotation=45, fontsize=6)
    plt.ylabel('Proportion')
    # plt.yticks(np.round(np.linspace(seiz_status_over_time[col].min(), seiz_status_over_time[col].max(), 10), 5),
    #            fontsize=6)
    plt.title(f"seiz_stat_{status_number}")
    if status_number > 0:
        plt.axhline(0.013, label='Ba Diop', color='lightsalmon')
    else:
        plt.axhline(1 - 3 * 0.013, label='Ba Diop', color='lightsalmon')
    plt.legend()
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Epilepsy/Large_run/"
            "large_pop_size_prop_seiz_stats.png", bbox_inches='tight')

