"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputs' results_folder
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

outputspath = Path('./outputs/t.mangal@imperial.ac.uk')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs('baseline_scenario.py', outputspath)

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract a specific logged output for all runs, e.g. prevalence:
extracted = extract_results(results_folder,
                            module="tlo.methods.hiv",
                            key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                            column="hiv_prev_adult_15plus",
                            index="date")

# 3) Get summary of the results for that log-element
prevalence = summarize(extracted)

# get the mean and 95% interval
prevalence_means = summarize(extracted, only_mean=False)




