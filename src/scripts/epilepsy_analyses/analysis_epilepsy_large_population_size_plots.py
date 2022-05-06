"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""
from pathlib import Path

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

inc_summary = summarize(inc, only_mean=True).mean()
inc_death_summary = summarize(inc_death, only_mean=True).mean()
