"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""
from pathlib import Path

import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_params,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
)

# NOTE THAT THIS FILE PATH IS UNIQUE EACH INDIVIDUAL AND WILL BE DIFFERENT FOR EACH USER
outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('analysis_epilepsy_calibrate_incidence_grid.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# for each logfile we will calculate by hand the incidence per 100,000 person years


def extract_inc_epilepsy_per_100000_person_years(results_folder):
    info = get_scenario_info(results_folder)
    ave_inc_per_draw = []
    for draw in range(info['number_of_draws']):
        inc_per_run = []
        for run in range(info['runs_per_draw']):
            ep_df: pd.DataFrame = \
                load_pickled_dataframes(results_folder, draw, run, "tlo.methods.epilepsy")["tlo.methods.epilepsy"]
            ep_df = ep_df['inc_epilepsy']
            ep_df['inc_per_p_y'] = np.divide(ep_df['n_incident_epilepsy'],
                                             np.subtract(np.multiply(ep_df['n_alive'], 1 / 3),
                                                         np.multiply(ep_df['n_incident_epilepsy'], 1 / 6)))
            ep_df['inc_per_100000_p_y'] = ep_df['inc_per_p_y'] * 100000
            inc_per_run.append(ep_df['inc_per_100000_p_y'].tolist())
        # calculate average incidence per draw
        ave_inc_per_draw.append([float(sum(col)) / len(col) for col in zip(*inc_per_run)])


extract_inc_epilepsy_per_100000_person_years(results_folder)
ep_inc = log['tlo.methods.epilepsy']['inc_epilepsy']

# use 4 month intervals to calculate incidence per person year, assume that any cases occurred half way through the
# quarter
ep_inc['inc_per_p_y'] = np.divide(ep_inc['n_incident_epilepsy'],
                                  np.subtract(np.multiply(ep_inc['n_alive'], 1 / 3),
                                              np.multiply(ep_inc['n_incident_epilepsy'], 1 / 6)))
ep_inc['inc_per_100000_p_y'] = ep_inc['inc_per_p_y'] * 100000
# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

target_incidence = 77
