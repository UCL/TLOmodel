"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from tlo.analysis.utils import (
    extract_params,
    extract_params_from_json,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('rti_incidence_parameterisation.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params_from_json(results_folder, 'rti_incidence_parameterisation.py', 'RTI', 'base_rate_injrti')
# 2) Extract a specific log series for all runs:
extracted = extract_results(results_folder,
                            module="tlo.methods.rti",
                            key="summary_1m",
                            column="incidence of rti per 100,000",
                            index="date")

# 3) Get summary of the results for that log-element
incidence = summarize(extracted)
# If only interested in the means
incidence_onlymeans = summarize(extracted, only_mean=True)
# get per parameter summaries
mean_incidence_overall = incidence.mean()
# get upper and lower estimates
mean_incidence_lower = mean_incidence_overall.loc[:, "lower"]
mean_incidence_upper = mean_incidence_overall.loc[:, "upper"]
lower_upper = np.array(list(zip(
    mean_incidence_lower.to_list(),
    mean_incidence_upper.to_list()
))).transpose()
# name of parmaeter that varies
param_name = 'RTI:base_rate_injrti'
# find the values that fall within our accepted range of incidence based on results of the GBD study
gbd_data = pd.read_csv('resources/gbd/ResourceFile_RTI_Deaths_and_Incidence.csv')
incidence_data = gbd_data.loc[gbd_data['measure'] == 'Incidence']
incidence_data = incidence_data.loc[incidence_data['metric'] == 'Rate']
incidence_data = incidence_data.loc[incidence_data['year'] > 2009]
expected_incidence_upper = incidence_data['upper'].mean()
expected_incidence_lower = incidence_data['lower'].mean()
expected_incidence = incidence_data['val'].mean()
per_param_average_incidence = mean_incidence_overall[:, 'mean'].values
yerr = abs(lower_upper - per_param_average_incidence)
in_accepted_range = np.where((per_param_average_incidence > expected_incidence_lower) &
                             (per_param_average_incidence < expected_incidence_upper))
xvals = range(info['number_of_draws'])
colors = ['lightsteelblue' if i not in in_accepted_range[0] else 'lightsalmon' for i in xvals]
best_fit_found = min(per_param_average_incidence, key = lambda x: abs(x - expected_incidence))
best_fit_index = np.where(per_param_average_incidence == best_fit_found)
colors[best_fit_index[0][0]] = 'gold'
print(f"best fitting parameter value = {params[best_fit_index[0][0]]}")
xlabels = [np.round(value, 5) for value in params]
fig, ax = plt.subplots()
ax.bar(
    x=xvals,
    height=mean_incidence_overall[:, 'mean'].values,
    yerr=yerr,
    color=colors,
)
ax.set_xticks(xvals)
ax.set_xticklabels(xlabels, rotation=90)
plt.xlabel(param_name)
plt.ylabel('Incidence of RTI per 100,000')
plt.title('Calibration of the base rate of rti injury which determines\n the overall incidence of RTI')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/base_rate_rti_{min(xlabels)}_"
            f"{max(xlabels)}.png", bbox_inches='tight')
