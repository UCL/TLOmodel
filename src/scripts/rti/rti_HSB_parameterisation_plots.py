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

outputspath = Path('./outputs/')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('rti_hsb_parameterisation.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract a specific log series for all runs:
extracted = extract_results(results_folder,
                            module="tlo.methods.rti",
                            key="summary_1m",
                            column="percent sought healthcare",
                            index="date")

# 3) Get summary of the results for that log-element
prop_sought_healthcare = summarize(extracted)
# If only interested in the means
prop_sought_healthcare_onlymeans = summarize(extracted, only_mean=True)
# get per parameter summaries
mean_overall = prop_sought_healthcare.mean()
# get upper and lower estimates
prop_sought_healthcare_lower = mean_overall.loc[:, "lower"]
prop_sought_healthcare_upper = mean_overall.loc[:, "upper"]
lower_upper = np.array(list(zip(
    prop_sought_healthcare_lower.to_list(),
    prop_sought_healthcare_upper.to_list()
))).transpose()
# name of parmaeter that varies
param_name = 'RTI:rt_emergency_care_ISS_score_cut_off'
# find the values that fall within our accepted range of health seeking behaviour based on results of Zafar et al
# doi: 10.1016/j.ijsu.2018.02.034
expected_hsb_upper = 0.85
expected_hsb_lower = 0.6533
per_param_average_hsb = mean_overall[:, 'mean'].values
yerr = abs(lower_upper - per_param_average_hsb)
in_accepted_range = np.where((per_param_average_hsb > expected_hsb_lower) &
                             (per_param_average_hsb < expected_hsb_upper))
xvals = range(info['number_of_draws'])
colors = ['lightsteelblue' if i not in in_accepted_range else 'lightsalmon' for i in xvals]
xlabels = [
    round(params.loc[(params.module_param == param_name)][['value']].loc[draw].value, 3)
    for draw in range(info['number_of_draws'])
]
fig, ax = plt.subplots()
ax.bar(
    x=xvals,
    height=mean_overall[:, 'mean'].values,
    yerr=yerr,
    color=colors
)
ax.set_xticks(xvals)
ax.set_xticklabels(xlabels)
plt.xlabel(param_name)
plt.ylabel('Percentage sought healthcare')
plt.title('Calibration of the ISS score which determines\nautomatic health care seeking')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/ISS_cutoff_score_{min(xlabels)}_"
            f"{max(xlabels)}")

# get the mean percentage sought healthcare in the simulation
mean_per_parameter = prop_sought_healthcare_onlymeans.mean()

# 4) Create some plots:



# i) bar plot to summarize as the value at the end of the run
prop_sought_healthcare_end = prop_sought_healthcare.iloc[[-1]]

height = prop_sought_healthcare_end.loc[:, (slice(None), "mean")].iloc[0].values
lower_upper = np.array(list(zip(
    prop_sought_healthcare_end.loc[:, (slice(None), "lower")].iloc[0].values,
    prop_sought_healthcare_end.loc[:, (slice(None), "upper")].iloc[0].values
))).transpose()

yerr = abs(lower_upper - height)

xvals = range(info['number_of_draws'])
xlabels = [
    round(params.loc[(params.module_param == param_name)][['value']].loc[draw].value, 3)
    for draw in range(info['number_of_draws'])
]

fig, ax = plt.subplots()
ax.bar(
    x=xvals,
    height=prop_sought_healthcare_end.loc[:, (slice(None), "mean")].iloc[0].values,
    yerr=yerr
)
ax.set_xticks(xvals)
ax.set_xticklabels(xlabels)
plt.xlabel(param_name)
plt.show()

# ii) plot to show time-series (means)
for draw in range(info['number_of_draws']):
    plt.plot(
        propinf.loc[:, (draw, "mean")].index, propinf.loc[:, (draw, "mean")].values,
        label=f"{param_name}={round(params.loc[(params.module_param == param_name)][['value']].loc[draw].value, 3)}"
    )
plt.xlabel(propinf.index.name)
plt.legend()
plt.show()

# iii) banded plot to show variation across runs
draw = 0
plt.plot(propinf.loc[:, (draw, "mean")].index, propinf.loc[:, (draw, "mean")].values, 'b')
plt.fill_between(
    propinf.loc[:, (draw, "mean")].index,
    propinf.loc[:, (draw, "lower")].values,
    propinf.loc[:, (draw, "upper")].values,
    color='b',
    alpha=0.5,
    label=f"{param_name}={round(params.loc[(params.module_param == param_name)][['value']].loc[draw].value, 3)}"
)
plt.xlabel(propinf.index.name)
plt.legend()
plt.show()
