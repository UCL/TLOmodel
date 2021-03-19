"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.dev.th_testing.th_utils import (
    extract_params,
    extract_results,
    get_folders,
    get_info,
    get_alog,
    summarize,
)

outputspath = Path('./outputs')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_folders('mockitis_single_sweep.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = get_alog(results_folder)

# get basic information about the results
info = get_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Define the log-element to extract:
log_element = {
    "component": "tlo.methods.mockitis",  # <-- the dataframe that is output
    "series": "['summary'].PropInf",  # <-- series in the dateframe to be extracted
    "index": "['summary'].date",  # <-- (optional) index to use
}

# 3) Get summary of the results for that log-element
propinf = summarize(extract_results(results_folder, log_element))

# if only interestedd in the means
propinf_onlymeans = summarize(extract_results(results_folder, log_element), only_mean=True)

# 4) Create some plots:

# name of parmaeter that varies
param_name = 'Mockitis:p_infection'

# i) bar plot to summarize as the value at the end of the run
propinf_end = propinf.iloc[[-1]]

height = propinf_end.loc[:, (slice(None), "mean")].iloc[0].values
lower_upper = np.array(list(zip(
    propinf_end.loc[:, (slice(None), "lower")].iloc[0].values,
    propinf_end.loc[:, (slice(None), "upper")].iloc[0].values
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
    height=propinf_end.loc[:, (slice(None), "mean")].iloc[0].values,
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
