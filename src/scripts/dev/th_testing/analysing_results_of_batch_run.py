"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""

from pathlib import Path
import pickle
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import os

outputspath = Path('./outputs')


# %%
def get_folders(batch_file_name: str, outputspath: Path) -> list:
    """Returns paths of folders assoicated with a batch_file, in chronological order."""
    # todo - get association with batchfile

    stub = batch_file_name.rstrip('.py')

    folders = [Path(f) for f in os.scandir(outputspath) if (f.is_dir() & f.name.startswith(stub))]

    folders.sort()
    return folders

# %% Utility function to get the information on the runs (number draws, run and their paths)
def get_info(results_folder: Path) -> dict:
    """Utility function to get the information on the runs (number draws, run and their paths)"""
    info = dict()
    draw_folders = [f for f in os.scandir(results_folder) if f.is_dir()]

    info['number_of_draws'] = len(draw_folders)

    run_folders = [f for f in os.scandir(draw_folders[0]) if f.is_dir()]
    info['runs_per_draw'] = len(run_folders)

    return info

# %% Utility function to open one log from within a batch set
def getalog(results_folder: Path) -> dict:
    folder = results_folder / str(0) / str(0)
    pickles = [f for f in os.scandir(folder) if f.name.endswith('.pickle')]

    output = dict()
    for p in pickles:
        name = p.name[:-len('.pickle')]
        output[name] = pickle.load(open(p.path, "rb"))

    return output

# %%
def extract_params(results_folder: Path) -> pd.DataFrame:
    """Utility function to unpack results to produce a dateframe that summarizes that parameters that change across
    the draws. It produces a dataframe with index of draw and columns of each parameters that is specified to be varied
     in the batch.
     NB. This does the extraction from run 0 in each draw, under the assumption that the over-written parameters are the
      same in each run."""

    # Get the paths for the draws
    draws = [f for f in os.scandir(results_folder) if f.is_dir()]

    list_of_param_changes = list()

    for d in draws:
        p = pickle.load(
            open(Path(d)/ str(0) / str('tlo.scenario.pickle'), "rb")
        )['override_parameter']

        p['module_param'] = p['module'] + ':' + p['name']
        p.index = [int(d.name)] * len(p.index)

        list_of_param_changes.append(p[['module_param', 'new_value']])

    params = pd.concat(list_of_param_changes)
    params.index.name = 'draw'
    params = params.rename(columns={'new_value': 'value'})
    params = params.sort_index()

    return params

# %%
def extract_results(results_folder: Path, log_element: dict) -> pd.DataFrame:
    """Utility function to unpack results to produce a dataframe that summaries one series from the log, with column
    multi-index for draw/run. If an 'index' component of the log_element is provided, this will only work if the index
    is the same in each run."""

    if 'index' in log_element:
        # extract the index from the first log, and use this ensure that all other are exactly the same.
        one_log_component = pickle.load(
            open(results_folder / str(0) / str(0) / str(log_element['component'] + '.pickle'), "rb")
        )
        index = eval(f"one_log_component{log_element['index']}")

    # get number of draws and numbers of runs
    info = get_info(results_folder)

    results = pd.DataFrame(columns=pd.MultiIndex.from_product(
        [range(info['number_of_draws']), range(info['runs_per_draw'])],
        names=["draw", "run"]
    ))

    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):
            try:
                log_component_file = results_folder / str(draw) / str(run) / str(log_element['component'] + '.pickle')
                log_component = pickle.load(open(log_component_file, "rb"))
                series = eval(f"log_component{log_element['series']}")
                results[draw, run] = series

                idx = eval(f"log_component{log_element['index']}")
                assert idx.equals(index)

            except:
                results[draw, run] = np.nan

    # if 'index' is provied, set this to be the index of the results
    if 'index' in log_element:
        results.index = index

    return results

# %%
def summarize(results: pd.DataFrame, only_mean: bool=False) -> pd.DataFrame:
    """Utility function to compute summary statistics that finds mean value and 95% credible intervals across the
    runs. """
    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                results.columns.unique(level='draw'),
                ["mean", "lower", "upper"]
            ],
            names=['draw', 'stat']),
        index = results.index
    )

    summary.loc[:, (slice(None), "mean")] = results.groupby(axis=1, by='draw').mean().values
    summary.loc[:, (slice(None), "lower")] = results.groupby(axis=1, by='draw').quantile(0.025).values
    summary.loc[:, (slice(None), "upper")] = results.groupby(axis=1, by='draw').quantile(0.975).values

    if only_mean:
        # Remove other metrics and simplify if 'only_mean' is required:
        om = summary.loc[:, (slice(None), "mean")]
        om.columns = [c[0] for c in om.columns.to_flat_index()]
        return om

    return summary

# %% Typical work-flow

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_folders('mockitis_batch.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = getalog(results_folder)

# get basic information about the results
info = get_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Define the log-element to extract:
log_element = {
    "component": "tlo.methods.mockitis",    # <-- the dataframe that is output
    "series": "['summary'].PropInf",        # <-- series in the dateframe to be extracted
    "index": "['summary'].date",            # <-- (optional) index to use
}

# 3) Get summary of the results for that log-element
propinf = summarize(extract_results(results_folder, log_element))

# if only interestedd in the means
propinf_onlymeans = summarize(extract_results(results_folder, log_element), only_mean=True)


# 4) Create some plots:

# name of parmaeter that varies
param_name='Mockitis:p_infection'

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
    round(params.loc[(params.module_param==param_name)][['value']].loc[draw].value, 3)
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
    plt.plot(propinf.loc[:, (draw, "mean")].index, propinf.loc[:, (draw, "mean")].values,
             label=f"{param_name}={round(params.loc[(params.module_param==param_name)][['value']].loc[draw].value, 3)}")
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
    color = 'b',
    alpha = 0.5,
    label=f"{param_name}={round(params.loc[(params.module_param==param_name)][['value']].loc[draw].value, 3)}"
)
plt.xlabel(propinf.index.name)
plt.legend()
plt.show()
