"""Collection of utilities for analysing results from the batchrun system"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def get_folders(batch_file_name: str, outputspath: Path) -> list:
    """Returns paths of folders assoicated with a batch_file, in chronological order."""
    stub = batch_file_name.rstrip('.py')
    folders = [Path(f) for f in os.scandir(outputspath) if (f.is_dir() & f.name.startswith(stub))]
    folders.sort()
    return folders


def get_info(results_folder: Path) -> dict:
    """Utility function to get the the number draws and the number of runs in a batch set."""
    info = dict()
    draw_folders = [f for f in os.scandir(results_folder) if f.is_dir()]

    info['number_of_draws'] = len(draw_folders)

    run_folders = [f for f in os.scandir(draw_folders[0]) if f.is_dir()]
    info['runs_per_draw'] = len(run_folders)

    return info


def get_alog(results_folder: Path) -> dict:
    """Utility function to create a dict contaning all the logs from the first run within a batch set."""
    folder = results_folder / str(0) / str(0)
    pickles = [f for f in os.scandir(folder) if f.name.endswith('.pickle')]

    output = dict()
    for p in pickles:
        name = p.name[:-len('.pickle')]
        output[name] = pickle.load(open(p.path, "rb"))

    return output


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
            open(Path(d) / str(0) / str('tlo.scenario.pickle'), "rb")
        )['override_parameter']

        p['module_param'] = p['module'] + ':' + p['name']
        p.index = [int(d.name)] * len(p.index)

        list_of_param_changes.append(p[['module_param', 'new_value']])

    params = pd.concat(list_of_param_changes)
    params.index.name = 'draw'
    params = params.rename(columns={'new_value': 'value'})
    params = params.sort_index()

    return params


def extract_results(results_folder: Path, log_element: dict) -> pd.DataFrame:
    """Utility function to unpack results to produce a dataframe that summaries one series from the log, with column
    multi-index for the draw/run. If an 'index' component of the log_element is provided, the dataframe uses that index
    (but note that this will only work if the index is the same in each run)."""

    if 'index' in log_element:
        # extract the index from the first log, and use this ensure that all other are exactly the same.
        __one_log_component__ = pickle.load(
            open(results_folder / str(0) / str(0) / str(log_element['component'] + '.pickle'), "rb")
        )
        index = eval(f"__one_log_component__{log_element['index']}")

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
                __log_component__ = pickle.load(open(log_component_file, "rb"))
                series = eval(f"__log_component__{log_element['series']}")
                results[draw, run] = series

                idx = eval(f"__log_component__{log_element['index']}")
                assert idx.equals(index)

            except ValueError:
                results[draw, run] = np.nan

    # if 'index' is provied, set this to be the index of the results
    if 'index' in log_element:
        results.index = index

    return results


def summarize(results: pd.DataFrame, only_mean: bool = False) -> pd.DataFrame:
    """Utility function to compute summary statistics that finds mean value and 95% interval across the runs for each
    draw."""
    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                results.columns.unique(level='draw'),
                ["mean", "lower", "upper"]
            ],
            names=['draw', 'stat']),
        index=results.index
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


def get_grid(params: pd.DataFrame, res: pd.Series):
    """Utility function to create the arrays needed to plot a heatmap.
    params:
        This is the dataframe of parameters with index=draw (made using `extract_params()`).
    In res:
        results of interest with index=draw (can be made using `extract_params()`)
    """

    res = pd.concat([params.pivot(columns='module_param', values='value'), res], axis=1)
    piv = res.pivot_table(index=res.columns[0], columns=res.columns[1], values=res.columns[2])

    grid = dict()
    grid[res.columns[0]], grid[res.columns[1]] = np.meshgrid(piv.index, piv.columns)
    grid[res.columns[2]] = piv.values

    return grid
