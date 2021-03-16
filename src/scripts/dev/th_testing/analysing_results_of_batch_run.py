"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' folder
"""

from pathlib import Path
import pickle
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import os

outputspath = Path('./outputs')


# import the class used to generate the batch (todo - way todo this neatly and in one go?)
from scripts.dev.th_testing.mockitis_batch import Mockitis_Batch
bd = Mockitis_Batch()
batch_file_name = 'mockitis_batch.py'


# %% Utility function to get the results folders that have been created by a run of batch script
def get_folders(batch_file_name: str, outputspath: Path) -> list:
    """Returns paths of folders assoicated with a batch script, in chronological order"""
    list_subfolders_with_paths = [Path(f) for f in os.scandir(outputspath) if f.is_dir()]
    list_subfolders_with_paths.sort()
    return list_subfolders_with_paths

#  %% Check that folder and definition of batch have the right structure! (todo)


# %% Utility function to unpack results to produce a dataframe that summaries one series from the log, with column
# multi-index for draw/run

def extract_results(folder: Path, log_element: dict) -> pd.DataFrame:
    results = pd.DataFrame(columns=pd.MultiIndex.from_product(
        [range(bd.number_of_draws), range(bd.runs_per_draw)],
        names=["draw", "run"]
    ))

    for draw in range(bd.number_of_draws):
        for run in range(bd.runs_per_draw):
            try:
                log_component_file = folder / str(draw) / str(run) / str(log_element['component'] + '.pickle')
                log_component = pickle.load(open(log_component_file, "rb"))
                series = eval(f"log_component{log_element['series']}")
                results[draw, run] = series
            except:
                results[draw, run] = np.nan

    return results

# %% Utility function to compute summary statistics that finds mean value and 95% credible intervals across the runs

def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                results.columns.unique(level='draw'),
                ["mean", "lower", "upper"]
            ],
            names=['draw', 'stat'])
    )

    summary.loc[:, (slice(None), "mean")] = results.groupby(axis=1, by='draw').mean().values
    summary.loc[:, (slice(None), "lower")] = results.groupby(axis=1, by='draw').quantile(0.025).values
    summary.loc[:, (slice(None), "upper")] = results.groupby(axis=1, by='draw').quantile(0.975).values

    return summary

# %% Typical work-flow

# 1) Find results_folder for most recent run
results_folder = get_folders(batch_file_name, outputspath)[-1]

# 2) Define the log-element to extract:
log_element = {
    "component": "tlo.methods.mockitis",                # <--- the dataframe that is output
    "series": "['summary'].PropInf"                     # <--- series in the dateframe to be extracted
}

# 3) Get summary of the results for that log-element
propinf = summarize(extract_results(results_folder, log_element))

# 4) Create a plot

# summarize as the value at the end of the run
propinf_end = propinf.iloc[[-1]]

height = propinf_end.loc[:, (slice(None), "mean")].iloc[0].values
lower_upper = np.array(list(zip(
    propinf_end.loc[:, (slice(None), "lower")].iloc[0].values,
    propinf_end.loc[:, (slice(None), "upper")].iloc[0].values
))).transpose()

yerr = abs(lower_upper - height)

plt.bar(
    x=propinf_end.columns.unique(level="draw").values,
    height=propinf_end.loc[:, (slice(None), "mean")].iloc[0].values,
    yerr=yerr
)

plt.show()
