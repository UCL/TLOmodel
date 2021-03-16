"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' folder
"""

from pathlib import Path
import pickle
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

# import the class used to generate the batch
from scripts.dev.th_testing.mockitis_batch import Mockitis_Batch
bd = Mockitis_Batch()

# Declare output folder:
# todo- is there a way to know this automatically from the batchfile? (name is predictable but not time-stamp;
#  perhaps could just find the most recent of all folders with that stub).
folder = Path('outputs/mockitis_batch-2021-03-15T125516Z')

# %% Utility function to unpack results to produce a dataframe that summaries one series from the log, with column
# multi-index for draw/run

def extract_results(dict: log_element):
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

def summarize(results):
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

# %%

# Define the log-element to extract:
log_element = {
    "component": "tlo.methods.mockitis",    # <--- the dataframe that is output
    "series": "['summary'].PropInf"                     # <--- series in the dateframe to be extracted
}

propinf = summarize(extract_results(log_element))

# summarize as the value at the end of the run
propinf_end = propinf.iloc[[-1]]

height = propinf_end.loc[:, (slice(None), "mean")].iloc[0].values
lower_upper = np.array(list(zip(
    propinf_end.loc[:, (slice(None), "lower")].iloc[0].values,
    propinf_end.loc[:, (slice(None), "upper")].iloc[0].values
))).transpose()

yerr = lower_upper - height

plt.bar(
    x=propinf_end.columns.unique(level="draw").values,
    height=propinf_end.loc[:, (slice(None), "mean")].iloc[0].values,
    yerr=yerr
)

plt.show()
