"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder

if running locally need to parse log files:
tlo parse-log /Users/tmangal/PycharmProjects/TLOmodel/outputs/mihpsa_runs-2025-04-14T130655Z/0/0

mihpsa_runs-2025-04-19T220218Z

"""

import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
from tlo import Date

datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")


# %% Analyse results of runs

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("mihpsa_runs.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder, draw=0)

# get basic information about the results
info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

scaling_factor = log['tlo.methods.population']['scaling_factor'].scaling_factor.values[0]


# %% Population attributes


stock_variables = [
    "N_PLHIV_00_14_C",
    "N_PLHIV_15_24_M",
    "N_PLHIV_15_24_F",
    "N_PLHIV_25_49_M",
    "N_PLHIV_25_49_F",
    "N_PLHIV_50_UP_M",
    "N_PLHIV_50_UP_F",
    "N_Total_00_14_C",
    "N_Total_15_24_M",
    "N_Total_15_24_F",
    "N_Total_25_49_M",
    "N_Total_25_49_F",
    "N_Total_50_UP_M",
    "N_Total_50_UP_F",
    "N_Diag_00_14_C",
    "N_Diag_15_UP_M",
    "N_Diag_15_UP_F",
    "N_ART_00_14_C",
    "N_ART_15_UP_M",
    "N_ART_15_UP_F",
    "N_VLS_15_UP_M",
    "N_VLS_15_UP_F",
    "N_PLHIV_15_UP_AIDS",
    "N_PLHIV_15_UP_NO_AIDS",
]


stocks_output = {}

for stock in stock_variables:
    result = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.hiv",
            key="stock_variables",
            column=stock,
            index="date",
            do_scaling=True,
        ),
        collapse_columns=False,
        only_mean=True
    )

    for draw in result.columns:
        if draw not in stocks_output:
            stocks_output[draw] = pd.DataFrame()  # Initialise DataFrame for the draw if not exists

        stocks_output[draw][stock] = result[draw]

with pd.ExcelWriter(results_folder / "deaths_project_outputs.xlsx", engine='openpyxl') as writer:
    # Iterate over the dictionary and write each DataFrame to a new sheet
    for draw, df in stocks_output.items():
        df = df.T  # Switch rows and columns
        # Writing each draw's DataFrame to a new sheet named after the draw
        df.to_excel(writer, sheet_name=f'Draw_{draw}', index=True)



