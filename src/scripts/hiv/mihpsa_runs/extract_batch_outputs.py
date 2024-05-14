"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
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

datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")


# %% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("mihpsa_minimal_scenario.py.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

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
]

Flow_variables = [
                "N_BirthAll",
                "N_BirthHIV",
                "N_BirthART",
                "N_NewHIV_00_14_C",
                "N_NewHIV_15_24_M",
                "N_NewHIV_15_24_F",
                "N_NewHIV_25_49_M",
                "N_NewHIV_25_49_F",
                "N_NewHIV_50_UP_M",
                "N_NewHIV_50_UP_F",
                "N_DeathsHIV_00_14_C",
                "N_DeathsHIV_15_UP_M",
                "N_DeathsHIV_15_UP_F",
                "N_DeathsAll_00_14_C",
                "N_DeathsAll_15_UP_M",
                "N_DeathsAll_15_UP_F",
                "N_YLL_00_14_C",
                "N_YLL_15_UP_M",
                "N_YLL_15_UP_F",
                "N_HIVTest_Facility_NEG_15_UP",
                "N_HIVTest_Facility_POS_15_UP",
                "N_HIVTest_Index_NEG_15_UP",
                "N_HIVTest_Index_POS_15_UP",
                "N_HIVTest_Community_NEG_15_UP",
                "N_HIVTest_Community_POS_15_UP",
                "N_HIVTest_SelfTest_POS_15_UP",
                "N_HIVTest_SelfTest_Dist",
                "N_Condom_Acts",
                "N_NewVMMC",
                "PY_PREP_ORAL_AGYW",
                "PY_PREP_ORAL_FSW",
                "PY_PREP_ORAL_MSM",
                "PY_PREP_INJECT_AGYW",
                "PY_PREP_INJECT_FSW",
                "PY_PREP_INJECT_MSM",
                "N_ART_ADH_15_UP_F",
                "N_ART_ADH_15_UP_M",
                "N_VL_TEST_15_UP",
                "N_VL_TEST_00_14",
                "N_OUTREACH_FSW",
                "N_OUTREACH_MSM",
                "N_EconEmpowerment",
                "N_CSE_15_19_F",
                "N_CSE_15_19_M"]

# %% extract results
# Load and format model results (with year as integer):

# ---------------------------------- HIV ---------------------------------- #
model_hiv_adult_prev = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_adult_15plus",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_adult_prev.index = model_hiv_adult_prev.index.year

model_hiv_adult_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_adult_inc.index = model_hiv_adult_inc.index.year
