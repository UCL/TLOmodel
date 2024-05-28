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
results_folder = get_scenario_outputs("mihpsa_runs.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

scaling_factor = log['tlo.methods.population']['scaling_factor'].scaling_factor.values[0]

# -----------------------------------------------------------------
# export one run
log0 = log['tlo.methods.hiv']['stock_variables']
# Select columns to be multiplied (excluding the first column 'First')
columns_to_multiply = log0.columns[1:]

# Multiply selected columns by scaling factor
log0[columns_to_multiply] = (log0[columns_to_multiply] * scaling_factor).astype(int)
log0.to_csv(outputspath / 'mihpsa_stock.csv')


log0F = log['tlo.methods.hiv']['flow_variables']
columns_to_multiply = log0F.columns[1:]
log0F[columns_to_multiply] = (log0F[columns_to_multiply] * scaling_factor).astype(int)

log0F.to_csv(outputspath / 'mihpsa_flow.csv')


# get test outputs

# N_HIVTest_Facility_NEG_15_UP
# N_HIVTest_Facility_POS_15_UP
# N_HIVTest_Index_NEG_15_UP
# N_HIVTest_Index_POS_15_UP
# N_HIVTest_Community_NEG_15_UP
# N_HIVTest_Community_POS_15_UP
# N_HIVTest_SelfTest_POS_15_UP
# N_HIVTest_SelfTest_Dist

# number tests age >=15, hiv_status=false
# Convert 'datetime' column to datetime type
log_tests = log['tlo.methods.hiv']['hiv_test']
log_tests['date'] = pd.to_datetime(log_tests['date'])

# Filter for age 15 years and up
log_tests_filtered = log_tests[log_tests['adult'] == True]

# Extract year from the datetime column
log_tests_filtered['year'] = log_tests_filtered['date'].dt.year

# Group by 'hiv_status' and 'year', then count the number of entries
result = log_tests_filtered.groupby(['hiv_status', 'year']).size().reset_index()

# scale to full population
result[0] = (result[0] * scaling_factor).astype(int)

result.to_csv(outputspath / 'mihpsa_tests.csv')


# get deaths

# N_DeathsHIV_00_14_C
# N_DeathsHIV_15_UP_M
# N_DeathsHIV_15_UP_F
#
# N_DeathsAll_00_14_C
# N_DeathsAll_15_UP_M
# N_DeathsAll_15_UP_F

deaths = log['tlo.methods.demography']['death']
deaths['date'] = pd.to_datetime(deaths['date'])
deaths['year'] = deaths['date'].dt.year

# create new column adult=true/false
deaths['adult'] = deaths['age'] > 14
deaths['adult'] = deaths['adult'].astype(bool)

deaths_hiv = deaths[deaths['label'] == 'AIDS']

# get HIV deaths, child
result_hiv = deaths_hiv.groupby(['adult', 'year']).size().reset_index()
result_hiv[0] = (result_hiv[0] * scaling_factor).astype(int)
result_hiv.to_csv(outputspath / 'mihpsa_hiv_deaths_child.csv')


# get HIV deaths, child
result_hiv_adult = deaths_hiv[deaths_hiv['adult'] == True]
result_hiv_adult = deaths_hiv.groupby(['year', 'sex']).size().reset_index()
result_hiv_adult[0] = (result_hiv_adult[0] * scaling_factor).astype(int)
result_hiv_adult.to_csv(outputspath / 'mihpsa_hiv_deaths_adult.csv')


# get all deaths, child
result_all = deaths.groupby(['adult', 'year']).size().reset_index()
result_all[0] = (result_all[0] * scaling_factor).astype(int)
result_all.to_csv(outputspath / 'mihpsa_all_deaths_child.csv')


# get HIV deaths, child
result_all_adult = deaths[deaths['adult'] == True]
result_all_adult = deaths.groupby(['year', 'sex']).size().reset_index()
result_all_adult[0] = (result_all_adult[0] * scaling_factor).astype(int)
result_all_adult.to_csv(outputspath / 'mihpsa_all_deaths_adult.csv')


# get deaths by single years of age, male and female
# male deaths
full_output = deaths_hiv.groupby(['age', 'year', 'sex']).size().reset_index()

# convert to wide format
# Filter rows by sex=M first, then sex=F
full_output_sorted = full_output.sort_values(by='sex', ascending=False)

# Pivot the DataFrame to wide format
pivot_df = full_output_sorted.pivot_table(index='age', columns='year', values=0, aggfunc='first')

# Display the pivot DataFrame
print(pivot_df)

# -----------------------------------------------------------------


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

flow_variables = [
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

# extract the dataframe for stock variables for each run
# find mean
# scale to full population size

log0 = load_pickled_dataframes(results_folder, draw=0, run=0)
log1 = load_pickled_dataframes(results_folder, draw=0, run=1)
log2 = load_pickled_dataframes(results_folder, draw=0, run=2)

stock0 = log0['tlo.methods.hiv']['stock_variables'].iloc[:, 1:]
stock1 = log1['tlo.methods.hiv']['stock_variables'].iloc[:, 1:]
stock2 = log2['tlo.methods.hiv']['stock_variables'].iloc[:, 1:]

mean_values = ((stock0 + stock1 + stock2) / 3) * scaling_factor
mean_values_rounded = mean_values.round().astype(int)
mean_values_rounded.to_csv(outputspath / 'MIHPSA_May2024/mihpsa_stock_FULL.csv')


flow0 = log0['tlo.methods.hiv']['flow_variables'].iloc[:, 1:]
flow1 = log1['tlo.methods.hiv']['flow_variables'].iloc[:, 1:]
flow2 = log2['tlo.methods.hiv']['flow_variables'].iloc[:, 1:]

mean_values = ((flow0 + flow1 + flow2) / 3) * scaling_factor
mean_values_rounded = mean_values.round().astype(int)
mean_values_rounded.to_csv(outputspath / 'MIHPSA_May2024/mihpsa_flow_FULL.csv')



