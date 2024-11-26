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
from tlo import Date

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
# # export one run
# log0 = log['tlo.methods.hiv']['stock_variables']
# # Select columns to be multiplied (excluding the first column 'First')
# columns_to_multiply = log0.columns[1:]
#
# # Multiply selected columns by scaling factor
# log0[columns_to_multiply] = (log0[columns_to_multiply] * scaling_factor).astype(int)
# # log0.to_csv(outputspath / 'mihpsa_stock.csv')
#
#
# log0F = log['tlo.methods.hiv']['flow_variables']
# columns_to_multiply = log0F.columns[1:]
# log0F[columns_to_multiply] = (log0F[columns_to_multiply] * scaling_factor).astype(int)
#
# # log0F.to_csv(outputspath / 'mihpsa_flow.csv')
#
#
# # get test outputs
#
# # N_HIVTest_Facility_NEG_15_UP
# # N_HIVTest_Facility_POS_15_UP
# # N_HIVTest_Index_NEG_15_UP
# # N_HIVTest_Index_POS_15_UP
# # N_HIVTest_Community_NEG_15_UP
# # N_HIVTest_Community_POS_15_UP
# # N_HIVTest_SelfTest_POS_15_UP
# # N_HIVTest_SelfTest_Dist
#
# # number tests age >=15, hiv_status=false
# # Convert 'datetime' column to datetime type
# log_tests = log['tlo.methods.hiv']['hiv_test']
# log_tests['date'] = pd.to_datetime(log_tests['date'])
#
# # Filter for age 15 years and up
# log_tests_filtered = log_tests[log_tests['adult'] == True]
#
# # Extract year from the datetime column
# log_tests_filtered['year'] = log_tests_filtered['date'].dt.year
#
# # Group by 'hiv_status' and 'year', then count the number of entries
# result = log_tests_filtered.groupby(['hiv_status', 'year']).size().reset_index()
#
# # scale to full population
# result[0] = (result[0] * scaling_factor).astype(int)
#
# # result.to_csv(outputspath / 'mihpsa_tests.csv')
#
#
# # get deaths
#
# # N_DeathsHIV_00_14_C
# # N_DeathsHIV_15_UP_M
# # N_DeathsHIV_15_UP_F
# #
# # N_DeathsAll_00_14_C
# # N_DeathsAll_15_UP_M
# # N_DeathsAll_15_UP_F
#
# deaths = log['tlo.methods.demography']['death']
# deaths['date'] = pd.to_datetime(deaths['date'])
# deaths['year'] = deaths['date'].dt.year
#
# # create new column adult=true/false
# deaths['adult'] = deaths['age'] > 14
# deaths['adult'] = deaths['adult'].astype(bool)
#
# deaths_hiv = deaths[deaths['label'] == 'AIDS']
#
# ### HIV DEATHS
# # get HIV deaths, child
# result_hiv = deaths_hiv.groupby(['adult', 'year']).size().reset_index()
# result_hiv = deaths_hiv[deaths_hiv['adult'] == False].groupby('year').size().reset_index(name='count')
# result_hiv["count"] = (result_hiv["count"] * scaling_factor).astype(int)
# # result_hiv.to_csv(outputspath / 'mihpsa_hiv_deaths_child.csv')
#
#
# # get HIV deaths, adult
# # result_hiv_adult = deaths_hiv[deaths_hiv['adult'] == True]
# result_hiv_adult = deaths_hiv[deaths_hiv['adult'] == True].groupby(['year', 'sex']).size().reset_index()
# result_hiv_adult[0] = (result_hiv_adult[0] * scaling_factor).astype(int)
# # result_hiv_adult.to_csv(outputspath / 'mihpsa_hiv_deaths_adult.csv')
#
#
# ### ALL DEATHS
# # get all deaths, child
# result_all = deaths.groupby(['adult', 'year']).size().reset_index()
# result_all[0] = (result_all[0] * scaling_factor).astype(int)
# # result_all.to_csv(outputspath / 'mihpsa_all_deaths_child.csv')
#
#
# # get all deaths, adult
# result_all_adult = deaths[deaths['adult'] == True]
# result_all_adult = deaths.groupby(['year', 'sex']).size().reset_index()
# result_all_adult[0] = (result_all_adult[0] * scaling_factor).astype(int)
# # result_all_adult.to_csv(outputspath / 'mihpsa_all_deaths_adult.csv')
#
#
# # get deaths by single years of age, male and female
# # male deaths
# full_output = deaths_hiv.groupby(['age', 'year', 'sex']).size().reset_index()
#
# # convert to wide format
# # Filter rows by sex=M first, then sex=F
# full_output_sorted = full_output.sort_values(by='sex', ascending=False)
#
# # Pivot the DataFrame to wide format
# pivot_df = full_output_sorted.pivot_table(index='age', columns='year', values=0, aggfunc='first')
#
# # Display the pivot DataFrame
# print(pivot_df)

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
    "N_PLHIV_15_UP_AIDS",
    "N_PLHIV_15_UP_NO_AIDS",
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

# %% extract the intervention scenarios

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

flows_output = {}

for flow in flow_variables:
    result = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.hiv",
            key="flow_variables",
            column=flow,
            index="date",
            do_scaling=True,
        ),
        collapse_columns=False,
        only_mean=True
    )

    for draw in result.columns:
        if draw not in flows_output:
            flows_output[draw] = pd.DataFrame()  # Initialise DataFrame for the draw if not exists

        flows_output[draw][flow] = result[draw]

# Output the stocks_output dict to an Excel workbook
with pd.ExcelWriter(results_folder / 'stocks_output.xlsx') as writer:
    for sheet_name, df in stocks_output.items():
        df.to_excel(writer, sheet_name=str(sheet_name))

with pd.ExcelWriter(results_folder / 'flows_output.xlsx') as writer:
    for sheet_name, df in flows_output.items():
        df.to_excel(writer, sheet_name=str(sheet_name))

# EXTRACT DEATHS
TARGET_PERIOD = (Date(2025, 1, 1), Date(2050, 12, 31))


def summarise_deaths(results_folder,
                                   label=None, age=None, sex=None):
    """ returns mean deaths for each year of the simulation
    values are aggregated across the runs of each draw
    for the specified cause
    """

    results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "label"])["person_id"].count()
        ),
        do_scaling=True,
    )
    # removes multi-index
    results_deaths = results_deaths.reset_index()

    # select only cause specified
    if label == 'AIDS':
        tmp = results_deaths.loc[
            (results_deaths.label == label)
        ]
    # otherwise all deaths
    else:
        tmp = results_deaths

    if age == 'children':
        tmp = tmp.loc[tmp.age < 15]
    else:
        tmp = tmp.loc[tmp.age >= 15]

    if sex == 'M':
        tmp = tmp.loc[tmp.sex == 'M']
    elif sex == 'F':
        tmp = tmp.loc[tmp.sex == 'F']

    # group deaths by year
    tmp = pd.DataFrame(tmp.groupby(["year"]).sum())

    # get mean for each draw
    mean_deaths = pd.concat({'mean': tmp.iloc[:, 1:].groupby(level=0, axis=1).mean()}, axis=1).swaplevel(axis=1)

    return mean_deaths


aids_deaths_children = summarise_deaths(results_folder,
                                                      label='AIDS',
                                                      age='children')

aids_deaths_men = summarise_deaths(results_folder,
                                                 label='AIDS',
                                                 age='adult',
                                                 sex='M')









#
# def get_aids_deaths_children(_df):
#     """Return total number of deaths where label='AIDS' and age<15 within the TARGET_PERIOD.
#     df returned: series with rows for each draw, values are total counts.
#     """
#     # Filter for label "AIDS" and age less than 15 within the TARGET_PERIOD
#     filtered_df = _df.loc[
#         (pd.to_datetime(_df.date).between(*TARGET_PERIOD)) &
#         (_df['label'] == 'AIDS') &
#         (_df['age'] < 15)
#         ]
#
#     # Group by the 'draw' column (assuming 'draw' is part of the DataFrame) and count the deaths
#     return filtered_df.groupby('label').size()
#
#
# aids_deaths_children = summarize(
#     extract_results(
#         results_folder,
#         module='tlo.methods.demography',
#         key='death',
#         custom_generate_series=get_aids_deaths_children,
#         do_scaling=True
#     ),
#     collapse_columns=False,
#     only_mean=True
# )
#
#
# def get_aids_deaths_men(_df):
#     """Return total number of deaths where label='AIDS' and age<15 within the TARGET_PERIOD.
#     df returned: single count value, not grouped by COD.
#     """
#     # Filter for label "AIDS" and age less than 15
#     filtered_df = _df.loc[
#         (pd.to_datetime(_df.date).between(*TARGET_PERIOD)) &
#         (_df['label'] == 'AIDS') &
#         (_df['age'] >= 15) &
#         (_df['sex'] == 'M')
#         ]
#
#     # Return the count of relevant deaths
#     return filtered_df.shape[0]  # This gives the total number of deaths
#
#
# aids_deaths_men = summarize(
#     extract_results(
#         results_folder,
#         module='tlo.methods.demography',
#         key='death',
#         custom_generate_series=get_aids_deaths_men,
#         do_scaling=True
#     ),
#     collapse_columns=False,
#     only_mean=True
# )
#
#
# def get_aids_deaths_women(_df):
#     """Return total number of deaths where label='AIDS' and age<15 within the TARGET_PERIOD.
#     df returned: single count value, not grouped by COD.
#     """
#     # Filter for label "AIDS" and age less than 15
#     filtered_df = _df.loc[
#         (pd.to_datetime(_df.date).between(*TARGET_PERIOD)) &
#         (_df['label'] == 'AIDS') &
#         (_df['age'] >= 15) &
#         (_df['sex'] == 'F')
#         ]
#
#     # Return the count of relevant deaths
#     return filtered_df.shape[0]  # This gives the total number of deaths
#
#
# aids_deaths_women = summarize(
#     extract_results(
#         results_folder,
#         module='tlo.methods.demography',
#         key='death',
#         custom_generate_series=get_aids_deaths_women,
#         do_scaling=True
#     ),
#     collapse_columns=False,
#     only_mean=True
# )
#
#
# def get_all_deaths_children(_df):
#     """Return total number of deaths where label='AIDS' and age<15 within the TARGET_PERIOD.
#     df returned: single count value, not grouped by COD.
#     """
#     # Filter for label "AIDS" and age less than 15
#     filtered_df = _df.loc[
#         (pd.to_datetime(_df.date).between(*TARGET_PERIOD)) &
#         (_df['age'] < 15)
#         ]
#
#     # Return the count of relevant deaths
#     return filtered_df.shape[0]  # This gives the total number of deaths
#
#
# all_deaths_children = summarize(
#     extract_results(
#         results_folder,
#         module='tlo.methods.demography',
#         key='death',
#         custom_generate_series=get_all_deaths_children,
#         do_scaling=True
#     ),
#     collapse_columns=False,
#     only_mean=True
# )
#
#
# def get_all_deaths_men(_df):
#     """Return total number of deaths where label='AIDS' and age<15 within the TARGET_PERIOD.
#     df returned: single count value, not grouped by COD.
#     """
#     # Filter for label "AIDS" and age less than 15
#     filtered_df = _df.loc[
#         (pd.to_datetime(_df.date).between(*TARGET_PERIOD)) &
#         (_df['age'] >= 15) &
#         (_df['sex'] == 'M')
#         ]
#
#     # Return the count of relevant deaths
#     return filtered_df.shape[0]  # This gives the total number of deaths
#
#
# all_deaths_men = summarize(
#     extract_results(
#         results_folder,
#         module='tlo.methods.demography',
#         key='death',
#         custom_generate_series=get_all_deaths_men,
#         do_scaling=True
#     ),
#     collapse_columns=False,
#     only_mean=True
# )
#
#
# def get_all_deaths_women(_df):
#     """Return total number of deaths where label='AIDS' and age<15 within the TARGET_PERIOD.
#     df returned: single count value, not grouped by COD.
#     """
#     # Filter for label "AIDS" and age less than 15
#     filtered_df = _df.loc[
#         (pd.to_datetime(_df.date).between(*TARGET_PERIOD)) &
#         (_df['age'] >= 15) &
#         (_df['sex'] == 'F')
#         ]
#
#     # Return the count of relevant deaths
#     return filtered_df.shape[0]  # This gives the total number of deaths
#
#
# all_deaths_women = summarize(
#     extract_results(
#         results_folder,
#         module='tlo.methods.demography',
#         key='death',
#         custom_generate_series=get_all_deaths_women,
#         do_scaling=True
#     ),
#     collapse_columns=False,
#     only_mean=True
# )
#
# # Create an empty dictionary to store dataframes for each draw
# draws_data = {}
#
# # Store each category of death count in a single DataFrame
# draws_data["aids_deaths_children"] = aids_deaths_children
# draws_data["aids_deaths_men"] = aids_deaths_men
# draws_data["aids_deaths_women"] = aids_deaths_women
# draws_data["all_deaths_children"] = all_deaths_children
# draws_data["all_deaths_men"] = all_deaths_men
# draws_data["all_deaths_women"] = all_deaths_women
#
# # Prepare a dictionary for each draw (assuming 'aids_deaths_children', 'aids_deaths_men', etc., have multi-draw output)
# combined_results = {}
#
# # Combine the data for each draw by matching the index of each category
# for draw_number in range(1, len(aids_deaths_children.columns) + 1):
#     draw_data = {}
#     for category, df in draws_data.items():
#         # Select the column for the current draw number (e.g., 1, 2, 3, etc.)
#         draw_data[category] = df[draw_number]
#
#     # Store the data for this draw in the combined_results dictionary
#     combined_results[draw_number] = pd.DataFrame(draw_data)
#
# # Output the combined results into an Excel file with one sheet per draw
# with pd.ExcelWriter('death_counts_by_draw.xlsx') as writer:
#     for draw_number, df in combined_results.items():
#         sheet_name = str(draw_number)  # Use the draw number as the sheet name
#         df.to_excel(writer, sheet_name=sheet_name)

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
mean_values_rounded.index = log0['tlo.methods.hiv']['stock_variables']['date']
df = mean_values_rounded[stock_variables]  # reorder columns
df.to_csv(outputspath / 'MIHPSA_Aug2024/mihpsa_stock_FULL.csv')

flow0 = log0['tlo.methods.hiv']['flow_variables'].iloc[:, 1:]
flow1 = log1['tlo.methods.hiv']['flow_variables'].iloc[:, 1:]
flow2 = log2['tlo.methods.hiv']['flow_variables'].iloc[:, 1:]

mean_values = ((flow0 + flow1 + flow2) / 3) * scaling_factor
mean_values_rounded = mean_values.round().astype(int)
mean_values_rounded.index = log0['tlo.methods.hiv']['stock_variables']['date']
df = mean_values_rounded[flow_variables]  # reorder columns
df.to_csv(outputspath / 'MIHPSA_Aug2024/mihpsa_flow_FULL.csv')


# ------------------------------------------------------------------------------
# %% extract deaths  and scale to population size


def get_child_hiv_deaths(log, scaling_factor):
    deaths = log['tlo.methods.demography']['death']
    deaths['date'] = pd.to_datetime(deaths['date'])
    deaths['year'] = deaths['date'].dt.year

    # Create new column adult=true/false
    deaths['adult'] = deaths['age'] > 14
    deaths['adult'] = deaths['adult'].astype(bool)

    # Filter for HIV deaths
    deaths_hiv = deaths[deaths['label'] == 'AIDS']

    # Get HIV deaths, child
    result_hiv = deaths_hiv[deaths_hiv['adult'] == False].groupby('year').size().reset_index(name='count')
    result_hiv["count"] = (result_hiv["count"] * scaling_factor).astype(int)

    return result_hiv


def mean_child_hiv_deaths(log0, log1, log2, scaling_factor):
    # Process each log file
    df1 = get_child_hiv_deaths(log0, scaling_factor)
    df2 = get_child_hiv_deaths(log1, scaling_factor)
    df3 = get_child_hiv_deaths(log2, scaling_factor)

    # Merge dataframes on 'year' and compute the mean counts
    merged_df = df1.merge(df2, on='year', suffixes=('_1', '_2')).merge(df3, on='year')
    merged_df['mean_count'] = merged_df[['count_1', 'count_2', 'count']].mean(axis=1)

    # Select relevant columns
    mean_counts_df = merged_df[['year', 'mean_count']]

    return mean_counts_df


child_hiv_deaths = mean_child_hiv_deaths(log0, log1, log2, scaling_factor)
child_hiv_deaths = child_hiv_deaths.round().astype(int)
child_hiv_deaths.to_csv(outputspath / 'MIHPSA_Aug2024/mean_child_hiv_deaths.csv')


def get_adult_hiv_deaths(log, scaling_factor):
    deaths = log['tlo.methods.demography']['death']
    deaths['date'] = pd.to_datetime(deaths['date'])
    deaths['year'] = deaths['date'].dt.year

    # Create new column adult=true/false
    deaths['adult'] = deaths['age'] > 14
    deaths['adult'] = deaths['adult'].astype(bool)

    # Filter for HIV deaths
    deaths_hiv = deaths[deaths['label'] == 'AIDS']

    # Get HIV deaths, child
    result_hiv = deaths_hiv[deaths_hiv['adult'] == True].groupby(['year', 'sex']).size().reset_index()
    result_hiv[0] = (result_hiv[0] * scaling_factor).astype(int)

    return result_hiv


def mean_adult_hiv_deaths(log0, log1, log2, scaling_factor):
    # Process each log file
    df1 = get_adult_hiv_deaths(log0, scaling_factor)
    df2 = get_adult_hiv_deaths(log1, scaling_factor)
    df3 = get_adult_hiv_deaths(log2, scaling_factor)

    # Merge dataframes on 'year' and compute the mean counts
    merged_df = df1.merge(df2, on=['year', 'sex'], suffixes=('_1', '_2')).merge(df3, on=['year', 'sex'])
    merged_df['mean_count'] = merged_df[['0_1', '0_2', 0]].mean(axis=1)

    # Select relevant columns
    mean_counts_df = merged_df[['year', 'sex', 'mean_count']]

    return mean_counts_df


adult_hiv_deaths = mean_adult_hiv_deaths(log0, log1, log2, scaling_factor)
adult_hiv_deaths["mean_count"] = adult_hiv_deaths["mean_count"].round().astype(int)
adult_hiv_deaths.to_csv(outputspath / 'MIHPSA_Aug2024/mean_adult_hiv_deaths.csv')


# ALL DEATHS
def get_child_all_deaths(log, scaling_factor):
    deaths = log['tlo.methods.demography']['death']
    deaths['date'] = pd.to_datetime(deaths['date'])
    deaths['year'] = deaths['date'].dt.year

    # Create new column adult=true/false
    deaths['adult'] = deaths['age'] > 14
    deaths['adult'] = deaths['adult'].astype(bool)

    # Get HIV deaths, child
    result = deaths[deaths['adult'] == False].groupby('year').size().reset_index(name='count')
    result["count"] = (result["count"] * scaling_factor).astype(int)

    return result


def mean_child_all_deaths(log0, log1, log2, scaling_factor):
    # Process each log file
    df1 = get_child_all_deaths(log0, scaling_factor)
    df2 = get_child_all_deaths(log1, scaling_factor)
    df3 = get_child_all_deaths(log2, scaling_factor)

    # Merge dataframes on 'year' and compute the mean counts
    merged_df = df1.merge(df2, on='year', suffixes=('_1', '_2')).merge(df3, on='year')
    merged_df['mean_count'] = merged_df[['count_1', 'count_2', 'count']].mean(axis=1)

    # Select relevant columns
    mean_counts_df = merged_df[['year', 'mean_count']]

    return mean_counts_df


child_all_deaths = mean_child_all_deaths(log0, log1, log2, scaling_factor)
child_all_deaths = child_all_deaths.round().astype(int)
child_all_deaths.to_csv(outputspath / 'MIHPSA_Aug2024/mean_child_all_deaths.csv')


def get_adult_all_deaths(log, scaling_factor):
    deaths = log['tlo.methods.demography']['death']
    deaths['date'] = pd.to_datetime(deaths['date'])
    deaths['year'] = deaths['date'].dt.year

    # Create new column adult=true/false
    deaths['adult'] = deaths['age'] > 14
    deaths['adult'] = deaths['adult'].astype(bool)

    # Get HIV deaths, child
    result = deaths[deaths['adult'] == True].groupby(['year', 'sex']).size().reset_index()
    result[0] = (result[0] * scaling_factor).astype(int)

    return result


def mean_adult_all_deaths(log0, log1, log2, scaling_factor):
    # Process each log file
    df1 = get_adult_all_deaths(log0, scaling_factor)
    df2 = get_adult_all_deaths(log1, scaling_factor)
    df3 = get_adult_all_deaths(log2, scaling_factor)

    # Merge dataframes on 'year' and compute the mean counts
    merged_df = df1.merge(df2, on=['year', 'sex'], suffixes=('_1', '_2')).merge(df3, on=['year', 'sex'])
    merged_df['mean_count'] = merged_df[['0_1', '0_2', 0]].mean(axis=1)

    # Select relevant columns
    mean_counts_df = merged_df[['year', 'sex', 'mean_count']]

    return mean_counts_df


adult_all_deaths = mean_adult_all_deaths(log0, log1, log2, scaling_factor)
adult_all_deaths["mean_count"] = adult_all_deaths["mean_count"].round().astype(int)
adult_all_deaths.to_csv(outputspath / 'MIHPSA_Aug2024/mean_adult_all_deaths.csv')


# -----------------------------------------------------------------------------------

def get_hiv_deaths_by_age(log, scaling_factor):
    deaths = log['tlo.methods.demography']['death']
    deaths['date'] = pd.to_datetime(deaths['date'])
    deaths['year'] = deaths['date'].dt.year

    # Filter for HIV deaths
    deaths_hiv = deaths[deaths['label'] == 'AIDS']

    # Get HIV deaths, child
    result = deaths_hiv.groupby(['year', 'age', 'sex']).size().reset_index(name='count')
    result["count"] = (result["count"] * scaling_factor).astype(int)

    return result


def mean_hiv_deaths_all_ages(log0, log1, log2, scaling_factor):
    # Process each log file
    df1 = get_hiv_deaths_by_age(log0, scaling_factor)
    df2 = get_hiv_deaths_by_age(log1, scaling_factor)
    df3 = get_hiv_deaths_by_age(log2, scaling_factor)

    # Generate a complete index for the required years and ages
    years = range(2010, 2051)
    ages = list(range(0, 81)) + ['80+']
    sexes = df1['sex'].unique()

    complete_index = pd.MultiIndex.from_product([years, ages, sexes], names=['year', 'age', 'sex'])

    # Reindex each dataframe to ensure all combinations are present
    df1 = df1.set_index(['year', 'age', 'sex']).reindex(complete_index, fill_value=0).reset_index()
    df2 = df2.set_index(['year', 'age', 'sex']).reindex(complete_index, fill_value=0).reset_index()
    df3 = df3.set_index(['year', 'age', 'sex']).reindex(complete_index, fill_value=0).reset_index()

    # Merge dataframes on 'year' and compute the mean counts
    merged_df = df1.merge(df2, on=['year', 'age', 'sex'], suffixes=('_1', '_2')).merge(df3, on=['year', 'age', 'sex'])
    merged_df['mean_count'] = merged_df[['count_1', 'count_2', 'count']].mean(axis=1)

    # Select relevant columns
    mean_counts_df = merged_df[['year', 'age', 'sex', 'mean_count']]

    return mean_counts_df


hiv_deaths_all_ages = mean_hiv_deaths_all_ages(log0, log1, log2, scaling_factor)
hiv_deaths_all_ages["mean_count"] = hiv_deaths_all_ages["mean_count"].round().astype(int)
hiv_deaths_all_ages2021 = hiv_deaths_all_ages.loc[hiv_deaths_all_ages.year >= 2023]
hiv_deaths_all_ages2021.to_csv(outputspath / 'MIHPSA_Aug2024/hiv_deaths_all_ages.csv')


# -----------------------------------------------------------------------------------

def process_hiv_tests(log, scaling_factor):
    # Convert 'datetime' column to datetime type
    log_tests = log['tlo.methods.hiv']['hiv_test']
    log_tests['date'] = pd.to_datetime(log_tests['date'])

    # Filter for age 15 years and up
    log_tests_filtered = log_tests[log_tests['adult'] == True]

    # Extract year from the datetime column
    log_tests_filtered['year'] = log_tests_filtered['date'].dt.year

    # Group by 'hiv_status' and 'year', then count the number of entries
    result1 = log_tests_filtered.groupby(['hiv_status', 'year']).size().reset_index(name='count')

    # Scale to full population
    result1['count'] = (result1['count'] * scaling_factor).astype(int)

    return result1


def mean_hiv_tests(log0, log1, log2, scaling_factor):
    # Process each log file
    df1 = process_hiv_tests(log0, scaling_factor)
    df2 = process_hiv_tests(log1, scaling_factor)
    df3 = process_hiv_tests(log2, scaling_factor)

    # Merge dataframes on 'hiv_status' and 'year'
    merged_df = df1.merge(df2, on=['hiv_status', 'year'], suffixes=('_1', '_2')).merge(df3, on=['hiv_status', 'year'])

    # Compute the mean of the 'count' columns
    merged_df['mean_count'] = merged_df[['count_1', 'count_2', 'count']].mean(axis=1)

    # Select relevant columns
    mean_counts_df = merged_df[['hiv_status', 'year', 'mean_count']]

    return mean_counts_df


mean_tests_df = mean_hiv_tests(log0, log1, log2, scaling_factor)
mean_tests_df["mean_count"] = mean_tests_df["mean_count"].round().astype(int)
mean_tests_df.to_csv(outputspath / 'MIHPSA_Aug2024/mean_tests_df.csv')
