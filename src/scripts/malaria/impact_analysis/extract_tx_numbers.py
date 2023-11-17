"""
Read in the output files generated by analysis_scenarios and plot outcomes for comparison
"""

import datetime
from pathlib import Path
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tlo import Date

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
)

# outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("effect_of_treatment_packages_combined.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# Create a list of strings summarizing the parameter values in the different draws
scenario_names = ['baseline', '-hiv', 'tb', '-malaria', '-all3']

# -----------------------------------------------------------------------------------------
# %% HS usage
# -----------------------------------------------------------------------------------------

# fraction of HCW time

hs_capacity = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="Capacity",
        column="average_Frac_Time_Used_Overall",
        index="date",
        do_scaling=False
    ),
    only_mean=True, collapse_columns=False
)


# ---------------------------------- PERSON-YEARS ---------------------------------- #
# for each scenario, return a df with the person-years logged in each draw/run
# to be used for calculating tb incidence or mortality rates


def get_person_years(_df):
    """ extract person-years for each draw/run
    sums across men and women
    will skip column if particular run has failed
    """
    years = pd.to_datetime(_df["date"]).dt.year
    py = pd.Series(dtype="int64", index=years)
    for year in years:
        tot_py = (
            (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["M"]).apply(pd.Series) +
            (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["F"]).apply(pd.Series)
        ).transpose()
        py[year] = tot_py.sum().values[0]

    py.index = pd.to_datetime(years, format="%Y")

    return py


py0 = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=False
    ),
    only_mean=True, collapse_columns=False
)

# scale HS capacity for person-years
# note py logged at start of yr, capacity logged at end of yr
py0.index = pd.to_datetime(py0.index, format='%Y-%m-%d').year
hs_capacity.index = pd.to_datetime(hs_capacity.index, format='%Y-%m-%d').year

scaled_hs_capacity = hs_capacity.divide(py0)

# ---------------------------------------------------------------------------------
# TREATMENT COUNTS
# ---------------------------------------------------------------------------------

years_of_simulation = 10


def summarise_appt_outputs(df_list, treatment_id):
    """ summarise the treatment counts across all draws/runs for one results folder
        requires a list of dataframes with all treatments listed with associated counts
    """
    number_runs = len(df_list)
    number_HSI_by_run = pd.DataFrame(index=np.arange(years_of_simulation), columns=np.arange(number_runs))
    column_names = [
        treatment_id + "_mean",
        treatment_id + "_lower",
        treatment_id + "_upper"]
    out = pd.DataFrame(columns=column_names)

    for i in range(number_runs):
        if treatment_id in df_list[i].columns:
            number_HSI_by_run.iloc[:, i] = pd.Series(df_list[i].loc[:, treatment_id])

    out.iloc[:, 0] = number_HSI_by_run.quantile(q=0.5, axis=1)
    out.iloc[:, 1] = number_HSI_by_run.quantile(q=0.025, axis=1)
    out.iloc[:, 2] = number_HSI_by_run.quantile(q=0.975, axis=1)

    return out


def sum_appt_by_id(results_folder, module, key, column, draw):
    """
    sum occurrences of each treatment_id over the simulation period for every run within a draw
    """

    info = get_scenario_info(results_folder)
    # create emtpy dataframe
    results = pd.DataFrame()

    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

        new = df[['date', column]].copy()
        tmp = pd.DataFrame(new[column].to_list())

        # sum each column to get total appts of each type over the simulation
        tmp2 = pd.DataFrame(tmp.sum())
        # add results to dataframe for output
        results = pd.concat([results, tmp2], axis=1)

    return results


def extract_appt_details(results_folder, module, key, column, draw):
    """
    extract list of dataframes with all treatments listed with associated counts
    """

    info = get_scenario_info(results_folder)

    df_list = list()

    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

        new = df[['date', column]].copy()
        df_list.append(pd.DataFrame(new[column].to_list()))

    # for column in each df, get median
    # list of treatment IDs
    list_tx_id = list(df_list[0].columns)
    results = pd.DataFrame(index=np.arange(years_of_simulation))

    # produce a list of numbers of every treatment_id
    for treatment_id in list_tx_id:
        tmp = summarise_appt_outputs(df_list, treatment_id)

        # append output to dataframe
        results = results.join(tmp)

    return results


scaling_factor = extract_results(
    results_folder,
    module="tlo.methods.population",
    key="scaling_factor",
    column="scaling_factor",
    index="date",
    do_scaling=False)


def summarise_grouped_appts(results_folder, module, key, column, draw):
    """
    extract list of dataframes with all treatments listed
    group treatment_id by stub
    then summarise
    keep firstattendance_emergency and non-emergency separate

    produces dataframe: rows=treatment_id_stub e.g. Hiv, Alri
    columns = number of appts for each run, median, lower and upper UIs
    """

    info = get_scenario_info(results_folder)

    # df_list = list()
    out = pd.DataFrame()

    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

        new = df[['date', column]].copy()
        # split the tx_id lists into separate columns ready for summarising
        new_df = pd.DataFrame(new[column].to_list())

        # get columns sums for each treatment_id
        out[run] = new_df.sum()

    # with treatment numbers for each run, group by prefix and sum
    grouped_df = out.groupby(out.index.str.split('_').str[0]).sum()
    # add back the first attendance appts separately
    grouped_df = grouped_df.append(out.loc['FirstAttendance_Emergency'])
    grouped_df = grouped_df.append(out.loc['FirstAttendance_NonEmergency'])
    grouped_df['median'] = grouped_df.iloc[:, 0:5].quantile(q=0.5, axis=1) * scaling_factor.values[0][0]
    grouped_df['lower'] = grouped_df.iloc[:, 0:5].quantile(q=0.025, axis=1) * scaling_factor.values[0][0]
    grouped_df['upper'] = grouped_df.iloc[:, 0:5].quantile(q=0.975, axis=1) * scaling_factor.values[0][0]

    return grouped_df


# extract numbers of appts
module = "tlo.methods.healthsystem.summary"
key = 'HSI_Event'
column = 'TREATMENT_ID'

# this returns every appt type with mean and lower/upper bounds
# todo this is not scaled
treatment_id0 = extract_appt_details(results_folder,
                                     module=module, key=key, column=column, draw=0)
treatment_id1 = extract_appt_details(results_folder,
                                     module=module, key=key, column=column, draw=1)
treatment_id2 = extract_appt_details(results_folder,
                                     module=module, key=key, column=column, draw=2)
treatment_id3 = extract_appt_details(results_folder,
                                     module=module, key=key, column=column, draw=3)
treatment_id4 = extract_appt_details(results_folder,
                                     module=module, key=key, column=column, draw=4)
treatment_id5 = extract_appt_details(results_folder,
                                     module=module, key=key, column=column, draw=5)

# get total counts of every appt type for each scenario
sum0 = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=0)
sum0['mean'] = sum0.mean(axis=1) * scaling_factor.values[0][0]

sum1 = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=1)
sum1['mean'] = sum1.mean(axis=1) * scaling_factor.values[0][0]

sum2 = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=2)
sum2['mean'] = sum2.mean(axis=1) * scaling_factor.values[0][0]

sum3 = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=3)
sum3['mean'] = sum3.mean(axis=1) * scaling_factor.values[0][0]

sum4 = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=4)
sum4['mean'] = sum4.mean(axis=1) * scaling_factor.values[0][0]

sum5 = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=5)
sum5['mean'] = sum5.mean(axis=1) * scaling_factor.values[0][0]


data_output = pd.concat([sum0['mean'], sum1['mean'], sum2['mean'], sum3['mean'], sum4['mean'],
                         sum5['mean']], axis=1)
data_output.to_csv(outputspath / ('treatment_numbers3' + '.csv'))

sum0.to_csv(outputspath / ('baseline_appt_numbers' + '.csv'))

# extract numbers of appts grouped by treatment_id stub
# median is taken across runs for grouped numbers of appts
# this is scaled
sum_tx0 = summarise_grouped_appts(results_folder,
                                  module=module, key=key, column=column, draw=0)

sum_tx4 = summarise_grouped_appts(results_folder,
                                  module=module, key=key, column=column, draw=4)

sum_tx5 = summarise_grouped_appts(results_folder,
                                  module=module, key=key, column=column, draw=5)


# summary table for output
# todo note these are scaled
output_table = pd.DataFrame({
    'baseline_median': sum_tx0['median'],
    'baseline_lower': sum_tx0['lower'],
    'baseline_upper': sum_tx0['upper'],
    'sc4_median': sum_tx4['median'],
    'sc4_lower': sum_tx4['lower'],
    'sc4_upper': sum_tx4['upper'],
    'sc5_median': sum_tx5['median'],
    'sc5_lower': sum_tx5['lower'],
    'sc5_upper': sum_tx5['upper']
})


# Define a custom rounding function
def round_to_nearest_100(x):
    return round(x, -2)


output_table = output_table.applymap(round_to_nearest_100)
# Convert all values to integers
output_table = output_table.astype(int, errors='ignore')

output_table.to_csv(outputspath / ('tx_id_numbers' + '.csv'))


# ---------------------------------------------------------------------------------
# HOW MUCH HS REQUIRED FOR BIG 3 PROGRAMME DELIVERY
# ---------------------------------------------------------------------------------

# summary table of numbers of appts required for hiv, tb and malaria
# need to be scaled for full pop
# note: assume HS use linearly scales with pop size

# comparison is baseline vs scenario 4
# select all appts starting with hiv, tb or malaria
module = "tlo.methods.healthsystem.summary"
key = 'HSI_Event'
column = 'TREATMENT_ID'

# treatment_id0 gives all appts with mean, lower and upper bounds
# todo this is not scaled

prog_appts = treatment_id0.filter(like='Hiv').columns | treatment_id0.filter(like='Malaria').columns | treatment_id0.filter(
    like='Tb').columns
prog_df = treatment_id0[prog_appts]

# add in column totals
prog_df.loc['Total'] = prog_df.sum()

# total numbers of OPD appts




# horizontal barplot

# Filter columns that end with 'mean'
mean_columns = [col for col in prog_df.columns if col.endswith('mean')]
stripped_mean_column_names = [col.replace('_mean', '') for col in mean_columns]

# Create a horizontal bar plot
# Create a color list based on column name criteria
colours = ['c' if col.startswith('Hiv') else 'm' if col.startswith('Malaria') else 'y' for col in mean_columns]

plt.figure(figsize=(10, 7))
plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
plt.barh(stripped_mean_column_names, prog_df[mean_columns].iloc[10] * scaling_factor.values[0][0], color=colours)
plt.xscale('log')

# Add labels and title
plt.xlabel('Values')
plt.ylabel('')
plt.yticks(fontsize=10)
plt.title('')

plt.savefig(outputspath / "baseline_appt_numbers.png")

# Show the plot
plt.show()




# ---------------------------------------------------------------------------------
# SQUEEZE FACTORS
# ---------------------------------------------------------------------------------


# extract mean squeeze factors for each draw, keep only the final yr for hiv, tb and malaria
def summarise_squeeze_factors(results_folder):
    # extract squeeze factors
    module = "tlo.methods.healthsystem.summary"
    key = 'HSI_Event'
    column = 'squeeze_factor'

    df = pd.DataFrame()

    for draw in range(scenario_info["number_of_draws"]):
        # extract the squeeze factors
        sf = extract_appt_details(results_folder, module=module, key=key, column=column, draw=draw)
        sf_mean = sf.loc[:, sf.columns.str.endswith('mean')]
        # List columns starting with 'Hiv,' 'Tb,' or 'malaria'
        select_sf = [col for col in sf_mean.columns if col.startswith(('Hiv', 'Tb', 'Malaria'))]
        # Keep only row 9 (0-indexed)
        result = sf.loc[9, select_sf]

        # append each column to dataframe
        df[draw] = result

    # return summary dataframe
    return df


squeeze_factors = summarise_squeeze_factors(results_folder)
