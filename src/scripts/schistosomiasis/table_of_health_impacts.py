""" use the outputs from scenario_runs.py and produce plots
and summary statistics for paper
"""

from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import pandas as pd
# import lacroix
import matplotlib.colors as colors
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from collections import defaultdict
import textwrap
from typing import Tuple

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
    unflatten_flattened_multi_index_in_logging,
)

resourcefilepath = Path("./resources")

output_folder = Path("./outputs/t.mangal@imperial.ac.uk")

results_folder = get_scenario_outputs("schisto_scenarios.py", output_folder)[-1]

# Declare path for output graphs from this script
def make_graph_file_name(name):
    return results_folder / f"Schisto_{name}.png"


# Name of species that being considered:
species = ('mansoni', 'haematobium')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


def get_parameter_names_from_scenario_file() -> Tuple[str]:
    """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
    from scripts.schistosomiasis.scenario_runs import (
        SchistoScenarios,
    )
    e = SchistoScenarios()
    return tuple(e._scenarios.keys())

param_names = get_parameter_names_from_scenario_file()


def set_param_names_as_column_index_level_0(_df):
    """Set the columns index (level 0) as the param_names."""
    ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


# %% FUNCTIONS ##################################################################
TARGET_PERIOD = (Date(2025, 1, 1), Date(2035, 12, 31))


# create table (one per scenario)
# todo
# extract DALYs per district (all cause)
# extract amount PZQ used per district - could do this using scenario set-up
# use the population logged in sub_group by district and map to scenario and coverage


# todo edit this, want columns=district, columns=age-groupPSAC etc
# todo for each draw

def find_number_in_subgroup(df, param_names):
    """
    Calculate the number of people in specified subgroups by district from the given DataFrame.

    Parameters:
    df (pd.DataFrame): A DataFrame with the first column as date and the second column as a dictionary
                       with district and age group as keys.
    param_names (tuple): A tuple of parameter names that dictate which age groups to include in the count.

    Returns:
    pd.Series: A Series with total counts for each age group across all dates.
    """
    # Initialize a dictionary to hold cumulative counts
    cumulative_counts = {
        'number_PSAC': 0,
        'number_SAC': 0,
        'number_Adults': 0,
        'number_All': 0
    }

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        district_data = row[1]  # Assuming the second column is the dictionary with district data

        # Initialize counts for the current row
        counts = {
            'PSAC': 0,
            'SAC': 0,
            'Adults': 0,
            'All': 0
        }

        # Iterate over the keys and values in the district data
        for key, value in district_data.items():
            # Extract the age group name from the key
            age_group = key.split('|')[1].split('=')[1]  # Extract age group name

            # Update counts based on the age group and param_names
            if 'SAC' in param_names and 'SAC' in age_group:
                counts['SAC'] += value
            if 'PSAC' in param_names and (age_group == 'PSAC' or age_group == 'SAC'):
                counts['PSAC'] += value
            if 'Adults' in param_names and age_group == 'Adults':
                counts['Adults'] += value
            counts['All'] += value  # Always include in total

        # Aggregate the counts for all rows
        cumulative_counts['number_PSAC'] += counts['PSAC']
        cumulative_counts['number_SAC'] += counts['SAC']
        cumulative_counts['number_Adults'] += counts['Adults']
        cumulative_counts['number_All'] += counts['All']

    # Convert cumulative counts to a pd.Series
    return pd.Series(cumulative_counts)



# Define the parameter names
param_names = (
    'Baseline',
    'MDA SAC with no WASH',
    'MDA SAC with WASH',
    'MDA PSAC with no WASH',
    'MDA PSAC with WASH',
    'MDA All with no WASH',
    'MDA All with WASH'
)

# Call the function and get the counts
subgroup_counts_series = find_number_in_subgroup(df, param_names)

# Display the result
print(subgroup_counts_series)


pzq_needed = extract_results(
    results_folder,
    module='tlo.methods.schisto',
    key='number_in_subgroup',
    custom_generate_series=find_number_in_subgroup,
    do_scaling=False
).pipe(set_param_names_as_column_index_level_0)
pzq_needed_summary = summarize(pzq_needed, only_mean=True)






def get_PZQ_required():
    """ estimate the amount of PZQ required for the MDA programme
    using the scenario coverage, frequency of rounds and
    number of people in eligible subgroups by district
    """


def extract_prevalence_all_species(number_infected: pd.DataFrame, number_subgroup: pd.DataFrame) -> pd.DataFrame:
    """ produce a dataframe of prevalence of any species for each month of each year (rows)
    by district (columns)
    """

    # Extract the first column as 'date'
    dates = number_infected.iloc[:, 0]

    # Initialize an empty dataframe for storing the results
    result_df = pd.DataFrame({'date': dates})

    # Loop through each row (date) in the two dataframes
    for i in range(len(number_infected)):
        infected_row = number_infected.iloc[i, 1]
        subgroup_row = number_subgroup.iloc[i, 1]

        # Sum values for each district across age groups in both dataframes
        district_sums_infected = {}
        district_sums_subgroup = {}

        for key, value in infected_row.items():
            district = key.split('|')[0].split('=')[1]
            if district not in district_sums_infected:
                district_sums_infected[district] = 0
            district_sums_infected[district] += value

        for key, value in subgroup_row.items():
            district = key.split('|')[0].split('=')[1]
            if district not in district_sums_subgroup:
                district_sums_subgroup[district] = 0
            district_sums_subgroup[district] += value

        # Compute the ratio for each district and store it
        for district in district_sums_infected:
            if district not in result_df.columns:
                result_df[district] = None
            infected_sum = district_sums_infected[district]
            subgroup_sum = district_sums_subgroup[district] if district in district_sums_subgroup else 1
            result_df.loc[i, district] = infected_sum / subgroup_sum if subgroup_sum != 0 else None

    # Filter for the last entry in 2024 and 2035
    last_2024 = result_df[result_df['date'].dt.year == 2024].iloc[-1]
    last_2035 = result_df[result_df['date'].dt.year == 2035].iloc[-1]

    # Create a new dataframe with districts as rows and values for 2024 and 2035
    district_columns = result_df.columns.drop('date')
    output_df = pd.DataFrame({'district': district_columns,
                              'prevalence2024': last_2024[district_columns].values,
                              'prevalence2035': last_2035[district_columns].values})

    return output_df


def extract_prevalence_by_run(results_folder, scenario_info, module, output_file):

    # Create an Excel writer object
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        for draw in range(scenario_info['draws_per_draw']):
            summary_list = []

            # For each run, get the dataframe of prevalence by district, then summarize and return
            for run in range(scenario_info['runs_per_draw']):
                # Load the number_infected and number_subgroup dataframes
                number_infected = load_pickled_dataframes(results_folder, draw, run, module)[module][
                    'number_infected_any_species']
                number_subgroup = load_pickled_dataframes(results_folder, draw, run, module)[module][
                    'number_in_subgroup']

                # Extract the 2024 and 2035 values for the current run
                tmp_df = extract_prevalence_all_species(number_infected, number_subgroup)

                # Add the resulting dataframe to the list of run summaries
                summary_list.append(tmp_df)

            # Concatenate all summaries into one dataframe
            all_runs_df = pd.concat(summary_list, ignore_index=True)

            # Group by district and calculate the median for prevalence2024 and prevalence2035
            output_df = all_runs_df.groupby('district').median().reset_index()

            # Write output_df to a sheet in the workbook
            sheet_name = param_names[draw]
            output_df.to_excel(writer, sheet_name=sheet_name, index=False)


extract_prevalence_by_run(results_folder, scenario_info, module='tlo.methods.schisto',
                          output_file=results_folder / 'table_of_prevalence.xlsx')
