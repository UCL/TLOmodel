""" use the outputs from scenario_runs.py and produce plots
and summary statistics for paper

JOB ID:
schisto_scenarios-2025-03-22T130153Z
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
    compute_summary_statistics,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    compute_summary_statistics,
    make_age_grp_lookup,
    make_age_grp_types,
    unflatten_flattened_multi_index_in_logging,
)

from scripts.costing.cost_estimation import (estimate_input_cost_of_scenarios,
                                             summarize_cost_data,
                                             do_stacked_bar_plot_of_cost_by_category,
                                             do_line_plot_of_cost,
                                             create_summary_treemap_by_cost_subgroup,
                                             estimate_projected_health_spending)

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

# %% FUNCTIONS ##################################################################
TARGET_PERIOD = (Date(2024, 1, 1), Date(2040, 12, 31))


def get_district_prevalence(_df):
    """Get the prevalence for every district """
    df = _df.copy()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to datetime
        df = df.set_index('date')  # Set date as index
    else:
        # If 'date' is already index, ensure it's a datetime type
        df.index = pd.to_datetime(df.index)

    # _df.index = pd.to_datetime(_df.index)
    _df_year = df[df.index.year == year]  # Filter by year

    _df_year.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split('|')) for col in _df_year.columns],
        names=['infection_status', 'district_of_residence', 'age_years']
    )

    # limit to relevant age-groups
    if age_group == 'SAC':
        age_group_filter = ['SAC']
    elif age_group == 'PSAC':
        age_group_filter = ['PSAC', 'SAC']  # Include both PSAC and SAC
    elif age_group == 'Adult':
        age_group_filter = ['Adults']
    elif age_group == 'Infant':
        age_group_filter = ['Infant']
    else:
        age_group_filter = ['Adults', 'Infant', 'PSAC', 'SAC']  # Include all age groups for 'All'

    selected_columns = [
        col for col in _df_year.columns
        if any(age in col[2] for age in age_group_filter)
    ]
    df_total = _df_year[selected_columns]
    district_sums = df_total.groupby(axis=1, level='district_of_residence').sum()

    # Set infection status filter
    infection_filter = []
    if 'High-infection' in infection_types:
        infection_filter.append('High-infection')
    if 'Moderate-infection' in infection_types:
        infection_filter.append('Moderate-infection')
    if 'Low-infection' in infection_types:
        infection_filter.append('Low-infection')

    selected_columns = [
        col for col in _df_year.columns
        if any(inf_status in col[0] for inf_status in infection_filter) and
           any(age in col[2] for age in age_group_filter)
    ]
    df_infected = _df_year[selected_columns]
    infected_numerator = df_infected.groupby('district_of_residence', axis=1).sum()

    proportion_infected = infected_numerator.div(district_sums).iloc[0]

    return proportion_infected


def extract_district_prevalence() -> pd.DataFrame:
    """ for each run/draw combination, extract the prevalence by district
    using the custom arguments for age-group and infection status
    """

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)
    module = 'tlo.methods.schisto'

    # Collect results from each draw/run
    res = dict()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            draw_run = (draw, run)

            try:
                _df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                output_from_eval: pd.Series = get_district_prevalence(_df)
                assert isinstance(output_from_eval, pd.Series), (
                    'Custom command does not generate a pd.Series'
                )

                res[draw_run] = output_from_eval

            except KeyError:
                # Some logs could not be found - probably because this run failed.
                res[draw_run] = None

    # Use pd.concat to compile results (skips dict items where the values is None)
    _concat = pd.concat(res, axis=1)
    _concat.columns.names = ['draw', 'run']  # name the levels of the columns multi-index
    return _concat


key = "infection_status_haematobium"
year = 2023
infection_types = ['High-infection', 'Moderate-infection', 'Low-infection']
age_group = 'All'

tmp = extract_district_prevalence()
median_district_prev2023 = tmp.groupby('draw', axis=1).median()

year = 2040
tmp = extract_district_prevalence()
median_district_prev2040 = tmp.groupby('draw', axis=1).median()

# get percentage change in schisto prevalence for each district
percentage_change = ((median_district_prev2040 - median_district_prev2023) / median_district_prev2023) * 100
percentage_change.index = percentage_change.index.str.replace('district_of_residence=', '', case=False)

#### PLOT percentage change in prevalence
# Plotting the heatmap with color coding by column (Scenario)
plt.figure(figsize=(12, 8))  # Increase figure size to accommodate 32 rows
ax = sns.heatmap(percentage_change, xticklabels=param_names, yticklabels=percentage_change.index,
                 annot=False, cmap='coolwarm', linewidths=0.5,
                 cbar_kws={'label': 'Percentage Change (%)'})

# Customizing the plot
plt.title("Percentage Change in Prevalence (2023 to 2040)", fontsize=16)
plt.ylabel('', fontsize=14)
plt.xlabel('', fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()


file_path = results_folder / f'prevalence_HML_All_haematobium {target_period()}.xlsx'

# Create an Excel writer object
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    # Write each DataFrame to a different sheet
    median_district_prev2023.to_excel(writer, sheet_name='Prev_2023', index=False)
    median_district_prev2023.to_excel(writer, sheet_name='Prev_2040', index=False)
    percentage_change.to_excel(writer, sheet_name='percentage_change', index=False)







