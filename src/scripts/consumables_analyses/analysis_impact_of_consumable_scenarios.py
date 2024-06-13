"""This file uses the results of the results of running `impact_of_cons_availability_intervention.py`
tob extract summary results for the manuscript - "Rethinking economic evaluation of
system level interventions.
I plan to run the simulation for a short period of 5 years (2020 - 2025) because
holding the consumable availability constant in the short run would be more justifiable
than holding it constant for a long period.
"""

import argparse
from pathlib import Path
import textwrap
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import Counter, defaultdict


from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    extract_results,
    get_color_cause_of_death_or_daly_label,
    make_age_grp_lookup,
    order_of_cause_of_death_or_daly_label,
    summarize,
)
import pickle

from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    summarize,
    write_log_to_excel,
    parse_log_file,
    COARSE_APPT_TYPE_TO_COLOR_MAP,
    SHORT_TREATMENT_ID_TO_COLOR_MAP,
    _standardize_short_treatment_id,
    bin_hsi_event_details,
    compute_mean_across_runs,
    get_coarse_appt_type,
    get_color_short_treatment_id,
    order_of_short_treatment_ids,
    plot_stacked_bar_chart,
    squarify_neat,
    unflatten_flattened_multi_index_in_logging,
)

outputspath = Path('./outputs/')
resourcefilepath = Path("./resources")

PREFIX_ON_FILENAME = '3'

# Declare period for which the results will be generated (defined inclusively)

TARGET_PERIOD = (Date(2010, 1, 1), Date(2011, 12, 31))

make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

_, age_grp_lookup = make_age_grp_lookup()

def target_period() -> str:
    """Returns the target period as a string of the form YYYY-YYYY"""
    return "-".join(str(t.year) for t in TARGET_PERIOD)

def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


# %% Gathering basic information

# Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('impact_of_consumable_scenarios.py', outputspath)
#results_folder = Path(outputspath/ 'impact_of_consumables_availability_intervention-2023-05-09T210307Z/')
results_folder = Path(outputspath / 'impact_of_consumables_scenarios-2024-06-10T201342Z/')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


# %% Extracting results from run

# 1. DALYs averted
#-----------------------------------------
# 1.1 Difference in total DALYs accrued
def extract_total_dalys(results_folder):

    def extract_dalys_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": df.drop(['date', 'sex', 'age_range', 'year'], axis = 1).sum().sum()})

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=extract_dalys_total,
        do_scaling=True
    )

total_dalys_accrued = extract_total_dalys(results_folder)
dalys_summarized = summarize(total_dalys_accrued)
dalys_summarized = dalys_summarized.unstack()

fig, ax = plt.subplots()

# Arrays to store the values for plotting
central_vals = []
lower_vals = []
upper_vals = []

# Extract values for each parameter
for i, _p in enumerate(params['value']):
    central_val = dalys_summarized[(i, 'mean')].values[0]
    lower_val = dalys_summarized[(i, 'lower')].values[0]
    upper_val = dalys_summarized[(i, 'upper')].values[0]

    central_vals.append(central_val)
    lower_vals.append(lower_val)
    upper_vals.append(upper_val)

# Generate the plot
scenarios = params['value'] #range(len(params))  # X-axis values representing time periods
colors = plt.cm.viridis(np.linspace(0, 1, len(params['value'])))  # Generate different colors for each bar

for i in range(len(scenarios)):
    ax.bar(scenarios[i], central_vals[i], color=colors[i], label=scenarios[i])
    ax.errorbar(scenarios[i], central_vals[i], yerr=[[central_vals[i] - lower_vals[i]], [upper_vals[i] - central_vals[i]]], fmt='o', color='black')

plt.xticks(scenarios, params['value'], rotation=45)
ax.set_xlabel('Scenarios')
ax.set_ylabel('Total DALYs accrued (in millions)')

# Format y-axis ticks to display in millions
formatter = FuncFormatter(lambda x, _: '{:,.0f}'.format(x / 1000000))
ax.yaxis.set_major_formatter(formatter)

#ax.set_ylim((0, 50))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()

# DALYs by disease area/intervention - for comparison of the magnitude of impact created by consumables interventions
def _extract_dalys_by_disease(_df: pd.DataFrame) -> pd.Series:
    """Construct a series with index disease and value of the total of DALYS (stacked) from the
    `dalys_stacked` key logged in `tlo.methods.healthburden`.
    N.B. This limits the time period of interest to 2010-2019"""
    _, calperiodlookup = make_calendar_period_lookup()

    return _df.loc[(_df['year'] >=2009) & (_df['year'] < 2012)]\
             .drop(columns=['date', 'sex', 'age_range', 'year'])\
             .sum(axis=0)

dalys_extracted_by_disease = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked",
    custom_generate_series=_extract_dalys_by_disease,
    do_scaling=True
)

dalys_by_disease_summarized = summarize(dalys_extracted_by_disease)
dalys_by_disease_summarized = dalys_by_disease_summarized.unstack()

# Figure - Focus on top 5 diseases across the 10 scenarios? 0r do a dot plot
# Assuming dalys_by_disease_summarized is your MultiIndex Series
# Convert it to a DataFrame for easier manipulation
dalys_by_disease_summarized_df = dalys_by_disease_summarized.reset_index()
dalys_by_disease_summarized_df = dalys_by_disease_summarized_df.rename(columns = {'level_2': 'disease', 0: 'DALYs'})

# Pivot the DataFrame to get 'draw' as columns, 'disease' as index, and 'DALYs' as values
pivot_df = dalys_by_disease_summarized_df.pivot_table(
    index='disease',
    columns=['draw', 'stat'],
    values='DALYs'
)
pivot_df = pivot_df.sort_values(by=(0, 'mean'), ascending=False)
pivot_df = pivot_df[0:9] # Keep only top 10 conditions

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Get the list of diseases for the x-axis
diseases = pivot_df.index

# Plot each draw with its confidence interval
for draw in pivot_df.columns.levels[0]:
    central_vals = pivot_df[(draw, 'mean')]
    lower_vals = pivot_df[(draw, 'lower')]
    upper_vals = pivot_df[(draw, 'upper')]

    ax.plot(diseases, central_vals, label=f'Draw {draw}') # TODO update label to name of scenario
    ax.fill_between(diseases, lower_vals, upper_vals, alpha=0.3)

# Customize plot
ax.set_xlabel('Cause of DALYs (Top 10)')
ax.set_ylabel('Total DALYs accrued (in millions)')

# Format y-axis ticks to display in millions
formatter = FuncFormatter(lambda x, _: '{:,.0f}'.format(x / 1000000))
ax.yaxis.set_major_formatter(formatter)

ax.set_title('DALYs by Cause')
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

fig.tight_layout()
plt.show()

# TODO update the plot above so that only three scenarions are represented

# 2. Mechanisms of impact
#-----------------------------------------
# Number of units of item which were needed but not made available for the top 25 items
# TODO ideally this should count the number of treatment IDs but this needs the detailed health system logger

# Cost of consumables?

# TODO Justify the focus on levels 1a and 1b - where do HSIs occur?; at what level is there most misallocation within districts
# TODO get graphs of percentage of successful HSIs under different scenarios for levels 1a and 1b

