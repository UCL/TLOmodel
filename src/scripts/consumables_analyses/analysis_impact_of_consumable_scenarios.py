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
import seaborn as sns

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
figurespath = Path(outputspath / 'impact_of_consumable_scenarios')
figurespath.mkdir(parents=True, exist_ok=True) # create directory if it doesn't exist
resourcefilepath = Path("./resources")

# Declare period for which the results will be generated (defined inclusively)

TARGET_PERIOD = (Date(2010, 1, 1), Date(2019, 12, 31))

make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

_, age_grp_lookup = make_age_grp_lookup()

def target_period() -> str:
    """Returns the target period as a string of the form YYYY-YYYY"""
    return "-".join(str(t.year) for t in TARGET_PERIOD)

def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


def create_line_plot_absolute_figure(_df, _plt_var, _index_var, keep_rows, metric, _plt_name):
    pivot_df = _df.pivot_table(
        index=_index_var,
        columns=['draw', 'stat'],
        values=_plt_var
    )
    pivot_df = pivot_df.sort_values(by=(0, 'mean'), ascending=False)
    pivot_df = pivot_df[0:keep_rows]  # Keep only top X conditions

    # Define Scnearios and colours
    scenarios = params['name_of_scenario']  # range(len(params))  # X-axis values representing time periods
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(params['value'])))  # Generate different colors for each bar
    #colors = plt.cm.viridis(np.linspace(0, 1, len(params['value'])))  # Generate different colors for each bar

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get the list of labels for the x-axis
    x_axis_label_names = pivot_df.index

    # Plot each draw with its confidence interval
    for draw in pivot_df.columns.levels[0]:
        central_vals = pivot_df[(draw, 'mean')]
        lower_vals = pivot_df[(draw, 'lower')]
        upper_vals = pivot_df[(draw, 'upper')]

        ax.plot(x_axis_label_names, central_vals, label= scenarios[draw], color = colors[draw])  # TODO update label to name of scenario
        ax.fill_between(x_axis_label_names, lower_vals, upper_vals, alpha=0.3, color = colors[draw])

    # Customize plot
    ax.set_ylabel(f'{metric}')

    # Format y-axis ticks to display in millions
    #formatter = FuncFormatter(lambda x, _: '{:,.0f}'.format(x / 1000000))
    #ax.yaxis.set_major_formatter(formatter)

    #ax.set_title('DALYs by Cause')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    fig.tight_layout()
    fig.savefig(figurespath / _plt_name)
    fig.show()
    plt.close(fig)

def create_line_plot_percentage_of_baseline(_df, _plt_var, _index_var, keep_rows, metric, _plt_name):
    pivot_df = _df.pivot_table(
        index=_index_var,
        columns=['draw', 'stat'],
        values=_plt_var
    )
    pivot_df = pivot_df.sort_values(by=(0, 'mean'), ascending=False)
    pivot_df = pivot_df[0:keep_rows]  # Keep only top X conditions

    # Define Scnearios and colours
    scenarios = params['name_of_scenario']  # range(len(params))  # X-axis values representing time periods
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(params['value'])))  # Generate different colors for each bar

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get the list of labels for the x-axis
    x_axis_label_names = pivot_df.index

    # Plot each draw with its confidence interval
    for draw in pivot_df.columns.levels[0]:
        if scenarios[draw] == 'Actual': # Because this is a comparative graph and 'default' is the baseline for comparison
            pass
        else:
            central_vals = pivot_df[(draw, 'mean')]/pivot_df[(0, 'mean')] * 100# this shows the % reduction in DALYs compared to baseline
            lower_vals = pivot_df[(draw, 'lower')]/pivot_df[(0, 'lower')] * 100
            upper_vals = pivot_df[(draw, 'upper')]/pivot_df[(0, 'upper')] * 100

            ax.plot(x_axis_label_names, central_vals, label= scenarios[draw], color = colors[draw])  # TODO update label to name of scenario
            ax.fill_between(x_axis_label_names, lower_vals, upper_vals, alpha=0.3, color = colors[draw])

    # Customize plot
    ax.set_ylabel(f"{metric} (percentage of 'Actual')")

    # Formatting y-axis as percentages
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))

    #ax.set_title('DALYs by Cause')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    fig.tight_layout()
    fig.savefig(figurespath / _plt_name)
    fig.show()
    plt.close(fig)


# %% Gathering basic information

# Find results_folder associated with a given batch_file and get most recent
#results_folder = get_scenario_outputs('impact_of_consumable_scenarios.py', outputspath)
results_folder = Path(outputspath / 'sakshi.mohan@york.ac.uk/impact_of_consumables_scenarios-2024-06-11T204007Z/')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
params_dict  = {'default': 'Actual', 'scenario1': 'Scenario 1', 'scenario2': 'Scenario 2',
                'scenario3': 'Scenario 3', 'scenario4': 'Scenario 4', 'scenario5': 'Scenario 5',
                'scenario6': 'Scenario 6', 'scenario7': 'Scenario 7', 'scenario8': 'Scenario 8',
                'all': 'Perfect'}
params_dict_df = pd.DataFrame.from_dict(params_dict, orient='index', columns=['name_of_scenario']).reset_index().rename(columns = {'index': 'value'})
params = params.merge(params_dict_df, on = 'value', how = 'left', validate = '1:1')


# %% Extracting results from run

# 1. DALYs accrued
#-----------------------------------------
# 1.1 Total DALYs accrued
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

total_dalys_accrued = summarize(extract_total_dalys(results_folder))
total_dalys_accrued = total_dalys_accrued.unstack()

fig, ax = plt.subplots()

# Arrays to store the values for plotting
central_vals = []
lower_vals = []
upper_vals = []

# Extract values for each parameter
for i, _p in enumerate(params['value']):
    central_val = total_dalys_accrued[(i, 'mean')].values[0]
    lower_val = total_dalys_accrued[(i, 'lower')].values[0]
    upper_val = total_dalys_accrued[(i, 'upper')].values[0]

    central_vals.append(central_val)
    lower_vals.append(lower_val)
    upper_vals.append(upper_val)

# Generate the plot
scenarios = params['name_of_scenario'] #range(len(params))  # X-axis values representing time periods
colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(params['value'])))  # Generate different colors for each bar

for i in range(len(scenarios)):
    ax.bar(scenarios[i], central_vals[i], color=colors[i], label=scenarios[i])
    ax.errorbar(scenarios[i], central_vals[i], yerr=[[central_vals[i] - lower_vals[i]], [upper_vals[i] - central_vals[i]]], fmt='o', color='black')

plt.xticks(scenarios, params['name_of_scenario'], rotation=45)
ax.set_xlabel('Scenarios')
ax.set_ylabel('Total DALYs accrued (in millions)')

# Format y-axis ticks to display in millions
formatter = FuncFormatter(lambda x, _: '{:,.0f}'.format(x / 1000000))
ax.yaxis.set_major_formatter(formatter)

#ax.set_ylim((0, 50))
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig(figurespath / 'main_result_DALYs.png')
fig.show()
plt.close(fig)

# 1.2 DALYs by disease area/intervention - for comparison of the magnitude of impact created by consumables interventions
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
dalys_by_disease_summarized_df = dalys_by_disease_summarized_df.rename(columns = {'level_2': 'cause', 0: 'DALYs'})

create_line_plot_absolute_figure(_df = dalys_by_disease_summarized_df,
                 _plt_var = "DALYs",
                 _index_var = "cause",
                 keep_rows = 10, # keep top 10 causes
                  metric='DALYs accrued',
                 _plt_name = 'DALYs_by_cause.png')

create_line_plot_percentage_of_baseline(_df = dalys_by_disease_summarized_df,
                 _plt_var = "DALYs",
                 _index_var = "cause",
                 keep_rows = 10, # keep top 10 causes
                  metric = 'DALYs accrued',
                 _plt_name = 'DALYs_by_cause_percentage.png')

# 2. Consumable demand not met
#-----------------------------------------
# Number of units of item which were needed but not made available for the top 25 items
# TODO ideally this should count the number of treatment IDs but this needs the detailed health system logger
def consumables_availability_figure(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """ 'Figure 3': Usage of consumables in the HealthSystem"""
    make_graph_file_name = lambda stub: output_folder / f"Fig3_consumables_availability_figure.png"  # noqa: E731

    def get_counts_of_items_requested(_df):
        _df = drop_outside_period(_df)

        counts_of_available = defaultdict(int)
        counts_of_not_available = defaultdict(int)

        for _, row in _df.iterrows():
            for item, num in row['Item_Available'].items():
                counts_of_available[item] += num
            for item, num in row['Item_NotAvailable'].items(): # eval(row['Item_NotAvailable'])
                counts_of_not_available[item] += num

        return pd.concat(
            {'Available': pd.Series(counts_of_available), 'Not_Available': pd.Series(counts_of_not_available)},
            axis=1
        ).fillna(0).astype(int).stack()

    cons_req = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='Consumables',
            custom_generate_series=get_counts_of_items_requested,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True
    )

    cons = cons_req.unstack()
    cons_names = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'consumables' / 'ResourceFile_Consumables_Items_and_Packages.csv'
    )[['Item_Code', 'Items']].set_index('Item_Code').drop_duplicates()
    cons_names.index = cons_names.index.astype(str)
    cons = cons.merge(cons_names, left_index=True, right_index=True, how='left').set_index('Items') #.astype(int)
    cons = cons.assign(total=cons.sum(1)).sort_values('total').drop(columns='total')

    cons.columns = pd.MultiIndex.from_tuples(cons.columns, names=['draw', 'stat', 'var'])
    cons_not_available = cons.loc[:, cons.columns.get_level_values(2) == 'Not_Available']
    cons_not_available.mean = cons_not_available.loc[:, cons_not_available.columns.get_level_values(1) == 'mean']
    cons_available = cons.loc[:, cons.columns.get_level_values(2) == 'Available']

    cons_not_available = cons_not_available.unstack().reset_index()
    cons_not_available = cons_not_available.rename(columns={0: 'qty_not_available'})

    create_line_plot_absolute_figure(_df=cons_not_available,
                     _plt_var="qty_not_available",
                     _index_var="Items",
                     keep_rows=25,  # keep top 25 demanded consumables
                     metric="consumable demand not met",
                     _plt_name='consumables_demand_not_met.png')

    create_line_plot_percentage_of_baseline(_df=cons_not_available,
                     _plt_var="qty_not_available",
                     _index_var="Items",
                     keep_rows=25,   # keep top 25 demanded consumables
                     metric = "consumable demand not met",
                     _plt_name='consumables_demand_not_met_percentage.png')

consumables_availability_figure(results_folder, outputspath, resourcefilepath)

# HSI affected by missing consumables
# We need healthsystem logger for this

# 3. Number of Health System Interactions
#-----------------------------------------
# HSIs taking place by level in the default scenario
def get_counts_of_hsis(_df):
    _df = drop_outside_period(_df)

    # Initialize an empty dictionary to store the total counts
    total_hsi_count = {}

    for date, appointment_dict in _df['Number_By_Appt_Type_Code_And_Level'].items():
        print(appointment_dict)
        for level, appointments_at_level in appointment_dict.items():
            print(level, appointments_at_level)
            total_hsi_count[level] = {}
            for appointment_type, count in appointments_at_level.items():
                print(appointment_type, count)
                if appointment_type in total_hsi_count:
                    total_hsi_count[level][appointment_type] += count
                else:
                    total_hsi_count[level][appointment_type] = count

    total_hsi_count_series = pd.Series(total_hsi_count)
    for level in ['0', '1a', '1b', '2', '3', '4']:
        appointments_at_level = pd.Series(total_hsi_count_series[total_hsi_count_series.index == level].values[0], dtype='int')
        # Create a list of tuples with the original index and the new level '1a'
        new_index_tuples = [(idx, level) for idx in appointments_at_level.index]
        # Create the new MultiIndex
        new_index = pd.MultiIndex.from_tuples(new_index_tuples, names=['Appointment', 'Level'])
        # Reindex the Series with the new MultiIndex
        appointments_at_level_multiindex = appointments_at_level.copy()
        appointments_at_level_multiindex.index = new_index
        if level == '0':
            appointments_all_levels = appointments_at_level_multiindex
        else:
            appointments_all_levels = pd.concat([appointments_all_levels, appointments_at_level_multiindex], axis = 0)

    return pd.Series(appointments_all_levels).fillna(0).astype(int)

hsi_count = summarize(
    extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_counts_of_hsis,
        do_scaling=True
    ),
    only_mean=True,
    collapse_columns=True
)

hsi = hsi_count.assign(baseline_values=hsi_count[(0, 'mean')]).sort_values('baseline_values').drop(columns='baseline_values')
hsi.columns = pd.MultiIndex.from_tuples(hsi.columns, names=['draw', 'stat'])
#hsi = hsi.unstack().reset_index()
hsi_stacked = hsi.stack().stack().reset_index()
hsi_stacked = hsi_stacked.rename(columns={0: 'hsis_requested'})

create_line_plot_absolute_figure(_df=hsi_stacked,
                 _plt_var="hsis_requested",
                 _index_var="Level",
                 keep_rows=6,  # show all levels
                 metric = 'HSIs',
                 _plt_name='hsis_requested.png')

create_line_plot_percentage_of_baseline(_df=hsi_stacked,
                 _plt_var="hsis_requested",
                 _index_var="Level",
                 keep_rows=6,  # show all levels
                 metric = 'HSIs',
                 _plt_name='hsis_requested.png')


# 4. Number of Services delivered
#-----------------------------------------
def get_counts_of_treatments(_df):
    _df = drop_outside_period(_df)

    counts_of_treatments = defaultdict(int)

    for _, row in _df.iterrows():
        for item, num in row['TREATMENT_ID'].items():
            counts_of_treatments[item] += num

    return pd.Series(counts_of_treatments).fillna(0).astype(int)


count_of_treatments_delivered = summarize(
    extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_counts_of_treatments,
        do_scaling=True
    ),
    only_mean=True,
    collapse_columns=True
)
count_of_treatments_delivered = count_of_treatments_delivered.assign(baseline_values=count_of_treatments_delivered[(0, 'mean')]).sort_values('baseline_values').drop(columns='baseline_values')
treatments_delivered = count_of_treatments_delivered.unstack().reset_index()
treatments_delivered = treatments_delivered.rename(columns={'level_2': 'Treatment_ID', 0: 'qty_delivered'})

create_line_plot_absolute_figure(_df=treatments_delivered,
                 _plt_var="qty_delivered",
                 _index_var="Treatment_ID",
                 keep_rows=20,  # show all levels
                 metric = 'Number of HSIs requested by Treatment ID',
                 _plt_name='treatments_delivered.png')

create_line_plot_percentage_of_baseline(_df=treatments_delivered,
                 _plt_var="qty_delivered",
                 _index_var="Treatment_ID",
                 keep_rows=20,  # show all levels
                 metric = 'number of HSIs requested by Treatment ID',
                 _plt_name='treatments_delivered_percentage.png')

# Cost of consumables?

# %% Summarizing input resourcefile data

# 1. Consumable availability by category and level
#--------------------------------------------------
tlo_availability_df = pd.read_csv(resourcefilepath  / 'healthsystem'/ 'consumables' / "ResourceFile_Consumables_availability_small.csv")

# Attach district, facility level, program to this dataset
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}
tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    on = ['Facility_ID'], how='left')
# Attach programs
programs = pd.read_csv(resourcefilepath / 'healthsystem'/ 'consumables' / "ResourceFile_Consumables_availability_and_usage.csv")[['category', 'item_code', 'module_name']]
programs = programs.drop_duplicates('item_code')
tlo_availability_df = tlo_availability_df.merge(programs, on = ['item_code'], how = 'left')

# Generate a heatmap
# Pivot the DataFrame
aggregated_df = tlo_availability_df.groupby(['category', 'Facility_Level'])['available_prop'].mean().reset_index()
heatmap_data = aggregated_df.pivot("category", "Facility_Level", "available_prop")

# Calculate the aggregate row and column
aggregate_col= tlo_availability_df.groupby('Facility_Level')['available_prop'].mean()
aggregate_row = tlo_availability_df.groupby('category')['available_prop'].mean()
overall_aggregate = tlo_availability_df['available_prop'].mean()

# Add aggregate row and column
heatmap_data['Average'] = aggregate_row
aggregate_col['Average'] = overall_aggregate
heatmap_data.loc['Average'] = aggregate_col

# Generate the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', cbar_kws={'label': 'Proportion of days on which consumable is available'})

# Customize the plot
#plt.title('Consumable availability by Facility Level and Category')
plt.xlabel('Facility Level')
plt.ylabel('Category')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.savefig(figurespath /'consumable_availability_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# TODO Justify the focus on levels 1a and 1b - where do HSIs occur?; at what level is there most misallocation within districts
# TODO get graphs of percentage of successful HSIs under different scenarios for levels 1a and 1b
# TODO is there a way to link consumables directly to DALYs (how many DALYs are lost due to stockouts of specific consumables)
# TODO why are there no appointments at level 1b

