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
import squarify

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

def do_bar_plot_with_ci(_df, annotations=None, xticklabels_horizontal_and_wrapped=False):
    """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
     extent of the error bar."""
    yerr = np.array([
        (_df['mean'] - _df['lower']).values,
        (_df['upper'] - _df['mean']).values,
    ])

    xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(params['value'])))  # Generate different colors for each bar

    fig, ax = plt.subplots()
    ax.bar(
        xticks.keys(),
        _df['mean'].values,
        yerr=yerr,
        alpha=1,
        color = colors,
        ecolor='black',
        capsize=10,
        label=xticks.values()
    )
    if annotations:
        for xpos, ypos, text in zip(xticks.keys(), _df['upper'].values, annotations):
            ax.text(xpos, ypos * 1.05, text, horizontalalignment='center', fontsize = 9)
    ax.set_xticks(list(xticks.keys()))
    if not xticklabels_horizontal_and_wrapped:
        # xticklabels will be vertical and not wrapped
        ax.set_xticklabels(list(xticks.values()), rotation=90)
    else:
        wrapped_labs = ["\n".join(textwrap.wrap(_lab, 20)) for _lab in xticks.values()]
        ax.set_xticklabels(wrapped_labs)
    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig, ax

def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = [i.year for i in TARGET_PERIOD]
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

def get_num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = [i.year for i in TARGET_PERIOD]
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum()
    )

def find_difference_relative_to_comparison(_ser: pd.Series,
                                           comparison: str,
                                           scaled: bool = False,
                                           drop_comparison: bool = True,
                                           ):
    """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
    within the runs (level 1), relative to where draw = `comparison`.
    The comparison is `X - COMPARISON`."""
    return _ser \
        .unstack(level=0) \
        .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
        .drop(columns=([comparison] if drop_comparison else [])) \
        .stack()

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
scenarios = params['name_of_scenario'] #range(len(params))  # X-axis values representing time periods
drop_scenarios = ['Scenario 4', 'Scenario 5'] # Drops scenarios which are no longer considered important for comparison

# %% Extracting results from run

# 1. DALYs accrued and averted
#-----------------------------------------
# 1.1 Total DALYs accrued
# Get total DALYs accrued
num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=True
    )

# %% Chart of total number of DALYS
num_dalys_summarized = summarize(num_dalys).loc[0].unstack()
num_dalys_summarized['scenario'] = scenarios.to_list()
num_dalys_summarized = num_dalys_summarized.set_index('scenario')

# Plot DALYS accrued (with xtickabels horizontal and wrapped)
name_of_plot = f'Total DALYs accrued, {target_period()}'
chosen_num_dalys_summarized = num_dalys_summarized[~num_dalys_summarized.index.isin(drop_scenarios)]
fig, ax = do_bar_plot_with_ci(
    (chosen_num_dalys_summarized / 1e6).clip(lower=0.0),
    annotations=[
        f"{round(row['mean']/1e6, 1)} \n ({round(row['lower']/1e6, 1)}-{round(row['upper']/1e6, 1)})"
        for _, row in chosen_num_dalys_summarized.clip(lower=0.0).iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
)
ax.set_title(name_of_plot)
ax.set_ylim(0, 120)
ax.set_yticks(np.arange(0, 120, 10))
ax.set_ylabel('Total DALYs accrued \n(Millions)')
fig.tight_layout()
fig.savefig(figurespath / name_of_plot.replace(' ', '_').replace(',', ''))
fig.show()
plt.close(fig)

# 1.2 Total DALYs averted
# Get absolute DALYs averted
num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison= 0) # sets the comparator to 0 which is the Status Quo scenario
        ).T
    ).iloc[0].unstack()
num_dalys_averted['scenario'] = scenarios.to_list()[1:10]
num_dalys_averted = num_dalys_averted.set_index('scenario')

# Get percentage DALYs averted
pc_dalys_averted = 100.0 * summarize(
    -1.0 *
    pd.DataFrame(
        find_difference_relative_to_comparison(
            num_dalys.loc[0],
            comparison= 0, # sets the comparator to 0 which is the Status Quo scenario
            scaled=True)
    ).T
).iloc[0].unstack()
pc_dalys_averted['scenario'] = scenarios.to_list()[1:10]
pc_dalys_averted = pc_dalys_averted.set_index('scenario')

# %% Chart of number of DALYs averted
# Plot DALYS averted (with xtickabels horizontal and wrapped)
name_of_plot = f'Additional DALYs Averted vs Actual, {target_period()}'
chosen_num_dalys_averted = num_dalys_averted[~num_dalys_averted.index.isin(drop_scenarios)]
chosen_pc_dalys_averted = pc_dalys_averted[~pc_dalys_averted.index.isin(drop_scenarios)]
fig, ax = do_bar_plot_with_ci(
    (chosen_num_dalys_averted / 1e6).clip(lower=0.0),
    annotations=[
        f"{round(row['mean'], 1)} % \n ({round(row['lower'], 1)}-{round(row['upper'], 1)}) %"
        for _, row in chosen_pc_dalys_averted.clip(lower=0.0).iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
)
ax.set_title(name_of_plot)
ax.set_ylim(0, 16)
ax.set_yticks(np.arange(0, 18, 2))
ax.set_ylabel('Additional DALYS Averted \n(Millions)')
fig.tight_layout()
fig.savefig(figurespath / name_of_plot.replace(' ', '_').replace(',', ''))
fig.show()
plt.close(fig)

# 1.2 DALYs by disease area/intervention - for comparison of the magnitude of impact created by consumables interventions
num_dalys_by_cause = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys_by_cause,
        do_scaling=True
    )
num_dalys_by_cause_summarized = summarize(num_dalys_by_cause).unstack(level = 0)
num_dalys_by_cause_summarized = num_dalys_by_cause_summarized.reset_index()
num_dalys_by_cause_summarized = num_dalys_by_cause_summarized.rename(columns = {'level_2':'cause', 0: 'DALYs_accrued'})
num_dalys_by_cause_summarized = num_dalys_by_cause_summarized.pivot(index=['draw','cause'], columns='stat', values='DALYs_accrued')

# Get top 10 causes until status quo
num_dalys_by_cause_status_quo = num_dalys_by_cause_summarized[num_dalys_by_cause_summarized.index.get_level_values(0) == 0]
num_dalys_by_cause_status_quo = num_dalys_by_cause_status_quo.sort_values('mean', ascending = False)
num_dalys_by_cause_status_quo =num_dalys_by_cause_status_quo[0:10]

for cause in num_dalys_by_cause_status_quo.index.get_level_values(1).unique():
    name_of_plot = f'Total DALYs accrued by {cause}, {target_period()}'
    chosen_num_dalys_by_cause_summarized = num_dalys_by_cause_summarized[~num_dalys_by_cause_summarized.index.get_level_values(0).isin([4,5])]
    chosen_num_dalys_by_cause_summarized = chosen_num_dalys_by_cause_summarized[chosen_num_dalys_by_cause_summarized.index.get_level_values(1) == cause]
    fig, ax = do_bar_plot_with_ci(
        (chosen_num_dalys_by_cause_summarized / 1e6).clip(lower=0.0),
        annotations=[
            f"{round(row['mean'] / 1e6, 1)} \n ({round(row['lower'] / 1e6, 1)}-{round(row['upper'] / 1e6, 1)})"
            for _, row in chosen_num_dalys_by_cause_summarized.clip(lower=0.0).iterrows()
        ],
        xticklabels_horizontal_and_wrapped=False,
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 30)
    ax.set_yticks(np.arange(0, 30, 5))
    ax.set_ylabel(f'Total DALYs accrued by {cause} \n(Millions)')
    fig.tight_layout()
    fig.savefig(figurespath / name_of_plot.replace(' ', '_').replace(',', '').replace('/', '_'))
    fig.show()
    plt.close(fig)

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

for disease in ['AIDS', 'Lower respiratory infections', 'Neonatal Disorders', 'Malaria', 'TB (non-AIDS)']:
    dalys_accrued = dalys_by_disease_summarized.xs(disease, level=2)
    fig, ax = plt.subplots()

    # Arrays to store the values for plotting
    central_vals = []
    lower_vals = []
    upper_vals = []

    # Extract values for each parameter
    for i, _p in enumerate(params['value']):
        central_val = dalys_accrued[(i, 'mean')]
        lower_val = dalys_accrued[(i, 'lower')]
        upper_val = dalys_accrued[(i, 'upper')]

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
    ax.set_title(disease)

    # Format y-axis ticks to display in millions
    formatter = FuncFormatter(lambda x, _: '{:,.0f}'.format(x / 1000000))
    ax.yaxis.set_major_formatter(formatter)

    #ax.set_ylim((0, 50))
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(figurespath / f'main_result_DALYs_{disease}.png')
    fig.show()
    plt.close(fig)


# Figure - Focus on top 5 diseases across the 10 scenarios? 0r do a dot plot
# Assuming dalys_by_disease_summarized is your MultiIndex Series
# Convert it to a DataFrame for easier manipulation
dalys_by_disease_summarized_df = dalys_by_disease_summarized.reset_index()
dalys_by_disease_summarized_df = dalys_by_disease_summarized_df.rename(columns = {'level_2': 'cause', 0: 'DALYs'})

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

consumables_availability_figure(results_folder, outputspath, resourcefilepath)

# TODO use squarify_plot to represent which consumables are most used in the system (by short Treatment_ID?) (not quantity but frequency)

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


# 4.1 Number of Services delivered by long Treatment_ID
#------------------------------------------------------
def get_counts_of_hsi_by_treatment_id(_df):
    """Get the counts of the short TREATMENT_IDs occurring"""
    _counts_by_treatment_id = _df \
        .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
        .apply(pd.Series) \
        .sum() \
        .astype(int)
    return _counts_by_treatment_id.groupby(level=0).sum()

counts_of_hsi_by_treatment_id = summarize(
    extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_counts_of_hsi_by_treatment_id,
        do_scaling=True
    ),
    only_mean=True,
    collapse_columns=True,
)

counts_of_hsi_by_treatment_id = counts_of_hsi_by_treatment_id.assign(baseline_values=counts_of_hsi_by_treatment_id[(0, 'mean')]).sort_values('baseline_values').drop(columns='baseline_values')
hsi_by_treatment_id = counts_of_hsi_by_treatment_id.unstack().reset_index()
hsi_by_treatment_id = hsi_by_treatment_id.rename(columns={'level_2': 'Treatment_ID', 0: 'qty_of_HSIs'})

# hsi[(0,'mean')].sum()/counts_of_hsi_by_treatment_id[(0,'mean')].sum()

# 4.2 Number of Services delivered by short Treatment ID
#--------------------------------------------------------
def get_counts_of_hsi_by_short_treatment_id(_df):
    """Get the counts of the short TREATMENT_IDs occurring (shortened, up to first underscore)"""
    _counts_by_treatment_id = get_counts_of_hsi_by_treatment_id(_df)
    _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
    return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()


counts_of_hsi_by_treatment_id_short = summarize(
    extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_counts_of_hsi_by_short_treatment_id,
        do_scaling=True
    ),
    only_mean=True,
    collapse_columns=True,
)

hsi_by_short_treatment_id = counts_of_hsi_by_treatment_id_short.unstack().reset_index()
hsi_by_short_treatment_id = hsi_by_short_treatment_id.rename(columns = {'level_2': 'Short_Treatment_ID', 0: 'qty_of_HSIs'})


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

