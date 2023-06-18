"""This file uses the results of
`impact_of_cons_regression_scenarios.py` and 'impact_of_consumables_availability.py'
to extract summary results for the June 2023 Think Tank Presentation.

The simulation was run for a short period of 6 years (2020 - 2025) because
holding the consumable availability constant in the short run would be more justifiable
than holding it constant for a long period.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tlo import Date

import matplotlib.pyplot as plt
import textwrap # to wrap x-tick labels in the DALY figures
import pandas as pd
import numpy as np
import pickle

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
path_for_figures = outputspath / "think_tank_figures_june2023"

PREFIX_ON_FILENAME = '3'

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2020, 1, 1), Date(2025, 12, 31))

def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

def formatting_hsi_df(_df):
    """Standard formatting for the HSI_Event log."""
    _df = _df.pipe(drop_outside_period) \
        .drop(_df.index[~_df.did_run]) \
        .reset_index(drop=True) \
        .drop(columns=['Person_ID', 'Squeeze_Factor', 'Facility_ID', 'did_run'])

    # Unpack the dictionary in `Number_By_Appt_Type_Code`.
    _df = _df.join(_df['Number_By_Appt_Type_Code'].apply(pd.Series).fillna(0.0)).drop(
        columns='Number_By_Appt_Type_Code')

    # Produce coarse version of TREATMENT_ID (just first level, which is the module)
    _df['TREATMENT_ID_SHORT'] = _df['TREATMENT_ID'].str.split('_').apply(lambda x: x[0])

    return _df

# %% Gathering basic information
# Find results_folder associated with a given batch_file and get most recent
#results_folder = get_scenario_outputs('impact_of_cons_availability_intervention.py', outputspath)
# Results folder for `impact_of_cons_regression_scenarios.py`
results_folder = Path(outputspath/ 'impact_of_consumables_availability_intervention-2023-06-14T232900Z/')
# Results folder for 'impact_of_consumables_availability.py'
results_folder_default = outputspath / 'default_scenario' # move the 'NO_CHANGE' scenario folder to a seperate folder

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


# %% Extracting results from run

# Create a list of strings summarizing the parameter values in the different draws
param_strings = [f"{row.module_param}={row.value}" for _, row in params.iterrows()]
scenario_strings = [f"{row.value}" for _, row in params.iterrows()]

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
summarised_total_dalys = summarize(total_dalys_accrued)

# Figure showing total DALYs accrued
def plot_summarized_total_dalys(summarised_total_dalys, param_strings, file_name):
    fig, ax = plt.subplots()
    number_of_draws = len(param_strings)
    statistic_values = {
        s: np.array(
            [summarised_total_dalys[(d, s)].values[0] for d in range(number_of_draws)]
        )
        for s in ["mean", "lower", "upper"]
    }
    ax.bar(
        param_strings,
        statistic_values["mean"],
        yerr=[
            statistic_values["mean"] - statistic_values["lower"],
            statistic_values["upper"] - statistic_values["mean"]
        ]
    )
    ax.set_ylabel("Total number of DALYs")

    # Clean x-tick labels
    label_mapping = {
                  'NO_CHANGE': 'Actual (2018)',
                  'scenario_fac_type': "Scenario: Facility level 1b",
                  'scenario_fac_owner': "Scenario: Facility owner CHAM",
                  'scenario_functional_computer': "Scenario: Functional computer available",
                  'scenario_incharge_drug_orders': "Scenario: Pharmacist in-charge of drug orders",
                  'scenario_functional_emergency_vehicle': "Scenario: Emergency vehicle available",
                  'scenario_service_diagnostic': "Scenario: Diagnostic services available",
                  'scenario_dist_todh': "Scenario: Distance to DHO - 0-10 kms",
                  'scenario_dist_torms': "Scenario: Distance to RMS - 0-10 kms" ,
                  'scenario_drug_order_fulfilment_freq_last_3mts': "Scenario: Monthly drug order fulfillment",
                  'scenario_all_features': "Scenario: All features"
    } # Dictionary mapping old x-tick labels to new ones

    xtick_labels = ax.get_xticklabels() # Get the current x-tick labels
    new_xtick_labels = [label_mapping.get(label.get_text(), label.get_text()) for label in xtick_labels] # Replace the x-tick labels using the mapping dictionary

    wrapped_labels = [textwrap.fill(label, 10) for label in new_xtick_labels] # Wrap the x-tick labels

    ax.set_xticklabels(wrapped_labels)  # Set the new x-tick labels
    ax.tick_params(axis="x", rotation=45, labelsize=7) # format x-tick labels

    plt.tight_layout()
    plt.savefig(path_for_figures / file_name, dpi=300)
    return fig, ax

# Plot the deaths by age across the two scenario draws as line plot with error bars
fig_1, ax_1 = plot_summarized_total_dalys(summarize(total_dalys_accrued), scenario_strings, 'dalys_comparison.png')

# 1.2 (Optional) Difference in total DALYs accrued by disease
def _extract_dalys_by_disease(_df: pd.DataFrame) -> pd.Series:
    """Construct a series with index disease and value of the total of DALYS (stacked) from the
    `dalys_stacked` key logged in `tlo.methods.healthburden`.
    N.B. This limits the time period of interest to 2010-2019"""
    _, calperiodlookup = make_calendar_period_lookup()

    return _df.loc[(_df['year'] >=2020) & (_df['year'] < 2025)]\
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
print(dalys_by_disease_summarized[[(0,  'mean'),(1,  'mean'), (2,  'mean')]])
# dalys_by_disease_summarized.to_csv(outputspath / 'dalys_by_disease.csv')

# 2. Services delivered
#-----------------------------------------
# 2.1 Total number of HSIs "completed"
hsi_alternatescenario = load_pickled_dataframes(results_folder, draw=0, run=0)['tlo.methods.healthsystem.summary']['HSI_Event']
hsi_default = load_pickled_dataframes(results_folder, draw=1, run=1)['tlo.methods.healthsystem.summary']['HSI_Event']

# 2.2 Number of HSIs completed by disease
# use 'TREATMENT_ID' on above data

# 3. Consumable availability
#-----------------------------------------
# Sample data
#consumable_alternatescenario = load_pickled_dataframes(results_folder, draw=0, run=2)['tlo.methods.healthsystem']['Consumables']
#consumable_default = load_pickled_dataframes(results_folder, draw=1, run=1)['tlo.methods.healthsystem']['Consumables']

def figure6_cons_use(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """ 'Figure 6': Usage of consumables in the HealthSystem"""

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig6_{stub}.png"  # noqa: E731

    def get_counts_of_items_requested(_df):
        _df = drop_outside_period(_df)

        counts_of_available = defaultdict(int)
        counts_of_not_available = defaultdict(int)

        for _, row in _df.iterrows():
            for item, num in eval(row['Item_Available']).items():
                counts_of_available[item] += num
            for item, num in eval(row['Item_NotAvailable']).items():
                counts_of_not_available[item] += num

        return pd.concat(
            {'Available': pd.Series(counts_of_available), 'Not_Available': pd.Series(counts_of_not_available)},
            axis=1
        ).fillna(0).astype(int).stack()

    cons_req = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='Consumables',
            custom_generate_series=get_counts_of_items_requested,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True
    )

# %% The problem with the above code is that it was written for one draw rather than 2 so the dataframe remains multiindex and the merge does not work

    # Merge in item names and prepare to plot:
    cons = cons_req.unstack()
    # cons = cons['mean'] # new line
    cons_names = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'consumables' / 'ResourceFile_Consumables_Items_and_Packages.csv'
    )[['Item_Code', 'Items']].set_index('Item_Code').drop_duplicates()

    #cons_multiindex_columns = cons.columns # remove this if only one draw is used
    #cons.columns = cons.columns.get_level_values(0) # remove this if only one draw is used

    cons = cons.merge(cons_names, left_index=True, right_index=True, how='left').set_index('Items')# .astype(int)
    # %% In order to make the astype(int) work, I could replace all the NaNs as 0s

    cons = cons[cons.Available.notna()] # new line
    cons = cons.assign(total=cons.sum(1)).sort_values('total').drop(columns='total')

    #cons.columns = cons_multiindex_columns # remove this if only one draw is used

    # 3.1 Instances of availability/unavailability of top 20 demanded consumables
    fig, ax = plt.subplots()
    name_of_plot = 'Demand For Consumables'
    (cons / 1e6).head(20).plot.barh(ax=ax, stacked=True)
    ax.set_title(name_of_plot)
    ax.set_ylabel('Item (20 most requested)')
    ax.set_xlabel('Number of requests (Millions)')
    ax.yaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    # 3.2 To 20 consumables not available
    fig, ax = plt.subplots()
    name_of_plot = 'Consumables Not Available'
    (cons['Not_Available'] / 1e6).sort_values(ascending = False).head(20).plot.barh(ax=ax) # change = updated to descending
    ax.set_title(name_of_plot)
    ax.set_ylabel('Item (20 most frequently not available when requested)')
    ax.set_xlabel('Number of requests (Millions)')
    ax.yaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    # 3.3 Proportion of HSIs for which consumable was recorded as not available

    def get_treatment_id_affecting_by_missing_consumables(_df):
        """Return frequency that a (short) TREATMENT_ID suffers from consumables not being available."""
        _df = drop_outside_period(_df)
        _df = _df.loc[(_df['Item_NotAvailable'] != '{}'), ['TREATMENT_ID', 'Item_NotAvailable']]
        _df['TREATMENT_ID_SHORT'] = _df['TREATMENT_ID'].map(lambda x: x.split('_')[0])
        return _df['TREATMENT_ID_SHORT'].value_counts()

    treatment_id_affecting_by_missing_consumables = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='Consumables',
            custom_generate_series=get_treatment_id_affecting_by_missing_consumables,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True
    )

    fig, ax = plt.subplots()
    name_of_plot = 'HSI Affected by Unavailable Consumables (by Short TREATMENT_ID)'
    squarify_neat(
        sizes=treatment_id_affecting_by_missing_consumables.values,
        label=treatment_id_affecting_by_missing_consumables.index,
        colormap=get_color_short_treatment_id,
        alpha=1,
        ax=ax,
        text_kwargs={'color': 'black', 'size': 8}
    )
    ax.set_axis_off()
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

figure6_cons_use(results_folder_default, path_for_figures, resourcefilepath)

'''
# The following code will be relevant once mode_appt_constraints = 2 is applied

# 4. Staff use
# Proportion of staff time demanded (This could also be measured as number of minutes of
# staff time required under the two scenarios)
# Load pickle files
staffusage_alternatescenario = load_pickled_dataframes(results_folder, draw=0, run=0)['tlo.methods.healthsystem']['Capacity']
staffusage_default = load_pickled_dataframes(results_folder, draw=1, run=1)['tlo.methods.healthsystem']['Capacity']

staffusage_alternatescenario['Frac_Time_Used_Overall']
staffusage_default['Frac_Time_Used_Overall']


# 3.4 Proportion of staff time demanded by disease and *facility level*
def figure4_hr_use_overall(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """ 'Figure 4': The level of usage of the HealthSystem HR Resources """

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig4_{stub}.png"  # noqa: E731

    def get_share_of_time_for_hw_in_each_facility_by_short_treatment_id(_df):
        _df = drop_outside_period(_df)
        _df = _df.set_index('date')
        _all = _df['Frac_Time_Used_Overall']
        _df = _df['Frac_Time_Used_By_Facility_ID'].apply(pd.Series)
        _df.columns = _df.columns.astype(int)
        _df = _df.reindex(columns=sorted(_df.columns))
        _df['All'] = _all
        return _df.groupby(pd.Grouper(freq="M")).mean().stack()  # find monthly averages and stack into series

    def get_share_of_time_used_for_each_officer_at_each_level(_df):
        _df = drop_outside_period(_df)
        _df = _df.set_index('date')
        _df = _df['Frac_Time_Used_By_OfficerType'].apply(pd.Series).mean()  # find mean over the period
        _df.index = unflatten_flattened_multi_index_in_logging(_df.index)
        return _df

    capacity_by_facility = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='Capacity',
            custom_generate_series=get_share_of_time_for_hw_in_each_facility_by_short_treatment_id,
            do_scaling=False
        ),
        only_mean=True,
        collapse_columns=True
    )

    capacity_by_officer = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='Capacity',
            custom_generate_series=get_share_of_time_used_for_each_officer_at_each_level,
            do_scaling=False
        ),
        only_mean=True,
        collapse_columns=True
    )

    # Find the levels of each facility
    mfl = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'organisation' / 'ResourceFile_Master_Facilities_List.csv'
    ).set_index('Facility_ID')

    def find_level_for_facility(id):
        return mfl.loc[id].Facility_Level

    color_for_level = {'0': 'blue', '1a': 'yellow', '1b': 'green', '2': 'grey', '3': 'orange', '4': 'black',
                       '5': 'white'}

    fig, ax = plt.subplots()
    name_of_plot = 'Usage of Healthcare Worker Time By Month'
    capacity_unstacked = capacity_by_facility.unstack()
    for i in capacity_unstacked.columns:
        if i != 'All':
            level = find_level_for_facility(i)
            h1, = ax.plot(capacity_unstacked[i].index, capacity_unstacked[i].values,
                          color=color_for_level[level], linewidth=0.5, label=f'Facility_Level {level}')

    h2, = ax.plot(capacity_unstacked['All'].index, capacity_unstacked['All'].values, color='red', linewidth=1.5)
    ax.set_title(name_of_plot)
    ax.set_xlabel('Month')
    ax.set_ylabel('Fraction of all time used\n(Average for the month)')
    ax.legend([h1, h2], ['Each Facility', 'All Facilities'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    name_of_plot = 'Usage of Healthcare Worker Time (Average)'
    capacity_unstacked_average = capacity_by_facility.unstack().mean()
    # levels = [find_level_for_facility(i) if i != 'All' else 'All' for i in capacity_unstacked_average.index]
    xpos_for_level = dict(zip((color_for_level.keys()), range(len(color_for_level))))
    for id, val in capacity_unstacked_average.items():
        if id != 'All':
            _level = find_level_for_facility(id)
            if _level != '5':
                xpos = xpos_for_level[_level]
                scatter = (np.random.rand() - 0.5) * 0.25
                h1, = ax.plot(xpos + scatter, val * 100, color=color_for_level[_level],
                              marker='.', markersize=15, label='Each Facility', linestyle='none')
    h2 = ax.axhline(y=capacity_unstacked_average['All'] * 100,
                    color='red', linestyle='--', label='Average')
    ax.set_title(name_of_plot)
    ax.set_xlabel('Facility_Level')
    ax.set_xticks(list(xpos_for_level.values()))
    ax.set_xticklabels(xpos_for_level.keys())
    ax.set_ylabel('Percent of Time Available That is Used\n')
    ax.legend(handles=[h1, h2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    name_of_plot = 'Usage of Healthcare Worker Time by Cadre and Facility_Level'
    (100.0 * capacity_by_officer.unstack()).T.plot.bar(ax=ax)
    ax.legend()
    ax.set_xlabel('Facility_Level')
    ax.set_ylabel('Percent of time that is used')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

figure4_hr_use_overall(results_folder, outputspath, resourcefilepath)

# Collapse capacity dataframe
collapsed_df_officer = capacity_by_officer.groupby('Officer_Type').mean()
collapsed_df_level = capacity_by_officer.groupby('Facility_Level').mean()

# Subset the dataframe for relevant officer categories
officer_types = ['Clinical', 'Nursing_and_Midwifery', 'Pharmacy']
idx = pd.IndexSlice # Create an IndexSlice object
capacity_by_officer_subset = capacity_by_officer.loc[idx[officer_types, :], :] # Subset the DataFrame using the IndexSlice object
'''

'''
# Scratch code
#-----------------------
# DALYs by age group and time
def _extract_dalys_by_age_group_and_time_period(_df: pd.DataFrame) -> pd.Series:
    """Construct a series with index age-rage/time-period and value of the total of DALYS (stacked) from the
    `dalys_stacked` key logged in `tlo.methods.healthburden`."""
    _, calperiodlookup = make_calendar_period_lookup()

    return _df.assign(
                Period=lambda x: x['year'].map(calperiodlookup).astype(make_calendar_period_type()),
            ).set_index('Period')\
             .drop(columns=['date', 'sex', 'age_range', 'year'])\
             .groupby(axis=0, level=0)\
             .sum()\
             .sum(axis=1)


#-----------------------
dalys_extracted = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked",
    custom_generate_series=_extract_dalys_by_age_group_and_time_period,
    do_scaling=True
)

dalys_summarized = summarize(dalys_extracted)
dalys_summarized = dalys_summarized.loc[dalys_summarized.index.isin(('2010-2014'))]

# Load the pickle file
file = results_folder / '0/0/tlo.methods.healthsystem.pickle'
with open(file, 'rb') as f:
    healthsystem_usage_alternatescenario = pickle.load(f)

file = results_folder / '1/0/tlo.methods.healthsystem.pickle'
with open(file, 'rb') as f:
    healthsystem_usage_default = pickle.load(f)

#-----------------------
# Extract excel file
# Created a new function because I wasn't sure how metadata needs to be specified
def write_log_to_excel_new(filename, log_dataframes):
    """Takes the output of parse_log_file() and creates an Excel file from dataframes"""
    sheets = list()
    sheet_count = 0
    for module, key_df in log_dataframes.items():
        for key, df in key_df.items():
            sheet_count += 1
            sheets.append([module, key, sheet_count])

    writer = pd.ExcelWriter(filename)
    index = pd.DataFrame(data=sheets, columns=['module', 'key', 'sheet'])
    index.to_excel(writer, sheet_name='Index')

    sheet_count = 0
    for module, key_df in log_dataframes.items():
        for key, df in key_df.items():
            sheet_count += 1
            df.to_excel(writer, sheet_name=f'Sheet {sheet_count}')
    writer.close() # AttributeError: 'OpenpyxlWriter' object has no attribute 'save'

# Write log to excel
#parse_log_file(results_folder, level: int = logging.INFO)
log_dataframes ={
            'healthsystem': {'Consumables': pd.DataFrame(),
                             'Capacity': pd.DataFrame()
                             },
            'healthburden': {'dalys_stacked': pd.DataFrame(),
                             'dalys': pd.DataFrame()
                             }
        }
#write_log_to_excel_new(results_folder, log_dataframes)

'''
