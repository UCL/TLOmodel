import argparse
from pathlib import Path
from tlo import Date
from collections import Counter, defaultdict

import calendar
import datetime
import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import ast
import math

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    summarize,
    create_pickles_locally,
    parse_log_file,
    unflatten_flattened_multi_index_in_logging
)

from scripts.costing.cost_estimation import (estimate_input_cost_of_scenarios,
                                             do_stacked_bar_plot_of_cost_by_category,
                                             do_line_plot_of_cost)
# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
costing_outputs_folder = Path('./outputs/costing')
if not os.path.exists(costing_outputs_folder):
    os.makedirs(costing_outputs_folder)
figurespath = costing_outputs_folder / "global_fund_roi_analysis"
if not os.path.exists(figurespath):
    os.makedirs(figurespath)

def summarize_cost_data(_df):
    _df = _df.stack()
    collapsed_df = _df.groupby(level='draw').agg([
            'mean',
            ('lower', lambda x: x.quantile(0.025)),
            ('upper', lambda x: x.quantile(0.975))
        ])

    collapsed_df = collapsed_df.unstack()
    collapsed_df.index = collapsed_df.index.set_names('stat', level=0)
    collapsed_df = collapsed_df.unstack(level='stat')
    return collapsed_df

# Load result files
#-------------------
#results_folder = get_scenario_outputs('htm_with_and_without_hss-2024-09-04T143044Z.py', outputfilepath)[0] # Tara's FCDO/GF scenarios version 1
#results_folder = get_scenario_outputs('hss_elements-2024-09-04T142900Z.py', outputfilepath)[0] # Tara's FCDO/GF scenarios version 1
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/t.mangal@imperial.ac.uk')
results_folder = get_scenario_outputs('htm_with_and_without_hss-2024-10-22T163743Z.py', outputfilepath)[0] # Tara's FCDO/GF scenarios version 2
#results_folder = get_scenario_outputs('hss_elements-2024-10-12T111649Z.py', outputfilepath)[0] # Tara's FCDO/GF scenarios version 2

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0) # look at one log (so can decide what to extract)
params = extract_params(results_folder)
population_scaling_factor = log['tlo.methods.demography']['scaling_factor']['scaling_factor'].iloc[0]
TARGET_PERIOD_INTERVENTION = (Date(2025, 1, 1), Date(2035, 12, 31))
relevant_period_for_costing = [i.year for i in TARGET_PERIOD_INTERVENTION]
htm_scenarios = {0:"Baseline", 1: "HSS PACKAGE: Perfect", 2: "HSS PACKAGE: Realistic", 3: "HIV Programs Scale-up WITHOUT HSS PACKAGE",
4: "HIV Programs Scale-up WITH FULL HSS PACKAGE", 5: "HIV Programs Scale-up WITH REALISTIC HSS PACKAGE", 6: "TB Programs Scale-up WITHOUT HSS PACKAGE",
7: "TB Programs Scale-up WITH FULL HSS PACKAGE", 8: "TB Programs Scale-up WITH REALISTIC HSS PACKAGE", 9: "Malaria Programs Scale-up WITHOUT HSS PACKAGE",
10: "Malaria Programs Scale-up WITH FULL HSS PACKAGE", 11: "Malaria Programs Scale-up WITH REALISTIC HSS PACKAGE", 12: "HTM Programs Scale-up WITHOUT HSS PACKAGE",
13: "HTM Programs Scale-up WITH FULL HSS PACKAGE", 14: "HTM Programs Scale-up WITH REALISTIC HSS PACKAGE", 15: "HTM Programs Scale-up WITH SUPPLY CHAINS", 16: "HTM Programs Scale-up WITH HRH"}
htm_scenarios_for_gf_report = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16]

hss_scenarios = {0: "Baseline", 1: "HRH Moderate Scale-up (1%)", 2: "HRH Scale-up Following Historical Growth", 3: "HRH Accelerated Scale-up (6%)",
                 4: "Increase Capacity at Primary Care Levels", 5: "Increase Capacity of CHW", 6: "Consumables Increased to 75th Percentile",
                 7: "Consumables Available at HIV levels", 8: "Consumables Available at EPI levels", 9: "Perfect Consumables Availability",
                 10: "HSS PACKAGE: Perfect", 11: "HSS PACKAGE: Realistic expansion, no change in HSB", 12: "HSS PACKAGE: Realistic expansion"}
hss_scenarios_for_gf_report = [0, 1, 3, 4, 6, 7, 8, 9, 10, 12]

# Load the list of districts and their IDs
district_dict = pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')[
    ['District_Num', 'District']].drop_duplicates()
district_dict = dict(zip(district_dict['District_Num'], district_dict['District']))

# Estimate standard input costs of scenario
#-----------------------------------------------------------------------------------------------------------------------
input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath , cost_only_used_staff=True) # summarise = True
#input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath , draws = draws_included, cost_only_used_staff=True, summarize = True)

# Add additional costs pertaining to simulation
#-----------------------------------------------------------------------------------------------------------------------
# Extract supply chain cost as a proportion of consumable costs to apply to malaria scale-up commodities
# Load primary costing resourcefile
workbook_cost = pd.read_excel((resourcefilepath / "costing/ResourceFile_Costing.xlsx"),
                              sheet_name=None)
# Read parameters for consumables costs
# Load consumables cost data
unit_price_consumable = workbook_cost["consumables"]
unit_price_consumable = unit_price_consumable.rename(columns=unit_price_consumable.iloc[0])
unit_price_consumable = unit_price_consumable[['Item_Code', 'Final_price_per_chosen_unit (USD, 2023)']].reset_index(
    drop=True).iloc[1:]
unit_price_consumable = unit_price_consumable[unit_price_consumable['Item_Code'].notna()]

# Assume that the cost of procurement, warehousing and distribution is a fixed proportion of consumable purchase costs
# The fixed proportion is based on Resource Mapping Expenditure data from 2018
resource_mapping_data = workbook_cost["resource_mapping_r7_summary"]
# Make sure values are numeric
expenditure_column = ['EXPENDITURE (USD) (Jul 2018 - Jun 2019)']
resource_mapping_data[expenditure_column] = resource_mapping_data[expenditure_column].apply(
    lambda x: pd.to_numeric(x, errors='coerce'))
supply_chain_expenditure = \
resource_mapping_data[resource_mapping_data['Cost Type'] == 'Supply Chain'][expenditure_column].sum()[0]
consumables_purchase_expenditure = \
resource_mapping_data[resource_mapping_data['Cost Type'] == 'Drugs and Commodities'][expenditure_column].sum()[0]
supply_chain_cost_proportion = supply_chain_expenditure / consumables_purchase_expenditure

# In this case malaria intervention scale-up costs were not included in the standard estimate_input_cost_of_scenarios function
list_of_draws_with_malaria_scaleup_parameters = params[(params.module_param == 'Malaria:scaleup_start_year')]
list_of_draws_with_malaria_scaleup_parameters.loc[:,'value'] = pd.to_numeric(list_of_draws_with_malaria_scaleup_parameters['value'])
list_of_draws_with_malaria_scaleup_implemented_in_costing_period = list_of_draws_with_malaria_scaleup_parameters[(list_of_draws_with_malaria_scaleup_parameters['value'] < max(relevant_period_for_costing))].index.to_list()

# 1. IRS costs
irs_coverage_rate = 0.8
districts_with_irs_scaleup = ['Kasungu', 'Mchinji', 'Lilongwe', 'Lilongwe City', 'Dowa', 'Ntchisi', 'Salima', 'Mangochi',
                              'Mwanza', 'Likoma', 'Nkhotakota']
# Convert above list of district names to numeric district identifiers
district_keys_with_irs_scaleup = [key for key, name in district_dict.items() if name in districts_with_irs_scaleup]
TARGET_PERIOD_MALARIA_SCALEUP = (Date(2024, 1, 1), Date(2035, 12, 31))

# Get population by district
def get_total_population_by_district(_df):
    years_needed = [i.year for i in TARGET_PERIOD_MALARIA_SCALEUP] # we only consider the population for the malaria scale-up period
    # because those are the years relevant for malaria scale-up costing
    _df['year'] = pd.to_datetime(_df['date']).dt.year
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    _df = pd.melt(_df.drop(columns = 'date'), id_vars = ['year']).rename(columns = {'variable': 'district'})
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .set_index(['year', 'district'])['value']
    )

district_population_by_year = extract_results(
    results_folder,
    module='tlo.methods.malaria',
    key='pop_district',
    custom_generate_series=get_total_population_by_district,
    do_scaling=True
)

def get_number_of_people_covered_by_malaria_scaleup(_df, list_of_districts_covered = None, draws_included = None):
    _df = pd.DataFrame(_df)
    # Reset the index to make 'district' a column
    _df = _df.reset_index()
    # Convert the 'district' column to numeric values
    _df['district'] = pd.to_numeric(_df['district'], errors='coerce')
    _df = _df.set_index(['year', 'district'])
    if list_of_districts_covered is not None:
        _df.loc[~_df.index.get_level_values('district').isin(list_of_districts_covered), :] = 0
    if draws_included is not None:
        _df.loc[:, ~_df.columns.get_level_values('draw').isin(draws_included)] = 0
    return _df

district_population_covered_by_irs_scaleup_by_year = get_number_of_people_covered_by_malaria_scaleup(district_population_by_year,
                                                                                                 list_of_districts_covered=district_keys_with_irs_scaleup,
                                                                                                 draws_included = list_of_draws_with_malaria_scaleup_implemented_in_costing_period)

irs_cost_per_person = unit_price_consumable[unit_price_consumable.Item_Code == 161]['Final_price_per_chosen_unit (USD, 2023)']
# The above unit cost already includes implementation - project management (17%), personnel (6%), vehicles (10%), equipment (6%), monitoring and evaluation (3%), training (3%),
# other commodities (3%) and buildings (2%) from Alonso et al (2021)
irs_multiplication_factor = irs_cost_per_person * irs_coverage_rate
total_irs_cost = irs_multiplication_factor.iloc[0] * district_population_covered_by_irs_scaleup_by_year # for districts and scenarios included
total_irs_cost = total_irs_cost.groupby(level='year').sum()

# 2. Bednet costs
bednet_coverage_rate = 0.7
# We can assume 3-year lifespan of a bednet, each bednet covering 1.8 people.
unit_cost_of_bednet = unit_price_consumable[unit_price_consumable.Item_Code == 160]['Final_price_per_chosen_unit (USD, 2023)'] * (1 + supply_chain_cost_proportion)
# We add supply chain costs (procurement + distribution + warehousing) because the unit_cost does not include this
annual_bednet_cost_per_person = unit_cost_of_bednet / 1.8 / 3
bednet_multiplication_factor = bednet_coverage_rate * annual_bednet_cost_per_person

district_population_covered_by_bednet_scaleup_by_year = get_number_of_people_covered_by_malaria_scaleup(district_population_by_year,
                                                                                                 draws_included = list_of_draws_with_malaria_scaleup_implemented_in_costing_period) # All districts covered

total_bednet_cost = bednet_multiplication_factor.iloc[0] * district_population_covered_by_bednet_scaleup_by_year  # for scenarios included
total_bednet_cost = total_bednet_cost.groupby(level='year').sum()

# Malaria scale-up costs - TOTAL
malaria_scaleup_costs = [
    (total_irs_cost.reset_index(), 'cost_of_IRS_scaleup'),
    (total_bednet_cost.reset_index(), 'cost_of_bednet_scaleup'),
]
def melt_and_label_malaria_scaleup_cost(_df, label):
    multi_index = pd.MultiIndex.from_tuples(_df.columns)
    _df.columns = multi_index

    # reshape dataframe and assign 'draw' and 'run' as the correct column headers
    melted_df = pd.melt(_df, id_vars=['year']).rename(columns={'variable_0': 'draw', 'variable_1': 'run'})
    # Replace item_code with consumable_name_tlo
    melted_df['cost_subcategory'] = label
    melted_df['cost_category'] = 'other'
    melted_df['cost_subgroup'] = 'NA'
    melted_df['Facility_Level'] = 'all'
    melted_df = melted_df.rename(columns={'value': 'cost'})
    return melted_df

# Iterate through additional costs, melt and concatenate
for df, label in malaria_scaleup_costs:
    new_df = melt_and_label_malaria_scaleup_cost(df, label)
    input_costs = pd.concat([input_costs, new_df], ignore_index=True)

# TODO Reduce the cost of Oxygen and Depo-medroxy temporarily which we figure out the issue with this

# Aggregate input costs for further analysis
input_costs_subset = input_costs[(input_costs['year'] >= relevant_period_for_costing[0]) & (input_costs['year'] <= relevant_period_for_costing[1])]
total_input_cost = input_costs_subset.groupby(['draw', 'run'])['cost'].sum()

# Calculate incremental cost
#-----------------------------------------------------------------------------------------------------------------------

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


# TODO the following calculation should first capture the different by run and then be summarised
incremental_scenario_cost = (pd.DataFrame(
            find_difference_relative_to_comparison(
                total_input_cost,
                comparison= 0) # sets the comparator to 0 which is the Actual scenario
        ).T.iloc[0].unstack()).T

# %%
# Monetary value of health impact
def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = relevant_period_for_costing # [i.year for i in TARGET_PERIOD_INTERVENTION]
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=True
    )

#num_dalys_summarized = summarize(num_dalys).loc[0].unstack()
#num_dalys_summarized['scenario'] = scenarios.to_list() # add when scenarios have names
#num_dalys_summarized = num_dalys_summarized.set_index('scenario')

# Get absolute DALYs averted
num_dalys_averted =(-1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison= 0) # sets the comparator to 0 which is the Actual scenario
        ).T.iloc[0].unstack(level = 'run'))

#num_dalys = num_dalys.loc[0].unstack()
#num_dalys_averted = num_dalys_averted[num_dalys_averted.index.get_level_values(0).isin(draws_included)]
#num_dalys_averted['scenario'] = scenarios.to_list()[1:12]
#num_dalys_averted = num_dalys_averted.set_index('scenario')

chosen_cet = 77.4 # based on Ochalek et al (2018) - the paper provided the value $61 in 2016 USD terms, this value is in 2023 USD terms
monetary_value_of_incremental_health = (num_dalys_averted * chosen_cet).clip(0.0)
max_ability_to_pay_for_implementation = (monetary_value_of_incremental_health - incremental_scenario_cost).clip(0.0) # monetary value - change in costs
#TODO check that the above calculation is correct

# Plot costs
#-----------------------------------------------------------------------------------------------------------------------
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'medical consumables', _disaggregate_by_subgroup = True, _year = [2018], _outputfilepath = figurespath)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'human resources for health', _disaggregate_by_subgroup = True, _year = [2018], _outputfilepath = figurespath)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'medical equipment', _disaggregate_by_subgroup = True, _year = [2018], _outputfilepath = figurespath)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'other', _year = [2018], _outputfilepath = figurespath)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _year = list(range(2020, 2030)), _outputfilepath = figurespath)

do_line_plot_of_cost(_df = input_costs, _cost_category = 'medical consumables', _year = 'all', _draws = [0], disaggregate_by= 'cost_subgroup',_outputfilepath = figurespath)
do_line_plot_of_cost(_df = input_costs, _cost_category = 'other', _year = 'all', _draws = [0], disaggregate_by= 'cost_subgroup',_outputfilepath = figurespath)
do_line_plot_of_cost(_df = input_costs, _cost_category = 'human resources for health', _year = 'all', _draws = [0], disaggregate_by= 'cost_subgroup',_outputfilepath = figurespath)
do_line_plot_of_cost(_df = input_costs, _cost_category = 'human resources for health', _year = 'all', _draws = [0], disaggregate_by= 'cost_subcategory', _outputfilepath = figurespath)
do_line_plot_of_cost(_df = input_costs, _cost_category = 'medical equipment', _year = 'all', _draws = None, _outputfilepath = figurespath)
do_line_plot_of_cost(_df = input_costs, _cost_category = 'other', _year = 'all', _draws = None, _outputfilepath = figurespath)
do_line_plot_of_cost(_df = input_costs, _cost_category = 'all', _year = 'all', disaggregate_by= 'cost_category', _draws = None, _outputfilepath = figurespath)

# 3. Return on Investment Plot
#----------------------------------------------------
# Plot ROI at various levels of cost
roi_outputs_folder = Path(figurespath / 'roi' / 'htm')
if not os.path.exists(roi_outputs_folder):
    os.makedirs(roi_outputs_folder)

# Iterate over each draw in monetary_value_of_incremental_health
for draw_index, row in monetary_value_of_incremental_health.iterrows():
    # Initialize an empty DataFrame to store values for each 'run'
    all_run_values = pd.DataFrame()

    # Create an array of implementation costs ranging from 0 to the max value of max ability to pay for the current draw
    implementation_costs = np.linspace(0, max_ability_to_pay_for_implementation.loc[draw_index].max(), 50)

    # Retrieve the corresponding row from incremental_scenario_cost for the same draw
    scenario_cost_row = incremental_scenario_cost.loc[draw_index]

    # Calculate the values for each individual run
    for run in scenario_cost_row.index:  # Assuming 'run' columns are labeled by numbers
        # Calculate the cost-effectiveness metric for the current run
        run_values = (row[run] - (implementation_costs + scenario_cost_row[run])) / (
                implementation_costs + scenario_cost_row[run])

        # Create a DataFrame with index as (draw_index, run) and columns as implementation costs
        run_df = pd.DataFrame([run_values], index=pd.MultiIndex.from_tuples([(draw_index, run)], names=['draw', 'run']),
                              columns=implementation_costs)

        # Append the run DataFrame to all_run_values
        all_run_values = pd.concat([all_run_values, run_df])

    collapsed_data = all_run_values.groupby(level='draw').agg([
            'mean',
            ('lower', lambda x: x.quantile(0.025)),
            ('upper', lambda x: x.quantile(0.975))
        ])

    collapsed_data = collapsed_data.unstack()
    collapsed_data.index = collapsed_data.index.set_names('implementation_cost', level=0)
    collapsed_data.index = collapsed_data.index.set_names('stat', level=1)
    collapsed_data = collapsed_data.reset_index().rename(columns = {0: 'roi'})
    #collapsed_data = collapsed_data.reorder_levels(['draw', 'stat', 'implementation_cost'])

    # Divide rows by the sum of implementation costs and incremental input cost
    mean_values = collapsed_data[collapsed_data['stat'] == 'mean'][['implementation_cost', 'roi']]
    lower_values = collapsed_data[collapsed_data['stat'] == 'lower'][['implementation_cost', 'roi']]
    upper_values = collapsed_data[collapsed_data['stat']  == 'upper'][['implementation_cost', 'roi']]

    # Plot mean line
    plt.plot(implementation_costs / 1e6, mean_values['roi'], label=f'{htm_scenarios[draw_index]}')
    # Plot the confidence interval as a shaded region
    plt.fill_between(implementation_costs / 1e6, lower_values['roi'], upper_values['roi'], alpha=0.2)

    plt.xlabel('Implementation cost, millions')
    plt.ylabel('Return on Investment')
    plt.title('Return on Investment of scenario at different levels of implementation cost')

    #plt.text(x=0.95, y=0.8,
    #         s=f"Monetary value of incremental health = USD {round(monetary_value_of_incremental_health.loc[draw_index]['mean'] / 1e6, 2)}m (USD {round(monetary_value_of_incremental_health.loc[draw_index]['lower'] / 1e6, 2)}m-{round(monetary_value_of_incremental_health.loc[draw_index]['upper'] / 1e6, 2)}m);\n "
    #           f"Incremental input cost of scenario = USD {round(scenario_cost_row['mean'] / 1e6, 2)}m (USD {round(scenario_cost_row['lower'] / 1e6, 2)}m-{round(scenario_cost_row['upper'] / 1e6, 2)}m)",
    #         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=9,
    #         weight='bold', color='black')

    # Show legend
    plt.legend()
    # Save
    plt.savefig(figurespath / f'roi/htm/draw{draw_index}_{htm_scenarios[draw_index]}_ROI.png', dpi=100,
                bbox_inches='tight')
    plt.close()

# 4. Plot Maximum ability-to-pay
#----------------------------------------------------
max_ability_to_pay_for_implementation_summarized = summarize_cost_data(max_ability_to_pay_for_implementation)
def do_bar_plot_with_ci(_df, annotations=None, xticklabels_horizontal_and_wrapped=False):
    """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
    extent of the error bar."""

    yerr = np.array([
        (_df['mean'] - _df['lower']).values,
        (_df['upper'] - _df['mean']).values,
    ])

    xticks = {(i+1): k for i, k in enumerate(_df.index)}

    fig, ax = plt.subplots()
    ax.bar(
        xticks.keys(),
        _df['mean'].values,
        yerr=yerr,
        alpha=1,
        ecolor='black',
        capsize=10,
        label=xticks.values()
    )
    '''
    if annotations:
        for xpos, ypos, text in zip(xticks.keys(), _df['upper'].values, annotations):
            ax.text(xpos, ypos * 1.05, text, horizontalalignment='center', fontsize=11)

    ax.set_xticks(list(xticks.keys()))
    if not xticklabels_horizontal_and_wrapped:
        wrapped_labs = ["\n".join(textwrap.wrap(_lab, 20)) for _lab in xticks.values()]
        ax.set_xticklabels(wrapped_labs, rotation=45, ha='right', fontsize=10)
    else:
        wrapped_labs = ["\n".join(textwrap.wrap(_lab, 20)) for _lab in xticks.values()]
        ax.set_xticklabels(wrapped_labs, fontsize=10)
    '''

    # Set font size for y-tick labels
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=11)

    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig, ax

# Plot Max ability to pay
name_of_plot = f'Maximum ability to pay, {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}' #f'Maximum ability to pay, {first_year_of_simulation} - {final_year_of_simulation}'
fig, ax = do_bar_plot_with_ci(
    (max_ability_to_pay_for_implementation_summarized / 1e6),
    annotations=[
        f"{round(row['mean']/1e6, 1)} \n ({round(row['lower']/1e6, 1)}-{round(row['upper']/1e6, 1)})"
        for _, row in max_ability_to_pay_for_implementation_summarized.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
)
ax.set_title(name_of_plot)
#ax.set_ylim(0, 120)
#ax.set_yticks(np.arange(0, 120, 10))
ax.set_ylabel('Maximum ability to pay \n(Millions)')
fig.tight_layout()
fig.savefig(figurespath / name_of_plot.replace(' ', '_').replace(',', ''))
fig.show()
plt.close(fig)

'''
#years_with_no_malaria_scaleup = set(TARGET_PERIOD).symmetric_difference(set(TARGET_PERIOD_MALARIA_SCALEUP))
#years_with_no_malaria_scaleup = sorted(list(years_with_no_malaria_scaleup))
#years_with_no_malaria_scaleup =  [i.year for i in years_with_no_malaria_scaleup]
'''
