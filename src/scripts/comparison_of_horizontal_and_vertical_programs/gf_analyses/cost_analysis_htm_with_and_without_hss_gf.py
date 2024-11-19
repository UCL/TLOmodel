"""Produce plots to show the impact each the healthcare system (overall health impact) when running under different
scenarios (scenario_vertical_programs_with_and_without_hss.py)

job ID:
results for FCDO and GF presentations Sept 2024:
htm_with_and_without_hss-2024-09-04T143044Z

results for updates 30Sept2024 (IRS in high-risk distr and reduced gen pop RDT):
htm_with_and_without_hss-2024-09-17T083150Z

with reduced consumables logging
htm_with_and_without_hss-2024-11-12T172503Z
"""

from pathlib import Path
from tlo import Date

import datetime
import os
import textwrap

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize
)

from scripts.costing.cost_estimation import (estimate_input_cost_of_scenarios,
                                             summarize_cost_data,
                                             do_stacked_bar_plot_of_cost_by_category,
                                             do_line_plot_of_cost,
                                             generate_roi_plots,
                                             generate_multiple_scenarios_roi_plot)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/t.mangal@imperial.ac.uk')
figurespath = Path('./outputs/global_fund_roi_analysis/htm_with_and_without_hss')
if not os.path.exists(figurespath):
    os.makedirs(figurespath)
roi_outputs_folder = Path(figurespath / 'roi')
if not os.path.exists(roi_outputs_folder):
    os.makedirs(roi_outputs_folder)

# Load result files
#------------------------------------------------------------------------------------------------------------------
results_folder = get_scenario_outputs('htm_with_and_without_hss-2024-11-12T172503Z.py', outputfilepath)[0]

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0) # look at one log (so can decide what to extract)
params = extract_params(results_folder)
info = get_scenario_info(results_folder)

# Declare default parameters for cost analysis
#------------------------------------------------------------------------------------------------------------------
# Population scaling factor for malaria scale-up projections
population_scaling_factor = log['tlo.methods.demography']['scaling_factor']['scaling_factor'].iloc[0]
# Load the list of districts and their IDs
district_dict = pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')[
    ['District_Num', 'District']].drop_duplicates()
district_dict = dict(zip(district_dict['District_Num'], district_dict['District']))

# Period relevant for costing
TARGET_PERIOD= (Date(2025, 1, 1), Date(2035, 12, 31))  # This is the period that is costed
relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))

# Scenarios
htm_scenarios = {0:"Baseline", 1: "HSS PACKAGE: Realistic", 2: "HIV Programs Scale-up WITHOUT HSS PACKAGE",
3: "HIV Programs Scale-up WITH REALISTIC HSS PACKAGE", 4: "TB Programs Scale-up WITHOUT HSS PACKAGE",
5: "TB Programs Scale-up WITH REALISTIC HSS PACKAGE", 6: "Malaria Programs Scale-up WITHOUT HSS PACKAGE",
7: "Malaria Programs Scale-up WITH REALISTIC HSS PACKAGE", 8: "HTM Programs Scale-up WITHOUT HSS PACKAGE",
9: "HTM Programs Scale-up WITH REALISTIC HSS PACKAGE", 10: "HTM Programs Scale-up WITH SUPPLY CHAINS", 11: "HTM Programs Scale-up WITH HRH"}

htm_scenarios_substitutedict = {0:"0", 1: "A", 2: "B", 3: "C",
4: "D", 5: "E", 6: "F",
7: "G", 8: "H", 9: "I",
10: "J", 11: "K"}

# Subset of scenarios included in analysis
htm_scenarios_for_report = list(range(0,12))

color_map = {
    'Baseline': '#9e0142',
    'HSS PACKAGE: Realistic': '#d8434e',
    'HIV Programs Scale-up WITHOUT HSS PACKAGE': '#f36b48',
    'HIV Programs Scale-up WITH REALISTIC HSS PACKAGE': '#fca45c',
    'TB Programs Scale-up WITHOUT HSS PACKAGE': '#fddc89',
    'TB Programs Scale-up WITH REALISTIC HSS PACKAGE': '#e7f7a0',
    'Malaria Programs Scale-up WITHOUT HSS PACKAGE': '#a5dc97',
    'Malaria Programs Scale-up WITH REALISTIC HSS PACKAGE': '#6dc0a6',
    'HTM Programs Scale-up WITHOUT HSS PACKAGE': '#438fba',
    'HTM Programs Scale-up WITH REALISTIC HSS PACKAGE': '#5e4fa2',
    'HTM Programs Scale-up WITH SUPPLY CHAINS': '#3c71aa',
    'HTM Programs Scale-up WITH HRH': '#2f6094',
}

# Cost-effectiveness threshold
chosen_cet = 199.620811947318 # This is based on the estimate from Lomas et al (2023)- $160.595987085533 in 2019 USD coverted to 2023 USD
# based on Ochalek et al (2018) - the paper provided the value $61 in 2016 USD terms, this value is $77.4 in 2023 USD terms
chosen_value_of_statistical_life = 834

# Discount rate
discount_rate = 0.03

# Define a function to create bar plots
def do_bar_plot_with_ci(_df, annotations=None, xticklabels_horizontal_and_wrapped=False):
    """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
    extent of the error bar."""

    # Calculate y-error bars
    yerr = np.array([
        (_df['mean'] - _df['lower']).values,
        (_df['upper'] - _df['mean']).values,
    ])

    # Map xticks based on the hss_scenarios dictionary
    xticks = {index: htm_scenarios.get(index, f"Scenario {index}") for index in _df.index}

    # Retrieve colors from color_map based on the xticks labels
    colors = [color_map.get(label, '#333333') for label in xticks.values()]  # default to grey if not found

    # Generate consecutive x positions for the bars, ensuring no gaps
    x_positions = np.arange(len(xticks))  # Consecutive integers for each bar position

    fig, ax = plt.subplots()
    ax.bar(
        x_positions,
        _df['mean'].values,
        yerr=yerr,
        color=colors,  # Set bar colors
        alpha=1,
        ecolor='black',
        capsize=10,
    )

    # Add optional annotations above each bar
    if annotations:
        for xpos, ypos, text in zip(x_positions, _df['upper'].values, annotations):
            ax.text(xpos, ypos * 1.05, text, horizontalalignment='center', fontsize=8)

    # Set x-tick labels with wrapped text if required
    wrapped_labs = ["\n".join(textwrap.wrap(label,30)) for label in xticks.values()]
    ax.set_xticks(x_positions)  # Set x-ticks to consecutive positions
    ax.set_xticklabels(wrapped_labs, rotation=45 if not xticklabels_horizontal_and_wrapped else 0, ha='right',
                       fontsize=7)

    # Set y-axis limit to upper max + 500
    ax.set_ylim(_df['lower'].min()*1.25, _df['upper'].max()*1.25)

    # Set font size for y-tick labels and grid
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=9)

    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig, ax

def do_standard_bar_plot_with_ci(_df, set_colors=None, annotations=None,
                        xticklabels_horizontal_and_wrapped=False,
                        put_labels_in_legend=True,
                        offset=1e6):
    """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
     extent of the error bar."""

    substitute_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    yerr = np.array([
        (_df['mean'] - _df['lower']).values,
        (_df['upper'] - _df['mean']).values,
    ])
# TODO should be above be 'median'
    xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

    if set_colors:
        colors = [color_map.get(series, 'grey') for series in _df.index]
    else:
        cmap = sns.color_palette('Spectral', as_cmap=True)
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
        colors = list(map(cmap, rescale(np.array(list(xticks.keys()))))) if put_labels_in_legend else None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        xticks.keys(),
        _df['mean'].values,
        yerr=yerr,
        ecolor='black',
        color=colors,
        capsize=10,
        label=xticks.values()
    )

    if annotations:
        for xpos, (ypos, text) in zip(xticks.keys(), zip(_df['upper'].values.flatten(), annotations)):
            annotation_y = ypos + offset

            ax.text(
                xpos,
                annotation_y,
                '\n'.join(text.split(' ', 1)),
                horizontalalignment='center',
                verticalalignment='bottom',  # Aligns text at the bottom of the annotation position
                fontsize='x-small',
                rotation='horizontal'
            )

    ax.set_xticks(list(xticks.keys()))

    if put_labels_in_legend:
        # Update xticks label with substitute labels
        # Insert legend with updated labels that shows correspondence between substitute label and original label
        xtick_values = [letter for letter, label in zip(substitute_labels, xticks.values())]
        xtick_legend = [f'{letter}: {label}' for letter, label in zip(substitute_labels, xticks.values())]
        h, legs = ax.get_legend_handles_labels()
        ax.legend(h, xtick_legend, loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
        ax.set_xticklabels(list(xtick_values))
    else:
        if not xticklabels_horizontal_and_wrapped:
            # xticklabels will be vertical and not wrapped
            ax.set_xticklabels(list(xticks.values()), rotation=90)
        else:
            wrapped_labs = ["\n".join(textwrap.wrap(_lab, 20)) for _lab in xticks.values()]
            ax.set_xticklabels(wrapped_labs)

    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.15, right=0.85)  # Adjust left and right margins

    return fig, ax

# Estimate standard input costs of scenario
#-----------------------------------------------------------------------------------------------------------------------
input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
                                               _years= list_of_relevant_years_for_costing, cost_only_used_staff= True,
                                               _discount_rate = discount_rate)

# TODO Remove the manual fix below once the logging for these is corrected
# Post-run fixes to costs due to challenges with calibration
input_costs.loc[input_costs.cost_subgroup == 'Oxygen, 1000 liters, primarily with oxygen cylinders', 'cost'] = \
    input_costs.loc[input_costs.cost_subgroup == 'Oxygen, 1000 liters, primarily with oxygen cylinders', 'cost']/10
#input_costs = apply_discounting_to_cost_data(input_costs, _discount_rate = discount_rate)

# Add additional costs pertaining to simulation (Only for scenarios with Malaria scale-up)
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
resource_mapping_data[resource_mapping_data['Cost Type'] == 'Drugs and Commodities'][expenditure_column].sum()[0] + \
resource_mapping_data[resource_mapping_data['Cost Type'] == 'HIV Drugs and Commodities'][expenditure_column].sum()[0]
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
year_of_malaria_scaleup_start = list_of_draws_with_malaria_scaleup_parameters.loc[:,'value'].reset_index()['value'][0]
final_year_for_costing = max(list_of_relevant_years_for_costing)
TARGET_PERIOD_MALARIA_SCALEUP = (Date(year_of_malaria_scaleup_start, 1, 1), Date(final_year_for_costing, 12, 31))

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
inflation_2011_to_2023 = 1.35
unit_cost_of_bednet = unit_price_consumable[unit_price_consumable.Item_Code == 160]['Final_price_per_chosen_unit (USD, 2023)'] + (8.27 - 3.36) * inflation_2011_to_2023
# Stelmach et al Tanzania https://pmc.ncbi.nlm.nih.gov/articles/PMC6169190/#_ad93_ (Price in 2011 USD) - This cost includes non-consumable costs - personnel, equipment, fuel, logistics and planning, shipping. The cost is measured per net distributed
# Note that the cost per net of $3.36 has been replaced with a cost of Malawi Kwacha 667 (2023) as per the Central Medical Stores Trust sales catalogue

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

# Extract input_costs for browsing
input_costs.groupby(['draw', 'run', 'cost_category', 'cost_subcategory', 'cost_subgroup','year'])['cost'].sum().to_csv(figurespath / 'cost_detailed.csv')

# %%
# Return on Invesment analysis
# Calculate incremental cost
# -----------------------------------------------------------------------------------------------------------------------
# Aggregate input costs for further analysis (this step is needed because the malaria specific scale-up costs start from the year or malaria scale-up implementation)
input_costs_subset = input_costs[
    (input_costs['year'] >= relevant_period_for_costing[0]) & (input_costs['year'] <= relevant_period_for_costing[1])]
total_input_cost = input_costs_subset.groupby(['draw', 'run'])['cost'].sum()
total_input_cost_summarized = summarize_cost_data(total_input_cost.unstack(level='run'))
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


incremental_scenario_cost = (pd.DataFrame(
    find_difference_relative_to_comparison(
        total_input_cost,
        comparison=0)  # sets the comparator to 0 which is the Actual scenario
).T.iloc[0].unstack()).T

# Keep only scenarios of interest
incremental_scenario_cost = incremental_scenario_cost[
    incremental_scenario_cost.index.get_level_values(0).isin(htm_scenarios_for_report)]

# Monetary value of health impact
# -----------------------------------------------------------------------------------------------------------------------
def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = relevant_period_for_costing
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    _df = _df.loc[_df.year.between(*years_needed)].drop(columns=['date', 'sex', 'age_range']).groupby('year').sum().sum(axis = 1)

    # Initial year and discount rate
    initial_year = min(_df.index.unique())

    # Calculate the discounted values
    discounted_values = _df / (1 + discount_rate) ** (_df.index - initial_year)

    return pd.Series(discounted_values.sum())

num_dalys = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_num_dalys,
    do_scaling=True
)

# Get absolute DALYs averted
num_dalys_averted = (-1.0 *
                     pd.DataFrame(
                         find_difference_relative_to_comparison(
                             num_dalys.loc[0],
                             comparison=0)  # sets the comparator to 0 which is the Actual scenario
                     ).T.iloc[0].unstack(level='run'))
num_dalys_averted = num_dalys_averted[num_dalys_averted.index.get_level_values(0).isin(htm_scenarios_for_report)]

# The monetary value of the health benefit is delta health times CET (negative values are set to 0)
def get_monetary_value_of_incremental_health(_num_dalys_averted, _chosen_value_of_life_year):
    monetary_value_of_incremental_health = (_num_dalys_averted * _chosen_value_of_life_year).clip(lower=0.0)
    return monetary_value_of_incremental_health

# TODO check that the above calculation is correct

# 3. Return on Investment Plot
# ----------------------------------------------------
# Combined ROI plot of relevant scenarios
# HTM scenarios
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [1,8,9,10,11],
                   _scenario_dict = htm_scenarios,
                   _outputfilepath=roi_outputs_folder,
                   _value_of_life_suffix = 'all_HTM_VSL')

# HIV scenarios
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [2,3],
                   _scenario_dict = htm_scenarios,
                   _outputfilepath=roi_outputs_folder,
                   _value_of_life_suffix = 'HIV_VSL')

# TB scenarios
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [4,5],
                   _scenario_dict = htm_scenarios,
                   _outputfilepath=roi_outputs_folder,
                   _value_of_life_suffix = 'TB_VSL')

# Malaria scenarios
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [6,7],
                   _scenario_dict = htm_scenarios,
                   _outputfilepath=roi_outputs_folder,
                   _value_of_life_suffix = 'Malaria_VSL')

# 4. Plot Maximum ability-to-pay at CET
# ----------------------------------------------------
max_ability_to_pay_for_implementation = (get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_cet) - incremental_scenario_cost).clip(
    lower=0.0)  # monetary value - change in costs
max_ability_to_pay_for_implementation_summarized = summarize_cost_data(max_ability_to_pay_for_implementation)
max_ability_to_pay_for_implementation_summarized = max_ability_to_pay_for_implementation_summarized[
    max_ability_to_pay_for_implementation_summarized.index.get_level_values(0).isin(htm_scenarios_for_report)]

# Plot Maximum ability to pay
name_of_plot = f'Maximum ability to pay at CET, {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (max_ability_to_pay_for_implementation_summarized / 1e6),
    annotations=[
        f"{round(row['mean'] / 1e6, 1)} \n ({round(row['lower'] / 1e6, 1)}-\n {round(row['upper'] / 1e6, 1)})"
        for _, row in max_ability_to_pay_for_implementation_summarized.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
)
ax.set_title(name_of_plot)
ax.set_ylabel('Maximum ability to pay \n(Millions)')
fig.tight_layout()
fig.savefig(roi_outputs_folder / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)

# Plot incremental costs
incremental_scenario_cost_summarized = summarize_cost_data(incremental_scenario_cost)
name_of_plot = f'Incremental scenario cost relative to baseline {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (incremental_scenario_cost_summarized / 1e6),
    annotations=[
        f"{round(row['mean'] / 1e6, 1)} \n ({round(row['lower'] / 1e6, 1)}- \n {round(row['upper'] / 1e6, 1)})"
        for _, row in incremental_scenario_cost_summarized.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
)
ax.set_title(name_of_plot)
ax.set_ylabel('Cost \n(USD Millions)')
fig.tight_layout()
fig.savefig(roi_outputs_folder / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)

# 4. Plot costs
# ----------------------------------------------------
input_costs_for_plot = input_costs_subset[input_costs_subset.draw.isin(htm_scenarios_for_report)]
# First summarize all input costs
input_costs_for_plot_summarized = input_costs_for_plot.groupby(['draw', 'year', 'cost_subcategory', 'Facility_Level', 'cost_subgroup', 'cost_category']).agg(
    mean=('cost', 'mean'),
    lower=('cost', lambda x: x.quantile(0.025)),
    upper=('cost', lambda x: x.quantile(0.975))
).reset_index()
input_costs_for_plot_summarized = input_costs_for_plot_summarized.melt(
    id_vars=['draw', 'year', 'cost_subcategory', 'Facility_Level', 'cost_subgroup', 'cost_category'],
    value_vars=['mean', 'lower', 'upper'],
    var_name='stat',
    value_name='cost'
)

do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'all', _year = list(range(2025, 2036)), _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'all', _year = [2025],  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'human resources for health',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'medical consumables',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'medical equipment',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'other',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
