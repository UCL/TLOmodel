"""
This script runs the generates results for the Horizontal versus Vertical investments paper.
The latest job_ID used for the analysis is -
Completed in Jan 2025:
htm_and_hss_runs-2025-01-16T135243Z
This is generated by ~/src/scripts/comparison_of_horizontal_and_vertical_programs/manuscript_analyses/scenario_hss_htm_paper.py
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
                                             generate_multiple_scenarios_roi_plot,
                                             estimate_projected_health_spending,
                                             apply_discounting_to_cost_data)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/t.mangal@imperial.ac.uk')
figurespath = Path('./outputs/horizontal_v_vertical')
if not os.path.exists(figurespath):
    os.makedirs(figurespath)
roi_outputs_folder = Path(figurespath / 'roi')
if not os.path.exists(roi_outputs_folder):
    os.makedirs(roi_outputs_folder)

# Load result files
#------------------------------------------------------------------------------------------------------------------
results_folder = get_scenario_outputs('htm_and_hss_runs-2025-01-16T135243Z.py', outputfilepath)[0]

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
htm_scenarios = {0:"Baseline", 8: "HSS Expansion Package",
                 9: "HIV Program Scale-up Without HSS Expansion", 17: "HIV Programs Scale-up With HSS Expansion Package",
                 18: "TB Program Scale-up Without HSS Expansion", 26: "TB Programs Scale-up With HSS Expansion Package",
                 27: "Malaria Program Scale-up Without HSS Expansion", 35: "Malaria Programs Scale-up With HSS Expansion Package",
                 36: "HTM Program Scale-up Without HSS Expansion",39: "HTM Program Scale-up With HRH Scale-up (6%)",
                 41: "HTM Program Scale-up With Consumables at 75th Percentile", 44: "HTM Programs Scale-up With HSS Expansion Package"}

# Use letters instead of full scenario name for figures
htm_scenarios_substitutedict = {0:"0", 8: "A", 9: "B", 17: "C",
18: "D", 26: "E", 27: "F",
35: "G", 36: "H", 39: "I",
41: "J", 44: "K"}

color_map = {
    'Baseline': '#9e0142',
    'HSS Expansion Package': '#d8434e',
    'HIV Program Scale-up Without HSS Expansion': '#f36b48',
    'HIV Programs Scale-up With HSS Expansion Package': '#fca45c',
    'TB Program Scale-up Without HSS Expansion': '#fddc89',
    'TB Programs Scale-up With HSS Expansion Package': '#e7f7a0',
    'Malaria Program Scale-up Without HSS Expansion': '#a5dc97',
    'Malaria Programs Scale-up With HSS Expansion Package': '#6dc0a6',
    'HTM Program Scale-up Without HSS Expansion': '#438fba',
    'HTM Programs Scale-up With HSS Expansion Package': '#5e4fa2',
    'HTM Program Scale-up With Consumables at 75th Percentile': '#3c71aa',
    'HTM Program Scale-up With HRH Scale-up (6%)': '#2f6094',
}

# Cost-effectiveness threshold
chosen_cet = 199.620811947318 # This is based on the estimate from Lomas et al (2023)- $160.595987085533 in 2019 USD coverted to 2023 USD
# based on Ochalek et al (2018) - the paper provided the value $61 in 2016 USD terms, this value is $77.4 in 2023 USD terms
chosen_value_of_statistical_life = 834 # This is based on Munthali et al (2020) National Planning Commission Report on
#"Medium and long-term impacts of a moderate lockdown (social restrictions) in response to the COVID-19 pandemic in Malawi"

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
        # Use htm_scenarios for the legend
        xtick_legend = [f'{letter}: {htm_scenarios.get(label, label)}' for letter, label in zip(substitute_labels, xticks.values())]
        xtick_values = [letter for letter, label in zip(substitute_labels, xticks.values())]

        h, legs = ax.get_legend_handles_labels()
        ax.legend(h, xtick_legend, loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
        ax.set_xticklabels(xtick_values)
    else:
        if not xticklabels_horizontal_and_wrapped:
            # xticklabels will be vertical and not wrapped
            ax.set_xticklabels(list(xticks.values()), rotation=90)
        else:
            wrapped_labs = ["\n".join(textwrap.wrap(_lab, 20)) for _lab in xticks.values()]
            ax.set_xticklabels(wrapped_labs)

    # Extend ylim to accommodate data labels
    ymin, ymax = ax.get_ylim()
    extension = 0.1 * (ymax - ymin) # 10% of range
    ax.set_ylim(ymin - extension, ymax + extension) # Set new y-axis limits with the extended range

    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #fig.tight_layout()
    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.15, right=0.85)  # Adjust left and right margins

    return fig, ax

# Estimate standard input costs of scenario
#-----------------------------------------------------------------------------------------------------------------------
input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
                                               _years= list_of_relevant_years_for_costing, cost_only_used_staff= True,
                                               _discount_rate = discount_rate, _draws = list(htm_scenarios.keys()))

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
def get_total_population_by_year(_df):
    years_needed = [i.year for i in TARGET_PERIOD_MALARIA_SCALEUP]  # Malaria scale-up period years
    _df['year'] = pd.to_datetime(_df['date']).dt.year

    # Validate that all necessary years are in the DataFrame
    if not set(years_needed).issubset(_df['year'].unique()):
        raise ValueError("Some years are not recorded in the dataset.")

    # Filter for relevant years and return the total population as a Series
    return _df.loc[_df['year'].between(min(years_needed), max(years_needed)), ['year', 'total']].set_index('year')[
        'total']


# Extract results with custom function
total_population_by_year = extract_results(
    results_folder,
    module='tlo.methods.demography',
    key='population',
    custom_generate_series=get_total_population_by_year,
    do_scaling=True
)

# Replicate population estimates for each district
district_ids = list(range(32))
replicated_population_by_year = pd.concat([total_population_by_year] * 32, axis=0)
replicated_population_by_year['District_Num'] = np.array(sorted(district_ids * (final_year_for_costing - year_of_malaria_scaleup_start + 1)), dtype=np.int64) # attach district number to each replicate
replicated_population_by_year = replicated_population_by_year.reset_index()

# Load proportional population distribution
population_2010 = pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')
population_proportion_by_district_2010 = (
    population_2010.groupby('District_Num')['Count']
    .sum()
    .pipe(lambda x: x / x.sum())  # Compute proportions
)
assert (population_proportion_by_district_2010.sum() == 1)

# Merge and compute district-level population by year
district_population_by_year = replicated_population_by_year.merge(population_proportion_by_district_2010, on='District_Num', how='left', validate='m:1')
district_population_by_year[total_population_by_year.columns]= district_population_by_year[total_population_by_year.columns].multiply(district_population_by_year['Count'], axis=0)
district_population_by_year = district_population_by_year.drop(columns = ['District_Num', 'Count'])

# Set multi-level columns and final formatting
district_population_by_year = (
    district_population_by_year
    .set_axis(pd.MultiIndex.from_tuples(district_population_by_year.columns, names=['draw', 'run']), axis=1)
    .rename(columns={'District_Num': 'district'})
    .set_index(['year', 'district'])
)
district_population_by_year.columns = pd.MultiIndex.from_tuples(district_population_by_year.columns)
district_population_by_year.columns.names = ['draw', 'run']

def get_number_of_people_covered_by_malaria_scaleup(_df, list_of_districts_covered = None, draws_included = None):
    _df = pd.DataFrame(_df)
    # Reset the index to make 'district' a column
    _df = _df.reset_index()
    # Convert the 'district' column to numeric values
    _df['district'] = pd.to_numeric(_df['district'], errors='coerce')
    _df = _df.set_index(['year', 'district'])
    # Zero out rows for districts not in the specified list
    if list_of_districts_covered is not None:
        mask = _df.index.get_level_values('district').isin(list_of_districts_covered)
        _df.loc[~mask, :] = 0  # Use mask to zero out unwanted rows

    # Zero out columns for draws not in the specified list
    if draws_included is not None:
        mask = _df.columns.get_level_values('draw').isin(draws_included)
        _df.loc[:, ~mask] = 0  # Use mask to zero out unwanted columns
    return _df

district_population_covered_by_irs_scaleup_by_year = get_number_of_people_covered_by_malaria_scaleup(district_population_by_year,
                                                                                                 list_of_districts_covered=district_keys_with_irs_scaleup,
                                                                                                 draws_included = list_of_draws_with_malaria_scaleup_implemented_in_costing_period)

irs_cost_per_person = unit_price_consumable[unit_price_consumable.Item_Code == 161]['Final_price_per_chosen_unit (USD, 2023)']
# This cost includes non-consumable costs - personnel, equipment, fuel, logistics and planning, shipping, PPE. The cost is measured per person protected. Based on Stelmach et al (2018)
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
    melted_df['cost_category'] = 'malaria scale-up'
    melted_df['cost_subgroup'] = 'NA'
    melted_df['Facility_Level'] = 'all'
    melted_df = melted_df.rename(columns={'value': 'cost'})
    return melted_df

# Iterate through additional costs, melt and concatenate
for df, label in malaria_scaleup_costs:
    new_df = melt_and_label_malaria_scaleup_cost(df, label)
    new_df = new_df[new_df['year'].isin(list_of_relevant_years_for_costing)]
    new_df = apply_discounting_to_cost_data(new_df, _discount_rate= discount_rate, _year = relevant_period_for_costing[0])
    input_costs = pd.concat([input_costs, new_df], ignore_index=True)

# Extract input_costs for browsing
input_costs = input_costs[input_costs['draw'].isin(list(htm_scenarios.keys()))]
input_costs.groupby(['draw', 'run', 'cost_category', 'cost_subcategory', 'cost_subgroup','year'])['cost'].sum().to_csv(figurespath / 'cost_detailed.csv')

# %%
# Return on Invesment analysis
# 1. Calculate incremental cost
# -----------------------------------------------------------------------------------------------------------------------
# Extract detailed input_costs
input_costs.groupby(['draw', 'run', 'cost_category', 'year'])['cost'].sum().to_csv(figurespath / 'detailed_costs_2025-2035.csv')

total_input_cost = input_costs.groupby(['draw', 'run'])['cost'].sum()
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
        comparison=0)  # sets the comparator to draw 0 which is the Actual scenario
).T.iloc[0].unstack()).T

# 2. Monetary value of health impact
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
num_dalys_averted = num_dalys_averted[num_dalys_averted.index.get_level_values('draw').isin(list(htm_scenarios.keys()))] # keep only relevant draws

# Plot DALYs
num_dalys_averted_summarised = summarize_cost_data(num_dalys_averted)
name_of_plot = f'Incremental DALYs averted compared to baseline {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (num_dalys_averted_summarised / 1e6),
    annotations=[
        f"{row['mean']/ 1e6:.2f} ({row['lower'] / 1e6 :.2f}- {row['upper'] / 1e6:.2f})"
        for _, row in num_dalys_averted_summarised.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
    put_labels_in_legend=True,
    offset=2,
)
ax.set_title(name_of_plot)
ax.set_ylabel('DALYs \n(Millions)')
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(roi_outputs_folder / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)

# The monetary value of the health benefit is delta health times CET (negative values are set to 0)
def get_monetary_value_of_incremental_health(_num_dalys_averted, _chosen_value_of_life_year):
    monetary_value_of_incremental_health = (_num_dalys_averted * _chosen_value_of_life_year).clip(lower=0.0)
    return monetary_value_of_incremental_health

# TODO check that the above calculation is correct

# 3. Estimate and plot ICERs
# ----------------------------------------------------
icers = incremental_scenario_cost.div(num_dalys_averted)  # Element-wise division
icers = icers.mask(num_dalys_averted < 0)
icers_summarized = summarize_cost_data(icers)

# Plot ICERs
name_of_plot = f'Incremental cost-effectiveness ratios (ICERs), {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (icers_summarized),
    annotations=[
        f"{row['mean']:.2f} ({row['lower'] :.2f}- \n {row['upper'] :.2f})"
        for _, row in icers_summarized.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
    put_labels_in_legend=True,
    offset=10,
)
ax.set_title(name_of_plot)
ax.set_ylabel('ICERs \n($/DALY averted)')
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(roi_outputs_folder / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)

# 4. Return on Investment Plot
# ----------------------------------------------------
projected_health_spending = estimate_projected_health_spending(resourcefilepath,
                                  results_folder,
                                 _years = list_of_relevant_years_for_costing,
                                 _discount_rate = discount_rate,
                                 _summarize = True)
projected_health_spending_baseline = projected_health_spending[projected_health_spending.index.get_level_values(0) == 0]['mean'][0]

# Combined ROI plot of relevant scenarios
htm_scenarios = {0:"Baseline", 8: "HSS Expansion Package",
                 9: "HIV Program Scale-up Without HSS Expansion", 17: "HIV Programs Scale-up With HSS Expansion Package",
                 18: "TB Program Scale-up Without HSS Expansion", 26: "TB Programs Scale-up With HSS Expansion Package",
                 27: "Malaria Program Scale-up Without HSS Expansion", 35: "Malaria Programs Scale-up With HSS Expansion Package",
                 36: "HTM Program Scale-up Without HSS Expansion",44: "HTM Programs Scale-up With HSS Expansion Package",
                 41: "HTM Program Scale-up With Consumables at 75th Percentile", 39: "HTM Program Scale-up With HRH Scale-up (6%)"}

# HTM with HSS versus HSS alone
draw_colors = {8: '#438FBA', 44:'#5E4FA2'}
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [8, 44],
                   _scenario_dict = htm_scenarios,
                   _outputfilepath=roi_outputs_folder,
                   _value_of_life_suffix = 'HSS_VSL',
                   _plot_vertical_lines_at = [0, 1e9, 3e9],
                    _year_suffix= f' ({str(relevant_period_for_costing[0])} - {str(relevant_period_for_costing[1])})',
                    _projected_health_spending = projected_health_spending_baseline,
                   _draw_colors = draw_colors)

# HTM scenarios with and without HSS
draw_colors = {36: '#438FBA', 44:'#5E4FA2'}
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [36, 44],
                   _scenario_dict = htm_scenarios,
                   _outputfilepath=roi_outputs_folder,
                   _value_of_life_suffix = 'HTM_VSL',
                   _plot_vertical_lines_at = [0, 1e9, 3e9],
                    _year_suffix= f' ({str(relevant_period_for_costing[0])}- {str(relevant_period_for_costing[1])})',
                   _projected_health_spending = projected_health_spending_baseline,
                   _draw_colors = draw_colors)

draw_colors = {36: '#438FBA', 44:'#5E4FA2'}
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [8, 36, 44],
                   _scenario_dict = htm_scenarios,
                   _outputfilepath=roi_outputs_folder,
                   _value_of_life_suffix = 'HSS_VSL',
                   _plot_vertical_lines_at = [0, 1e9, 3e9],
                    _year_suffix= f' ({str(relevant_period_for_costing[0])} - {str(relevant_period_for_costing[1])})',
                    _projected_health_spending = projected_health_spending_baseline,
                   _draw_colors = draw_colors)

# HIV scenarios with and without HSS
draw_colors = {9: '#438FBA', 17:'#5E4FA2'}
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [9,17],
                   _scenario_dict = htm_scenarios,
                   _outputfilepath=roi_outputs_folder,
                   _year_suffix=f' ({str(relevant_period_for_costing[0])}- {str(relevant_period_for_costing[1])})',
                   _value_of_life_suffix = 'HIV_VSL',
                   _draw_colors = draw_colors)

# TB scenarios with and without HSS
draw_colors = {18: '#438FBA', 26:'#5E4FA2'}
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [18,26],
                   _scenario_dict = htm_scenarios,
                   _outputfilepath=roi_outputs_folder,
                   _year_suffix=f' ({str(relevant_period_for_costing[0])}- {str(relevant_period_for_costing[1])})',
                   _value_of_life_suffix = 'TB_VSL',
                   _draw_colors = draw_colors,
                   _y_axis_lim = 30)

# Malaria scenarios with and without HSS
draw_colors = {27: '#438FBA', 35:'#5E4FA2'}
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [27,35],
                   _scenario_dict = htm_scenarios,
                   _outputfilepath=roi_outputs_folder,
                   _year_suffix=f' ({str(relevant_period_for_costing[0])}- {str(relevant_period_for_costing[1])})',
                   _value_of_life_suffix = 'Malaria_VSL',
                   _draw_colors = draw_colors)

# 5. Plot Maximum ability-to-pay at CET
# ----------------------------------------------------
max_ability_to_pay_for_implementation = (get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_cet) - incremental_scenario_cost).clip(
    lower=0.0)  # monetary value - change in costs
max_ability_to_pay_for_implementation_summarized = summarize_cost_data(max_ability_to_pay_for_implementation)

# Plot Maximum ability to pay
name_of_plot = f'Maximum ability to pay at CET, {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (max_ability_to_pay_for_implementation_summarized / 1e6),
    annotations=[
        f"{row['mean'] / projected_health_spending_baseline :.2%} ({row['lower'] / projected_health_spending_baseline :.2%}- \n {row['upper'] / projected_health_spending_baseline:.2%})"
        for _, row in max_ability_to_pay_for_implementation_summarized.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
    put_labels_in_legend=True,
    offset=50,
)
ax.set_title(name_of_plot)
ax.set_ylabel('Maximum ability to pay \n(Millions)')
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(roi_outputs_folder / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)

# Plot incremental costs
incremental_scenario_cost_summarized = summarize_cost_data(incremental_scenario_cost)
name_of_plot = f'Incremental scenario cost relative to baseline {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (incremental_scenario_cost_summarized / 1e6),
    annotations=[
        f"{row['mean'] / projected_health_spending_baseline :.2%} ({row['lower'] / projected_health_spending_baseline :.2%}- {row['upper'] / projected_health_spending_baseline:.2%})"
        for _, row in incremental_scenario_cost_summarized.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
    put_labels_in_legend=True,
    offset=50,
)
ax.set_title(name_of_plot)
ax.set_ylabel('Cost \n(USD Millions)')
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(roi_outputs_folder / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)

# 6. Plot costs
# ----------------------------------------------------
# First summarize all input costs
input_costs_for_plot_summarized = input_costs.groupby(['draw', 'year', 'cost_subcategory', 'Facility_Level', 'cost_subgroup', 'cost_category']).agg(
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

do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'all', _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'all', _year = [2025],  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'human resources for health',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'medical consumables',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'medical equipment',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'malaria scale-up',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = htm_scenarios_substitutedict)

# Plost costs over time
# First remove discounting
def remove_discounting(_df, _discount_rate=0, _year = None):
    if _year == None:
        # Initial year and discount rate
        initial_year = min(_df['year'].unique())
    else:
        initial_year = _year

    # Calculate the discounted values
    _df.loc[:, 'cost'] = _df['cost'] * ((1 + _discount_rate) ** (_df['year'] - initial_year))
    return _df
input_costs_for_plot_summarized_undiscounted = remove_discounting(input_costs_for_plot_summarized,
                                                                  _discount_rate = discount_rate)


# Baseline
do_line_plot_of_cost(_df = input_costs_for_plot_summarized, _cost_category='all',
                         _year=list_of_relevant_years_for_costing, _draws= [0],
                         disaggregate_by= 'cost_category',
                         _outputfilepath = figurespath)

# HSS alone
do_line_plot_of_cost(_df = input_costs_for_plot_summarized, _cost_category='all',
                         _year=list_of_relevant_years_for_costing, _draws= [8],
                         disaggregate_by= 'cost_category',
                         _outputfilepath = figurespath)

# HTM without HSS
do_line_plot_of_cost(_df = input_costs_for_plot_summarized, _cost_category='all',
                         _year=list_of_relevant_years_for_costing, _draws= [36],
                         disaggregate_by= 'cost_category',
                         _outputfilepath = figurespath)

# HTM with HSS
do_line_plot_of_cost(_df = input_costs_for_plot_summarized, _cost_category='all',
                         _year=list_of_relevant_years_for_costing, _draws= [44],
                         disaggregate_by= 'cost_category',
                         _outputfilepath = figurespath)
