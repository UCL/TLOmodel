"""Produce plots to show the impact each the healthcare system (overall health impact) when running under different
scenarios (scenario_impact_of_healthsystem.py)

with reduced consumables logging
/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/hss_elements-2024-11-12T172311Z
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
                                             estimate_projected_health_spending)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/t.mangal@imperial.ac.uk')
figurespath = Path('./outputs/global_fund_roi_analysis/hss_elements/')
if not os.path.exists(figurespath):
    os.makedirs(figurespath)
roi_outputs_folder_gf = Path(figurespath / 'gf/roi')
if not os.path.exists(roi_outputs_folder_gf):
    os.makedirs(roi_outputs_folder_gf)
roi_outputs_folder_fcdo = Path(figurespath / 'fcdo/roi')
if not os.path.exists(roi_outputs_folder_fcdo):
    os.makedirs(roi_outputs_folder_fcdo)

# Load result files
# ------------------------------------------------------------------------------------------------------------------
results_folder = get_scenario_outputs('hss_elements-2024-11-12T172311Z.py', outputfilepath)[0]

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0)  # look at one log (so can decide what to extract)
params = extract_params(results_folder)

# Declare default parameters for cost analysis
# ------------------------------------------------------------------------------------------------------------------
# Period relevant for costing
TARGET_PERIOD_INTERVENTION = (Date(2025, 1, 1), Date(2035, 12, 31))  # This is the period that is costed
relevant_period_for_costing = [i.year for i in TARGET_PERIOD_INTERVENTION]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))

# Scenarios
hss_scenarios = {0: "Baseline", 1: "HRH Moderate Scale-up (1%)", 2: "HRH Scale-up Following Historical Growth", 3: "HRH Accelerated Scale-up (6%)",
                 4: "Increase Capacity at Primary Care Levels", 5: "Consumables Increased to 75th Percentile",
                 6: "Consumables Available at HIV levels", 7: "Consumables Available at EPI levels", 8: "HSS PACKAGE: Realistic expansion"}
hs_scenarios_substitutedict = {0:"0", 1: "A", 2: "B", 3: "C",
4: "D", 5: "E", 6: "F",
7: "G", 8: "H"}
hss_scenarios_for_report = [0, 1, 2, 3, 4, 5, 6, 7, 8]
color_map = {
    'Baseline': '#a50026',
    'HRH Moderate Scale-up (1%)': '#d73027',
    'HRH Scale-up Following Historical Growth': '#f46d43',
    'HRH Accelerated Scale-up (6%)': '#fdae61',
    'Increase Capacity at Primary Care Levels': '#fee08b',
    'Consumables Increased to 75th Percentile': '#d9ef8b',
    'Consumables Available at HIV levels': '#a6d96a',
    'Consumables Available at EPI levels': '#66bd63',
    'HSS PACKAGE: Realistic expansion': '#3288bd'
}

# Cost-effectiveness threshold
chosen_cet = 199.620811947318  # This is based on the estimate from Lomas et al (2023)- $160.595987085533 in 2019 USD coverted to 2023 USD
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
    xticks = {index: hss_scenarios.get(index, f"Scenario {index}") for index in _df.index}

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
    wrapped_labs = ["\n".join(textwrap.wrap(label, 25)) for label in xticks.values()]
    ax.set_xticks(x_positions)  # Set x-ticks to consecutive positions
    ax.set_xticklabels(wrapped_labs, rotation=45 if not xticklabels_horizontal_and_wrapped else 0, ha='right',
                       fontsize=8)

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
        xtick_legend = [f'{letter}: {hss_scenarios.get(label, label)}' for letter, label in zip(substitute_labels, xticks.values())]
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

    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #fig.tight_layout()
    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.15, right=0.85)  # Adjust left and right margins

    return fig, ax

# Estimate standard input costs of scenario
# -----------------------------------------------------------------------------------------------------------------------
input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
                                               _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
                                               _discount_rate = discount_rate)
# _draws = htm_scenarios_for_gf_report --> this subset is created after calculating malaria scale up costs
# TODO Remove the manual fix below once the logging for these is corrected
input_costs.loc[input_costs.cost_subgroup == 'Oxygen, 1000 liters, primarily with oxygen cylinders', 'cost'] = \
    input_costs.loc[input_costs.cost_subgroup == 'Oxygen, 1000 liters, primarily with oxygen cylinders', 'cost']/10
#input_costs.loc[input_costs.cost_subgroup == 'Depot-Medroxyprogesterone Acetate 150 mg - 3 monthly', 'cost'] =\
#    input_costs.loc[input_costs.cost_subgroup == 'Depot-Medroxyprogesterone Acetate 150 mg - 3 monthly', 'cost']/7
#input_costs = apply_discounting_to_cost_data(input_costs, _discount_rate = discount_rate)

# %%
# Return on Invesment analysis
# Calculate incremental cost
# -----------------------------------------------------------------------------------------------------------------------
# Aggregate input costs for further analysis
input_costs_subset = input_costs[
    (input_costs['year'] >= relevant_period_for_costing[0]) & (input_costs['year'] <= relevant_period_for_costing[1])]
# TODO the above step may not longer be needed
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

# Monetary value of health impact
# -----------------------------------------------------------------------------------------------------------------------
def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = relevant_period_for_costing  # [i.year for i in TARGET_PERIOD_INTERVENTION]
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
num_dalys_averted_fcdo_scenarios = num_dalys_averted[
    num_dalys_averted.index.get_level_values(0).isin(hss_scenarios_for_report)]
num_dalys_averted_gf_scenarios = num_dalys_averted[
    num_dalys_averted.index.get_level_values(0).isin(hss_scenarios_for_report)]

# The monetary value of the health benefit is delta health times CET (negative values are set to 0)
def get_monetary_value_of_incremental_health(_num_dalys_averted, _chosen_value_of_life_year):
    monetary_value_of_incremental_health = (_num_dalys_averted * _chosen_value_of_life_year).clip(lower=0.0)
    return monetary_value_of_incremental_health

# TODO check that the above calculation is correct

# 3. Return on Investment Plot
# ----------------------------------------------------
projected_health_spending = estimate_projected_health_spending(resourcefilepath,
                                  results_folder,
                                 _years = list_of_relevant_years_for_costing,
                                 _discount_rate = discount_rate,
                                 _summarize = True)
projected_health_spending_baseline = projected_health_spending[projected_health_spending.index.get_level_values(0) == 0]['mean'][0]

# FCDO
# Combined ROI plot of relevant scenarios
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [1,2,3,4],
                   _scenario_dict = hss_scenarios,
                   _outputfilepath=roi_outputs_folder_fcdo,
                   _value_of_life_suffix = 'HR_VSL',
                    _year_suffix= f' ({str(relevant_period_for_costing[0])} - {str(relevant_period_for_costing[1])})',
                    _projected_health_spending = projected_health_spending_baseline)

# Combined ROI plot of relevant scenarios
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [5,6,7],
                   _scenario_dict = hss_scenarios,
                   _outputfilepath=roi_outputs_folder_fcdo,
                   _value_of_life_suffix = 'Consumables_VSL',
                   _y_axis_lim= 100,
                    _year_suffix= f' ({str(relevant_period_for_costing[0])} - {str(relevant_period_for_costing[1])})',
                    _projected_health_spending = projected_health_spending_baseline)

# Combined ROI plot of relevant scenarios
generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health=get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_value_of_statistical_life),
                   _incremental_input_cost=incremental_scenario_cost,
                   _draws = [8],
                   _scenario_dict = hss_scenarios,
                   _outputfilepath=roi_outputs_folder_fcdo,
                   _value_of_life_suffix = 'HSS_VSL',
                    _year_suffix= f' ({str(relevant_period_for_costing[0])} - {str(relevant_period_for_costing[1])})',
                    _projected_health_spending = projected_health_spending_baseline)

# 4. Plot Maximum ability-to-pay at CET
# ----------------------------------------------------
max_ability_to_pay_for_implementation = (get_monetary_value_of_incremental_health(num_dalys_averted, _chosen_value_of_life_year = chosen_cet) - incremental_scenario_cost).clip(
    lower=0.0)  # monetary value - change in costs
max_ability_to_pay_for_implementation_summarized = summarize_cost_data(max_ability_to_pay_for_implementation)
max_ability_to_pay_for_implementation_summarized_fcdo = max_ability_to_pay_for_implementation_summarized[
    max_ability_to_pay_for_implementation_summarized.index.get_level_values(0).isin(hss_scenarios_for_report)]
max_ability_to_pay_for_implementation_summarized_gf = max_ability_to_pay_for_implementation_summarized[
    max_ability_to_pay_for_implementation_summarized.index.get_level_values(0).isin(hss_scenarios_for_report)]

# FCDO
# Plot Maximum ability to pay
name_of_plot = f'Maximum ability to pay at CET, {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (max_ability_to_pay_for_implementation_summarized_fcdo / 1e6),
    annotations=[
        f"{round(row['mean'] / 1e6, 1)} \n ({round(row['lower'] / 1e6, 1)}-\n {round(row['upper'] / 1e6, 1)})"
        for _, row in max_ability_to_pay_for_implementation_summarized_fcdo.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
)
ax.set_title(name_of_plot)
ax.set_ylabel('Maximum ability to pay \n(Millions)')
fig.tight_layout()
fig.savefig(roi_outputs_folder_fcdo / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)

# Global Fund
# Plot Maximum ability to pay
name_of_plot = f'Maximum ability to pay at CET, {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (max_ability_to_pay_for_implementation_summarized_gf / 1e6),
    annotations=[
        f"{round(row['mean'] / 1e6, 1)} \n ({round(row['lower'] / 1e6, 1)}-\n {round(row['upper'] / 1e6, 1)})"
        for _, row in max_ability_to_pay_for_implementation_summarized_gf.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
)
ax.set_title(name_of_plot)
ax.set_ylabel('Maximum ability to pay \n(Millions)')
fig.tight_layout()
fig.savefig(roi_outputs_folder_gf / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)

# Plot incremental costs
incremental_scenario_cost_summarized = summarize_cost_data(incremental_scenario_cost)
# Keep only scenarios of interest
incremental_scenario_cost_summarized_fcdo = incremental_scenario_cost_summarized[
    incremental_scenario_cost_summarized.index.get_level_values(0).isin(hss_scenarios_for_report)]
incremental_scenario_cost_summarized_gf = incremental_scenario_cost_summarized[
    incremental_scenario_cost_summarized.index.get_level_values(0).isin(hss_scenarios_for_report)]

# FCDO
name_of_plot = f'Incremental scenario cost relative to baseline {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (incremental_scenario_cost_summarized_fcdo / 1e6),
    annotations=[
        f"{round(row['mean'] / 1e6, 1)} \n ({round(row['lower'] / 1e6, 1)}- \n {round(row['upper'] / 1e6, 1)})"
        for _, row in incremental_scenario_cost_summarized_fcdo.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
)
ax.set_title(name_of_plot)
ax.set_ylabel('Cost \n(USD Millions)')
fig.tight_layout()
fig.savefig(roi_outputs_folder_fcdo / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)

# Global Fund
name_of_plot = f'Incremental scenario cost relative to baseline {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (incremental_scenario_cost_summarized_gf / 1e6),
    annotations=[
        f"{round(row['mean'] / 1e6, 1)} \n ({round(row['lower'] / 1e6, 1)}- \n {round(row['upper'] / 1e6, 1)})"
        for _, row in incremental_scenario_cost_summarized_gf.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
)
ax.set_title(name_of_plot)
ax.set_ylabel('Cost \n(USD Millions)')
fig.tight_layout()
fig.savefig(roi_outputs_folder_gf / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)

# 4. Plot costs
# ----------------------------------------------------
# FCDO
input_costs_for_plot = input_costs_subset[input_costs_subset.draw.isin(hss_scenarios_for_report)]
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

do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'all', _disaggregate_by_subgroup = False, _outputfilepath = Path(figurespath / 'fcdo'), _scenario_dict = hs_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'all', _year = [2025],  _disaggregate_by_subgroup = False, _outputfilepath = Path(figurespath / 'fcdo'), _scenario_dict = hs_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'human resources for health',  _disaggregate_by_subgroup = False, _outputfilepath = Path(figurespath / 'fcdo'), _scenario_dict = hs_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'medical consumables',  _disaggregate_by_subgroup = False, _outputfilepath = Path(figurespath / 'fcdo'), _scenario_dict = hs_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'medical equipment',  _disaggregate_by_subgroup = False, _outputfilepath = Path(figurespath / 'fcdo'), _scenario_dict = hs_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'other',  _disaggregate_by_subgroup = False, _outputfilepath = Path(figurespath / 'fcdo'), _scenario_dict = hs_scenarios_substitutedict)

# Global Fund
input_costs_for_plot = input_costs_subset[input_costs_subset.draw.isin(hss_scenarios_for_report)]
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

do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'all', _disaggregate_by_subgroup = False, _outputfilepath = Path(figurespath / 'gf'), _scenario_dict = hs_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'all', _year = [2025],  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = hs_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'human resources for health',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = hs_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'medical consumables',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = hs_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'medical equipment',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = hs_scenarios_substitutedict)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'other',  _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = hs_scenarios_substitutedict)
