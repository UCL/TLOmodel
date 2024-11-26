
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
figurespath = Path('./outputs/costing/overview/')
if not os.path.exists(figurespath):
    os.makedirs(figurespath)

# Load result files
# ------------------------------------------------------------------------------------------------------------------
results_folder = get_scenario_outputs('hss_elements-2024-11-12T172311Z.py', outputfilepath)[0]
#results_folder = Path('./outputs/cost_scenarios-2024-11-26T164353Z')

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0)  # look at one log (so can decide what to extract)
params = extract_params(results_folder)

# Declare default parameters for cost analysis
# ------------------------------------------------------------------------------------------------------------------
# Period relevant for costing
TARGET_PERIOD_INTERVENTION = (Date(2020, 1, 1), Date(2030, 12, 31))  # This is the period that is costed
relevant_period_for_costing = [i.year for i in TARGET_PERIOD_INTERVENTION]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))

# Scenarios
cost_scenarios = {0: "Real world", 1: "Perfect health system"}

# Costing parameters
discount_rate = 0.03

# Estimate standard input costs of scenario
# -----------------------------------------------------------------------------------------------------------------------
input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath, _draws = [0],
                                               _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
                                               _discount_rate = discount_rate, summarize = True)
# _draws = htm_scenarios_for_gf_report --> this subset is created after calculating malaria scale up costs
# TODO Remove the manual fix below once the logging for these is corrected
input_costs.loc[input_costs.cost_subgroup == 'Oxygen, 1000 liters, primarily with oxygen cylinders', 'cost'] = \
    input_costs.loc[input_costs.cost_subgroup == 'Oxygen, 1000 liters, primarily with oxygen cylinders', 'cost']/10

input_costs_undiscounted = estimate_input_cost_of_scenarios(results_folder, resourcefilepath, _draws = [0,1],
                                               _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
                                               _discount_rate = 0, summarize = True)
# _draws = htm_scenarios_for_gf_report --> this subset is created after calculating malaria scale up costs
# TODO Remove the manual fix below once the logging for these is corrected
input_costs_undiscounted.loc[input_costs_undiscounted.cost_subgroup == 'Oxygen, 1000 liters, primarily with oxygen cylinders', 'cost'] = \
    input_costs_undiscounted.loc[input_costs_undiscounted.cost_subgroup == 'Oxygen, 1000 liters, primarily with oxygen cylinders', 'cost']/10

# Get figures for overview paper
# -----------------------------------------------------------------------------------------------------------------------
# Figure 1: Estimated costs by cost category
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'all', _disaggregate_by_subgroup = False, _outputfilepath = figurespath, _scenario_dict = cost_scenarios)

# Figure 2: Estimated costs by year
do_line_plot_of_cost(_df = input_costs_undiscounted, _cost_category='all',
                         _year='all', _draws= [0],
                         disaggregate_by= 'cost_category',
                         _outputfilepath = figurespath)
do_line_plot_of_cost(_df = input_costs_undiscounted, _cost_category='all',
                         _year='all', _draws= [1],
                         disaggregate_by= 'cost_category',
                         _outputfilepath = figurespath)

# Figure 3: Comparison of model-based cost estimates with actual expenditure recorded for 2018/19 and budget planned for 2020/21-2022/23

# Figure 4: Total cost by scenario assuming 0% discount rate
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_undiscounted,
                                        _cost_category = 'all',
                                        _disaggregate_by_subgroup = False,
                                        _outputfilepath = figurespath,
                                        _scenario_dict = cost_scenarios,
                                        _add_figname_suffix = '_UNDISCOUNTED')

# Figure 5: Total cost by scenario applying changing discount rates

# Get tables for overview paper
# -----------------------------------------------------------------------------------------------------------------------
# Group data and aggregate cost for each draw and stat
def generate_detail_cost_table(_groupby_var, _groupby_var_name, _longtable = False):
    edited_input_costs = input_costs.copy()
    edited_input_costs[_groupby_var] = edited_input_costs[_groupby_var].replace('_', ' ', regex=True)
    edited_input_costs[_groupby_var] = edited_input_costs[_groupby_var].replace('%', '\%', regex=True)
    edited_input_costs[_groupby_var] = edited_input_costs[_groupby_var].replace('&', '\&', regex=True)

    grouped_costs = edited_input_costs.groupby(['cost_category', _groupby_var, 'draw', 'stat'])['cost'].sum()
    # Format the 'cost' values before creating the LaTeX table
    grouped_costs = grouped_costs.apply(lambda x: f"{float(x):,.2f}")
    # Remove underscores from all column values

    # Create a pivot table to restructure the data for LaTeX output
    pivot_data = {}
    for draw in [0, 1]:
        draw_data = grouped_costs.xs(draw, level='draw').unstack(fill_value=0)  # Unstack to get 'stat' as columns
        # Concatenate 'mean' with 'lower-upper' in the required format
        pivot_data[draw] = draw_data['mean'].astype(str) + ' [' + \
                           draw_data['lower'].astype(str) + '-' + \
                           draw_data['upper'].astype(str) + ']'

    # Combine draw data into a single DataFrame
    table_data = pd.concat([pivot_data[0], pivot_data[1]], axis=1, keys=['draw=0', 'draw=1']).reset_index()

    # Rename columns for clarity
    table_data.columns = ['Cost Category', _groupby_var_name, 'Real World', 'Perfect Health System']

    # Replace '\n' with '\\' for LaTeX line breaks
    #table_data['Real World'] = table_data['Real World'].apply(lambda x: x.replace("\n", "\\\\"))
    #table_data['Perfect Health System'] = table_data['Perfect Health System'].apply(lambda x: x.replace("\n", "\\\\"))

    # Convert to LaTeX format with horizontal lines after every row
    latex_table = table_data.to_latex(
        longtable=_longtable,  # Use the longtable environment for large tables
        column_format='|R{4cm}|R{5cm}|R{3.5cm}|R{3.5cm}|',
        caption=f"Summarized Costs by Category and {_groupby_var_name}",
        label=f"tab:cost_by_{_groupby_var}",
        position="h",
        index=False,
        escape=False,  # Prevent escaping special characters like \n
        header=True
    )

    # Add \hline after the header and after every row for horizontal lines
    latex_table = latex_table.replace("\\\\", "\\\\ \\hline")  # Add \hline after each row
    #latex_table = latex_table.replace("_", " ")  # Add \hline after each row

    # Specify the file path to save
    latex_file_path = figurespath / f'cost_by_{_groupby_var}.tex'

    # Write to a file
    with open(latex_file_path, 'w') as latex_file:
        latex_file.write(latex_table)

    # Print latex for reference
    print(latex_table)

# Table : Cost by cost subcategory
generate_detail_cost_table(_groupby_var = 'cost_subcategory', _groupby_var_name = 'Cost Subcategory')
# Table : Cost by cost subgroup
generate_detail_cost_table(_groupby_var = 'cost_subgroup', _groupby_var_name = 'Category Subgroup', _longtable = True)

