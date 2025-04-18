""" This script will do the bootstrap analysis for the CEA metrics with optimized performance (without joblib) """

import random
from pathlib import Path
import os
from typing import List, Dict, Any
import datetime
from math import e
from openpyxl import Workbook
from openpyxl import load_workbook
import scipy.stats as stats
import pickle

import numpy.random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from tlo.lm import LinearModel, LinearModelType, Predictor

# Store all the scenarios results
scenario_dfs = {}  # Initialize empty dictionary of dataframes
summary_statistics = {} # Initialize empty dictionary of dataframes


scenarios = ['baseline_ant', 'baseline_ant_with_po_level2',
             'baseline_ant_with_po_level1b',
             'baseline_ant_with_po_level1a', 'baseline_ant_with_po_level0',
             'existing_psa', 'existing_psa_with_po_level2', 'existing_psa_with_po_level1b',
             'existing_psa_with_po_level1a', 'existing_psa_with_po_level0',
             'planned_psa', 'planned_psa_with_po_level2', 'planned_psa_with_po_level1b',
             'planned_psa_with_po_level1a', 'planned_psa_with_po_level0'
             ]

dx_accuracy = 'imperfect'
sa_name = ''
# sa_name = 'reduced_hw_dx'
# sa_name = 'perfect_hw_dx'
# sa_name = 'reduce_referral'
# sa_name = 'reduce_ox_effect'
# sa_name = 'reduce_incidence'
# sa_name = 'reduce_mortality'
# sa_name = 'planned_psa_70'
# sa_name = 'remove_death_adjustment'
# sa_name = 'oxygen_cost_50'
# sa_name = 'po_cost_double'
# sa_name = 'out_inpatient_cost_double'



# Date for saving the image for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Open all scenario outputs
for scenario in scenarios:
    with open(f'debug_output_{scenario}_{dx_accuracy}_{sa_name}.pkl', 'rb') as f:
        scenario_output = pickle.load(f)
        scenario_dfs[scenario] = pd.DataFrame(scenario_output)


def get_scenario_differences(scenario_dfs, scenario, wtp=80):
    """Calculate point estimate differences and economic metrics for all scenarios relative to baseline."""
    # deaths averted and incremental costs
    deaths_averted = (scenario_dfs['baseline_ant']['mortality_outcome'].astype(int) -
                      scenario_dfs[scenario]['mortality_outcome'].astype(int)).sum()
    dalys_averted = (scenario_dfs['baseline_ant']['DALYs_discounted'] -
                     scenario_dfs[scenario]['DALYs_discounted']).sum()

    incremental_cost = (scenario_dfs[scenario]['total_costs'] - scenario_dfs['baseline_ant']['total_costs']).sum()
    mortality_reduction = deaths_averted / scenario_dfs['baseline_ant']['mortality_outcome'].astype(int).sum()

    # economic metrics
    icer_deaths = incremental_cost / deaths_averted
    icer_dalys = incremental_cost / dalys_averted
    inhb = dalys_averted - (incremental_cost / wtp)
    inmb = (dalys_averted * wtp) - incremental_cost

    return {
        'deaths_averted': deaths_averted,
        'dalys_averted': dalys_averted,
        'incremental_cost': incremental_cost,
        'mortality_reduction': mortality_reduction,
        'icer_deaths': icer_deaths,
        'icer_dalys': icer_dalys,
        'inhb': inhb,
        'inmb': inmb
    }



def prepare_bootstrap_data(scenario_dfs, scenario1, scenario2, dx_accuracy):
    """ Prepare data arrays for bootstrap analysis - convert pandas DataFrame into numpy array
    for faster computational run """
    df1 = scenario_dfs[scenario1]
    df2 = scenario_dfs[scenario2]

    # Convert key columns to numpy arrays
    mortality1 = df1['mortality_outcome'].astype(int).values
    mortality2 = df2['mortality_outcome'].astype(int).values
    dalys1 = df1['DALYs_discounted'].values
    dalys2 = df2['DALYs_discounted'].values
    costs1 = df1['total_costs'].values
    costs2 = df2['total_costs'].values

    # Prepare oxygen data
    oxygen_need_array = np.array([1 if x == '<90%' else 0 for x in df2['oxygen_saturation']])

    # Create facility and oxygen provided arrays
    # facility_col = f'final_facility_scenario_{scenario2}_{dx_accuracy}_hw_dx'
    # oxygen_provided_col = f'oxygen_provided_scenario_{scenario2}_{dx_accuracy}_hw_dx'
    # facility_fup_col = f'final_facility_follow_up_scenario_{scenario2}_{dx_accuracy}_hw_dx'
    # oxygen_provided_fup_col = f'oxygen_provided_follow_up_scenario_{scenario2}_{dx_accuracy}_hw_dx'

    # Create facility and oxygen provided arrays:
    facility_array = np.array([(x == '2' or x == '1b') for x in
                              df2[f'final_facility_scenario_{scenario2}_{dx_accuracy}_hw_dx']])
    oxygen_array = np.array([False if x is None else bool(x) for x in
                            df2[f'oxygen_provided_scenario_{scenario2}_{dx_accuracy}_hw_dx']])

    facility_fup_array = np.array([(x == '2' or x == '1b') for x in
                                  df2[f'final_facility_follow_up_scenario_{scenario2}_{dx_accuracy}_hw_dx']])
    oxygen_fup_array = np.array([False if x is None else bool(x) for x in
                                df2[f'oxygen_provided_follow_up_scenario_{scenario2}_{dx_accuracy}_hw_dx']])

    return {
        'mortality1': mortality1,
        'mortality2': mortality2,
        'dalys1': dalys1,
        'dalys2': dalys2,
        'costs1': costs1,
        'costs2': costs2,
        'oxygen_need_array': oxygen_need_array,
        'facility_array': facility_array,
        'oxygen_array': oxygen_array,
        'facility_fup_array': facility_fup_array,
        'oxygen_fup_array': oxygen_fup_array,
        'n': len(df1)
    }

def calculate_economic_uncertainty(scenario_dfs, scenario1, scenario2, wtp=80, n_bootstrap=1000):
    """Optimized function to calculate economic uncertainty using vectorized NumPy operations."""
    # Get point estimates
    point_differences = get_scenario_differences(scenario_dfs, scenario=scenario2, wtp=wtp)

    # Prepare data for bootstrap
    bootstrap_data = prepare_bootstrap_data(scenario_dfs, scenario1, scenario2, dx_accuracy)
    n = bootstrap_data['n']

    # Pre-allocate arrays for results
    bootstrap_deaths_averted = np.empty(n_bootstrap)
    bootstrap_dalys_averted = np.empty(n_bootstrap)
    bootstrap_cost_diff = np.empty(n_bootstrap)
    bootstrap_mortality_reduction = np.empty(n_bootstrap)
    bootstrap_accessed_oxygen = np.empty(n_bootstrap)
    bootstrap_icer_deaths = np.empty(n_bootstrap)
    bootstrap_icer_dalys = np.empty(n_bootstrap)
    bootstrap_inhbs = np.empty(n_bootstrap)
    bootstrap_inmbs = np.empty(n_bootstrap)

    # Use a single random number generator
    rng = np.random.RandomState(1)  # Fixed seed for reproducibility

    # Run bootstrap iterations sequentially
    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)

        # Extract resampled values for this iteration
        mortality_scenario1_resampled = bootstrap_data['mortality1'][indices]
        mortality_scenario2_resampled = bootstrap_data['mortality2'][indices]
        dalys_scenario1_resampled = bootstrap_data['dalys1'][indices]
        dalys_scenario2_resampled = bootstrap_data['dalys2'][indices]
        costs_scenario1_resampled = bootstrap_data['costs1'][indices]
        costs_scenario2_resampled = bootstrap_data['costs2'][indices]

        # Calculate oxygen metrics
        need_oxygen = np.sum(bootstrap_data['oxygen_need_array'][indices])

        # Use boolean arrays for faster calculation
        facility_indices = bootstrap_data['facility_array'][indices]
        oxygen_indices = bootstrap_data['oxygen_array'][indices]
        oxygen_provided = np.sum(facility_indices & oxygen_indices)

        facility_fup_indices = bootstrap_data['facility_fup_array'][indices]
        oxygen_fup_indices = bootstrap_data['oxygen_fup_array'][indices]
        fup_oxygen_provided = np.sum(facility_fup_indices & oxygen_fup_indices)

        # Avoid division by zero
        if need_oxygen > 0:
            access_to_oxygen = (oxygen_provided + fup_oxygen_provided) / need_oxygen
        else:
            access_to_oxygen = 0

        bootstrap_accessed_oxygen[i] = access_to_oxygen

        # Calculate differences
        mort_diff = mortality_scenario1_resampled - mortality_scenario2_resampled
        dalys_diff = dalys_scenario1_resampled - dalys_scenario2_resampled
        cost_diff = costs_scenario2_resampled - costs_scenario1_resampled

        # Calculate population-level metrics
        total_mort_diff = np.sum(mort_diff)
        bootstrap_deaths_averted[i] = total_mort_diff

        total_dalys_diff = np.sum(dalys_diff)
        bootstrap_dalys_averted[i] = total_dalys_diff

        total_cost_diff = np.sum(cost_diff)
        bootstrap_cost_diff[i] = total_cost_diff

        # Calculate mortality reduction (avoid division by zero)
        mort1_mean = np.mean(mortality_scenario1_resampled)
        if mort1_mean > 0:
            mortality_reduction = np.mean(mort_diff) / mort1_mean
        else:
            mortality_reduction = 0

        bootstrap_mortality_reduction[i] = mortality_reduction

        # Calculate economic metrics
        if total_mort_diff != 0:
            icer_deaths = total_cost_diff / total_mort_diff
        else:
            icer_deaths = np.nan

        bootstrap_icer_deaths[i] = icer_deaths

        if total_dalys_diff != 0:
            icer_dalys = total_cost_diff / total_dalys_diff
        else:
            icer_dalys = np.nan

        bootstrap_icer_dalys[i] = icer_dalys

        inhb = total_dalys_diff - (total_cost_diff / wtp)
        bootstrap_inhbs[i] = inhb

        inmb = (total_dalys_diff * wtp) - total_cost_diff
        bootstrap_inmbs[i] = inmb

    # Calculate statistics on bootstrap results
    skewness = stats.skew(bootstrap_icer_deaths, nan_policy='omit')
    kurtosis = stats.kurtosis(bootstrap_icer_deaths, nan_policy='omit')

    return {
        'deaths_averted': point_differences['deaths_averted'],
        'dalys_averted': point_differences['dalys_averted'],
        'incremental_cost': point_differences['incremental_cost'],
        'icer_deaths': point_differences['icer_deaths'],
        'icer_dalys': point_differences['icer_dalys'],
        'inhb': point_differences['inhb'],
        'inmb': point_differences['inmb'],
        'death_averted_mean': np.mean(bootstrap_deaths_averted),
        'death_averted_ci': np.percentile(bootstrap_deaths_averted, [2.5, 97.5]),
        'dalys_averted_mean': np.mean(bootstrap_dalys_averted),
        'dalys_averted_ci': np.percentile(bootstrap_dalys_averted, [2.5, 97.5]),
        'cost_diff_mean': np.mean(bootstrap_cost_diff),
        'cost_diff_ci': np.percentile(bootstrap_cost_diff, [2.5, 97.5]),
        'mortality_reduction_mean': np.mean(bootstrap_mortality_reduction),
        'mortality_reduction_ci': np.percentile(bootstrap_mortality_reduction, [2.5, 97.5]),
        'access_to_oxygen_mean': np.mean(bootstrap_accessed_oxygen),
        'access_to_oxygen_ci': np.percentile(bootstrap_accessed_oxygen, [2.5, 97.5]),
        'icer_deaths_mean': np.mean(bootstrap_icer_deaths),
        'bootstrap_normality': (skewness, kurtosis),
        'icer_deaths_ci': np.percentile(bootstrap_icer_deaths, [2.5, 97.5]),
        'icer_dalys_mean': np.mean(bootstrap_icer_dalys),
        'icer_dalys_ci': np.percentile(bootstrap_icer_dalys, [2.5, 97.5]),
        'inhb_mean': np.mean(bootstrap_inhbs),
        'inhb_ci': np.percentile(bootstrap_inhbs, [2.5, 97.5]),
        'inmb_mean': np.mean(bootstrap_inmbs),
        'inmb_ci': np.percentile(bootstrap_inmbs, [2.5, 97.5]),
    }


# Run the analysis
if __name__ == "__main__":
    for scenario_compare in scenarios[1:]:
        print(f"Processing scenario: {scenario_compare}")
        start_time = datetime.datetime.now()

        summary_statistics[scenario_compare] = \
            calculate_economic_uncertainty(
                scenario_dfs=scenario_dfs,
                scenario1='baseline_ant',
                scenario2=scenario_compare,
                wtp=80,
                n_bootstrap=1000
            )

        end_time = datetime.datetime.now()
        print(f"Completed in {end_time - start_time}")

# Save results
with open(f'debug_output_bootstrap_scenario_statistics_{dx_accuracy}{sa_name}.pkl', 'wb') as f:
    pickle.dump(summary_statistics, f)

print("Bootstrap complete and saved.")


# Open the summary statistics
with open(f'debug_output_bootstrap_scenario_statistics_{dx_accuracy}{sa_name}.pkl', 'rb') as f:
    bootstrap_results = pickle.load(f)


scenario_statistics = bootstrap_results


def get_frontier_points(scenario_statistics):
    """Get points for cost-effectiveness frontier."""
    # Extract mean costs and effects for each scenario
    # Add origin point (baseline)
    points = [{'scenario': 'baseline_ant', 'cost': 0, 'effect': 0}]

    # Extract mean costs and effects for each scenario
    for scenario in scenario_statistics:
        points.append({
            'scenario': scenario,
            'cost': scenario_statistics[scenario]['cost_diff_mean'],
            'effect': scenario_statistics[scenario]['dalys_averted_mean']
        })

    # Sort by effects
    points = sorted(points, key=lambda x: x['effect'])

    # Find frontier points (non-dominated strategies)
    # Start with baseline point (0,0)
    frontier = [points[0]]

    for i in range(1, len(points)):
        current_point = points[i]

        while len(frontier) >= 2:
            # Calculate slopes between last three points
            previous = frontier[-1]
            second_previous = frontier[-2]

            # Calculate ICERs
            slope_last = (current_point['cost'] - previous['cost']) / (current_point['effect'] - previous['effect'])
            slope_second_last = (previous['cost'] - second_previous['cost']) / (previous['effect'] - second_previous['effect'])

            # If current point creates a more efficient frontier, remove the last point
            if slope_last < slope_second_last:
                frontier.pop()
            else:
                break

        # Add point to frontier
        frontier.append(current_point)

    # Filter out points that are dominated (higher cost, same or lower effect)
    final_frontier = frontier
    # for i in range(2, len(frontier)):
    #     if (frontier[i]['cost'] / frontier[i]['effect']) < (frontier[i-1]['cost'] / frontier[i-1]['effect']):
    #         final_frontier.append(frontier[i])

    return final_frontier


def plot_ce_plane(scenario_statistics):
    """
    Create cost-effectiveness plane plot with confidence intervals.

    Parameters:
    - all_differences: dictionary of differences from get_scenario_differences()
    - wtp: willingness to pay threshold (for reference line)
    """

    base_scenarios = ['baseline_ant', 'existing_psa', 'planned_psa']
    po_levels = ['level2', 'level1b', 'level1a', 'level0']

    # Get unique scenarios
    scenarios = list(scenario_statistics.keys())

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Find max y value (including CI upper bounds)
    max_cost = max(scenario_statistics[s]['cost_diff_ci'][1] for s in scenarios)
    # Find max and min x values for DALYs
    max_dalys = max(scenario_statistics[s]['dalys_averted_ci'][1] for s in scenarios)
    min_dalys = min(scenario_statistics[s]['dalys_averted_ci'][0] for s in scenarios)

    # Plot reference lines
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Colors for different scenarios

    # Define base colors for each scenario
    BASE_COLOURS = {
        'baseline_ant': {
            'base': '#004080', # Navy Blue
            'level2': '#1a5da0', # Cobalt Blue
            'level1b': '#3379c0',  # Royal Blue
            'level1a': '#4d96df',  # Steel Blue
            'level0': '#66b2ff'  # Cornflower Blue
        },
        'existing_psa': {
            'base': '#e65100', # Dark Orange
            'level2': '#ec670a', # Medium-Dark Orange
            'level1b': '#f37c13',  # Medium Orange
            'level1a': '#f9921d',  # Medium-Light Orange
            'level0': '#ffa726'  # Light Orange
        },
        'planned_psa': {
            'base': '#1b5e20', # Dark Green
            'level2': '#3e7c42', # Medium-Dark Green
            'level1b': '#609a64',  # Medium Green
            'level1a': '#83b885',  # Medium-Light Green
            'level0': '#a5d6a7'  # Light Green
        }
    }
    colours = BASE_COLOURS

    # Define markers by PO implementation level
    po_level_markers = {
        'base': 'o',  # circle
        'level2': 's',   # square
        'level1b': 'd',  # diamond
        'level1a': '*',  # star
        'level0': '^'   # triangle
    }

    # Add point for baseline_ant
    plt.scatter(0, 0, color=colours['baseline_ant']['base'], marker=po_level_markers['base'],
                label='baseline_ant')
    # Plot each scenario
    for scenario in scenarios:
        # Get pre-calculated statistics
        cost_stats = {'mean': scenario_statistics[scenario]['cost_diff_mean'],
                      'ci_lower': scenario_statistics[scenario]['cost_diff_ci'][0],
                      'ci_upper': scenario_statistics[scenario]['cost_diff_ci'][1]}
        daly_stats = {'mean': scenario_statistics[scenario]['dalys_averted_mean'],
                      'ci_lower': scenario_statistics[scenario]['dalys_averted_ci'][0],
                      'ci_upper': scenario_statistics[scenario]['dalys_averted_ci'][1]}

        # Determine base scenario and PO level
        base_scenario = scenario.split('_with_')[0] if '_with_' in scenario else scenario

        if '_with_' in scenario:
            # Determine PO level
            po_level = next((po for po in po_levels if po in scenario), 'base')
            colour = colours[base_scenario][po_level]
            marker = po_level_markers[po_level]
        else:
            # Base scenario without PO level
            colour = colours[base_scenario]['base']
            marker = po_level_markers['base']

        # Plot point (mean values)
        plt.scatter(daly_stats['mean'], cost_stats['mean'],
                    color=colour, marker=marker,
                    label=scenario)

        # For error bars, use same color with slight transparency
        if isinstance(colour, str):  # If hex color
            error_color = mcolors.to_rgba(colour, alpha=0.3)
        else:  # If already rgba
            error_color = (*colour[:3], 0.3)

        # Plot confidence intervals using pre-calculated CIs
        plt.errorbar(daly_stats['mean'], cost_stats['mean'],
                     yerr=[[cost_stats['mean'] - cost_stats['ci_lower']],
                           [cost_stats['ci_upper'] - cost_stats['mean']]],
                     xerr=[[daly_stats['mean'] - daly_stats['ci_lower']],
                           [daly_stats['ci_upper'] - daly_stats['mean']]],
                     color=error_color, capsize=5)

    # Get and plot frontier
    frontier_points = get_frontier_points(scenario_statistics)
    frontier_x = [p['effect'] for p in frontier_points]
    frontier_y = [p['cost'] for p in frontier_points]
    plt.plot(frontier_x, frontier_y, 'k--', color='grey', alpha=0.2, label='CE frontier')

    # Get ICER value and format it
    n_points = len(frontier_points)
    for i in range(1, n_points):
        frontier_scenario = frontier_points[i]['scenario']
        icer_value = scenario_statistics[frontier_scenario]['icer_dalys_mean']
        icer_value_rounded = (np.round(icer_value)).astype(int)
        icer_text = f"ICER={icer_value_rounded}$/DALY"
        # Add ICER annotation
        plt.annotate(
            icer_text,
            xy=(scenario_statistics[frontier_scenario]['dalys_averted_mean'],
                scenario_statistics[frontier_scenario]['cost_diff_mean']-85000))

    # Set y-axis limits to match max cost
    plt.ylim(-max_cost * 0.1, max_cost * 1.1)  # Add 10% padding

    # Format y-axis to show values in 100,000s
    def y_fmt(y, pos):
        """Format y-axis to show values in full units, add commas"""
        return '{:,.0f}'.format(y)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_fmt))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(y_fmt))

    # Customize plot
    plt.xlabel('DALYs Averted', fontsize=14)
    plt.ylabel('Incremental Cost ($)', fontsize=14)
    plt.title('Cost-Effectiveness Plane')

    # Get the current handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Create custom labels dictionary (map original labels to new ones)
    custom_labels = {
        'baseline_ant': 'No_Ox_&_No_PO',
        'baseline_ant_with_po_level2': 'No_Ox_&_PO_1level',
        'baseline_ant_with_po_level1b': 'No_Ox_&_PO_2levels',
        'baseline_ant_with_po_level1a': 'No_Ox_&_PO_3levels',
        'baseline_ant_with_po_level0': 'No_Ox_&_PO_4levels',
        'existing_psa': 'Low_Ox_&_No_PO',
        'existing_psa_with_po_level2': 'Low_Ox_&_PO_1level',
        'existing_psa_with_po_level1b': 'Low_Ox_&_PO_2levels',
        'existing_psa_with_po_level1a': 'Low_Ox_&_PO_3levels',
        'existing_psa_with_po_level0': 'Low_Ox_&_PO_4levels',
        'planned_psa': 'High_Ox_&_No_PO',
        'planned_psa_with_po_level2': 'High_Ox_&_PO_1level',
        'planned_psa_with_po_level1b': 'High_Ox_&_PO_2levels',
        'planned_psa_with_po_level1a': 'High_Ox_&_PO_3levels',
        'planned_psa_with_po_level0': 'High_Ox_&_PO_4levels',
    }

    # Replace labels with custom ones where available
    new_labels = [custom_labels.get(label, label) for label in labels]
    frontier_handle = [handles[0]]
    frontier_label = [labels[0]]
    scenarios_handle = handles[1:]
    scenarios_label = new_labels[1:]

    # Assuming scenarios_handle and scenarios_label are your original lists
    n_items = len(scenarios_label)
    n_cols = 5
    n_rows = (n_items + n_cols - 1) // n_cols
    # This is the key part: create the correct indexing order
    indices = []
    for col in range(n_cols):
        for row in range(n_rows):
            idx = row * n_cols + col  # This is the original index in row-major order
            if idx < n_items:
                indices.append(idx)

    # Reorder handles and labels using these indices
    reordered_handles = [scenarios_handle[i] for i in indices]
    reordered_labels = [scenarios_label[i] for i in indices]

    # Position legend at the bottom of the plot
    legend_frontier = plt.gca().legend(
        frontier_handle, frontier_label, loc='upper center', bbox_to_anchor=(0.1, -0.1), ncol=1,
        frameon=False)
    plt.gca().add_artist(legend_frontier)
    # plt.legend(scenarios_handle, scenarios_label, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5,
    #            frameon=False)
    plt.legend(reordered_handles, reordered_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5,
               frameon=False)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    return plt.gcf()

# Apply
fig = plot_ce_plane(bootstrap_results)
plt.show()

# Optionally save the figure
fig.savefig(f'ce_plane{sa_name}.png', dpi=300, bbox_inches='tight')

# # # Save a dataframe with the key information for the paper
key_table = {}
cleaned_values = {}
for scenario in bootstrap_results.keys():
    # Initialize the dictionary for this scenario
    cleaned_values[scenario] = {}
    for key, values in bootstrap_results[scenario].items():
        if key.startswith(('dalys_averted', 'inhb')):
            values_abs = np.abs(values.astype(int))  # absolute value
            rounded_values = (np.round(values / 100) * 100).astype(int) if \
                (values_abs > 1000).all() else np.round(values).astype(int)
            cleaned_values[scenario][key] = rounded_values
        elif key.startswith(('icer', 'death')):
            if isinstance(values, np.ndarray):
                values_clean = np.nan_to_num(values, nan=0)
                rounded_values = np.round(values_clean).astype(int)
                # rounded_values = (np.round(values)).astype(int)
            else:
                values_clean = np.nan_to_num(values, nan=0)
                rounded_values = int(round(values_clean)) if np.isfinite(values_clean) else 0
                # rounded_values = values.astype(int)
            cleaned_values[scenario][key] = rounded_values
        elif key.startswith(('cost', 'inmb')) or 'cost' in key:
            values_abs = np.abs(values.astype(int))
            rounded_values = (np.round(values / 1000) * 1000).astype(int) if \
                (values_abs > 10000).all() else np.round(values).astype(int)
            cleaned_values[scenario][key] = rounded_values
        elif key.startswith(('mortality', 'access')):
            if isinstance(values, np.ndarray):
                rounded_values = ['{:.1f}'.format(v*100) for v in values]
            else:
                rounded_values = '{:.1f}'.format(values*100)
            cleaned_values[scenario][key] = rounded_values
        else:
            cleaned_values[scenario][key] = values

# Create the table with key outputs with their 95% CI
key_table = pd.DataFrame(columns=['scenario', 'access_to_oxygen',
                                  'cost_diff',
                                  'mortality_reduction', 'death_averted',
                                  'icer_deaths', 'dalys_averted',
                                  'icer_dalys', 'inhb', 'inmb'])


def format_values_for_table(data_dict, key):
    mean = data_dict.get(f'{key}_mean', None)
    ci = data_dict.get(f'{key}_ci', None)

    if mean is not None and ci is not None and len(ci) == 2:
        return f"{mean} ({ci[0]}, {ci[1]})"
    return None

# Convert the dictionary to rows in the DataFrame
rows = []
for scenario, dict in cleaned_values.items():
    row = {
        'scenario': scenario,
        'access_to_oxygen': format_values_for_table(dict, key='access_to_oxygen'),
        'cost_diff': format_values_for_table(dict, key='cost_diff'),
        'mortality_reduction': format_values_for_table(dict, key='mortality_reduction'),
        'deaths_averted': format_values_for_table(dict, key='death_averted'),
        'icer_deaths': format_values_for_table(dict, key='icer_deaths'),
        'dalys_averted': format_values_for_table(dict, key='dalys_averted'),
        'icer_dalys': format_values_for_table(dict, key='icer_dalys'),
        'inhb': format_values_for_table(dict, key='inhb'),
        'inmb': format_values_for_table(dict, key='inmb'),
    }
    rows.append(row)

key_table = pd.DataFrame(rows)


