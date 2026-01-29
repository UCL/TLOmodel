"""Produce outputs for Impact of Improved Consumables Availability Paper
"""
import datetime
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from scripts.costing.cost_estimation import (
    clean_consumable_name,
    create_summary_treemap_by_cost_subgroup,
    do_line_plot_of_cost,
    do_stacked_bar_plot_of_cost_by_category,
    estimate_input_cost_of_scenarios,
    summarize_cost_data
)
from tlo import Date
from tlo.analysis.utils import (
    compute_summary_statistics,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    create_pickles_locally
)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/')
figurespath = Path('./outputs/consumables_impact_analysis/manuscript')
if not os.path.exists(figurespath):
    os.makedirs(figurespath)
path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"

# Load result files
# ------------------------------------------------------------------------------------------------------------------
results_folder = get_scenario_outputs('consumables_costing-2026-01-29T121903Z.py', outputfilepath)[0] # Dec 2025 runs
#create_pickles_locally(scenario_output_dir = "./outputs/consumables_costing-2026-01-29T121903Z") # from .log files

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0)  # look at one log (so can decide what to extract)
params = extract_params(results_folder)
info = get_scenario_info(results_folder)

# Declare default parameters for cost analysis
# ------------------------------------------------------------------------------------------------------------------
# Period relevant for costing
TARGET_PERIOD = (Date(2010, 1, 1), Date(2012, 12, 31))  # This is the period that is costed
relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
list_of_years_for_plot = list(range(2010, 2012))
number_of_years_costed = relevant_period_for_costing[1] - 2010 + 1

discount_rate_health = 0
chosen_metric = 'mean'

# Scenarios
cons_scenarios = {
    0:  "Baseline availability – Default health system",
    1:  "Baseline availability – Perfect health system",

    2:  "Non-therapeutic consumables – Default health system",
    3:  "Non-therapeutic consumables – Perfect health system",

    4:  "Vital medicines – Default health system",
    5:  "Vital medicines – Perfect health system",

    6:  "Pharmacist-managed stocks – Default health system",
    7:  "Pharmacist-managed stocks – Perfect health system",

    8:  "75th percentile facility – Default health system",
    9:  "75th percentile facility – Perfect health system",

    10: "90th percentile facility – Default health system",
    11: "90th percentile facility – Perfect health system",

    12: "Best facility – Default health system",
    13: "Best facility – Perfect health system",

    14: "District pooling – Default health system",
    15: "District pooling – Perfect health system",

    16: "Neighbourhood pooling – Default health system",
    17: "Neighbourhood pooling – Perfect health system",

    18: "Large radius pairwise exchanges – Default health system",
    19: "Large radius pairwise exchanges – Perfect health system",

    20: "Small radius pairwise exchanges – Default health system",
    21: "Small radius pairwise exchanges – Perfect health system",

    22: "Perfect availability – Default health system",
    23: "Perfect availability – Perfect health system",
}

# Function to get incremental values
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

# Define a function to create bar plots
def do_standard_bar_plot_with_ci(_df: pd.DataFrame, set_colors=None, annotations=None,
                                 xticklabels_horizontal_and_wrapped=False,
                                 put_labels_in_legend=True,
                                 offset=1e6):
    """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
     extent of the error bar."""

    substitute_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    yerr = np.array([
        (_df[chosen_metric] - _df['lower']).values,
        (_df['upper'] - _df[chosen_metric]).values,
    ])

    xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

    if set_colors is not None:
        # dict mapping -> use index keys; list/tuple/Series -> use as-is
        if isinstance(set_colors, dict):
            colors = [set_colors.get(k, 'grey') for k in _df.index]
            # Optional debug:
            # missing = [k for k in _df.index if k not in set_colors]
            # if missing: print("No color for:", missing)
        else:
            colors = list(set_colors)
    else:
        cmap = sns.color_palette('Spectral', as_cmap=True)
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
        colors = list(map(cmap, rescale(np.array(list(xticks.keys()))))) if put_labels_in_legend else None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        xticks.keys(),
        _df[chosen_metric].values,
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
        # Use all_manuscript_scenarios for the legend
        xtick_legend = [f'{letter}: {cons_scenarios.get(label, label)}' for letter, label in
                        zip(substitute_labels, xticks.values())]
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
    extension = 0.1 * (ymax - ymin)  # 10% of range
    ax.set_ylim(ymin - extension, ymax + extension)  # Set new y-axis limits with the extended range

    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # fig.tight_layout()
    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.15, right=0.5, top=0.88)

    return fig, ax

#
def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = relevant_period_for_costing
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    _df = _df.loc[_df.year.between(*years_needed)].drop(columns=['date', 'sex', 'age_range']).groupby(
        'year').sum().sum(axis=1)

    # Initial year and discount rate
    initial_year = min(_df.index.unique())

    # Calculate the discounted values
    discounted_values = _df / (1 + discount_rate_health) ** (_df.index - initial_year)

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

# Plot DALYs
num_dalys_averted_summarized = summarize_cost_data(num_dalys_averted, _metric=chosen_metric)
num_dalys_averted_subset_for_figure = num_dalys_averted_summarized[
    num_dalys_averted_summarized.index.get_level_values('draw').isin(list(cons_scenarios.keys()))]
name_of_plot = f'Incremental DALYs averted compared to baseline {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (num_dalys_averted_subset_for_figure / 1e6),
    annotations=[
        f"{row[chosen_metric] / 1e6:.2f} ({row['lower'] / 1e6 :.2f}- {row['upper'] / 1e6:.2f})"
        for _, row in num_dalys_averted_subset_for_figure.iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
    put_labels_in_legend=True,
    offset=0.05,
)
#ax.set_title(name_of_plot)
ax.set_ylabel('DALYs \n(Millions)')
ax.set_ylim(bottom=0)
fig.savefig(figurespath / name_of_plot.replace(' ', '_').replace(',', ''))
plt.close(fig)
