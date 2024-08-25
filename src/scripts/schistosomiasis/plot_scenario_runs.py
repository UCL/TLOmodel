""" use the outputs from scenario_runs.py and produce plots
and summary statistics for paper
"""

from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import pandas as pd
# import lacroix
import matplotlib.colors as colors
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from collections import defaultdict
import textwrap

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
    unflatten_flattened_multi_index_in_logging,
)


resourcefilepath = Path("./resources")
# outputpath = Path("./outputs")

output_folder = Path("./outputs/t.mangal@imperial.ac.uk")

# results_folder = get_scenario_outputs("schisto_calibration.py", outputpath)[-1]
results_folder = get_scenario_outputs("schisto_scenarios.py", output_folder)[-1]

# Declare path for output graphs from this script
def make_graph_file_name(name):
    return output_folder / f"Schisto_{name}.png"


# Name of species that being considered:
species = ('mansoni', 'haematobium')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# %% FUNCTIONS ##################################################################
TARGET_PERIOD = (Date(2025, 1, 1), Date(2035, 12, 31))
param_names = []  # todo can use params to label scenarios??


def target_period() -> str:
    """Returns the target period as a string of the form YYYY-YYYY"""
    return "-".join(str(t.year) for t in TARGET_PERIOD)


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


def set_param_names_as_column_index_level_0(_df):
    """Set the columns index (level 0) as the param_names."""
    ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


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


def do_bar_plot_with_ci(_df, annotations=None, xticklabels_horizontal_and_wrapped=False, put_labels_in_legend=True):
    """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
     extent of the error bar."""

    substitute_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    yerr = np.array([
        (_df['mean'] - _df['lower']).values,
        (_df['upper'] - _df['mean']).values,
    ])

    xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

    # Define colormap (used only with option `put_labels_in_legend=True`)
    cmap = plt.get_cmap("tab20")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
    colors = list(map(cmap, rescale(np.array(list(xticks.keys()))))) if put_labels_in_legend else None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        xticks.keys(),
        _df['mean'].values,
        yerr=yerr,
        alpha=0.8,
        ecolor='black',
        color=colors,
        capsize=10,
        label=xticks.values()
    )
    if annotations:
        for xpos, ypos, text in zip(xticks.keys(), _df['upper'].values, annotations):
            ax.text(xpos, ypos*1.15, text, horizontalalignment='center', rotation='vertical', fontsize='x-small')
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
    fig.tight_layout()

    return fig, ax



# %% OUTPUTS ##################################################################

# todo Total DALYs 2025-2035 for each scenario
# this will only reflect high-intensity infections
num_dalys = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_num_dalys,
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0)

num_dalys_summarized = summarize(num_dalys).loc[0].unstack().reindex(param_names)

name_of_plot = f'All Scenarios: DALYs, {target_period()}'
fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6)
ax.set_title(name_of_plot)
ax.set_ylabel('(Millions)')
ax.axhline(num_dalys_summarized.loc['Baseline', 'mean'] / 1e6, color='black', alpha=0.5)
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)




# todo person-years infected with low/moderate/high intensity infections
# stacked bar plot for each scenario
# separate for mansoni and haematobium


# todo table, rows=districts,
# column level0 with and without WASH:
# classify districts into low/moderate/high burden
# columns=[HML burden, person-years of low/moderate/high infection, # PZQ tablets]


# todo hiv incidence by draw


# todo bladder cancer incidence by draw


# todo elimination
# years to reach elimination as PH problem
# -- Elimination as a PH problem is defined as reducing the prevalence of heavy infections to less than 1% of population
# years to reach elimination of transmission
# years to reach morbidity control (heavy infections below threshold)
# -- morbidity control is reducing the prevalence of heavy infections to below 5% of the population
# -- (e.g. ≥400 EPG for mansoni or ≥50 eggs/10 mL urine for haematobium).

















