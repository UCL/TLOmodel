""" use the outputs from scenario_runs.py and produce plots
and summary statistics for paper

JOB ID:
schisto_scenarios-2025-03-22T130153Z
"""

from pathlib import Path
import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
# import lacroix
import matplotlib.colors as colors
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from collections import defaultdict
import textwrap
from typing import Tuple

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    compare_number_of_deaths,
    extract_params,
    compute_summary_statistics,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    compute_summary_statistics,
    make_age_grp_lookup,
    make_age_grp_types,
    unflatten_flattened_multi_index_in_logging,
)


resourcefilepath = Path("./resources")

output_folder = Path("./outputs/t.mangal@imperial.ac.uk")

results_folder = get_scenario_outputs("schisto_scenarios.py", output_folder)[-1]


# Declare path for output graphs from this script
def make_graph_file_name(name):
    return results_folder / f"Schisto_{name}.png"


# Name of species that being considered:
species = ('mansoni', 'haematobium')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# %% FUNCTIONS ##################################################################
TARGET_PERIOD = (Date(2024, 1, 1), Date(2040, 12, 31))


def get_parameter_names_from_scenario_file() -> Tuple[str]:
    """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
    from scripts.schistosomiasis.scenario_runs import (
        SchistoScenarios,
    )
    e = SchistoScenarios()
    return tuple(e._scenarios.keys())


param_names = get_parameter_names_from_scenario_file()


def target_period() -> str:
    """Returns the target period as a string of the form YYYY-YYYY"""
    return "-".join(str(t.year) for t in TARGET_PERIOD)


def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


def set_param_names_as_column_index_level_0(_df):
    """Set the columns index (level 0) as the param_names."""
    ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


def find_difference_relative_to_comparison_series(
    _ser: pd.Series,
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


def find_difference_relative_to_comparison_dataframe(_df: pd.DataFrame, **kwargs):
    """Apply `find_difference_relative_to_comparison_series` to each row in a dataframe"""
    return pd.concat({
        _idx: find_difference_relative_to_comparison_series(row, **kwargs)
        for _idx, row in _df.iterrows()
    }, axis=1).T


########################################################################################
# %% EXTRACT DALYS
########################################################################################


def get_total_num_dalys(_df):
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


total_num_dalys = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_total_num_dalys,
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0)

summary_total_num_dalys = \
compute_summary_statistics(total_num_dalys, central_measure='mean').loc[0].unstack().reindex(
    param_names)
summary_total_num_dalys.to_csv(results_folder / f'summary_total_num_dalys_{target_period()}.csv')


# DALYS all-cause

def round_to_nearest_100(x):
    return 100 * round(x / 100)


def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'year']) \
        .sum()


# extract dalys by cause with mean and upper/lower intervals
# With 'collapse_columns', if number of draws is 1, then collapse columns multi-index:

num_dalys_by_cause = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)

summary_dalys_by_cause = compute_summary_statistics(num_dalys_by_cause,
    central_measure='mean',
)

summary_dalys_by_cause = round_to_nearest_100(summary_dalys_by_cause)
summary_dalys_by_cause = summary_dalys_by_cause.astype(int)
summary_dalys_by_cause.to_csv(results_folder / (f'summary_dalys_by_cause {target_period()}.csv'))


# todo li_wealth replaced by district_of_residence in DALY logger
def get_num_dalys_by_defined_label(_df):
    """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
    y = _df \
        .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
        .drop(columns=['date', 'year', 'district_of_residence']) \
        .sum(axis=0)

    # define course cause mapper for HIV, TB, MALARIA and OTHER
    causes = {
        'Schistosomiasis': 'Schisto',
        'AIDS': 'HIV/AIDS',
        'Cancer (Bladder)': 'Bladder cancer',
        'Childhood Diarrhoea': 'Diarrhoea',
        'Lower respiratory infections': 'ALRI',
        '': 'Other',  # defined in order to use this dict to determine ordering of the causes in output
    }
    causes_relabels = y.index.map(causes).fillna('Other')

    return y.groupby(by=causes_relabels).sum()[list(causes.values())]


num_dalys_by_defined_label = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_by_wealth_stacked_by_age_and_time",
    custom_generate_series=get_num_dalys_by_defined_label,
    do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)

summary_num_dalys_by_label = compute_summary_statistics(num_dalys_by_defined_label, central_measure='mean')
summary_num_dalys_by_label.to_csv(results_folder / f'summary_num_dalys_by_label{target_period()}.csv')


def num_dalys_by_district(_df):
    """Return total number of DALYs (summed across all disease columns) by district within TARGET_PERIOD, as a Series."""
    return (
        _df
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
        .drop(columns=['date', 'year'])
        .groupby('district_of_residence')
        .sum()
        .sum(axis=1)  # sum across disease columns
    )


summary_dalys_by_district_unscaled = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_by_wealth_stacked_by_age_and_time",
        custom_generate_series=num_dalys_by_district,
        do_scaling=False,
    ),
    central_measure='mean',
)

summary_dalys_by_district_unscaled = round_to_nearest_100(summary_dalys_by_district_unscaled)
summary_dalys_by_district_unscaled = summary_dalys_by_district_unscaled.astype(int)
summary_dalys_by_district_unscaled.to_csv(results_folder / (f'summary_dalys_by_district_unscaled {target_period()}.csv'))


########################################################################################
# %% EXTRACT DALYS AVERTED
########################################################################################

# total number of DALYs averted, summed across all causes and all districts

total_dalys_averted_vs_pauseWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='Pause WASH, no MDA'
    ),
    central_measure='mean'
)
total_dalys_averted_vs_pauseWASH.to_csv(results_folder / f'total_dalys_averted_vs_pauseWASH{target_period()}.csv')

total_dalys_averted_vs_ContinueWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='Continue WASH, no MDA'
    ),
    central_measure='mean'
)
total_dalys_averted_vs_ContinueWASH.to_csv(results_folder / f'total_dalys_averted_vs_ContinueWASH{target_period()}.csv')

total_dalys_averted_vs_scaleupWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='Scale-up WASH, no MDA'
    ),
    central_measure='mean'
)
total_dalys_averted_vs_scaleupWASH.to_csv(results_folder / f'total_dalys_averted_vs_scaleupWASH{target_period()}.csv')



# percentage DALYs averted, summed across all causes and all districts
pc_dalys_averted_vs_pauseWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='Pause WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)
pc_dalys_averted_vs_pauseWASH.to_csv(results_folder / f'pc_dalys_averted_vs_pauseWASH{target_period()}.csv')

pc_dalys_averted_vs_ContinueWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='Continue WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)
pc_dalys_averted_vs_ContinueWASH.to_csv(results_folder / f'pc_dalys_averted_vs_continueWASH{target_period()}.csv')

pc_dalys_averted_vs_scaleupWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='Scale-up WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)
pc_dalys_averted_vs_scaleupWASH.to_csv(results_folder / f'pc_dalys_averted_vs_scaleupWASH{target_period()}.csv')




# dalys averted by cause
dalys_by_label_averted_vs_pauseWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        num_dalys_by_cause,
        comparison='Pause WASH, no MDA'
    ),
    central_measure='mean'
)
dalys_by_label_averted_vs_pauseWASH.to_csv(results_folder / f'dalys_by_label_averted_vs_pauseWASH{target_period()}.csv')


dalys_by_label_averted_vs_continueWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        num_dalys_by_cause,
        comparison='Continue WASH, no MDA'
    ),
    central_measure='mean'
)
dalys_by_label_averted_vs_continueWASH.to_csv(results_folder / f'dalys_by_label_averted_vs_continueWASH{target_period()}.csv')


dalys_by_label_averted_vs_scaleupWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        num_dalys_by_cause,
        comparison='Scale-up WASH, no MDA'
    ),
    central_measure='mean'
)
dalys_by_label_averted_vs_scaleupWASH.to_csv(results_folder / f'dalys_by_label_averted_vs_scaleupWASH{target_period()}.csv')




# PERCENTAGE DALYS AVERTED BY CAUSE
pc_dalys_by_label_averted_vs_PauseWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        num_dalys_by_cause,
        comparison='Pause WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)
pc_dalys_by_label_averted_vs_PauseWASH.to_csv(results_folder / f'pc_dalys_by_label_averted_vs_PauseWASH{target_period()}.csv')

pc_dalys_by_label_averted_vs_ContinueWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        num_dalys_by_cause,
        comparison='Continue WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)
pc_dalys_by_label_averted_vs_ContinueWASH.to_csv(results_folder / f'pc_dalys_by_label_averted_vs_ContinueWASH{target_period()}.csv')


pc_dalys_by_label_averted_vs_scaleupWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        num_dalys_by_cause,
        comparison='Scale-up WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)
pc_dalys_by_label_averted_vs_scaleupWASH.to_csv(results_folder / f'pc_dalys_by_label_averted_vs_scaleupWASH{target_period()}.csv')




#######################################################################################
# %% PLOTS DALYS AVERTED
#######################################################################################

def plot_clustered_bars_with_error_bars(df: pd.DataFrame):
    """
    Plots a clustered bar chart with error bars based on a multi-index DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with MultiIndex columns (draw names and 'stat' levels 'lower', 'mean', 'upper')
                       and a row index representing categories.
    Returns:
    fig, ax: The figure and axes of the plot.
    """
    # Choose a color palette
    colors = plt.colormaps['Spectral']  # Get the Spectral colormap
    n_draws = df.columns.levels[0].shape[0]  # Number of draws
    color_list = [colors(i / n_draws) for i in range(n_draws)]  # Create a list of colors for each draw

    # Extract the data for plotting
    means = df.xs('central', level='stat', axis=1)
    lowers = df.xs('lower', level='stat', axis=1)
    uppers = df.xs('upper', level='stat', axis=1)

    # Number of groups (categories)
    n_groups = means.shape[0]

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar width and spacing
    bar_width = 0.15  # Width of each individual bar
    cluster_gap = 0.2  # Extra space between clusters

    # Calculate positions of the bars with cluster gaps
    index = np.arange(n_groups) * (n_draws * bar_width + cluster_gap)

    # Plot each draw's mean with error bars
    for i, draw in enumerate(means.columns):
        ax.bar(index + (i * bar_width), means[draw], bar_width,
               label=draw,
               yerr=[means[draw] - lowers[draw], uppers[draw] - means[draw]],
               color=color_list[i],
               capsize=4)

    # Add vertical dashed lines precisely between clusters
    for i in range(1, n_groups):
        ax.axvline(x=index[i] - (cluster_gap / 2), color='grey', linestyle='--')

    # Add horizontal line at y=0
    ax.axhline(y=0, color='grey', linestyle='-')

    # Labeling
    ax.set_xlabel('')
    ax.set_xticks(index + (n_draws - 1) * bar_width / 2)
    ax.set_xticklabels(df.index, rotation=0)

    # Add a legend for the draws
    ax.legend(title='Scenario')

    # Return the figure and axes
    return fig, ax


# DALYs averted vs BASELINE
order_for_plotting_vs_continueWASH = [
    'Continue WASH, MDA SAC',
    'Continue WASH, MDA PSAC',
    'Continue WASH, MDA All',
    'Scale-up WASH, no MDA',
    'Scale-up WASH, MDA SAC',
    'Scale-up WASH, MDA PSAC',
    'Scale-up WASH, MDA All',
]

pc_dalys_averted_ContinueWASH_ordered = pc_dalys_by_label_averted_vs_ContinueWASH.reindex(
    columns=order_for_plotting_vs_continueWASH, level=0)
pc_dalys_averted_ContinueWASH_ordered = pc_dalys_averted_ContinueWASH_ordered.drop(index='Other')


name_of_plot = f'Percentage reduction in DALYs (comparator=continued WASH) {target_period()}'
fig, ax = plot_clustered_bars_with_error_bars(pc_dalys_averted_ContinueWASH_ordered)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in DALYs')
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)


# DALYs averted vs WASH scale-up
order_for_plotting_vs_scaleupWASH = [
        'Continue WASH, no MDA',
    'Continue WASH, MDA SAC',
    'Continue WASH, MDA PSAC',
    'Continue WASH, MDA All',
    'Scale-up WASH, MDA SAC',
    'Scale-up WASH, MDA PSAC',
    'Scale-up WASH, MDA All',
]
pc_dalys_by_label_averted_vs_scaleupWASH_ordered = pc_dalys_by_label_averted_vs_scaleupWASH.reindex(columns=order_for_plotting_vs_scaleupWASH, level=0)
pc_dalys_by_label_averted_vs_scaleupWASH_ordered = pc_dalys_by_label_averted_vs_scaleupWASH_ordered.drop(index='Other')

name_of_plot = f'Percentage reduction in DALYs (comparator=scale-up WASH) {target_period()}'
fig, ax = plot_clustered_bars_with_error_bars(pc_dalys_by_label_averted_vs_scaleupWASH_ordered)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in DALYs')
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)


# -----------------------------------------------------------------------------


# %% INCIDENCE OF ASSOCIATED CONDITIONS -----------------------------------------------------------------------------

# get incidence of associated disorders

# diarrhoea
diarrhoea = extract_results(
    results_folder,
    module="tlo.methods.diarrhoea",
    key="incident_case",
    custom_generate_series=(
        lambda df: pd.Series(
            df.query(f"{TARGET_PERIOD[0].year} <= date.dt.year <= {TARGET_PERIOD[1].year}")["person_id"].count(),
            index=["total"]
        )
    ),
    do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)
diarrhoea.to_csv(results_folder / (f'diarrhoea_incidence {target_period()}.csv'))

diarrhoea_averted_vs_PauseWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='Pause WASH, no MDA'
    ),
    central_measure='mean'
)

diarrhoea_pc_dalys_averted_vs_PauseWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='Pause WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)


diarrhoea_averted_vs_ContinueWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='Continue WASH, no MDA'
    ),
    central_measure='mean'
)

diarrhoea_pc_dalys_averted_vs_ContinueWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='Continue WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)

diarrhoea_averted_vs_scaleupWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='Scale-up WASH, no MDA'
    ),
    central_measure='mean'
)

diarrhoea_pc_dalys_averted_vs_scaleupWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='Scale-up WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)



# ALRI
alri = extract_results(
    results_folder,
    module="tlo.methods.alri",
    key="event_counts",
    custom_generate_series=(
        lambda df: pd.Series(
            df.query(f"{TARGET_PERIOD[0].year} <= date.dt.year <= {TARGET_PERIOD[1].year}")["incident_cases"].sum(),
            index=["total"]
        )
    ),
    do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)
alri.to_csv(results_folder / (f'alri_incidence {target_period()}.csv'))


alri_averted_vs_PauseWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='Pause WASH, no MDA'
    ),
    central_measure='mean'
)

alri_pc_dalys_averted_vs_PauseWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='Pause WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)


alri_averted_vs_ContinueWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='Continue WASH, no MDA'
    ),
    central_measure='mean'
)

alri_pc_dalys_averted_vs_ContinueWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='Continue WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)

alri_averted_vs_scaleupWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='Scale-up WASH, no MDA'
    ),
    central_measure='mean'
)

alri_pc_dalys_averted_vs_scaleupWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='Scale-up WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)



# HIV
hiv = extract_results(
    results_folder,
    module="tlo.methods.hiv",
    key="summary_inc_and_prev_for_adults_and_children_and_fsw",
    custom_generate_series=(
        lambda df: pd.Series(
            df.query(f"{TARGET_PERIOD[0].year} <= date.dt.year <= {TARGET_PERIOD[1].year}")["hiv_adult_inc_1549"].sum(),
            index=["total"]
        )
    ),
    do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)
hiv.to_csv(results_folder / (f'hiv_incidence {target_period()}.csv'))

hiv_averted_vs_PauseWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='Pause WASH, no MDA'
    ),
    central_measure='mean'
)

hiv_pc_dalys_averted_vs_PauseWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='Pause WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)


hiv_averted_vs_ContinueWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='Continue WASH, no MDA'
    ),
    central_measure='mean'
)

hiv_pc_dalys_averted_vs_ContinueWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='Continue WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)

hiv_averted_vs_scaleupWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='Scale-up WASH, no MDA'
    ),
    central_measure='mean'
)

hiv_pc_dalys_averted_vs_scaleupWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='Scale-up WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)



# bladder cancer
bladder = extract_results(
    results_folder,
    module="tlo.methods.bladder_cancer",
    key="summary_stats",
    custom_generate_series=(
        lambda df: pd.Series(
            df.query(f"{TARGET_PERIOD[0].year} <= date.dt.year <= {TARGET_PERIOD[1].year}")["total_tis_t1"].sum(),
            index=["total"]
        )
    ),
    do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)
bladder.to_csv(results_folder / (f'bladder_incidence {target_period()}.csv'))

bladder_averted_vs_PauseWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='Pause WASH, no MDA'
    ),
    central_measure='mean'
)

bladder_pc_dalys_averted_vs_PauseWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='Pause WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)


bladder_averted_vs_ContinueWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='Continue WASH, no MDA'
    ),
    central_measure='mean'
)

bladder_pc_dalys_averted_vs_ContinueWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='Continue WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)


bladder_averted_vs_scaleupWASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='Scale-up WASH, no MDA'
    ),
    central_measure='mean'
)

bladder_pc_dalys_averted_vs_scaleupWASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='Scale-up WASH, no MDA',
        scaled=True
    ),
    central_measure='mean'
)


# %% plot incidence of diseases with each MDR strategy vs Baseline


# combine the dataframes for plotting
schisto_row = (pc_dalys_by_label_averted_vs_ContinueWASH.loc['Schistosomiasis'])
combined_df = pd.concat([schisto_row.to_frame().T,
                         diarrhoea_pc_dalys_averted_vs_ContinueWASH,
                         alri_pc_dalys_averted_vs_ContinueWASH,
                         hiv_pc_dalys_averted_vs_ContinueWASH,
                         bladder_pc_dalys_averted_vs_ContinueWASH], ignore_index=True)
combined_df.index = ['Schistosomiasis', 'Diarrhoea', 'ALRI', 'HIV', 'Bladder cancer']
combined_df.to_csv(results_folder / (f'percentage_dalys_averted_vs_baseline_{target_period()}.csv'))

combined_df_ordered = combined_df.reindex(columns=order_for_plotting_vs_continueWASH, level=0)


name_of_plot = f'Percentage reduction in incidence (comparator=continued WASH) {target_period()}'
fig, ax = plot_clustered_bars_with_error_bars(combined_df_ordered)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in incidence')
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)



# %% plot incidence of diseases with each MDR strategy vs WASH

# combine the dataframes for plotting
schisto_row = (pc_dalys_by_label_averted_vs_scaleupWASH.loc['Schistosomiasis'])
combined_df = pd.concat([schisto_row.to_frame().T,
                         diarrhoea_pc_dalys_averted_vs_scaleupWASH,
                         alri_pc_dalys_averted_vs_scaleupWASH,
                         hiv_pc_dalys_averted_vs_scaleupWASH,
                         bladder_pc_dalys_averted_vs_scaleupWASH], ignore_index=True)
combined_df.index = ['Schistosomiasis', 'Diarrhoea', 'ALRI', 'HIV', 'Bladder cancer']
combined_df.to_csv(results_folder / (f'percentage_dalys_averted_vs_scaleup_WASH_{target_period()}.csv'))

combined_df_ordered = combined_df.reindex(columns=order_for_plotting_vs_scaleupWASH, level=0)


name_of_plot = f'Percentage reduction in incidence (comparator=WASH scale-up) {target_period()}'
fig, ax = plot_clustered_bars_with_error_bars(combined_df_ordered)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in incidence')
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)


##################################################################################
# %%  PERSON-YEARS INFECTED
##################################################################################

# stacked bar plot for each scenario
# not separate for mansoni and haematobium

def get_person_years_infected(_df):
    """Get the person-years for each draw, summed over every district """

    tmp = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]

    # limit to specific age-groups
    if age == 'SAC':
        df = tmp.filter(regex=r'\bSAC\b')  # matches 'SAC' exactly, not 'PSAC'
    elif age == 'PSAC':
        df = tmp.filter(like='PSAC')
    elif age == 'adult':
        df = tmp.filter(like='Adult')
    else:
        df = tmp  # include all age-groups

    if inf == 'HML':
        df_filtered = df.filter(regex='(High-infection|Moderate-infection|Low-infection)')
    if inf == 'HM':
        df_filtered = df.filter(regex='(Moderate-infection|High-infection)')
    if inf == 'H':
        df_filtered = df.filter(regex='(High-infection)')
    if inf == 'ML':
        df_filtered = df.filter(regex='(Moderate-infection|Low-infection)')
    if inf == 'M':
        df_filtered = df.filter(regex='(Moderate-infection)')
    if inf == 'L':
        df_filtered = df.filter(regex='(Low-infection)')

    person_years = df_filtered.sum(axis=1).sum() / 365.25

    return pd.Series(person_years)


# produce dataframes of PY averted for each age-group
# ages = ['PSAC', 'SAC', 'Adults']
# inf = 'HML'  # 'HML' or any combination

# # Initialise empty lists to hold the results for each age group
# num_py_averted_vs_scaleup_WASH_results = []
# pc_py_averted_vs_scaleup_WASH_results = []
#
# for age in ages:
#     person_years = extract_results(
#         results_folder,
#         module="tlo.methods.schisto",
#         key="Schisto_person_days_infected",
#         custom_generate_series=get_person_years_infected,
#         do_scaling=False,  # switch to True for full runs
#     ).pipe(set_param_names_as_column_index_level_0)
#
#     person_years_summary = compute_summary_statistics(person_years, central_measure='median')
#
#     num_py_averted_vs_WASH = compute_summary_statistics(
#         -1.0 * find_difference_relative_to_comparison_dataframe(
#             person_years,
#             comparison='Scale-up WASH, no MDA'
#         ),
#         central_measure='median'
#     )
#
#     pc_py_averted_vs_WASH = 100.0 * compute_summary_statistics(
#         -1.0 * find_difference_relative_to_comparison_dataframe(
#             person_years,
#             comparison='Scale-up WASH, no MDA',
#             scaled=True
#         ),
#         central_measure='median'
#     )
#
#     # Append the results to the corresponding lists
#     num_py_averted_vs_scaleup_WASH_results.append(num_py_averted_vs_WASH)
#     pc_py_averted_vs_scaleup_WASH_results.append(pc_py_averted_vs_WASH)
#
# # Combine results into two DataFrames, with age groups as a single-level row index
# num_py_averted_vs_scaleup_WASH_results = pd.concat(num_py_averted_vs_scaleup_WASH_results, keys=ages, axis=0)
# pc_py_averted_vs_scaleup_WASH_results = pd.concat(pc_py_averted_vs_scaleup_WASH_results, keys=ages, axis=0)
#
# num_py_averted_vs_scaleup_WASH_results.index = num_py_averted_vs_scaleup_WASH_results.index.get_level_values(0)
# pc_py_averted_vs_scaleup_WASH_results.index = pc_py_averted_vs_scaleup_WASH_results.index.get_level_values(0)
#
# num_py_averted_vs_scaleup_WASH_results.to_csv(results_folder / (f'num_py_averted_vs_scaleup_WASH_results {target_period()}.csv'))
# pc_py_averted_vs_scaleup_WASH_results.to_csv(results_folder / (f'pc_py_averted_vs_scaleup_WASH_results {target_period()}.csv'))
#
#
# # repeat for baseline comparator
# num_py_averted_vs_baseline_results = []
# pc_py_averted_vs_baseline_results = []
#
# for age in ages:
#     person_years = extract_results(
#         results_folder,
#         module="tlo.methods.schisto",
#         key="Schisto_person_days_infected",
#         custom_generate_series=get_person_years_infected,
#         do_scaling=False,  # switch to True for full runs
#     ).pipe(set_param_names_as_column_index_level_0)
#
#     person_years_summary = compute_summary_statistics(person_years, central_measure='median')
#
#     num_py_averted_vs_baseline = compute_summary_statistics(
#         -1.0 * find_difference_relative_to_comparison_dataframe(
#             person_years,
#             comparison='Continue WASH, no MDA'
#         ),
#         central_measure='median'
#     )
#
#     pc_py_averted_vs_baseline = 100.0 * compute_summary_statistics(
#         -1.0 * find_difference_relative_to_comparison_dataframe(
#             person_years,
#             comparison='Continue WASH, no MDA',
#             scaled=True
#         ),
#         central_measure='median'
#     )
#
#     # Append the results to the corresponding lists
#     num_py_averted_vs_baseline_results.append(num_py_averted_vs_baseline)
#     pc_py_averted_vs_baseline_results.append(pc_py_averted_vs_baseline)
#
# # Combine results into two DataFrames, with age groups as a single-level row index
# num_py_averted_vs_baseline_results = pd.concat(num_py_averted_vs_baseline_results, keys=ages, axis=0)
# pc_py_averted_vs_baseline_results = pd.concat(pc_py_averted_vs_baseline_results, keys=ages, axis=0)
#
# num_py_averted_vs_baseline_results.index = num_py_averted_vs_baseline_results.index.get_level_values(0)
# pc_py_averted_vs_baseline_results.index = pc_py_averted_vs_baseline_results.index.get_level_values(0)
#
# num_py_averted_vs_baseline_results.to_csv(results_folder / (f'num_py_averted_vs_baseline_results {target_period()}.csv'))
# pc_py_averted_vs_baseline_results.to_csv(results_folder / (f'pc_py_averted_vs_baseline_results {target_period()}.csv'))


# def plot_averted_points_with_errorbars(_df):
#     # Set the color palette for the age groups
#     age_groups = _df.index
#     num_age_groups = len(age_groups)
#     age_colors = sns.color_palette("Set2", num_age_groups)
#
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     # X positions for each draw
#     x_positions = np.arange(len(_df.columns.levels[0]))
#
#     # Stagger points slightly for each age group
#     stagger = np.linspace(-0.2, 0.2, num_age_groups)
#
#     # Loop over each draw (column level 0)
#     for i, draw in enumerate(_df.columns.levels[0]):
#         mean_values = _df[(draw, 'central')]
#         lower_values = _df[(draw, 'lower')]
#         upper_values = _df[(draw, 'upper')]
#
#         # Plot each age group with staggered x positions
#         for j, age_group in enumerate(age_groups):
#             ax.errorbar(
#                 x=x_positions[i] + stagger[j],
#                 y=mean_values[age_group],
#                 yerr=[[mean_values[age_group] - lower_values[age_group]],
#                       [upper_values[age_group] - mean_values[age_group]]],
#                 fmt='o',  # Points for the mean values
#                 color=age_colors[j],  # Assign color per age group
#                 capsize=5,
#                 label=age_group if i == 0 else ""  # Only label age groups once
#             )
#
#     # Add grey vertical dashed lines between each draw
#     for i in range(1, len(x_positions)):
#         ax.axvline(x_positions[i] - 0.5, color='grey', linestyle='--')
#
#     # Add horizontal grey line at y=0
#     ax.axhline(0, color='grey', linestyle='-')
#
#     # Customize ticks and labels
#     ax.set_xticks(x_positions)
#     ax.set_xticklabels(_df.columns.levels[0], rotation=45, ha='right')
#     ax.set_ylim(top=80)
#
#     # Add legend for age groups
#     handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', label=age_group)
#                for age_group, color in zip(age_groups, age_colors)]
#     ax.legend(handles=handles, title="Age Group")
#
#     return fig, ax
#
#
# pc_py_averted_vs_WASH_results_ordered = pc_py_averted_vs_scaleup_WASH_results.reindex(columns=order_for_plotting_vs_scaleup_WASH, level=0)
#
# name_of_plot = f'Percentage reduction in person-years infected with Schistosomiasis vs WASH scale-up {target_period()}'
# fig, ax = plot_averted_points_with_errorbars(pc_py_averted_vs_WASH_results_ordered)
# ax.set_title(name_of_plot)
# ax.set_ylabel('Percentage reduction in Person-Years Infected')
# ax.set_ylim(-40, 40)
# fig.tight_layout()
# fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
# fig.show()
# plt.close(fig)
#
#
# # todo replot this
# pc_py_averted_vs_baseline_ordered = pc_py_averted_vs_baseline_results.reindex(columns=order_for_plotting_vs_baseline, level=0)
#
# name_of_plot = f'Percentage reduction in person-years infected with Schistosomiasis vs baseline {target_period()}'
# fig, ax = plot_averted_points_with_errorbars(pc_py_averted_vs_baseline_ordered)
# ax.set_title(name_of_plot)
# ax.set_ylabel('Percentage reduction in Person-Years Infected')
# ax.set_ylim(-40, 40)
# fig.tight_layout()
# fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
# fig.show()
# plt.close(fig)


def generate_py_averted_by_age(results_folder, comparator_scenario):

    def wrap_get_person_years_infected(age_group, inf_level):
        """
        this makes sure that the arguments age and inf are updated within each
        call in the loop "for age in ages:
        """
        def wrapper(df):
            globals()['age'] = age_group
            globals()['inf'] = inf_level
            return get_person_years_infected(df)

        return wrapper

    num_py_averted_results = []
    pc_py_averted_results = []

    ages = ['SAC', 'PSAC', 'Adults']  # needed for get_person_years_infected
    inf = 'HML'  # needed for get_person_years_infected

    for age in ages:
        custom_series_func = wrap_get_person_years_infected(age, inf)

        # Extract age-specific person-years infected
        person_years = extract_results(
                    results_folder,
                    module="tlo.methods.schisto",
                    key="Schisto_person_days_infected",
                    custom_generate_series=custom_series_func,
                    do_scaling=True,
                ).pipe(set_param_names_as_column_index_level_0)

        # Compute absolute difference vs comparator
        num_py_averted = compute_summary_statistics(
            -1.0 * find_difference_relative_to_comparison_dataframe(
                person_years,
                comparison=comparator_scenario,
            ),
            central_measure='median'
        )

        pc_py_averted = 100.0 * compute_summary_statistics(
            -1.0 * find_difference_relative_to_comparison_dataframe(
                person_years,
                comparison=comparator_scenario,
                scaled=True
            ),
            central_measure='median'
        )

        # Append with age-group as row label
        num_py_averted_results.append(num_py_averted.assign(age=age))
        pc_py_averted_results.append(pc_py_averted.assign(age=age))

    # Combine into single DataFrames with 'age' as index
    num_py_averted_df = pd.concat(num_py_averted_results).set_index('age')
    pc_py_averted_df = pd.concat(pc_py_averted_results).set_index('age')

    # Save results
    suffix = f"{comparator_scenario.replace(' ', '_')}_{target_period()}"
    num_py_averted_df.to_csv(results_folder / f'num_py_averted_vs_{suffix}.csv')
    pc_py_averted_df.to_csv(results_folder / f'pc_py_averted_vs_{suffix}.csv')

    return num_py_averted_df, pc_py_averted_df


# todo need to get three sets of comparators
# this will generate comparisons across every scenario, then limit to those that are relevant
py_averted_pause = generate_py_averted_by_age(results_folder, comparator_scenario='Pause WASH, no MDA')
py_averted_continue = generate_py_averted_by_age(results_folder, comparator_scenario='Continue WASH, no MDA')
py_averted_scaleup = generate_py_averted_by_age(results_folder, comparator_scenario='Scale-up WASH, no MDA')


def plot_grouped_averted_points(data_dict, ylabel=None, ylim=None):

    row_labels = ['Pause', 'Continue', 'Scale-up']
    scenario_keywords = ['MDA SAC', 'MDA PSAC', 'MDA All']
    age_groups = ['SAC', 'PSAC', 'Adults']
    palette = sns.color_palette("Set2", len(age_groups))
    stagger = np.linspace(-0.2, 0.2, len(age_groups))

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    plt.subplots_adjust(hspace=0)  # No space between rows

    for row_idx, row_label in enumerate(row_labels):
        ax = axes[row_idx]
        df = data_dict[row_label]

        # Extract relevant draws, exclude 'MDA None'
        all_draws = df.columns.levels[0]
        selected_draws = [d for d in all_draws if row_label in d and 'MDA None' not in d]

        # Sort draws into groups by MDA strategy
        grouped_draws = []
        for keyword in scenario_keywords:
            grouped_draws.extend([d for d in selected_draws if keyword in d])

        x_positions = np.arange(len(grouped_draws))
        group_boundaries = []
        idx = 0
        for keyword in scenario_keywords:
            count = sum(keyword in d for d in grouped_draws)
            if count > 0:
                idx += count
                group_boundaries.append(idx - 0.5)

        # Plot each draw with staggered points for age groups
        for i, draw in enumerate(grouped_draws):
            for j, age_group in enumerate(age_groups):
                mean = df.loc[age_group, (draw, 'central')]
                lower = df.loc[age_group, (draw, 'lower')]
                upper = df.loc[age_group, (draw, 'upper')]
                ax.errorbar(
                    x=i + stagger[j],
                    y=mean,
                    yerr=[[mean - lower], [upper - mean]],
                    fmt='o',
                    color=palette[j],
                    capsize=5,
                    label=age_group if (i == 0 and row_idx == 0) else ""
                )

        # Vertical dashed lines between MDA groups
        for boundary in group_boundaries[:-1]:
            ax.axvline(boundary, color='grey', linestyle='--')

        ax.axhline(0, color='grey', linestyle='-')
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0, top=ylim)  # Cut at 0

        if row_idx < 2:
            ax.set_xticks([])
        else:
            # Set simplified x-axis labels for bottom row
            label_positions = []
            start = 0
            for keyword in scenario_keywords:
                count = sum(keyword in d for d in grouped_draws)
                if count > 0:
                    mid = start + (count - 1) / 2
                    label_positions.append(mid)
                    start += count
            ax.set_xticks(label_positions)
            ax.set_xticklabels(['MDA SAC', 'MDA PSAC', 'MDA ALL'])
            ax.tick_params(axis='x', which='both', length=0)  # Remove tick marks

        # Right-side panel label
        ax.text(1.02, 0.5, row_label, transform=ax.transAxes,
                rotation=270, va='center', fontsize=12, fontweight='bold')

    # Legend: top left of first panel
    handles = [plt.Line2D([0], [0], marker='o', color=palette[i], linestyle='', label=age)
               for i, age in enumerate(age_groups)]
    axes[0].legend(handles=handles, title='Age Group', loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    return fig


# todo check this plotting the correct figures as code has been updated
fig = plot_grouped_averted_points({
    'Pause': py_averted_pause[1],
    'Continue': py_averted_continue[1],
    'Scale-up': py_averted_scaleup[1]
}, ylabel='% py averted', ylim=40)
plt.show()

fig = plot_grouped_averted_points({
    'Pause': py_averted_pause[0] / 1_000_000,
    'Continue': py_averted_continue[0] / 1_000_000,
    'Scale-up': py_averted_scaleup[0] / 1_000_000
}, ylabel='Number py averted, millions', ylim=40)
plt.show()




##################################################################################
# %%  PREVALENCE OF INFECTION BY SPECIES / AGE-GROUP
##################################################################################

def get_prevalence_infection(_df):
    """Get the prevalence every year of the simulation """

    # select the last entry for each year
    _df.set_index('date', inplace=True)
    df = _df.resample('Y').last()

    # limit to specific age-groups
    # if age=All, then don't filter - all age-groups included
    if age == 'SAC':
        df = df.filter(like='SAC')
    if age == 'Adult':
        df = df.filter(like='Adult')
    if age == 'PSAC':
        df = df.filter(like='PSAC')
    if age == 'Infant':
        df = df.filter(like='Infant')

    # Aggregate the sums of infection statuses by district_of_residence and year
    # this df is filtered by age-group
    district_sum = df.sum(axis=1)

    if inf == 'HML':
        df_filtered = df.filter(regex='(High-infection|Moderate-infection|Low-infection)')
    if inf == 'HM':
        df_filtered = df.filter(regex='(Moderate-infection|High-infection)')
    if inf == 'ML':
        df_filtered = df.filter(regex='(Moderate-infection|Low-infection)')
    if inf == 'H':
        df_filtered = df.filter(regex='(High-infection)')
    if inf == 'M':
        df_filtered = df.filter(regex='(Moderate-infection)')
    if inf == 'L':
        df_filtered = df.filter(regex='(Low-infection)')

    infected = df_filtered.sum(axis=1)

    prop_infected = infected.div(district_sum)

    return prop_infected


age = 'All'  # SAC, Adult, all, infant, PSAC
inf = 'HML'

prev_haem_HML_All = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_haematobium",
    custom_generate_series=get_prevalence_infection,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)
prev_haem_HML_All.index = prev_haem_HML_All.index.year
prev_haem_HML_All.to_csv(results_folder / (f'prev_haem_HML_All {target_period()}.csv'))


prev_mansoni_HML_All = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_mansoni",
    custom_generate_series=get_prevalence_infection,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)
prev_mansoni_HML_All.index = prev_mansoni_HML_All.index.year
prev_mansoni_HML_All.to_csv(results_folder / (f'prev_mansoni_HML_All {target_period()}.csv'))


age = 'All'  # SAC, Adult, all, infant, PSAC
inf = 'H'

prev_haem_H_All = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_haematobium",
    custom_generate_series=get_prevalence_infection,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)
prev_haem_H_All.index = prev_haem_H_All.index.year
prev_haem_H_All.to_csv(results_folder / (f'prev_haem_H_All {target_period()}.csv'))


prev_mansoni_H_All = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_mansoni",
    custom_generate_series=get_prevalence_infection,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)

prev_mansoni_H_All.index = prev_mansoni_H_All.index.year
prev_mansoni_H_All.to_csv(results_folder / (f'prev_mansoni_H_All {target_period()}.csv'))


summary_prev_mansoni_H_All = compute_summary_statistics(prev_mansoni_H_All, central_measure='median')
summary_prev_mansoni_H_All.to_csv(results_folder / (f'summary_prev_mansoni_H_All {target_period()}.csv'))

summary_prev_haem_H_All = compute_summary_statistics(prev_haem_H_All, central_measure='median')
summary_prev_haem_H_All.to_csv(results_folder / (f'summary_prev_haem_H_All {target_period()}.csv'))



####################################################################################
# %%  PLOT PREVALENCE OF INFECTION BY SPECIES OVER TIME - INDIVIDUAL RUNS
####################################################################################


# def plot_lines_by_draw(df: pd.DataFrame, title):
#     """
#     Plot lines by (draw, run), grouped by draw colour. Excludes draws starting with 'Pause' except 'Pause WASH, no MDA'.
#     """
#
#     # Exclude year 2010
#     df = df[df.index != 2010]
#     # df = df[df.index >= 2024]
#
    # # Desired draw order for legend
    # ordered_draws = [
    #     'Pause WASH, no MDA',
    #     'Pause WASH, MDA SAC',
    #     'Pause WASH, MDA PSAC',
    #     'Pause WASH, MDA All',
    #     'Continue WASH, no MDA',
    #     'Continue WASH, MDA SAC',
    #     'Continue WASH, MDA PSAC',
    #     'Continue WASH, MDA All',
    #     'Scale-up WASH, no MDA',
    #     'Scale-up WASH, MDA SAC',
    #     'Scale-up WASH, MDA PSAC',
    #     'Scale-up WASH, MDA All'
    # ]
    #
    # # Filter columns: exclude draws starting with "Pause" unless exact match
    # filtered_cols = [col for col in df.columns
    #                  if (col[0] in ordered_draws)]
    #
    # df = df[filtered_cols]
    #
    # # Assign colours using a striking colormap
    # colours = plt.colormaps['tab10'](np.linspace(0, 1, len(ordered_draws)))
    # colour_map = dict(zip(ordered_draws, colours))
#
#     # Set up plot
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     # Plot each (draw, run) line with colour by draw
#     for (draw, run) in df.columns:
#         ax.plot(df.index, df[(draw, run)],
#                 label=f"{draw} - {run}",
#                 color=colour_map[draw],
#                 alpha=0.8)
#
#     # One legend entry per draw, in the specified order
#     custom_lines = [Line2D([0], [0], color=colour_map[draw], lw=1) for draw in ordered_draws if draw in colour_map]
#     ax.legend(custom_lines, [draw for draw in ordered_draws if draw in colour_map],
#               title="", loc='upper right', frameon=True)
#
#     # Axis labelling
#     ax.set_xlabel("")
#     ax.set_ylabel("Prevalence")
#     ax.set_title(title)
#
#     # Layout
#     fig.tight_layout()
#
#     return fig, ax


# name_of_plot = f'Prevalence of high-intensity infections (mansoni), all ages {target_period()}'
# fig, ax = plot_lines_by_draw(prev_mansoni_H_All, title=name_of_plot)
# fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
# plt.show()
#
# name_of_plot = f'Prevalence of high-intensity infections (haematobium), all ages {target_period()}'
# fig, ax = plot_lines_by_draw(prev_haem_H_All, title=name_of_plot)
# fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
# plt.show()




####################################################################################
# %%  PLOT PREVALENCE OF INFECTION BY SPECIES OVER TIME - SUMMARY BANDS
####################################################################################

#
# def plot_draws_with_ci(df: pd.DataFrame, title: str):
#     """
#     Plot median lines with shaded confidence intervals by draw.
#     Custom colours based on MDA category and line styles based on WASH scale.
#
#     Parameters:
#     df (pd.DataFrame): DataFrame with index = year, columns = MultiIndex (draw, stat)
#     title (str): Title for the plot
#
#     Returns:
#     fig, ax: matplotlib figure and axis
#     """
#
#     # Exclude year 2010
#     # todo change this for full time-series
#     # df = df[df.index != 2010]
#     df = df[df.index >= 2023]
#
#     # Desired draw order for legend and colour assignment
#     ordered_draws = [
#         'Pause WASH, no MDA',
#         'Continue WASH, no MDA',
#         'Continue WASH, MDA SAC',
#         'Continue WASH, MDA PSAC',
#         'Continue WASH, MDA All',
#         'Scale-up WASH, no MDA',
#         'Scale-up WASH, MDA SAC',
#         'Scale-up WASH, MDA PSAC',
#         'Scale-up WASH, MDA All'
#     ]
#
#     # Filter draws to include only those in the specified order
#     filtered_draws = [draw for draw in df.columns.levels[0] if draw in ordered_draws]
#     df = df.loc[:, df.columns.get_level_values(0).isin(filtered_draws)]
#
#     # Define preset colours based on MDA type
#     mda_colours = {
#         'no MDA': '#1b9e77',
#         'MDA SAC': '#d95f02',
#         'MDA PSAC': '#7570b3',
#         'MDA All': '#e7298a'
#     }
#
#     def get_colour(draw):
#         for key in mda_colours:
#             if key in draw:
#                 return mda_colours[key]
#         return '#000000'  # default black
#
#     def get_linestyle(draw):
#         if 'Scale-up WASH' in draw:
#             return '-'
#         elif 'Continue WASH' in draw:
#             return '--'
#         elif 'Pause WASH' in draw:
#             return (0, (1, 1))  # dotted
#         return '-'  # fallback
#
#     # Set up plot
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     # Plot median and CI for each draw
#     for draw in ordered_draws:
#         if draw in df.columns.get_level_values(0):
#             median = df[(draw, 'central')]
#             lower = df[(draw, 'lower')]
#             upper = df[(draw, 'upper')]
#
#             colour = get_colour(draw)
#             linestyle = get_linestyle(draw)
#
#             ax.plot(df.index, median, label=draw, color=colour, linestyle=linestyle, lw=1.5)
#             ax.fill_between(df.index, lower, upper, color=colour, alpha=0.3)
#
#     # Custom legend
#     custom_lines = [Line2D([0], [0], color=get_colour(draw), linestyle=get_linestyle(draw), lw=1.5)
#                     for draw in ordered_draws if draw in df.columns.get_level_values(0)]
#     ax.legend(custom_lines, [draw for draw in ordered_draws if draw in df.columns.get_level_values(0)],
#               loc='upper right', frameon=True)
#
#     # Axis labelling
#     ax.set_xlabel("")
#     ax.set_ylabel("Prevalence")
#     ax.set_title(title)
#
#     fig.tight_layout()
#     return fig, ax




def plot_combined_prevalence(summary_prev_mansoni_H_All, summary_prev_haem_H_All, target_period):
    """
    Plot combined summary prevalence for mansoni and haematobium infections, all ages,
    as two vertically stacked subplots with consistent legend formatting.
    Legends are placed side-by-side in the top-right corner of the top subplot.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 15), sharex=True)

    mda_colours = {
        'no MDA': '#1b9e77',
        'MDA SAC': '#d95f02',
        'MDA PSAC': '#7570b3',
        'MDA All': '#e7298a'
    }

    def get_colour(draw):
        for key in mda_colours:
            if key in draw:
                return mda_colours[key]
        return '#000000'

    def get_linestyle(draw):
        if 'Scale-up WASH' in draw:
            return '-'
        elif 'Continue WASH' in draw:
            return '--'
        elif 'Pause WASH' in draw:
            return (0, (1, 1))
        return '-'

    ordered_draws = [
        'Pause WASH, no MDA',
        'Pause WASH, MDA SAC',
        'Pause WASH, MDA PSAC',
        'Pause WASH, MDA All',
        'Continue WASH, no MDA',
        'Continue WASH, MDA SAC',
        'Continue WASH, MDA PSAC',
        'Continue WASH, MDA All',
        'Scale-up WASH, no MDA',
        'Scale-up WASH, MDA SAC',
        'Scale-up WASH, MDA PSAC',
        'Scale-up WASH, MDA All'
    ]

    def plot_on_ax(df, ax, title):
        df = df[df.index >= 2023]
        df = df.loc[:, df.columns.get_level_values(0).isin(ordered_draws)]
        for draw in ordered_draws:
            if draw in df.columns.get_level_values(0):
                median = df[(draw, 'central')]
                lower = df[(draw, 'lower')]
                upper = df[(draw, 'upper')]
                ax.plot(df.index, median, label=draw,
                        color=get_colour(draw), linestyle=get_linestyle(draw), lw=1.5)
                ax.fill_between(df.index, lower, upper,
                                color=get_colour(draw), alpha=0.3)
        ax.set_title(title)
        ax.set_ylabel("Prevalence")
        ax.set_ylim(0, 0.035)
        ax.set_xticks(df.index)
        ax.set_xticklabels(df.index.astype(int))

    # Plot mansoni
    plot_on_ax(summary_prev_mansoni_H_All, axes[0],
               f'Summary prevalence of high-intensity infections (mansoni), all ages {target_period()}')

    # Plot haematobium
    plot_on_ax(summary_prev_haem_H_All, axes[1],
               f'Summary prevalence of high-intensity infections (haematobium), all ages {target_period()}')
    axes[1].set_xlabel("Year")

    # Create legends
    colour_legend = [
        Line2D([0], [0], color=col, lw=1)
        for col in mda_colours.values()
    ]
    colour_labels = list(mda_colours.keys())

    context_styles = {
        'Pause WASH': (0, (1, 1)),
        'Continue WASH': '--',
        'Scale-up WASH': '-'
    }
    style_legend = [
        Line2D([0], [0], color='black', linestyle=ls, lw=1)
        for ls in context_styles.values()
    ]
    style_labels = list(context_styles.keys())

    # Add legends to top axis: WASH context right of MDA strategy
    legend1 = axes[0].legend(
        colour_legend, colour_labels,
        title="MDA Strategy",
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        frameon=True
    )
    axes[0].add_artist(legend1)

    legend2 = axes[0].legend(
        style_legend, style_labels,
        title="WASH Context",
        loc='upper right',
        bbox_to_anchor=(0.85, 1.0),  # adjust leftward within plot
        frameon=True
    )

    fig.tight_layout()
    return fig

fig = plot_combined_prevalence(summary_prev_mansoni_H_All, summary_prev_haem_H_All, target_period)
fig.savefig(make_graph_file_name("combined_summary_prevalence_high_intensity_all_ages"))
plt.show()









