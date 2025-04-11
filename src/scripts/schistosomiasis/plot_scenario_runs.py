""" use the outputs from scenario_runs.py and produce plots
and summary statistics for paper

JOB ID:
schisto_scenarios-2025-03-22T130153Z
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

from scripts.costing.cost_estimation import (estimate_input_cost_of_scenarios,
                                             summarize_cost_data,
                                             do_stacked_bar_plot_of_cost_by_category,
                                             do_line_plot_of_cost,
                                             create_summary_treemap_by_cost_subgroup,
                                             estimate_projected_health_spending)

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
# todo update
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


# %% EXTRACT DALYS


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

num_dalys_compute_summary_statistics = \
compute_summary_statistics(total_num_dalys, central_measure='median').loc[0].unstack().reindex(
    param_names)
num_dalys_compute_summary_statistics.to_csv(results_folder / f'total_num_dalys_{target_period()}.csv')


def get_total_num_dalys_by_label(_df):
    """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
    y = _df \
        .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
        .drop(columns=['date', 'year', 'li_wealth']) \
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


total_num_dalys_by_label = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_by_wealth_stacked_by_age_and_time",
    custom_generate_series=get_total_num_dalys_by_label,
    do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)

total_num_dalys_by_label_compute_summary_statistics = compute_summary_statistics(total_num_dalys_by_label, central_measure='median')
total_num_dalys_by_label_compute_summary_statistics.to_csv(results_folder / f'total_num_dalys_by_label_{target_period()}.csv')


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


# total number of DALYs
total_num_dalys_averted_vs_baseline = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='No WASH, no MDA'
    ),
    central_measure='median'
)
total_num_dalys_averted_vs_baseline.to_csv(results_folder / f'total_num_dalys_averted_vs_baseline{target_period()}.csv')

total_num_dalys_averted_vs_baseline_vs_WASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='WASH only'
    ),
    central_measure='median'
)
total_num_dalys_averted_vs_baseline_vs_WASH.to_csv(results_folder / f'total_num_dalys_averted_vs_baseline_vs_WASH{target_period()}.csv')


# NUMBERS OF DALYS BY CAUSE
num_dalys_by_label_averted_vs_baseline = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys_by_label,
        comparison='No WASH, no MDA'
    ),
    central_measure='median'
)
num_dalys_by_label_averted_vs_baseline.to_csv(results_folder / f'num_dalys_by_label_averted_vs_baseline{target_period()}.csv')


num_dalys_by_label_averted_vs_WASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys_by_label,
        comparison='WASH only'
    ),
    central_measure='median'
)
num_dalys_by_label_averted_vs_WASH.to_csv(results_folder / f'num_dalys_by_label_averted_vs_WASH{target_period()}.csv')


# PERCENTAGE DALYS AVERTED - TOTAL
pc_dalys_averted_total = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='No WASH, no MDA',
        scaled=True
    ),
    central_measure='median'
)
pc_dalys_averted_total.to_csv(results_folder / f'pc_dalys_averted_total{target_period()}.csv')

pc_dalys_averted_WASH_total = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='WASH only',
        scaled=True
    ),
    central_measure='median'
)
pc_dalys_averted_WASH_total.to_csv(results_folder / f'pc_dalys_averted_WASH_total{target_period()}.csv')

# PERCENTAGE DALYS AVERTED BY CAUSE
pc_dalys_averted = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys_by_label,
        comparison='No WASH, no MDA',
        scaled=True
    ),
    central_measure='median'
)
pc_dalys_averted.to_csv(results_folder / f'pc_dalys_averted{target_period()}.csv')

pc_dalys_averted_WASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys_by_label,
        comparison='WASH only',
        scaled=True
    ),
    central_measure='median'
)
pc_dalys_averted_WASH.to_csv(results_folder / f'pc_dalys_averted_WASH{target_period()}.csv')


# %% PLOTS DALYS RELATIVE TO WASH ONLY

order_for_plotting = [
    'No WASH, no MDA',
    'MDA SAC with no WASH',
    'MDA PSAC with no WASH',
    'MDA All with no WASH',
    'MDA SAC with WASH',
    'MDA PSAC with WASH',
    'MDA All with WASH'
]


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
pc_dalys_averted_WASH_ordered = pc_dalys_averted_WASH.reindex(columns=order_for_plotting, level=0)
pc_dalys_averted_WASH_ordered = pc_dalys_averted_WASH_ordered.drop(index='Other')

name_of_plot = f'Percentage reduction in DALYs versus WASH only {target_period()}'
fig, ax = plot_clustered_bars_with_error_bars(pc_dalys_averted_WASH_ordered)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in DALYs')
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)


# DALYs averted vs BASELINE
order_for_plotting_vs_baseline = [
    'MDA SAC with no WASH',
    'MDA PSAC with no WASH',
    'MDA All with no WASH',
    'WASH only'
    'MDA SAC with WASH',
    'MDA PSAC with WASH',
    'MDA All with WASH'
]
pc_dalys_averted_ordered = pc_dalys_averted.reindex(columns=order_for_plotting_vs_baseline, level=0)
pc_dalys_averted__ordered = pc_dalys_averted_ordered.drop(index='Other')

name_of_plot = f'Percentage reduction in DALYs versus Baseline {target_period()}'
fig, ax = plot_clustered_bars_with_error_bars(pc_dalys_averted_ordered)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in DALYs')
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)


# %% DALYS by ALL CAUSES-----------------------------------------------------------------------------
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

daly_summary = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True,
    ),
    central_measure='median',
)

daly_summary = round_to_nearest_100(daly_summary)
daly_summary = daly_summary.astype(int)
daly_summary.to_csv(results_folder / (f'DALYs_by_cause {target_period()}.csv'))

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


diarrhoea_averted_vs_WASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='WASH only'
    ),
    central_measure='median'
)

diarrhoea_pc_dalys_averted_vs_WASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='WASH only',
        scaled=True
    ),
    central_measure='median'
)

diarrhoea_averted_vs_baseline = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='No WASH, no MDA'
    ),
    central_measure='median'
)

diarrhoea_pc_dalys_averted_vs_baseline = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='No WASH, no MDA',
        scaled=True
    ),
    central_measure='median'
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

alri_averted_vs_WASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='WASH only'
    ),
    central_measure='median'
)

alri_pc_dalys_averted_vs_WASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='WASH only',
        scaled=True
    ),
    central_measure='median'
)

alri_averted_vs_baseline = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='No WASH, no MDA'
    ),
    central_measure='median'
)

alri_pc_dalys_averted_vs_baseline = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='No WASH, no MDA',
        scaled=True
    ),
    central_measure='median'
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


hiv_averted_vs_WASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='WASH only'
    ),
    central_measure='median'
)

hiv_pc_dalys_averted_vs_WASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='WASH only',
        scaled=True
    ),
    central_measure='median'
)

hiv_averted_vs_baseline = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='No WASH, no MDA'
    ),
    central_measure='median'
)

hiv_pc_dalys_averted_vs_baseline = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='No WASH, no MDA',
        scaled=True
    ),
    central_measure='median'
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

bladder_averted_vs_WASH = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='WASH only'
    ),
    central_measure='median'
)

bladder_pc_dalys_averted_vs_WASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='WASH only',
        scaled=True
    ),
    central_measure='median'
)

bladder_averted_vs_baseline = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='No WASH, no MDA'
    ),
    central_measure='median'
)

bladder_pc_dalys_averted_vs_baseline = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='No WASH, no MDA',
        scaled=True
    ),
    central_measure='median'
)


# %% plot incidence of diseases with each MDR strategy vs WASH

# combine the dataframes for plotting
schisto_row = (pc_dalys_averted_WASH.loc['Schisto'])
combined_df = pd.concat([schisto_row.to_frame().T,
                         diarrhoea_pc_dalys_averted_vs_WASH,
                         alri_pc_dalys_averted_vs_WASH,
                         hiv_pc_dalys_averted_vs_WASH,
                         bladder_pc_dalys_averted_vs_WASH], ignore_index=True)
combined_df.index = ['Schistosomiasis', 'Diarrhoea', 'ALRI', 'HIV', 'Bladder cancer']
combined_df.to_csv(results_folder / (f'percentage_dalys_averted_vs_WASH_{target_period()}.csv'))

combined_df_ordered = combined_df.reindex(columns=order_for_plotting, level=0)


name_of_plot = f'Percentage reduction in incidence versus WASH {target_period()}'
fig, ax = plot_clustered_bars_with_error_bars(combined_df_ordered)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in incidence')
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)


# %% plot incidence of diseases with each MDR strategy vs Baseline


# combine the dataframes for plotting
schisto_row = (pc_dalys_averted.loc['Schisto'])
combined_df = pd.concat([schisto_row.to_frame().T,
                         diarrhoea_pc_dalys_averted_vs_baseline,
                         alri_pc_dalys_averted_vs_baseline,
                         hiv_pc_dalys_averted_vs_baseline,
                         bladder_pc_dalys_averted_vs_baseline], ignore_index=True)
combined_df.index = ['Schistosomiasis', 'Diarrhoea', 'ALRI', 'HIV', 'Bladder cancer']
combined_df.to_csv(results_folder / (f'percentage_dalys_averted_vs_baseline_{target_period()}.csv'))

combined_df_ordered = combined_df.reindex(columns=order_for_plotting, level=0)


name_of_plot = f'Percentage reduction in incidence versus Baseline {target_period()}'
fig, ax = plot_clustered_bars_with_error_bars(combined_df_ordered)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in incidence')
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)


# %%  -----------------------------------------------------------------------------

# PERSON-YEARS INFECTED
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
    if inf == 'ML':
        df_filtered = df.filter(regex='(Moderate-infection|Low-infection)')
    if inf == 'M':
        df_filtered = df.filter(regex='(Moderate-infection)')
    if inf == 'L':
        df_filtered = df.filter(regex='(Low-infection)')

    person_years = df_filtered.sum(axis=1).sum() / 365.25

    return pd.Series(person_years)


# produce dataframes of PY averted for each age-group
ages = ['PSAC', 'SAC', 'Adults']
inf = 'HML'  # 'HML' or any combination

# Initialise empty lists to hold the results for each age group
num_py_averted_vs_WASH_results = []
pc_py_averted_vs_WASH_results = []

for age in ages:
    person_years = extract_results(
        results_folder,
        module="tlo.methods.schisto",
        key="Schisto_person_days_infected",
        custom_generate_series=get_person_years_infected,
        do_scaling=False,  # switch to True for full runs
    ).pipe(set_param_names_as_column_index_level_0)

    person_years_summary = compute_summary_statistics(person_years, central_measure='median')

    num_py_averted_vs_WASH = compute_summary_statistics(
        -1.0 * find_difference_relative_to_comparison_dataframe(
            person_years,
            comparison='WASH only'
        ),
        central_measure='median'
    )

    pc_py_averted_vs_WASH = 100.0 * compute_summary_statistics(
        -1.0 * find_difference_relative_to_comparison_dataframe(
            person_years,
            comparison='WASH only',
            scaled=True
        ),
        central_measure='median'
    )

    # Append the results to the corresponding lists
    num_py_averted_vs_WASH_results.append(num_py_averted_vs_WASH)
    pc_py_averted_vs_WASH_results.append(pc_py_averted_vs_WASH)

# Combine results into two DataFrames, with age groups as a single-level row index
num_py_averted_vs_WASH_results = pd.concat(num_py_averted_vs_WASH_results, keys=ages, axis=0)
pc_py_averted_vs_WASH_results = pd.concat(pc_py_averted_vs_WASH_results, keys=ages, axis=0)

num_py_averted_vs_WASH_results.index = num_py_averted_vs_WASH_results.index.get_level_values(0)
pc_py_averted_vs_WASH_results.index = pc_py_averted_vs_WASH_results.index.get_level_values(0)

num_py_averted_vs_WASH_results.to_csv(results_folder / (f'num_py_averted_vs_WASH_results {target_period()}.csv'))
pc_py_averted_vs_WASH_results.to_csv(results_folder / (f'pc_py_averted_vs_WASH_results {target_period()}.csv'))


def plot_averted_points_with_errorbars(_df):
    # Set the color palette for the age groups
    age_groups = _df.index
    num_age_groups = len(age_groups)
    age_colors = sns.color_palette("Set2", num_age_groups)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # X positions for each draw
    x_positions = np.arange(len(_df.columns.levels[0]))

    # Stagger points slightly for each age group
    stagger = np.linspace(-0.2, 0.2, num_age_groups)

    # Loop over each draw (column level 0)
    for i, draw in enumerate(_df.columns.levels[0]):
        mean_values = _df[(draw, 'central')]
        lower_values = _df[(draw, 'lower')]
        upper_values = _df[(draw, 'upper')]

        # Plot each age group with staggered x positions
        for j, age_group in enumerate(age_groups):
            ax.errorbar(
                x=x_positions[i] + stagger[j],
                y=mean_values[age_group],
                yerr=[[mean_values[age_group] - lower_values[age_group]],
                      [upper_values[age_group] - mean_values[age_group]]],
                fmt='o',  # Points for the mean values
                color=age_colors[j],  # Assign color per age group
                capsize=5,
                label=age_group if i == 0 else ""  # Only label age groups once
            )

    # Add grey vertical dashed lines between each draw
    for i in range(1, len(x_positions)):
        ax.axvline(x_positions[i] - 0.5, color='grey', linestyle='--')

    # Add horizontal grey line at y=0
    ax.axhline(0, color='grey', linestyle='-')

    # Customize ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(_df.columns.levels[0], rotation=45, ha='right')
    ax.set_ylim(top=80)

    # Add legend for age groups
    handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', label=age_group)
               for age_group, color in zip(age_groups, age_colors)]
    ax.legend(handles=handles, title="Age Group")

    return fig, ax


pc_py_averted_vs_WASH_results_ordered = pc_py_averted_vs_WASH_results.reindex(columns=order_for_plotting, level=0)

name_of_plot = f'Percentage reduction in person-years infected with Schistosomiasis vs WASH {target_period()}'
fig, ax = plot_averted_points_with_errorbars(pc_py_averted_vs_WASH_results_ordered)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in Person-Years Infected')
ax.set_ylim(-100, 100)
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)


# %% PREVALENCE OF INFECTION BY SPECIES / AGE-GROUP

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
inf = 'H'
prev = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_haematobium",
    custom_generate_series=get_prevalence_infection,
    do_scaling=False,
)
prev.index = prev.index.year
prev.to_csv(results_folder / (f'prevalence_H_ALL_haematobium {target_period()}.csv'))


prevM = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_mansoni",
    custom_generate_series=get_prevalence_infection,
    do_scaling=False,
)
prevM.index = prevM.index.year
prevM.to_csv(results_folder / (f'prevalence_H_ALL_mansoni {target_period()}.csv'))



# %% PREVALENCE OF INFECTION OVERALL (BOTH SPECIES) BY DISTRICT

number_infected = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="number_infected_any_species",
    column="number_infected",
    do_scaling=False,
)

number_in_district = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="number_in_subgroup",
    column="number_alive",
    do_scaling=False,
)


def get_numbers_infected_any_species(_df):
    """Return a DataFrame with one row per year, columns as multi-index (draw, run, district),
    and values as the sum of counts across all age groups for each district in each draw/run for each year."""

    records = []

    # Iterate through the rows (each year)
    for year, row in _df.iterrows():
        for (draw, run), entry in row.items():
            if not entry:  # Skip if the entry is empty
                continue
            if isinstance(entry, dict):  # Ensure the entry is a dictionary
                for composite_key, value in entry.items():
                    split_keys = dict(kv.split("=") for kv in composite_key.split("|"))
                    district = split_keys.get("district_of_residence")
                    if district:  # Ensure district is available
                        records.append({
                            "year": year,
                            "draw": draw,
                            "run": run,
                            "district": district,
                            "count": value
                        })

    # Convert the flattened records into a DataFrame
    long_df = pd.DataFrame(records)

    # Group by (year, draw, run, district) and sum the counts
    grouped = (
        long_df
        .groupby(["year", "draw", "run", "district"])["count"]
        .sum()
        .rename("summed_value")
        .to_frame()
    )

    # Reshape the data so that we have multi-index columns (draw, run, district)
    result = (
        grouped
        .unstack(["draw", "run", "district"])  # Unstack to create the multi-index columns
        .droplevel(0, axis=1)  # Drop the 'number_infected' level
    )

    return result


total_number_infected = get_numbers_infected_any_species(number_infected)
total_number_in_district = get_numbers_infected_any_species(number_in_district)

if total_number_infected.columns.equals(total_number_in_district.columns):
    # Perform element-wise division for matching columns
    result = total_number_infected / total_number_in_district

result.to_csv(results_folder / (f'prevalence_any_infection_all_ages_district{target_period()}.csv'))

# summarise the prevalence for each district by draw
median_by_draw_district = result.groupby(level=['draw', 'district'], axis=1).median()
median_by_draw_district.to_csv(results_folder / (f'median_prevalence_any_infection_all_ages_district{target_period()}.csv'))


# %% GET PZQ USED FOR SCENARIOS -----------------------------------------------------------------------------

# numbers of PZQ doses - these are 1mg doses
# includes MDA and treatment
def get_counts_of_items_requested(_df):
    _df = drop_outside_period(_df)

    counts_of_available = defaultdict(int)
    counts_of_not_available = defaultdict(int)
    counts_of_used = defaultdict(int)

    for _, row in _df.iterrows():
        for item, num in row['Item_Available'].items():
            counts_of_available[item] += num
        for item, num in row['Item_NotAvailable'].items():
            counts_of_not_available[item] += num
        for item, num in row['Item_Used'].items():
            counts_of_used[item] += num

    return pd.concat(
        {'Item_Available': pd.Series(counts_of_available),
         'Not_Available': pd.Series(counts_of_not_available),
         'Item_Used': pd.Series(counts_of_used)},
        axis=1
    ).fillna(0).astype(int).stack()

cons_req = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Consumables',
        custom_generate_series=get_counts_of_items_requested,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

cons = cons_req.unstack()
# item 286 is Praziquantel 600mg_1000_CMST
pzq_use = cons_req.loc['286']

# attach costs to PZQ: 0.0000406606 USD
PZQ_item_cost = 0.0000406606
pzq_cost = pd.DataFrame(pzq_use.iloc[-1] * PZQ_item_cost).T
pzq_cost.index = ['pzq_costs']
pzq_use = pd.concat([pzq_use, pzq_cost])
pzq_use.to_csv(results_folder / (f'pzq_use {target_period()}.csv'))

summary_pzq_cost = compute_summary_statistics(pzq_use)


def plot_simple_barplot(_df, title=None, ylab=None):
    """
    Plot barplot by scenario using the 'central' value for height,
    and 'lower' and 'upper' for the error bars.

    Assumes the DataFrame has a MultiIndex with 'draw' and 'stat' levels,
    where 'stat' includes 'lower', 'central', 'upper'.
    """

    # Extract 'central', 'lower', and 'upper' columns
    df_plot = _df['central']
    lower = _df['lower']
    upper = _df['upper']

    # Compute error bars
    yerr = [df_plot - lower, upper - df_plot]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        df_plot.index,
        df_plot,
        yerr=yerr,
        capsize=5,
        color='skyblue',
        edgecolor='black'
    )

    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xticklabels(df_plot.index, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


# Convert summary_pzq_cost.loc['pzq_costs'] to DataFrame with 'central', 'lower', and 'upper'
pzq_costs2 = summary_pzq_cost.loc['pzq_costs'].unstack(level='stat')
plot_simple_barplot(pzq_costs2, title='PZQ Costs', ylab='PZQ Costs, USD')



# HSIs
def get_total_num_treatment_episdoes(_df):
    """Return total number of treatments within the TARGET_PERIOD."""
    # Ensure 'date' is a datetime column if not already
    _df['date'] = pd.to_datetime(_df['date'])

    # Filter rows based on the TARGET_PERIOD (date range)
    filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]

    # Sum only the numeric columns (exclude 'date' and non-numeric columns)
    y = filtered_df.select_dtypes(include='number').sum(axis=0)

    return y


treatment_episodes = extract_results(
        results_folder,
        module='tlo.methods.schisto',
        key='schisto_treatment_episodes',
        custom_generate_series=get_total_num_treatment_episdoes,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)


pzq_plus_tx_episodes = pd.concat([pzq_use, treatment_episodes, total_num_dalys])

pzq_plus_tx_episodes.to_csv(results_folder / (f'pzq_plus_tx_episodes {target_period()}.csv'))


summary_treatment_episodes = compute_summary_statistics(treatment_episodes,
                                                central_measure='median')
df_reshaped = summary_treatment_episodes.stack(level='draw')
df_reshaped = df_reshaped.reset_index(level=0, drop=True)

plot_simple_barplot(df_reshaped,
                    title='Number Treatment Episodes',
                    ylab='Number treatment episodes')


## ICERS

# need the delta costs
comparison_pzq_costs_vs_WASH = find_difference_relative_to_comparison_series(
        pzq_plus_tx_episodes.loc['pzq_costs'],
        comparison='WASH only'
    )

# get the delta DALYS
num_dalys_averted_vs_WASH = -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='WASH only'
    )

num_dalys_averted_vs_WASH = num_dalys_averted_vs_WASH.T

# Step 1: Align dataset1 (costs) with dataset2 (health outcomes)
aligned_data1 = comparison_pzq_costs_vs_WASH.reindex_like(num_dalys_averted_vs_WASH)  # Align dataset1 to dataset2's index

# Step 2: Compute ICER = health_diff / cost_diff
# Extract values from both Series (dataset1) and DataFrame (dataset2)
cost_diff = aligned_data1.values.flatten()  # Dataset 1 (cost differences)
health_diff = num_dalys_averted_vs_WASH.values.flatten()  # Dataset 2 (health outcomes)

# Compute ICER (health difference / cost difference)
icer = health_diff / cost_diff

# Extract the 'draw' level from the MultiIndex for color coding
draw_labels = aligned_data1.index.get_level_values('draw').values.flatten()  # Get 'draw' labels

# Step 3: Create a dictionary of unique draw labels to colours
unique_draws = np.unique(draw_labels)  # Unique draw labels
colormap = plt.get_cmap('tab20')  # You can choose another colour map if needed
colors = {label: colormap(i / len(unique_draws)) for i, label in enumerate(unique_draws)}

# Step 4: Assign a colour to each data point based on its draw label
point_colors = [colors[label] for label in draw_labels]

# Step 5: Create the scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(cost_diff, health_diff, c=point_colors)  # Colour by draw labels

# Add the legend (position it outside the plot to the right)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10) for label in unique_draws]
plt.legend(handles, unique_draws, title="Scenario", bbox_to_anchor=(1.05, 0.5), loc='center left')

# Labels and title
plt.xlabel('Cost Difference')
plt.ylabel('Health Difference')
plt.title('Incremental Cost-Effectiveness Ratio \n compared to WASH only')
plt.grid(True)

plt.xlim(-4e7, 4e7)  # Extend the x-axis
plt.ylim(-4e6, 4e6)  # Extend the y-axis

# Adding quadrant guidelines (vertical and horizontal lines)
plt.axhline(0, color='black',linewidth=1, linestyle='--')  # Horizontal line at y=0
plt.axvline(0, color='black',linewidth=1, linestyle='--')  # Vertical line at x=0

# Add text annotations for the quadrants
plt.text(1e7, 3.5e6, 'Cost-effective and beneficial', fontsize=12, color='green', ha='left', va='top')
plt.text(-3e7, 3.5e6, 'Cost-effective but harmful', fontsize=12, color='orange', ha='left', va='top')
plt.text(-3e7, -3.5e6, 'Dominated', fontsize=12, color='red', ha='left', va='bottom')
plt.text(1e7, -3.5e6, 'Cost-ineffective and harmful', fontsize=12, color='blue', ha='left', va='bottom')

# Adjust layout to ensure the legend fits outside the plot
plt.tight_layout()
plt.show()



def get_counts_of_hsi_by_treatment_id(_df):
    """Get the counts of full TREATMENT_IDs occurring within the target period."""
    _counts_by_treatment_id = _df \
        .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
        .apply(pd.Series) \
        .sum() \
        .astype(int)

    return _counts_by_treatment_id


counts_of_hsi_by_treatment_id = extract_results(
    results_folder,
    module='tlo.methods.healthsystem.summary',
    key='HSI_Event',
    custom_generate_series=get_counts_of_hsi_by_treatment_id,
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0).fillna(0.0).sort_index()

median_num_hsi_by_treatment_id = compute_summary_statistics(counts_of_hsi_by_treatment_id, central_measure='median')
median_num_hsi_by_treatment_id.to_csv(results_folder / (f'median_num_hsi_by_treatment_id {target_period()}.csv'))



# %% ICERS - comparator is WASH only

# costs for each scenario


# Total DALYs by scenario




# %% todo what do you want by district?
# above is prevalence by district - way to present this - table / figure
# summarise across runs for each scenario
# think about heavy infections only by district
# person-years infected by district
# person-years infected heavy infection only by district
# get schisto DALYS for each scenario
# get costs for each scenario





scaling_factor_district = load_pickled_dataframes(
                results_folder, draw=0, run=0, name='tlo.methods.population'
            )['tlo.methods.population']['scaling_factor_district']['scaling_factor_district'].values[0]

