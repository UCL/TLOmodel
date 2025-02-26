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
from typing import Tuple

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

output_folder = Path("./outputs/t.mangal@imperial.ac.uk")

# results_folder = get_scenario_outputs("schisto_calibration.py", outputpath)[-1]
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
TARGET_PERIOD = (Date(2024, 1, 1), Date(2028, 12, 31))

# figure out data end for each run
hiv_inc = extract_results(
    results_folder,
    module="tlo.methods.hiv",
    key="summary_inc_and_prev_for_adults_and_children_and_fsw",
    column="hiv_adult_inc_1549",
    index="date",
    do_scaling=False
)


# first run fails in 2030 (0/3)
# for now, replace with another run


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

num_dalys_summarized = summarize(total_num_dalys).loc[0].unstack().reindex(param_names)
num_dalys_summarized.to_csv(results_folder / f'total_num_dalys_{target_period()}.csv')


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
total_num_dalys_by_label_summarized = summarize(total_num_dalys_by_label)


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


total_num_dalys_by_label_results_averted_vs_baseline = summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys_by_label,
        comparison='Baseline'
    ),
    only_mean=True
)
total_num_dalys_by_label_results_averted_vs_baseline.to_csv(results_folder / f'total_num_dalys_by_label_results_averted_vs_baseline{target_period()}.csv')

total_num_dalys_by_label_results_averted_vs_WASHonly = summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys_by_label,
        comparison='WASH only'
    ),
    only_mean=True
)
total_num_dalys_by_label_results_averted_vs_WASHonly.to_csv(results_folder / f'total_num_dalys_by_label_results_averted_vs_WASHonly{target_period()}.csv')




pc_dalys_averted = 100.0 * summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys_by_label,
        comparison='Baseline',
        scaled=True
    ),
    only_mean=False
)

pc_dalys_averted_WASHonly = 100.0 * summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys_by_label,
        comparison='WASH only',
        scaled=True
    ),
    only_mean=False
)
pc_dalys_averted_WASHonly.to_csv(results_folder / f'pc_dalys_averted_WASHonly{target_period()}.csv')


# %% PLOTS DALYS RELATIVE TO BASELINE

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
    means = df.xs('mean', level='stat', axis=1)
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


name_of_plot = f'Percentage reduction in DALYs from baseline {target_period()}'
fig, ax = plot_clustered_bars_with_error_bars(pc_dalys_averted)
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

daly_summary = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True,
    ),
    only_mean=True,
    collapse_columns=False,
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
    key="diarrhoea_episodes",
    custom_generate_series=(
        lambda df: pd.Series(
            df.query(f"{TARGET_PERIOD[0].year} <= date.dt.year <= {TARGET_PERIOD[1].year}")["number_episodes"].sum(),
            index=["total"]
        )
    ),
    do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)
diarrhoea.to_csv(results_folder / (f'diarrhoea_incidence {target_period()}.csv'))


diarrhoea_averted_vs_baseline = summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='Baseline'
    ),
    only_mean=True
)

diarrhoea_pc_dalys_averted = 100.0 * summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        diarrhoea,
        comparison='Baseline',
        scaled=True
    ),
    only_mean=False
)


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

alri_averted_vs_baseline = summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='Baseline'
    ),
    only_mean=True
)

alri_pc_dalys_averted = 100.0 * summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        alri,
        comparison='Baseline',
        scaled=True
    ),
    only_mean=False
)

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


hiv_averted_vs_baseline = summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='Baseline'
    ),
    only_mean=True
)

hiv_pc_dalys_averted = 100.0 * summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        hiv,
        comparison='Baseline',
        scaled=True
    ),
    only_mean=False
)

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

bladder_averted_vs_baseline = summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='Baseline'
    ),
    only_mean=True
)

bladder_pc_dalys_averted = 100.0 * summarize(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        bladder,
        comparison='Baseline',
        scaled=True
    ),
    only_mean=False
)

schisto_row = (pc_dalys_averted.loc['Schisto'])
other_row = (pc_dalys_averted.loc['Other'])
combined_df = pd.concat([schisto_row.to_frame().T,
                         diarrhoea_pc_dalys_averted,
                         alri_pc_dalys_averted,
                         hiv_pc_dalys_averted,
                         bladder_pc_dalys_averted,
                         other_row.to_frame().T], ignore_index=True)
combined_df.index = ['Schisto', 'Diarrhoea', 'ALRI', 'HIV', 'Bladder cancer', 'Other']

name_of_plot = f'Percentage reduction in incidence from baseline {target_period()}'
fig, ax = plot_clustered_bars_with_error_bars(combined_df)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in incidence')
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)


# %% GET COSTS FOR SCENARIOS -----------------------------------------------------------------------------

#  numbers of PZQ doses - these are individual tablets
# includes MDA and treatment
def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


def get_counts_of_items_requested(_df):
    _df = drop_outside_period(_df)

    counts_of_available = defaultdict(int)
    counts_of_not_available = defaultdict(int)

    for _, row in _df.iterrows():
        for item, num in row['Item_Available'].items():
            counts_of_available[item] += num
        for item, num in row['Item_NotAvailable'].items():
            counts_of_not_available[item] += num

    return pd.concat(
        {'Item_Available': pd.Series(counts_of_available), 'Not_Available': pd.Series(counts_of_not_available)},
        axis=1
    ).fillna(0).astype(int).stack()

cons_req = summarize(
    extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Consumables',
        custom_generate_series=get_counts_of_items_requested,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0),
    only_mean=True,
    collapse_columns=True
)

cons = cons_req.unstack()
# item 286 is Praziquantel 600mg_1000_CMST
# todo these are showing as all unavailable, ie they've been requested and the HSI has run because they're optional
# so use the quantities requested
pzq_use = cons_req.loc['286']
pzq_use.to_csv(results_folder / (f'pzq_use {target_period()}.csv'))


















# -----------------------------------------------------------------------------

# todo person-years infected with low/moderate/high intensity infections by district and total
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
total_num_py_averted_results = []
pc_py_averted_results = []

for age in ages:
    person_years = extract_results(
        results_folder,
        module="tlo.methods.schisto",
        key="Schisto_person_days_infected",
        custom_generate_series=get_person_years_infected,
        do_scaling=False,  # switch to True for full runs
    ).pipe(set_param_names_as_column_index_level_0)

    person_years_summary = summarize(person_years, only_mean=True)

    total_num_py_averted_vs_baseline = summarize(
        -1.0 * find_difference_relative_to_comparison_dataframe(
            person_years,
            comparison='Baseline'
        ),
        only_mean=True
    )

    pc_py_averted = 100.0 * summarize(
        -1.0 * find_difference_relative_to_comparison_dataframe(
            person_years,
            comparison='Baseline',
            scaled=True
        ),
        only_mean=False
    )

    # Append the results to the corresponding lists
    total_num_py_averted_results.append(total_num_py_averted_vs_baseline)
    pc_py_averted_results.append(pc_py_averted)

# Combine results into two DataFrames, with age groups as a single-level row index
total_num_py_averted_df = pd.concat(total_num_py_averted_results, keys=ages, axis=0)
pc_py_averted_df = pd.concat(pc_py_averted_results, keys=ages, axis=0)

total_num_py_averted_df.index = total_num_py_averted_df.index.get_level_values(0)
pc_py_averted_df.index = pc_py_averted_df.index.get_level_values(0)


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
        mean_values = _df[(draw, 'mean')]
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
    ax.set_ylim(top=50)

    # Add legend for age groups
    handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', label=age_group)
               for age_group, color in zip(age_groups, age_colors)]
    ax.legend(handles=handles, title="Age Group")

    return fig, ax


name_of_plot = f'Percentage reduction in person-years infected with Schistosomiasis from baseline {target_period()}'
fig, ax = plot_averted_points_with_errorbars(pc_py_averted_df)
ax.set_title(name_of_plot)
ax.set_ylabel('Percentage reduction in Person-Years')
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
fig.show()
plt.close(fig)


# todo table, rows=districts, columns=diff in PY for each scenario 2035
# both species combined, all ages
# column level0 with and without WASH:
# classify districts into low/moderate/high burden
# columns=[HML burden, person-years of low/moderate/high infection, # PZQ tablets]


# todo elimination
# years to reach elimination as PH problem
# -- Elimination as a PH problem is defined as reducing the prevalence of heavy infections to less than 1% of population
# years to reach elimination of transmission
# years to reach morbidity control (heavy infections below threshold)

# -- morbidity control is reducing the prevalence of heavy infections to below 5% of the population
# -- (e.g. ≥400 EPG for mansoni or ≥50 eggs/10 mL urine for haematobium).
def get_prevalence_heavy_infection(_df):
    """Get the prevalence every year of the simulation """

    # select the last entry for each year
    _df.set_index('date', inplace=True)
    df = _df.resample('Y').last()

    # df = df.filter(like='Likoma')

    # limit to SAC
    if age == 'SAC':
        df = df.filter(like='SAC')
    if age == 'adult':
        df = df.filter(like='Adult')

    # Aggregate the sums of infection statuses by district_of_residence and year
    district_sum = df.sum(axis=1)

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

    infected = df_filtered.sum(axis=1)

    prop_infected = infected.div(district_sum)

    return prop_infected


age = 'SAC'  # SAC, adult, all
inf = 'HM'
prev = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_haematobium",
    custom_generate_series=get_prevalence_heavy_infection,
    do_scaling=False,
)
prev.index = prev.index.year
