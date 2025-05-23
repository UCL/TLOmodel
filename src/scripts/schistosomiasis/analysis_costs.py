"""
Use the outputs from scenario_runs.py to produce plots
and summary statistics for paper.

JOB ID:

PosixPath('outputs/t.mangal@imperial.ac.uk/schisto_scenarios-2025-04-25T130018Z')

"""

# todo should use district scaling factor so everything is weighted correctly


# ==============================================================================
# 📦 IMPORTS
# ==============================================================================

from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import matplotlib.colors as colors
from collections import defaultdict
import textwrap
from typing import Tuple, Optional, List

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd, make_age_grp_types, parse_log_file, compare_number_of_deaths,
    extract_params, compute_summary_statistics, extract_results, get_scenario_info,
    get_scenario_outputs, load_pickled_dataframes, make_age_grp_lookup,
    unflatten_flattened_multi_index_in_logging
)

# from scripts.costing.cost_estimation import (
#     estimate_input_cost_of_scenarios, summarize_cost_data,
#     do_stacked_bar_plot_of_cost_by_category, do_line_plot_of_cost,
#     create_summary_treemap_by_cost_subgroup, estimate_projected_health_spending
# )


# ==============================================================================
# 📁 FILE PATHS AND PARAMETERS
# ==============================================================================

resourcefilepath = Path("./resources")
output_folder = Path("./outputs/t.mangal@imperial.ac.uk")
results_folder = get_scenario_outputs("schisto_scenarios.py", output_folder)[-1]


# Output graph file generator
def make_graph_file_name(name):
    return results_folder / f"Schisto_{name}.png"


species = ('mansoni', 'haematobium')

# Scenario and parameters info
log = load_pickled_dataframes(results_folder)
scenario_info = get_scenario_info(results_folder)
params = extract_params(results_folder)

# Simulation time period
TARGET_PERIOD = (Date(2024, 1, 1), Date(2040, 12, 31))


# ==============================================================================
# %% 🛠️ UTILITY FUNCTIONS
# ==============================================================================

def get_parameter_names_from_scenario_file() -> Tuple[str]:
    """Return tuple of scenario names from the scenario file."""
    from scripts.schistosomiasis.scenario_runs import SchistoScenarios
    return tuple(SchistoScenarios()._scenarios.keys())


param_names = get_parameter_names_from_scenario_file()


def target_period() -> str:
    """Return target period as a string of the form 'YYYY-YYYY'."""
    return "-".join(str(t.year) for t in TARGET_PERIOD)


def drop_outside_period(_df):
    """Return a dataframe restricted to dates within the TARGET_PERIOD."""
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


# ==============================================================================
# %% 📊 DATA EXTRACTION FUNCTIONS - HEALTH
# ==============================================================================

# todo get dalys by cause and year (for discounting)

dalys_by_year_and_cause = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
    custom_generate_series=(
        lambda df_: df_.drop(
            columns=(['date', 'sex', 'age_range']),
        ).groupby(['year']).sum().stack()
    ),
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0)

dalys_by_year_and_cause.index = dalys_by_year_and_cause.index.set_names('label', level=1)

# this gives schisto dalys for each run, by year
schisto_dalys_by_year = dalys_by_year_and_cause.xs('Schistosomiasis', level='label')


def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'year']) \
        .sum()


# sum DALYs by cause over target period
num_dalys_by_cause = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)

total_schisto_dalys = num_dalys_by_cause.loc[num_dalys_by_cause.index == 'Schistosomiasis']

summary_total_dalys = compute_summary_statistics(num_dalys_by_cause, central_measure='mean')

# ==============================================================================
# %% 📊 DATA EXTRACTION FUNCTIONS - HEALTH SYSTEM
# ==============================================================================

def get_counts_of_items_requested(_df):
    """
    Return counts of requested items over the period — Available, NotAvailable, Used.
    Output as pd.Series with stacked labels.
    """
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

    return pd.concat({
        'Item_Available': pd.Series(counts_of_available),
        'Not_Available': pd.Series(counts_of_not_available),
        'Item_Used': pd.Series(counts_of_used)
    }, axis=1).fillna(0).astype(int).stack()


def get_counts_of_cons_by_year(_df):
    """
    Return annual total of item '286' usage.
    'Item_Used' is a dictionary of {item: count}.
    """
    # _df = drop_outside_period(_df)
    _df['Year'] = pd.to_datetime(_df['date']).dt.year

    def sum_item_counts(series_of_dicts):
        # Convert each dict into a Series and sum, skipping NaNs
        df_items = series_of_dicts.dropna().apply(pd.Series)
        summed = df_items.sum()
        return summed.get('286', 0)  # safely retrieve item '286'

    counts_by_year = _df.groupby('Year')['Item_Used'].apply(sum_item_counts)

    return counts_by_year.astype(int)


def get_total_num_treatment_episodes(_df):
    """Return total number of treatments within the TARGET_PERIOD."""
    # Ensure 'date' is a datetime column if not already
    _df['date'] = pd.to_datetime(_df['date'])

    # Filter rows based on the TARGET_PERIOD (date range)
    filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]

    # Sum only the numeric columns (exclude 'date' and non-numeric columns)
    y = filtered_df.select_dtypes(include='number').sum(axis=0)

    return y


# ==============================================================================
# %% ✅ ECONOMIC FUNCTIONS
# ==============================================================================


def compute_icer(
    dalys_averted: pd.DataFrame,
    comparison_costs: pd.DataFrame,
    discount_rate_dalys: float = 0.0,
    discount_rate_costs: float = 0.0,
    return_summary: bool = True
) -> pd.DataFrame | pd.Series:
    """
    Compute ICERs comparing costs and DALYs averted over a TARGET_PERIOD by run and draw.
    """
    global TARGET_PERIOD
    start_year, end_year = TARGET_PERIOD

    # Restrict index to target period (years)
    mask = (dalys_averted.index >= start_year.year) & (dalys_averted.index <= end_year.year)

    # Extract the target period data
    dalys_period = dalys_averted.loc[mask]
    costs_period = comparison_costs.loc[mask]

    # Calculate years since start for discounting
    years_since_start = dalys_period.index.values - start_year.year

    # Discount factors
    discount_weights_dalys = 1 / ((1 + discount_rate_dalys) ** years_since_start)
    discount_weights_costs = 1 / ((1 + discount_rate_costs) ** years_since_start)

    # Apply discounting only to target period data
    if discount_rate_dalys != 0.0:
        dalys_period = dalys_period.multiply(discount_weights_dalys, axis=0)

    if discount_rate_costs != 0.0:
        costs_period = costs_period.multiply(discount_weights_costs, axis=0)

    # Sum discounted DALYs and costs over target period (rows)
    total_dalys = dalys_period.sum(axis=0)  # indexed by (run, draw)
    total_costs = costs_period.sum(axis=0)  # indexed by (run, draw)

    # Compute ICER per (run, draw) pair
    icers = total_costs / total_dalys

    # Prepare DataFrame for output
    icers_df = icers.reset_index()
    icers_df.columns = ['run', 'draw', 'icer']

    if return_summary:
        # Group by draw and summarise ICER over runs
        summary = (
            icers_df
            .groupby('draw')['icer']
            .agg(
                mean='mean',
                lower=lambda x: np.quantile(x, 0.025),
                upper=lambda x: np.quantile(x, 0.975)
            )
        )
        return summary
    else:
        # Return all ICERs for every run and draw
        return icers_df


def compute_nhb(
    dalys_averted: pd.DataFrame,
    comparison_costs: pd.DataFrame,
    discount_rate_dalys: float = 0.03,
    discount_rate_costs: float = 0.03,
    threshold: float = 150,
    return_summary: bool = True
) -> pd.DataFrame | pd.Series:
    """
    Compute Net Health Benefit (NHB) using DALYs averted and incremental costs over TARGET_PERIOD.

    Parameters:
    - dalys_averted: pd.DataFrame with index as years, columns as MultiIndex (run, draw), values = DALYs averted
    - comparison_costs: pd.DataFrame with same structure, values = incremental costs
    - discount_rate_dalys: annual discount rate for DALYs
    - discount_rate_costs: annual discount rate for costs
    - threshold: cost-effectiveness threshold (e.g. $ per DALY averted)
    - return_summary: if True, summarise across runs for each draw

    Returns:
    - pd.Series (summary) or pd.DataFrame with NHB values
    """
    global TARGET_PERIOD
    start_year, end_year = TARGET_PERIOD

    # Restrict to target period
    mask = (dalys_averted.index >= start_year.year) & (dalys_averted.index <= end_year.year)
    dalys_period = dalys_averted.loc[mask]
    costs_period = comparison_costs.loc[mask]

    # Years since start for discounting
    years_since_start = dalys_period.index.values - start_year.year

    # Discount factors
    discount_weights_dalys = 1 / ((1 + discount_rate_dalys) ** years_since_start)
    discount_weights_costs = 1 / ((1 + discount_rate_costs) ** years_since_start)

    # Apply discounting
    if discount_rate_dalys != 0.0:
        dalys_period = dalys_period.multiply(discount_weights_dalys, axis=0)

    if discount_rate_costs != 0.0:
        costs_period = costs_period.multiply(discount_weights_costs, axis=0)

    # Total discounted DALYs and costs per (run, draw)
    total_dalys_averted = dalys_period.sum(axis=0)
    total_costs = costs_period.sum(axis=0)

    # NHB
    nhb = total_dalys_averted - (total_costs / threshold)
    if not isinstance(nhb.index, pd.MultiIndex):
        nhb.index = pd.MultiIndex.from_tuples(nhb.index, names=['run', 'draw'])

    nhb_df = nhb.reset_index()
    nhb_df.columns = ['run', 'draw', 'nhb']

    if return_summary:
        summary = (
            nhb_df
            .groupby('draw')['nhb']
            .agg(mean='mean', lower=lambda x: np.quantile(x, 0.025), upper=lambda x: np.quantile(x, 0.975))
        )
        return summary
    else:
        return nhb_df


# ==============================================================================
# %% ✅ PLOT FUNCTIONS
# ==============================================================================


def plot_npv_scatter(npv_df, column, yaxis_label, title):
    """
    Plot the Net Present Value (NPV) of interventions for each draw and scenario, showing the scatter of NPVs across draws.

    Parameters
    ----------
    npv_df : pd.DataFrame
        DataFrame containing the NPV of the intervention for each draw and scenario.
        It should have a multi-index (run, draw) and the 'NPV_Intervention' column.
    """
    # Reset index to make 'run' and 'draw' columns for easier plotting
    npv_df_reset = npv_df.reset_index()

    # Set plot style
    sns.set(style="whitegrid")

    # Create a scatter plot for the NPV of each draw and scenario
    plt.figure(figsize=(12, 6))
    scatter_plot = sns.scatterplot(x='draw', y=column, hue='draw', data=npv_df_reset, palette='viridis')

    # Adding labels and title
    plt.xlabel('', fontsize=14)
    plt.ylabel(yaxis_label, fontsize=14)
    plt.title(title, fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    scatter_plot.legend(title=None)

    # Show the plot
    plt.tight_layout()
    plt.show()



def plot_icers_with_nhb_isocurves(
    dalys_averted: pd.DataFrame,
    incremental_costs: pd.DataFrame,
    discount_rate_dalys: float = 0.03,
    discount_rate_costs: float = 0.03,
    summary: bool = True,
    lambda_values: Optional[List[float]] = None,
    isocurves=False,
):
    """
    Plot ICERs with Net Health Benefit (NHB) isocurves using Malawi λ values,
    adding one label per draw near the cluster of its points.
    """
    global TARGET_PERIOD
    start_year, end_year = TARGET_PERIOD

    # Filter to target period
    mask = (dalys_averted.index >= start_year.year) & (dalys_averted.index <= end_year.year)
    dalys_period = dalys_averted.loc[mask]
    costs_period = incremental_costs.loc[mask]

    # Calculate years since start year for discounting
    years_since_start = dalys_period.index.values - start_year.year

    # Discount factors
    if discount_rate_dalys != 0:
        discount_factors_dalys = 1 / ((1 + discount_rate_dalys) ** years_since_start)
        dalys_disc = dalys_period.multiply(discount_factors_dalys, axis=0)
    else:
        dalys_disc = dalys_period

    if discount_rate_costs != 0:
        discount_factors_costs = 1 / ((1 + discount_rate_costs) ** years_since_start)
        costs_disc = costs_period.multiply(discount_factors_costs, axis=0)
    else:
        costs_disc = costs_period

    # Sum discounted DALYs and costs over years
    dalys_sum = dalys_disc.sum(axis=0)
    costs_sum = costs_disc.sum(axis=0)

    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        'dalys_averted': dalys_sum,
        'incremental_costs': costs_sum,
    }).reset_index()

    # Default Malawi λ values if not provided
    if lambda_values is None:
        lambda_values = [150, 300, 600]

    # Use a punchier colour palette
    draws = df['draw'].unique()
    palette = sns.color_palette("bright", n_colors=len(draws))
    color_map = dict(zip(draws, palette))

    plt.figure(figsize=(10, 8))

    if summary:
        summary_list = []
        for draw in draws:
            subset = df[df['draw'] == draw]
            mean_dalys = subset['dalys_averted'].mean()
            se_dalys = subset['dalys_averted'].std() / np.sqrt(len(subset))
            ci_dalys = 1.96 * se_dalys

            mean_costs = subset['incremental_costs'].mean()
            se_costs = subset['incremental_costs'].std() / np.sqrt(len(subset))
            ci_costs = 1.96 * se_costs

            summary_list.append({
                'draw': draw,
                'mean_dalys_averted': mean_dalys,
                'ci_dalys': ci_dalys,
                'mean_incremental_costs': mean_costs,
                'ci_costs': ci_costs
            })

        summary_df = pd.DataFrame(summary_list)

        for _, row in summary_df.iterrows():
            plt.errorbar(
                row['mean_incremental_costs'], row['mean_dalys_averted'],
                xerr=row['ci_costs'], yerr=row['ci_dalys'],
                fmt='o', capsize=4,
                color=color_map[row['draw']],
                label=str(row['draw']),
                markeredgewidth=0
            )

        # Add one label per draw at the mean point
        # for _, row in summary_df.iterrows():
        #     plt.text(
        #         row['mean_incremental_costs'], row['mean_dalys_averted'],
        #         f"{row['draw']}",
        #         fontsize=10,
        #         color=color_map[row['draw']],
        #         weight='bold',
        #         horizontalalignment='left',
        #         verticalalignment='bottom',
        #         alpha=0.9
        #     )

    else:
        for draw in draws:
            subset = df[df['draw'] == draw]
            plt.scatter(
                subset['incremental_costs'], subset['dalys_averted'],
                label=str(draw),
                color=color_map[draw],
                alpha=0.7,
                edgecolors='none',
                s=50
            )

            # Add one label per draw at mean coordinates
            # mean_x = subset['incremental_costs'].mean()
            # mean_y = subset['dalys_averted'].mean()
            # plt.text(
            #     mean_x, mean_y,
            #     f"{draw}",
            #     fontsize=10,
            #     color=color_map[draw],
            #     weight='bold',
            #     horizontalalignment='left',
            #     verticalalignment='bottom',
            #     alpha=0.9
            # )

    # Plot NHB isocurves without legend entry
    xlims = plt.xlim()
    xmin, xmax = max(0, xlims[0]), xlims[1]
    xvals = np.linspace(xmin, xmax, 200)

    if isocurves:
        for lam in lambda_values:
            yvals = xvals / lam
            plt.plot(xvals, yvals, linestyle='--', color='grey', alpha=0.8)

            label_x = xmax * 0.95
            label_y = label_x / lam
            plt.text(label_x, label_y, f'λ = {lam}', fontsize=10, color='grey',
                     verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel('Incremental Costs')
    plt.ylabel('DALYs Averted')
    plt.title('ICERs')
    plt.grid(True)
    plt.legend(title='Draw', loc='best')
    plt.tight_layout()

    plt.show()




# ==============================================================================
# %% ✅ EXTRACTED RESULTS (HEALTH SYSTEM)
# ==============================================================================

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

# todo item 1735 is Praziquantel, 600 mg (donated)

# attach costs to PZQ: 0.0000406606 USD
PZQ_item_cost = 0.0000406606
pzq_cost = pd.DataFrame(pzq_use.iloc[-1] * PZQ_item_cost).T
pzq_cost.index = ['pzq_costs']
pzq_use = pd.concat([pzq_use, pzq_cost])
pzq_use.to_csv(results_folder / (f'pzq_use {target_period()}.csv'))

summary_pzq_cost = compute_summary_statistics(pzq_use)


pzq_use_vs_PauseWASH = compute_summary_statistics(
    find_difference_relative_to_comparison_dataframe(
        pzq_cost,
        comparison='Pause WASH, no MDA'
    ),
    central_measure='mean'
)
pzq_use_vs_PauseWASH.to_csv(results_folder / f'pzq_use_vs_PauseWASH{target_period()}.csv')

pzq_use_vs_ContinueWASH = compute_summary_statistics(
    find_difference_relative_to_comparison_dataframe(
        pzq_cost,
        comparison='Continue WASH, no MDA'
    ),
    central_measure='mean'
)
pzq_use_vs_ContinueWASH.to_csv(results_folder / f'pzq_use_vs_ContinueWASH{target_period()}.csv')

pzq_use_vs_scaleupWASH = compute_summary_statistics(
    find_difference_relative_to_comparison_dataframe(
        pzq_cost,
        comparison='Scale-up WASH, no MDA'
    ),
    central_measure='mean'
)
pzq_use_vs_scaleupWASH.to_csv(results_folder / f'pzq_use_vs_scaleupWASH{target_period()}.csv')



mda_episodes = extract_results(
        results_folder,
        module='tlo.methods.schisto',
        key='schisto_mda_episodes',
        custom_generate_series=get_total_num_treatment_episodes,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)


summary_mda_episodes = compute_summary_statistics(mda_episodes,
                                                central_measure='median')
summary_mda_episodes = summary_mda_episodes.stack(level='draw')
summary_mda_episodes = summary_mda_episodes.reset_index(level=0, drop=True)
summary_mda_episodes.to_csv(results_folder / f'summary_mda_episodes{target_period()}.csv')


# UNIT COSTS PER RUN
unit_cost_per_mda = 2.26 - 0.05  # full cost - consumables

mda_episodes_cost_excl_cons = mda_episodes * unit_cost_per_mda

summary_mda_episodes_cost_excl_cons = compute_summary_statistics(mda_episodes_cost_excl_cons,
                                                central_measure='mean')
summary_mda_episodes_cost_excl_cons = summary_mda_episodes_cost_excl_cons.stack(level='draw')
summary_mda_episodes_cost_excl_cons = summary_mda_episodes_cost_excl_cons.reset_index(level=0, drop=True)
summary_mda_episodes_cost_excl_cons.to_csv(results_folder / f'summary_mda_episodes_cost_excl_cons{target_period()}.csv')


# calculate the costs incurred from non-cons costs, i.e. HRH, implementation etc
costs_excl_cons_vs_PauseWASH = compute_summary_statistics(
    find_difference_relative_to_comparison_dataframe(
        mda_episodes_cost_excl_cons,
        comparison='Pause WASH, no MDA'
    ),
    central_measure='mean'
)
costs_excl_cons_vs_PauseWASH.to_csv(results_folder / f'costs_excl_cons_vs_PauseWASH{target_period()}.csv')

costs_excl_cons_vs_ContinueWASH = compute_summary_statistics(
    find_difference_relative_to_comparison_dataframe(
        mda_episodes_cost_excl_cons,
        comparison='Continue WASH, no MDA'
    ),
    central_measure='mean'
)
costs_excl_cons_vs_ContinueWASH.to_csv(results_folder / f'costs_excl_cons_vs_ContinueWASH{target_period()}.csv')

costs_excl_cons_vs_scaleupWASH = compute_summary_statistics(
    find_difference_relative_to_comparison_dataframe(
        mda_episodes_cost_excl_cons,
        comparison='Scale-up WASH, no MDA'
    ),
    central_measure='mean'
)
costs_excl_cons_vs_scaleupWASH.to_csv(results_folder / f'costs_excl_cons_vs_scaleupWASH{target_period()}.csv')



# ==============================================================================
# %% ✅ GET OUTPUTS BY RUN FOR ICER / NHB CALCULATION
# ==============================================================================

######################
# SCHISTO DALYS
######################
num_dalys_by_year_run = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
    custom_generate_series=(
        lambda df_: df_.drop(
            columns=(['date', 'sex', 'age_range']),
        ).groupby(['year']).sum().stack()
    ),
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0)

schisto_dalys_by_year_run = num_dalys_by_year_run.loc[
    num_dalys_by_year_run.index.get_level_values(1) == 'Schistosomiasis'
].droplevel(1)

schisto_dalys_averted_by_year_run_vs_pause = -1.0 * find_difference_relative_to_comparison_dataframe(
    schisto_dalys_by_year_run,
    comparison='Pause WASH, no MDA'
)
schisto_dalys_averted_by_year_run_vs_continue = -1.0 * find_difference_relative_to_comparison_dataframe(
    schisto_dalys_by_year_run,
    comparison='Continue WASH, no MDA'
)
schisto_dalys_averted_by_year_run_vs_scaleup = -1.0 * find_difference_relative_to_comparison_dataframe(
    schisto_dalys_by_year_run,
    comparison='Scale-up WASH, no MDA'
)

# produce dataframe with the 3 comparators combined into one
def select_draws_by_keyword(df, keyword):
    """Select columns where the first level of the column MultiIndex contains the keyword (case-insensitive)."""
    mask = df.columns.get_level_values(1).str.contains(keyword, case=False)
    return df.loc[:, mask]

# Select desired columns from each dataframe
df1_sel = select_draws_by_keyword(schisto_dalys_averted_by_year_run_vs_pause, 'Pause')
df2_sel = select_draws_by_keyword(schisto_dalys_averted_by_year_run_vs_continue, 'Continue')
df3_sel = select_draws_by_keyword(schisto_dalys_averted_by_year_run_vs_scaleup, 'Scale-up')

# Concatenate the selected columns horizontally
schisto_dalys_averted_by_year_run_combined = pd.concat([df1_sel, df2_sel, df3_sel], axis=1)
schisto_dalys_averted_by_year_run_combined.to_csv(results_folder / f'schisto_dalys_averted_by_year_run_combined{target_period()}.csv')


#################
# add disability for light infections



# def get_low_intensity_person_years_infected(_df):
#     """Get the person-years for each draw, summed over every district """
#
#     df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]
#
#     df_filtered = df.filter(regex='(Low-infection)')
#
#     person_years = df_filtered.sum(axis=1).sum() / 365.25
#
#     return pd.Series(person_years)


def get_person_years_infected_by_year(_df: pd.DataFrame) -> pd.Series:
    """
    Get person-years infected per year, summed over all districts, age groups, and species,
    for infection level = Low-infection.
    """
    # Filter to target period
    # df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)].copy()
    df = _df
    # Extract year and drop original date column
    df['year'] = pd.to_datetime(df['date']).dt.year
    df.drop(columns='date', inplace=True)

    # Filter columns for infection_level=Low-infection only
    df_filtered = df.filter(regex=r'infection_level=Low-infection')

    # Convert columns to MultiIndex
    columns_split = df_filtered.columns.str.extract(
        r'species=([^|]+)\|age_group=([^|]+)\|infection_level=([^|]+)\|district=([^|]+)'
    )
    df_filtered.columns = pd.MultiIndex.from_frame(columns_split, names=['species', 'age_group', 'infection_level', 'district'])

    # Group by year and sum over all dimensions
    py_by_year = (
        df_filtered
        .groupby(df['year'])
        .sum()
        .sum(axis=1)  # sum over species, age group, infection level, district
        / 365.25      # convert person-days to person-years
    )

    return py_by_year


person_years_light_infection = extract_results(
            results_folder,
            module="tlo.methods.schisto",
            key="Schisto_person_days_infected",
            custom_generate_series=get_person_years_infected_by_year,
            do_scaling=True,
        ).pipe(set_param_names_as_column_index_level_0)


# convert to years lived with disability
disability_weight_light_inf = person_years_light_infection * 0.01  # (from updated GBD)

disability_weight_light_inf_averted_by_year_run_vs_pause = -1.0 * find_difference_relative_to_comparison_dataframe(
    disability_weight_light_inf,
    comparison='Pause WASH, no MDA'
)
disability_weight_light_inf_averted_by_year_run_vs_continue = -1.0 * find_difference_relative_to_comparison_dataframe(
    disability_weight_light_inf,
    comparison='Continue WASH, no MDA'
)
disability_weight_light_inf_averted_by_year_run_vs_scaleup = -1.0 * find_difference_relative_to_comparison_dataframe(
    disability_weight_light_inf,
    comparison='Scale-up WASH, no MDA'
)

# Select desired columns from each dataframe
df1_sel = select_draws_by_keyword(disability_weight_light_inf_averted_by_year_run_vs_pause, 'Pause')
df2_sel = select_draws_by_keyword(disability_weight_light_inf_averted_by_year_run_vs_continue, 'Continue')
df3_sel = select_draws_by_keyword(disability_weight_light_inf_averted_by_year_run_vs_scaleup, 'Scale-up')

# Concatenate the selected columns horizontally
disability_weight_light_inf_averted_by_year_run_combined = pd.concat([df1_sel, df2_sel, df3_sel], axis=1)
disability_weight_light_inf_averted_by_year_run_combined.to_csv(results_folder / f'disability_weight_light_inf_averted_by_year_run_combined{target_period()}.csv')



######################
# COSTS PER RUN
######################
# - only item 286 here

pzq_cons_req_by_year = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Consumables',
        custom_generate_series=get_counts_of_cons_by_year,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

pzq_costs_req_by_year = pzq_cons_req_by_year * PZQ_item_cost


mda_episodes_per_year = extract_results(
        results_folder,
        module='tlo.methods.schisto',
        key='schisto_mda_episodes',
        column='mda_episodes',
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).set_index(pzq_costs_req_by_year.index)


# unit costs applied per mda episode
unit_cost_per_year = mda_episodes_per_year * unit_cost_per_mda
total_cost_per_year = pzq_costs_req_by_year + unit_cost_per_year

# COMPARISON
costs_incurred_by_year_run_vs_pause = find_difference_relative_to_comparison_dataframe(
    total_cost_per_year,
    comparison='Pause WASH, no MDA'
)
costs_incurred_by_year_run_vs_continue = find_difference_relative_to_comparison_dataframe(
    total_cost_per_year,
    comparison='Continue WASH, no MDA'
)
costs_incurred_by_year_run_vs_scaleup = find_difference_relative_to_comparison_dataframe(
    total_cost_per_year,
    comparison='Scale-up WASH, no MDA'
)


# Select desired columns from each dataframe
df1_sel = select_draws_by_keyword(costs_incurred_by_year_run_vs_pause, 'Pause')
df2_sel = select_draws_by_keyword(costs_incurred_by_year_run_vs_continue, 'Continue')
df3_sel = select_draws_by_keyword(costs_incurred_by_year_run_vs_scaleup, 'Scale-up')

# Concatenate the selected columns horizontally
costs_incurred_by_year_run_combined = pd.concat([df1_sel, df2_sel, df3_sel], axis=1)
costs_incurred_by_year_run_combined.to_csv(results_folder / f'costs_incurred_by_year_run_combined{target_period()}.csv')


## just PZQ costs only for ICER

pzq_costs_incurred_by_year_run_vs_pause = find_difference_relative_to_comparison_dataframe(
    pzq_costs_req_by_year,
    comparison='Pause WASH, no MDA'
)
pzq_costs_incurred_by_year_run_vs_continue = find_difference_relative_to_comparison_dataframe(
    pzq_costs_req_by_year,
    comparison='Continue WASH, no MDA'
)
pzq_costs_incurred_by_year_run_vs_scaleup = find_difference_relative_to_comparison_dataframe(
    pzq_costs_req_by_year,
    comparison='Scale-up WASH, no MDA'
)


# Select desired columns from each dataframe
df1_sel = select_draws_by_keyword(pzq_costs_incurred_by_year_run_vs_pause, 'Pause')
df2_sel = select_draws_by_keyword(pzq_costs_incurred_by_year_run_vs_continue, 'Continue')
df3_sel = select_draws_by_keyword(pzq_costs_incurred_by_year_run_vs_scaleup, 'Scale-up')

# Concatenate the selected columns horizontally
pzq_costs_incurred_by_year_run_combined = pd.concat([df1_sel, df2_sel, df3_sel], axis=1)
pzq_costs_incurred_by_year_run_combined.to_csv(results_folder / f'pzq_costs_incurred_by_year_run_combined{target_period()}.csv')


# ==============================================================================
# 📊 CALCULATE ICERS AND NHB
# ==============================================================================
# if adding light infection disability weight:
# todo
dalys_averted_including_light_inf = disability_weight_light_inf_averted_by_year_run_combined + schisto_dalys_averted_by_year_run_combined.iloc[:-1]



icer_no_discount_summary = compute_icer(dalys_averted=schisto_dalys_averted_by_year_run_combined.iloc[:-1],
    comparison_costs=costs_incurred_by_year_run_combined,
    discount_rate_dalys=0,  # no discounting
    discount_rate_costs=0,
    return_summary=True)

icer_no_discount_summary.to_csv(results_folder / (f'icer_no_discount_summary{target_period()}.csv'))


icer_no_discount_CONS_ONLY_summary = compute_icer(dalys_averted=schisto_dalys_averted_by_year_run_combined.iloc[:-1],
    comparison_costs=pzq_costs_incurred_by_year_run_combined,
    discount_rate_dalys=0,  # no discounting
    discount_rate_costs=0,
    return_summary=True)

icer_no_discount_CONS_ONLY_summary.to_csv(results_folder / (f'icer_no_discount_CONS_ONLY_summary{target_period()}.csv'))


icer_no_discount_LIGHT_INF_summary = compute_icer(dalys_averted=dalys_averted_including_light_inf,
    comparison_costs=costs_incurred_by_year_run_combined,
    discount_rate_dalys=0,  # no discounting
    discount_rate_costs=0,
    return_summary=True)

icer_no_discount_LIGHT_INF_summary.to_csv(results_folder / (f'icer_no_discount_LIGHT_INF_summary{target_period()}.csv'))



icer_no_discount_by_run = compute_icer(dalys_averted=schisto_dalys_averted_by_year_run_combined.iloc[:-1],
    comparison_costs=costs_incurred_by_year_run_combined,
    discount_rate_dalys=0,  # no discounting
    discount_rate_costs=0,
    return_summary=False)

icer_no_discount_by_run.to_csv(results_folder / (f'icer_no_discount_by_run{target_period()}.csv'))


icer_discount_costs_summary = compute_icer(dalys_averted=schisto_dalys_averted_by_year_run_combined.iloc[:-1],
    comparison_costs=costs_incurred_by_year_run_combined,
    discount_rate_dalys=0,  # no discounting
    discount_rate_costs=0.03,
    return_summary=True)

icer_discount_costs_summary.to_csv(results_folder / (f'icer_discount_costs_summary{target_period()}.csv'))

icer_discount_costs_by_run = compute_icer(dalys_averted=schisto_dalys_averted_by_year_run_combined.iloc[:-1],
    comparison_costs=costs_incurred_by_year_run_combined,
    discount_rate_dalys=0,  # no discounting
    discount_rate_costs=0.03,
    return_summary=False)

icer_discount_costs_by_run.to_csv(results_folder / (f'icer_discount_costs_by_run{target_period()}.csv'))


nhb_no_discount_150 = compute_nhb(
    dalys_averted=schisto_dalys_averted_by_year_run_combined.iloc[:-1],
    comparison_costs=costs_incurred_by_year_run_combined,
    discount_rate_dalys=0,
    discount_rate_costs=0,
    threshold=150,
    return_summary=True
)
nhb_no_discount_150.to_csv(results_folder / (f'nhb_no_discount_150_{target_period()}.csv'))

nhb_no_discount_150_CONS_ONLY = compute_nhb(
    dalys_averted=schisto_dalys_averted_by_year_run_combined.iloc[:-1],
    comparison_costs=pzq_costs_incurred_by_year_run_combined,
    discount_rate_dalys=0,
    discount_rate_costs=0,
    threshold=150,
    return_summary=True
)
nhb_no_discount_150_CONS_ONLY.to_csv(results_folder / (f'nhb_no_discount_150_CONS_ONLY{target_period()}.csv'))


nhb_no_discount_150_LIGHT_INF = compute_nhb(
    dalys_averted=dalys_averted_including_light_inf,
    comparison_costs=costs_incurred_by_year_run_combined,
    discount_rate_dalys=0,
    discount_rate_costs=0,
    threshold=150,
    return_summary=True
)
nhb_no_discount_150_LIGHT_INF.to_csv(results_folder / (f'nhb_no_discount_150_LIGHT_INF_{target_period()}.csv'))



# ==============================================================================
# 📊 GENERATE FIGURES
# ==============================================================================


lambda_values = [500, 1000, 1500]  # willingness-to-pay per DALY averted
# with low WTP, isocurve goes super high and data points appear all squashed

plot_icers_with_nhb_isocurves(
    dalys_averted=schisto_dalys_averted_by_year_run_combined.iloc[:-1],  # Indexed by year; columns multi-index (run, draw)
    incremental_costs=costs_incurred_by_year_run_combined,
    discount_rate_dalys=0,
    discount_rate_costs=0,
    summary=True,
    lambda_values=lambda_values,
    isocurves=False
)




plot_npv_scatter(npv_results, column='NPV_Intervention',
                 yaxis_label='Net Present Value of Intervention',
                 title='Net Present Value Compared with WASH only')

plot_npv_scatter(npv_results, column='Cost_per_DALY_Averted',
                 yaxis_label='Cost per DALY averted (discounted',
                 title='Cost per DALY averted (discounted) compared with WASH only')



# ==============================================================================
# %% ✅ COSTING MODULE
# ==============================================================================

# Period relevant for costing
relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
list_of_years_for_plot = list(range(2024, 2041))
number_of_years_costed = relevant_period_for_costing[1] - 2024 + 1

# Costing parameters
discount_rate = 0.03

# Estimate standard input costs of scenario
# -----------------------------------------------------------------------------------------------------------------------
# Standard 3% discount rate
input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
                                               _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
                                               _discount_rate=discount_rate, summarize=True)






# Undiscounted costs
input_costs_undiscounted = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
                                               _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
                                               _discount_rate=0, summarize=False)

# todo find costs of MDA administration and add
# https://pmc.ncbi.nlm.nih.gov/articles/PMC9272108/?utm_source=chatgpt.com#s3






