"""
Use the outputs from scenario_runs.py to produce plots
and summary statistics for paper.

JOB ID:
schisto_scenarios-2025-03-22T130153Z
"""

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
from typing import Tuple

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd, make_age_grp_types, parse_log_file, compare_number_of_deaths,
    extract_params, compute_summary_statistics, extract_results, get_scenario_info,
    get_scenario_outputs, load_pickled_dataframes, make_age_grp_lookup,
    unflatten_flattened_multi_index_in_logging
)

from scripts.costing.cost_estimation import (
    estimate_input_cost_of_scenarios, summarize_cost_data,
    do_stacked_bar_plot_of_cost_by_category, do_line_plot_of_cost,
    create_summary_treemap_by_cost_subgroup, estimate_projected_health_spending
)


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
    """Promote param names to the first level of the dataframe's column index."""
    _df.columns = pd.MultiIndex.from_product([param_names, _df.columns])
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


def total_dalys_by_year(_df):
    """Return total number of DALYs (Stacked) by year within the TARGET_PERIOD.
    Throw error if not a record for every year in the TARGET_PERIOD (to guard against runs that crashed).
    """
    years_needed = [i.year for i in TARGET_PERIOD]
    years_present = _df.year.unique()
    assert set(years_present).issuperset(years_needed), "Some years are not recorded."

    dalys_by_year = (
        _df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range'])
        .groupby('year')
        .sum()
        .sum(axis=1)  # sum across all causes (columns) within each year
    )

    return dalys_by_year




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
    _df = drop_outside_period(_df)
    _df['Year'] = pd.to_datetime(_df['date']).dt.year

    counts_by_year = (
        _df
        .groupby('Year')['Item_Used']
        .apply(lambda x: x.apply(pd.Series).sum())
        .apply(lambda s: s.get('286', 0))
    )

    return counts_by_year.astype(int)


def get_total_num_treatment_episdoes(_df):
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


def compute_icer(dalys_averted, comparison_pzq_costs):
    """
    Compute the Incremental Cost-Effectiveness Ratio (ICER) comparing PZQ costs and DALYs averted

    ICER is computed as:
        ICER = health_diff / cost_diff

    Parameters:
    -----------
    num_dalys_averted_vs_WASH : pd.DataFrame
        DataFrame of DALYs averted, indexed by scenario (rows) and draw (columns).
    comparison_pzq_costs_vs_WASH : pd.Series
        Series of cost differences, indexed to align with num_dalys_averted_vs_WASH.

    Returns:
    --------
    pd.Series
        Series of ICER values, indexed by draw.
    """

    # Transpose DALYs DataFrame to match the costs index structure
    dalys_averted = dalys_averted.T

    # Align cost data to match the DALYs data index
    aligned_costs = comparison_pzq_costs.reindex_like(dalys_averted)

    # Flatten values for ICER calculation
    cost_diff = aligned_costs.values.flatten()
    health_diff = dalys_averted.values.flatten()

    # Compute ICER
    icer_values = health_diff / cost_diff

    # Create a Series indexed by the 'draw' level
    icer_series = pd.Series(icer_values, index=aligned_costs.index.get_level_values('draw'))

    return icer_series


def calculate_npv_and_cost_per_daly(
    annual_num_dalys_averted: pd.DataFrame,
    annual_costs: pd.DataFrame,
    discount_factors: pd.Series
) -> pd.DataFrame:
    """
    Calculate the net present value (NPV) of DALYs averted and costs,
    compute the net present value (NPV) of the intervention, and calculate the cost per DALY averted.

    Parameters
    ----------
    annual_num_dalys_averted : pd.DataFrame
        DataFrame of annual DALYs averted by scenario. Rows indexed by year (int), columns are scenarios.
    annual_costs : pd.DataFrame
        DataFrame of annual incremental costs by scenario. Rows indexed by year (int), columns are scenarios.
    discount_factors : pd.Series
        Series of discount factors for each year, indexed by year (int).

    Returns
    -------
    pd.DataFrame
        DataFrame with NPV of DALYs averted, NPV of costs, net present value of the intervention, and cost per DALY averted for each scenario.
    """

    # Ensure consistent index names for alignment
    annual_num_dalys_averted.index.name = 'year'
    annual_costs.index.name = 'year'
    discount_factors.index.name = 'year'

    # Apply discount factors to DALYs averted and costs
    discounted_dalys_averted = annual_num_dalys_averted.multiply(discount_factors, axis=0)
    discounted_costs = annual_costs.multiply(discount_factors, axis=0)

    # Compute total NPV for DALYs averted and costs
    total_dalys_averted = discounted_dalys_averted.sum()
    total_costs = discounted_costs.sum()

    # Calculate net present value (NPV) for the intervention
    npv_intervention = total_dalys_averted - total_costs

    # Calculate cost per DALY averted
    cost_per_daly_averted = total_costs / total_dalys_averted

    # Compile results into a tidy DataFrame
    results = pd.DataFrame({
        'NPV_DALYs_Averted': total_dalys_averted,
        'NPV_Costs': total_costs,
        'NPV_Intervention': npv_intervention,  # Net Present Value of the intervention
        'Cost_per_DALY_Averted': cost_per_daly_averted
    })

    return results


def compute_discounted_nhb(df, discount_rate=0.03, threshold=150):
    """
    Compute discounted Net Health Benefit (NHB) from DALYs and costs.

    Parameters:
    - df: pd.DataFrame with columns ['scenario', 'year', 'dalys', 'cost']
    - discount_rate: annual discount rate (default 3%)
    - threshold: cost-effectiveness threshold per DALY averted (e.g. 150 USD)

    Returns:
    - pd.DataFrame with NHB for each scenario
    """

    df = df.copy()
    df['discount_factor'] = 1 / ((1 + discount_rate) ** df['year'])
    df['discounted_dalys'] = df['dalys'] * df['discount_factor']
    df['discounted_cost'] = df['cost'] * df['discount_factor']

    # Calculate NHB per scenario
    results = (
        df.groupby('scenario')
          .agg(total_dalys=('discounted_dalys', 'sum'),
               total_cost=('discounted_cost', 'sum'))
          .assign(nhb=lambda d: -d['total_dalys'] - d['total_cost'] / threshold)
    )

    return results


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



treatment_episodes = extract_results(
        results_folder,
        module='tlo.methods.schisto',
        key='schisto_treatment_episodes',
        custom_generate_series=get_total_num_treatment_episdoes,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)


summary_treatment_episodes = compute_summary_statistics(treatment_episodes,
                                                central_measure='median')
df_reshaped = summary_treatment_episodes.stack(level='draw')
df_reshaped = df_reshaped.reset_index(level=0, drop=True)


# ==============================================================================
# %% ✅ EXTRACTED RESULTS (HEALTH)
# ==============================================================================

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

# create df with PZQ use, number tx episodes and dalys
pzq_plus_tx_episodes = pd.concat([pzq_use, treatment_episodes, total_num_dalys])
pzq_plus_tx_episodes.to_csv(results_folder / (f'pzq_plus_tx_episodes {target_period()}.csv'))



# Total DALYs per year
dalys_per_year = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=total_dalys_by_year,
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0)


num_dalys_averted_vs_WASH = -1.0 * find_difference_relative_to_comparison_dataframe(
        total_num_dalys,
        comparison='WASH only'
    )


# ==============================================================================
# 📊 EXTRACT COSTS
# ==============================================================================


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
comparison_pzq_costs_vs_WASH = comparison_pzq_costs_vs_WASH.reindex_like(num_dalys_averted_vs_WASH)  # Align index


icer_WASH = compute_icer(dalys_averted=num_dalys_averted_vs_WASH, comparison_pzq_costs=comparison_pzq_costs_vs_WASH)
icer_WASH.to_csv(results_folder / (f'icer_WASH {target_period()}.csv'))



# discount rate
discount_rate = 0.03

# Define years and discount factors
years = np.arange(2024, 2041)
discount_factors = pd.Series(
    1 / ((1 + discount_rate) ** (years - 2024)),
    index=years
)

npv_results = calculate_npv_and_cost_per_daly(
    annual_num_dalys_averted=annual_num_dalys_averted_vs_WASH,
    annual_costs=annual_pzq_cost_annual_vs_WASH,
    discount_factors=discount_factors
)

npv_results.to_csv(results_folder / f'npv_results{target_period()}.csv')




# ==============================================================================
# 📊 GENERATE FIGURES
# ==============================================================================


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
input_costs_undiscounted = estimate_input_cost_of_scenarios(results_folder, resourcefilepath, _draws=[0, 3, 5, 8],
                                               _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
                                               _discount_rate=0, summarize=True)








