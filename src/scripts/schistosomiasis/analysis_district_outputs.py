""" use the outputs from scenario_runs.py and produce plots
and summary statistics for paper

JOB ID:
schisto_scenarios-2025-03-22T130153Z
"""

from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import pandas as pd
# import lacroix
import matplotlib.colors as colors
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from collections import defaultdict
import textwrap
from typing import Tuple
from scipy.stats import norm

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

tmp = log['tlo.methods.population']['scaling_factor_district']
district_scaling_factor = pd.DataFrame(tmp.iloc[0]['scaling_factor_district'], index=[0]).T
district_scaling_factor.columns = ['scaling_factor']


#################################################################################
# %% USEFUL FUNCTIONS
#################################################################################

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


def select_draws_by_keyword(df, keyword, level):
    """Select columns where the first level of the column MultiIndex contains the keyword (case-insensitive)."""
    mask = df.columns.get_level_values(level).str.contains(keyword, case=False)
    return df.loc[:, mask]


def sum_by_year_all_districts(df: pd.DataFrame, target_period: tuple) -> pd.DataFrame:
    """
    Restrict to years within TARGET_PERIOD and sum across all districts.
    Returns a DataFrame indexed by year with columns as (draw, run).
    """
    # Filter to years in TARGET_PERIOD
    start_year, end_year = TARGET_PERIOD

    df_filtered = df.loc[start_year.year:end_year.year]

    # Sum across districts (i.e. group by year)
    df_yearly = df_filtered.groupby(level='year').sum()

    return df_yearly


def compute_stepwise_effects_by_wash_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each WASH strategy, compute stepwise comparisons:
    - MDA SAC vs no MDA
    - MDA PSAC vs MDA SAC
    - MDA All vs MDA PSAC

    Returns a DataFrame with same index as input and columns for each comparison.
    """
    wash_strategies = ['Pause WASH', 'Continue WASH', 'Scale-up WASH']
    mda_steps = ['no MDA', 'MDA SAC', 'MDA PSAC', 'MDA All']
    comparisons = [
        ('MDA SAC', 'no MDA'),
        ('MDA PSAC', 'MDA SAC'),
        ('MDA All', 'MDA PSAC'),
    ]

    result_frames = []

    for wash in wash_strategies:
        for comp_to, comp_from in comparisons:
            draw_from = f'{wash}, {comp_from}'
            draw_to = f'{wash}, {comp_to}'
            # Ensure both draws exist in df
            if draw_from in df.columns.get_level_values('draw') and draw_to in df.columns.get_level_values('draw'):
                diff = df.xs(draw_to, level='draw', axis=1) - df.xs(draw_from, level='draw', axis=1)
                # Add informative column names
                diff.columns = pd.MultiIndex.from_product(
                    [[wash], [f"{comp_to} vs {comp_from}"], diff.columns],
                    names=['wash_strategy', 'comparison', 'run']
                )
                result_frames.append(diff)

    if not result_frames:
        raise ValueError("No matching comparisons could be computed. Check input column structure.")

    # Concatenate all comparison results along columns
    return pd.concat(result_frames, axis=1)



def compute_number_averted_vs_noMDA_within_wash_strategies(
    df: pd.DataFrame,
    wash_strategies: tuple = ("Pause WASH", "Continue WASH", "Scale-up WASH"),
    results_path: Path = None,
    filename_prefix: str = 'dalys_averted_by_year_run_district',
    target_period: tuple = None,
    averted_or_incurred: str = 'averted',
) -> pd.DataFrame:
    """
    Computes value (dalys or number py) averted by comparing each WASH strategy's MDA scenarios
    to the corresponding 'no MDA' baseline within the same strategy group.
    """
    if averted_or_incurred == 'averted':
        scale = -1.0
    else:
        scale = 1.0

    comparator_results = []

    for strategy in wash_strategies:
        comparator_draw = f"{strategy}, no MDA"
        # Skip the comparator in selection
        relevant_draws = [col for col in df.columns.get_level_values(0).unique()
                          if strategy in col and col != comparator_draw]
        if not relevant_draws:
            continue

        # Filter to relevant columns for this strategy
        df_subset = df.loc[:, df.columns.get_level_values(0).isin([comparator_draw] + relevant_draws)]

        # Compute difference relative to the strategy's 'no MDA' comparator
        diff_df = scale * find_difference_relative_to_comparison_dataframe(df_subset, comparison=comparator_draw)

        # Select only the relevant MDA draws
        comparator_results.append(diff_df)

    # Concatenate results across all strategies
    combined_df = pd.concat(comparator_results, axis=1)

    if results_path:
        period_str = f"_{target_period[0].year}-{target_period[1].year}" if target_period else ""
        output_file = results_path / f"{filename_prefix}{period_str}.xlsx"
        combined_df.to_excel(output_file)

    return combined_df


def compute_number_averted_vs_SAC_within_wash_strategies(
    df: pd.DataFrame,
    wash_strategies: tuple = ("Pause WASH", "Continue WASH", "Scale-up WASH"),
    results_path: Path = None,
    filename_prefix: str = 'dalys_averted_by_year_run_district',
    target_period: tuple = None,
    averted_or_incurred: str = 'averted',
) -> pd.DataFrame:
    """
    Computes value (dalys or number py) averted by comparing each WASH strategy's MDA scenarios
    to the corresponding 'no MDA' baseline within the same strategy group.
    """
    if averted_or_incurred == 'averted':
        scale = -1.0
    else:
        scale = 1.0

    comparator_results = []

    for strategy in wash_strategies:
        comparator_draw = f"{strategy}, MDA SAC"
        # Skip the comparator in selection
        relevant_draws = [col for col in df.columns.get_level_values(0).unique()
                          if strategy in col and col != comparator_draw]
        if not relevant_draws:
            continue

        # Filter to relevant columns for this strategy
        df_subset = df.loc[:, df.columns.get_level_values(0).isin([comparator_draw] + relevant_draws)]

        # Compute difference relative to the strategy's 'no MDA' comparator
        diff_df = scale * find_difference_relative_to_comparison_dataframe(df_subset, comparison=comparator_draw)

        # Select only the relevant MDA draws
        comparator_results.append(diff_df)

    # Concatenate results across all strategies
    combined_df = pd.concat(comparator_results, axis=1)

    if results_path:
        period_str = f"_{target_period[0].year}-{target_period[1].year}" if target_period else ""
        output_file = results_path / f"{filename_prefix}{period_str}.xlsx"
        combined_df.to_excel(output_file)

    return combined_df


def format_summary_for_output(
    df: pd.DataFrame,
    stack_level: str = 'draw',
    scale_factor: float = 1_000_000,
    stat_central: str = 'central',
    stat_lower: str = 'lower',
    stat_upper: str = 'upper',
    filename_prefix: str = 'table',
    filename: str = None,
) -> pd.DataFrame:
    """
    Format a DataFrame with uncertainty statistics into a readable table and save as Excel.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with formatted strings showing central (lower–upper).
    """

    # Scale the data
    scaled_df = df / scale_factor

    # Stack to long format on the specified level
    long_df = scaled_df.stack(level=stack_level)

    # Apply formatting per row
    formatted = long_df.apply(
        lambda row: f"{row[stat_central]:.2f} ({row[stat_lower]:.2f}–{row[stat_upper]:.2f})",
        axis=1
    )

    # Unstack to wide format, columns = stack_level values
    formatted_df = formatted.unstack(level=-1)

    # Save to Excel
    output_path = results_folder / f'{filename_prefix}_{filename}_{target_period()}.xlsx'
    formatted_df.to_excel(output_path)

    return formatted_df


def compute_icer_district(
    dalys_averted: pd.DataFrame,
    comparison_costs: pd.DataFrame,
    discount_rate_dalys: float = 0.0,
    discount_rate_costs: float = 0.0,
    return_summary: bool = True
) -> pd.DataFrame | pd.Series:
    """
    Compute ICERs comparing costs and DALYs averted over a TARGET_PERIOD by district, run, and scenario.

    Assumes:
    - Row MultiIndex: ['year', 'District']
    - Column MultiIndex: ['wash_strategy', 'comparison', 'run']
    - TARGET_PERIOD is a global tuple of (start_year, end_year) as datetime or int
    """
    global TARGET_PERIOD
    start_year, end_year = TARGET_PERIOD

    # Filter index by year
    years = dalys_averted.index.get_level_values('year')
    mask = (years >= start_year.year) & (years <= end_year.year)
    dalys_period = dalys_averted.loc[mask]
    costs_period = comparison_costs.loc[mask]

    # Discounting
    years_since_start = years[mask] - start_year.year
    discount_weights_dalys = 1 / ((1 + discount_rate_dalys) ** years_since_start)
    discount_weights_costs = 1 / ((1 + discount_rate_costs) ** years_since_start)

    if discount_rate_dalys != 0.0:
        dalys_period = dalys_period.mul(discount_weights_dalys.values, axis=0)

    if discount_rate_costs != 0.0:
        costs_period = costs_period.mul(discount_weights_costs.values, axis=0)

    # Sum over years within each district
    total_dalys = dalys_period.groupby(level='district_of_residence').sum()
    total_costs = costs_period.groupby(level='District').sum()

    # Compute ICERs
    icers = total_costs / total_dalys

    # Convert to long format
    icers_long = (
        icers
        .stack(['wash_strategy', 'comparison', 'run'])
        .rename('icer')
        .reset_index()
        .rename(columns={'level_0': 'District'})
    )

    if return_summary:
        summary = (
            icers_long
            .groupby(['District', 'wash_strategy', 'comparison'])['icer']
            .agg(
                mean='mean',
                lower=lambda x: np.quantile(x, 0.025),
                upper=lambda x: np.quantile(x, 0.975)
            )
            .reset_index()
        )
        return summary
    else:
        return icers_long


def compute_icer_national(
    dalys_averted: pd.DataFrame,
    comparison_costs: pd.DataFrame,
    discount_rate_dalys: float = 0.0,
    discount_rate_costs: float = 0.0,
    return_summary: bool = True
) -> pd.DataFrame | pd.Series:
    """
    Compute ICERs comparing costs and DALYs averted over TARGET_PERIOD.

    Assumes:
    - Row index: year (single-level)
    - Column MultiIndex: (wash_strategy, comparison, run)
    - TARGET_PERIOD is a global tuple (start_year, end_year), with year as int or datetime
    """
    global TARGET_PERIOD
    start_year, end_year = TARGET_PERIOD

    # Check row index
    if not dalys_averted.index.name == "year":
        raise ValueError("dalys_averted must have a single-level index named 'year'")

    # Filter by year
    mask = (dalys_averted.index >= start_year.year) & (dalys_averted.index <= end_year.year)
    dalys_period = dalys_averted.loc[mask]
    costs_period = comparison_costs.loc[mask]

    # Discounting
    years = dalys_period.index
    years_since_start = years - start_year.year
    discount_weights_dalys = 1 / ((1 + discount_rate_dalys) ** years_since_start)
    discount_weights_costs = 1 / ((1 + discount_rate_costs) ** years_since_start)

    if discount_rate_dalys != 0.0:
        dalys_period = dalys_period.mul(discount_weights_dalys.values[:, None])
    if discount_rate_costs != 0.0:
        costs_period = costs_period.mul(discount_weights_costs.values[:, None])

    # Sum over years
    total_dalys = dalys_period.sum(axis=0)
    total_costs = costs_period.sum(axis=0)

    # ICER calculation
    icer = total_costs / total_dalys  # Result: Series with MultiIndex (wash_strategy, comparison, run)

    # Format
    icer = icer.rename("icer").reset_index()

    if return_summary:
        summary = (
            icer.groupby(["wash_strategy", "comparison"])["icer"]
            .agg(
                mean="mean",
                lower=lambda x: np.quantile(x, 0.025),
                upper=lambda x: np.quantile(x, 0.975)
            )
            .reset_index()
        )
        return summary
    else:
        return icer



def combine_on_keyword(df1: pd.DataFrame, df2: pd.DataFrame, keyword: str = "MDA All") -> pd.DataFrame:
    """
    Combine two DataFrames with identical MultiIndex columns (draw, run),
    taking columns containing `keyword` from df2 and all others from df1.
    """
    # Ensure the column MultiIndex names
    if df1.columns.names != df2.columns.names:
        raise ValueError("df1 and df2 must have the same column MultiIndex names")

    # Identify which columns to take from df2
    draw_level = df1.columns.names.index("draw")
    draws = df1.columns.get_level_values(draw_level)
    mask = draws.str.contains(keyword)

    # Build the result
    result = df1.copy(deep=False)  # shallow copy of values will be overwritten
    # Overwrite only the masked columns with df2's values
    cols_to_replace = df1.columns[mask]
    result.loc[:, cols_to_replace] = df2.loc[:, cols_to_replace]

    return result



#################################################################################
# %% DISTRICT FUNCTIONS - PREVALENCE
#################################################################################

def get_prevalence_infection_all_ages_by_district(_df):
    """Return yearly prevalence of infection (any age) by district as a pd.Series indexed by (year, district)."""

    global inf

    _df = _df.copy()
    _df.set_index('date', inplace=True)

    def parse_columns(cols):
        tuples = []
        for col in cols:
            parts = col.split('|')
            values = [p.split('=')[1] for p in parts]
            tuples.append(tuple(values))
        return tuples

    _df.columns = pd.MultiIndex.from_tuples(parse_columns(_df.columns),
                                           names=['infection_status', 'district_of_residence', 'age_years'])

    # Sum across ages
    df = _df.groupby(level=['infection_status', 'district_of_residence'], axis=1).sum()

    # Total population by district
    total_by_district = df.groupby(level='district_of_residence', axis=1).sum()

    inf_categories_dict = {
        'HML': ['High-infection', 'Moderate-infection', 'Low-infection'],
        'HM': ['High-infection', 'Moderate-infection'],
        'ML': ['Moderate-infection', 'Low-infection'],
        'H': ['High-infection'],
        'M': ['Moderate-infection'],
        'L': ['Low-infection'],
    }

    if inf not in inf_categories_dict:
        raise ValueError(f"Unknown inf='{inf}' — must be one of {list(inf_categories_dict)}")

    inf_categories = inf_categories_dict[inf]

    infected_by_district = df.loc[:, df.columns.get_level_values('infection_status').isin(inf_categories)]
    infected_by_district = infected_by_district.groupby(level='district_of_residence', axis=1).sum()

    prevalence_df = infected_by_district.divide(total_by_district)

    # Convert datetime index to year (integer)
    prevalence_df.index = prevalence_df.index.year

    # Convert prevalence DataFrame (years x districts) to Series indexed by (year, district)
    prevalence_series = prevalence_df.stack(dropna=False)
    prevalence_series.index.names = ['year', 'district']

    return prevalence_series


inf = 'H'  # define outside function, set before calling
prev_haem_H_All_district = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_haematobium",
    custom_generate_series=get_prevalence_infection_all_ages_by_district,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)

prev_haem_H_All_district.to_excel(results_folder / (f'prev_haem_H_year_district {target_period()}.xlsx'))

prev_mansoni_H_All_district = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_mansoni",
    custom_generate_series=get_prevalence_infection_all_ages_by_district,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)

prev_mansoni_H_All_district.to_excel(results_folder / (f'prev_mansoni_H_year_district {target_period()}.xlsx'))


# -------------------- prevalence of any infection


inf = 'HML'  # define outside function, set before calling
prev_haem_HML_All_district = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_haematobium",
    custom_generate_series=get_prevalence_infection_all_ages_by_district,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)

prev_haem_HML_All_district.to_excel(results_folder / (f'prev_haem_HML_year_summary {target_period()}.xlsx'))

prev_mansoni_HML_All_district = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_mansoni",
    custom_generate_series=get_prevalence_infection_all_ages_by_district,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)

prev_mansoni_HML_All_district.to_excel(results_folder / (f'prev_mansoni_HML_All_district {target_period()}.xlsx'))




def calc_mean_and_ci(df, ci=0.95):
    """
    Calculate mean and confidence intervals over the 'run' level of columns,
    grouped by 'year', 'district', and 'draw' (first level of columns).

    Parameters:
    - df: pd.DataFrame with MultiIndex rows including 'year' and 'district',
          and MultiIndex columns with levels ['draw', 'run'].
    - ci: confidence level (default 0.95 for 95% CI).

    Returns:
    - pd.DataFrame indexed by ['year', 'district', 'draw'] with columns ['mean', 'lower_ci', 'upper_ci'].
    """
    # Extract MultiIndex names for columns
    draw_level = df.columns.names.index('draw')
    run_level = df.columns.names.index('run')

    # Group by draw (level=0) to aggregate over runs (level=1)
    mean_df = df.groupby(axis=1, level='draw').mean()
    std_df = df.groupby(axis=1, level='draw').std()
    count_df = df.groupby(axis=1, level='draw').count()

    # Standard error across runs
    se_df = std_df / np.sqrt(count_df)

    # z-score for the two-tailed confidence interval
    z = norm.ppf(1 - (1 - ci) / 2)

    # Calculate confidence intervals
    lower_ci = mean_df - z * se_df
    upper_ci = mean_df + z * se_df

    # Stack the dataframes to long format for merging
    mean_long = mean_df.stack().rename('mean')
    lower_long = lower_ci.stack().rename('lower_ci')
    upper_long = upper_ci.stack().rename('upper_ci')

    # Combine into one DataFrame
    result = pd.concat([mean_long, lower_long, upper_long], axis=1).reset_index()

    # Rename columns for clarity
    result = result.rename(columns={'level_3': 'draw'}) if 'level_3' in result.columns else result

    return result


prev_haem_HML_All_district_summary = calc_mean_and_ci(prev_haem_HML_All_district)
prev_haem_HML_All_district_summary.to_excel(results_folder / (f'prev_haem_HML_All_district_summary {target_period()}.xlsx'))


prev_mansoni_HML_All_district_summary = calc_mean_and_ci(prev_mansoni_HML_All_district)
prev_mansoni_HML_All_district_summary.to_excel(results_folder / (f'prev_mansoni_HML_All_district_summary {target_period()}.xlsx'))


####################################################################################
# %%  NATIONAL PREVALENCE - SCALED BY DISTRICT
####################################################################################


def get_national_prevalence_scaled(_df):
    """
    Return mean national prevalence per year using district-level scaling.

    _df: DataFrame with columns indexed like 'infection_status=...|district_of_residence=...|age_years=...'
    Assumes global 'inf' and 'district_scaling_factor' are defined.
    """

    global inf, district_scaling_factor

    _df = _df.copy()
    _df.set_index('date', inplace=True)

    def parse_columns(cols):
        tuples = []
        for col in cols:
            parts = col.split('|')
            values = [p.split('=')[1] for p in parts]
            tuples.append(tuple(values))
        return tuples

    # Convert columns to MultiIndex
    _df.columns = pd.MultiIndex.from_tuples(parse_columns(_df.columns),
                                            names=['infection_status', 'district_of_residence', 'age_years'])

    # Collapse age groups
    df = _df.groupby(level=['infection_status', 'district_of_residence'], axis=1).sum()

    # Total population per district
    total_by_district = df.groupby(level='district_of_residence', axis=1).sum()

    # Select infection categories
    inf_categories_dict = {
        'HML': ['High-infection', 'Moderate-infection', 'Low-infection'],
        'HM': ['High-infection', 'Moderate-infection'],
        'ML': ['Moderate-infection', 'Low-infection'],
        'H': ['High-infection'],
        'M': ['Moderate-infection'],
        'L': ['Low-infection'],
    }

    if inf not in inf_categories_dict:
        raise ValueError(f"Unknown inf='{inf}' — must be one of {list(inf_categories_dict)}")

    inf_categories = inf_categories_dict[inf]
    infected_by_district = df.loc[:, df.columns.get_level_values('infection_status').isin(inf_categories)]
    infected_by_district = infected_by_district.groupby(level='district_of_residence', axis=1).sum()

    # Check all districts match scaling factor
    sf = district_scaling_factor
    sf = sf['scaling_factor']  # convert to Series indexed by district

    missing = set(infected_by_district.columns) - set(sf.index)
    if missing:
        raise ValueError(f"Missing scaling factors for districts: {missing}")

    # Apply scaling
    sf.index = sf.index.astype(str)
    infected_by_district.columns = infected_by_district.columns.astype(str)

    # Make sure all required districts are present
    if not set(infected_by_district.columns).issubset(sf.index):
        missing = set(infected_by_district.columns) - set(sf.index)
        raise ValueError(f"Missing scaling factors for districts: {missing}")

    # Apply scaling
    scaled_infected = infected_by_district.multiply(sf, axis=1)
    scaled_population = total_by_district.multiply(sf, axis=1)

    # Convert to year index
    scaled_infected.index = scaled_infected.index.year
    scaled_population.index = scaled_population.index.year

    # National sums
    national_infected = scaled_infected.sum(axis=1)
    national_population = scaled_population.sum(axis=1)

    # Debug: Check for zero population
    if (national_population == 0).any():
        print("Warning: Some years have zero national population. Those will return NaN.")

    # Compute prevalence
    national_prevalence = national_infected / national_population

    return national_prevalence


inf = 'HML'  # define outside function, set before calling


prev_haem_national = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_haematobium",
    custom_generate_series=get_national_prevalence_scaled,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)

prev_haem_national.to_excel(results_folder / (f'prev_haem_national {target_period()}.xlsx'))


prev_mansoni_national = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="infection_status_mansoni",
    custom_generate_series=get_national_prevalence_scaled,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)

prev_mansoni_national.to_excel(results_folder / (f'prev_mansoni_national {target_period()}.xlsx'))


# todo this returns very small CI
def calc_mean_and_ci(df, ci=0.95):
    from scipy.stats import norm
    import numpy as np

    # z-score for two-tailed CI
    z = norm.ppf(1 - (1 - ci) / 2)

    # Mean over 'run' (level='run')
    mean_df = df.groupby(axis=1, level='draw').mean()
    std_df = df.groupby(axis=1, level='draw').std()
    count_df = df.groupby(axis=1, level='draw').count()

    se_df = std_df / np.sqrt(count_df)

    lower_ci = mean_df - z * se_df
    upper_ci = mean_df + z * se_df

    # Convert to long format
    mean_long = mean_df.stack().rename('mean')
    lower_long = lower_ci.stack().rename('lower_ci')
    upper_long = upper_ci.stack().rename('upper_ci')

    result = pd.concat([mean_long, lower_long, upper_long], axis=1).reset_index()

    # Rename columns if needed
    if 'level_1' in result.columns:
        result = result.rename(columns={'level_1': 'draw'})

    return result


def calc_mean_and_ci_quantiles(df, ci=0.95):
    """
    Calculate mean and empirical confidence intervals (based on quantiles) over the 'run' level of columns,
    grouped by 'draw' (first level of columns).

    Parameters:
    - df: pd.DataFrame with MultiIndex columns ['draw', 'run'] and rows indexed by at least 'date'.
    - ci: confidence level, default 0.95 (95% CI).

    Returns:
    - pd.DataFrame with columns ['date', 'draw', 'mean', 'lower_ci', 'upper_ci'].
    """
    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q

    # Mean over runs grouped by draw
    mean_df = df.groupby(axis=1, level='draw').mean()

    # Quantiles over runs grouped by draw
    lower_ci_df = df.groupby(axis=1, level='draw').quantile(lower_q)
    upper_ci_df = df.groupby(axis=1, level='draw').quantile(upper_q)

    # Stack to long format
    mean_long = mean_df.stack().rename('mean')
    lower_long = lower_ci_df.stack().rename('lower_ci')
    upper_long = upper_ci_df.stack().rename('upper_ci')

    # Combine into single DataFrame
    result = pd.concat([mean_long, lower_long, upper_long], axis=1).reset_index()

    # Rename columns for clarity if needed
    # Expected columns: ['date', 'draw', 'mean', 'lower_ci', 'upper_ci']
    # If your index names differ, adjust accordingly

    return result


def calc_mean_and_range_ci(df):
    """
    Calculate mean and empirical confidence intervals as range (min to max)
    over the 'run' level of columns, grouped by 'draw' (first level of columns).

    Parameters:
    - df: pd.DataFrame with MultiIndex columns ['draw', 'run'] and rows indexed by at least 'date'.

    Returns:
    - pd.DataFrame with columns ['date', 'draw', 'mean', 'lower_ci', 'upper_ci'].
    """
    # Mean over runs grouped by draw
    mean_df = df.groupby(axis=1, level='draw').mean()

    # Min and max over runs grouped by draw
    lower_ci_df = df.groupby(axis=1, level='draw').min()
    upper_ci_df = df.groupby(axis=1, level='draw').max()

    # Stack to long format
    mean_long = mean_df.stack().rename('mean')
    lower_long = lower_ci_df.stack().rename('lower_ci')
    upper_long = upper_ci_df.stack().rename('upper_ci')

    # Combine into single DataFrame
    result = pd.concat([mean_long, lower_long, upper_long], axis=1).reset_index()

    return result


prev_haem_national_summary = calc_mean_and_range_ci(prev_haem_national)
prev_haem_national_summary.to_excel(results_folder / (f'prev_haem_national_summary {target_period()}.xlsx'))


prev_mansoni_national_summary = calc_mean_and_range_ci(prev_mansoni_national)
prev_mansoni_national_summary.to_excel(results_folder / (f'prev_mansoni_national_summary {target_period()}.xlsx'))







####################################################################################
# %%  YEAR REACHING EPHP
####################################################################################

def get_first_years_below_threshold(
    df: pd.DataFrame,
    threshold: float = 0.015,
    year_range: tuple = (2024, 2040)
) -> pd.DataFrame:
    """Return the first year each district drops below the threshold for each strategy."""
    df = df.loc[df.index.get_level_values("year") >= year_range[0]]
    df_mean_runs = df.groupby(axis=1, level="draw").mean()

    below = (df_mean_runs < threshold).reset_index()
    long_format = below.melt(id_vars=["year", "district"], var_name="draw", value_name="below_threshold")
    below_threshold = long_format[long_format["below_threshold"]]

    first_years = below_threshold.groupby(["district", "draw"])["year"].min().reset_index(name="year_ephp")
    return first_years


first_years_ephp_df_haem = get_first_years_below_threshold(prev_haem_H_All_district)
first_years_ephp_df_haem.to_excel(results_folder / (f'first_years_ephp_df_haem {target_period()}.xlsx'))



first_years_ephp_df_mansoni = get_first_years_below_threshold(prev_mansoni_H_All_district)
first_years_ephp_df_mansoni.to_excel(results_folder / (f'first_years_ephp_df_mansoni {target_period()}.xlsx'))






#################################################################################
# %% PREVALENCE OF INFECTION OVERALL (BOTH SPECIES) BY DISTRICT
#################################################################################

number_infected = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="number_infected_any_species",
    column="number_infected",
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)

number_in_district = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="number_in_subgroup",
    column="number_alive",
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)


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

result.index = pd.Index(range(2010, 2041), name="year")
result.to_excel(results_folder / f'prevalence_any_infection_all_ages_district{target_period()}.xlsx')

# summarise the prevalence for each district by draw
median_by_draw_district = result.groupby(level=['draw', 'district'], axis=1).median()
median_by_draw_district.to_excel(results_folder / (f'median_prevalence_any_infection_all_ages_by_year_district{target_period()}.xlsx'))


#################################################################################
# %% PERSON-YEARS INFECTED BY DISTRICT
#################################################################################

def get_person_years_infected_by_district(_df: pd.DataFrame) -> pd.Series:
    """
    Get person-years infected by district, summed over the specified time period and infection levels.
    """
    # Filter to target period
    df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)].drop(columns='date')

    # Filter columns by infection level
    pattern = '|'.join(infection_levels)
    df_filtered = df.filter(regex=f'infection_level=({pattern})')

    # Convert column names to a MultiIndex
    columns_split = df_filtered.columns.str.extract(r'species=([^|]+)\|age_group=([^|]+)\|infection_level=([^|]+)\|district=([^|]+)')
    df_filtered.columns = pd.MultiIndex.from_frame(columns_split, names=['species', 'age_group', 'infection_level', 'district'])

    # Sum across time (i.e., sum values for each column over the period)
    summed_by_time = df_filtered.sum(axis=0)

    # Group by district and sum
    py_by_district = summed_by_time.groupby('district').sum() / 365.25

    return py_by_district


infection_levels = ['Low-infection', 'Moderate-infection', 'High-infection']

py_district = extract_results(
        results_folder,
        module="tlo.methods.schisto",
        key="Schisto_person_days_infected",
        custom_generate_series=get_person_years_infected_by_district,
        do_scaling=False,
    ).pipe(set_param_names_as_column_index_level_0)


# need to multiply by district-level scaling factor
scaled_py_district = py_district.mul(district_scaling_factor['scaling_factor'], axis=0)
scaled_py_district.to_excel(results_folder / f'num_py_infected_by_district_{target_period()}.xlsx')

# summarise the py infected for each district by draw
median_py_district = scaled_py_district.groupby(level=['draw'], axis=1).median()
median_py_district.to_excel(results_folder / (f'median_py_any_infection_by_year_district{target_period()}.xlsx'))


num_py_averted_pause = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        scaled_py_district,
        comparison='Pause WASH, no MDA'
    ),
    central_measure='mean'
)
num_py_averted_continue = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        scaled_py_district,
        comparison='Continue WASH, no MDA'
    ),
    central_measure='mean'
)
num_py_averted_scaleup = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        scaled_py_district,
        comparison='Scale-up WASH, no MDA'
    ),
    central_measure='mean'
)

# Select desired columns from each dataframe
df1_sel = select_draws_by_keyword(num_py_averted_pause, 'Pause', level=0)
df2_sel = select_draws_by_keyword(num_py_averted_continue, 'Continue', level=0)
df3_sel = select_draws_by_keyword(num_py_averted_scaleup, 'Scale-up', level=0)

# Concatenate the selected columns horizontally
num_py_averted_combined = pd.concat([df1_sel, df2_sel, df3_sel], axis=1)
num_py_averted_combined.to_csv(results_folder / f'num_py_averted_by_district{target_period()}.xlsx')

# to get total national-level py averted weighted by district
# sum the total py across all districts
num_py_summed_series = pd.DataFrame(scaled_py_district.sum(axis=0)).T

py_averted_national_pause = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        num_py_summed_series,
        comparison='Pause WASH, no MDA'
    ),
    central_measure='mean'
)
py_averted_national_continue = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        num_py_summed_series,
        comparison='Continue WASH, no MDA'
    ),
    central_measure='mean'
)
py_averted_national_scaleup = compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        num_py_summed_series,
        comparison='Scale-up WASH, no MDA'
    ),
    central_measure='mean'
)
df1_sel = select_draws_by_keyword(py_averted_national_pause, 'Pause', level=0)
df2_sel = select_draws_by_keyword(py_averted_national_continue, 'Continue', level=0)
df3_sel = select_draws_by_keyword(py_averted_national_scaleup, 'Scale-up', level=0)

# Concatenate the selected columns horizontally
num_py_averted_combined_national = pd.concat([df1_sel, df2_sel, df3_sel], axis=1)
num_py_averted_combined_national.to_excel(results_folder / f'num_py_averted_national{target_period()}.xlsx')


#################################################################################
# %% PERSON-YEARS INFECTED BY AGE AND DISTRICT
#################################################################################


def get_person_years_infected_by_district_and_age(_df: pd.DataFrame) -> pd.Series:
    """
    Get person-years infected by district and age group, summed over the specified time period and infection levels.
    """
    # Filter to target period
    df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)].drop(columns='date')

    # Filter columns by infection level
    pattern = '|'.join(infection_levels)
    df_filtered = df.filter(regex=f'infection_level=({pattern})')

    # Convert column names to a MultiIndex
    columns_split = df_filtered.columns.str.extract(
        r'species=([^|]+)\|age_group=([^|]+)\|infection_level=([^|]+)\|district=([^|]+)'
    )
    df_filtered.columns = pd.MultiIndex.from_frame(columns_split, names=['species', 'age_group', 'infection_level', 'district'])

    # Sum across time (i.e., sum values for each column over the period)
    summed_by_time = df_filtered.sum(axis=0)

    # Group by district and age_group, summing over infection levels and species
    py_by_district_age = summed_by_time.groupby(['district', 'age_group']).sum() / 365.25

    return py_by_district_age


py_district_age = extract_results(
    results_folder,
    module="tlo.methods.schisto",
    key="Schisto_person_days_infected",
    custom_generate_series=get_person_years_infected_by_district_and_age,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)

# Multiply by district scaling factor (broadcasting across age_groups)
scaled_py_district_age = py_district_age.mul(
    district_scaling_factor['scaling_factor'],
    axis=0,
    level='district'  # ensures correct alignment on the first level of the index
)

scaled_py_district_age.to_excel(results_folder / f'num_py_infected_by_district_and_age_{target_period()}.xlsx')

summary_py_by_district_age = compute_summary_statistics(scaled_py_district_age,
                                                        central_measure='mean')

summary_py_by_district_age.to_excel(results_folder / f'summary_num_py_infected_by_district_and_age_{target_period()}.xlsx')


# format into nice table for output
df_million = summary_py_by_district_age / 1_000_000

# Rearrange into long format: rows = (district, age_group, draw), columns = stat
df_long = df_million.stack(level='draw')  # columns now: lower, central, upper
formatted = df_long.apply(
    lambda row: f"{row['central']:.2f} ({row['lower']:.2f}–{row['upper']:.2f})", axis=1
)
# Return to wide format: rows = (district, age_group), columns = draw
formatted_df = formatted.unstack(level=-1)

output_path = results_folder / f'table_summary_py_by_district_age_{target_period()}.xlsx'
formatted_df.to_excel(output_path)


# get the total py infected nationally by age-group
by_age_group = scaled_py_district_age.groupby(level='age_group').sum()

# Step 2: Aggregate the age groups
adults_total = by_age_group.loc['Adults']
infants_psac_total = by_age_group.loc[['Infant', 'PSAC']].sum()
sac_total = by_age_group.loc['SAC']

# Step 3: Combine into single DataFrame and convert to millions
age_group_summary = pd.DataFrame({
    'Adults': adults_total,
    'Infants+PSAC': infants_psac_total,
    'SAC': sac_total
}).T

age_group_summary2 = compute_summary_statistics(age_group_summary,
                                                central_measure='mean')

# format into nice table for output
tmp = age_group_summary2 / 1_000_000

# Rearrange into long format: rows = (district, age_group, draw), columns = stat
df_long = tmp.stack(level='draw')  # columns now: lower, central, upper
formatted = df_long.apply(
    lambda row: f"{row['central']:.2f} ({row['lower']:.2f}–{row['upper']:.2f})", axis=1
)
# Return to wide format: rows = (district, age_group), columns = draw
formatted_df = formatted.unstack(level=-1)

output_path = results_folder / f'table_summary_py_national_age_{target_period()}.xlsx'
formatted_df.to_excel(output_path)



# -------------------------- national PY infected by age
# todo get the national numbers of py infected by age scaled by district

# Sum across districts for each age_group, for every (draw, run) combination
df_summed = scaled_py_district_age.groupby(level='age_group').sum()
# Rearrange so we can group over 'draw' in columns
# The result will be: index=age_group, columns=draw, values=mean over runs
mean_by_draw = df_summed.groupby(axis=1, level='draw').mean()
se_by_draw = df_summed.groupby(axis=1, level='draw').sem()  # standard error of the mean

output_path = results_folder / f'mean_py_age_national{target_period()}.xlsx'
mean_by_draw.to_excel(output_path)

output_path = results_folder / f'se_py_age_national{target_period()}.xlsx'
se_by_draw.to_excel(output_path)



###########################################################################################################
# %% get DALYs by district / national
###########################################################################################################

num_dalys_by_year_run_district = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_by_wealth_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
    custom_generate_series=(
        lambda df_: df_.drop(
            columns=(['date']),
        ).groupby(['year', 'district_of_residence']).sum().stack()
    ),
    do_scaling=False
).pipe(set_param_names_as_column_index_level_0)

dalys_schisto_district = num_dalys_by_year_run_district.loc[
    num_dalys_by_year_run_district.index.get_level_values(2) == 'Schistosomiasis'].droplevel(2)

# remove values stored for 2041 (all zeros)
dalys_schisto_district = dalys_schisto_district.loc[dalys_schisto_district.index.get_level_values(0) <= 2040]

# Extract district names from index level 1
districts = dalys_schisto_district.index.get_level_values(1)

# Align scaling factors to the district index in df_schisto
scaling_factors = district_scaling_factor.loc[districts].iloc[:, 0].values

# Multiply df_schisto by scaling factors, broadcasting over rows
dalys_schisto_district_scaled = dalys_schisto_district.multiply(scaling_factors, axis=0)
dalys_schisto_district_scaled.to_excel(results_folder / f'schisto_dalys_by_year_run_district{target_period()}.xlsx')


# === national total DALYs  =========================================================

# get the total national level DALYs incurred 2024-2040 weighted by district
dalys_summed_by_year = sum_by_year_all_districts(dalys_schisto_district_scaled, TARGET_PERIOD)
dalys_summed_by_year.to_excel(results_folder / f'schisto_dalys_by_year_run_national{target_period()}.xlsx')

# get the total dalys nationally
dalys_national = pd.DataFrame(dalys_summed_by_year.sum()).T
dalys_national_summary = compute_summary_statistics(dalys_national,
                                                central_measure='mean')

formatted_table = format_summary_for_output(dalys_national_summary, filename='summary_dalys_national')


# === DALYs averted by district =========================================================

dalys_averted_district_compared_noMDA = compute_number_averted_vs_noMDA_within_wash_strategies(
    dalys_schisto_district_scaled,
    results_path=results_folder,
    filename_prefix='schisto_dalys_averted_by_year_run_district',
    target_period=TARGET_PERIOD,
    averted_or_incurred='averted'
)
dalys_averted_district_compared_noMDA.to_excel(results_folder / f'dalys_averted_district_compared_noMDA{target_period()}.xlsx')


dalys_averted_district_compared_SAC = compute_number_averted_vs_SAC_within_wash_strategies(
    dalys_schisto_district_scaled,
    results_path=results_folder,
    filename_prefix='schisto_dalys_averted_by_year_run_district',
    target_period=TARGET_PERIOD,
    averted_or_incurred='averted'
)
dalys_averted_district_compared_SAC.to_excel(results_folder / f'dalys_averted_district_compared_SAC{target_period()}.xlsx')


# --- Incremental DALYs averted by district ---

# incremental dalys averted - compare each prog in turn to the last one
incremental_dalys_averted_district = -1 * compute_stepwise_effects_by_wash_strategy(dalys_schisto_district_scaled)
incremental_dalys_averted_district.to_excel(results_folder / f'stepwise_dalys_averted_year_district{target_period()}.xlsx')

years = incremental_dalys_averted_district.index.get_level_values('year')
mask = (years >= TARGET_PERIOD[0].year) & (years <= TARGET_PERIOD[1].year)

sum_incremental_dalys_averted_district = (
    incremental_dalys_averted_district.loc[mask]
    .groupby('district_of_residence')
    .sum()
)
sum_incremental_dalys_averted_district.to_excel(results_folder / f'sum_incremental_dalys_averted_district{target_period()}.xlsx')


# === DALYs averted national =========================================================

dalys_averted_national_compared_noMDA = compute_number_averted_vs_noMDA_within_wash_strategies(
    dalys_summed_by_year,
    results_path=results_folder,
    filename_prefix='schisto_dalys_averted_by_year_run_national',
    target_period=TARGET_PERIOD,
    averted_or_incurred='averted'
)

# incremental dalys averted - compare each prog in turn to the last one
incremental_dalys_averted_national = -1 * compute_stepwise_effects_by_wash_strategy(dalys_summed_by_year)
incremental_dalys_averted_national.to_excel(results_folder / f'stepwise_dalys_averted_year_national{target_period()}.xlsx')


#################################################################################
# %% COSTS FOR MDA
#################################################################################

def get_counts_of_mda_by_year_district(_df):
    """
    Returns a Series with the count of MDA episodes per district per year,
    ensuring all districts have an entry for every year (zeros filled where absent).
    """
    _df['Year'] = pd.to_datetime(_df['date']).dt.year

    # Flatten the data into (Year, District, Count) triplets
    records = []

    for _, row in _df.iterrows():
        year = row['Year']
        district_counts = row['mda_episodes_district']['ss_MDA_treatment_counter']
        for district, count in district_counts.items():
            records.append((year, district, count))

    df_counts = pd.DataFrame(records, columns=['Year', 'District', 'Count'])

    # Aggregate counts per (Year, District)
    grouped = df_counts.groupby(['Year', 'District'])['Count'].sum()

    # Get all unique years and districts
    all_years = df_counts['Year'].unique()
    all_districts = df_counts['District'].unique()

    # Create a complete MultiIndex of all year-district pairs
    full_index = pd.MultiIndex.from_product(
        [all_years, all_districts],
        names=['Year', 'District']
    )

    # Reindex the grouped counts to the full index, filling missing with 0
    result = grouped.reindex(full_index, fill_value=0)

    return result.astype(int)


mda_episodes_per_year_district = extract_results(
        results_folder,
        module='tlo.methods.schisto',
        key='schisto_mda_episodes_by_district',
        custom_generate_series=get_counts_of_mda_by_year_district,
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0)

# Extract district names from index level 1
districts = mda_episodes_per_year_district.index.get_level_values(1)

# Align scaling factors to the district index in df_schisto
scaling_factors = district_scaling_factor.loc[districts].iloc[:, 0].values

# Multiply df_schisto by scaling factors, broadcasting over rows
mda_episodes_per_year_district_scaled = mda_episodes_per_year_district.multiply(scaling_factors, axis=0)
# change index name for function sum_by_year_all_districts
mda_episodes_per_year_district_scaled.index = mda_episodes_per_year_district_scaled.index.set_names(
    ['year' if name == 'Year' else name for name in mda_episodes_per_year_district_scaled.index.names])

# assign costs - full including consumables
cons_cost_per_mda = 0.05  # assuming all children
cons_cost_per_mda_incl_adults = 0.081  # weighted mean across children and adults
prog_delivery_cost_per_mda = 1.27

full_cost_per_mda = prog_delivery_cost_per_mda + cons_cost_per_mda  # assuming all children
full_cost_per_mda_incl_adults = prog_delivery_cost_per_mda + cons_cost_per_mda_incl_adults


# === Costs incurred =========================================================

# --- Full costs ---
full_costs_per_year_district_child = mda_episodes_per_year_district_scaled * full_cost_per_mda
full_costs_per_year_district_adults = mda_episodes_per_year_district_scaled * full_cost_per_mda_incl_adults
full_costs_per_year_district = combine_on_keyword(full_costs_per_year_district_child,
                                                  full_costs_per_year_district_adults, keyword="MDA All")

full_costs_per_year_national = sum_by_year_all_districts(full_costs_per_year_district, TARGET_PERIOD)


# --- Program costs only ---
prog_costs_per_year_district = mda_episodes_per_year_district_scaled * prog_delivery_cost_per_mda
prog_costs_per_year_national = sum_by_year_all_districts(prog_costs_per_year_district, TARGET_PERIOD)

# sum across target period
prog_costs_per_year_national_sum = pd.DataFrame(prog_costs_per_year_national.sum(axis=0)).T
prog_costs_per_year_national_summary = compute_summary_statistics(prog_costs_per_year_national_sum,
                                                central_measure='mean')
fmt = format_summary_for_output(prog_costs_per_year_national_summary, filename='prog_costs_per_year_national')


# --- Cons costs only ---
cons_costs_per_year_district_child = mda_episodes_per_year_district_scaled * cons_cost_per_mda
cons_costs_per_year_district_adults = mda_episodes_per_year_district_scaled * cons_cost_per_mda_incl_adults
cons_costs_per_year_district = combine_on_keyword(cons_costs_per_year_district_child,
                                                  cons_costs_per_year_district_adults, keyword="MDA All")


cons_costs_per_year_national = sum_by_year_all_districts(cons_costs_per_year_district, TARGET_PERIOD)

# sum across target period
cons_costs_per_year_national_sum = pd.DataFrame(cons_costs_per_year_national.sum(axis=0)).T
cons_costs_per_year_national_summary = compute_summary_statistics(cons_costs_per_year_national_sum,
                                                central_measure='mean')
fmt = format_summary_for_output(cons_costs_per_year_national_summary, filename='cons_costs_per_year_national')


#################################################################################
# %% COSTS RELATIVE TO COMPARATOR
#################################################################################


# === Costs incurred relative to comparators NATIONAL =========================================================

full_costs_relative_noMDA = compute_number_averted_vs_noMDA_within_wash_strategies(
    full_costs_per_year_national,
    results_path=results_folder,
    filename_prefix='full_costs_per_year_national_compared_noMDA',
    target_period=TARGET_PERIOD,
    averted_or_incurred='incurred',
)
full_costs_relative_noMDA.to_excel(results_folder / f'full_costs_relative_noMDA{target_period()}.xlsx')

# incremental costs incurred - compare each prog in turn to the last one
incremental_full_costs_incurred_per_year_national = compute_stepwise_effects_by_wash_strategy(full_costs_per_year_national)
incremental_full_costs_incurred_per_year_national.to_excel(results_folder / f'incremental_full_costs_incurred_per_year_national{target_period()}.xlsx')


# --- Cons costs incurred relative to comparator ---

cons_costs_relative_noMDA = compute_number_averted_vs_noMDA_within_wash_strategies(
    cons_costs_per_year_national,
    results_path=results_folder,
    filename_prefix='cons_costs_per_year_national_compared_noMDA',
    target_period=TARGET_PERIOD,
    averted_or_incurred='incurred',
)
cons_costs_relative_noMDA.to_excel(results_folder / f'cons_costs_relative_noMDA{target_period()}.xlsx')


# incremental costs incurred - compare each prog in turn to the last one
incremental_cons_costs_incurred_per_year_national = compute_stepwise_effects_by_wash_strategy(cons_costs_per_year_national)
incremental_cons_costs_incurred_per_year_national.to_excel(results_folder / f'incremental_cons_costs_incurred_per_year_national{target_period()}.xlsx')


# === Costs incurred relative to comparators DISTRICT =========================================================

full_costs_relative_noMDA_district = compute_number_averted_vs_noMDA_within_wash_strategies(
    full_costs_per_year_district,
    results_path=results_folder,
    filename_prefix='full_costs_per_year_district_compared_noMDA',
    target_period=TARGET_PERIOD,
    averted_or_incurred='incurred',
)
full_costs_relative_noMDA_district.to_excel(results_folder / f'full_costs_relative_noMDA_district{target_period()}.xlsx')

full_costs_relative_SAC_district = compute_number_averted_vs_SAC_within_wash_strategies(
    full_costs_per_year_district,
    results_path=results_folder,
    filename_prefix='full_costs_per_year_district_compared_noMDA',
    target_period=TARGET_PERIOD,
    averted_or_incurred='incurred',
)
full_costs_relative_SAC_district.to_excel(results_folder / f'full_costs_relative_SAC_district{target_period()}.xlsx')

# incremental costs incurred - compare each prog in turn to the last one
incremental_full_costs_incurred_per_year_district = compute_stepwise_effects_by_wash_strategy(full_costs_per_year_district)
incremental_full_costs_incurred_per_year_district.to_excel(results_folder / f'incremental_full_costs_incurred_per_year_district{target_period()}.xlsx')

years = incremental_full_costs_incurred_per_year_district.index.get_level_values('year')
mask = (years >= TARGET_PERIOD[0].year) & (years <= TARGET_PERIOD[1].year)

sum_incremental_full_costs_incurred_district = (
    incremental_full_costs_incurred_per_year_district.loc[mask]
    .groupby('District')
    .sum()
)
sum_incremental_full_costs_incurred_district.to_excel(results_folder / f'sum_incremental_full_costs_incurred_district{target_period()}.xlsx')


# --- Cons costs incurred relative to comparator ---
cons_costs_relative_noMDA_district = compute_number_averted_vs_noMDA_within_wash_strategies(
    cons_costs_per_year_district,
    results_path=results_folder,
    filename_prefix='cons_costs_per_year_district_compared_noMDA',
    target_period=TARGET_PERIOD,
    averted_or_incurred='incurred',
)
cons_costs_relative_noMDA_district.to_excel(results_folder / f'cons_costs_relative_noMDA_district{target_period()}.xlsx')

cons_costs_relative_SAC_district = compute_number_averted_vs_SAC_within_wash_strategies(
    cons_costs_per_year_district,
    results_path=results_folder,
    filename_prefix='cons_costs_per_year_district_compared_noMDA',
    target_period=TARGET_PERIOD,
    averted_or_incurred='incurred',
)
cons_costs_relative_SAC_district.to_excel(results_folder / f'cons_costs_relative_SAC_district{target_period()}.xlsx')


# incremental costs incurred - compare each prog in turn to the last one
incremental_cons_costs_incurred_per_year_district = compute_stepwise_effects_by_wash_strategy(cons_costs_per_year_district)
incremental_cons_costs_incurred_per_year_district.to_excel(results_folder / f'incremental_cons_costs_incurred_per_year_district{target_period()}.xlsx')


years = incremental_cons_costs_incurred_per_year_district.index.get_level_values('year')
mask = (years >= TARGET_PERIOD[0].year) & (years <= TARGET_PERIOD[1].year)

sum_incremental_cons_costs_incurred_district = (
    incremental_cons_costs_incurred_per_year_district.loc[mask]
    .groupby('District')
    .sum()
)
sum_incremental_cons_costs_incurred_district.to_excel(results_folder / f'sum_incremental_cons_costs_incurred_district{target_period()}.xlsx')



#################################################################################
# %% ICERS
#################################################################################


icer_national = compute_icer_national(
    dalys_averted=incremental_dalys_averted_national,
    comparison_costs=incremental_full_costs_incurred_per_year_national,
    discount_rate_dalys=0.0,
    discount_rate_costs=0.0,
    return_summary=True
)

icer_national["formatted"] = icer_national.apply(
    lambda row: f"{row['mean']:.2f} ({row['lower']:.2f}–{row['upper']:.2f})", axis=1
)
icer_national.to_excel(results_folder / f'icer_national_{target_period()}.xlsx')

# --- ICERS for consumables costs only ---

icer_national_cons_only = compute_icer_national(
    dalys_averted=incremental_dalys_averted_national,
    comparison_costs=incremental_cons_costs_incurred_per_year_national,
    discount_rate_dalys=0.0,
    discount_rate_costs=0.0,
    return_summary=True
)

icer_national_cons_only["formatted"] = icer_national_cons_only.apply(
    lambda row: f"{row['mean']:.2f} ({row['lower']:.2f}–{row['upper']:.2f})", axis=1
)
icer_national_cons_only.to_excel(results_folder / f'icer_national_cons_only_{target_period()}.xlsx')


# --- ICERS district ---

icer_district = compute_icer_district(
    dalys_averted=incremental_dalys_averted_district,
    comparison_costs=incremental_full_costs_incurred_per_year_district,
    discount_rate_dalys=0.0,
    discount_rate_costs=0.0,
    return_summary=True
)

icer_district["formatted"] = icer_district.apply(
    lambda row: f"{row['mean']:.2f} ({row['lower']:.2f}–{row['upper']:.2f})", axis=1
)
icer_district.to_excel(results_folder / f'icer_district_{target_period()}.xlsx')

# --- ICERS district for consumables costs only ---

icer_district_cons_only = compute_icer_district(
    dalys_averted=incremental_dalys_averted_district,
    comparison_costs=incremental_cons_costs_incurred_per_year_district,
    discount_rate_dalys=0.0,
    discount_rate_costs=0.0,
    return_summary=True
)

icer_district_cons_only["formatted"] = icer_district_cons_only.apply(
    lambda row: f"{row['mean']:.2f} ({row['lower']:.2f}–{row['upper']:.2f})", axis=1
)
icer_district_cons_only.to_excel(results_folder / f'icer_district_cons_only_{target_period()}.xlsx')



#################################################################################
# %% NHB
#################################################################################

# todo the dalys averted and costs need to be compared to no MDA
def compute_nhb(
    dalys_averted: pd.DataFrame,
    comparison_costs: pd.DataFrame,
    discount_rate_dalys: float = 0.03,
    discount_rate_costs: float = 0.03,
    threshold: float = 150,
    return_summary: bool = True
) -> pd.DataFrame | pd.Series:
    """
    Compute Net Health Benefit (NHB) using DALYs averted and incremental costs by district over TARGET_PERIOD.

    Parameters:
    - dalys_averted: pd.DataFrame with MultiIndex (year, district), columns MultiIndex (run, draw)
    - comparison_costs: pd.DataFrame with same structure
    - discount_rate_dalys: annual discount rate for DALYs
    - discount_rate_costs: annual discount rate for costs
    - threshold: cost-effectiveness threshold ($ per DALY averted)
    - return_summary: if True, summarise across runs for each draw and district

    Returns:
    - pd.DataFrame or pd.Series with NHB values by draw and district
    """
    global TARGET_PERIOD
    start_year, end_year = TARGET_PERIOD

    # Restrict to target period
    mask = (dalys_averted.index.get_level_values(0) >= start_year.year) & (
        dalys_averted.index.get_level_values(0) <= end_year.year
    )
    dalys_period = dalys_averted.loc[mask]
    costs_period = comparison_costs.loc[mask]

    # Years since start for discounting
    years_since_start = dalys_period.index.get_level_values(0) - start_year.year

    # Discount weights
    discount_weights_dalys = 1 / ((1 + discount_rate_dalys) ** years_since_start)
    discount_weights_costs = 1 / ((1 + discount_rate_costs) ** years_since_start)

    # Apply discounting
    dalys_period = dalys_period.mul(discount_weights_dalys.values[:, None], axis=0)
    costs_period = costs_period.mul(discount_weights_costs.values[:, None], axis=0)

    # Sum over time, keeping districts
    dalys_summed = dalys_period.groupby(level=1).sum()
    costs_summed = costs_period.groupby(level=1).sum()

    # Compute NHB
    nhb = dalys_summed - (costs_summed / threshold)

    # Reshape from wide (columns: run, draw) to long
    nhb = nhb.stack(level=[0, 1]).rename("nhb").reset_index()
    nhb.columns = ['district', 'run', 'draw', 'nhb']

    if return_summary:
        def ci_bounds(x):
            mean = x.mean()
            se = x.std(ddof=1) / (len(x) ** 0.5)
            return pd.Series({
                'mean': mean,
                'lower': mean - 1.96 * se,
                'upper': mean + 1.96 * se
            })

        summary = nhb.groupby(['district', 'draw'])['nhb'].apply(ci_bounds)
        summary = summary.unstack().reset_index()
        summary.columns = ['district', 'draw', 'mean', 'lower', 'upper']
        return summary
    else:
        return nhb


nhb_district_vs_noMDA = compute_nhb(
    dalys_averted=dalys_averted_district_compared_noMDA,
    comparison_costs=cons_costs_relative_noMDA_district,
    discount_rate_dalys=0.0,
    threshold=120,
    discount_rate_costs=0.0,
    return_summary=True
)

nhb_district_vs_noMDA.to_csv(results_folder / f'nhb_district_vs_noMDA{target_period()}.csv')

nhb_district_vs_SAC = compute_nhb(
    dalys_averted=dalys_averted_district_compared_SAC,
    comparison_costs=cons_costs_relative_SAC_district,
    discount_rate_dalys=0.0,
    threshold=120,
    discount_rate_costs=0.0,
    return_summary=True
)

nhb_district_vs_SAC.to_csv(results_folder / f'nhb_district_vs_SAC{target_period()}.csv')

nhb_district_vs_SAC_full_costs = compute_nhb(
    dalys_averted=dalys_averted_district_compared_SAC,
    comparison_costs=full_costs_relative_SAC_district,
    discount_rate_dalys=0.0,
    threshold=120,
    discount_rate_costs=0.0,
    return_summary=True
)

nhb_district_vs_SAC_full_costs.to_csv(results_folder / f'nhb_district_vs_SAC_full_costs{target_period()}.csv')



# for each district under each WASH assumption, get the preferred strategy
# preferred strategy chosen by maximum NHB
# if nhb compared to SAC is negative, then return SAC as preferred strategy



# def get_best_draw_per_district(df, keyword):
#     """
#     For each district, return the draw containing the keyword with the highest mean.
#     If this best draw has a negative mean and the fallback draw '{keyword}, MDA SAC' is not present,
#     return a row with draw set to that fallback and mean/lower/upper as NaN.
#
#     Parameters:
#         df (pd.DataFrame): MultiIndex (district, draw) DataFrame with ['mean', 'lower', 'upper'].
#         keyword (str): Draw prefix, e.g., "Continue WASH", "Pause WASH", etc.
#
#     Returns:
#         pd.DataFrame: DataFrame with index=district and columns ['draw', 'mean', 'lower', 'upper'].
#     """
#     df_filtered = df[df.index.get_level_values("draw").str.contains(keyword)]
#     df_reset = df_filtered.reset_index()
#
#     # Get best draw (highest mean) per district
#     best_draws = (
#         df_reset
#         .sort_values("mean", ascending=False)
#         .groupby("district")
#         .first()
#         .loc[:, ["draw", "mean", "lower", "upper"]]
#     )
#
#     fallback_draw = f"{keyword}, MDA SAC"
#
#     # Handle districts where best draw is negative
#     for district in best_draws.index[best_draws["mean"] < 0]:
#         try:
#             # Check if fallback draw exists
#             fallback_row = df.loc[(district, fallback_draw)]
#             best_draws.loc[district] = {
#                 "draw": fallback_draw,
#                 "mean": fallback_row["mean"],
#                 "lower": fallback_row["lower"],
#                 "upper": fallback_row["upper"],
#             }
#         except KeyError:
#             # If fallback draw is missing, assign NaNs but retain fallback draw name
#             best_draws.loc[district] = {
#                 "draw": fallback_draw,
#                 "mean": np.nan,
#                 "lower": np.nan,
#                 "upper": np.nan,
#             }
#
#     return best_draws


def get_best_draw_per_district(df, keyword):
    """
    For each district, return the draw containing the keyword with the highest mean.
    If this best draw has a negative mean and the fallback draw '{keyword}, MDA SAC' is not present,
    return a row with draw set to that fallback and mean/lower/upper as NaN.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['district', 'draw', 'mean', 'lower', 'upper'].
        keyword (str): Draw prefix, e.g., "Continue WASH", "Pause WASH", etc.

    Returns:
        pd.DataFrame: DataFrame indexed by district with columns ['draw', 'mean', 'lower', 'upper'].
    """
    # Filter rows where 'draw' contains keyword
    df_filtered = df[df['draw'].str.contains(keyword, na=False)].copy()

    # Get best draw (highest mean) per district
    best_draws = (
        df_filtered
        .sort_values('mean', ascending=False)
        .groupby('district')
        .first()
        .loc[:, ['draw', 'mean', 'lower', 'upper']]
        .copy()
    )

    fallback_draw = f"{keyword}, MDA SAC"

    # For districts where best mean < 0, try to replace with fallback
    for district in best_draws.index[best_draws['mean'] < 0]:
        fallback_row = df[(df['district'] == district) & (df['draw'] == fallback_draw)]
        if not fallback_row.empty:
            best_draws.loc[district] = fallback_row.iloc[0][['draw', 'mean', 'lower', 'upper']]
        else:
            best_draws.loc[district] = [fallback_draw, np.nan, np.nan, np.nan]

    return best_draws


# Apply to each scenario
pause_wash_best_strategy = get_best_draw_per_district(nhb_district_vs_SAC, "Pause WASH")
continue_wash_best_strategy = get_best_draw_per_district(nhb_district_vs_SAC, "Continue WASH")
scaleup_wash_best_strategy = get_best_draw_per_district(nhb_district_vs_SAC, "Scale-up WASH")
# todo all districts except 2 return PSAC (1 for pause WASH)

pause_wash_best_strategy.to_csv(results_folder / f'pause_wash_best_strategy{target_period()}.csv')
continue_wash_best_strategy.to_csv(results_folder / f'continue_wash_best_strategy{target_period()}.csv')
scaleup_wash_best_strategy.to_csv(results_folder / f'scaleup_wash_best_strategy{target_period()}.csv')


pause_wash_best_full_costs = get_best_draw_per_district(nhb_district_vs_SAC_full_costs, "Pause WASH")
# todo all districts return no MDA with full costs

