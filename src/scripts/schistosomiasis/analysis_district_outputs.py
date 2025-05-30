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



def compute_number_averted_within_wash_strategies(
    df: pd.DataFrame,
    wash_strategies: tuple = ("Pause WASH", "Continue WASH", "Scale-up WASH"),
    results_path: Path = None,
    filename_prefix: str = 'dalys_averted_by_year_run_district',
    target_period: tuple = None
) -> pd.DataFrame:
    """
    Computes value (dalys or number py) averted by comparing each WASH strategy's MDA scenarios
    to the corresponding 'no MDA' baseline within the same strategy group.
    """
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
        diff_df = -1.0 * find_difference_relative_to_comparison_dataframe(df_subset, comparison=comparator_draw)

        # Select only the relevant MDA draws
        comparator_results.append(diff_df)

    # Concatenate results across all strategies
    combined_df = pd.concat(comparator_results, axis=1)

    if results_path:
        period_str = f"_{target_period[0].year}-{target_period[1].year}" if target_period else ""
        output_file = results_path / f"{filename_prefix}{period_str}.xlsx"
        combined_df.to_excel(output_file)

    return combined_df


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

prev_mansoni_H_All_district.to_excel(results_folder / (f'prev_mansoni_H_year_district {target_period()}.xslx'))


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


###########################################################################################################
# get DALYs by district
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

# format into nice table for output
tmp = dalys_national_summary / 1_000_000

# Rearrange into long format: rows = (district, age_group, draw), columns = stat
df_long = tmp.stack(level='draw')  # columns now: lower, central, upper
formatted = df_long.apply(
    lambda row: f"{row['central']:.2f} ({row['lower']:.2f}–{row['upper']:.2f})", axis=1
)
# Return to wide format: rows = (district, age_group), columns = draw
formatted_df = formatted.unstack(level=-1)

output_path = results_folder / f'table_summary_dalys_national_{target_period()}.xlsx'
formatted_df.to_excel(output_path)


# === DALYs averted by district =========================================================

schisto_dalys_combined = compute_number_averted_within_wash_strategies(
    dalys_schisto_district_scaled,
    results_path=results_folder,
    filename_prefix='schisto_dalys_averted_by_year_run_district',
    target_period=TARGET_PERIOD
)


# --- Incremental DALYs averted by district ---

# incremental dalys averted - compare each prog in turn to the last one
comparison_df = -1 * compute_stepwise_effects_by_wash_strategy(dalys_schisto_district_scaled)
comparison_df.to_excel(results_folder / f'stepwise_dalys_averted_year_district{target_period()}.xlsx')







##########################
# get mda episodes costs by district / year / run


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

# assign costs - full including consumables
unit_cost_per_mda_incl_cons = 2.26

costs_mda_episodes_per_year_district_scaled = mda_episodes_per_year_district_scaled * unit_cost_per_mda_incl_cons

# costs incurred

# calculate the costs averted from non-cons costs, i.e. HRH, implementation etc
costs_district_vs_PauseWASH = find_difference_relative_to_comparison_dataframe(
        costs_mda_episodes_per_year_district_scaled,
        comparison='Pause WASH, no MDA'
    )
costs_district_vs_PauseWASH.to_csv(results_folder / f'costs_district_vs_PauseWASH{target_period()}.csv')

costs_district_vs_ContinueWASH = find_difference_relative_to_comparison_dataframe(
        costs_mda_episodes_per_year_district_scaled,
        comparison='Continue WASH, no MDA'
    )
costs_district_vs_ContinueWASH.to_csv(results_folder / f'costs_district_vs_ContinueWASH{target_period()}.csv')

costs_district_vs_scaleupWASH = find_difference_relative_to_comparison_dataframe(
        costs_mda_episodes_per_year_district_scaled,
        comparison='Scale-up WASH, no MDA'
    )
costs_district_vs_scaleupWASH.to_csv(results_folder / f'costs_district_vs_scaleupWASH{target_period()}.csv')

# Select desired columns from each dataframe
df1_sel = select_draws_by_keyword(costs_district_vs_PauseWASH, 'Pause')
df2_sel = select_draws_by_keyword(costs_district_vs_ContinueWASH, 'Continue')
df3_sel = select_draws_by_keyword(costs_district_vs_scaleupWASH, 'Scale-up')

# Concatenate the selected columns horizontally
costs_incurred_by_district_year_run_combined = pd.concat([df1_sel, df2_sel, df3_sel], axis=1)
costs_incurred_by_district_year_run_combined.to_csv(results_folder / f'costs_incurred_by_district_year_run_combined{target_period()}.csv')


##########################
# calculate ICER

def compute_icer(
    dalys_averted: pd.DataFrame,
    comparison_costs: pd.DataFrame,
    discount_rate_dalys: float = 0.0,
    discount_rate_costs: float = 0.0,
    return_summary: bool = True
) -> pd.DataFrame | pd.Series:
    """
    Compute ICERs comparing costs and DALYs averted over a TARGET_PERIOD by district, run, and draw.

    Assumes:
    - Row MultiIndex: (year, district)
    - Column MultiIndex: (draw, run)
    - TARGET_PERIOD is a global tuple of (start_year, end_year) as datetime or int
    """
    global TARGET_PERIOD
    start_year, end_year = TARGET_PERIOD

    # Filter index by year (first level)
    years = dalys_averted.index.get_level_values(0)
    mask = (years >= start_year.year) & (years <= end_year.year)
    dalys_period = dalys_averted.loc[mask]
    costs_period = comparison_costs.loc[mask]

    # Extract years for discounting from index level 0
    years_since_start = years[mask] - start_year.year

    # Discount weights per year
    discount_weights_dalys = 1 / ((1 + discount_rate_dalys) ** years_since_start)
    discount_weights_costs = 1 / ((1 + discount_rate_costs) ** years_since_start)

    # Apply discounting only if rates are nonzero
    if discount_rate_dalys != 0.0:
        # Discount dalys_period by multiplying rows by discount weights
        dalys_period = dalys_period.mul(discount_weights_dalys.values, axis=0)

    if discount_rate_costs != 0.0:
        costs_period = costs_period.mul(discount_weights_costs.values, axis=0)

    # Sum over years **within each district**
    # Group by district (level 1 of index)
    total_dalys = dalys_period.groupby(level=1).sum()
    total_costs = costs_period.groupby(level=1).sum()

    # Compute ICER = total_costs / total_dalys for each district, run, draw
    icers = total_costs / total_dalys

    # Rearrange to long form DataFrame with columns: district, draw, run, icer
    icers_long = (
        icers
        .stack([0, 1])  # stack draw, run columns to index
        .rename('icer')
        .reset_index()   # columns: district, draw, run, icer
    )

    if return_summary:
        # Summarise ICER across runs for each district and draw
        summary = (
            icers_long
            .groupby(['level_0', 'draw'])['icer']
            .agg(
                mean='mean',
                lower=lambda x: np.quantile(x, 0.025),
                upper=lambda x: np.quantile(x, 0.975)
            )
            .reset_index()
        )
        return summary
    else:
        # Return all ICERs (district, draw, run, icer)
        return icers


icer_district = compute_icer(
    dalys_averted=schisto_dalys_averted_by_year_run_district_combined,
    comparison_costs=costs_incurred_by_district_year_run_combined,
    discount_rate_dalys=0.0,
    discount_rate_costs=0.0,
    return_summary=True
)
icer_district.to_csv(results_folder / f'icer_district_{target_period()}.csv')


def plot_icer_three_panels(df, context="Continue_WASH"):
    """
    Plot ICER by district for three categories ('MDA SAC', 'MDA PSAC', 'MDA All')
    Only draws containing 'Continue WASH' are included.
    """
    # Filter draws containing 'Continue WASH'
    df_filtered = df[df['draw'].str.contains(context, na=False)]

    categories = ['MDA SAC', 'MDA PSAC', 'MDA All']
    titles = {
        'MDA SAC': f'{context} MDA SAC',
        'MDA PSAC': 'MDA PSAC',
        'MDA All': 'MDA All'
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharey=True)

    for ax, category in zip(axes, categories):
        subset = df_filtered[df_filtered['draw'].str.contains(category, na=False)]
        if subset.empty:
            ax.text(0.5, 0.5, f'No data for {category}', ha='center', va='center')
            ax.set_title(titles[category])
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Sort districts alphabetically to keep consistent order
        subset = subset.sort_values('level_0')

        # Plot points
        sns.pointplot(
            data=subset,
            x='level_0',
            y='mean',
            join=False,
            color='blue',
            ax=ax
        )

        # Add error bars manually
        x_vals = range(len(subset))
        y_vals = subset['mean'].values
        y_err_lower = y_vals - subset['lower'].values
        y_err_upper = subset['upper'].values - y_vals

        ax.errorbar(
            x=x_vals,
            y=y_vals,
            yerr=[y_err_lower, y_err_upper],
            fmt='none',
            ecolor='blue',
            elinewidth=1,
            capsize=3,
            alpha=0.7
        )

        ax.axhline(500, color='grey', linestyle='--', linewidth=1)
        ax.set_ylim(0, None)
        ax.set_title(titles[category])

        # Show x-axis labels only on the bottom plot (last subplot)
        if category != 'MDA All':
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('District')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        ax.set_ylabel('ICER')

    plt.tight_layout()
    plt.show()


plot_icer_three_panels(icer_district, context='Continue WASH')

plot_icer_three_panels(icer_district, context='Scale-up WASH')

########################
# NHB


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
        summary = (
            nhb.groupby(['district', 'draw'])['nhb']
            .agg(mean='mean', lower=lambda x: np.quantile(x, 0.025), upper=lambda x: np.quantile(x, 0.975))
        )
        return summary
    else:
        return nhb



nhb_district = compute_nhb(
    dalys_averted=schisto_dalys_averted_by_year_run_district_combined,
    comparison_costs=costs_incurred_by_district_year_run_combined,
    discount_rate_dalys=0.0,
    threshold=150,
    discount_rate_costs=0.0,
    return_summary=True
)

nhb_district.to_csv(results_folder / f'nhb_district{target_period()}.csv')


#############################################################################
# Kaplan-Meier proportion districts reaching EPHP each year

def extract_mda_label(draw_name: str) -> str:
    # Define possible MDA labels to look for in draw name
    mda_labels = ["no MDA", "MDA SAC", "MDA PSAC", "MDA All"]
    for label in mda_labels:
        if label.replace(" ", "").lower() in draw_name.replace(" ", "").lower():
            return label
    # Default if none matched
    return "Other"


def plot_ephp_km_panels(
    df: pd.DataFrame,
    threshold: float = 0.015,
    year_range: tuple = (2024, 2040),
    alpha: float = 1.0,
    figsize: tuple = (8, 12)
):
    """
    Plot Kaplan-Meier-style curves in three vertically stacked panels showing the proportion of districts
    reaching prevalence < threshold by year. Panels are grouped by draw naming patterns: 'Pause', 'Continue', 'Scale-up'.

    Parameters:
        df : pd.DataFrame
            DataFrame with MultiIndex (year, district) and columns with MultiIndex (draw, run)
        threshold : float
            Prevalence threshold for defining EPHP
        year_range : tuple
            Range of years to display on x-axis
        alpha : float
            Transparency for individual draw lines
        figsize : tuple
            Size of the overall figure
    """

    def extract_mda_label(draw_name: str) -> str:
        """Extract MDA category for legend from draw name."""
        mda_labels = ["no MDA", "MDA SAC", "MDA PSAC", "MDA All"]
        draw_lower = draw_name.replace(" ", "").lower()
        for label in mda_labels:
            if label.replace(" ", "").lower() in draw_lower:
                return label
        return "Other"

    # Remove pre-2024 data
    df = df.loc[df.index.get_level_values("year") >= 2024]

    # Step 1: mean across runs for each draw
    df_mean_runs = df.groupby(axis=1, level="draw").mean()

    # Step 2: identify years where prevalence < threshold
    below = (df_mean_runs < threshold).reset_index()
    long_format = below.melt(id_vars=["year", "district"], var_name="draw", value_name="below_threshold")
    below_threshold = long_format[long_format["below_threshold"]]

    # First year each district reaches threshold, by draw
    first_years = below_threshold.groupby(["district", "draw"])["year"].min().reset_index(name="year_ephp")

    # Setup for panel plots
    draw_filters = {
        "Pause": "Pause",
        "Continue": "Continue",
        "Scale-up": "Scale-up"
    }

    total_districts = df.index.get_level_values("district").nunique()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True, sharey=True)

    # Use darker palette, enough colors for up to 20 draws per panel
    palette = sns.color_palette("dark", n_colors=20)

    for ax, (title, substr) in zip(axes, draw_filters.items()):
        # Filter draws by name substring
        filtered = first_years[first_years["draw"].str.contains(substr)]

        # Count cumulative districts reaching EPHP by year
        ephp_counts = (
            filtered.groupby(["draw", "year_ephp"])
            .size()
            .groupby(level=0)
            .cumsum()
            .reset_index(name="num_districts")
        )
        ephp_counts["prop_districts"] = ephp_counts["num_districts"] / total_districts
        ephp_counts = ephp_counts[ephp_counts["year_ephp"].between(*year_range)]

        # Prepare unique draws and assign colours
        draw_list = ephp_counts["draw"].unique()
        color_dict = dict(zip(draw_list, palette[:len(draw_list)]))

        # To avoid duplicate legend labels for same MDA category
        plotted_labels = set()

        for draw, data in ephp_counts.groupby("draw"):
            label = extract_mda_label(draw)
            if label not in plotted_labels:
                plot_label = label
                plotted_labels.add(label)
            else:
                plot_label = None  # Don't repeat label in legend

            ax.step(
                data["year_ephp"],
                data["prop_districts"],
                where="post",
                label=plot_label,
                color=color_dict[draw],
                alpha=alpha,
                linewidth=1.5,
            )

        ax.set_title(title)
        ax.set_ylabel("Proportion < {:.1f}%".format(threshold * 100))
        ax.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.15)
        ax.legend(loc="upper left", fontsize="small", title="")

    axes[-1].set_xlabel("Year")
    plt.suptitle("Progress Toward EPHP by Year and Strategy", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


plot_ephp_km_panels(prev_haem_H_All_district)
plot_ephp_km_panels(prev_mansoni_H_All_district)
