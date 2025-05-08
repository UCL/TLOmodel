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


#################################################################################
# %% DISTRICT FUNCTIONS
#################################################################################


def get_district_prevalence(_df):
    """Get the prevalence for every district """
    df = _df.copy()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to datetime
        df = df.set_index('date')  # Set date as index
    else:
        # If 'date' is already index, ensure it's a datetime type
        df.index = pd.to_datetime(df.index)

    # _df.index = pd.to_datetime(_df.index)
    _df_year = df[df.index.year == year]  # Filter by year

    _df_year.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split('|')) for col in _df_year.columns],
        names=['infection_status', 'district_of_residence', 'age_years']
    )

    # limit to relevant age-groups
    if age_group == 'SAC':
        age_group_filter = ['SAC']
    elif age_group == 'PSAC':
        age_group_filter = ['PSAC', 'SAC']  # Include both PSAC and SAC
    elif age_group == 'Adult':
        age_group_filter = ['Adults']
    elif age_group == 'Infant':
        age_group_filter = ['Infant']
    else:
        age_group_filter = ['Adults', 'Infant', 'PSAC', 'SAC']  # Include all age groups for 'All'

    selected_columns = [
        col for col in _df_year.columns
        if any(age in col[2] for age in age_group_filter)
    ]
    df_total = _df_year[selected_columns]
    district_sums = df_total.groupby(axis=1, level='district_of_residence').sum()

    # Set infection status filter
    infection_filter = []
    if 'High-infection' in infection_types:
        infection_filter.append('High-infection')
    if 'Moderate-infection' in infection_types:
        infection_filter.append('Moderate-infection')
    if 'Low-infection' in infection_types:
        infection_filter.append('Low-infection')

    selected_columns = [
        col for col in _df_year.columns
        if any(inf_status in col[0] for inf_status in infection_filter) and
           any(age in col[2] for age in age_group_filter)
    ]
    df_infected = _df_year[selected_columns]
    infected_numerator = df_infected.groupby('district_of_residence', axis=1).sum()

    proportion_infected = infected_numerator.div(district_sums).iloc[0]

    return proportion_infected


def extract_district_prevalence() -> pd.DataFrame:
    """ for each run/draw combination, extract the prevalence by district
    using the custom arguments for age-group and infection status
    """

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)
    module = 'tlo.methods.schisto'

    # Collect results from each draw/run
    res = dict()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            draw_run = (draw, run)

            try:
                _df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                output_from_eval: pd.Series = get_district_prevalence(_df)
                assert isinstance(output_from_eval, pd.Series), (
                    'Custom command does not generate a pd.Series'
                )

                res[draw_run] = output_from_eval

            except KeyError:
                # Some logs could not be found - probably because this run failed.
                res[draw_run] = None

    # Use pd.concat to compile results (skips dict items where the values is None)
    _concat = pd.concat(res, axis=1)
    _concat.columns.names = ['draw', 'run']  # name the levels of the columns multi-index
    return _concat


def analyse_and_plot_schisto_prevalence_change(keys, year_start, year_end, age_group, infection_types):
    for key in keys:
        print(f"Processing: {key}")

        # Set global key
        globals()['key'] = key  # Required because extract_district_prevalence uses it

        # Extract data for start year
        globals()['year'] = year_start
        prev_start = extract_district_prevalence()
        median_prev_start = prev_start.groupby('draw', axis=1).median()

        # Extract data for end year
        globals()['year'] = year_end
        prev_end = extract_district_prevalence()
        median_prev_end = prev_end.groupby('draw', axis=1).median()

        # Calculate percentage change
        percentage_change = ((median_prev_end - median_prev_start) / median_prev_start) * 100
        percentage_change.index = percentage_change.index.str.replace('district_of_residence=', '', case=False)
        percentage_change.columns = param_names

        # Plot full heatmap
        plot_percentage_change_heatmap(
            df=percentage_change,
            title=f"Percentage Change in Prevalence ({key}, {year_start} to {year_end})",
            filename_suffix=f"{key}_ALL_SCENARIOS"
        )

        # Filtered subset for specific scenarios
        ordered_draws = [
            'Pause WASH, no MDA',
            'Continue WASH, no MDA',
            'Continue WASH, MDA SAC',
            'Continue WASH, MDA PSAC',
            'Continue WASH, MDA All',
            'Scale-up WASH, no MDA',
            'Scale-up WASH, MDA SAC',
            'Scale-up WASH, MDA PSAC',
            'Scale-up WASH, MDA All'
        ]
        filtered_cols = [col for col in ordered_draws if col in percentage_change.columns]
        percentage_change_subset = percentage_change[filtered_cols]

        # Plot filtered heatmap
        plot_percentage_change_heatmap(
            df=percentage_change_subset,
            title=f"Percentage Change in Prevalence ({key}, {year_start} to {year_end})",
            filename_suffix=f"{key}"
        )

        # Export to Excel
        export_to_excel(
            filename=f"prevalence_HML_{age_group}_{key} {target_period()}.xlsx",
            prev_start=median_prev_start,
            prev_end=median_prev_end,
            change_df=percentage_change
        )


def plot_percentage_change_heatmap(df, title, filename_suffix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df, xticklabels=df.columns, yticklabels=df.index,
        annot=False, cmap='coolwarm', linewidths=0.5,  # coolwarm
        cbar_kws={'label': 'Percentage Change (%)'}
    )
    plt.title(title, fontsize=16)
    plt.ylabel('', fontsize=14)
    plt.xlabel('', fontsize=14)
    plt.tight_layout()
    name_of_plot = f'{title} {target_period()}'
    plt.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    plt.show()


def export_to_excel(filename, prev_start, prev_end, change_df):
    file_path = results_folder / filename
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        prev_start.to_excel(writer, sheet_name='Prev_Start', index=True)
        prev_end.to_excel(writer, sheet_name='Prev_End', index=True)
        change_df.to_excel(writer, sheet_name='Percentage_Change', index=True)


# call plots

infection_keys = ['infection_status_haematobium', 'infection_status_mansoni']
year_start = 2023
year_end = 2040
age_group = 'All'
infection_types = ['High-infection', 'Moderate-infection', 'Low-infection']

analyse_and_plot_schisto_prevalence_change(
    keys=infection_keys,
    year_start=year_start,
    year_end=year_end,
    age_group=age_group,
    infection_types=infection_types
)




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

result.to_csv(results_folder / (f'prevalence_any_infection_all_ages_district{target_period()}.csv'))

# summarise the prevalence for each district by draw
median_by_draw_district = result.groupby(level=['draw', 'district'], axis=1).median()
median_by_draw_district.to_csv(results_folder / (f'median_prevalence_any_infection_all_ages_district{target_period()}.csv'))





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
        do_scaling=False,  # todo need to scale
    ).pipe(set_param_names_as_column_index_level_0)


# need to multiply by district-level scaling factor
scaled_py_district = py_district.mul(district_scaling_factor['scaling_factor'], axis=0)

pc_py_averted_district_vs_scaleup_WASH = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        scaled_py_district,
        comparison='Scale-up WASH, no MDA',
        scaled=True
    ),
    central_measure='median'
)
pc_py_averted_district_vs_scaleup_WASH.to_csv(results_folder / f'pc_py_averted_district_vs_scaleup_WASH{target_period()}.csv')


pc_py_averted_district_vs_baseline = 100.0 * compute_summary_statistics(
    -1.0 * find_difference_relative_to_comparison_dataframe(
        scaled_py_district,
        comparison='Continue WASH, no MDA',
        scaled=True
    ),
    central_measure='median'
)
pc_py_averted_district_vs_baseline.to_csv(results_folder / f'pc_py_averted_district_vs_baseline{target_period()}.csv')


# heatmap - vs WASH scale-up
data_to_plot = pc_py_averted_district_vs_scaleup_WASH.xs('central', level='stat', axis=1)
data_to_plot = data_to_plot.loc[:, ~data_to_plot.columns.get_level_values(0).str.startswith('Pause') |
                            (data_to_plot.columns.get_level_values(0) == 'Pause WASH, no MDA')]


title = f"Percentage change in person-years infected vs WASH scale-up \n any species, all ages {target_period()}"
plot_percentage_change_heatmap(data_to_plot, title, filename_suffix="PY_")


#heatmap - vs BASELINE
data_to_plot = pc_py_averted_district_vs_baseline.xs('central', level='stat', axis=1)
data_to_plot = data_to_plot.loc[:, ~data_to_plot.columns.get_level_values(0).str.startswith('Pause') |
                            (data_to_plot.columns.get_level_values(0) == 'Pause WASH, no MDA')]


title = f"Percentage change in person-years infected vs continued WASH improvements \n any species, all ages {target_period()}"
plot_percentage_change_heatmap(data_to_plot, title, filename_suffix="PY_")





###########################################################################################################
