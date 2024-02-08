"""
Read in the output files generated by analysis_scenarios
generate life tables to estimate life expectancy for each run/draw
produce summary statistics
"""

import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

from tlo.analysis.utils import (
    extract_results,
    get_scenario_info,
    load_pickled_dataframes,
    summarize,
)


# HELPER FUNCTIONS
def _map_age_to_age_group(_df):
    """
    Maps ages to age-groups in 5-year intervals and adds a new column 'age-group' to the DataFrame.

    Args:
    - dataframe (pd.DataFrame): The DataFrame containing the age data.

    Returns:
    - pd.DataFrame: The DataFrame with the 'age-group' column added.
    """
    # Define age groups in 5-year intervals
    age_groups = ['0'] + ['1-4'] + [f'{start}-{start + 4}' for start in range(5, 90, 5)] + ['90']

    # Create a new column 'age-group' based on the age-to-age-group mapping
    _df['age_group'] = pd.cut(_df['age'], bins=[0] + [1] + list(range(5, 95, 5)) + [float('inf')],
                              labels=age_groups, right=False)

    return pd.Series(_df['age_group'])

def _extract_person_years(results_folder, _draw, _run):
    """Helper function to get the multiplier from the simulation
    Note that if the scaling factor cannot be found a `KeyError` is thrown."""
    return load_pickled_dataframes(
        results_folder, _draw, _run, 'tlo.methods.demography'
    )['tlo.methods.demography']['person_years']


def _create_multi_index_columns(results_folder: Path):
    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    return pd.MultiIndex.from_product([range(info['number_of_draws']),
                                       range(info['runs_per_draw'])],
                                      names=['draw', 'run'])


def _num_deaths_by_age_group(results_folder, target_period):
    """ produces dataframe with mean (+ 95% UI) number of deaths
    for each draw by age-group
    dataframe returned: rows=age-gp, columns=draw median, draw lower, draw upper
    """

    def extract_deaths_by_age_group(df: pd.DataFrame) -> pd.Series:
        # Call the function to add the 'age-group' column
        age_group = _map_age_to_age_group(df)
        return df.loc[pd.to_datetime(df.date).dt.date.between(*target_period, inclusive='both')].groupby([age_group, df["sex"]]).size()

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_by_age_group,
        do_scaling=False
    )


def _aggregate_person_years_by_age(results_folder, target_period):
    """ extract person-years for each draw/run
    calculate for men and women separately
    return a dataframe with index=age-groups and columns=person-years
    """

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    # Create an empty DataFrame to store all outputs
    output = pd.DataFrame()

    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            _df = _extract_person_years(results_folder, _draw=draw, _run=run)

            # create empty dataframe to store outputs from each run
            tmp = pd.DataFrame(columns=['M', 'F'])

            for sex in ['M', 'F']:
                py = _df[sex]  # extract values for each sex
                # create dataframe one row per year and one column per age_year
                new_df = pd.DataFrame(py.tolist())
                dates = _df.date.dt.date
                new_df = new_df.loc[dates.between(*target_period, inclusive='both')]

                # sum values for each age (single years)
                py_by_single_age_years = new_df.sum(numeric_only=True, axis=0).reset_index()
                py_by_single_age_years = py_by_single_age_years.rename(columns={'index': 'age', 0: 'person_years'})

                # convert single age years to float for mapping
                py_by_single_age_years['age'] = py_by_single_age_years['age'].astype(float)
                # map single age bands to age-groups
                py_by_single_age_years['age_group'] = _map_age_to_age_group(py_by_single_age_years)

                summary = py_by_single_age_years.groupby(["age_group"])["person_years"].sum()
                tmp[sex] = summary

            # then join each draw/run in a new column
            output = pd.concat([output, pd.concat([tmp['M'], tmp['F']], ignore_index=True)], axis=1)

    # Create a MultiIndex for rows using age group and 'Male' and 'Female'
    multi_index_rows_male = pd.MultiIndex.from_product([summary.index, ['M']], names=['age_group', 'sex'])
    multi_index_rows_female = pd.MultiIndex.from_product([summary.index, ['F']], names=['age_group', 'sex'])
    output.index = multi_index_rows_male.append(multi_index_rows_female)

    # multi-index columns
    multi_index_columns = _create_multi_index_columns(results_folder)
    output.columns = multi_index_columns

    return output


def _estimate_life_expectancy(_person_years_at_risk, _number_of_deaths_in_interval):
    """
    for a single run, estimate life expectancy for males and females
    return: pd.Series
    """

    estimated_life_expectancy_at_birth = dict()

    # first age-group is 0, then 1-4, 5-9, 10-14 etc. 22 categories in total
    level_0_values = _person_years_at_risk.index.get_level_values('age_group').unique()

    # Extract interval width
    interval_width = [5 if '90' in interval else int(interval.split('-')[1]) - int(interval.split('-')[0]) + 1
    if '-' in interval else 1 for interval in level_0_values.categories]

    number_age_groups = len(interval_width)
    fraction_of_last_age_survived = pd.Series([0.5] * number_age_groups, index=level_0_values)

    # separate male and female data
    for sex in ['M', 'F']:
        person_years_by_sex = _person_years_at_risk.xs(key=sex, level='sex')
        number_of_deaths_by_sex = _number_of_deaths_in_interval.xs(key=sex, level='sex')

        death_rate_in_interval = number_of_deaths_by_sex / person_years_by_sex
        # if no deaths or person-years, produces nan
        death_rate_in_interval = death_rate_in_interval.fillna(0)
        # if no deaths in age 90+, set death rate equal to value in age 85-89
        if death_rate_in_interval.loc['90'] == 0:
            death_rate_in_interval.loc['90'] = death_rate_in_interval.loc['85-89']

        # Calculate the probability of dying in the interval
        # condition checks whether the observed number deaths is significantly higher than would be expected
        # based on population years at risk and survival fraction
        # if true, suggests very high mortality rates and returns value 1
        condition = number_of_deaths_by_sex > (
            person_years_by_sex / interval_width / fraction_of_last_age_survived)
        probability_of_dying_in_interval = pd.Series(index=number_of_deaths_by_sex.index, dtype=float)
        probability_of_dying_in_interval[condition] = 1
        probability_of_dying_in_interval[~condition] = interval_width * death_rate_in_interval / (
            1 + interval_width * (1 - fraction_of_last_age_survived) * death_rate_in_interval)
        # all those surviving to final interval die during this interval
        probability_of_dying_in_interval.loc['90'] = 1

        # number_alive_at_start_of_interval
        # keep dtype as float in case using aggregated outputs
        # note range stops BEFORE the specified number
        number_alive_at_start_of_interval = pd.Series(index=range(number_age_groups), dtype=float)
        number_alive_at_start_of_interval[0] = 100_000  # hypothetical cohort
        for i in range(1, number_age_groups):
            number_alive_at_start_of_interval[i] = (1 - probability_of_dying_in_interval[i - 1]) * \
                                                   number_alive_at_start_of_interval[i - 1]

        # number_dying_in_interval
        number_dying_in_interval = pd.Series(index=range(number_age_groups), dtype=float)
        for i in range(0, number_age_groups - 1):
            number_dying_in_interval[i] = number_alive_at_start_of_interval[i] - number_alive_at_start_of_interval[
                i + 1]

        number_dying_in_interval[number_age_groups - 1] = number_alive_at_start_of_interval[number_age_groups - 1]

        # person-years lived in interval
        py_lived_in_interval = pd.Series(index=range(number_age_groups), dtype=float)
        for i in range(0, number_age_groups - 1):
            py_lived_in_interval[i] = interval_width[i] * (
                number_alive_at_start_of_interval[i + 1] + fraction_of_last_age_survived[i] * number_dying_in_interval[
                i])
        py_lived_in_interval[number_age_groups - 1] = number_alive_at_start_of_interval[number_age_groups - 1] / \
                                                      death_rate_in_interval[number_age_groups - 1]

        # person-years lived beyond start of interval
        # have to iterate backwards for this
        py_lived_beyond_start_of_interval = pd.Series(index=range(number_age_groups), dtype=float)
        py_lived_beyond_start_of_interval[number_age_groups - 1] = py_lived_in_interval[number_age_groups - 1]
        for i in range((number_age_groups - 2), -1, -1):
            py_lived_beyond_start_of_interval[i] = py_lived_beyond_start_of_interval[i + 1] + py_lived_in_interval[i]

        # calculate observed life expectancy at start of interval
        # if number of people alive at start of interval=0, condition returns true and observed life expectancy=0
        condition = number_alive_at_start_of_interval == 0
        observed_life_expectancy = pd.Series(index=range(number_age_groups), dtype=float)
        observed_life_expectancy[condition] = 0
        observed_life_expectancy[~condition] = py_lived_beyond_start_of_interval / number_alive_at_start_of_interval

        # estimated life expectancy from birth
        estimated_life_expectancy_at_birth[sex] = observed_life_expectancy[0]

    return estimated_life_expectancy_at_birth


def get_life_expectancy_estimates(
    results_folder: Path,
    target_period: Tuple[datetime.date, datetime.date],
    summary: bool = True
) -> pd.DataFrame:
    """
    produces sets of life expectancy estimates for each draw/run
    calls:
    *1 _num_deaths_by_age_group
    *2 _aggregate_person_years_by_age

    Args:
    - results_folder (PosixPath): The path to the results folder containing log, `tlo.methods.demography`
    - target period (tuple of dates): declare the date range (inclusively) in which life expectancy is to be estimated
    - summary (bool): declare whether to return a summarized value (mean with 95% uncertainty intervals)
        or return the estimate for each draw/run

    Returns:
    - pd.DataFrame: The DataFrame with the life expectancy estimates (in years)
     for every draw/run in the results folder; or, with option `summary=True` summarized (central, lower,
     upper estimates) for each draw.

    example use:
    test = produce_life_expectancy_estimates(results_folder, median=True,
        target_period=(Date(2019, 1, 1), Date(2020, 1, 1)))

    """

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    # extract numbers of deaths (by age-group, within the target_period)
    deaths = _num_deaths_by_age_group(results_folder, target_period)

    # extract person-years (by age-group, within the target_period)
    person_years = _aggregate_person_years_by_age(results_folder, target_period)

    # Initialize an empty list to collect life expectancies
    le_for_each_draw_and_run = dict()

    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):
            le_for_each_draw_and_run[(draw, run)] = _estimate_life_expectancy(
                _number_of_deaths_in_interval=deaths.loc[:, (draw, run)],
                _person_years_at_risk=person_years.loc[:, (draw, run)]
            )

    output = pd.DataFrame.from_dict(le_for_each_draw_and_run)
    output.index.name = "sex"
    output.columns = output.columns.set_names(level=[0, 1], names=['draw', 'run'])

    if not summary:
        return output

    else:
        return summarize(results=output, only_mean=False, collapse_columns=False)
