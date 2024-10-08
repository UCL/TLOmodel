"""
Read in the output files generated by analysis_scenarios
generate life tables to estimate life expectancy for each run/draw
produce summary statistics
"""

import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from tlo.analysis.utils import (
    extract_results,
    get_scenario_info,
    load_pickled_dataframes,
    summarize,
)


def _map_age_to_age_group(age: pd.Series) -> pd.Series:
    """
    Returns age-groups used in the calculation of life-expectancy.

    Args:
    - age (pd.Series): The pd.Series containing ages

    Returns:
    - pd.Series: Series of the 'age-group', corresponding the `age` argument.
    """
    # Define age groups in 5-year intervals
    age_groups = ['0'] + ['1-4'] + [f'{start}-{start + 4}' for start in range(5, 90, 5)] + ['90']

    return pd.cut(
        age,
        bins=[0] + [1] + list(range(5, 95, 5)) + [float('inf')],
        labels=age_groups, right=False
    )


def _extract_person_years(results_folder, _draw, _run) -> pd.Series:
    """Returns the person-years that are logged."""
    return load_pickled_dataframes(
        results_folder, _draw, _run, 'tlo.methods.demography'
    )['tlo.methods.demography']['person_years']


def _num_deaths_by_age_group(results_folder, target_period) -> pd.DataFrame:
    """Returns dataframe with number of deaths by sex/age-group within the target period for each draw/run
    (dataframe returned: index=sex/age-grp, columns=draw/run)
    """

    def extract_deaths_by_age_group(df: pd.DataFrame) -> pd.Series:
        age_group = _map_age_to_age_group(df['age'])
        return df.loc[
            pd.to_datetime(df.date).dt.date.between(*target_period, inclusive='both')
        ].groupby([age_group, df["sex"]]).size()

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_by_age_group,
        do_scaling=False
    )


def _aggregate_person_years_by_age(results_folder, target_period) -> pd.DataFrame:
    """ Returns person-years in each sex/age-group for each draw/run (as pd.DataFrame with index=sex/age-groups and
    columns=draw/run)
    """
    info = get_scenario_info(results_folder)
    py_by_sex_and_agegroup = dict()
    for draw in range(info["number_of_draws"]):
        for run in range(info["runs_per_draw"]):
            _df = _extract_person_years(results_folder, _draw=draw, _run=run)

            # mask for entries with dates within the target period
            mask = _df.date.dt.date.between(*target_period, inclusive="both")

            # Compute PY within time-period and summing within age-group, for each sex
            py_by_sex_and_agegroup[(draw, run)] = pd.concat({
                sex: _df.loc[mask, sex]
                        .apply(pd.Series)
                        .sum(axis=0)
                        .pipe(lambda x: x.groupby(_map_age_to_age_group(x.index.astype(float))).sum())
                for sex in ["M", "F"]}
            )

    # Format as pd.DataFrame with multiindex in index (sex/age-group) and columns (draw/run)
    py_by_sex_and_agegroup = pd.DataFrame.from_dict(py_by_sex_and_agegroup)
    py_by_sex_and_agegroup.index = py_by_sex_and_agegroup.index.set_names(
        level=[0, 1], names=["sex", "age_group"]
    )
    py_by_sex_and_agegroup.columns = py_by_sex_and_agegroup.columns.set_names(
        level=[0, 1], names=["draw", "run"]
    )

    return py_by_sex_and_agegroup


def calculate_probability_of_dying(interval_width, fraction_of_last_age_survived, sex, _person_years_at_risk,
                                   _number_of_deaths_in_interval) -> pd.DataFrame:
    """Returns the probability of dying in each interval"""

    person_years_by_sex = _person_years_at_risk.xs(key=sex, level='sex')

    number_of_deaths_by_sex = _number_of_deaths_in_interval.xs(key=sex, level='sex')

    death_rate_in_interval = number_of_deaths_by_sex / person_years_by_sex

    death_rate_in_interval = death_rate_in_interval.fillna(0)

    if death_rate_in_interval.loc['90'] == 0:
        death_rate_in_interval.loc['90'] = death_rate_in_interval.loc['85-89']

    condition = number_of_deaths_by_sex > (

        person_years_by_sex / interval_width / interval_width)

    probability_of_dying_in_interval = pd.Series(index=number_of_deaths_by_sex.index, dtype=float)

    probability_of_dying_in_interval[condition] = 1

    probability_of_dying_in_interval[~condition] = interval_width * death_rate_in_interval / (

        1 + interval_width * (1 - fraction_of_last_age_survived) * death_rate_in_interval)

    probability_of_dying_in_interval.at['90'] = 1
    return probability_of_dying_in_interval, death_rate_in_interval


def _estimate_life_expectancy(
    _person_years_at_risk: pd.Series,
    _number_of_deaths_in_interval: pd.Series
) -> Dict[str, float]:
    """
    For a single run, estimate life expectancy for males and females
    returns: Dict (keys by "M" and "F" for the sex, values the estimated life-expectancy at birth).
    """

    estimated_life_expectancy_at_birth = dict()

    # first age-group is 0, then 1-4, 5-9, 10-14 etc. 22 categories in total
    age_group_labels = _person_years_at_risk.index.get_level_values('age_group').unique()

    # Extract interval width
    interval_width = [
        5 if '90' in interval else int(interval.split('-')[1]) - int(interval.split('-')[0]) + 1
        if '-' in interval else 1 for interval in age_group_labels.categories
    ]
    number_age_groups = len(interval_width)
    fraction_of_last_age_survived = pd.Series([0.5] * number_age_groups, index=age_group_labels)

    # separate male and female data
    for sex in ['M', 'F']:
        probability_of_dying_in_interval, death_rate_in_interval = calculate_probability_of_dying(interval_width,
                                                                                                  fraction_of_last_age_survived,
                                                                                                  sex,
                                                                                                  _person_years_at_risk,
                                                                                                  _number_of_deaths_in_interval)
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
                _number_of_deaths_in_interval=deaths[(draw, run)],
                _person_years_at_risk=person_years[(draw, run)]
            )

    output = pd.DataFrame.from_dict(le_for_each_draw_and_run)
    output.index.name = "sex"
    output.columns = output.columns.set_names(level=[0, 1], names=['draw', 'run'])

    if not summary:
        return output

    else:
        return summarize(results=output, only_mean=False, collapse_columns=False)


def _calculate_probability_of_premature_death_for_single_run(
    age_before_which_death_is_defined_as_premature: int,
    person_years_at_risk: pd.Series,
    number_of_deaths_in_interval: pd.Series
) -> Dict[str, float]:
    """
    For a single run, estimate the probability of dying before the defined premature age for males and females.
    Returns: Dict (keys by "M" and "F" for the sex, values the estimated probability of dying before the defined
    premature age).
    """
    probability_of_premature_death = dict()

    age_group_labels = person_years_at_risk.index.get_level_values('age_group').unique()
    interval_width = [
        5 if '90' in interval else int(interval.split('-')[1]) - int(interval.split('-')[0]) + 1
        if '-' in interval else 1 for interval in age_group_labels.categories
    ]
    number_age_groups = len(interval_width)
    fraction_of_last_age_survived = pd.Series([0.5] * number_age_groups, index=age_group_labels)

    for sex in ['M', 'F']:
        probability_of_dying_in_interval, death_rate_in_interval = calculate_probability_of_dying(interval_width,
                                                                                                  fraction_of_last_age_survived,
                                                                                                  sex,
                                                                                                  person_years_at_risk,
                                                                                                  number_of_deaths_in_interval)

        # Calculate cumulative probability of dying before the defined premature age
        cumulative_probability_of_dying = 0
        proportion_alive_at_start_of_interval = 1.0

        for age_group, prob in probability_of_dying_in_interval.items():
            if int(age_group.split('-')[0]) >= age_before_which_death_is_defined_as_premature:
                break
            cumulative_probability_of_dying += proportion_alive_at_start_of_interval * prob
            proportion_alive_at_start_of_interval *= (1 - prob)

        probability_of_premature_death[sex] = cumulative_probability_of_dying

    return probability_of_premature_death


def get_probability_of_premature_death(
    results_folder: Path,
    target_period: Tuple[datetime.date, datetime.date],
    summary: bool = True,
    age_before_which_death_is_defined_as_premature: int = 70
) -> pd.DataFrame:
    """
    Produces sets of probability of premature death for each draw/run.

    Args:
    - results_folder (PosixPath): The path to the results folder containing log, `tlo.methods.demography`
    - target period (tuple of dates): Declare the date range (inclusively) in which the probability is to be estimated.
    - summary (bool): Declare whether to return a summarized value (mean with 95% uncertainty intervals)
        or return the estimate for each draw/run.
    - age_before_which_death_is_defined_as_premature (int): proposed in defined in Norheim et al.(2015) to be 70 years

    Returns:
    - pd.DataFrame: The DataFrame with the probability estimates for every draw/run in the results folder;
     or, with option `summary=True`, summarized (central, lower, upper estimates) for each draw.
    """
    info = get_scenario_info(results_folder)
    deaths = _num_deaths_by_age_group(results_folder, target_period)
    person_years = _aggregate_person_years_by_age(results_folder, target_period)

    prob_for_each_draw_and_run = dict()

    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):
            prob_for_each_draw_and_run[(draw, run)] = _calculate_probability_of_premature_death_for_single_run(
                age_before_which_death_is_defined_as_premature=age_before_which_death_is_defined_as_premature,
                number_of_deaths_in_interval=deaths[(draw, run)],
                person_years_at_risk=person_years[(draw, run)]
            )

    output = pd.DataFrame.from_dict(prob_for_each_draw_and_run)
    output.index.name = "sex"
    output.columns = output.columns.set_names(level=[0, 1], names=['draw', 'run'])

    if not summary:
        return output

    else:
        return summarize(results=output, only_mean=False, collapse_columns=False)
