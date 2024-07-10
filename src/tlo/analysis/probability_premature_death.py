"""
Read in the output files generated by analysis_scenarios
generate life tables to estimate probability of premature death (defined as before age 70) for each run/draw
produce summary statistics
"""

import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tlo.analysis.life_expectancy import _aggregate_person_years_by_age, _num_deaths_by_age_group

from tlo.analysis.utils import get_scenario_info, summarize

def _calculate_probability_of_dying_before_70(
    _person_years_at_risk: pd.Series,
    _number_of_deaths_in_interval: pd.Series
) -> Dict[str, float]:
    """
    For a single run, estimate the probability of dying before age 70 for males and females.
    Returns: Dict (keys by "M" and "F" for the sex, values the estimated probability of dying before age 70).
    """
    probability_of_dying_before_70 = dict()

    age_group_labels = _person_years_at_risk.index.get_level_values('age_group').unique()
    interval_width = [
        5 if '90' in interval else int(interval.split('-')[1]) - int(interval.split('-')[0]) + 1
        if '-' in interval else 1 for interval in age_group_labels.categories
    ]
    number_age_groups = len(interval_width)
    fraction_of_last_age_survived = pd.Series([0.5] * number_age_groups, index=age_group_labels)

    for sex in ['M', 'F']:
        person_years_by_sex = _person_years_at_risk.xs(key=sex, level='sex')
        number_of_deaths_by_sex = _number_of_deaths_in_interval.xs(key=sex, level='sex')

        death_rate_in_interval = number_of_deaths_by_sex / person_years_by_sex
        death_rate_in_interval = death_rate_in_interval.fillna(0)
        if death_rate_in_interval.loc['90'] == 0:
            death_rate_in_interval.loc['90'] = death_rate_in_interval.loc['85-89']

        condition = number_of_deaths_by_sex > (
            person_years_by_sex / interval_width / fraction_of_last_age_survived)
        probability_of_dying_in_interval = pd.Series(index=number_of_deaths_by_sex.index, dtype=float)
        probability_of_dying_in_interval[condition] = 1
        probability_of_dying_in_interval[~condition] = interval_width * death_rate_in_interval / (
            1 + interval_width * (1 - fraction_of_last_age_survived) * death_rate_in_interval)
        probability_of_dying_in_interval.at['90'] = 1

        # Calculate cumulative probability of dying before age 70
        cumulative_probability_of_dying = 0
        number_alive_at_start_of_interval = 1.0

        for age_group, prob in probability_of_dying_in_interval.items():
            if age_group in ['70-74', '75-79', '80-84', '85-89', '90']:
                break
            cumulative_probability_of_dying += number_alive_at_start_of_interval * prob
            number_alive_at_start_of_interval *= (1 - prob)

        probability_of_dying_before_70[sex] = cumulative_probability_of_dying

    return probability_of_dying_before_70


def get_probability_of_dying_before_70(
    results_folder: Path,
    target_period: Tuple[datetime.date, datetime.date],
    summary: bool = True
) -> pd.DataFrame:
    """
    Produces sets of probability of dying before a specified age for each draw/run.

    Args:
    - results_folder (PosixPath): The path to the results folder containing log, `tlo.methods.demography`
    - target period (tuple of dates): Declare the date range (inclusively) in which the probability is to be estimated.
    - summary (bool): Declare whether to return a summarized value (mean with 95% uncertainty intervals)
        or return the estimate for each draw/run.

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
            prob_for_each_draw_and_run[(draw, run)] = _calculate_probability_of_dying_before_70(
                _number_of_deaths_in_interval=deaths[(draw, run)],
                _person_years_at_risk=person_years[(draw, run)]
            )

    output = pd.DataFrame.from_dict(prob_for_each_draw_and_run)
    output.index.name = "sex"
    output.columns = output.columns.set_names(level=[0, 1], names=['draw', 'run'])

    if not summary:
        return output

    else:
        return summarize(results=output, only_mean=False, collapse_columns=False)
