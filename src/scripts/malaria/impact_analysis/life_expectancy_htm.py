"""
Read in the output files generated by analysis_scenarios
generate life tables to estimate life expectancy for each run/draw
produce summary statistics

this version uses the HTM mortality rates from the EXCL HTM scenario,
i.e. without any active HTM programmes
adds on the 'other' mortality rates from the status quo
then calculates life expectancy, fixing background mortality rates
to the actual level
-> an estimation of the extent to which indirect effects contribute to
the increase in life expectancy with active HTM programmes
"""

import datetime
from pathlib import Path
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Dict, Tuple

from tlo import Date

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
)

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")  # use for azure batch runs

# outputspath = Path("./outputs")  # use for local runs

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("exclude_services_Mar2024.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# get number of draws and numbers of runs
info = get_scenario_info(results_folder)


# HELPER FUNCTIONS


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


def _get_multiplier(_draw, _run):
    """Helper function to get the multiplier from the simulation."""
    return load_pickled_dataframes(results_folder, _draw, _run, 'tlo.methods.population'
                                   )['tlo.methods.population']['scaling_factor']['scaling_factor'].values[0]


def _extract_person_years(results_folder, _draw, _run) -> pd.Series:
    """Returns the person-years that are logged."""
    return load_pickled_dataframes(
        results_folder, _draw, _run, 'tlo.methods.demography'
    )['tlo.methods.demography']['person_years']


def _create_multi_index_columns():
    return pd.MultiIndex.from_product([range(info['number_of_draws']),
                                       range(info['runs_per_draw'])],
                                      names=['draw', 'run'])


def _num_deaths_by_age_group(results_folder, target_period, cause) -> pd.DataFrame:
    """Returns dataframe with number of deaths by sex/age-group within the target period for each draw/run
    (dataframe returned: index=sex/age-grp, columns=draw/run)
    """

    if cause not in ['HTM', 'other']:
        raise ValueError("Cause must be either 'HTM' or 'other'")

    def extract_deaths_by_age_group(df: pd.DataFrame) -> pd.Series:
        age_group = _map_age_to_age_group(df['age'])
        if cause == 'HTM':
            filtered_df = df[df['label'].isin(['AIDS', 'Malaria', 'TB (non-AIDS)'])]
        elif cause == 'other':
            # exclude deaths due to HTM
            filtered_df = df[~df['label'].isin(['AIDS', 'TB', 'Malaria'])]

        return filtered_df.loc[
            pd.to_datetime(filtered_df.date).dt.date.between(*target_period, inclusive='both')
        ].groupby([age_group, filtered_df["sex"]]).size()

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


# get adjusted mortality rates - fixing mortality due ot 'other'
# causes using scenario 0 (status quo)
def _adjusted_mortality_rates(
    results_folder,
    target_period,
    sex) -> pd.DataFrame:

    # calculate mortality rates due to causes HTM for scenario EXCL HTM
    deathsHTM = _num_deaths_by_age_group(results_folder, target_period, cause='HTM')

    # extract person-years (by age-group, within the target_period)
    person_years = _aggregate_person_years_by_age(results_folder, target_period)

    tmp = deathsHTM.xs(key=sex, level='sex')
    tmp2 = person_years.xs(key=sex, level='sex')

    mortality_rate_HTM = tmp.div(tmp2)

    # get mortality rates due to cause 'other' from draw 0
    deathsOther = _num_deaths_by_age_group(results_folder, target_period, cause='other')
    tmp3 = deathsOther.xs(key=sex, level='sex')

    mortality_rate_Other = tmp3.div(tmp2)

    # sum mortality_rate_Other draw0 with mortality_rate_HTM draw4
    # Select subsets of columns from both DataFrames
    subset1 = mortality_rate_Other.iloc[:, :5]  # Select columns 0 to 4 from mortality_rate_Other
    subset2 = mortality_rate_HTM.iloc[:, 20:25]  # Select columns 20 to 24 from mortality_rate_HTM

    # Align the columns by reindexing
    # Ensure the column levels are aligned before reindexing
    subset1.columns = subset1.columns.droplevel(0)
    subset2.columns = subset2.columns.droplevel(0)

    # Add the two subsets together
    summed_mortality_rates = subset1 + subset2
    median_adjusted_mortality_rates = summed_mortality_rates.median(axis=1)

    return median_adjusted_mortality_rates


target_period = (datetime.date(2019, 1, 1), datetime.date(2020, 1, 1))

M_adjusted_mortality = _adjusted_mortality_rates(
    results_folder=results_folder,
    target_period=target_period,
    sex='M')

F_adjusted_mortality = _adjusted_mortality_rates(
    results_folder=results_folder,
    target_period=target_period,
    sex='F')

# enter manually into life expectancy spreadsheet column M_i to get life expectancy at birth
