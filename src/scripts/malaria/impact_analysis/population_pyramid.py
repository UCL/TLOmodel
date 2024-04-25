

import plotly

import pandas as pd
import plotly.graph_objects as gp

import datetime
from pathlib import Path

# import lacroix
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

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



outputspath = Path("./outputs/t.mangal@imperial.ac.uk")
results_folder = get_scenario_outputs("exclude_services_Mar2024.py", outputspath)[-1]

# get basic information about the results
scenario_info = get_scenario_info(results_folder)


scaling_factor = extract_results(
    results_folder,
    module="tlo.methods.population",
    key="scaling_factor",
    column="scaling_factor",
    index="date",
    do_scaling=False)


def _extract_person_years(results_folder, _draw, _run) -> pd.Series:
    """Returns the person-years that are logged."""
    return load_pickled_dataframes(
        results_folder, _draw, _run, 'tlo.methods.demography'
    )['tlo.methods.demography']['person_years']


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


# extract person-years for 2019 only
target_period = (datetime.date(2019, 1, 1), datetime.date(2020, 1, 1))

person_years = _aggregate_person_years_by_age(results_folder, target_period)
median_py = person_years.groupby(level=0, axis=1).median(0.5)

male_py = median_py.xs(key='M', level='sex') * scaling_factor.values[0][0]
female_py = median_py.xs(key='F', level='sex') * scaling_factor.values[0][0]


y_age = female_py.index
# reverse the order of the age categories
x_M = male_py[::-1]
x_F = female_py[::-1] * -1

# Reverse the order of the age groups
y_age_reversed = y_age[::-1]

colour1 = '#343579'
colour2 = '#F8485E'
# Create dummy patches for legend
patch1 = patches.Patch(color=colour1, label='Status quo')
patch2 = patches.Patch(color=colour2, label='Excluding HTM')


# Create the bar plots with reversed order
ax1 = sns.barplot(x=x_M.loc[:, 0], y=y_age_reversed, order=y_age_reversed, color=colour1, alpha=0.9, label="Male")
ax2 = sns.barplot(x=x_F.loc[:, 0], y=y_age_reversed, order=y_age_reversed, color=colour1, alpha=0.9, label="Female")

# Add the second series on top with a different color scheme
ax1 = sns.barplot(x=x_M.loc[:, 4], y=y_age_reversed, order=y_age_reversed, color=colour2, alpha=0.9, label="")
ax2 = sns.barplot(x=x_F.loc[:, 4], y=y_age_reversed, order=y_age_reversed, color=colour2, alpha=0.9, label="")

plt.title("")
plt.xlabel("Female -- Male    ")
plt.ylabel('Age group')
plt.grid()
plt.xticks(ticks=[-1000000, -500000, 0, 500000, 1000000],
labels=['1', '0.5', '0', '0.5', '1'])

plt.legend(handles=[patch1, patch2], title="")

plt.savefig(outputspath / "Mar2024_HTMresults/PopulationPyramid.png")

plt.show()

# ------------------------------------------------------------------------------------------------------
# get person-years by year

def get_person_years(_df):
    """ extract person-years for each draw/run
    sums across men and women
    will skip column if particular run has failed
    """
    years = pd.to_datetime(_df["date"]).dt.year
    py = pd.Series(dtype="int64", index=years)
    for year in years:
        tot_py = (
            (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["M"]).apply(pd.Series) +
            (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["F"]).apply(pd.Series)
        ).transpose()
        py[year] = tot_py.sum().values[0]

    py.index = pd.to_datetime(years, format="%Y")

    return py


py0 = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=True
)

# select draw 0 and draw 4
new_df = py0.loc[:, py0.columns.get_level_values('draw').isin([0, 4])]


# plot population size for baseline and Excl HTM

col = ['#343579', '#F8485E']
col_repeated = [color for color in col for _ in range(5)]
i=0

for column in new_df.columns:
    plt.plot(new_df[column], label=column, color=col_repeated[i])
    i+=1

plt.yticks(ticks=[14000000, 15000000, 16000000, 17000000, 18000000, 19000000],
    labels=['14.0', '15.0', '16.0', '17.0', '18.0', '19.0'])

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Population size, millions')
plt.title('')

legend_labels = ['Status quo', 'Excluding HTM']
legend_handles = [Line2D([0], [0], color=color, lw=2) for color in col]
plt.legend(handles=legend_handles, labels=legend_labels)

plt.savefig(outputspath / "Mar2024_HTMresults/PopulationSize.png")

plt.show()
