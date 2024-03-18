"""
Read in the output files generated by analysis_scenarios and plot outcomes for comparison
"""

import datetime
from pathlib import Path
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
import seaborn as sns

# outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("exclude_HTM_services.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# --------------------------------------------------------------------
# EXTRACT DATA


# mean dalys by cause with UI by year

results = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
    custom_generate_series=(
        lambda df_: df_.drop(
            columns=(['date', 'sex', 'age_range']),
        ).groupby(['year']).sum().stack()
    ),
    do_scaling=True
)
# indices are year/label
results.index = results.index.set_names('label', level=1)

median_dalys = results.groupby(level=0, axis=1).median(0.5)
lower_dalys = results.groupby(level=0, axis=1).quantile(0.025)
upper_dalys = results.groupby(level=0, axis=1).quantile(0.975)


# person-years (total) mean by year

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


person_years = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=True
    ),
    only_mean=True, collapse_columns=False
)

person_years.index = person_years.index.year


# # calculate DALYs per 100,000 population by year
def edit_data_for_plotting(draw):

    median = median_dalys.loc[:, draw].reset_index(0)
    median = median.reset_index(0)
    median = median.rename({draw: 'dalys'}, axis=1)

    lower = lower_dalys.loc[:, draw].reset_index(0)
    lower = lower.reset_index(0)
    lower = lower.rename({draw: 'dalys_lower'}, axis=1)

    upper = upper_dalys.loc[:, draw].reset_index(0)
    upper = upper.reset_index(0)
    upper = upper.rename({draw: 'dalys_upper'}, axis=1)

    py = person_years.loc[:, draw].reset_index(0)
    py = py.rename({'date': 'year', draw: 'py'}, axis=1)

    # map person-years by year to the rows
    median['py'] = median['year'].map(py.set_index('year')['py'])
    median['dalys_per_100_000'] = (median['dalys'] / median['py']) * 100_000

    lower['py'] = lower['year'].map(py.set_index('year')['py'])
    lower['dalys_per_100_000_lower'] = (lower['dalys_lower'] / lower['py']) * 100_000

    upper['py'] = upper['year'].map(py.set_index('year')['py'])
    upper['dalys_per_100_000_upper'] = (upper['dalys_upper'] / upper['py']) * 100_000

    new_df = pd.merge(median, lower,  how='left', on=['label', 'year'])
    new_df = pd.merge(new_df, upper,  how='left', on=['label', 'year'])

    return new_df


dalys0 = edit_data_for_plotting(draw=0)
dalys4 = edit_data_for_plotting(draw=4)

# plots
# Set the style of seaborn
sns.set(style="whitegrid")

# Define a color palette with 6 colors
colors = sns.color_palette("husl", 6)


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                    constrained_layout=True,
                                    figsize=(16, 8))
fig.suptitle('')

# ALRI
d0 = dalys0.loc[(dalys0.label == 'Lower respiratory infections') & (dalys0.year < 2020)]

ax1.plot(d0['year'], d0['dalys_per_100_000'],
         color=colors[0])
ax1.fill_between(d0['year'], d0['dalys_per_100_000_lower'].astype(float),
                 d0['dalys_per_100_000_upper'].astype(float),
                 color=colors[0], alpha=0.2)

d4 = dalys4.loc[(dalys4.label == 'Lower respiratory infections') & (dalys4.year < 2020)]

ax1.plot(d4['year'], d4['dalys_per_100_000'],
         color=colors[3])
ax1.fill_between(d4['year'], d4['dalys_per_100_000_lower'].astype(float),
                 d4['dalys_per_100_000_upper'].astype(float),
                 color=colors[3], alpha=0.2)

ax1.set_ylim(0, 9000)
ax1.set(title='ALRI',
        ylabel='DALYs per 100,000',
        xlabel='Year')


# Diarrhoea
d0 = dalys0.loc[(dalys0.label == 'Childhood Diarrhoea') & (dalys0.year < 2020)]

ax2.plot(d0['year'], d0['dalys_per_100_000'],
         color=colors[0])
ax2.fill_between(d0['year'], d0['dalys_per_100_000_lower'].astype(float),
                 d0['dalys_per_100_000_upper'].astype(float),
                 color=colors[0], alpha=0.2)

d4 = dalys4.loc[(dalys4.label == 'Childhood Diarrhoea') & (dalys4.year < 2020)]

ax2.plot(d4['year'], d4['dalys_per_100_000'],
         color=colors[3])
ax2.fill_between(d4['year'], d4['dalys_per_100_000_lower'].astype(float),
                 d4['dalys_per_100_000_upper'].astype(float),
                 color=colors[3], alpha=0.2)

ax2.set_ylim(0, 4000)
ax2.set(title='Childhood diarrhoea',
        ylabel='DALYs per 100,000',
        xlabel='Year')

# Maternal disorders
d0 = dalys0.loc[(dalys0.label == 'Maternal Disorders') & (dalys0.year < 2020)]

ax3.plot(d0['year'], d0['dalys_per_100_000'],
         color=colors[0])
ax3.fill_between(d0['year'], d0['dalys_per_100_000_lower'].astype(float),
                 d0['dalys_per_100_000_upper'].astype(float),
                 color=colors[0], alpha=0.2)

d4 = dalys4.loc[(dalys4.label == 'Maternal Disorders') & (dalys4.year < 2020)]

ax3.plot(d4['year'], d4['dalys_per_100_000'],
         color=colors[3])
ax3.fill_between(d4['year'], d4['dalys_per_100_000_lower'].astype(float),
                 d4['dalys_per_100_000_upper'].astype(float),
                 color=colors[3], alpha=0.2)

ax3.set_ylim(0, 2000)
ax3.set(title='Maternal disorders',
        ylabel='DALYs per 100,000',
        xlabel='Year')

# Neonatal disorders
d0 = dalys0.loc[(dalys0.label == 'Neonatal Disorders') & (dalys0.year < 2020)]

ax4.plot(d0['year'], d0['dalys_per_100_000'],
         color=colors[0])
ax4.fill_between(d0['year'], d0['dalys_per_100_000_lower'].astype(float),
                 d0['dalys_per_100_000_upper'].astype(float),
                 color=colors[0], alpha=0.2)

d4 = dalys4.loc[(dalys4.label == 'Neonatal Disorders') & (dalys4.year < 2020)]

ax4.plot(d4['year'], d4['dalys_per_100_000'],
         color=colors[3])
ax4.fill_between(d4['year'], d4['dalys_per_100_000_lower'].astype(float),
                 d4['dalys_per_100_000_upper'].astype(float),
                 color=colors[3], alpha=0.2)

ax4.set_ylim(0, 9000)
ax4.set(title='Neonatal disorders',
        ylabel='DALYs per 100,000',
        xlabel='Year')


# Heart disease
d0 = dalys0.loc[(dalys0.label == 'Heart Disease') & (dalys0.year < 2020)]

ax5.plot(d0['year'], d0['dalys_per_100_000'],
         color=colors[0])
ax5.fill_between(d0['year'], d0['dalys_per_100_000_lower'].astype(float),
                 d0['dalys_per_100_000_upper'].astype(float),
                 color=colors[0], alpha=0.2)

d4 = dalys4.loc[(dalys4.label == 'Heart Disease') & (dalys4.year < 2020)]

ax5.plot(d4['year'], d4['dalys_per_100_000'],
         color=colors[3])
ax5.fill_between(d4['year'], d4['dalys_per_100_000_lower'].astype(float),
                 d4['dalys_per_100_000_upper'].astype(float),
                 color=colors[3], alpha=0.2)

ax5.set_ylim(0, 600)
ax5.set(title='Heart Disease',
        ylabel='DALYs per 100,000',
        xlabel='Year')

# Kidney disease
d0 = dalys0.loc[(dalys0.label == 'Kidney Disease') & (dalys0.year < 2020)]

ax6.plot(d0['year'], d0['dalys_per_100_000'],
         color=colors[0])
ax6.fill_between(d0['year'], d0['dalys_per_100_000_lower'].astype(float),
                 d0['dalys_per_100_000_upper'].astype(float),
                 color=colors[0], alpha=0.2)

d4 = dalys4.loc[(dalys4.label == 'Kidney Disease') & (dalys4.year < 2020)]

ax6.plot(d4['year'], d4['dalys_per_100_000'],
         color=colors[3])
ax6.fill_between(d4['year'], d4['dalys_per_100_000_lower'].astype(float),
                 d4['dalys_per_100_000_upper'].astype(float),
                 color=colors[3], alpha=0.2)

ax6.set_ylim(0, 200)
ax6.set(title='Kidney Disease',
        ylabel='DALYs per 100,000',
        xlabel='Year')

fig.savefig(outputspath / "DALYs_per_100k.png")

plt.show()



####################
# add column totals
# d0.loc['Total']= d0.sum()
# d4.loc['Total']= d4.sum()


# plot population size for baseline and Excl HTM

plt.plot(d0.year, d0.py, label='Line 1', color='blue', marker='o')
plt.plot(d4.year, d4.py, label='Line 2', color='orange', marker='s')
plt.ylabel('Population size')
plt.legend(labels=['Status quo', 'Excluding HTM'])
plt.show()
