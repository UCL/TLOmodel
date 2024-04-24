"""This file uses the results of the scenario runs to generate plots

*1 Epi outputs (incidence and mortality)

"""

import datetime
from pathlib import Path

# import lacroix
from matplotlib import pyplot as plt
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

# Find results_folder associated with a given batch_file (and get most recent [-1])
# results_folder = get_scenario_outputs("effect_of_treatment_packages", outputspath)[-1]
results_folder = get_scenario_outputs("exclude_services_Mar2024.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# colour scheme
# berry = lacroix.colorList('CranRaspberry')  # ['#F2B9B8', '#DF7878', '#E40035', '#009A90', '#0054A4', '#001563']

# -----------------------------------------------------------------------------------------
# %% Epi outputs
# -----------------------------------------------------------------------------------------


def summarize_median(results: pd.DataFrame) -> pd.DataFrame:
    """ edit existing utility function to return:
    median and 95% quantiles
    """

    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                results.columns.unique(level='draw'),
                ["median", "lower", "upper"]
            ],
            names=['draw', 'stat']),
        index=results.index
    )

    summary.loc[:, (slice(None), "median")] = results.groupby(axis=1, by='draw').quantile(0.5).values
    summary.loc[:, (slice(None), "lower")] = results.groupby(axis=1, by='draw').quantile(0.025).values
    summary.loc[:, (slice(None), "upper")] = results.groupby(axis=1, by='draw').quantile(0.975).values

    return summary


# ---------------------------------- HIV ---------------------------------- #

# HIV incidence
hiv_inc = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False
    )
)

hiv_cases = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="n_new_infections_adult_1549",
        index="date",
        do_scaling=False
    )
)

adult_pop = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="pop_total",
        index="date",
        do_scaling=False
    )
)

adult_plhiv = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="total_plhiv",
        index="date",
        do_scaling=False
    )
)

# ---------------------------------- PERSON-YEARS ---------------------------------- #
# for each scenario, return a df with the person-years logged in each draw/run
# to be used for calculating tb incidence or mortality rates


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
    do_scaling=False
)


# ---------------------------------- TB ---------------------------------- #
# number new active tb cases - convert to incidence rate
def tb_inc_func(results_folder):
    inc = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False
    )

    inc.columns = inc.columns.get_level_values(0)

    # divide each run of tb incidence by py from that run
    # tb logger starts at 2011-01-01, demog starts at 2010-01-01
    # extract py log from 2011
    py = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=False
    )
    py.columns = py.columns.get_level_values(0)

    inc_per_py = inc / py
    # remove first row (2010) as nan
    inc_per_py = inc_per_py.iloc[1:]

    # Calculate the mean of each row for each column name
    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                inc.columns.unique(level='draw'),
                ["median", "lower", "upper"]
            ],
            names=['draw', 'stat']),
        index=inc.index
    )

    summary.loc[:, (slice(None), "median")] = inc_per_py.groupby(axis=1, by='draw').quantile(0.5).values
    summary.loc[:, (slice(None), "lower")] = inc_per_py.groupby(axis=1, by='draw').quantile(0.025).values
    summary.loc[:, (slice(None), "upper")] = inc_per_py.groupby(axis=1, by='draw').quantile(0.975).values

    return summary


tb_inc = tb_inc_func(results_folder)


# ---------------------------------- MALARIA ---------------------------------- #

# malaria incidence
# value is per 1000py
mal_inc = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.malaria",
        key="incidence",
        column="inc_1000py_2_10",
        index="date",
        do_scaling=False
    )
)


# ---------------------------------- DEATHS ---------------------------------- #

# get full outputs of numbers of deaths

results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "cause"])["person_id"].count()
        )
)

results_deaths.to_csv(outputspath / "Mar2024_HTMresults/full_deaths.csv")


# plot AIDS deaths by yr
def summarise_aids_deaths(results_folder):
    results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "cause"])["person_id"].count()
        ),
        do_scaling=False,
    )
    # removes multi-index
    results_deaths = results_deaths.reset_index()

    # select only cause AIDS_TB and AIDS_non_TB
    tmp = results_deaths.loc[
        (results_deaths.cause == "AIDS_TB") | (results_deaths.cause == "AIDS_non_TB")
        ]

    # group deaths by year
    tmp = pd.DataFrame(tmp.groupby(["year"]).sum())

    py = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=False
    )
    # years = pd.to_datetime(_df["date"]).dt.year
    # py = pd.Series(dtype="int64", index=years)
    py.index = tmp.index

    deaths_per_py = tmp.iloc[:, 1:26].div(py) * 1000

    # get median and UI
    tmp2 = pd.concat({
        'median': deaths_per_py.groupby(level=0, axis=1).median(0.5),
        'lower': deaths_per_py.groupby(level=0, axis=1).quantile(0.025),
        'upper': deaths_per_py.groupby(level=0, axis=1).quantile(0.975)
    }, axis=1).swaplevel(axis=1)

    return tmp2


def summarise_deaths_for_one_cause(results_folder, cause):
    results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "cause"])["person_id"].count()
        ),
        do_scaling=False,
    )
    # removes multi-index
    results_deaths = results_deaths.reset_index()

    # select only cause specified
    tmp = results_deaths.loc[
        (results_deaths.cause == cause)
    ]

    # group deaths by year
    tmp = pd.DataFrame(tmp.groupby(["year"]).sum())

    py = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=False
    )

    py.index = tmp.index

    deaths_per_py = tmp.iloc[:, 1:26].div(py) * 1000

    # get median and UI
    tmp2 = pd.concat({
        'median': deaths_per_py.groupby(level=0, axis=1).median(0.5),
        'lower': deaths_per_py.groupby(level=0, axis=1).quantile(0.025),
        'upper': deaths_per_py.groupby(level=0, axis=1).quantile(0.975)
    }, axis=1).swaplevel(axis=1)

    return tmp2


aids_deaths = summarise_aids_deaths(results_folder)
tb_deaths = summarise_deaths_for_one_cause(results_folder, 'TB')
malaria_deaths = summarise_deaths_for_one_cause(results_folder, 'Malaria')

# remove first yr of deaths data (2010) to align with inc outputs
aids_deaths = aids_deaths.iloc[1:]
tb_deaths = tb_deaths.iloc[1:]
malaria_deaths = malaria_deaths.iloc[1:]

# ---------------------------------- SMOOTH DATA ---------------------------------- #
#  create smoothed lines
data_x = hiv_inc.index.year  # 2011 onwards

start_year = 2011
end_year = 2020
increment = 0.25

# Create the array
xvals = np.arange(start_year, end_year + increment, increment)


def create_smoothed_lines(data_x, df_y):
    """
    pass a dataframe into function
    this will smooth the data according to the lowess parameters defined
    and return a df with smoothed data across the xvals array
    """

    smoothed_df = pd.DataFrame()

    # extract each column in turn to smooth
    for column_name in df_y.columns:

        y = df_y[column_name]
        lowess = sm.nonparametric.lowess(endog=y, exog=data_x, xvals=xvals, frac=0.85, it=0)
        smoothed_df[column_name] = lowess

    smoothed_df.columns = smoothed_df.columns.to_flat_index()

    return smoothed_df


# smooth outputs for plots
smoothed_hiv_inc = create_smoothed_lines(data_x, hiv_inc) * 1000
smoothed_tb_inc = create_smoothed_lines(data_x, tb_inc) * 1000
smoothed_mal_inc = create_smoothed_lines(data_x, mal_inc)

# scaled numbers of deaths per year
smoothed_aids_deaths = create_smoothed_lines(data_x, aids_deaths)
smoothed_tb_deaths = create_smoothed_lines(data_x, tb_deaths)
smoothed_malaria_deaths = create_smoothed_lines(data_x, malaria_deaths)

# ---------------------------------- PLOTS ---------------------------------- #

plt.style.use('default')  # to reset

font = {'family': 'sans-serif',
        'color': 'black',
        'weight': 'bold',
        'size': 11,
        }

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
colors = ['#1E22AA', '#9B26B6', '#F8485E', '#FF8F1C', '#30B700']


# Set x-axis tick positions and labels at every second data point
xvals_for_ticks = xvals[0::4]
xlabels_for_ticks = [int(val) for val in xvals_for_ticks]

# colors = sns.color_palette("deep", 5)
gridcol = '#EDEDED'


fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4,
                                    constrained_layout=True,
                                    figsize=(16, 8))
fig.suptitle('')

for i, scenario in enumerate(smoothed_hiv_inc.filter(like='median')):
    ax1.plot(xvals, smoothed_hiv_inc.filter(like='median')[scenario], label=scenario,
             color=colors[i], zorder=2)

for i, scenario in enumerate(smoothed_hiv_inc.filter(like='lower')):
    ax1.fill_between(xvals, np.array(smoothed_hiv_inc.loc[:, [(i, 'lower')]]).flatten(),
                 np.array(smoothed_hiv_inc.loc[:, [(i, 'upper')]]).flatten(), color=colors[i],
                 alpha=0.2, zorder=2)

ax1.grid(True, linestyle='-', color=gridcol, zorder=1)
ax1.set(title='HIV',
        ylabel='HIV Incidence, per 1000py')
ax1.set_xticks(xvals_for_ticks)
ax1.set_xticklabels("")
ax1.set_ylim(0, 14)

# TB incidence
for i, scenario in enumerate(smoothed_tb_inc.filter(like='median')):
    ax2.plot(xvals, smoothed_tb_inc.filter(like='median')[scenario], label=scenario,
             color=colors[i], zorder=2)

for i, scenario in enumerate(smoothed_tb_inc.filter(like='lower')):
    ax2.fill_between(xvals, np.array(smoothed_tb_inc.loc[:, [(i, 'lower')]]).flatten(),
                 np.array(smoothed_tb_inc.loc[:, [(i, 'upper')]]).flatten(), color=colors[i],
                 alpha=0.2, zorder=2)

ax2.grid(True, linestyle='-', color=gridcol, zorder=1)
ax2.set(title='TB',
        ylabel='TB Incidence, per 1000py')
ax2.set_xticks(xvals_for_ticks)
ax2.set_xticklabels("")
ax2.set_ylim(0, 6)


# Malaria incidence
for i, scenario in enumerate(smoothed_mal_inc.filter(like='median')):
    ax3.plot(xvals, smoothed_mal_inc.filter(like='median')[scenario], label=scenario,
             color=colors[i], zorder=2)

for i, scenario in enumerate(smoothed_mal_inc.filter(like='lower')):
    ax3.fill_between(xvals, np.array(smoothed_mal_inc.loc[:, [(i, 'lower')]]).flatten(),
                 np.array(smoothed_mal_inc.loc[:, [(i, 'upper')]]).flatten(), color=colors[i],
                 alpha=0.2, zorder=2)

ax3.grid(True, linestyle='-', color=gridcol, zorder=1)
ax3.set(title='Malaria',
        ylabel='Malaria Incidence, per 1000py')
ax3.set_xticks(xvals_for_ticks)
ax3.set_xticklabels("")
ax3.set_ylim(0, 900)

# empty plot
ax4.axis('off')

# AIDS deaths
for i, scenario in enumerate(smoothed_aids_deaths.filter(like='median')):
    ax5.plot(xvals, smoothed_aids_deaths.filter(like='median')[scenario], label=scenario,
             color=colors[i], zorder=2)

for i, scenario in enumerate(smoothed_aids_deaths.filter(like='lower')):
    ax5.fill_between(xvals, np.array(smoothed_aids_deaths.loc[:, [(i, 'lower')]]).flatten(),
                 np.array(smoothed_aids_deaths.loc[:, [(i, 'upper')]]).flatten(), color=colors[i],
                 alpha=0.2, zorder=2)

ax5.grid(True, linestyle='-', color=gridcol, zorder=1)
ax5.set(title='',
        ylabel='AIDS mortality rate per 1000py')
ax5.set_xticks(xvals_for_ticks)
ax5.set_xticklabels(xlabels_for_ticks)
ax5.set_ylim(0, 7)
ax5.tick_params(axis='x', rotation=70)

# TB deaths
for i, scenario in enumerate(smoothed_tb_deaths.filter(like='median')):
    ax6.plot(xvals, smoothed_tb_deaths.filter(like='median')[scenario], label=scenario,
             color=colors[i], zorder=2)

for i, scenario in enumerate(smoothed_tb_deaths.filter(like='lower')):
    ax6.fill_between(xvals, np.array(smoothed_tb_deaths.loc[:, [(i, 'lower')]]).flatten(),
                 np.array(smoothed_tb_deaths.loc[:, [(i, 'upper')]]).flatten(), color=colors[i],
                 alpha=0.2, zorder=2)

ax6.grid(True, linestyle='-', color=gridcol, zorder=1)
ax6.set(title='',
        ylabel='TB mortality rate per 1000py')
ax6.set_xticks(xvals_for_ticks)
ax6.set_xticklabels(xlabels_for_ticks)
ax6.set_ylim(0, 2)
ax6.tick_params(axis='x', rotation=70)

# Malaria deaths
for i, scenario in enumerate(smoothed_malaria_deaths.filter(like='median')):
    ax7.plot(xvals, smoothed_malaria_deaths.filter(like='median')[scenario], label=scenario,
             color=colors[i], zorder=2)

for i, scenario in enumerate(smoothed_malaria_deaths.filter(like='lower')):
    ax7.fill_between(xvals, np.array(smoothed_malaria_deaths.loc[:, [(i, 'lower')]]).flatten(),
                 np.array(smoothed_malaria_deaths.loc[:, [(i, 'upper')]]).flatten(), color=colors[i],
                 alpha=0.2, zorder=2)

ax7.grid(True, linestyle='-', color=gridcol, zorder=1)
ax7.set(title='',
        ylabel='Malaria mortality rate per 1000py')
ax7.set_xticks(xvals_for_ticks)
ax7.set_xticklabels(xlabels_for_ticks)
ax7.set_ylim(0, 5)
ax7.tick_params(axis='x', rotation=70)
ax7.legend(loc='upper right',
           labels=['Status Quo', 'Exclude HIV services', 'Exclude TB services',
                   'Exclude Malaria services', 'Exclude HTM services'],
           bbox_to_anchor=(1.8, 1.0))

# empty plot
ax8.axis('off')

# fig.savefig(outputspath / "Mar2024_HTMresults/Epi_outputs_excl_htm_mortality.png")

plt.show()










# %%:  ---------------------------------- DALYS ---------------------------------- #
TARGET_PERIOD = (Date(2010, 1, 1), Date(2036, 1, 1))


def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'year']) \
        .sum()


def return_daly_summary(results_folder):
    dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True
    )
    dalys.columns = dalys.columns.get_level_values(0)

    return dalys

# each year extract dalys for malaria

malaria_daly = pd.DataFrame()

for year in range(2010, 2020):

    TARGET_PERIOD = (Date(year, 1, 1), Date(year, 12, 31))
    dalys = return_daly_summary(results_folder)

    # select draw=0 and label=malaria
    selected_data = dalys.loc['Malaria', 0]
    selected_df = selected_data.to_frame().T

    # Append the DataFrame to the original DataFrame
    malaria_daly = malaria_daly.append(selected_df)

malaria_daly.index = range(2010, 2020)


# plot malaria DALYs
malaria_daly.plot()
plt.title("Malaria DALYs by year")
plt.show()
