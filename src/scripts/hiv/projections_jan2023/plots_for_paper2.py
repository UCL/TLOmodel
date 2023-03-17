"""This file uses the results of the scenario runs to generate plots

*1 Epi outputs (incidence and mortality)
*2 Care cascades (HIV and TB)
*3 Treatment delays (HIV and TB)
*4 DALYS averted

"""

import datetime
from pathlib import Path
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import lacroix
import math
from scipy import interpolate
import statsmodels.api as sm

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
from tlo import Date

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]

# colour scheme
berry = lacroix.colorList('CranRaspberry')  # ['#F2B9B8', '#DF7878', '#E40035', '#009A90', '#0054A4', '#001563']
baseline_colour = berry[5]  # '#001563'
sc1_colour = berry[3]  # '#009A90'
sc2_colour = berry[2]  # '#E40035'


# -----------------------------------------------------------------------------------------
# %% Epi outputs
# -----------------------------------------------------------------------------------------


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
    results0,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=False
)

py1 = extract_results(
    results1,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=False
)

py2 = extract_results(
    results2,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=False
)


# ---------------------------------- HIV ---------------------------------- #

# HIV incidence

def hiv_adult_inc(results_folder):
    inc = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False
    )

    inc.columns = inc.columns.get_level_values(0)
    inc_summary = pd.DataFrame(index=inc.index, columns=["median", "lower", "upper"])
    inc_summary["median"] = inc.median(axis=1)
    inc_summary["lower"] = inc.quantile(q=0.025, axis=1)
    inc_summary["upper"] = inc.quantile(q=0.975, axis=1)

    return inc_summary


inc0 = hiv_adult_inc(results0)
inc1 = hiv_adult_inc(results1)
inc2 = hiv_adult_inc(results2)


# ---------------------------------- TB ---------------------------------- #

# number new active tb cases
def tb_inc(results_folder):
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
    # extract py log from 2011-2035
    py = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=False
    )
    py.columns = py.columns.get_level_values(0)

    inc_py = inc / py.iloc[:, 1:26]
    inc_summary = pd.DataFrame(index=inc.index, columns=["median", "lower", "upper"])
    inc_summary["median"] = inc_py.median(axis=1)
    inc_summary["lower"] = inc_py.quantile(q=0.025, axis=1)
    inc_summary["upper"] = inc_py.quantile(q=0.975, axis=1)

    return inc_summary


tb_inc0 = tb_inc(results0)
tb_inc1 = tb_inc(results1)
tb_inc2 = tb_inc(results2)


# %%:  ---------------------------------- HIV deaths ---------------------------------- #

# AIDS deaths

def summarise_aids_deaths(results_folder, person_years):
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
    tmp2 = pd.DataFrame(tmp.groupby(["year"]).sum())

    # divide each draw/run by the respective person-years from that run
    # need to reset index as they don't match exactly (date format)
    tmp3 = tmp2.reset_index(drop=True) / (person_years.reset_index(drop=True))

    aids_deaths = {}  # empty dict

    aids_deaths["median_aids_deaths_rate_100kpy"] = (
                                                        tmp3.astype(float).quantile(0.5, axis=1)
                                                    ) * 100000
    aids_deaths["lower_aids_deaths_rate_100kpy"] = (
                                                       tmp3.astype(float).quantile(0.025, axis=1)
                                                   ) * 100000
    aids_deaths["upper_aids_deaths_rate_100kpy"] = (
                                                       tmp3.astype(float).quantile(0.975, axis=1)
                                                   ) * 100000

    return aids_deaths


aids_deaths0 = summarise_aids_deaths(results0, py0)
aids_deaths1 = summarise_aids_deaths(results1, py1)
aids_deaths2 = summarise_aids_deaths(results2, py2)


# %%:  ---------------------------------- TB deaths ---------------------------------- #


# deaths due to TB (not including TB-HIV)
def summarise_tb_deaths(results_folder, person_years):
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
    tmp = results_deaths.loc[results_deaths.cause == "TB"]

    # group deaths by year
    tmp2 = pd.DataFrame(tmp.groupby(["year"]).sum())

    # divide each draw/run by the respective person-years from that run
    # need to reset index as they don't match exactly (date format)
    tmp3 = tmp2.reset_index(drop=True) / (person_years.reset_index(drop=True))

    tb_deaths = {}  # empty dict

    tb_deaths["median_tb_deaths_rate_100kpy"] = (
                                                    tmp3.astype(float).quantile(0.5, axis=1)
                                                ) * 100000
    tb_deaths["lower_tb_deaths_rate_100kpy"] = (
                                                   tmp3.astype(float).quantile(0.025, axis=1)
                                               ) * 100000
    tb_deaths["upper_tb_deaths_rate_100kpy"] = (
                                                   tmp3.astype(float).quantile(0.975, axis=1)
                                               ) * 100000

    return tb_deaths


tb_deaths0 = summarise_tb_deaths(results0, py0)
tb_deaths1 = summarise_tb_deaths(results1, py1)
tb_deaths2 = summarise_tb_deaths(results2, py2)


#  create smoothed lines
def create_smoothed_lines(data_x, data_y):

    # xnew = np.linspace(data_x.min(), data_x.max(), interpolation_points)
    # bspline = interpolate.make_interp_spline(data_x, data_y, k=3)
    # smoothed_data = bspline(xnew)

    smoothed_data = sm.nonparametric.lowess(data_y, data_x, frac=0.7)

    return smoothed_data


data_x = inc0.index[11:].year  # 2022 onwards

# hiv incidence
s_hiv_inc0 = create_smoothed_lines(data_x, (inc0["median"][11:] * 100000))
s_hiv_inc0_l = create_smoothed_lines(data_x, (inc0["lower"][11:] * 100000))
s_hiv_inc0_u = create_smoothed_lines(data_x, (inc0["upper"][11:] * 100000))

s_hiv_inc1 = create_smoothed_lines(data_x, (inc1["median"][11:] * 100000))
s_hiv_inc1_l = create_smoothed_lines(data_x, (inc1["lower"][11:] * 100000))
s_hiv_inc1_u = create_smoothed_lines(data_x, (inc1["upper"][11:] * 100000))

s_hiv_inc2 = create_smoothed_lines(data_x, (inc2["median"][11:] * 100000))
s_hiv_inc2_l = create_smoothed_lines(data_x, (inc2["lower"][11:] * 100000))
s_hiv_inc2_u = create_smoothed_lines(data_x, (inc2["upper"][11:] * 100000))

# tb incidence
s_tb_inc0 = create_smoothed_lines(data_x, (tb_inc0["median"][11:] * 100000))
s_tb_inc0_l = create_smoothed_lines(data_x, (tb_inc0["lower"][11:] * 100000))
s_tb_inc0_u = create_smoothed_lines(data_x, (tb_inc0["upper"][11:] * 100000))

s_tb_inc1 = create_smoothed_lines(data_x, (tb_inc1["median"][11:] * 100000))
s_tb_inc1_l = create_smoothed_lines(data_x, (tb_inc1["lower"][11:] * 100000))
s_tb_inc1_u = create_smoothed_lines(data_x, (tb_inc1["upper"][11:] * 100000))

s_tb_inc2 = create_smoothed_lines(data_x, (tb_inc2["median"][11:] * 100000))
s_tb_inc2_l = create_smoothed_lines(data_x, (tb_inc2["lower"][11:] * 100000))
s_tb_inc2_u = create_smoothed_lines(data_x, (tb_inc2["upper"][11:] * 100000))

# aids deaths
data_x2 = py0.index[12:].year

s_aids = create_smoothed_lines(data_x2, aids_deaths0["median_aids_deaths_rate_100kpy"][12:])
s_aids_l = create_smoothed_lines(data_x2, aids_deaths0["lower_aids_deaths_rate_100kpy"][12:])
s_aids_u = create_smoothed_lines(data_x2, aids_deaths0["upper_aids_deaths_rate_100kpy"][12:])

s_aids1 = create_smoothed_lines(data_x2, aids_deaths1["median_aids_deaths_rate_100kpy"][12:])
s_aids1_l = create_smoothed_lines(data_x2, aids_deaths1["lower_aids_deaths_rate_100kpy"][12:])
s_aids1_u = create_smoothed_lines(data_x2, aids_deaths1["upper_aids_deaths_rate_100kpy"][12:])

s_aids2 = create_smoothed_lines(data_x2, aids_deaths2["median_aids_deaths_rate_100kpy"][12:])
s_aids2_l = create_smoothed_lines(data_x2, aids_deaths2["lower_aids_deaths_rate_100kpy"][12:])
s_aids2_u = create_smoothed_lines(data_x2, aids_deaths2["upper_aids_deaths_rate_100kpy"][12:])

# tb deaths
s_tb_death = create_smoothed_lines(data_x2, tb_deaths0["median_tb_deaths_rate_100kpy"][12:])
s_tb_death_l = create_smoothed_lines(data_x2, tb_deaths0["lower_tb_deaths_rate_100kpy"][12:])
s_tb_death_u = create_smoothed_lines(data_x2, tb_deaths0["upper_tb_deaths_rate_100kpy"][12:])

s_tb_death1 = create_smoothed_lines(data_x2, tb_deaths1["median_tb_deaths_rate_100kpy"][12:])
s_tb_death1_l = create_smoothed_lines(data_x2, tb_deaths1["lower_tb_deaths_rate_100kpy"][12:])
s_tb_death1_u = create_smoothed_lines(data_x2, tb_deaths1["upper_tb_deaths_rate_100kpy"][12:])

s_tb_death2 = create_smoothed_lines(data_x2, tb_deaths2["median_tb_deaths_rate_100kpy"][12:])
s_tb_death2_l = create_smoothed_lines(data_x2, tb_deaths2["lower_tb_deaths_rate_100kpy"][12:])
s_tb_death2_u = create_smoothed_lines(data_x2, tb_deaths2["upper_tb_deaths_rate_100kpy"][12:])




# %%:  ---------------------------------- PLOTS ---------------------------------- #

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                             sharex=True,
                                             constrained_layout=True,
                                             figsize=(9, 8))
fig.suptitle('')

# HIV incidence
line1, = ax1.plot(data_x, inc0["median"][11:] * 100000)  # create empty plot to set axes

ax1.plot(data_x, s_hiv_inc0[:, 1], "-", color=baseline_colour)
ax1.fill_between(data_x, s_hiv_inc0_l[:, 1], s_hiv_inc0_u[:, 1], color=baseline_colour,
                 alpha=0.2)

ax1.plot(data_x, s_hiv_inc1[:, 1], "-", color=sc1_colour)
ax1.fill_between(data_x, s_hiv_inc1_l[:, 1], s_hiv_inc1_u[:, 1], color=sc1_colour, alpha=0.2)

ax1.plot(data_x, s_hiv_inc2[:, 1], "-", color=sc2_colour)
ax1.fill_between(data_x, s_hiv_inc2_l[:, 1], s_hiv_inc2_u[:, 1], color=sc2_colour, alpha=0.2)

ax1.set_ylim([0, 250])
line1.remove()
ax1.set(title='HIV',
        ylabel='Incidence per 100,000 py')

# TB incidence
ax2.plot(data_x, s_tb_inc0[:, 1], "-", color=baseline_colour)
ax2.fill_between(data_x, s_tb_inc0_l[:, 1], s_tb_inc0_u[:, 1],
                 color=baseline_colour, alpha=0.2)

ax2.plot(data_x, s_tb_inc1[:, 1], "-", color=sc1_colour)
ax2.fill_between(data_x, s_tb_inc1_l[:, 1], s_tb_inc1_u[:, 1], color=sc1_colour,
                 alpha=0.2)

ax2.plot(data_x, s_tb_inc2[:, 1], "-", color=sc2_colour)
ax2.fill_between(data_x, s_tb_inc2_l[:, 1], s_tb_inc2_u[:, 1], color=sc2_colour,
                 alpha=0.2)

ax2.set_ylim([0, 250])

ax2.set(title='TB',
        ylabel='')

# HIV deaths
ax3.plot(data_x2, s_aids[:, 1], "-", color=baseline_colour)
ax3.fill_between(data_x2, s_aids_l[:, 1],
                 s_aids_u[:, 1], color=baseline_colour, alpha=0.2)

ax3.plot(data_x2, s_aids1[:, 1], "-", color=sc1_colour)
ax3.fill_between(data_x2, s_aids1_l[:, 1],
                 s_aids1_u[:, 1], color=sc1_colour, alpha=0.2)

ax3.plot(data_x2, s_aids2[:, 1], "-", color=sc2_colour)
ax3.fill_between(data_x2, s_aids2_l[:, 1],
                 s_aids2_u[:, 1], color=sc2_colour, alpha=0.2)

ax3.set_ylim([0, 100])

ax3.set(title='',
        ylabel='Mortality per 100,000 py')

# TB deaths
ax4.plot(data_x2, s_tb_death[:, 1], "-", color=baseline_colour)
ax4.fill_between(data_x2, s_tb_death_l[:, 1],
                 s_tb_death_u[:, 1], color=baseline_colour, alpha=0.2)

ax4.plot(data_x2, s_tb_death1[:, 1], "-", color=sc1_colour)
ax4.fill_between(data_x2, s_tb_death1_l[:, 1],
                 s_tb_death1_u[:, 1], color=sc1_colour, alpha=0.2)

ax4.plot(data_x2, s_tb_death2[:, 1], "-", color=sc2_colour)
ax4.fill_between(data_x2, s_tb_death2_l[:, 1],
                 s_tb_death2_u[:, 1], color=sc2_colour, alpha=0.2)

ax4.set(title='',
        ylabel='')
ax4.set_ylim([0, 100])

# plt.tick_params(axis="both", which="major", labelsize=10)

# handles for legend
l_baseline = mlines.Line2D([], [], color=baseline_colour, label="Baseline")
l_sc1 = mlines.Line2D([], [], color=sc1_colour, label="Constrained scale-up")
l_sc2 = mlines.Line2D([], [], color=sc2_colour, label="Unconstrained scale-up")

plt.legend(handles=[l_baseline, l_sc1, l_sc2])

# fig.savefig(outputspath / "Epi_outputs_focussed.png")

plt.show()
