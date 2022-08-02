"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
plots created:
4-panel plot HIV and TB incidence and deaths

"""

import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]
results3 = get_scenario_outputs("scenario3.py", outputspath)[-1]
results4 = get_scenario_outputs("scenario4.py", outputspath)[-1]

# %% Analyse results of runs

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results0 / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results0)

# get basic information about the results
info = get_scenario_info(results0)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results0)


# %% extract results

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

py3 = extract_results(
    results3,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=False
)

py4 = extract_results(
    results4,
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
inc3 = hiv_adult_inc(results3)
inc4 = hiv_adult_inc(results4)


# # Make plot
# fig, ax = plt.subplots()
# ax.plot(inc0.index, inc0["median"], "-", color="C3")
# ax.fill_between(inc0.index, inc0["lower"], inc0["upper"], color="C3", alpha=0.2)
#
# ax.plot(inc1.index, inc1["median"], "-", color="C0")
# ax.fill_between(inc1.index, inc1["lower"], inc1["upper"], color="C0", alpha=0.2)
#
# ax.plot(inc2.index, inc2["median"], "-", color="C2")
# ax.fill_between(inc2.index, inc2["lower"], inc2["upper"], color="C2", alpha=0.2)
#
# ax.plot(inc3.index, inc3["median"], "-", color="C4")
# ax.fill_between(inc3.index, inc3["lower"], inc3["upper"], color="C4", alpha=0.2)
#
# ax.plot(inc4.index, inc4["median"], "-", color="C6")
# ax.fill_between(inc4.index, inc4["lower"], inc4["upper"], color="C6", alpha=0.2)
#
# fig.subplots_adjust(left=0.15)
# plt.title("HIV incidence in adults 15-49")
# plt.ylabel("HIV incidence")
# plt.legend(["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])
#
# plt.show()


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
tb_inc3 = tb_inc(results3)
tb_inc4 = tb_inc(results4)


# # Make plot
# fig, ax = plt.subplots()
# ax.plot(tb_inc0.index, tb_inc0["median"], "-", color="C3")
# ax.fill_between(tb_inc0.index, tb_inc0["lower"], tb_inc0["upper"], color="C3", alpha=0.2)
#
# ax.plot(tb_inc1.index, tb_inc1["median"], "-", color="C0")
# ax.fill_between(tb_inc1.index, tb_inc1["lower"], tb_inc1["upper"], color="C0", alpha=0.2)
#
# ax.plot(tb_inc2.index, tb_inc2["median"], "-", color="C2")
# ax.fill_between(tb_inc2.index, tb_inc2["lower"], tb_inc2["upper"], color="C2", alpha=0.2)
#
# ax.plot(tb_inc3.index, tb_inc3["median"], "-", color="C4")
# ax.fill_between(tb_inc3.index, tb_inc3["lower"], tb_inc3["upper"], color="C4", alpha=0.2)
#
# ax.plot(tb_inc4.index, tb_inc4["median"], "-", color="C6")
# ax.fill_between(tb_inc4.index, tb_inc4["lower"], tb_inc4["upper"], color="C6", alpha=0.2)
#
# fig.subplots_adjust(left=0.15)
# plt.title("Active TB incidence")
# plt.ylabel("Active TB cases per 100,000 population")
# plt.legend(["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])
#
# plt.show()


# ---------------------------------- HIV deaths ---------------------------------- #

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
aids_deaths3 = summarise_aids_deaths(results3, py3)
aids_deaths4 = summarise_aids_deaths(results4, py4)


# ---------------------------------- TB deaths ---------------------------------- #


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
tb_deaths3 = summarise_tb_deaths(results3, py3)
tb_deaths4 = summarise_tb_deaths(results4, py4)

# ---------------------------------- PLOTS ---------------------------------- #

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                             sharex=True,
                                             constrained_layout=True,
                                             figsize=(9, 8))
fig.suptitle('')

# HIV incidence
ax1.plot(inc0.index, inc0["median"], "-", color="C3")
ax1.fill_between(inc0.index, inc0["lower"], inc0["upper"], color="C3", alpha=0.2)

ax1.plot(inc1.index, inc1["median"], "-", color="C0")
ax1.fill_between(inc1.index, inc1["lower"], inc1["upper"], color="C0", alpha=0.2)

ax1.plot(inc2.index, inc2["median"], "-", color="C2")
ax1.fill_between(inc2.index, inc2["lower"], inc2["upper"], color="C2", alpha=0.2)

ax1.plot(inc3.index, inc3["median"], "-", color="C4")
ax1.fill_between(inc3.index, inc3["lower"], inc3["upper"], color="C4", alpha=0.2)

ax1.plot(inc4.index, inc4["median"], "-", color="C6")
ax1.fill_between(inc4.index, inc4["lower"], inc4["upper"], color="C6", alpha=0.2)

ax1.set(title='HIV incidence in adults 15-49',
        ylabel='HIV incidence')

# TB incidence
ax2.plot(tb_inc0.index, tb_inc0["median"], "-", color="C3")
ax2.fill_between(tb_inc0.index, tb_inc0["lower"], tb_inc0["upper"], color="C3", alpha=0.2)

ax2.plot(tb_inc1.index, tb_inc1["median"], "-", color="C0")
ax2.fill_between(tb_inc1.index, tb_inc1["lower"], tb_inc1["upper"], color="C0", alpha=0.2)

ax2.plot(tb_inc2.index, tb_inc2["median"], "-", color="C2")
ax2.fill_between(tb_inc2.index, tb_inc2["lower"], tb_inc2["upper"], color="C2", alpha=0.2)

ax2.plot(tb_inc3.index, tb_inc3["median"], "-", color="C4")
ax2.fill_between(tb_inc3.index, tb_inc3["lower"], tb_inc3["upper"], color="C4", alpha=0.2)

ax2.plot(tb_inc4.index, tb_inc4["median"], "-", color="C6")
ax2.fill_between(tb_inc4.index, tb_inc4["lower"], tb_inc4["upper"], color="C6", alpha=0.2)

ax2.set(title='Active TB incidence',
        ylabel='Active TB/100k')

# HIV deaths
ax3.plot(py0.index, aids_deaths0["median_aids_deaths_rate_100kpy"], "-", color="C3")
ax3.fill_between(py0.index, aids_deaths0["lower_aids_deaths_rate_100kpy"],
                 aids_deaths0["upper_aids_deaths_rate_100kpy"], color="C3", alpha=0.2)

ax3.plot(py0.index, aids_deaths1["median_aids_deaths_rate_100kpy"], "-", color="C0")
ax3.fill_between(py0.index, aids_deaths1["lower_aids_deaths_rate_100kpy"],
                 aids_deaths1["upper_aids_deaths_rate_100kpy"], color="C0", alpha=0.2)

ax3.plot(py0.index, aids_deaths2["median_aids_deaths_rate_100kpy"], "-", color="C2")
ax3.fill_between(py0.index, aids_deaths2["lower_aids_deaths_rate_100kpy"],
                 aids_deaths2["upper_aids_deaths_rate_100kpy"], color="C2", alpha=0.2)

ax3.plot(py0.index, aids_deaths3["median_aids_deaths_rate_100kpy"], "-", color="C4")
ax3.fill_between(py0.index, aids_deaths3["lower_aids_deaths_rate_100kpy"],
                 aids_deaths3["upper_aids_deaths_rate_100kpy"], color="C4", alpha=0.2)

ax3.plot(py0.index, aids_deaths4["median_aids_deaths_rate_100kpy"], "-", color="C6")
ax3.fill_between(py0.index, aids_deaths4["lower_aids_deaths_rate_100kpy"],
                 aids_deaths4["upper_aids_deaths_rate_100kpy"], color="C6", alpha=0.2)

ax3.set(title='Mortality due to AIDS (inc TB)',
        ylabel='Mortality/100k')

# TB deaths
ax4.plot(py0.index, tb_deaths0["median_tb_deaths_rate_100kpy"], "-", color="C3")
ax4.fill_between(py0.index, tb_deaths0["lower_tb_deaths_rate_100kpy"],
                 tb_deaths0["upper_tb_deaths_rate_100kpy"], color="C3", alpha=0.2)

ax4.plot(py0.index, tb_deaths1["median_tb_deaths_rate_100kpy"], "-", color="C0")
ax4.fill_between(py0.index, tb_deaths1["lower_tb_deaths_rate_100kpy"],
                 tb_deaths1["upper_tb_deaths_rate_100kpy"], color="C0", alpha=0.2)

ax4.plot(py0.index, tb_deaths2["median_tb_deaths_rate_100kpy"], "-", color="C2")
ax4.fill_between(py0.index, tb_deaths2["lower_tb_deaths_rate_100kpy"],
                 tb_deaths2["upper_tb_deaths_rate_100kpy"], color="C2", alpha=0.2)

ax4.plot(py0.index, tb_deaths3["median_tb_deaths_rate_100kpy"], "-", color="C4")
ax4.fill_between(py0.index, tb_deaths3["lower_tb_deaths_rate_100kpy"],
                 tb_deaths3["upper_tb_deaths_rate_100kpy"], color="C4", alpha=0.2)

ax4.plot(py0.index, tb_deaths4["median_tb_deaths_rate_100kpy"], "-", color="C6")
ax4.fill_between(py0.index, tb_deaths4["lower_tb_deaths_rate_100kpy"],
                 tb_deaths4["upper_tb_deaths_rate_100kpy"], color="C6", alpha=0.2)

ax4.set(title='Mortality due to TB',
        ylabel='Mortality/100k')

plt.tick_params(axis="both", which="major", labelsize=10)

plt.legend(labels=["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])
plt.show()


# ----------------------------------------------------------------------------
# ---------------------------------- TREATMENT COVERAGE ---------------------------------- #

# hiv treatment coverage
def hiv_tx_coverage(results_folder):
    hiv_cov = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="art_coverage_adult",
        index="date",
        do_scaling=False
    )

    hiv_cov.columns = hiv_cov.columns.get_level_values(0)
    hiv_cov_summary = pd.DataFrame(index=hiv_cov.index, columns=["median", "lower", "upper"])
    hiv_cov_summary["median"] = hiv_cov.median(axis=1)
    hiv_cov_summary["lower"] = hiv_cov.quantile(q=0.025, axis=1)
    hiv_cov_summary["upper"] = hiv_cov.quantile(q=0.975, axis=1)

    return hiv_cov_summary


hiv_tx0 = hiv_tx_coverage(results0)
hiv_tx1 = hiv_tx_coverage(results1)
hiv_tx2 = hiv_tx_coverage(results2)
hiv_tx3 = hiv_tx_coverage(results3)
hiv_tx4 = hiv_tx_coverage(results4)

# Make plot
fig, ax = plt.subplots()
ax.plot(hiv_tx0.index, hiv_tx0["median"], "-", color="C3")
ax.fill_between(hiv_tx0.index, hiv_tx0["lower"], hiv_tx0["upper"], color="C3", alpha=0.2)

ax.plot(hiv_tx1.index, hiv_tx1["median"], "-", color="C0")
ax.fill_between(hiv_tx1.index, hiv_tx1["lower"], hiv_tx1["upper"], color="C0", alpha=0.2)

ax.plot(hiv_tx2.index, hiv_tx2["median"], "-", color="C2")
ax.fill_between(hiv_tx2.index, hiv_tx2["lower"], hiv_tx2["upper"], color="C2", alpha=0.2)

ax.plot(hiv_tx3.index, hiv_tx3["median"], "-", color="C4")
ax.fill_between(hiv_tx3.index, hiv_tx3["lower"], hiv_tx3["upper"], color="C4", alpha=0.2)

ax.plot(hiv_tx4.index, hiv_tx4["median"], "-", color="C6")
ax.fill_between(hiv_tx4.index, hiv_tx4["lower"], hiv_tx4["upper"], color="C6", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.ylim((0, 1.0))
plt.title("HIV treatment coverage")
plt.ylabel("HIV treatment coverage")
plt.legend(["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

plt.show()


# ---------------------------------- TB TREATMENT DELAY ---------------------------------- #


def extract_tx_delay(results_folder: Path,
                     module: str,
                     key: str,
                     column: str = None,
                     ):
    """Utility function to unpack results
    edited version for utils.py
    """

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    # Collect results from each draw/run
    res = dict()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            draw_run = (draw, run)

            try:
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                test = df[column]
                test2 = test.apply(pd.to_numeric, errors="coerce")
                res[draw_run] = test2

            except KeyError:
                # Some logs could not be found - probably because this run failed.
                res[draw_run] = None

    return res


tb_tx_delay_adult_sc0_dict = extract_tx_delay(results_folder=results0,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayAdults")

tb_tx_delay_adult_sc1_dict = extract_tx_delay(results_folder=results1,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayAdults")

tb_tx_delay_adult_sc2_dict = extract_tx_delay(results_folder=results2,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayAdults")

tb_tx_delay_adult_sc3_dict = extract_tx_delay(results_folder=results3,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayAdults")

tb_tx_delay_adult_sc4_dict = extract_tx_delay(results_folder=results4,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayAdults")

# convert dict to dataframe
tb_tx_delay_adult_sc0 = pd.DataFrame(tb_tx_delay_adult_sc0_dict.items())
tb_tx_delay_adult_sc1 = pd.DataFrame(tb_tx_delay_adult_sc1_dict.items())
tb_tx_delay_adult_sc2 = pd.DataFrame(tb_tx_delay_adult_sc2_dict.items())
tb_tx_delay_adult_sc3 = pd.DataFrame(tb_tx_delay_adult_sc3_dict.items())
tb_tx_delay_adult_sc4 = pd.DataFrame(tb_tx_delay_adult_sc4_dict.items())

# need to collapse all draws/runs together
# set up empty list with columns for each year
# values will be variable length lists of delays
years = list((range(2010, 2035, 1)))


def summarise_tx_delay(treatment_delay_df):
    """
    extract all treatment delays from all draws/runs
    for each scenario and collapse into lists, with
    one list per year
    """
    list_delays = [[] for i in range(25)]

    # for each row of tb_tx_delay_adult_sc0 0-14 [draws, runs]:
    for i in range(treatment_delay_df.shape[0]):

        # separate each row into its arrays 0-25 [years]
        tmp = treatment_delay_df.loc[i, 1]

        # combine them into a list, with items separated from array
        # e.g. tmp[0] has values for 2010
        for j in range(25):
            tmp2 = tmp[j]

            list_delays[j] = [*list_delays[j], *tmp2]

    return list_delays


list_tx_delay0 = summarise_tx_delay(tb_tx_delay_adult_sc0)
list_tx_delay1 = summarise_tx_delay(tb_tx_delay_adult_sc1)
list_tx_delay2 = summarise_tx_delay(tb_tx_delay_adult_sc2)
list_tx_delay3 = summarise_tx_delay(tb_tx_delay_adult_sc3)
list_tx_delay4 = summarise_tx_delay(tb_tx_delay_adult_sc4)

# convert lists to df
# todo note nans are both false positive and fillers for dataframe
delay0 = pd.DataFrame(list_tx_delay0).T
delay0.columns = years
# convert wide to long format
delay0 = delay0.reset_index()
delay0_scatter = pd.melt(delay0, id_vars='index', value_vars=years)

delay1 = pd.DataFrame(list_tx_delay1).T
delay1.columns = years
# convert wide to long format
delay1 = delay1.reset_index()
delay1_scatter = pd.melt(delay1, id_vars='index', value_vars=years)

delay2 = pd.DataFrame(list_tx_delay2).T
delay2.columns = years
# convert wide to long format
delay2 = delay2.reset_index()
delay2_scatter = pd.melt(delay2, id_vars='index', value_vars=years)

delay3 = pd.DataFrame(list_tx_delay3).T
delay3.columns = years
# convert wide to long format
delay3 = delay3.reset_index()
delay3_scatter = pd.melt(delay3, id_vars='index', value_vars=years)


## plots
plt.scatter(delay1_scatter.variable, delay1_scatter.value, s=10, alpha=0.4)

plt.scatter(delay2_scatter.variable, delay2_scatter.value, s=10, alpha=0.4)
# plt.ylim((0, 1095))
plt.title("TB treatment delays")
plt.ylabel("Days from diagnosis to treatment")
plt.legend(["Scenario 1", "Scenario 2"])

plt.show()


# scenario 1 delays 2030
delay1_hist = delay1_scatter.loc[delay1_scatter['variable'] == 2030]
delay1_hist = delay1_hist.loc[(delay1_hist['value'] <= 365) & (delay1_hist['value'] >= 0)]

delay2_hist = delay2_scatter.loc[delay2_scatter['variable'] == 2030]
delay2_hist = delay2_hist.loc[(delay2_hist['value'] <= 365) & (delay2_hist['value'] >= 0)]


plt.hist(delay1_hist.value, range=[0, 50], bins=60, alpha=0.3)
plt.hist(delay2_hist.value, range=[0, 50], bins=60, alpha=0.3)
plt.ylabel("Frequency")
plt.title("TB treatment delays 2030")
plt.legend(["Scenario 1", "Scenario 2"])
plt.show()


plt.hist2d(delay2_scatter.value, delay2_scatter.variable, bins=10, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.show()
