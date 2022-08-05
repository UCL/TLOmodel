"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
plots created:
4-panel plot HIV and TB incidence and deaths

"""

import datetime
from pathlib import Path
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import lacroix

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


# -----------------------------------------------------------------------------------------
# %% Plots for health system usage
# -----------------------------------------------------------------------------------------


# ---------------------------------- Fraction HCW time-------------------------------------

# fraction of HCW time
# output difference in fraction HCW time from baseline for each scenario
def summarise_frac_hcws(results_folder):
    capacity0 = extract_results(
        results0,
        module="tlo.methods.healthsystem.summary",
        key="Capacity",
        column="average_Frac_Time_Used_Overall",
    )

    capacity = extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="Capacity",
        column="average_Frac_Time_Used_Overall",
    )

    tmp = (capacity.subtract(capacity0) / capacity0) * 100

    hcw = pd.DataFrame(index=capacity.index, columns=["median", "lower", "upper"])
    hcw["median"] = tmp.median(axis=1)
    hcw["lower"] = tmp.quantile(q=0.025, axis=1)
    hcw["upper"] = tmp.quantile(q=0.975, axis=1)

    return hcw


hcw1 = summarise_frac_hcws(results1)
hcw2 = summarise_frac_hcws(results2)
hcw3 = summarise_frac_hcws(results3)
hcw4 = summarise_frac_hcws(results4)

# ---------------------------------- Appt usage -------------------------------------

years_of_simulation = 26


def summarise_treatment_counts(df_list, treatment_id):
    """ summarise the treatment counts across all draws/runs for one results folder
        requires a list of dataframes with all treatments listed with associated counts
    """
    number_runs = len(df_list)
    number_HSI_by_run = pd.DataFrame(index=np.arange(years_of_simulation), columns=np.arange(number_runs))
    column_names = [
        treatment_id + "_median",
        treatment_id + "_lower",
        treatment_id + "_upper"]
    out = pd.DataFrame(columns=column_names)

    for i in range(number_runs):
        if treatment_id in df_list[i].columns:
            number_HSI_by_run.iloc[:, i] = pd.Series(df_list[i].loc[:, treatment_id])

    out.iloc[:, 0] = number_HSI_by_run.median(axis=1)
    out.iloc[:, 1] = number_HSI_by_run.quantile(q=0.025, axis=1)
    out.iloc[:, 2] = number_HSI_by_run.quantile(q=0.975, axis=1)

    return out


def treatment_counts(results_folder, module, key, column):
    info = get_scenario_info(results_folder)

    df_list = list()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            # check if anything contained in folder (some runs failed)
            folder = results_folder / str(draw) / str(run)
            p: os.DirEntry
            pickles = [p for p in os.scandir(folder) if p.name.endswith('.pickle')]
            if pickles:
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

                new = df[['date', column]].copy()
                df_list.append(pd.DataFrame(new[column].to_list()))

    # for column in each df, get median
    # list of treatment IDs
    list_tx_id = list(df_list[0].columns)
    results = pd.DataFrame(index=np.arange(years_of_simulation))

    for treatment_id in list_tx_id:
        tmp = summarise_treatment_counts(df_list, treatment_id)

        # append output to dataframe
        results = results.join(tmp)

    return results


def treatment_counts_full(results_folder, module, key, column, treatment_id):
    info = get_scenario_info(results_folder)

    df_list = list()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            # check if anything contained in folder (some runs failed)
            folder = results_folder / str(draw) / str(run)
            p: os.DirEntry
            pickles = [p for p in os.scandir(folder) if p.name.endswith('.pickle')]
            if pickles:
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

                new = df[['date', column]].copy()
                df_list.append(pd.DataFrame(new[column].to_list()))

    # join all treatment_id outputs from every draw/run
    results = pd.DataFrame(index=np.arange(years_of_simulation))
    for i in range(len(df_list)):
        tmp = df_list[i][treatment_id]
        # append output to dataframe
        results.loc[:, i] = tmp

    return results


tx_id0 = treatment_counts(results_folder=results0,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID")

tx_id1 = treatment_counts(results_folder=results1,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID")

tx_id2 = treatment_counts(results_folder=results2,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID")

tx_id3 = treatment_counts(results_folder=results3,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID")

tx_id4 = treatment_counts(results_folder=results4,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID")

# get full treatment counts for each draw
tb_test_counts0 = treatment_counts_full(results_folder=results0,
                                        module="tlo.methods.healthsystem.summary",
                                        key="HSI_Event",
                                        column="TREATMENT_ID",
                                        treatment_id="Tb_Test_Screening")

tb_test_counts1 = treatment_counts_full(results_folder=results1,
                                        module="tlo.methods.healthsystem.summary",
                                        key="HSI_Event",
                                        column="TREATMENT_ID",
                                        treatment_id="Tb_Test_Screening")

tb_test_counts2 = treatment_counts_full(results_folder=results2,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID",
                          treatment_id="Tb_Test_Screening")

tb_test_counts3 = treatment_counts_full(results_folder=results3,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID",
                          treatment_id="Tb_Test_Screening")

tb_test_counts4 = treatment_counts_full(results_folder=results4,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID",
                          treatment_id="Tb_Test_Screening")

# tb treatment
tb_tx_counts0 = treatment_counts_full(results_folder=results0,
                                        module="tlo.methods.healthsystem.summary",
                                        key="HSI_Event",
                                        column="TREATMENT_ID",
                                        treatment_id="Tb_Treatment")

tb_tx_counts1 = treatment_counts_full(results_folder=results1,
                                        module="tlo.methods.healthsystem.summary",
                                        key="HSI_Event",
                                        column="TREATMENT_ID",
                                        treatment_id="Tb_Treatment")

tb_tx_counts2 = treatment_counts_full(results_folder=results2,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID",
                          treatment_id="Tb_Treatment")

tb_tx_counts3 = treatment_counts_full(results_folder=results3,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID",
                          treatment_id="Tb_Treatment")

tb_tx_counts4 = treatment_counts_full(results_folder=results4,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID",
                          treatment_id="Tb_Treatment")


# select treatments relating to TB and HIV
tx0 = tx_id0[tx_id0.columns[pd.Series(tx_id0.columns).str.startswith(('Hiv', 'Tb'))]]
# drop lower and upper columns - keep only median
tx0 = tx0.loc[:, ~tx0.columns.str.contains('lower')]
tx0 = tx0.loc[:, ~tx0.columns.str.contains('upper')]
tx0 = tx0.T  # transpose for plotting heatmap
tx0 = tx0.fillna(1)  # replce nan with 0
tx0_norm = tx0.divide(tx0.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx0_norm.loc["Tb_Prevention_Ipt_median"] = tx0_norm.loc["Tb_Prevention_Ipt_median"] / tx0_norm.loc[
    "Tb_Prevention_Ipt_median", 4]
tx0_norm.loc["Hiv_Prevention_Prep_median"] = tx0_norm.loc["Hiv_Prevention_Prep_median"] / tx0_norm.loc[
    "Hiv_Prevention_Prep_median", 13]

tx1 = tx_id1[tx_id1.columns[pd.Series(tx_id1.columns).str.startswith(('Hiv', 'Tb'))]]
tx1 = tx1.loc[:, ~tx1.columns.str.contains('lower')]
tx1 = tx1.loc[:, ~tx1.columns.str.contains('upper')]
tx1 = tx1.T  # transpose for plotting heatmap
tx1 = tx1.fillna(1)  # replce nan with 0
tx1_norm = tx1.divide(tx1.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx1_norm.loc["Tb_Prevention_Ipt_median"] = tx1_norm.loc["Tb_Prevention_Ipt_median"] / tx1_norm.loc[
    "Tb_Prevention_Ipt_median", 4]
tx1_norm.loc["Hiv_Prevention_Prep_median"] = tx1_norm.loc["Hiv_Prevention_Prep_median"] / tx1_norm.loc[
    "Hiv_Prevention_Prep_median", 13]

tx2 = tx_id2[tx_id2.columns[pd.Series(tx_id1.columns).str.startswith(('Hiv', 'Tb'))]]
tx2 = tx2.loc[:, ~tx2.columns.str.contains('lower')]
tx2 = tx2.loc[:, ~tx2.columns.str.contains('upper')]
tx2 = tx2.T  # transpose for plotting heatmap
tx2 = tx2.fillna(1)  # replce nan with 0
tx2_norm = tx2.divide(tx2.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx2_norm.loc["Tb_Prevention_Ipt_median"] = tx2_norm.loc["Tb_Prevention_Ipt_median"] / tx2_norm.loc[
    "Tb_Prevention_Ipt_median", 4]
tx2_norm.loc["Hiv_Prevention_Prep_median"] = tx2_norm.loc["Hiv_Prevention_Prep_median"] / tx2_norm.loc[
    "Hiv_Prevention_Prep_median", 13]

tx3 = tx_id3[tx_id3.columns[pd.Series(tx_id1.columns).str.startswith(('Hiv', 'Tb'))]]
tx3 = tx3.loc[:, ~tx3.columns.str.contains('lower')]
tx3 = tx3.loc[:, ~tx3.columns.str.contains('upper')]
tx3 = tx3.T  # transpose for plotting heatmap
tx3 = tx3.fillna(1)  # replce nan with 0
tx3_norm = tx3.divide(tx3.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx3_norm.loc["Tb_Prevention_Ipt_median"] = tx3_norm.loc["Tb_Prevention_Ipt_median"] / tx3_norm.loc[
    "Tb_Prevention_Ipt_median", 4]
tx3_norm.loc["Hiv_Prevention_Prep_median"] = tx3_norm.loc["Hiv_Prevention_Prep_median"] / tx3_norm.loc[
    "Hiv_Prevention_Prep_median", 13]

tx4 = tx_id4[tx_id4.columns[pd.Series(tx_id1.columns).str.startswith(('Hiv', 'Tb'))]]
tx4 = tx4.loc[:, ~tx4.columns.str.contains('lower')]
tx4 = tx4.loc[:, ~tx4.columns.str.contains('upper')]
tx4 = tx4.T  # transpose for plotting heatmap
tx4 = tx4.fillna(1)  # replce nan with 0
tx4_norm = tx4.divide(tx4.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx4_norm.loc["Tb_Prevention_Ipt_median"] = tx4_norm.loc["Tb_Prevention_Ipt_median"] / tx4_norm.loc[
    "Tb_Prevention_Ipt_median", 4]
tx4_norm.loc["Hiv_Prevention_Prep_median"] = tx4_norm.loc["Hiv_Prevention_Prep_median"] / tx4_norm.loc[
    "Hiv_Prevention_Prep_median", 13]

# rename treatment IDs
appt_types = ["TB test", "HIV test", "TB X-ray", "HIV tx", "VMMC",
              "TB tx", "TB follow-up", "TB IPT", "PrEP"]
tx0_norm.index = appt_types
tx1_norm.index = appt_types
tx2_norm.index = appt_types
tx3_norm.index = appt_types
tx4_norm.index = appt_types

years = list((range(2010, 2036, 1)))

tx0_norm.columns = years
tx1_norm.columns = years
tx2_norm.columns = years
tx3_norm.columns = years
tx4_norm.columns = years

# ---------------------------------- PLOTS ------------------------------------

plt.style.use('ggplot')
cmap = sns.cm.mako

berry = lacroix.colorList('CranRaspberry')
berry_sns = sns.color_palette(berry)  # creates a seaborn palette.

width = 0.15
years_num = pd.Series(years)

# Make plot
fig = plt.figure(figsize=(10, 6))

# heatmap scenario 0?
ax0 = plt.subplot2grid((2, 3), (0, 0))  # 2 rows, 3 cols
sns.heatmap(tx3_norm,
            xticklabels=False,
            yticklabels=1,
            vmin=0,
            vmax=3,
            linewidth=0.5,
            cmap=cmap,
            cbar=True,
            cbar_kws={
                'pad': .02,
                'ticks': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            },
            )
ax0.set_title("Scenario 3", size=10)

# heatmap scenario 2?
ax1 = plt.subplot2grid((2, 3), (1, 0))
sns.heatmap(tx4_norm,
            xticklabels=5,
            yticklabels=1,
            vmin=0,
            vmax=3,
            linewidth=0.5,
            cmap=cmap,
            cbar=True,
            cbar_kws={
                'pad': .02,
                'ticks': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            },
            )
ax1.set_title("Scenario 4", size=10)

# Frac HCW time
ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
ax2.yaxis.tick_right()

ax2.bar(years_num[12:26], hcw1["median"].loc[12:25], width, color=berry[4])
ax2.bar(years_num[12:26] + width, hcw2["median"].loc[12:25], width, color=berry[3])
ax2.bar(years_num[12:26] + (width*2), hcw3["median"].loc[12:25], width, color=berry[2])
ax2.bar(years_num[12:26] + (width*3), hcw4["median"].loc[12:25], width, color=berry[1])

# ax2.plot(years, hcw1["median"], "-", color=berry[4])
# ax2.fill_between(years, hcw1["lower"], hcw1["upper"], color=berry[4], alpha=0.2)
# ax2.plot(years, hcw2["median"], "-", color=berry[3])
# ax2.fill_between(years, hcw2["lower"], hcw2["upper"], color=berry[3], alpha=0.2)
# ax2.plot(years, hcw3["median"], "-", color=berry[2])
# ax2.fill_between(years, hcw3["lower"], hcw3["upper"], color=berry[2], alpha=0.2)
# ax2.plot(years, hcw4["median"], "-", color=berry[1])
# ax2.fill_between(years, hcw4["lower"], hcw4["upper"], color=berry[1], alpha=0.2)

ax2.set_ylabel("% difference", rotation=-90, labelpad=20)
ax2.yaxis.set_label_position("right")
ax2.legend(["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

fig.savefig(outputspath / "HS_use.png")

plt.show()


# -----------------------------------------------------------------------------------------
# %% Plots for epi outputs
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
ax1.plot(inc0.index, inc0["median"] * 100000, "-", color=berry[5])
ax1.fill_between(inc0.index, inc0["lower"] * 100000, inc0["upper"] * 100000, color=berry[5], alpha=0.2)

ax1.plot(inc1.index, inc1["median"] * 100000, "-", color=berry[4])
ax1.fill_between(inc1.index, inc1["lower"] * 100000, inc1["upper"] * 100000, color=berry[4], alpha=0.2)

ax1.plot(inc2.index, inc2["median"] * 100000, "-", color=berry[3])
ax1.fill_between(inc2.index, inc2["lower"] * 100000, inc2["upper"] * 100000, color=berry[3], alpha=0.2)

ax1.plot(inc3.index, inc3["median"] * 100000, "-", color=berry[2])
ax1.fill_between(inc3.index, inc3["lower"] * 100000, inc3["upper"] * 100000, color=berry[2], alpha=0.2)

ax1.plot(inc4.index, inc4["median"] * 100000, "-", color=berry[1])
ax1.fill_between(inc4.index, inc4["lower"] * 100000, inc4["upper"] * 100000, color=berry[1], alpha=0.2)

ax1.set_ylim([0, 700])

ax1.set(title='HIV',
        ylabel='Incidence per 100,000 py')

# TB incidence
ax2.plot(tb_inc0.index, tb_inc0["median"] * 100000, "-", color=berry[5])
ax2.fill_between(tb_inc0.index, tb_inc0["lower"] * 100000, tb_inc0["upper"] * 100000, color=berry[5], alpha=0.2)

ax2.plot(tb_inc1.index, tb_inc1["median"] * 100000, "-", color=berry[4])
ax2.fill_between(tb_inc1.index, tb_inc1["lower"] * 100000, tb_inc1["upper"] * 100000, color=berry[4], alpha=0.2)

ax2.plot(tb_inc2.index, tb_inc2["median"] * 100000, "-", color=berry[3])
ax2.fill_between(tb_inc2.index, tb_inc2["lower"] * 100000, tb_inc2["upper"] * 100000, color=berry[3], alpha=0.2)

ax2.plot(tb_inc3.index, tb_inc3["median"] * 100000, "-", color=berry[2])
ax2.fill_between(tb_inc3.index, tb_inc3["lower"] * 100000, tb_inc3["upper"] * 100000, color=berry[2], alpha=0.2)

ax2.plot(tb_inc4.index, tb_inc4["median"] * 100000, "-", color=berry[1])
ax2.fill_between(tb_inc4.index, tb_inc4["lower"] * 100000, tb_inc4["upper"] * 100000, color=berry[1], alpha=0.2)

ax2.set_ylim([0, 700])

ax2.set(title='TB',
        ylabel='')

# HIV deaths
ax3.plot(py0.index, aids_deaths0["median_aids_deaths_rate_100kpy"], "-", color=berry[5])
ax3.fill_between(py0.index, aids_deaths0["lower_aids_deaths_rate_100kpy"],
                 aids_deaths0["upper_aids_deaths_rate_100kpy"], color=berry[5], alpha=0.2)

ax3.plot(py0.index, aids_deaths1["median_aids_deaths_rate_100kpy"], "-", color=berry[4])
ax3.fill_between(py0.index, aids_deaths1["lower_aids_deaths_rate_100kpy"],
                 aids_deaths1["upper_aids_deaths_rate_100kpy"], color=berry[4], alpha=0.2)

ax3.plot(py0.index, aids_deaths2["median_aids_deaths_rate_100kpy"], "-", color=berry[3])
ax3.fill_between(py0.index, aids_deaths2["lower_aids_deaths_rate_100kpy"],
                 aids_deaths2["upper_aids_deaths_rate_100kpy"], color=berry[3], alpha=0.2)

ax3.plot(py0.index, aids_deaths3["median_aids_deaths_rate_100kpy"], "-", color=berry[2])
ax3.fill_between(py0.index, aids_deaths3["lower_aids_deaths_rate_100kpy"],
                 aids_deaths3["upper_aids_deaths_rate_100kpy"], color=berry[2], alpha=0.2)

ax3.plot(py0.index, aids_deaths4["median_aids_deaths_rate_100kpy"], "-", color=berry[1])
ax3.fill_between(py0.index, aids_deaths4["lower_aids_deaths_rate_100kpy"],
                 aids_deaths4["upper_aids_deaths_rate_100kpy"], color=berry[1], alpha=0.2)

ax3.set_ylim([0, 300])

ax3.set(title='',
        ylabel='Mortality per 100,000 py')

# TB deaths
ax4.plot(py0.index, tb_deaths0["median_tb_deaths_rate_100kpy"], "-", color=berry[5])
ax4.fill_between(py0.index, tb_deaths0["lower_tb_deaths_rate_100kpy"],
                 tb_deaths0["upper_tb_deaths_rate_100kpy"], color=berry[5], alpha=0.2)

ax4.plot(py0.index, tb_deaths1["median_tb_deaths_rate_100kpy"], "-", color=berry[4])
ax4.fill_between(py0.index, tb_deaths1["lower_tb_deaths_rate_100kpy"],
                 tb_deaths1["upper_tb_deaths_rate_100kpy"], color=berry[4], alpha=0.2)

ax4.plot(py0.index, tb_deaths2["median_tb_deaths_rate_100kpy"], "-", color=berry[3])
ax4.fill_between(py0.index, tb_deaths2["lower_tb_deaths_rate_100kpy"],
                 tb_deaths2["upper_tb_deaths_rate_100kpy"], color=berry[3], alpha=0.2)

ax4.plot(py0.index, tb_deaths3["median_tb_deaths_rate_100kpy"], "-", color=berry[2])
ax4.fill_between(py0.index, tb_deaths3["lower_tb_deaths_rate_100kpy"],
                 tb_deaths3["upper_tb_deaths_rate_100kpy"], color=berry[2], alpha=0.2)

ax4.plot(py0.index, tb_deaths4["median_tb_deaths_rate_100kpy"], "-", color=berry[1])
ax4.fill_between(py0.index, tb_deaths4["lower_tb_deaths_rate_100kpy"],
                 tb_deaths4["upper_tb_deaths_rate_100kpy"], color=berry[1], alpha=0.2)

ax4.set(title='',
        ylabel='')
ax4.set_ylim([0, 100])

plt.tick_params(axis="both", which="major", labelsize=10)

plt.legend(labels=["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

fig.savefig(outputspath / "Epi_outputs.png")

plt.show()


# -----------------------------------------------------------------------------------------
# %% TB treatment cascade
# -----------------------------------------------------------------------------------------








# ---------------------------------- TREATMENT COVERAGE ---------------------------------- #

# get scaling factor for numbers of tests performed and treatments requested
# scaling factor 145.39609
sf = extract_results(
    results0,
    module="tlo.methods.population",
    key="scaling_factor",
    column="scaling_factor",
    index="date",
    do_scaling=False)


# tb proportion diagnosed
# todo note this will include false positives and cases from previous year
def tb_proportion_diagnosed(results_folder):

    tb_case = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False
    )

    tb_dx = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbNewDiagnosis",
        index="date",
        do_scaling=False
    )

    prop_dx = tb_dx.divide(tb_case, axis='columns')
    prop_dx_out = pd.DataFrame(index=prop_dx.index, columns=["median", "lower", "upper"])
    prop_dx_out["median"] = prop_dx.median(axis=1)
    prop_dx_out["lower"] = prop_dx.quantile(q=0.025, axis=1)
    prop_dx_out["upper"] = prop_dx.quantile(q=0.975, axis=1)

    # replace values >1 with 1
    prop_dx_out[prop_dx_out > 1] = 1

    return prop_dx_out


def tb_proportion_diagnosed_full(results_folder):

    tb_case = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False
    )

    tb_dx = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbNewDiagnosis",
        index="date",
        do_scaling=False
    )

    prop_dx = tb_dx.divide(tb_case, axis='columns')
    prop_dx = prop_dx.T.reset_index(drop=True).T
    # replace values >1 with 1
    prop_dx[prop_dx > 1] = 1

    return prop_dx


tb_dx0 = tb_proportion_diagnosed(results0)
tb_dx1 = tb_proportion_diagnosed(results1)
tb_dx2 = tb_proportion_diagnosed(results2)
tb_dx3 = tb_proportion_diagnosed(results3)
tb_dx4 = tb_proportion_diagnosed(results4)

tb_dx_full0 = tb_proportion_diagnosed_full(results0)
tb_dx_full1 = tb_proportion_diagnosed_full(results1)
tb_dx_full2 = tb_proportion_diagnosed_full(results2)
tb_dx_full3 = tb_proportion_diagnosed_full(results3)
tb_dx_full4 = tb_proportion_diagnosed_full(results4)


# tb treatment coverage
def tb_tx_coverage(results_folder):
    tx_cov = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbTreatmentCoverage",
        index="date",
        do_scaling=False
    )

    tx_cov.columns = tx_cov.columns.get_level_values(0)
    tx_cov_summary = pd.DataFrame(index=tx_cov.index, columns=["median", "lower", "upper"])
    tx_cov_summary["median"] = tx_cov.median(axis=1)
    tx_cov_summary["lower"] = tx_cov.quantile(q=0.025, axis=1)
    tx_cov_summary["upper"] = tx_cov.quantile(q=0.975, axis=1)

    return tx_cov_summary

def tb_tx_coverage_full(results_folder):
    tx_cov = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbTreatmentCoverage",
        index="date",
        do_scaling=False
    )

    tx_cov.columns = tx_cov.columns.get_level_values(0)

    return tx_cov


tb_tx0 = tb_tx_coverage(results0)
tb_tx1 = tb_tx_coverage(results1)
tb_tx2 = tb_tx_coverage(results2)
tb_tx3 = tb_tx_coverage(results3)
tb_tx4 = tb_tx_coverage(results4)

tb_tx_full0 = tb_tx_coverage_full(results0)
tb_tx_full1 = tb_tx_coverage_full(results1)
tb_tx_full2 = tb_tx_coverage_full(results2)
tb_tx_full3 = tb_tx_coverage_full(results3)
tb_tx_full4 = tb_tx_coverage_full(results4)

# ---------------------------------- PLOTS ---------------------------------- #
# TB test appts and TB proportion diagnosed

x0 = tx_id0["Tb_Test_Screening_median"].values[1:26]
x1 = tx_id1["Tb_Test_Screening_median"].values[1:26]
x2 = tx_id2["Tb_Test_Screening_median"].values[1:26]
x3 = tx_id3["Tb_Test_Screening_median"].values[1:26]
x4 = tx_id4["Tb_Test_Screening_median"].values[1:26]

y = years_num.values[1:26]

z0 = tb_dx0["median"].values
z1 = tb_dx1["median"].values
z2 = tb_dx2["median"].values
z3 = tb_dx3["median"].values
z4 = tb_dx4["median"].values


plt.style.use('ggplot')


fig = plt.figure()
fig.tight_layout()
ax = plt.axes(projection='3d')

ax.plot(y, x0, z0, color=berry[5], label="Scenario 0");
for i in range(len(tb_dx_full0.columns)):
    ax.plot(y, tb_test_counts0.iloc[1:26, i], tb_dx_full0.iloc[:, i], color=berry[5], alpha=0.1);

ax.plot(y, x1, z1, color=berry[4], label="Scenario 1");
for i in range(len(tb_dx_full0.columns)):
    ax.plot(y, tb_test_counts1.iloc[1:26, i], tb_dx_full1.iloc[:, i], color=berry[4], alpha=0.1);

ax.plot(y, x2, z2, color=berry[3], label="Scenario 2");
for i in range(len(tb_dx_full0.columns)):
    ax.plot(y, tb_test_counts2.iloc[1:26, i], tb_dx_full2.iloc[:, i], color=berry[3], alpha=0.1);

ax.plot(y, x3, z3, color=berry[2], label="Scenario 3");
for i in range(len(tb_dx_full0.columns)):
    ax.plot(y, tb_test_counts3.iloc[1:26, i], tb_dx_full3.iloc[:, i], color=berry[2], alpha=0.1);

ax.plot(y, x4, z4, color=berry[1], label="Scenario 4");
for i in range(len(tb_dx_full0.columns)):
    ax.plot(y, tb_test_counts4.iloc[1:26, i], tb_dx_full4.iloc[:, i], color=berry[1], alpha=0.1);

ax.set(facecolor='w')

xLabel = ax.set_xlabel('\nYear', linespacing=1.5)
yLabel = ax.set_ylabel('\nNo. treatment appts', linespacing=2.1)
zLabel = ax.set_zlabel('\nProportion treated', linespacing=1.4)

plt.legend(bbox_to_anchor=(-0.4,0.4), loc="center left", facecolor="white")

# fig.savefig(outputspath / "Tb_diagnosed.png")

plt.show()

#--------------------------------------------
# TB test appts and TB proportion diagnosed

x0 = tx_id0["Tb_Treatment_median"].values[1:26]
x1 = tx_id1["Tb_Treatment_median"].values[1:26]
x2 = tx_id2["Tb_Treatment_median"].values[1:26]
x3 = tx_id3["Tb_Treatment_median"].values[1:26]
x4 = tx_id4["Tb_Treatment_median"].values[1:26]

y = years_num.values[1:26]

z0 = tb_tx0["median"].values
z1 = tb_tx1["median"].values
z2 = tb_tx2["median"].values
z3 = tb_tx3["median"].values
z4 = tb_tx4["median"].values


fig = plt.figure()
fig.tight_layout()
ax = plt.axes(projection='3d')

ax.plot(y, x0, z0, color=berry[5], label="Scenario 0");
for i in range(len(tb_tx_full0.columns)):
    ax.plot(y, tb_tx_counts0.iloc[1:26, i], tb_tx_full0.iloc[:, i], color=berry[5], alpha=0.1);

ax.plot(y, x1, z1, color=berry[4], label="Scenario 1");
for i in range(len(tb_tx_full1.columns)):
    ax.plot(y, tb_tx_counts1.iloc[1:26, i], tb_tx_full1.iloc[:, i], color=berry[4], alpha=0.1);

ax.plot(y, x2, z2, color=berry[3], label="Scenario 2");
for i in range(len(tb_tx_full2.columns)):
    ax.plot(y, tb_tx_counts2.iloc[1:26, i], tb_tx_full2.iloc[:, i], color=berry[3], alpha=0.1);

ax.plot(y, x3, z3, color=berry[2], label="Scenario 3");
for i in range(len(tb_tx_full3.columns)):
    ax.plot(y, tb_tx_counts3.iloc[1:26, i], tb_tx_full3.iloc[:, i], color=berry[2], alpha=0.1);

ax.plot(y, x4, z4, color=berry[1], label="Scenario 4");
for i in range(len(tb_tx_full4.columns)):
    ax.plot(y, tb_tx_counts4.iloc[1:26, i], tb_tx_full4.iloc[:, i], color=berry[1], alpha=0.1);

ax.set(facecolor='w')

xLabel = ax.set_xlabel('\nYear', linespacing=1.5)
yLabel = ax.set_ylabel('\nNo. treatment appts', linespacing=2.1)
zLabel = ax.set_zlabel('\nProportion treated', linespacing=1.4)

ax.set_zlim3d(0.3, 0.8)
ax.set_ylim3d(150, 400)

plt.legend(bbox_to_anchor=(-0.4,0.4), loc="center left", facecolor="white")

# fig.savefig(outputspath / "Tb_treated.png")

plt.show()
