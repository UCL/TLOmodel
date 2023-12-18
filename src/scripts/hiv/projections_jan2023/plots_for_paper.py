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
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import lacroix
import math

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


# %%:  ---------------------------------- Appt usage -------------------------------------

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

tx2 = tx_id2[tx_id2.columns[pd.Series(tx_id2.columns).str.startswith(('Hiv', 'Tb'))]]
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

# rename treatment IDs
appt_types = ["TB test", "HIV test", "HIV tx", "VMMC", "HIV \n infant prophylaxis", "TB X-ray",
              "TB tx", "TB follow-up", "TB IPT", "PrEP"]
tx0_norm.index = appt_types
tx1_norm.index = appt_types
tx2_norm.index = appt_types

# insert zeros for IPT and PrEP pre-introduction (actual values are slightly above 0)
tx0_norm.loc['TB IPT'][0:4] = 0
tx1_norm.loc['TB IPT'][0:4] = 0
tx2_norm.loc['TB IPT'][0:4] = 0

tx0_norm.loc['PrEP'][0:8] = 0
tx1_norm.loc['PrEP'][0:8] = 0
tx2_norm.loc['PrEP'][0:8] = 0


years = list((range(2010, 2036, 1)))

tx0_norm.columns = years
tx1_norm.columns = years
tx2_norm.columns = years


# %%:  ---------------------------------- PLOTS ------------------------------------

plt.style.use('ggplot')
cmap = sns.cm.mako_r

width = 0.1
years_num = pd.Series(years)

tx0_norm.columns = years_num
tx1_norm.columns = years_num
tx2_norm.columns = years_num

# Make plot
fig, axs = plt.subplots(ncols=3, nrows=1,
                        # sharex=True,
                        # sharey=True,
                        constrained_layout=True,
                        figsize=(14, 5),
                        gridspec_kw=dict(width_ratios=[4,4,4]))
sns.heatmap(tx0_norm,
                 xticklabels=5,
                 yticklabels=1,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=False,
            ax=axs[0]
            )
axs[0].set_title("Baseline", size=10)

sns.heatmap(tx1_norm,
                 xticklabels=5,
                 yticklabels=False,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=False,
            ax=axs[1]
            )
axs[1].set_title("Constrained scale-up", size=10)

sns.heatmap(tx2_norm,
                 xticklabels=5,
                 yticklabels=False,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=True,
            ax=axs[2]
            )
axs[2].set_title("Constrained scale-up no constraints", size=10)

plt.tick_params(axis="both", which="major", labelsize=9)
fig.savefig(outputspath / "HS_use.png")
plt.show()




# frac HCW time

berry = lacroix.colorList('CranRaspberry')
baseline_colour = berry[5]
sc1_colour = berry[3]
sc2_colour = berry[2]


fig, ax = plt.subplots()

ax.bar(years_num[13:26], hcw1["median"].loc[13:25], width, color=sc1_colour)
ax.bar(years_num[13:26] + width, hcw2["median"].loc[13:25], width, color=sc2_colour)

ax.set_ylabel("% difference", rotation=90, labelpad=15)
ax.set_ylim([-10, 10])

ax.yaxis.set_label_position("left")
ax.legend(["Constrained scale-up", "Constrained scale-up \n no constraints"], frameon=False)
plt.tight_layout()
fig.savefig(outputspath / "Frac_HWC_time.png")
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


# %%:  ---------------------------------- PLOTS ---------------------------------- #

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                             sharex=True,
                                             constrained_layout=True,
                                             figsize=(9, 8))
fig.suptitle('')

# HIV incidence
ax1.plot(inc0.index, inc0["median"] * 100000, "-", color=baseline_colour)
ax1.fill_between(inc0.index, inc0["lower"] * 100000, inc0["upper"] * 100000, color=baseline_colour, alpha=0.2)

ax1.plot(inc1.index, inc1["median"] * 100000, "-", color=sc1_colour)
ax1.fill_between(inc1.index, inc1["lower"] * 100000, inc1["upper"] * 100000, color=sc1_colour, alpha=0.2)

ax1.plot(inc2.index, inc2["median"] * 100000, "-", color=sc2_colour)
ax1.fill_between(inc2.index, inc2["lower"] * 100000, inc2["upper"] * 100000, color=sc2_colour, alpha=0.2)

ax1.set_ylim([0, 700])

ax1.set(title='HIV',
        ylabel='Incidence per 100,000 py')

# TB incidence
ax2.plot(tb_inc0.index, tb_inc0["median"] * 100000, "-", color=baseline_colour)
ax2.fill_between(tb_inc0.index, tb_inc0["lower"] * 100000, tb_inc0["upper"] * 100000, color=baseline_colour, alpha=0.2)

ax2.plot(tb_inc1.index, tb_inc1["median"] * 100000, "-", color=sc1_colour)
ax2.fill_between(tb_inc1.index, tb_inc1["lower"] * 100000, tb_inc1["upper"] * 100000, color=sc1_colour, alpha=0.2)

ax2.plot(tb_inc2.index, tb_inc2["median"] * 100000, "-", color=sc2_colour)
ax2.fill_between(tb_inc2.index, tb_inc2["lower"] * 100000, tb_inc2["upper"] * 100000, color=sc2_colour, alpha=0.2)

ax2.set_ylim([0, 700])

ax2.set(title='TB',
        ylabel='')

# HIV deaths
ax3.plot(py0.index, aids_deaths0["median_aids_deaths_rate_100kpy"], "-", color=baseline_colour)
ax3.fill_between(py0.index, aids_deaths0["lower_aids_deaths_rate_100kpy"],
                 aids_deaths0["upper_aids_deaths_rate_100kpy"], color=baseline_colour, alpha=0.2)

ax3.plot(py0.index, aids_deaths1["median_aids_deaths_rate_100kpy"], "-", color=sc1_colour)
ax3.fill_between(py0.index, aids_deaths1["lower_aids_deaths_rate_100kpy"],
                 aids_deaths1["upper_aids_deaths_rate_100kpy"], color=sc1_colour, alpha=0.2)

ax3.plot(py0.index, aids_deaths2["median_aids_deaths_rate_100kpy"], "-", color=sc2_colour)
ax3.fill_between(py0.index, aids_deaths2["lower_aids_deaths_rate_100kpy"],
                 aids_deaths2["upper_aids_deaths_rate_100kpy"], color=sc2_colour, alpha=0.2)

ax3.set_ylim([0, 300])

ax3.set(title='',
        ylabel='Mortality per 100,000 py')

# TB deaths
ax4.plot(py0.index, tb_deaths0["median_tb_deaths_rate_100kpy"], "-", color=baseline_colour)
ax4.fill_between(py0.index, tb_deaths0["lower_tb_deaths_rate_100kpy"],
                 tb_deaths0["upper_tb_deaths_rate_100kpy"], color=baseline_colour, alpha=0.2)

ax4.plot(py0.index, tb_deaths1["median_tb_deaths_rate_100kpy"], "-", color=sc1_colour)
ax4.fill_between(py0.index, tb_deaths1["lower_tb_deaths_rate_100kpy"],
                 tb_deaths1["upper_tb_deaths_rate_100kpy"], color=sc1_colour, alpha=0.2)

ax4.plot(py0.index, tb_deaths2["median_tb_deaths_rate_100kpy"], "-", color=sc2_colour)
ax4.fill_between(py0.index, tb_deaths2["lower_tb_deaths_rate_100kpy"],
                 tb_deaths2["upper_tb_deaths_rate_100kpy"], color=sc2_colour, alpha=0.2)

ax4.set(title='',
        ylabel='')
ax4.set_ylim([0, 100])

plt.tick_params(axis="both", which="major", labelsize=10)

# handles for legend
l_baseline = mlines.Line2D([], [], color=baseline_colour, label="Baseline")
l_sc1 = mlines.Line2D([], [], color=sc1_colour, label="Constrained scale-up")
l_sc2 = mlines.Line2D([], [], color=sc2_colour, label="Constrained scale-up \n no constraints")

plt.legend(handles=[l_baseline, l_sc1, l_sc2])

fig.savefig(outputspath / "Epi_outputs.png")

plt.show()

# %% EPI OUTPUTS - start at 2022

# %%:  ---------------------------------- PLOTS ---------------------------------- #

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                             sharex=True,
                                             constrained_layout=True,
                                             figsize=(9, 8))
fig.suptitle('')

# HIV incidence
ax1.plot(inc0.index[11:], inc0["median"][11:] * 100000, "-", color=baseline_colour)
ax1.fill_between(inc0.index[11:], inc0["lower"][11:] * 100000, inc0["upper"][11:] * 100000, color=baseline_colour, alpha=0.2)

ax1.plot(inc1.index[11:], inc1["median"][11:] * 100000, "-", color=sc1_colour)
ax1.fill_between(inc1.index[11:], inc1["lower"][11:] * 100000, inc1["upper"][11:] * 100000, color=sc1_colour, alpha=0.2)

ax1.plot(inc2.index[11:], inc2["median"][11:] * 100000, "-", color=sc2_colour)
ax1.fill_between(inc2.index[11:], inc2["lower"][11:] * 100000, inc2["upper"][11:] * 100000, color=sc2_colour, alpha=0.2)

ax1.set_ylim([0, 250])

ax1.set(title='HIV',
        ylabel='Incidence per 100,000 py')

# TB incidence
ax2.plot(tb_inc0.index[11:], tb_inc0["median"][11:] * 100000, "-", color=baseline_colour)
ax2.fill_between(tb_inc0.index[11:], tb_inc0["lower"][11:] * 100000, tb_inc0["upper"][11:] * 100000, color=baseline_colour, alpha=0.2)

ax2.plot(tb_inc1.index[11:], tb_inc1["median"][11:] * 100000, "-", color=sc1_colour)
ax2.fill_between(tb_inc1.index[11:], tb_inc1["lower"][11:] * 100000, tb_inc1["upper"][11:] * 100000, color=sc1_colour, alpha=0.2)

ax2.plot(tb_inc2.index[11:], tb_inc2["median"][11:] * 100000, "-", color=sc2_colour)
ax2.fill_between(tb_inc2.index[11:], tb_inc2["lower"][11:] * 100000, tb_inc2["upper"][11:] * 100000, color=sc2_colour, alpha=0.2)

ax2.set_ylim([0, 250])

ax2.set(title='TB',
        ylabel='')

# HIV deaths
ax3.plot(py0.index[12:], aids_deaths0["median_aids_deaths_rate_100kpy"][12:], "-", color=baseline_colour)
ax3.fill_between(py0.index[12:], aids_deaths0["lower_aids_deaths_rate_100kpy"][12:],
                 aids_deaths0["upper_aids_deaths_rate_100kpy"][12:], color=baseline_colour, alpha=0.2)

ax3.plot(py0.index[12:], aids_deaths1["median_aids_deaths_rate_100kpy"][12:], "-", color=sc1_colour)
ax3.fill_between(py0.index[12:], aids_deaths1["lower_aids_deaths_rate_100kpy"][12:],
                 aids_deaths1["upper_aids_deaths_rate_100kpy"][12:], color=sc1_colour, alpha=0.2)

ax3.plot(py0.index[12:], aids_deaths2["median_aids_deaths_rate_100kpy"][12:], "-", color=sc2_colour)
ax3.fill_between(py0.index[12:], aids_deaths2["lower_aids_deaths_rate_100kpy"][12:],
                 aids_deaths2["upper_aids_deaths_rate_100kpy"][12:], color=sc2_colour, alpha=0.2)

ax3.set_ylim([0, 100])

ax3.set(title='',
        ylabel='Mortality per 100,000 py')

# TB deaths
ax4.plot(py0.index[12:], tb_deaths0["median_tb_deaths_rate_100kpy"][12:], "-", color=baseline_colour)
ax4.fill_between(py0.index[12:], tb_deaths0["lower_tb_deaths_rate_100kpy"][12:],
                 tb_deaths0["upper_tb_deaths_rate_100kpy"][12:], color=baseline_colour, alpha=0.2)

ax4.plot(py0.index[12:], tb_deaths1["median_tb_deaths_rate_100kpy"][12:], "-", color=sc1_colour)
ax4.fill_between(py0.index[12:], tb_deaths1["lower_tb_deaths_rate_100kpy"][12:],
                 tb_deaths1["upper_tb_deaths_rate_100kpy"][12:], color=sc1_colour, alpha=0.2)

ax4.plot(py0.index[12:], tb_deaths2["median_tb_deaths_rate_100kpy"][12:], "-", color=sc2_colour)
ax4.fill_between(py0.index[12:], tb_deaths2["lower_tb_deaths_rate_100kpy"][12:],
                 tb_deaths2["upper_tb_deaths_rate_100kpy"][12:], color=sc2_colour, alpha=0.2)

ax4.set(title='',
        ylabel='')
ax4.set_ylim([0, 100])

plt.tick_params(axis="both", which="major", labelsize=10)

# handles for legend
l_baseline = mlines.Line2D([], [], color=baseline_colour, label="Baseline")
l_sc1 = mlines.Line2D([], [], color=sc1_colour, label="Constrained scale-up")
l_sc2 = mlines.Line2D([], [], color=sc2_colour, label="Constrained scale-up \n no constraints")

plt.legend(handles=[l_baseline, l_sc1, l_sc2])

fig.savefig(outputspath / "Epi_outputs_focussed.png")

plt.show()


# %%:  ---------------------------------- DALYS ---------------------------------- #
TARGET_PERIOD = (Date(2023, 1, 1), Date(2036, 1, 1))


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
    # combine two labels for non-AIDS TB (this now fixed in latest code)
    dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
    dalys.drop(['non_AIDS_TB'], inplace=True)
    out = pd.DataFrame()
    out['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
    out['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
    out['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)

    return out


dalys0 = return_daly_summary(results0)
dalys1 = return_daly_summary(results1)
dalys2 = return_daly_summary(results2)


dalys0.loc['Column_Total'] = dalys0.sum(numeric_only=True, axis=0)
dalys1.loc['Column_Total'] = dalys1.sum(numeric_only=True, axis=0)
dalys2.loc['Column_Total'] = dalys2.sum(numeric_only=True, axis=0)

# create full table for export
daly_table = pd.DataFrame()
daly_table['scenario0'] = dalys0['median'].astype(str) + \
                          " (" + dalys0['lower'].astype(str) + " - " + \
                          dalys0['upper'].astype(str) + ")"
daly_table['scenario1'] = dalys1['median'].astype(str) + \
                          " (" + dalys1['lower'].astype(str) + " - " + \
                          dalys1['upper'].astype(str) + ")"
daly_table['scenario2'] = dalys2['median'].astype(str) + \
                          " (" + dalys2['lower'].astype(str) + " - " + \
                          dalys2['upper'].astype(str) + ")"

daly_table.to_csv(outputspath / "daly_summary.csv")


# extract dalys averted by each scenario relative to scenario 0
# comparison should be run-by-run
full_dalys0 = extract_results(
    results0,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys0.loc['Column_Total'] = full_dalys0.sum(numeric_only=True, axis=0)

full_dalys1 = extract_results(
    results1,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys1.loc['Column_Total'] = full_dalys1.sum(numeric_only=True, axis=0)

full_dalys2 = extract_results(
    results2,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys2.loc['Column_Total'] = full_dalys2.sum(numeric_only=True, axis=0)

writer = pd.ExcelWriter(r"outputs/t.mangal@imperial.ac.uk/full_dalys.xlsx")
full_dalys0.to_excel(writer, sheet_name='sc0')
full_dalys1.to_excel(writer, sheet_name='sc1')
full_dalys2.to_excel(writer, sheet_name='sc2')
writer.save()


# DALYs averted: baseline - scenario
# positive value will be DALYs averted due to interventions
# negative value will be higher DALYs reported, therefore increased health burden
sc1_sc0 = full_dalys0.subtract(full_dalys1, fill_value=0)
sc1_sc0_median = sc1_sc0.median(axis=1)
sc1_sc0_lower = sc1_sc0.quantile(q=0.025, axis=1)
sc1_sc0_upper = sc1_sc0.quantile(q=0.975, axis=1)

sc2_sc0 = full_dalys0.subtract(full_dalys2, fill_value=0)
sc2_sc0_median = sc2_sc0.median(axis=1)
sc2_sc0_lower = sc2_sc0.quantile(q=0.025, axis=1)
sc2_sc0_upper = sc2_sc0.quantile(q=0.975, axis=1)


# create full table for export
daly_averted_table = pd.DataFrame()
daly_averted_table['cause'] = sc1_sc0_median.index
daly_averted_table['scenario1_med'] = [int(round(x,-3)) for x in sc1_sc0_median]
daly_averted_table['scenario1_low'] = [int(round(x,-3)) for x in sc1_sc0_lower]
daly_averted_table['scenario1_upp'] = [int(round(x,-3)) for x in sc1_sc0_upper]
daly_averted_table['scenario2_med'] = [int(round(x,-3)) for x in sc2_sc0_median]
daly_averted_table['scenario2_low'] = [int(round(x,-3)) for x in sc2_sc0_lower]
daly_averted_table['scenario2_upp'] = [int(round(x,-3)) for x in sc2_sc0_upper]


daly_averted_table.to_csv(outputspath / "daly_averted_summary.csv")

aids_dalys_diff = [sc1_sc0_median['AIDS'],
              sc2_sc0_median['AIDS']]
tb_dalys_diff = [sc1_sc0_median['TB (non-AIDS)'],
              sc2_sc0_median['TB (non-AIDS)']]
total_dalys_diff = [sc1_sc0_median['Column_Total'],
              sc2_sc0_median['Column_Total']]


aids_dalys = [dalys0.loc['AIDS', 'median'],
              dalys1.loc['AIDS', 'median'],
              dalys2.loc['AIDS', 'median']]
tb_dalys = [dalys0.loc['TB (non-AIDS)', 'median'],
              dalys1.loc['TB (non-AIDS)', 'median'],
              dalys2.loc['TB (non-AIDS)', 'median']]
total_dalys = [dalys0.loc['Column_Total', 'median'],
              dalys1.loc['Column_Total', 'median'],
              dalys2.loc['Column_Total', 'median']]

labels = ['Baseline', 'Constrained scale-up', 'Constrained scale-up \n no constraints']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

# plots of total DALYs
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, aids_dalys, width, label='AIDS', color=baseline_colour)
rects2 = ax.bar(x, tb_dalys, width, label='TB', color=sc1_colour)
rects3 = ax.bar(x + width, total_dalys, width, label='Total', color=sc2_colour)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('DALYs')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
fig.savefig(outputspath / "DALYS.png")

plt.show()



# plots of diff in DALYs from baseline
labels = ['Constrained scale-up', 'Constrained scale-up \n no constraints']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width, aids_dalys_diff, width, label='AIDS', color=baseline_colour)
rects2 = ax.bar(x, tb_dalys_diff, width, label='TB', color=sc1_colour)
rects3 = ax.bar(x + width, total_dalys_diff, width, label='Total', color=sc2_colour)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('DALYs')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
fig.savefig(outputspath / "DALYS_averted.png")

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


tb_dx_full0 = tb_proportion_diagnosed_full(results0)
tb_dx_full1 = tb_proportion_diagnosed_full(results1)
tb_dx_full2 = tb_proportion_diagnosed_full(results2)


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

tb_tx_full0 = tb_tx_coverage_full(results0)
tb_tx_full1 = tb_tx_coverage_full(results1)
tb_tx_full2 = tb_tx_coverage_full(results2)


# %%:  ---------------------------------- PLOTS ---------------------------------- #

scale = sf[0][0].values[0]


# Make plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                             sharex=True,
                                             constrained_layout=True,
                                             figsize=(9, 8))
fig.suptitle('')

# TB tests
ax1.plot(tb_dx0.index, tx_id0["Tb_Test_Screening_median"][1:26] * sf[0][0].values[0], "-", color=baseline_colour)
ax1.fill_between(tb_dx0.index, tx_id0["Tb_Test_Screening_lower"][1:26] * sf[0][0].values[0],
                 tx_id0["Tb_Test_Screening_upper"][1:26] * sf[0][0].values[0], color=baseline_colour, alpha=0.2)

ax1.plot(tb_dx0.index, tx_id1["Tb_Test_Screening_median"][1:26] * sf[0][0].values[0], "-", color=sc1_colour)
ax1.fill_between(tb_dx0.index, tx_id1["Tb_Test_Screening_lower"][1:26] * sf[0][0].values[0],
                 tx_id1["Tb_Test_Screening_upper"][1:26] * sf[0][0].values[0], color=sc1_colour, alpha=0.2)

ax1.plot(tb_dx0.index, tx_id2["Tb_Test_Screening_median"][1:26] * sf[0][0].values[0], "-", color=sc2_colour)
ax1.fill_between(tb_dx0.index, tx_id2["Tb_Test_Screening_lower"][1:26] * sf[0][0].values[0],
                 tx_id2["Tb_Test_Screening_upper"][1:26] * sf[0][0].values[0], color=sc2_colour, alpha=0.2)
# ax1.set_ylim([5000000, 10000000])

ax1.set(title='',
       ylabel='No. screening appts')

ax1.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])

# TB start treatment
ax2.plot(tb_dx0.index, tx_id0["Tb_Treatment_median"][1:26] * sf[0][0].values[0], "-", color=baseline_colour)
ax2.fill_between(tb_dx0.index, tx_id0["Tb_Treatment_lower"][1:26] * sf[0][0].values[0],
                 tx_id0["Tb_Treatment_upper"][1:26] * sf[0][0].values[0], color=baseline_colour, alpha=0.2)

ax2.plot(tb_dx0.index, tx_id1["Tb_Treatment_median"][1:26] * sf[0][0].values[0], "-", color=sc1_colour)
ax2.fill_between(tb_dx0.index, tx_id1["Tb_Treatment_lower"][1:26] * sf[0][0].values[0],
                 tx_id1["Tb_Treatment_upper"][1:26] * sf[0][0].values[0], color=sc1_colour, alpha=0.2)

ax2.plot(tb_dx0.index, tx_id2["Tb_Treatment_median"][1:26] * sf[0][0].values[0], "-", color=sc2_colour)
ax2.fill_between(tb_dx0.index, tx_id2["Tb_Treatment_lower"][1:26] * sf[0][0].values[0],
                 tx_id2["Tb_Treatment_upper"][1:26] * sf[0][0].values[0], color=sc2_colour, alpha=0.2)

# ax2.set_ylim([10000, 60000])

ax2.set(title='',
       ylabel='No. treatment appts')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# TB proportion diagnosed
ax3.plot(tb_dx0.index, tb_dx0["median"], "-", color=baseline_colour)
ax3.fill_between(tb_dx0.index, tb_dx0["lower"], tb_dx0["upper"], color=baseline_colour, alpha=0.2)

ax3.plot(tb_dx1.index, tb_dx1["median"], "-", color=sc1_colour)
ax3.fill_between(tb_dx1.index, tb_dx1["lower"], tb_dx1["upper"], color=sc1_colour, alpha=0.2)

ax3.plot(tb_dx2.index, tb_dx2["median"], "-", color=sc2_colour)
ax3.fill_between(tb_dx2.index, tb_dx2["lower"], tb_dx2["upper"], color=sc2_colour, alpha=0.2)

ax3.set_ylim([0, 1.1])
ax3.set(title='',
       ylabel='Proportion diagnosed')


# TB treatment coverage
ax4.plot(tb_tx0.index, tb_tx0["median"], "-", color=baseline_colour)
ax4.fill_between(tb_tx0.index, tb_tx0["lower"], tb_tx0["upper"], color=baseline_colour, alpha=0.2)

ax4.plot(tb_tx1.index, tb_tx1["median"], "-", color=sc1_colour)
ax4.fill_between(tb_tx1.index, tb_tx1["lower"], tb_tx1["upper"], color=sc1_colour, alpha=0.2)

ax4.plot(tb_tx2.index, tb_tx2["median"], "-", color=sc2_colour)
ax4.fill_between(tb_tx2.index, tb_tx2["lower"], tb_tx2["upper"], color=sc2_colour, alpha=0.2)

ax4.set_ylim([0, 1.1])

ax4.set(title='',
       ylabel='Proportion treated')

plt.tick_params(axis="both", which="major", labelsize=10)
fig.savefig(outputspath / "TBtreatment_cascade_4panel.png")

plt.show()



# Make 3-panel plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                             constrained_layout=True,
                                             figsize=(12, 4))
fig.suptitle('')

# TB tests
ax1.plot(tb_dx0.index, tx_id0["Tb_Test_Screening_median"][1:26] * sf[0][0].values[0], "-", color=baseline_colour)
ax1.fill_between(tb_dx0.index, tx_id0["Tb_Test_Screening_lower"][1:26] * sf[0][0].values[0],
                 tx_id0["Tb_Test_Screening_upper"][1:26] * sf[0][0].values[0], color=baseline_colour, alpha=0.2)

ax1.plot(tb_dx0.index, tx_id1["Tb_Test_Screening_median"][1:26] * sf[0][0].values[0], "-", color=sc1_colour)
ax1.fill_between(tb_dx0.index, tx_id1["Tb_Test_Screening_lower"][1:26] * sf[0][0].values[0],
                 tx_id1["Tb_Test_Screening_upper"][1:26] * sf[0][0].values[0], color=sc1_colour, alpha=0.2)

ax1.plot(tb_dx0.index, tx_id2["Tb_Test_Screening_median"][1:26] * sf[0][0].values[0], "-", color=sc2_colour)
ax1.fill_between(tb_dx0.index, tx_id2["Tb_Test_Screening_lower"][1:26] * sf[0][0].values[0],
                 tx_id2["Tb_Test_Screening_upper"][1:26] * sf[0][0].values[0], color=sc2_colour, alpha=0.2)

# ax1.set_ylim([5000000, 10000000])

ax1.set(title='',
       ylabel='No. test appts')

ax1.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])

# TB start treatment
ax2.plot(tb_dx0.index, tx_id0["Tb_Treatment_median"][1:26] * sf[0][0].values[0], "-", color=baseline_colour)
ax2.fill_between(tb_dx0.index, tx_id0["Tb_Treatment_lower"][1:26] * sf[0][0].values[0],
                 tx_id0["Tb_Treatment_upper"][1:26] * sf[0][0].values[0], color=baseline_colour, alpha=0.2)

ax2.plot(tb_dx0.index, tx_id1["Tb_Treatment_median"][1:26] * sf[0][0].values[0], "-", color=sc1_colour)
ax2.fill_between(tb_dx0.index, tx_id1["Tb_Treatment_lower"][1:26] * sf[0][0].values[0],
                 tx_id1["Tb_Treatment_upper"][1:26] * sf[0][0].values[0], color=sc1_colour, alpha=0.2)

ax2.plot(tb_dx0.index, tx_id2["Tb_Treatment_median"][1:26] * sf[0][0].values[0], "-", color=sc2_colour)
ax2.fill_between(tb_dx0.index, tx_id2["Tb_Treatment_lower"][1:26] * sf[0][0].values[0],
                 tx_id2["Tb_Treatment_upper"][1:26] * sf[0][0].values[0], color=sc2_colour, alpha=0.2)

# ax2.set_ylim([10000, 60000])

ax2.set(title='',
       ylabel='No. treatment appts')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# TB treatment coverage
ax3.plot(tb_tx0.index, tb_tx0["median"], "-", color=baseline_colour)
ax3.fill_between(tb_tx0.index, tb_tx0["lower"], tb_tx0["upper"], color=baseline_colour, alpha=0.2)

ax3.plot(tb_tx1.index, tb_tx1["median"], "-", color=sc1_colour)
ax3.fill_between(tb_tx1.index, tb_tx1["lower"], tb_tx1["upper"], color=sc1_colour, alpha=0.2)

ax3.plot(tb_tx2.index, tb_tx2["median"], "-", color=sc2_colour)
ax3.fill_between(tb_tx2.index, tb_tx2["lower"], tb_tx2["upper"], color=sc2_colour, alpha=0.2)

ax3.set_ylim([0, 1.1])

ax3.set(title='',
       ylabel='Proportion treated')

plt.tick_params(axis="both", which="major", labelsize=10)
fig.savefig(outputspath / "TBtreatment_cascade_3panel.png")

plt.show()



# %%:  ---------------------------------- HIV TREATMENT CASCADE ---------------------------------- #

# hiv proportion diagnosed
def hiv_proportion_diagnosed(results_folder):

    hiv_dx = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="dx_adult",
        index="date",
        do_scaling=False
    )

    hiv_dx.columns = hiv_dx.columns.get_level_values(0)
    dx_summary = pd.DataFrame(index=hiv_dx.index, columns=["median", "lower", "upper"])
    dx_summary["median"] = hiv_dx.median(axis=1)
    dx_summary["lower"] = hiv_dx.quantile(q=0.025, axis=1)
    dx_summary["upper"] = hiv_dx.quantile(q=0.975, axis=1)

    return dx_summary

hiv_dx0 = hiv_proportion_diagnosed(results0)
hiv_dx1 = hiv_proportion_diagnosed(results1)
hiv_dx2 = hiv_proportion_diagnosed(results2)


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


# %%:  ---------------------------------- PLOTS ---------------------------------- #

scale = sf[0][0].values[0]


# Make plot
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2,
                                             constrained_layout=True,
                                             figsize=(9, 4))
fig.suptitle('')

# HIV start treatment
ax1.plot(hiv_dx0.index, tx_id0["Hiv_Treatment_median"][1:26] * sf[0][0].values[0], "-", color=baseline_colour)
ax1.fill_between(hiv_dx0.index, tx_id0["Hiv_Treatment_lower"][1:26] * sf[0][0].values[0],
                 tx_id0["Hiv_Treatment_upper"][1:26] * sf[0][0].values[0], color=baseline_colour, alpha=0.2)

ax1.plot(hiv_dx0.index, tx_id1["Hiv_Treatment_median"][1:26] * sf[0][0].values[0], "-", color=sc1_colour)
ax1.fill_between(hiv_dx0.index, tx_id1["Hiv_Treatment_lower"][1:26] * sf[0][0].values[0],
                 tx_id1["Hiv_Treatment_upper"][1:26] * sf[0][0].values[0], color=sc1_colour, alpha=0.2)

ax1.plot(hiv_dx0.index, tx_id2["Hiv_Treatment_median"][1:26] * sf[0][0].values[0], "-", color=sc2_colour)
ax1.fill_between(hiv_dx0.index, tx_id2["Hiv_Treatment_lower"][1:26] * sf[0][0].values[0],
                 tx_id2["Hiv_Treatment_upper"][1:26] * sf[0][0].values[0], color=sc2_colour, alpha=0.2)

# ax1.set_ylim([10000, 60000])

ax1.set(title='',
       ylabel='No. treatment appts')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax1.set_ylim(0, 2000000)

# HIV treatment coverage
ax2.plot(hiv_tx0.index, hiv_tx0["median"], "-", color=baseline_colour)
ax2.fill_between(hiv_tx0.index, hiv_tx0["lower"], hiv_tx0["upper"], color=baseline_colour, alpha=0.2)

ax2.plot(hiv_tx1.index, hiv_tx1["median"], "-", color=sc1_colour)
ax2.fill_between(hiv_tx1.index, hiv_tx1["lower"], hiv_tx1["upper"], color=sc1_colour, alpha=0.2)

ax2.plot(hiv_tx2.index, hiv_tx2["median"], "-", color=sc2_colour)
ax2.fill_between(hiv_tx2.index, hiv_tx2["lower"], hiv_tx2["upper"], color=sc2_colour, alpha=0.2)

ax2.set_ylim([0, 1.1])

ax2.set(title='',
       ylabel='Proportion treated')
ax2.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])

plt.tick_params(axis="both", which="major", labelsize=10)
fig.savefig(outputspath / "Hivtreatment_cascade_2panel.png")

plt.show()



#-----------------------------------------------------------------------------
# %%:  HIV testing cascade

scale = sf[0][0].values[0]


# Make plot
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2,
                                             constrained_layout=True,
                                             figsize=(9, 4))
fig.suptitle('')

# HIV testing appts
ax1.plot(hiv_dx0.index, tx_id0["Hiv_Test_median"][1:26] * sf[0][0].values[0], "-", color=baseline_colour)
ax1.fill_between(hiv_dx0.index, tx_id0["Hiv_Test_lower"][1:26] * sf[0][0].values[0],
                 tx_id0["Hiv_Test_upper"][1:26] * sf[0][0].values[0], color=baseline_colour, alpha=0.2)

ax1.plot(hiv_dx0.index, tx_id1["Hiv_Test_median"][1:26] * sf[0][0].values[0], "-", color=sc1_colour)
ax1.fill_between(hiv_dx0.index, tx_id1["Hiv_Test_lower"][1:26] * sf[0][0].values[0],
                 tx_id1["Hiv_Test_upper"][1:26] * sf[0][0].values[0], color=sc1_colour, alpha=0.2)

ax1.plot(hiv_dx0.index, tx_id2["Hiv_Test_median"][1:26] * sf[0][0].values[0], "-", color=sc2_colour)
ax1.fill_between(hiv_dx0.index, tx_id2["Hiv_Test_lower"][1:26] * sf[0][0].values[0],
                 tx_id2["Hiv_Test_upper"][1:26] * sf[0][0].values[0], color=sc2_colour, alpha=0.2)

# ax1.set_ylim([10000, 60000])

ax1.set(title='',
       ylabel='No. testing appts')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax1.set_ylim(0, 2000000)

# TB diagnosis
ax2.plot(hiv_dx0.index, hiv_dx0["median"], "-", color=baseline_colour)
ax2.fill_between(hiv_dx0.index, hiv_dx0["lower"], hiv_dx0["upper"], color=baseline_colour, alpha=0.2)

ax2.plot(hiv_dx1.index, hiv_dx1["median"], "-", color=sc1_colour)
ax2.fill_between(hiv_dx1.index, hiv_dx1["lower"], hiv_dx1["upper"], color=sc1_colour, alpha=0.2)

ax2.plot(hiv_dx2.index, hiv_dx2["median"], "-", color=sc2_colour)
ax2.fill_between(hiv_dx2.index, hiv_dx2["lower"], hiv_dx2["upper"], color=sc2_colour, alpha=0.2)

ax2.set_ylim([0, 1.1])

ax2.set(title='',
       ylabel='Proportion diagnosed')
ax2.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])

plt.tick_params(axis="both", which="major", labelsize=10)
fig.savefig(outputspath / "Hivdiagnosis_cascade_2panel.png")

plt.show()


# %%:  ---------------------------------- Treatment delays -------------------------------------

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


# convert dict to dataframe
tb_tx_delay_adult_sc0 = pd.DataFrame(tb_tx_delay_adult_sc0_dict.items())
tb_tx_delay_adult_sc1 = pd.DataFrame(tb_tx_delay_adult_sc1_dict.items())
tb_tx_delay_adult_sc2 = pd.DataFrame(tb_tx_delay_adult_sc2_dict.items())

# need to collapse all draws/runs together
# set up empty list with columns for each year
# values will be variable length lists of delays
years = list((range(2010, 2035, 1)))

# todo this should be for years post-2022
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

# replace nan with negative number (false positive)
list_tx_delay0 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay0]
list_tx_delay1 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay1]
list_tx_delay2 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay2]

# convert lists to df
# todo note nans are fillers for dataframe
delay0 = pd.DataFrame(list_tx_delay0).T
delay0.columns = years
# convert wide to long format
delay0 = delay0.reset_index()
delay0_scatter = pd.melt(delay0, id_vars='index', value_vars=years)
delay0_scatter['value_weeks'] = round(delay0_scatter.value / 7)
delay0_scatter.loc[delay0_scatter['value_weeks'] >= 10, 'value_weeks'] = 10
delay0_scatter = delay0_scatter[delay0_scatter['value'].notna()]

delay1 = pd.DataFrame(list_tx_delay1).T
delay1.columns = years
# convert wide to long format
delay1 = delay1.reset_index()
delay1_scatter = pd.melt(delay1, id_vars='index', value_vars=years)
delay1_scatter['value_weeks'] = round(delay1_scatter.value / 7)
delay1_scatter.loc[delay1_scatter['value_weeks'] >= 10, 'value_weeks'] = 10
delay1_scatter = delay1_scatter[delay1_scatter['value'].notna()]

delay2 = pd.DataFrame(list_tx_delay2).T
delay2.columns = years
# convert wide to long format
delay2 = delay2.reset_index()
delay2_scatter = pd.melt(delay2, id_vars='index', value_vars=years)
delay2_scatter['value_weeks'] = round(delay2_scatter.value / 7)
delay2_scatter.loc[delay2_scatter['value_weeks'] >= 10, 'value_weeks'] = 10
delay2_scatter = delay2_scatter[delay2_scatter['value'].notna()]


# scenario 1 delays 2023-2035
# aggregate values over 10 weeks
delay0_hist = delay0_scatter.loc[delay0_scatter['variable'] >= 2023]
delay0_hist = delay0_hist.loc[
    (delay0_hist['value_weeks'] >= 1) & (delay0_hist['value'] <= 1095)]  # exclude negative values (false +ve)

delay1_hist = delay1_scatter.loc[delay1_scatter['variable'] >= 2023]
delay1_hist = delay1_hist.loc[
    (delay1_hist['value_weeks'] >= 1) & (delay1_hist['value'] <= 1095)]

delay2_hist = delay2_scatter.loc[delay2_scatter['variable'] >= 2023]
delay2_hist = delay2_hist.loc[
    (delay2_hist['value_weeks'] >= 1) & (delay2_hist['value'] <= 1095)]


## TREATMENT DELAY CHILDREN

tb_tx_delay_child_sc0_dict = extract_tx_delay(results_folder=results0,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayChildren")

tb_tx_delay_child_sc1_dict = extract_tx_delay(results_folder=results1,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayChildren")

tb_tx_delay_child_sc2_dict = extract_tx_delay(results_folder=results2,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayChildren")

# convert dict to dataframe
tb_tx_delay_child_sc0 = pd.DataFrame(tb_tx_delay_child_sc0_dict.items())
tb_tx_delay_child_sc1 = pd.DataFrame(tb_tx_delay_child_sc1_dict.items())
tb_tx_delay_child_sc2 = pd.DataFrame(tb_tx_delay_child_sc2_dict.items())

list_tx_delay0 = summarise_tx_delay(tb_tx_delay_child_sc0)
list_tx_delay1 = summarise_tx_delay(tb_tx_delay_child_sc1)
list_tx_delay2 = summarise_tx_delay(tb_tx_delay_child_sc2)

# replace nan with negative number (false positive)
list_tx_delay0 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay0]
list_tx_delay1 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay1]
list_tx_delay2 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay2]


# convert lists to df
# todo note nans are fillers for dataframe
delay0child = pd.DataFrame(list_tx_delay0).T
delay0child.columns = years
# convert wide to long format
delay0child = delay0child.reset_index()
delay0_scatterchild = pd.melt(delay0child, id_vars='index', value_vars=years)
delay0_scatterchild['value_weeks'] = round(delay0_scatterchild.value / 7)
delay0_scatterchild.loc[delay0_scatterchild['value_weeks'] >= 10, 'value_weeks'] = 10
delay0_scatterchild = delay0_scatterchild[delay0_scatterchild['value'].notna()]

delay1child = pd.DataFrame(list_tx_delay1).T
delay1child.columns = years
# convert wide to long format
delay1child = delay1child.reset_index()
delay1_scatterchild = pd.melt(delay1child, id_vars='index', value_vars=years)
delay1_scatterchild['value_weeks'] = round(delay1_scatterchild.value / 7)
delay1_scatterchild.loc[delay1_scatterchild['value_weeks'] >= 10, 'value_weeks'] = 10
delay1_scatterchild = delay1_scatterchild[delay1_scatterchild['value'].notna()]

delay2child = pd.DataFrame(list_tx_delay2).T
delay2child.columns = years
# convert wide to long format
delay2child = delay2child.reset_index()
delay2_scatterchild = pd.melt(delay2child, id_vars='index', value_vars=years)
delay2_scatterchild['value_weeks'] = round(delay2_scatterchild.value / 7)
delay2_scatterchild.loc[delay2_scatterchild['value_weeks'] >= 10, 'value_weeks'] = 10
delay2_scatterchild = delay2_scatterchild[delay2_scatterchild['value'].notna()]

# aggregate values over 10 weeks
delay0_histchild = delay0_scatterchild.loc[delay0_scatterchild['variable'] >= 2023]
delay0_histchild = delay0_histchild.loc[
    (delay0_histchild['value_weeks'] >= 1) & (delay0_histchild['value'] <= 1095)]  # exclude negative values (false +ve)

delay1_histchild = delay1_scatterchild.loc[delay1_scatterchild['variable'] >= 2023]
delay1_histchild = delay1_histchild.loc[
    (delay1_histchild['value_weeks'] >= 1) & (delay1_histchild['value'] <= 1095)]

delay2_histchild = delay2_scatterchild.loc[delay2_scatterchild['variable'] >= 2023]
delay2_histchild = delay2_histchild.loc[
    (delay2_histchild['value_weeks'] >= 1) & (delay2_histchild['value'] <= 1095)]


counts, bins, bars = plt.hist(delay0_hist.value_weeks, bins=range(0,11))

colours = [baseline_colour, sc1_colour, sc2_colour]
bins = range(1, 12)
labels = ["", "1", "2", "3", "4", "5", "6", "7", "8", "9", "≥ 10"]

## plot
plt.style.use('ggplot')
# fig, ax = plt.subplots()
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2,
                                             sharey=True,
                                             constrained_layout=True,
                                             figsize=(9, 4))
fig.suptitle('')

ax1.hist([list(delay0_hist.value_weeks),
                               list(delay1_hist.value_weeks),
                               list(delay2_hist.value_weeks),
          ],
         bins=bins,
         align='right',
         color=colours,
         density=True)

ax1.set_xticks(bins)
ax1.set_xticklabels(labels)
ax1.patch.set_edgecolor('grey')
ax1.patch.set_linewidth('1')

ax1.set(title='Adults',
        ylabel='Density',
        xLabel="Treatment delay, weeks")
ax1.set_ylim([0, 1.0])


ax2.hist([list(delay0_histchild.value_weeks),
                               list(delay1_histchild.value_weeks),
                               list(delay2_histchild.value_weeks),
          ],
         bins=bins,
         align='right',
         color=colours,
         density=True)

ax2.set_xticks(bins)
ax2.set_xticklabels(labels)
ax2.patch.set_edgecolor('grey')
ax2.patch.set_linewidth('1')

ax2.set(title='Children',
        ylabel='',
        xLabel="Treatment delay, weeks")
ax2.set_ylim([0, 1.0])

plt.tick_params(axis="both", which="major", labelsize=10)

plt.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])
fig.savefig(outputspath / "Tb_treatment_delay.png")

plt.show()



# %%:  ---------------------------------- TB false positives -------------------------------------

# show false positives put on treatment per 100,000 population

tmp1 = delay0_scatter.groupby('variable').count()

false0 = delay0_scatter.groupby('variable')['value'].apply(lambda x: ((x<=0) | (x>1095)).count()).reset_index(name='count')
false1 = delay1_scatter.groupby('variable')['value'].apply(lambda x: ((x<=0) | (x>1095)).count()).reset_index(name='count')
false2 = delay2_scatter.groupby('variable')['value'].apply(lambda x: ((x<=0) | (x>1095)).count()).reset_index(name='count')

# todo note these are aggregated across all runs
# plt.style.use('ggplot')
# fig, ax = plt.subplots()
#
# ax.plot(years_num[13:25], false0["count"].loc[13:24], "-", color=berry[5])
# ax.plot(years_num[13:25], false1["count"].loc[13:24], "-", color=berry[4])
# ax.plot(years_num[13:25], false2["count"].loc[13:24], "-", color=berry[3])
# ax.plot(years_num[13:25], false3["count"].loc[13:24], "-", color=berry[2])
# ax.plot(years_num[13:25], false4["count"].loc[13:24], "-", color=berry[1])
#
# ax.patch.set_edgecolor('grey')
# ax.patch.set_linewidth('1')
#
# plt.ylabel("number false positives")
# plt.xlabel("Year")
# # plt.ylim((0, 1.0))
# plt.title("")
# plt.legend(labels=["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])
# plt.show()


def tb_false_pos_adults(results_folder):
    false_pos = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_false_positive",
        column="tbPropFalsePositiveAdults",
        index="date",
        do_scaling=False
    )

    false_pos.columns = false_pos.columns.get_level_values(0)
    false_pos_summary = pd.DataFrame(index=false_pos.index, columns=["median", "lower", "upper"])
    false_pos_summary["median"] = false_pos.median(axis=1)
    false_pos_summary["lower"] = false_pos.quantile(q=0.025, axis=1)
    false_pos_summary["upper"] = false_pos.quantile(q=0.975, axis=1)

    return false_pos_summary

def tb_false_pos_children(results_folder):
    false_pos = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_false_positive",
        column="tbPropFalsePositiveChildren",
        index="date",
        do_scaling=False
    )

    false_pos.columns = false_pos.columns.get_level_values(0)
    false_pos_summary = pd.DataFrame(index=false_pos.index, columns=["median", "lower", "upper"])
    false_pos_summary["median"] = false_pos.median(axis=1)
    false_pos_summary["lower"] = false_pos.quantile(q=0.025, axis=1)
    false_pos_summary["upper"] = false_pos.quantile(q=0.975, axis=1)

    return false_pos_summary

false_pos0 = tb_false_pos_adults(results0)
false_pos1 = tb_false_pos_adults(results1)
false_pos2 = tb_false_pos_adults(results2)


false_pos_child0 = tb_false_pos_children(results0)
false_pos_child1 = tb_false_pos_children(results1)
false_pos_child2 = tb_false_pos_children(results2)


plt.style.use('ggplot')
fig, ax = plt.subplots()

ax.plot(years_num[13:26], false_pos0["median"][12:25], "-", color=baseline_colour)
ax.fill_between(years_num[13:26], false_pos0["lower"][12:25], false_pos0["upper"][12:25],
                color=baseline_colour, alpha=0.2)

ax.plot(years_num[13:26], false_pos1["median"][12:25], "-", color=sc1_colour)
ax.fill_between(years_num[13:26], false_pos1["lower"][12:25], false_pos1["upper"][12:25],
                color=sc1_colour, alpha=0.2)

ax.plot(years_num[13:26], false_pos2["median"][12:25], "-", color=sc2_colour)
ax.fill_between(years_num[13:26], false_pos2["lower"][12:25], false_pos2["upper"][12:25],
                color=sc2_colour, alpha=0.2)

ax.patch.set_edgecolor('grey')
ax.patch.set_linewidth('1')

plt.ylabel("Proportion false positives")

plt.xlabel("Year")
plt.ylim((0, 0.5))
plt.title("")
plt.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])
fig.savefig(outputspath / "Tb_false_positives.png")
plt.show()



## plot false positives for all years
plt.style.use('ggplot')
fig, ax = plt.subplots()

ax.plot(years_num[1:26], false_pos0["median"], "-", color=baseline_colour)
ax.fill_between(years_num[1:26], false_pos0["lower"], false_pos0["upper"],
                color=baseline_colour, alpha=0.2)

ax.plot(years_num[1:26], false_pos1["median"], "-", color=sc1_colour)
ax.fill_between(years_num[1:26], false_pos1["lower"], false_pos1["upper"],
                color=sc1_colour, alpha=0.2)

ax.plot(years_num[1:26], false_pos2["median"], "-", color=sc2_colour)
ax.fill_between(years_num[1:26], false_pos2["lower"], false_pos2["upper"],
                color=sc2_colour, alpha=0.2)

ax.patch.set_edgecolor('grey')
ax.patch.set_linewidth('1')

plt.ylabel("Proportion false positives")

plt.xlabel("Year")
plt.ylim((0, 0.5))
plt.title("")
plt.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])
fig.savefig(outputspath / "Tb_false_positives_all_years.png")
plt.show()


## plot false positives for all years - CHILDREN
plt.style.use('ggplot')
fig, ax = plt.subplots()

ax.plot(years_num[1:26], false_pos_child0["median"], "-", color=baseline_colour)
ax.fill_between(years_num[1:26], false_pos_child0["lower"], false_pos_child0["upper"],
                color=baseline_colour, alpha=0.2)

ax.plot(years_num[1:26], false_pos_child1["median"], "-", color=sc1_colour)
ax.fill_between(years_num[1:26], false_pos_child1["lower"], false_pos_child1["upper"],
                color=sc1_colour, alpha=0.2)

ax.plot(years_num[1:26], false_pos_child2["median"], "-", color=sc2_colour)
ax.fill_between(years_num[1:26], false_pos_child2["lower"], false_pos_child2["upper"],
                color=sc2_colour, alpha=0.2)

ax.patch.set_edgecolor('grey')
ax.patch.set_linewidth('1')

plt.ylabel("Proportion false positives")

plt.xlabel("Year")
# plt.ylim((0, 0.5))
plt.title("")
plt.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])
fig.savefig(outputspath / "Tb_false_positives_all_years_children.png")
plt.show()


# %%:  ---------------------------------- PrEP IMPACT ---------------------------------- #

# HIV incidence in AGYW

def hiv_agyw_inc(results_folder):
    inc = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="infections_by_2age_groups_and_sex",
        column="n_new_infections_female_1524",
        index="date",
        do_scaling=True
    )

    inc.columns = inc.columns.get_level_values(0)
    inc_summary = pd.DataFrame(index=inc.index, columns=["median", "lower", "upper"])
    inc_summary["median"] = inc.median(axis=1)
    inc_summary["lower"] = inc.quantile(q=0.025, axis=1)
    inc_summary["upper"] = inc.quantile(q=0.975, axis=1)

    return inc_summary


agyw_inc0 = hiv_agyw_inc(results0)
agyw_inc1 = hiv_agyw_inc(results1)
agyw_inc2 = hiv_agyw_inc(results2)


baseline_num_infections = agyw_inc0["median"][11:26]
# multiply by scaling factor to get numbers of expected infections


## plot HIV incidence in AGYW
plt.style.use('ggplot')
fig, ax = plt.subplots()

ax.plot(years_num[12:26], agyw_inc0["median"][11:26], "-", color=baseline_colour)
ax.fill_between(years_num[12:26], agyw_inc0["lower"][11:26], agyw_inc0["upper"][11:26],
                color=baseline_colour, alpha=0.2)

ax.plot(years_num[12:26], agyw_inc1["median"][11:26], "-", color=sc1_colour)
ax.fill_between(years_num[12:26], agyw_inc1["lower"][11:26], agyw_inc1["upper"][11:26],
                color=sc1_colour, alpha=0.2)

ax.plot(years_num[12:26], agyw_inc2["median"][11:26], "-", color=sc2_colour)
ax.fill_between(years_num[12:26], agyw_inc2["lower"][11:26], agyw_inc2["upper"][11:26],
                color=sc2_colour, alpha=0.2)

ax.patch.set_edgecolor('grey')
ax.patch.set_linewidth('1')

plt.ylabel("Number new HIV infections")

plt.xlabel("Year")
plt.ylim((0, 7500))
plt.title("")
plt.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])
fig.savefig(outputspath / "Incidence_HIV_AGYW.png")
plt.show()



# %%: get consumables availability

# get consumables spreadsheet
cons_availability = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_availability_small.csv")
items_list = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_Items_and_Packages.csv")

# import master facilities list to get facility levels mapped to facility ID
master_fac = pd.read_csv(
    resourcefilepath / "healthsystem/organisation/ResourceFile_Master_Facilities_List.csv")

# map facility levels to facility ID in consumables spreadsheet
cons_full = pd.merge(cons_availability, master_fac,
                     left_on=["Facility_ID"], right_on=["Facility_ID"], how='left')

# groupby item code & facility level -> average availability by facility level for all items
average_cons_availability = cons_full.groupby(["item_code", "Facility_Level"])["available_prop"].mean().reset_index()


def get_item_codes_from_package_name(lookup_df: pd.DataFrame, package: str) -> dict:

    return int(pd.unique(lookup_df.loc[lookup_df["Intervention_Pkg"] == package, "Item_Code"]))


def get_item_code_from_item_name(lookup_df: pd.DataFrame, item: str) -> int:
    """Helper function to provide the item_code (an int) when provided with the name of the item"""
    return int(pd.unique(lookup_df.loc[lookup_df["Items"] == item, "Item_Code"])[0])

## TB consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["sputum_test"] = get_item_codes_from_package_name(items_list, "Microscopy Test")
item_codes_dict["xpert_test"] = get_item_codes_from_package_name(items_list, "Xpert test")
item_codes_dict["chest_xray"] = get_item_code_from_item_name(items_list, "X-ray")

# treatment
item_codes_dict["tb_tx_adult"] = get_item_code_from_item_name(items_list, "Cat. I & III Patient Kit A")
item_codes_dict["tb_tx_child"] = get_item_code_from_item_name(items_list, "Cat. I & III Patient Kit B")
item_codes_dict["tb_retx_adult"] = get_item_code_from_item_name(items_list, "Cat. II Patient Kit A1")
item_codes_dict["tb_retx_child"] = get_item_code_from_item_name(items_list, "Cat. II Patient Kit A2")
item_codes_dict["tb_mdrtx"] = get_item_code_from_item_name(items_list, "Category IV")
item_codes_dict["tb_ipt"] = get_item_code_from_item_name(items_list, "Isoniazid/Pyridoxine, tablet 300 mg")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 184, "item_code"] = "Sputum test"
selected_cons_availability.loc[selected_cons_availability.item_code == 187, "item_code"] = "GeneXpert test"
selected_cons_availability.loc[selected_cons_availability.item_code == 175, "item_code"] = "Chest X-ray"
selected_cons_availability.loc[selected_cons_availability.item_code == 176, "item_code"] = "Adult treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 178, "item_code"] = "Child treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 177, "item_code"] = "Adult retreatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 179, "item_code"] = "Child retreatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 180, "item_code"] = "MDR treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 192, "item_code"] = "IPT"

df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc=np.mean)

ax = sns.heatmap(df_heatmap, annot=True)
plt.tight_layout()

plt.xlabel('Facility level')
plt.ylabel('')
# plt.savefig(outputspath / "cons_availability.png", bbox_inches='tight')
plt.show()


## HIV consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["HIV test"] = get_item_code_from_item_name(items_list, "Test, HIV EIA Elisa")
item_codes_dict["Viral load"] = get_item_codes_from_package_name(items_list, "Viral Load")
item_codes_dict["VMMC"] = get_item_code_from_item_name(items_list, "male circumcision kit, consumables (10 procedures)_1_IDA")

# treatment
item_codes_dict["Adult PrEP"] = get_item_code_from_item_name(items_list, "Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg")
item_codes_dict["Infant PrEP"] = get_item_code_from_item_name(items_list, "Nevirapine, oral solution, 10 mg/ml")
item_codes_dict['First-line ART regimen: adult'] = get_item_code_from_item_name(items_list, "First-line ART regimen: adult")
item_codes_dict['First-line ART regimen: adult: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Cotrimoxizole, 960mg pppy")

# ART for older children aged ("ART_age_cutoff_younger_child" < age <= "ART_age_cutoff_older_child"):
item_codes_dict['First line ART regimen: older child'] = get_item_code_from_item_name(items_list, "First line ART regimen: older child")
item_codes_dict['First line ART regimen: older child: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Sulfamethoxazole + trimethropin, tablet 400 mg + 80 mg")

# ART for younger children aged (age < "ART_age_cutoff_younger_child"):
item_codes_dict['First line ART regimen: young child'] = get_item_code_from_item_name(items_list, "First line ART regimen: young child")
item_codes_dict['First line ART regimen: young child: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Sulfamethoxazole + trimethropin, oral suspension, 240 mg, 100 ml")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 196, "item_code"] = "HIV test"
selected_cons_availability.loc[selected_cons_availability.item_code == 190, "item_code"] = "Viral load"
selected_cons_availability.loc[selected_cons_availability.item_code == 197, "item_code"] = "VMMC"
selected_cons_availability.loc[selected_cons_availability.item_code == 1191, "item_code"] = "Adult PrEP"
selected_cons_availability.loc[selected_cons_availability.item_code == 198, "item_code"] = "Infant PrEP"
selected_cons_availability.loc[selected_cons_availability.item_code == 2671, "item_code"] = "Adult ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 204, "item_code"] = "Adult cotrimoxazole"
selected_cons_availability.loc[selected_cons_availability.item_code == 2672, "item_code"] = "Child ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 162, "item_code"] = "Child cotrimoxazole"
selected_cons_availability.loc[selected_cons_availability.item_code == 2673, "item_code"] = "Infant ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 202, "item_code"] = "Infant cotrimoxazole"

df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc=np.mean)

ax = sns.heatmap(df_heatmap, annot=True)
plt.tight_layout()

plt.xlabel('Facility level')
plt.ylabel('')
# plt.savefig(outputspath / "cons_availability_HIV.png", bbox_inches='tight')
plt.show()



# %%: get consumables availability - one month plots

# get consumables spreadsheet
cons_availability = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_availability_small.csv")
items_list = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_Items_and_Packages.csv")

# import master facilities list to get facility levels mapped to facility ID
master_fac = pd.read_csv(
    resourcefilepath / "healthsystem/organisation/ResourceFile_Master_Facilities_List.csv")

# map facility levels to facility ID in consumables spreadsheet
cons_full = pd.merge(cons_availability, master_fac,
                     left_on=["Facility_ID"], right_on=["Facility_ID"], how='left')


# select month for representation - Dec (use jan for hiv?)
cons_dec = cons_full.loc[cons_full.month == 1]

# groupby item code & facility level -> average availability by facility level for all items
average_cons_availability = cons_dec.groupby(["item_code", "Facility_Level"])["available_prop"].mean().reset_index()


def get_item_codes_from_package_name(lookup_df: pd.DataFrame, package: str) -> dict:

    return int(pd.unique(lookup_df.loc[lookup_df["Intervention_Pkg"] == package, "Item_Code"]))


def get_item_code_from_item_name(lookup_df: pd.DataFrame, item: str) -> int:
    """Helper function to provide the item_code (an int) when provided with the name of the item"""
    return int(pd.unique(lookup_df.loc[lookup_df["Items"] == item, "Item_Code"])[0])

## TB consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["sputum_test"] = get_item_codes_from_package_name(items_list, "Microscopy Test")
item_codes_dict["xpert_test"] = get_item_codes_from_package_name(items_list, "Xpert test")
item_codes_dict["chest_xray"] = get_item_code_from_item_name(items_list, "X-ray")

# treatment
item_codes_dict["tb_tx_adult"] = get_item_code_from_item_name(items_list, "Cat. I & III Patient Kit A")
item_codes_dict["tb_tx_child"] = get_item_code_from_item_name(items_list, "Cat. I & III Patient Kit B")
item_codes_dict["tb_retx_adult"] = get_item_code_from_item_name(items_list, "Cat. II Patient Kit A1")
item_codes_dict["tb_retx_child"] = get_item_code_from_item_name(items_list, "Cat. II Patient Kit A2")
item_codes_dict["tb_mdrtx"] = get_item_code_from_item_name(items_list, "Category IV")
item_codes_dict["tb_ipt"] = get_item_code_from_item_name(items_list, "Isoniazid/Pyridoxine, tablet 300 mg")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 184, "item_code"] = "Sputum test"
selected_cons_availability.loc[selected_cons_availability.item_code == 187, "item_code"] = "GeneXpert test"
selected_cons_availability.loc[selected_cons_availability.item_code == 175, "item_code"] = "Chest X-ray"
selected_cons_availability.loc[selected_cons_availability.item_code == 176, "item_code"] = "Adult treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 178, "item_code"] = "Child treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 177, "item_code"] = "Adult retreatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 179, "item_code"] = "Child retreatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 180, "item_code"] = "MDR treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 192, "item_code"] = "IPT"

df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc=np.mean)

ax = sns.heatmap(df_heatmap, annot=True)
plt.tight_layout()

plt.xlabel('Facility level')
plt.ylabel('')
plt.savefig(outputspath / "tb_cons_availability_dec.png", bbox_inches='tight')
plt.show()


## HIV consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["HIV test"] = get_item_code_from_item_name(items_list, "Test, HIV EIA Elisa")
item_codes_dict["Viral load"] = get_item_codes_from_package_name(items_list, "Viral Load")
item_codes_dict["VMMC"] = get_item_code_from_item_name(items_list, "male circumcision kit, consumables (10 procedures)_1_IDA")

# treatment
item_codes_dict["Adult PrEP"] = get_item_code_from_item_name(items_list, "Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg")
item_codes_dict["Infant PrEP"] = get_item_code_from_item_name(items_list, "Nevirapine, oral solution, 10 mg/ml")
item_codes_dict['First-line ART regimen: adult'] = get_item_code_from_item_name(items_list, "First-line ART regimen: adult")
item_codes_dict['First-line ART regimen: adult: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Cotrimoxizole, 960mg pppy")

# ART for older children aged ("ART_age_cutoff_younger_child" < age <= "ART_age_cutoff_older_child"):
item_codes_dict['First line ART regimen: older child'] = get_item_code_from_item_name(items_list, "First line ART regimen: older child")
item_codes_dict['First line ART regimen: older child: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Sulfamethoxazole + trimethropin, tablet 400 mg + 80 mg")

# ART for younger children aged (age < "ART_age_cutoff_younger_child"):
item_codes_dict['First line ART regimen: young child'] = get_item_code_from_item_name(items_list, "First line ART regimen: young child")
item_codes_dict['First line ART regimen: young child: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Sulfamethoxazole + trimethropin, oral suspension, 240 mg, 100 ml")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 196, "item_code"] = "HIV test"
selected_cons_availability.loc[selected_cons_availability.item_code == 190, "item_code"] = "Viral load"
selected_cons_availability.loc[selected_cons_availability.item_code == 197, "item_code"] = "VMMC"
selected_cons_availability.loc[selected_cons_availability.item_code == 1191, "item_code"] = "Adult PrEP"
selected_cons_availability.loc[selected_cons_availability.item_code == 198, "item_code"] = "Infant PrEP"
selected_cons_availability.loc[selected_cons_availability.item_code == 2671, "item_code"] = "Adult ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 204, "item_code"] = "Adult cotrimoxazole"
selected_cons_availability.loc[selected_cons_availability.item_code == 2672, "item_code"] = "Child ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 162, "item_code"] = "Child cotrimoxazole"
selected_cons_availability.loc[selected_cons_availability.item_code == 2673, "item_code"] = "Infant ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 202, "item_code"] = "Infant cotrimoxazole"

df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc=np.mean)

ax = sns.heatmap(df_heatmap, annot=True)
plt.tight_layout()

plt.xlabel('')
plt.ylabel('')
plt.savefig(outputspath / "hiv_cons_availability_dec.png", bbox_inches='tight')
plt.show()
