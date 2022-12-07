"""This file plots the outputs from the baseline model run
compared with data
"""

import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

# outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

outputspath = Path("./outputs/tbh03@ic.ac.uk")

# %% read in data files for plots
# load all the data for calibration

# TB WHO data
xls_tb = pd.ExcelFile(resourcefilepath / "ResourceFile_TB.xlsx")

data_tb_who = pd.read_excel(xls_tb, sheet_name="WHO_activeTB2020")
data_tb_who = data_tb_who.loc[
    (data_tb_who.year >= 2010)
]  # include only years post-2010
data_tb_who.index = data_tb_who["year"]
data_tb_who = data_tb_who.drop(columns=["year"])

# TB latent data (Houben & Dodd 2016)
data_tb_latent = pd.read_excel(xls_tb, sheet_name="latent_TB2014_summary")
data_tb_latent_all_ages = data_tb_latent.loc[data_tb_latent.Age_group == "0_80"]
data_tb_latent_estimate = data_tb_latent_all_ages.proportion_latent_TB.values[0]
data_tb_latent_lower = abs(
    data_tb_latent_all_ages.proportion_latent_TB_lower.values[0]
    - data_tb_latent_estimate
)
data_tb_latent_upper = abs(
    data_tb_latent_all_ages.proportion_latent_TB_upper.values[0]
    - data_tb_latent_estimate
)
data_tb_latent_yerr = [data_tb_latent_lower, data_tb_latent_upper]

# TB treatment coverage
data_tb_ntp = pd.read_excel(xls_tb, sheet_name="NTP2019")
data_tb_ntp.index = data_tb_ntp["year"]
data_tb_ntp = data_tb_ntp.drop(columns=["year"])

# HIV resourcefile
xls = pd.ExcelFile(resourcefilepath / "ResourceFile_HIV.xlsx")

# HIV UNAIDS data
data_hiv_unaids = pd.read_excel(xls, sheet_name="unaids_infections_art2021")
data_hiv_unaids.index = data_hiv_unaids["year"]
data_hiv_unaids = data_hiv_unaids.drop(columns=["year"])

# HIV UNAIDS data
data_hiv_unaids_deaths = pd.read_excel(xls, sheet_name="unaids_mortality_dalys2021")
data_hiv_unaids_deaths.index = data_hiv_unaids_deaths["year"]
data_hiv_unaids_deaths = data_hiv_unaids_deaths.drop(columns=["year"])

# AIDSinfo (UNAIDS)
data_hiv_aidsinfo = pd.read_excel(xls, sheet_name="children0_14_prev_AIDSinfo")
data_hiv_aidsinfo.index = data_hiv_aidsinfo["year"]
data_hiv_aidsinfo = data_hiv_aidsinfo.drop(columns=["year"])

# unaids program performance
data_hiv_program = pd.read_excel(xls, sheet_name="unaids_program_perf")
data_hiv_program.index = data_hiv_program["year"]
data_hiv_program = data_hiv_program.drop(columns=["year"])

# MPHIA HIV data - age-structured
data_hiv_mphia_inc = pd.read_excel(xls, sheet_name="MPHIA_incidence2015")
# 2015
data_hiv_mphia_inc_estimate = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49") & (data_hiv_mphia_inc.year == 2015), "total_percent_annual_incidence"
].values
data_hiv_mphia_inc_lower = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49") & (data_hiv_mphia_inc.year == 2015), "total_percent_annual_incidence_lower"
].values
data_hiv_mphia_inc_upper = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49") & (data_hiv_mphia_inc.year == 2015), "total_percent_annual_incidence_upper"
].values
data_hiv_mphia_inc_yerr = [
    abs(data_hiv_mphia_inc_lower - data_hiv_mphia_inc_estimate),
    abs(data_hiv_mphia_inc_upper - data_hiv_mphia_inc_estimate),
]
# 2020
data_hiv_mphia_inc_estimate20 = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49") & (data_hiv_mphia_inc.year == 2020), "total_percent_annual_incidence"
].values
data_hiv_mphia_inc_lower20 = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49") & (data_hiv_mphia_inc.year == 2020), "total_percent_annual_incidence_lower"
].values
data_hiv_mphia_inc_upper20 = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49") & (data_hiv_mphia_inc.year == 2020), "total_percent_annual_incidence_upper"
].values
data_hiv_mphia_inc_yerr20 = [
    abs(data_hiv_mphia_inc_lower20 - data_hiv_mphia_inc_estimate20),
    abs(data_hiv_mphia_inc_upper20 - data_hiv_mphia_inc_estimate20),
]


data_hiv_mphia_prev = pd.read_excel(xls, sheet_name="MPHIA_prevalence_art2015")

# DHS HIV data
data_hiv_dhs_prev = pd.read_excel(xls, sheet_name="DHS_prevalence")

# MoH HIV testing data
data_hiv_moh_tests = pd.read_excel(xls, sheet_name="MoH_numbers_tests")
data_hiv_moh_tests.index = data_hiv_moh_tests["year"]
data_hiv_moh_tests = data_hiv_moh_tests.drop(columns=["year"])

# MoH HIV ART data
# todo this is quarterly
data_hiv_moh_art = pd.read_excel(xls, sheet_name="MoH_number_art")

# %% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
# results_folder = get_scenario_outputs("scenario0.py", outputspath)[-1]

results_folder = get_scenario_outputs("long_run_all_diseases.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# choose which draw to summarise / visualise
draw = 0

# %% extract results
# Load and format model results (with year as integer):

# summarise outputs across all draws
def summarize_across_draws(results: pd.DataFrame, only_mean: bool = False) -> pd.DataFrame:
    """Utility function to compute summary statistics

    Finds mean value and 95% interval across the runs for all runs/draws
    """

    summary = pd.DataFrame(index=results.index, columns=["median", "lower", "upper"])
    summary["median"] = results.median(axis=1)
    summary["lower"] = results.quantile(q=0.025, axis=1)
    summary["upper"] = results.quantile(q=0.975, axis=1)

    return summary

# ---------------------------------- HIV ---------------------------------- #

model_hiv_adult_prev = summarize_across_draws(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_adult_15plus",
        index="date",
        do_scaling=False,
    ),
)
model_hiv_adult_prev.index = model_hiv_adult_prev.index.year

model_hiv_adult_inc = summarize_across_draws(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False,
    ),
)
model_hiv_adult_inc.index = model_hiv_adult_inc.index.year

model_hiv_child_prev = summarize_across_draws(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_child",
        index="date",
        do_scaling=False,
    ),
)
model_hiv_child_prev.index = model_hiv_child_prev.index.year

model_hiv_child_inc = summarize_across_draws(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_child_inc",
        index="date",
        do_scaling=False,
    ),
)
model_hiv_child_inc.index = model_hiv_child_inc.index.year

model_hiv_fsw_prev = summarize_across_draws(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_fsw",
        index="date",
        do_scaling=False,
    ),
)
model_hiv_fsw_prev.index = model_hiv_fsw_prev.index.year


# ---------------------------------- PERSON-YEARS ---------------------------------- #

# function to extract person-years by year
# call this for each run and then take the mean to use as denominator for mortality / incidence etc.
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


py = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=False
)

py_mean = py.mean(axis=1)

# ---------------------------------- TB ---------------------------------- #

model_tb_inc = summarize_across_draws(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
    ),
)
model_tb_inc.index = model_tb_inc.index.year
activeTB_inc_rate = model_tb_inc["median"] / py_mean.values[1:26] * 100000

activeTB_inc_rate.index = model_tb_inc.index
activeTB_inc_rate_low = pd.Series(
    (model_tb_inc["lower"].values / py_mean.values[1:26]) * 100000
)
activeTB_inc_rate_low.index = model_tb_inc.index
activeTB_inc_rate_high = pd.Series(
    (model_tb_inc["upper"].values / py_mean.values[1:26]) * 100000
)
activeTB_inc_rate_high.index = model_tb_inc.index

model_tb_mdr = summarize_across_draws(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_mdr",
        column="tbPropActiveCasesMdr",
        index="date",
        do_scaling=False,
    ),
)

model_tb_mdr.index = model_tb_mdr.index.year

model_tb_hiv_prop = summarize_across_draws(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="prop_active_tb_in_plhiv",
        index="date",
        do_scaling=False,
    ),
)

model_tb_hiv_prop.index = model_tb_hiv_prop.index.year

# ---------------------------------- DEATHS ---------------------------------- #

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

results_deaths = results_deaths.reset_index()

# todo summarise across runs
aids_non_tb_deaths_table = results_deaths.loc[results_deaths.cause == "AIDS_non_TB"]
aids_tb_deaths_table = results_deaths.loc[results_deaths.cause == "AIDS_TB"]
tb_deaths_table = results_deaths.loc[results_deaths.cause == "TB"]


# ------------ summarise deaths producing df for each draw


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


aids_deaths = summarise_aids_deaths(results_folder, py)


# TB deaths excluding HIV
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


tb_deaths = summarise_tb_deaths(results_folder, py)


# ---------------------------------- PROGRAM COVERAGE ---------------------------------- #

# TB treatment coverage
model_tb_tx = summarize_across_draws(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbTreatmentCoverage",
        index="date",
        do_scaling=False,
    ),
)

model_tb_tx.index = model_tb_tx.index.year

# HIV treatment coverage
model_hiv_tx = summarize_across_draws(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="art_coverage_adult",
        index="date",
        do_scaling=False,
    ),
)

model_hiv_tx.index = model_hiv_tx.index.year


# %% Function to make standard plot to compare model and data
def make_plot(
    model=None,
    model_low=None,
    model_high=None,
    data_name=None,
    data_mid=None,
    data_low=None,
    data_high=None,
    xlab=None,
    ylab=None,
    title_str=None,
    xlim=None,
    ylim_lower=None,
    ylim_upper=None
):
    assert model is not None
    assert title_str is not None

    # Make plot
    fig, ax = plt.subplots()
    ax.plot(model.index, model.values, "-", color="C3")
    if (model_low is not None) and (model_high is not None):
        ax.fill_between(model_low.index, model_low, model_high, color="C3", alpha=0.2)

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, "-", color="C0")
    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index, data_low, data_high, color="C0", alpha=0.2)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_ylabel(ylab)

    if xlim is not None:
        ax.set_xlim(2010, xlim)

    if ylim_lower is not None:
        ax.set_ylim(ylim_lower, ylim_upper)

    plt.title(title_str)
    plt.legend(["TLO", data_name])
    # plt.gca().set_ylim(bottom=0)
    # plt.savefig(outputspath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format='pdf')


# %% make plots

# HIV - prevalence among in adults aged 15-49

make_plot(
    title_str="HIV Prevalence in Adults Aged 15-49 (%)",
    model=model_hiv_adult_prev["median"] * 100,
    model_low=model_hiv_adult_prev["lower"] * 100,
    model_high=model_hiv_adult_prev["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_unaids["prevalence_age15_49"] * 100,
    data_low=data_hiv_unaids["prevalence_age15_49_lower"] * 100,
    data_high=data_hiv_unaids["prevalence_age15_49_upper"] * 100,
    xlab="Year",
    ylab="HIV prevalence (%)"
)

# data: MPHIA, 2015 and 2020
plt.plot(
    model_hiv_adult_prev.index[6],
    data_hiv_mphia_prev.loc[
        (data_hiv_mphia_prev.age == "Total 15-49") & (data_hiv_mphia_prev.year == 2015),
        "total percent hiv positive"].values,
    "gx",
)
plt.plot(
    model_hiv_adult_prev.index[11],
    data_hiv_mphia_prev.loc[
        (data_hiv_mphia_prev.age == "15-49") & (data_hiv_mphia_prev.year == 2020),
        "total percent hiv positive"].values,
    "gx",
)

# data: DHS
x_values = [model_hiv_adult_prev.index[0], model_hiv_adult_prev.index[5]]
y_values = data_hiv_dhs_prev.loc[
    (data_hiv_dhs_prev.Year >= 2010), "HIV prevalence among general population 15-49"
]
y_lower = abs(
    y_values
    - (
        data_hiv_dhs_prev.loc[
            (data_hiv_dhs_prev.Year >= 2010),
            "HIV prevalence among general population 15-49 lower",
        ]
    )
)
y_upper = abs(
    y_values
    - (
        data_hiv_dhs_prev.loc[
            (data_hiv_dhs_prev.Year >= 2010),
            "HIV prevalence among general population 15-49 upper",
        ]
    )
)
plt.errorbar(x_values, y_values, yerr=[y_lower, y_upper], fmt="ko")

plt.ylim((0, 15))
plt.xlim(2010, 2023)

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
green_cross = mlines.Line2D(
    [], [], linewidth=0, color="g", marker="x", markersize=7, label="MPHIA"
)
orange_ci = mlines.Line2D([], [], color="black", marker=".", markersize=15, label="DHS")
plt.legend(handles=[red_line, blue_line, green_cross, orange_ci])

plt.savefig(make_graph_file_name("HIV_Prevalence_in_Adults"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV Incidence in adults aged 15-49 per 100 population
make_plot(
    title_str="HIV Incidence in Adults Aged 15-49 per 100 population",
    model=model_hiv_adult_inc["median"] * 100,
    model_low=model_hiv_adult_inc["lower"] * 100,
    model_high=model_hiv_adult_inc["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_unaids["incidence_per1000_age15_49"] / 10,
    data_low=data_hiv_unaids["incidence_per1000_age15_49_lower"] / 10,
    data_high=data_hiv_unaids["incidence_per1000_age15_49_upper"] / 10,
    xlab="Year",
    ylab="HIV incidence per 100 population"
)

# MPHIA
plt.errorbar(
    model_hiv_adult_inc.index[6],
    data_hiv_mphia_inc_estimate[0],
    yerr=[[data_hiv_mphia_inc_yerr[0][0]], [data_hiv_mphia_inc_yerr[1][0]]],
    fmt="gx",
)
plt.errorbar(
    model_hiv_adult_inc.index[11],
    data_hiv_mphia_inc_estimate20[0],
    yerr=[[data_hiv_mphia_inc_yerr20[0][0]], [data_hiv_mphia_inc_yerr20[1][0]]],
    fmt="gx",
)
plt.ylim(0, 1.0)
plt.xlim(2010, 2023)
#
# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
orange_ci = mlines.Line2D(
    [], [], color="green", marker="x", markersize=8, label="MPHIA"
)
plt.legend(handles=[red_line, blue_line, orange_ci])

plt.savefig(make_graph_file_name("HIV_Incidence_in_Adults"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV Prevalence Children
make_plot(
    title_str="HIV Prevalence in Children 0-14 (%)",
    model=model_hiv_child_prev["median"] * 100,
    model_low=model_hiv_child_prev["lower"] * 100,
    model_high=model_hiv_child_prev["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_aidsinfo["prevalence_0_14"] * 100,
    data_low=data_hiv_aidsinfo["prevalence_0_14_lower"] * 100,
    data_high=data_hiv_aidsinfo["prevalence_0_14_upper"] * 100,
    xlab="Year",
    ylab="HIV prevalence in children Aged 0-14 (%)",
    xlim=2023,
    ylim_lower=0,
    ylim_upper=5
)

# MPHIA
plt.plot(
    model_hiv_child_prev.index[6],
    data_hiv_mphia_prev.loc[
        data_hiv_mphia_prev.age == "Total 0-14", "total percent hiv positive"
    ].values[0],
    "gx",
)

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
green_cross = mlines.Line2D(
    [], [], linewidth=0, color="g", marker="x", markersize=7, label="MPHIA"
)
plt.legend(handles=[red_line, blue_line, green_cross])
plt.savefig(make_graph_file_name("HIV_Prevalence_in_Children"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV Incidence Children
make_plot(
    title_str="HIV Incidence in Children Aged 0-14 per 100 population)",
    model=model_hiv_child_inc["median"] * 100,
    model_low=model_hiv_child_inc["lower"] * 100,
    model_high=model_hiv_child_inc["upper"] * 100,
    data_mid=data_hiv_aidsinfo["incidence0_14_per100py"],
    data_low=data_hiv_aidsinfo["incidence0_14_per100py_lower"],
    data_high=data_hiv_aidsinfo["incidence0_14_per100py_upper"],
    xlab="Year",
    ylab="HIV incidence per 100 population",
    xlim=2023,
    ylim_lower=0,
    ylim_upper=0.35
)

red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")

plt.legend(handles=[red_line, blue_line])

plt.savefig(make_graph_file_name("HIV_Incidence_in_Children"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV prevalence among female sex workers:
make_plot(
    title_str="HIV Prevalence among Female Sex Workers (%)",
    model=model_hiv_fsw_prev["median"] * 100,
    model_low=model_hiv_fsw_prev["lower"] * 100,
    model_high=model_hiv_fsw_prev["upper"] * 100,
    xlab="Year",
    ylab="HIV prevalence among FSW",
    xlim=2023,
    ylim_lower=0,
    ylim_upper=100
)

red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")

plt.legend(handles=[red_line])

plt.savefig(make_graph_file_name("HIV_Prevalence_FSW"))

plt.show()

# ----------------------------- TB -------------------------------------- #

# Active TB incidence per 100,000 person-years - annual outputs

make_plot(
    title_str="Active TB Incidence (per 100,000 person-years)",
    model=activeTB_inc_rate,
    model_low=activeTB_inc_rate_low,
    model_high=activeTB_inc_rate_high,
    data_name="WHO",
    data_mid=data_tb_who["incidence_per_100k"],
    data_low=data_tb_who["incidence_per_100k_low"],
    data_high=data_tb_who["incidence_per_100k_high"],
    xlab="Year",
    ylab="Active TB incidence per 100,000 py",
    xlim=2023,
    ylim_lower=0,
    ylim_upper=750
)

plt.savefig(make_graph_file_name("TB_Incidence"))

plt.show()


# ---------------------------------------------------------------------- #

# proportion TB cases that are MDR
make_plot(
    title_str="Proportion of active TB cases that are MDR",
    model=model_tb_mdr["median"],
    model_low=model_tb_mdr["lower"],
    model_high=model_tb_mdr["upper"],
    xlim=2023,
    ylim_lower=0,
    ylim_upper=0.08,
    xlab="Year",
    ylab="Proportion of MDR cases"
)
# data from ResourceFile_TB sheet WHO_mdrTB2017
plt.errorbar(model_tb_mdr.index[7], 0.0075, yerr=[[0.0059], [0.0105]], fmt="ob")

red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_dot = mlines.Line2D([], [], color="b", marker=".", markersize=15, label="WHO")

plt.legend(handles=[red_line, blue_dot])

plt.savefig(make_graph_file_name("Proportion_TB_Cases_MDR"))

plt.show()

# ---------------------------------------------------------------------- #

# proportion TB cases that are HIV+
# expect around 60% falling to 50% by 2017

make_plot(
    title_str="Proportion of active cases that are HIV+",
    model=model_tb_hiv_prop["median"],
    model_low=model_tb_hiv_prop["lower"],
    model_high=model_tb_hiv_prop["upper"],
    xlim=2023,
    ylim_lower=0,
    ylim_upper=1,
    xlab="Year",
    ylab="Proportion of MDR cases"
)

red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")

plt.legend(handles=[red_line])

plt.savefig(make_graph_file_name("Proportion_TB_Cases_HIV"))

plt.show()

# ---------------------------------------------------------------------- #

# AIDS deaths (including HIV/TB deaths)

# Make plot
fig, ax = plt.subplots()

ax.plot(data_hiv_unaids_deaths.index[0:9],
        aids_deaths["median_aids_deaths_rate_100kpy"][0:9], "-", color="C3")
ax.fill_between(data_hiv_unaids_deaths.index[0:9],
                aids_deaths["lower_aids_deaths_rate_100kpy"][0:9],
                aids_deaths["upper_aids_deaths_rate_100kpy"][0:9], color="C3", alpha=0.2)


ax.plot(data_hiv_unaids_deaths.index, data_hiv_unaids_deaths["AIDS_mortality_per_100k"], "-", color="C0")
ax.fill_between(data_hiv_unaids_deaths.index,
                data_hiv_unaids_deaths["AIDS_mortality_per_100k_lower"],
                data_hiv_unaids_deaths["AIDS_mortality_per_100k_upper"], color="C0", alpha=0.2)

ax.set_ylabel("AIDS mortality per 100,000")
ax.set_xlabel("Year")

plt.ylim((0, 500))
plt.xlim(2010, 2023)
plt.title("AIDS mortality per 100,000 population")

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")

plt.legend(handles=[red_line, blue_line])

plt.savefig(make_graph_file_name("AIDS_mortality"))

plt.show()
#

# ---------------------------------------------------------------------- #

# TB deaths (excluding HIV/TB deaths)
fig, ax = plt.subplots()

ax.plot(data_tb_who.index[0:9],
        tb_deaths["median_tb_deaths_rate_100kpy"][0:9], "-", color="C3")
ax.fill_between(data_tb_who.index[0:9],
                tb_deaths["lower_tb_deaths_rate_100kpy"][0:9],
                tb_deaths["upper_tb_deaths_rate_100kpy"][0:9], color="C3", alpha=0.2)

ax.plot(data_tb_who.index[0:9], data_tb_who["mortality_tb_excl_hiv_per_100k"][0:9], "-", color="C0")
ax.fill_between(data_tb_who.index[0:9],
                data_tb_who["mortality_tb_excl_hiv_per_100k_low"][0:9],
                data_tb_who["mortality_tb_excl_hiv_per_100k_high"][0:9], color="C0", alpha=0.2)

ax.set_ylabel("TB mortality per 100,000")
ax.set_xlabel("Year")

plt.ylim((0, 100))
plt.xlim(2010, 2023)
plt.title("TB mortality per 100,000 population")

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="WHO")

plt.legend(handles=[red_line, blue_line])

plt.savefig(make_graph_file_name("TB_mortality"))

plt.show()

# ---------------------------------------------------------------------- #

# TB treatment coverage
make_plot(
    title_str="TB treatment coverage (%)",
    model=model_tb_tx["median"] * 100,
    model_low=model_tb_tx["lower"] * 100,
    model_high=model_tb_tx["upper"] * 100,
    data_name="NTP",
    data_mid=data_tb_ntp["treatment_coverage"],
    xlim=2023,
    ylim_lower=0,
    ylim_upper=100,
    xlab="Year",
    ylab="TB treatment coverage (%)"
)
plt.savefig(make_graph_file_name("TB_treatment_coverage"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV treatment coverage
make_plot(
    title_str="HIV treatment coverage",
    model=model_hiv_tx["median"] * 100,
    model_low=model_hiv_tx["lower"] * 100,
    model_high=model_hiv_tx["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_unaids["ART_coverage_all_HIV_adults"],
    data_low=data_hiv_unaids["ART_coverage_all_HIV_adults_lower"],
    data_high=data_hiv_unaids["ART_coverage_all_HIV_adults_upper"],
    xlim=2023,
    ylim_lower=0,
    ylim_upper=100,
    xlab="Year",
    ylab="HIV treatment coverage (%)"
)

plt.savefig(make_graph_file_name("HIV_treatment_coverage"))

plt.show()

# ---------------------------------------------------------------------- #
# %%: DEATHS - GBD COMPARISON
# ---------------------------------------------------------------------- #
# get numbers of deaths from model runs
results_deaths = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    ),
    do_scaling=True,
)

results_deaths = results_deaths.reset_index()

# results_deaths.columns.get_level_values(1)
# Index(['', '', 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype='object', name='run')
#
# results_deaths.columns.get_level_values(0)  # this is higher level
# Index(['year', 'cause', 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype='object', name='draw')

# AIDS deaths
# select cause of death
tmp = results_deaths.loc[
    (results_deaths.cause == "AIDS_TB") | (results_deaths.cause == "AIDS_non_TB")
    ]
# select draw - drop columns where draw != 0, but keep year and cause
tmp2 = tmp.loc[
       :,
       ].copy()  # selects only columns for draw=0 (removes year/cause)

# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_AIDS = pd.DataFrame(tmp2.groupby(["year"]).sum())

# double check all columns are float64 or quantile argument will fail
model_2012_median = model_deaths_AIDS.iloc[2].quantile(0.5)
model_2017_median = model_deaths_AIDS.iloc[5].quantile(0.5)
model_2012_low = model_deaths_AIDS.iloc[2].quantile(0.025)
model_2017_low = model_deaths_AIDS.iloc[5].quantile(0.025)
model_2012_high = model_deaths_AIDS.iloc[2].quantile(0.975)
model_2017_high = model_deaths_AIDS.iloc[5].quantile(0.975)

# get GBD estimates from any log_filepath
# death_compare = compare_number_of_deaths('outputs/tb_transmission_runs__2022-09-07T181342.log', resourcefilepath)
death_compare = compare_number_of_deaths('outputs/tb_transmission_runs__2022-12-02T170813.log',
                                         resourcefilepath)

# sim.log_filepath example: 'outputs/tb_transmission_runs__2022-12-02T170813.log'

# include all ages and both sexes
deaths2010 = death_compare.loc[("2010-2014", slice(None), slice(None), "AIDS")].sum()
deaths2015 = death_compare.loc[("2015-2019", slice(None), slice(None), "AIDS")].sum()

# include all ages and both sexes
deaths2010_TB = death_compare.loc[("2010-2014", slice(None), slice(None), "non_AIDS_TB")].sum()
deaths2015_TB = death_compare.loc[("2015-2019", slice(None), slice(None), "non_AIDS_TB")].sum()

x_vals = [1, 2, 3, 4]
labels = ["2010-2014", "2010-2014", "2015-2019", "2015-2019"]
col = ["mediumblue", "mediumseagreen", "mediumblue", "mediumseagreen"]
# handles for legend
blue_patch = mpatches.Patch(color="mediumblue", label="GBD")
green_patch = mpatches.Patch(color="mediumseagreen", label="TLO")

# plot AIDS deaths
y_vals = [
    deaths2010["GBD_mean"],
    model_2012_median,
    deaths2015["GBD_mean"],
    model_2017_median,
]
y_lower = [
    abs(deaths2010["GBD_lower"] - deaths2010["GBD_mean"]),
    abs(model_2012_low - model_2012_median),
    abs(deaths2015["GBD_lower"] - deaths2015["GBD_mean"]),
    abs(model_2017_low - model_2017_median),
]
y_upper = [
    abs(deaths2010["GBD_upper"] - deaths2010["GBD_mean"]),
    abs(model_2012_high - model_2012_median),
    abs(deaths2015["GBD_upper"] - deaths2015["GBD_mean"]),
    abs(model_2017_high - model_2017_median),
]
plt.bar(x_vals, y_vals, color=col)
plt.errorbar(
    x_vals, y_vals,
    yerr=[y_lower, y_upper],
    ls="none",
    marker="o",
    markeredgecolor="red",
    markerfacecolor="red",
    ecolor="red",
)
plt.xticks(ticks=x_vals, labels=labels)
plt.title("Deaths per year due to AIDS")
plt.ylabel("Numbers of deaths per year")
plt.xlabel("Year")
plt.legend(handles=[blue_patch, green_patch])
plt.tight_layout()
plt.savefig(make_graph_file_name("AIDS_deaths_with_GBD"))
plt.show()

# -------------------------------------------------------------------------------------

# TB deaths
# select cause of death
tmp = results_deaths.loc[(results_deaths.cause == "TB")]
# select draw - drop columns where draw != 0, but keep year and cause
tmp2 = tmp.loc[
       :,
       ].copy()  # selects only columns for draw=0 (removes year/cause)

# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_TB = pd.DataFrame(tmp2.groupby(["year"]).sum())

# double check all columns are float64 or quantile argument will fail
model_2010_median = model_deaths_TB.iloc[2].quantile(0.5)  # uses 2012 value
model_2015_median = model_deaths_TB.iloc[5].quantile(0.5)  # uses 2017 value
model_2010_low = model_deaths_TB.iloc[2].quantile(0.025)
model_2015_low = model_deaths_TB.iloc[5].quantile(0.025)
model_2010_high = model_deaths_TB.iloc[2].quantile(0.975)
model_2015_high = model_deaths_TB.iloc[5].quantile(0.975)

x_vals = [1, 2, 3, 4]
labels = ["2010-2014", "2010-2014", "2015-2019", "2015-2019"]
col = ["mediumblue", "mediumseagreen", "mediumblue", "mediumseagreen"]
# handles for legend
blue_patch = mpatches.Patch(color="mediumblue", label="GBD")
green_patch = mpatches.Patch(color="mediumseagreen", label="TLO")

# plot AIDS deaths
y_vals = [
    deaths2015_TB["GBD_mean"],
    model_2010_median,
    deaths2015_TB["GBD_mean"],
    model_2015_median,
]
y_lower = [
    abs(deaths2015_TB["GBD_lower"] - deaths2015_TB["GBD_mean"]),
    abs(model_2010_low - model_2010_median),
    abs(deaths2015_TB["GBD_lower"] - deaths2015_TB["GBD_mean"]),
    abs(model_2015_low - model_2015_median),
]
y_upper = [
    abs(deaths2015_TB["GBD_upper"] - deaths2015_TB["GBD_mean"]),
    abs(model_2010_high - model_2010_median),
    abs(deaths2015_TB["GBD_upper"] - deaths2015_TB["GBD_mean"]),
    abs(model_2015_high - model_2015_median),
]
plt.bar(x_vals, y_vals, color=col)
plt.errorbar(
    x_vals, y_vals,
    yerr=[y_lower, y_upper],
    ls="none",
    marker="o",
    markeredgecolor="red",
    markerfacecolor="red",
    ecolor="red",
)
plt.xticks(ticks=x_vals, labels=labels)
plt.title("Deaths per year due to TB")
plt.ylabel("Numbers of deaths per year")
plt.xlabel("Year")
plt.legend(handles=[blue_patch, green_patch])
plt.tight_layout()
plt.savefig(make_graph_file_name("TB_deaths_with_GBD"))
plt.show()
