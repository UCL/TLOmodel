"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder

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

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

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
data_hiv_mphia_inc_estimate = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence"
].values[0]
data_hiv_mphia_inc_lower = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence_lower"
].values[0]
data_hiv_mphia_inc_upper = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence_upper"
].values[0]
data_hiv_mphia_inc_yerr = [
    abs(data_hiv_mphia_inc_lower - data_hiv_mphia_inc_estimate),
    abs(data_hiv_mphia_inc_upper - data_hiv_mphia_inc_estimate),
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
results_folder = get_scenario_outputs("scenario_batch_runs.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# %% extract results
# Load and format model results (with year as integer):
# ---------------------------------- HIV ---------------------------------- #
hiv_adult_prev = summarize(extract_results(
    results_folder,
    module="tlo.methods.hiv",
    key="summary_inc_and_prev_for_adults_and_children_and_fsw",
    column="hiv_prev_adult_15plus",
    index="date",
    do_scaling=False
),
    only_mean=False,
    collapse_columns=True
)

# Make plot
fig, ax = plt.subplots()
# baseline
ax.plot(hiv_adult_prev.index, hiv_adult_prev[(0, 'mean')], "-", color="C0")
ax.fill_between(hiv_adult_prev.index, hiv_adult_prev[(0, 'lower')], hiv_adult_prev[(0, 'upper')], color="C0", alpha=0.2)
# sc1
ax.plot(hiv_adult_prev.index, hiv_adult_prev[(1, 'mean')], "-", color="C1")
ax.fill_between(hiv_adult_prev.index, hiv_adult_prev[(1, 'lower')], hiv_adult_prev[(1, 'upper')], color="C1", alpha=0.2)
# sc2
ax.plot(hiv_adult_prev.index, hiv_adult_prev[(2, 'mean')], "-", color="C2")
ax.fill_between(hiv_adult_prev.index, hiv_adult_prev[(2, 'lower')], hiv_adult_prev[(2, 'upper')], color="C2", alpha=0.2)
# sc3
ax.plot(hiv_adult_prev.index, hiv_adult_prev[(3, 'mean')], "-", color="C3")
ax.fill_between(hiv_adult_prev.index, hiv_adult_prev[(3, 'lower')], hiv_adult_prev[(3, 'upper')], color="C3", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("HIV prevalence in adults")
plt.ylabel("HIV prevalence")
plt.legend(["Baseline", "Scenario 1", "Scenario 2", "Scenario 3"])

plt.show()

########## HIV incidence ############################
hiv_adult_inc = summarize(extract_results(
    results_folder,
    module="tlo.methods.hiv",
    key="summary_inc_and_prev_for_adults_and_children_and_fsw",
    column="hiv_adult_inc_1549",
    index="date",
    do_scaling=False
),
    only_mean=False,
    collapse_columns=True
)

# Make plot
fig, ax = plt.subplots()
# baseline
ax.plot(hiv_adult_inc.index, hiv_adult_inc[(0, 'mean')], "-", color="C0")
ax.fill_between(hiv_adult_inc.index, hiv_adult_inc[(0, 'lower')], hiv_adult_inc[(0, 'upper')], color="C0", alpha=0.2)
# sc1
ax.plot(hiv_adult_inc.index, hiv_adult_inc[(1, 'mean')], "-", color="C1")
ax.fill_between(hiv_adult_inc.index, hiv_adult_inc[(1, 'lower')], hiv_adult_inc[(1, 'upper')], color="C1", alpha=0.2)
# sc2
ax.plot(hiv_adult_inc.index, hiv_adult_inc[(2, 'mean')], "-", color="C2")
ax.fill_between(hiv_adult_inc.index, hiv_adult_inc[(2, 'lower')], hiv_adult_inc[(2, 'upper')], color="C2", alpha=0.2)
# sc3
ax.plot(hiv_adult_inc.index, hiv_adult_inc[(3, 'mean')], "-", color="C3")
ax.fill_between(hiv_adult_inc.index, hiv_adult_inc[(3, 'lower')], hiv_adult_inc[(3, 'upper')], color="C3", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("HIV incidence in adults 15-49")
plt.ylabel("HIV incidence")
plt.legend(["Baseline", "Scenario 1", "Scenario 2", "Scenario 3"])

plt.show()

# ---------------------------------- PERSON-YEARS ---------------------------------- #
draw = 0

# todo need mean py for each draw
# function to extract person-years by year
# call this for each run and then take the mean to use as denominator for mortality / incidence etc.

def get_person_years(draw, run):
    log = load_pickled_dataframes(results_folder, draw, run)

    py_ = log["tlo.methods.demography"]["person_years"]
    years = pd.to_datetime(py_["date"]).dt.year
    py = pd.Series(dtype="int64", index=years)
    for year in years:
        tot_py = (
            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series) +
            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
        ).transpose()
        py[year] = tot_py.sum().values[0]

    py.index = pd.to_datetime(years, format="%Y")

    return py


# for each draw, get mean py across all runs
py_summary = pd.DataFrame(data=None, columns=range(0, info["number_of_draws"]))

# draw number (default = 0) is specified above
for draw in range(0, info["number_of_draws"]):
    tmp = pd.DataFrame(data=None, columns=range(0, info["runs_per_draw"]))

    for run in range(0, info["runs_per_draw"]):
        tmp.iloc[:, run] = get_person_years(draw, run)

    py_summary.iloc[:, draw] = tmp.mean(axis=1)

# ---------------------------------- TB ---------------------------------- #


tb_inc = summarize(extract_results(
    results_folder,
    module="tlo.methods.tb",
    key="tb_incidence",
    column="num_new_active_tb",
    index="date",
    do_scaling=False
),
    only_mean=False,
    collapse_columns=True
)

# active tb incidence rates
activeTB_inc_rate0 = pd.Series(
    (tb_inc[(0, 'mean')].values / py_summary[0].values[1:41]) * 100000
)
activeTB_inc_rate0_lower = pd.Series(
    (tb_inc[(0, 'lower')].values / py_summary[0].values[1:41]) * 100000
)
activeTB_inc_rate0_upper = pd.Series(
    (tb_inc[(0, 'upper')].values / py_summary[0].values[1:41]) * 100000
)

activeTB_inc_rate1 = pd.Series(
    (tb_inc[(1, 'mean')].values / py_summary[1].values[1:41]) * 100000
)
activeTB_inc_rate1_lower = pd.Series(
    (tb_inc[(1, 'lower')].values / py_summary[1].values[1:41]) * 100000
)
activeTB_inc_rate1_upper = pd.Series(
    (tb_inc[(1, 'upper')].values / py_summary[1].values[1:41]) * 100000
)

activeTB_inc_rate2 = pd.Series(
    (tb_inc[(2, 'mean')].values / py_summary[2].values[1:41]) * 100000
)
activeTB_inc_rate2_lower = pd.Series(
    (tb_inc[(2, 'lower')].values / py_summary[2].values[1:41]) * 100000
)
activeTB_inc_rate2_upper = pd.Series(
    (tb_inc[(2, 'upper')].values / py_summary[2].values[1:41]) * 100000
)

activeTB_inc_rate3 = pd.Series(
    (tb_inc[(3, 'mean')].values / py_summary[3].values[1:41]) * 100000
)
activeTB_inc_rate3_lower = pd.Series(
    (tb_inc[(3, 'lower')].values / py_summary[3].values[1:41]) * 100000
)
activeTB_inc_rate3_upper = pd.Series(
    (tb_inc[(3, 'upper')].values / py_summary[3].values[1:41]) * 100000
)

# Make plot
fig, ax = plt.subplots()
# baseline
ax.plot(tb_inc.index, activeTB_inc_rate0, "-", color="C0")
ax.fill_between(tb_inc.index, activeTB_inc_rate0_lower, activeTB_inc_rate0_upper, color="C0", alpha=0.2)
# sc1
ax.plot(tb_inc.index, activeTB_inc_rate1, "-", color="C1")
ax.fill_between(tb_inc.index, activeTB_inc_rate1_lower, activeTB_inc_rate1_upper, color="C0", alpha=0.2)
# sc2
ax.plot(tb_inc.index, activeTB_inc_rate2, "-", color="C2")
ax.fill_between(tb_inc.index, activeTB_inc_rate2_lower, activeTB_inc_rate2_upper, color="C0", alpha=0.2)
# sc3
ax.plot(tb_inc.index, activeTB_inc_rate3, "-", color="C3")
ax.fill_between(tb_inc.index, activeTB_inc_rate3_lower, activeTB_inc_rate3_upper, color="C0", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.ylim((0, 500))
plt.title("Active TB incidence")
plt.ylabel("TB incidence")
plt.legend(["Baseline", "Scenario 1", "Scenario 2", "Scenario 3"])

plt.show()


# ---------------------------------- DEATHS ---------------------------------- #

results_deaths = summarize(extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    ),
    do_scaling=False,
),
    only_mean=False,
    collapse_columns=True
)

# make year and cause of death into column
results_deaths = results_deaths.reset_index()
tmp = results_deaths.drop(results_deaths[results_deaths.cause == "Other"].index)

