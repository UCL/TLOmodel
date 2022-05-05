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
results_folder1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results_folder3 = get_scenario_outputs("scenario3.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder1 / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder1)

# get basic information about the results
info = get_scenario_info(results_folder1)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder1)

# todo rename all draws to 0 to get summarised outputs in each results folder
draw = 0

# %% extract results
# Load and format model results (with year as integer):
# ---------------------------------- HIV ---------------------------------- #
hiv_adult_prev1 = extract_results(
        results_folder1,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_adult_15plus",
        index="date",
        do_scaling=False
)
# flatten multi-index
hiv_adult_prev1.columns = hiv_adult_prev1.columns.get_level_values(0)
hiv_adult_prev1["median"] = hiv_adult_prev1.median(axis=1)
hiv_adult_prev1["lower"] = hiv_adult_prev1.quantile(q=0.025, axis=1)
hiv_adult_prev1["upper"] = hiv_adult_prev1.quantile(q=0.975, axis=1)

hiv_adult_prev3 = extract_results(
        results_folder3,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_adult_15plus",
        index="date",
        do_scaling=False
)
# flatten multi-index
hiv_adult_prev3.columns = hiv_adult_prev3.columns.get_level_values(0)
hiv_adult_prev3["median"] = hiv_adult_prev3.median(axis=1)
hiv_adult_prev3["lower"] = hiv_adult_prev3.quantile(q=0.025, axis=1)
hiv_adult_prev3["upper"] = hiv_adult_prev3.quantile(q=0.975, axis=1)

# Make plot
fig, ax = plt.subplots()
ax.plot(hiv_adult_prev1.index, hiv_adult_prev1["median"], "-", color="C3")
ax.fill_between(hiv_adult_prev1.index, hiv_adult_prev1["lower"], hiv_adult_prev1["upper"], color="C3", alpha=0.2)

ax.plot(hiv_adult_prev3.index, hiv_adult_prev3["median"], "-", color="C2")
ax.fill_between(hiv_adult_prev3.index, hiv_adult_prev3["lower"], hiv_adult_prev3["upper"], color="C2", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("HIV prevalence in adults")
plt.ylabel("HIV prevalence")
plt.legend(["Scenario 1", "Scenario 3"])

plt.show()




########## HIV incidence ############################
hiv_adult_inc1 = extract_results(
        results_folder1,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False
)
# flatten multi-index
hiv_adult_inc1.columns = hiv_adult_inc1.columns.get_level_values(0)
hiv_adult_inc1["median"] = hiv_adult_inc1.median(axis=1)
hiv_adult_inc1["lower"] = hiv_adult_inc1.quantile(q=0.025, axis=1)
hiv_adult_inc1["upper"] = hiv_adult_inc1.quantile(q=0.975, axis=1)

hiv_adult_inc3 = extract_results(
        results_folder3,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False
)
# flatten multi-index
hiv_adult_inc3.columns = hiv_adult_inc3.columns.get_level_values(0)
hiv_adult_inc3["median"] = hiv_adult_inc3.median(axis=1)
hiv_adult_inc3["lower"] = hiv_adult_inc3.quantile(q=0.025, axis=1)
hiv_adult_inc3["upper"] = hiv_adult_inc3.quantile(q=0.975, axis=1)

# Make plot
fig, ax = plt.subplots()
ax.plot(hiv_adult_inc1.index, hiv_adult_inc1["median"], "-", color="C3")
ax.fill_between(hiv_adult_inc1.index, hiv_adult_inc1["lower"], hiv_adult_inc1["upper"], color="C3", alpha=0.2)

ax.plot(hiv_adult_inc3.index, hiv_adult_inc3["median"], "-", color="C2")
ax.fill_between(hiv_adult_inc3.index, hiv_adult_inc3["lower"], hiv_adult_inc3["upper"], color="C2", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("HIV incidence in adults 15-49")
plt.ylabel("HIV incidence")
plt.legend(["Scenario 1", "Scenario 3"])

plt.show()

# ---------------------------------- PERSON-YEARS ---------------------------------- #

# function to extract person-years by year
# call this for each run and then take the mean to use as denominator for mortality / incidence etc.
def get_person_years(draw, run):
    log = load_pickled_dataframes(results_folder1, draw, run)

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


# for draw 0, get py for all runs
number_runs = info["runs_per_draw"]
py_summary = pd.DataFrame(data=None, columns=range(0, number_runs))

# draw number (default = 0) is specified above
for run in range(0, number_runs):
    py_summary.iloc[:, run] = get_person_years(draw, run)

py_summary["mean"] = py_summary.mean(axis=1)

# ---------------------------------- TB ---------------------------------- #

tb_inc1 = extract_results(
        results_folder1,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
)

tb_inc1.columns = tb_inc1.columns.get_level_values(0)
tb_inc1["median"] = tb_inc1.median(axis=1)
tb_inc1["lower"] = tb_inc1.quantile(q=0.025, axis=1)
tb_inc1["upper"] = tb_inc1.quantile(q=0.975, axis=1)

activeTB_inc_rate1 = pd.Series(
    (tb_inc1["median"].values / py_summary["mean"].values[1:41]) * 100000
)
activeTB_inc_rate1.index = tb_inc1.index
activeTB_inc_rate_low1 = pd.Series(
    (tb_inc1["lower"].values / py_summary["mean"].values[1:41]) * 100000
)
activeTB_inc_rate_low1.index = tb_inc1.index
activeTB_inc_rate_high1 = pd.Series(
    (tb_inc1["upper"].values / py_summary["mean"].values[1:41]) * 100000
)
activeTB_inc_rate_high1.index = tb_inc1.index


tb_inc3 = extract_results(
        results_folder3,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
)

tb_inc3.columns = tb_inc3.columns.get_level_values(0)
tb_inc3["median"] = tb_inc3.median(axis=1)
tb_inc3["lower"] = tb_inc3.quantile(q=0.025, axis=1)
tb_inc3["upper"] = tb_inc3.quantile(q=0.975, axis=1)

activeTB_inc_rate3 = pd.Series(
    (tb_inc3["median"].values / py_summary["mean"].values[1:41]) * 100000
)
activeTB_inc_rate3.index = tb_inc3.index
activeTB_inc_rate_low3 = pd.Series(
    (tb_inc3["lower"].values / py_summary["mean"].values[1:41]) * 100000
)
activeTB_inc_rate_low3.index = tb_inc3.index
activeTB_inc_rate_high3 = pd.Series(
    (tb_inc3["upper"].values / py_summary["mean"].values[1:41]) * 100000
)
activeTB_inc_rate_high3.index = tb_inc3.index


fig, ax = plt.subplots()
ax.plot(activeTB_inc_rate1.index, activeTB_inc_rate1, "-", color="C1")
ax.fill_between(activeTB_inc_rate1.index, activeTB_inc_rate_low1, activeTB_inc_rate_high1, color="C2", alpha=0.2)

ax.plot(activeTB_inc_rate3.index, activeTB_inc_rate3, "-", color="C2")
ax.fill_between(activeTB_inc_rate3.index, activeTB_inc_rate_low3, activeTB_inc_rate_high3, color="C2", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("TB incidence per 100,000 population")
plt.ylabel("TB incidence")
plt.legend(["Scenario 1", "Scenario 3"])

plt.show()



## latent TB ######################################

tb_latent1 = extract_results(
        results_folder1,
        module="tlo.methods.tb",
        key="tb_prevalence",
        column="tbPrevLatent",
        index="date",
        do_scaling=False
)
# flatten multi-index
tb_latent1.columns = tb_latent1.columns.get_level_values(0)
tb_latent1["median"] = tb_latent1.median(axis=1)
tb_latent1["lower"] = tb_latent1.quantile(q=0.025, axis=1)
tb_latent1["upper"] = tb_latent1.quantile(q=0.975, axis=1)


tb_latent3 = extract_results(
        results_folder3,
        module="tlo.methods.tb",
        key="tb_prevalence",
        column="tbPrevLatent",
        index="date",
        do_scaling=False
)
# flatten multi-index
tb_latent3.columns = tb_latent3.columns.get_level_values(0)
tb_latent3["median"] = tb_latent3.median(axis=1)
tb_latent3["lower"] = tb_latent3.quantile(q=0.025, axis=1)
tb_latent3["upper"] = tb_latent3.quantile(q=0.975, axis=1)

# Make plot
fig, ax = plt.subplots()
ax.plot(tb_latent1.index, tb_latent1["median"], "-", color="C3")
ax.fill_between(tb_latent1.index, tb_latent1["lower"], tb_latent1["upper"], color="C3", alpha=0.2)

ax.plot(tb_latent3.index, tb_latent3["median"], "-", color="C2")
ax.fill_between(tb_latent3.index, tb_latent3["lower"], tb_latent3["upper"], color="C2", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("Latent TB prevalence")
plt.ylabel("Prevalence")
plt.legend(["Scenario 1", "Scenario 3"])

plt.show()





model_hiv_child_prev = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_child",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_child_prev.index = model_hiv_child_prev.index.year

model_hiv_child_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_child_inc",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_child_inc.index = model_hiv_child_inc.index.year

model_hiv_fsw_prev = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_fsw",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_fsw_prev.index = model_hiv_fsw_prev.index.year


# ---------------------------------- PERSON-YEARS ---------------------------------- #

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


# for draw 0, get py for all runs
number_runs = info["runs_per_draw"]
py_summary = pd.DataFrame(data=None, columns=range(0, number_runs))

# draw number (default = 0) is specified above
for run in range(0, number_runs):
    py_summary.iloc[:, run] = get_person_years(draw, run)

py_summary["mean"] = py_summary.mean(axis=1)

# ---------------------------------- TB ---------------------------------- #

model_tb_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_tb_inc.index = model_tb_inc.index.year
activeTB_inc_rate = pd.Series(
    (model_tb_inc["mean"].values / py_summary["mean"].values) * 100000
)
activeTB_inc_rate.index = model_tb_inc.index
activeTB_inc_rate_low = pd.Series(
    (model_tb_inc["lower"].values / py_summary["mean"].values) * 100000
)
activeTB_inc_rate_low.index = model_tb_inc.index
activeTB_inc_rate_high = pd.Series(
    (model_tb_inc["upper"].values / py_summary["mean"].values) * 100000
)
activeTB_inc_rate_high.index = model_tb_inc.index

model_tb_latent = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_prevalence",
        column="tbPrevLatent",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)

model_tb_latent.index = model_tb_latent.index.year

model_tb_mdr = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_mdr",
        column="tbPropActiveCasesMdr",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)

model_tb_mdr.index = model_tb_mdr.index.year

model_tb_hiv_prop = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="prop_active_tb_in_plhiv",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
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
       :, ("draw" == draw)
       ].copy()  # selects only columns for draw=0 (removes year/cause)
# join year and cause back to df - needed for groupby
frames = [tmp["year"], tmp["cause"], tmp2]
tmp3 = pd.concat(frames, axis=1)

# create new column names, dependent on number of runs in draw
base_columns = ["year", "cause"]
run_columns = ["run" + str(x) for x in range(0, info["runs_per_draw"])]
base_columns.extend(run_columns)
tmp3.columns = base_columns
tmp3 = tmp3.set_index("year")

# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_AIDS = pd.DataFrame(tmp3.groupby(["year"]).sum())

# double check all columns are float64 or quantile argument will fail
cols = [
    col
    for col in model_deaths_AIDS.columns
    if model_deaths_AIDS[col].dtype == "float64"
]
model_deaths_AIDS["median"] = (
    model_deaths_AIDS[cols].astype(float).quantile(0.5, axis=1)
)
model_deaths_AIDS["lower"] = (
    model_deaths_AIDS[cols].astype(float).quantile(0.025, axis=1)
)
model_deaths_AIDS["upper"] = (
    model_deaths_AIDS[cols].astype(float).quantile(0.975, axis=1)
)

# AIDS mortality rates per 100k person-years
total_aids_deaths_rate_100kpy = pd.Series(
    (model_deaths_AIDS["median"].values / py_summary["mean"].values) * 100000
)
total_aids_deaths_rate_100kpy_lower = pd.Series(
    (model_deaths_AIDS["lower"].values / py_summary["mean"].values) * 100000
)
total_aids_deaths_rate_100kpy_upper = pd.Series(
    (model_deaths_AIDS["upper"].values / py_summary["mean"].values) * 100000
)
total_aids_deaths_rate_100kpy.index = model_deaths_AIDS.index
total_aids_deaths_rate_100kpy_lower.index = model_deaths_AIDS.index
total_aids_deaths_rate_100kpy_upper.index = model_deaths_AIDS.index

# HIV/TB deaths
# select cause of death
tmp = results_deaths.loc[
    (results_deaths.cause == "AIDS_TB")
]
# select draw - drop columns where draw != 0, but keep year and cause
tmp2 = tmp.loc[
       :, ("draw" == draw)
       ].copy()  # selects only columns for draw=0 (removes year/cause)
# join year and cause back to df - needed for groupby
frames = [tmp["year"], tmp["cause"], tmp2]
tmp3 = pd.concat(frames, axis=1)

# create new column names, dependent on number of runs in draw
base_columns = ["year", "cause"]
run_columns = ["run" + str(x) for x in range(0, info["runs_per_draw"])]
base_columns.extend(run_columns)
tmp3.columns = base_columns
tmp3 = tmp3.set_index("year")

# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_AIDS_TB = pd.DataFrame(tmp3.groupby(["year"]).sum())

# double check all columns are float64 or quantile argument will fail
cols = [
    col
    for col in model_deaths_AIDS_TB.columns
    if model_deaths_AIDS_TB[col].dtype == "float64"
]
model_deaths_AIDS_TB["median"] = (
    model_deaths_AIDS_TB[cols].astype(float).quantile(0.5, axis=1)
)
model_deaths_AIDS_TB["lower"] = (
    model_deaths_AIDS_TB[cols].astype(float).quantile(0.025, axis=1)
)
model_deaths_AIDS_TB["upper"] = (
    model_deaths_AIDS_TB[cols].astype(float).quantile(0.975, axis=1)
)

# AIDS_TB mortality rates per 100k person-years
total_aids_TB_deaths_rate_100kpy = pd.Series(
    (model_deaths_AIDS_TB["median"].values / py_summary["mean"].values) * 100000
)
total_aids_TB_deaths_rate_100kpy_lower = pd.Series(
    (model_deaths_AIDS_TB["lower"].values / py_summary["mean"].values) * 100000
)
total_aids_TB_deaths_rate_100kpy_upper = pd.Series(
    (model_deaths_AIDS_TB["upper"].values / py_summary["mean"].values) * 100000
)
total_aids_TB_deaths_rate_100kpy.index = model_deaths_AIDS_TB.index
total_aids_TB_deaths_rate_100kpy_lower.index = model_deaths_AIDS_TB.index
total_aids_TB_deaths_rate_100kpy_upper.index = model_deaths_AIDS_TB.index

# TB deaths
# select cause of death
tmp = results_deaths.loc[(results_deaths.cause == "TB")]
# rename columns
run_columns = [
    "draw" + str(y) + "run" + str(x)
    for y in range(0, info["number_of_draws"])
    for x in range(0, info["runs_per_draw"])
]
base_columns = ["year", "cause"]
base_columns.extend(run_columns)
tmp.columns = base_columns

# add in any missing years
year_series = pd.DataFrame(
    model_hiv_adult_prev.index
)  # some years missing from TB deaths outputs

# fill with zeros
tmp2 = pd.merge(
    left=year_series, right=tmp, left_on="date", right_on="year", how="left"
)
tmp2 = tmp2.fillna(0)

# select draw
draw_name = str("draw" + str(draw))
model_deaths_TB = tmp2.loc[:, tmp2.columns.str.contains(draw_name)]

# join year and cause back to df - needed for groupby
frames = [tmp2["date"], model_deaths_TB]
model_deaths_TB = pd.concat(frames, axis=1)

model_deaths_TB = model_deaths_TB.set_index("date")

# double check all columns are float64 or quantile argument will fail
cols = [
    col for col in model_deaths_TB.columns if model_deaths_TB[col].dtype == "float64"
]
model_deaths_TB["median"] = model_deaths_TB[cols].astype(float).quantile(0.5, axis=1)
model_deaths_TB["lower"] = model_deaths_TB[cols].astype(float).quantile(0.025, axis=1)
model_deaths_TB["upper"] = model_deaths_TB[cols].astype(float).quantile(0.975, axis=1)

# TB mortality rates per 100k person-years
tot_tb_non_hiv_deaths_rate_100kpy = pd.Series(
    (model_deaths_TB["median"].values / py_summary["mean"].values) * 100000
)
tot_tb_non_hiv_deaths_rate_100kpy_lower = pd.Series(
    (model_deaths_TB["lower"].values / py_summary["mean"].values) * 100000
)
tot_tb_non_hiv_deaths_rate_100kpy_upper = pd.Series(
    (model_deaths_TB["upper"].values / py_summary["mean"].values) * 100000
)
tot_tb_non_hiv_deaths_rate_100kpy.index = model_deaths_AIDS.index
tot_tb_non_hiv_deaths_rate_100kpy_lower.index = model_deaths_AIDS.index
tot_tb_non_hiv_deaths_rate_100kpy_upper.index = model_deaths_AIDS.index

# ---------------------------------- PROGRAM COVERAGE ---------------------------------- #


# TB treatment coverage
model_tb_tx = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbTreatmentCoverage",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)

model_tb_tx.index = model_tb_tx.index.year

# HIV treatment coverage
model_hiv_tx = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="art_coverage_adult",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
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
        ax.set_xlabel(ylab)

    plt.title(title_str)
    plt.legend(["TLO", data_name])
    # plt.gca().set_ylim(bottom=0)
    # plt.savefig(outputspath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format='pdf')


# %% make plots

# HIV - prevalence among in adults aged 15-49

make_plot(
    title_str="HIV Prevalence in Adults Aged 15-49 (%)",
    model=model_hiv_adult_prev["mean"] * 100,
    model_low=model_hiv_adult_prev["lower"] * 100,
    model_high=model_hiv_adult_prev["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_unaids["prevalence_age15_49"] * 100,
    data_low=data_hiv_unaids["prevalence_age15_49_lower"] * 100,
    data_high=data_hiv_unaids["prevalence_age15_49_upper"] * 100,
)

# data: MPHIA
plt.plot(
    model_hiv_adult_prev.index[6],
    data_hiv_mphia_prev.loc[
        data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"
    ].values[0],
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
plt.xlabel = ("Year",)
plt.ylabel = "HIV prevalence (%)"

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
    model=model_hiv_adult_inc["mean"] * 100,
    model_low=model_hiv_adult_inc["lower"] * 100,
    model_high=model_hiv_adult_inc["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_unaids["incidence_per_1000"] / 10,
    data_low=data_hiv_unaids["incidence_per_1000_lower"] / 10,
    data_high=data_hiv_unaids["incidence_per_1000_upper"] / 10,
)

plt.xlabel = ("Year",)
plt.ylabel = "HIV incidence per 1000 population"

# MPHIA
plt.errorbar(
    model_hiv_adult_inc.index[6],
    data_hiv_mphia_inc_estimate,
    yerr=[[data_hiv_mphia_inc_yerr[0]], [data_hiv_mphia_inc_yerr[1]]],
    fmt="gx",
)

plt.ylim(0, 1.0)
plt.xlim(2010, 2020)
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
    model=model_hiv_child_prev["mean"] * 100,
    model_low=model_hiv_child_prev["lower"] * 100,
    model_high=model_hiv_child_prev["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_aidsinfo["prevalence_0_14"] * 100,
    data_low=data_hiv_aidsinfo["prevalence_0_14_lower"] * 100,
    data_high=data_hiv_aidsinfo["prevalence_0_14_upper"] * 100,
    xlab="Year",
    ylab="HIV prevalence (%)",
)

# MPHIA
plt.plot(
    model_hiv_child_prev.index[6],
    data_hiv_mphia_prev.loc[
        data_hiv_mphia_prev.age == "Total 0-14", "total percent hiv positive"
    ].values[0],
    "gx",
)

plt.xlim = (2010, 2020)
plt.ylim = (0, 5)

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
    title_str="HIV Incidence in Children (0-14) (per 100 pyar)",
    model=model_hiv_child_inc["mean"] * 100,
    model_low=model_hiv_child_inc["lower"] * 100,
    model_high=model_hiv_child_inc["upper"] * 100,
    data_mid=data_hiv_aidsinfo["incidence0_14_per100py"],
    data_low=data_hiv_aidsinfo["incidence0_14_per100py_lower"],
    data_high=data_hiv_aidsinfo["incidence0_14_per100py_upper"],
)
plt.savefig(make_graph_file_name("HIV_Incidence_in_Children"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV prevalence among female sex workers:
make_plot(
    title_str="HIV Prevalence among Female Sex Workers (%)",
    model=model_hiv_fsw_prev["mean"] * 100,
    model_low=model_hiv_fsw_prev["lower"] * 100,
    model_high=model_hiv_fsw_prev["upper"] * 100,
)
plt.savefig(make_graph_file_name("HIV_Prevalence_FSW"))

plt.show()

# ----------------------------- TB -------------------------------------- #

# Active TB incidence per 100,000 person-years - annual outputs

make_plot(
    title_str="Active TB Incidence (per 100k person-years)",
    model=activeTB_inc_rate,
    model_low=activeTB_inc_rate_low,
    model_high=activeTB_inc_rate_high,
    data_name="WHO_TB",
    data_mid=data_tb_who["incidence_per_100k"],
    data_low=data_tb_who["incidence_per_100k_low"],
    data_high=data_tb_who["incidence_per_100k_high"],
)
plt.savefig(make_graph_file_name("TB_Incidence"))

plt.show()

# ---------------------------------------------------------------------- #

# latent TB prevalence
make_plot(
    title_str="Latent TB prevalence",
    model=model_tb_latent["mean"],
    model_low=model_tb_latent["lower"],
    model_high=model_tb_latent["upper"],
)
# add latent TB estimate from Houben & Dodd 2016 (value for year=2014)
plt.errorbar(
    model_tb_latent.index[4],
    data_tb_latent_estimate,
    yerr=[[data_tb_latent_yerr[0]], [data_tb_latent_yerr[1]]],
    fmt="o",
)
plt.ylim = (0, 0.5)
plt.legend(["Model", "", "Houben & Dodd"])
plt.savefig(make_graph_file_name("Latent_TB_Prevalence"))

plt.show()

# ---------------------------------------------------------------------- #

# proportion TB cases that are MDR

make_plot(
    title_str="Proportion of active TB cases that are MDR",
    model=model_tb_mdr["mean"],
    model_low=model_tb_mdr["lower"],
    model_high=model_tb_mdr["upper"],
)
# data from ResourceFile_TB sheet WHO_mdrTB2017
plt.errorbar(model_tb_mdr.index[7], 0.0075, yerr=[[0.0059], [0.0105]], fmt="o")
plt.legend(["TLO", "WHO"])
plt.savefig(make_graph_file_name("Proportion_TB_Cases_MDR"))

plt.show()

# ---------------------------------------------------------------------- #

# proportion TB cases that are HIV+
# expect around 60% falling to 50% by 2017

make_plot(
    title_str="Proportion of active cases that are HIV+",
    model=model_tb_hiv_prop["mean"],
    model_low=model_tb_hiv_prop["lower"],
    model_high=model_tb_hiv_prop["upper"],
)
plt.savefig(make_graph_file_name("Proportion_TB_Cases_MDR"))

plt.show()

# ---------------------------------------------------------------------- #

# AIDS deaths (including HIV/TB deaths)
make_plot(
    title_str="Mortality to HIV-AIDS per 100,000 capita",
    model=total_aids_deaths_rate_100kpy,
    model_low=total_aids_deaths_rate_100kpy_lower,
    model_high=total_aids_deaths_rate_100kpy_upper,
    data_name="UNAIDS",
    data_mid=data_hiv_unaids_deaths["AIDS_mortality_per_100k"],
    data_low=data_hiv_unaids_deaths["AIDS_mortality_per_100k_lower"],
    data_high=data_hiv_unaids_deaths["AIDS_mortality_per_100k_upper"],
)
plt.savefig(make_graph_file_name("AIDS_mortality"))

plt.show()

# ---------------------------------------------------------------------- #

# AIDS/TB deaths
make_plot(
    title_str="Mortality to HIV-AIDS-TB per 100,000 capita",
    model=total_aids_TB_deaths_rate_100kpy,
    model_low=total_aids_TB_deaths_rate_100kpy_lower,
    model_high=total_aids_TB_deaths_rate_100kpy_upper,
    data_name="WHO",
    data_mid=data_tb_who["mortality_tb_hiv_per_100k"],
    data_low=data_tb_who["mortality_tb_hiv_per_100k_low"],
    data_high=data_tb_who["mortality_tb_hiv_per_100k_high"],
)
plt.savefig(make_graph_file_name("AIDS_TB_mortality"))

plt.show()

# ---------------------------------------------------------------------- #

# TB deaths (excluding HIV/TB deaths)
make_plot(
    title_str="TB mortality rate per 100,000 population",
    model=tot_tb_non_hiv_deaths_rate_100kpy,
    model_low=tot_tb_non_hiv_deaths_rate_100kpy_lower,
    model_high=tot_tb_non_hiv_deaths_rate_100kpy_upper,
    data_name="WHO",
    data_mid=data_tb_who["mortality_tb_excl_hiv_per_100k"],
    data_low=data_tb_who["mortality_tb_excl_hiv_per_100k_low"],
    data_high=data_tb_who["mortality_tb_excl_hiv_per_100k_high"],
)
plt.savefig(make_graph_file_name("TB_mortality"))

plt.show()

# ---------------------------------------------------------------------- #

# TB treatment coverage
make_plot(
    title_str="TB treatment coverage",
    model=model_tb_tx["mean"] * 100,
    model_low=model_tb_tx["lower"] * 100,
    model_high=model_tb_tx["upper"] * 100,
    data_name="NTP",
    data_mid=data_tb_ntp["treatment_coverage"],
)
plt.savefig(make_graph_file_name("TB_treatment_coverage"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV treatment coverage
make_plot(
    title_str="HIV treatment coverage",
    model=model_hiv_tx["mean"] * 100,
    model_low=model_hiv_tx["lower"] * 100,
    model_high=model_hiv_tx["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_unaids["ART_coverage_all_HIV_adults"],
    data_low=data_hiv_unaids["ART_coverage_all_HIV_adults_lower"],
    data_high=data_hiv_unaids["ART_coverage_all_HIV_adults_upper"],
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
       :, ("draw" == draw)
       ].copy()  # selects only columns for draw=0 (removes year/cause)
# join year and cause back to df - needed for groupby
frames = [tmp["year"], tmp["cause"], tmp2]
tmp3 = pd.concat(frames, axis=1)

# create new column names, dependent on number of runs in draw
base_columns = ["year", "cause"]
run_columns = ["run" + str(x) for x in range(0, info["runs_per_draw"])]
base_columns.extend(run_columns)
tmp3.columns = base_columns
tmp3 = tmp3.set_index("year")

# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_AIDS = pd.DataFrame(tmp3.groupby(["year"]).sum())

# double check all columns are float64 or quantile argument will fail
model_2010_median = model_deaths_AIDS.iloc[2].quantile(0.5)
model_2015_median = model_deaths_AIDS.iloc[5].quantile(0.5)
model_2010_low = model_deaths_AIDS.iloc[2].quantile(0.025)
model_2015_low = model_deaths_AIDS.iloc[5].quantile(0.025)
model_2010_high = model_deaths_AIDS.iloc[2].quantile(0.975)
model_2015_high = model_deaths_AIDS.iloc[5].quantile(0.975)

# get GBD estimates from any log_filepath
death_compare = compare_number_of_deaths('outputs/Logfile__2021-10-05T223107.log', resourcefilepath)
# sim.log_filepath example: 'outputs/Logfile__2021-10-04T155631.log'

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
    model_2010_median,
    deaths2015["GBD_mean"],
    model_2015_median,
]
y_lower = [
    abs(deaths2010["GBD_lower"] - deaths2010["GBD_mean"]),
    abs(model_2010_low - model_2010_median),
    abs(deaths2015["GBD_lower"] - deaths2015["GBD_mean"]),
    abs(model_2015_low - model_2015_median),
]
y_upper = [
    abs(deaths2010["GBD_upper"] - deaths2010["GBD_mean"]),
    abs(model_2010_high - model_2010_median),
    abs(deaths2015["GBD_upper"] - deaths2015["GBD_mean"]),
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
plt.title("Deaths per year due to AIDS")
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
       :, ("draw" == draw)
       ].copy()  # selects only columns for draw=0 (removes year/cause)
# join year and cause back to df - needed for groupby
frames = [tmp["year"], tmp["cause"], tmp2]
tmp3 = pd.concat(frames, axis=1)

# create new column names, dependent on number of runs in draw
base_columns = ["year", "cause"]
run_columns = ["run" + str(x) for x in range(0, info["runs_per_draw"])]
base_columns.extend(run_columns)
tmp3.columns = base_columns
tmp3 = tmp3.set_index("year")

# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_TB = pd.DataFrame(tmp3.groupby(["year"]).sum())

# double check all columns are float64 or quantile argument will fail
model_2010_median = model_deaths_TB.iloc[2].quantile(0.5)
model_2015_median = model_deaths_TB.iloc[5].quantile(0.5)
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
plt.legend(handles=[blue_patch, green_patch])
plt.tight_layout()
plt.savefig(make_graph_file_name("TB_deaths_with_GBD"))
plt.show()
