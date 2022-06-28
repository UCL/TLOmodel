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
# Load and format model results (with year as integer):
# ---------------------------------- HIV ---------------------------------- #
hiv_adult_prev = extract_results(
    results0,
    module="tlo.methods.hiv",
    key="summary_inc_and_prev_for_adults_and_children_and_fsw",
    column="hiv_prev_adult_15plus",
    index="date",
    do_scaling=False
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

