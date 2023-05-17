"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
function weighted_mean_for_data_comparison can be used to select which parameter sets to use
make plots for top 5 parameter sets just to make sure they are looking ok
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

outputspath = Path("./outputs/nic503@york.ac.uk")

# %% read in data files for plots


# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("analysis_impact_of_noxpert_diagnosis.py", outputspath)[-1]

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
activeTB_inc_rate = (model_tb_inc.divide(py_summary["mean"].values[1:10], axis=0)) * 100000

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

# ---------------------------------- Extracting relevant outcomes ---------------------------------- #

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


results_deaths = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    ),
    do_scaling=False,
)

results_deaths = results_deaths.reset_index()

# summarise across runs
aids_tb_deaths_table = results_deaths.loc[results_deaths.cause == "AIDS_TB"]
tb_deaths_table = results_deaths.loc[results_deaths.cause == "TB"]

# ------------ summarise deaths producing df for each draw

# TB deaths excluding HIV
tb_deaths = {}  # dict of df

for draw in info["number_of_draws"]:
    draw = draw

    # rename dataframe
    name = "model_deaths_TB_draw" + str(draw)
    # select cause of death
    tmp = results_deaths.loc[
        (results_deaths.cause == "TB")
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
    tb_deaths[name] = pd.DataFrame(tmp3.groupby(["year"]).sum())

    # double check all columns are float64 or quantile argument will fail
    cols = [
        col
        for col in tb_deaths[name].columns
        if tb_deaths[name][col].dtype == "float64"
    ]
    tb_deaths[name]["median"] = (
        tb_deaths[name][cols].astype(float).quantile(0.5, axis=1)
    )
    tb_deaths[name]["lower"] = (
        tb_deaths[name][cols].astype(float).quantile(0.025, axis=1)
    )
    tb_deaths[name]["upper"] = (
        tb_deaths[name][cols].astype(float).quantile(0.975, axis=1)
    )

    # AIDS_TB mortality rates per 100k person-years
    tb_deaths[name]["TB_death_rate_100kpy"] = (
            tb_deaths[name]["median"].values / py_summary["mean"].values) * 100000

    tb_deaths[name]["TB_death_rate_100kpy_lower"] = (
        tb_deaths[name]["lower"].values / py_summary["mean"].values) * 100000

    tb_deaths[name]["TB_death_rate_100kpy_upper"] = (
        tb_deaths[name]["upper"].values / py_summary["mean"].values) * 100000


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


#
# # TB deaths (excluding HIV/TB deaths)
# make_plot(
#     title_str="TB mortality rate per 100,000 population",
#     model=tot_tb_non_hiv_deaths_rate_100kpy,
#     model_low=tot_tb_non_hiv_deaths_rate_100kpy_lower,
#     model_high=tot_tb_non_hiv_deaths_rate_100kpy_upper,
#     )
# plt.savefig(make_graph_file_name("TB_mortality"))
#
# plt.show()
#
# # ---------------------------------------------------------------------- #
#
# # TB treatment coverage
# make_plot(
#     title_str="TB treatment coverage",
#     model=model_tb_tx["mean"] * 100,
#     model_low=model_tb_tx["lower"] * 100,
#     model_high=model_tb_tx["upper"] * 100,
#     data_name="NTP",
#     data_mid=data_tb_ntp["treatment_coverage"],
# )
# # plt.savefig(make_graph_file_name("TB_treatment_coverage"))
#
# plt.show()
#
# # ---------------------------------------------------------------------- #
#
# # TB deaths
# # select cause of death
# tmp = results_deaths.loc[(results_deaths.cause == "TB")]
# # select draw - drop columns where draw != 0, but keep year and cause
# tmp2 = tmp.loc[
#        :, ("draw" == draw)
#        ].copy()  # selects only columns for draw=0 (removes year/cause)
# # join year and cause back to df - needed for groupby
# frames = [tmp["year"], tmp["cause"], tmp2]
# tmp3 = pd.concat(frames, axis=1)
#
# # create new column names, dependent on number of runs in draw
# base_columns = ["year", "cause"]
# run_columns = ["run" + str(x) for x in range(0, info["runs_per_draw"])]
# base_columns.extend(run_columns)
# tmp3.columns = base_columns
# tmp3 = tmp3.set_index("year")
#
# # sum rows for each year (2 entries)
# # for each run need to combine deaths in each year, may have different numbers of runs
# model_deaths_TB = pd.DataFrame(tmp3.groupby(["year"]).sum())
#
# # double check all columns are float64 or quantile argument will fail
# model_2010_median = model_deaths_TB.iloc[2].quantile(0.5)
# model_2015_median = model_deaths_TB.iloc[5].quantile(0.5)
# model_2010_low = model_deaths_TB.iloc[2].quantile(0.025)
# model_2015_low = model_deaths_TB.iloc[5].quantile(0.025)
# model_2010_high = model_deaths_TB.iloc[2].quantile(0.975)
# model_2015_high = model_deaths_TB.iloc[5].quantile(0.975)
#
# plt.bar(x_vals, y_vals, color=col)
# plt.errorbar(
#     x_vals, y_vals,
#     yerr=[y_lower, y_upper],
#     ls="none",
#     marker="o",
#     markeredgecolor="lightskyblue",
#     markerfacecolor="lightskyblue",
#     ecolor="lightskyblue",
# )
# plt.xticks(ticks=x_vals, labels=labels)
# plt.title("Deaths per year due to TB")
# plt.legend(handles=[blue_patch, green_patch, red_patch])
# plt.tight_layout()
# # plt.savefig(make_graph_file_name("TB_deaths_with_GBD"))
# plt.show()
