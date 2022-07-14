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


# py0 = extract_results(
#     results0,
#     module="tlo.methods.demography",
#     key="person_years",
#     custom_generate_series=get_person_years,
#     do_scaling=False
# )
#
# py1 = extract_results(
#     results1,
#     module="tlo.methods.demography",
#     key="person_years",
#     custom_generate_series=get_person_years,
#     do_scaling=False
# )
#
# py2 = extract_results(
#     results2,
#     module="tlo.methods.demography",
#     key="person_years",
#     custom_generate_series=get_person_years,
#     do_scaling=False
# )
#
# py3 = extract_results(
#     results3,
#     module="tlo.methods.demography",
#     key="person_years",
#     custom_generate_series=get_person_years,
#     do_scaling=False
# )
#
# py4 = extract_results(
#     results4,
#     module="tlo.methods.demography",
#     key="person_years",
#     custom_generate_series=get_person_years,
#     do_scaling=False
# )


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

# Make plot
fig, ax = plt.subplots()
ax.plot(inc0.index, inc0["median"], "-", color="C3")
ax.fill_between(inc0.index, inc0["lower"], inc0["upper"], color="C3", alpha=0.2)

ax.plot(inc1.index, inc1["median"], "-", color="C0")
ax.fill_between(inc1.index, inc1["lower"], inc1["upper"], color="C0", alpha=0.2)

ax.plot(inc2.index, inc2["median"], "-", color="C2")
ax.fill_between(inc2.index, inc2["lower"], inc2["upper"], color="C2", alpha=0.2)

ax.plot(inc3.index, inc3["median"], "-", color="C4")
ax.fill_between(inc3.index, inc3["lower"], inc3["upper"], color="C4", alpha=0.2)

ax.plot(inc4.index, inc4["median"], "-", color="C6")
ax.fill_between(inc4.index, inc4["lower"], inc4["upper"], color="C6", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("HIV incidence in adults 15-49")
plt.ylabel("HIV incidence")
plt.legend(["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

plt.show()


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

# Make plot
fig, ax = plt.subplots()
ax.plot(tb_inc0.index, tb_inc0["median"], "-", color="C3")
ax.fill_between(tb_inc0.index, tb_inc0["lower"], tb_inc0["upper"], color="C3", alpha=0.2)

ax.plot(tb_inc1.index, tb_inc1["median"], "-", color="C0")
ax.fill_between(tb_inc1.index, tb_inc1["lower"], tb_inc1["upper"], color="C0", alpha=0.2)

ax.plot(tb_inc2.index, tb_inc2["median"], "-", color="C2")
ax.fill_between(tb_inc2.index, tb_inc2["lower"], tb_inc2["upper"], color="C2", alpha=0.2)

ax.plot(tb_inc3.index, tb_inc3["median"], "-", color="C4")
ax.fill_between(tb_inc3.index, tb_inc3["lower"], tb_inc3["upper"], color="C4", alpha=0.2)

ax.plot(tb_inc4.index, tb_inc4["median"], "-", color="C6")
ax.fill_between(tb_inc4.index, tb_inc4["lower"], tb_inc4["upper"], color="C6", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("Active TB incidence")
plt.ylabel("Active TB cases per 100,000 population")
plt.legend(["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

plt.show()


# ---------------------------------- HIV deaths ---------------------------------- #






# ---------------------------------- TB deaths ---------------------------------- #






# ---------------------------------- TREATMENT COVERAGE ---------------------------------- #


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

tb_tx0 = tb_tx_coverage(results0)
tb_tx1 = tb_tx_coverage(results1)
tb_tx2 = tb_tx_coverage(results2)
tb_tx3 = tb_tx_coverage(results3)
tb_tx4 = tb_tx_coverage(results4)

# Make plot
fig, ax = plt.subplots()
ax.plot(tb_tx0.index, tb_tx0["median"], "-", color="C3")
ax.fill_between(tb_tx0.index, tb_tx0["lower"], tb_tx0["upper"], color="C3", alpha=0.2)

ax.plot(tb_tx1.index, tb_tx1["median"], "-", color="C0")
ax.fill_between(tb_tx1.index, tb_tx1["lower"], tb_tx1["upper"], color="C0", alpha=0.2)

ax.plot(tb_tx2.index, tb_tx2["median"], "-", color="C2")
ax.fill_between(tb_tx2.index, tb_tx2["lower"], tb_tx2["upper"], color="C2", alpha=0.2)

ax.plot(tb_tx3.index, tb_tx3["median"], "-", color="C4")
ax.fill_between(tb_tx3.index, tb_tx3["lower"], tb_tx3["upper"], color="C4", alpha=0.2)

ax.plot(tb_tx4.index, tb_tx4["median"], "-", color="C6")
ax.fill_between(tb_tx4.index, tb_tx4["lower"], tb_tx4["upper"], color="C6", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.ylim((0, 1.0))
plt.title("TB treatment coverage")
plt.ylabel("TB treatment coverage")
plt.legend(["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

plt.show()





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


tmp = extract_tx_delay(results_folder=results0,
                       module="tlo.methods.tb",
                       key="tb_treatment_delays",
                       column="tbTreatmentDelayAdults")









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

