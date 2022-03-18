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
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/sejjil0@ucl.ac.uk")

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("baseline_alri_scenario.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)
# tlo.methods.deviance_measure[deviance_measure]

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract incident cases for all runs, and get summary of the results for that log-element
# uses draw/run as index by default
alri_incident_cases = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.alri",
        key="event_counts",
        column='incident_cases',
        index="date",
        do_scaling=False),
    collapse_columns=True)

alri_incident_cases.index = alri_incident_cases.index.year

alri_incident_cases.to_csv(outputspath / ("batch_run_results" + ".csv"))

mean_per_year = alri_incident_cases.groupby(axis=1, by=alri_incident_cases.columns.get_level_values('stat')).mean()

# plot the deviance against parameters
fig, ax = plt.subplots()

plt.plot(mean_per_year.index, mean_per_year['mean'])
plt.fill_between(
    mean_per_year.index,
    mean_per_year['lower'],
    mean_per_year['upper'],
    alpha=0.5,
)

plt.title("ALRI incident cases")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Number of cases")
# plt.gca().set_xlim(start_date, end_date)
plt.legend(["Model"])
plt.tight_layout()

plt.show()

# get the resource file
xls_alri = pd.ExcelFile(resourcefilepath / "ResourceFile_ALRI.xlsx")

# ALRI GBD data
data_alri_gbd = pd.read_excel(xls_alri, sheet_name="GBD_Malawi_estimates")
data_alri_gbd = data_alri_gbd.loc[
    (data_alri_gbd.Year >= 2010)
]  # include only years post-2010
data_alri_gbd.index = data_alri_gbd["Year"]
data_alri_gbd = data_alri_gbd.drop(columns=["Year"])

# ALRI McAllister et al 2019 (incidence)
data_alri_mcallister = pd.read_excel(xls_alri, sheet_name="McAllister_2019")
# data_alri_mcallister = data_alri_mcallister.loc[
#     (data_alri_mcallister.year >= 2010)]  # include only years post-2010
data_alri_mcallister.index = data_alri_mcallister["Year"]
# data_alri_mcallister = data_alri_mcallister.drop(columns=["Year"])

# %%%%% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("baseline_alri_scenario.py", outputspath)[-1]

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

# %%%%% extract results
# Load and format model results (with year as integer):

model_alri_incidence = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.alri",
        key="event_counts",
        column="incident_cases",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_alri_incidence.index = model_alri_incidence.index.year


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

    return


# for draw 0, get py for all runs
number_runs = info["runs_per_draw"]
py_summary = pd.DataFrame(data=None, columns=range(0, number_runs))

# draw number (default = 0) is specified above
for run in range(0, number_runs):
    py_summary.iloc[:, run] = get_person_years(draw, run)

py_summary["mean"] = py_summary.mean(axis=1)


# Incidence rate outputted from the ALRI model - using the tracker to get the number of cases per year
inc_rate = (model_alri_incidence.div(py_summary["mean"].values, axis=0).dropna()) * 100
