"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
from tlo.util import read_csv_files

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/sejjil0@ucl.ac.uk")

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("baseline_alri_scenario.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract incident cases and death for all runs, and get summary of the results for that log-element
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

# death count from model
alri_death_count = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.alri",
        key="event_counts",
        column='deaths',
        index="date",
        do_scaling=False),
    collapse_columns=True)

# set the index as year value (not date)
alri_incident_cases.index = alri_incident_cases.index.year
alri_death_count.index = alri_death_count.index.year

# store the mean of mean/lower/upper of the runs for each draw
alri_incident_cases.to_csv(outputspath / ("batch_run_incident_results" + ".csv"))
alri_death_count.to_csv(outputspath / ("batch_run_death_results" + ".csv"))

# get the mean/lowe/upper values per year for all draws
mean_cases_per_year = \
    alri_incident_cases.groupby(axis=1, by=alri_incident_cases.columns.get_level_values('stat')).mean()

mean_deaths_per_year = \
    alri_death_count.groupby(axis=1, by=alri_death_count.columns.get_level_values('stat')).mean()


# --------------------------------------- PERSON-YEARS (DENOMINATOR) ---------------------------------------
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
        tot_py.index = tot_py.index.astype(int)
        py[year] = tot_py.loc[0:4].sum().values[0]

    return py


# for each draw, get py for all runs
number_draws = info['number_of_draws']
number_runs = info["runs_per_draw"]

# create empty dataframe for collecting the results from get_person_years(draw, run) function
py_summary_per_run = pd.DataFrame(data=None, columns=range(0, number_runs))

all_draws_py_summary = pd.DataFrame(
    columns=pd.MultiIndex.from_product(
        [list(range(0, number_draws)), ['mean', 'lower', 'upper']],
        names=['draw', 'stat']), index=pd.to_datetime(log["tlo.methods.demography"]["person_years"]["date"]).dt.year
)

for draw in range(0, number_draws):
    for run in range(0, number_runs):
        py_summary_per_run.iloc[:, run] = get_person_years(draw, run)

    # get the mean, lower, upper values of the runs for the draw
    py_summary_mean_df = pd.DataFrame(data=None, columns=['mean', 'lower', 'upper'], index=py_summary_per_run.index)
    py_summary_mean_df.loc[:, 'mean'] = py_summary_per_run.mean(axis=1).values
    py_summary_mean_df.loc[:, 'lower'] = py_summary_per_run.quantile(0.025, axis=1).values
    py_summary_mean_df.loc[:, 'upper'] = py_summary_per_run.quantile(0.975, axis=1).values
    # add an index-column [draw] to py_summary_mean_df
    py_summary_mean_df.columns = pd.MultiIndex.from_product([[draw], list(py_summary_mean_df.columns)],
                                                            names=['draw', 'stat'])

    # update the index of empty dataframe all_draws_py_summary
    # year_index = py_summary_mean_df.index
    # all_draws_py_summary.reindex(year_index)
    all_draws_py_summary.loc[:, [draw]] = py_summary_mean_df.loc[:, [draw]].values

# get the mean of mean/ lower/ upper values per year across all draws
mean_py_per_year = all_draws_py_summary.groupby(axis=1, by=all_draws_py_summary.columns.get_level_values('stat')).mean()

# ---------------------------------------------------------------------------------------------------
# # # # # INCIDENCE & MORTALITY RATE (CASES/ DENOMINATOR) # # # # #

# calculate the incidence and mortality mean and upper/lower mean values
# mean incidence & mortality
alri_incidence_mean = pd.Series(
    (mean_cases_per_year['mean'] / mean_py_per_year['mean']) * 100).dropna()
alri_death_mean = pd.Series(
    (mean_deaths_per_year['mean'] / mean_py_per_year['mean']) * 100000).dropna()

# lower limit
alri_incidence_lower = pd.Series(
    (mean_cases_per_year['lower'] / mean_py_per_year['lower']) * 100).dropna()
alri_death_lower = pd.Series(
    (mean_deaths_per_year['lower'] / mean_py_per_year['lower']) * 100000).dropna()

# upper limit
alri_incidence_upper = pd.Series(
    (mean_cases_per_year['upper'] / mean_py_per_year['upper']) * 100).dropna()
alri_death_upper = pd.Series(
    (mean_deaths_per_year['upper'] / mean_py_per_year['upper']) * 100000).dropna()

# ----------------------------------- CREATE PLOTS - SINGLE RUN FIGURES -----------------------------------
# INCIDENCE & MORTALITY RATE - OUTPUT OVERTIME
start_date = 2010
end_date = 2031

# import GBD data for Malawi's ALRI burden estimates
GBD_data = read_csv_files(
    Path(resourcefilepath) / "ResourceFile_Alri",
    files="GBD_Malawi_estimates",
    )
# import McAllister estimates for Malawi's ALRI incidence
McAllister_data = read_csv_files(
    Path(resourcefilepath) / "ResourceFile_Alri",
    files="McAllister_2019",
    )

plt.style.use("ggplot")
# ------------------------------------

# INCIDENCE RATE
# # # # # ALRI incidence per 100 child-years # # # # #
fig = plt.figure()

# GBD estimates
plt.plot(GBD_data.Year, GBD_data.Incidence_per100_children, label='GBD')
plt.fill_between(
    GBD_data.Year,
    GBD_data.Incidence_per100_lower,
    GBD_data.Incidence_per100_upper,
    alpha=0.5,
)
# McAllister et al 2019 estimates
years_with_data = McAllister_data.dropna(axis=0)
plt.plot(years_with_data.Year, years_with_data.Incidence_per100_children, label='McAllister')
plt.fill_between(
    years_with_data.Year,
    years_with_data.Incidence_per100_lower,
    years_with_data.Incidence_per100_upper,
    alpha=0.5,
)
# model output
plt.plot(alri_incidence_mean.index, alri_incidence_mean, color='teal', label='Model')
plt.fill_between(
    alri_incidence_mean.index,
    alri_incidence_lower,
    alri_incidence_upper,
    color='teal',
    alpha=0.5,
)

plt.title("ALRI incidence per 100 child-years")
plt.xlabel("Year")
plt.ylabel("Incidence (/100cy)")
plt.xticks(ticks=np.arange(start_date, end_date), rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend()
plt.tight_layout()

plt.show()

# --------------------------------------------------

# MORTALITY RATE
# # # # # ALRI mortality per 100,000 children # # # # #
fig1 = plt.figure()

# GBD estimates
plt.plot(GBD_data.Year, GBD_data.Death_per100k_children, label='GBD')  # GBD data
plt.fill_between(
    GBD_data.Year,
    GBD_data.Death_per100k_lower,
    GBD_data.Death_per100k_upper,
    alpha=0.5,
)
# model output
plt.plot(alri_death_mean.index, alri_death_mean, color='teal', label='Model')  # model
plt.fill_between(
    alri_death_mean.index,
    alri_death_lower,
    alri_death_upper,
    color='teal',
    alpha=0.5
)
plt.title("ALRI Mortality per 100,000 children")
plt.xlabel("Year")
plt.xticks(ticks=np.arange(start_date, end_date), rotation=90)
plt.ylabel("Mortality (/100k)")
plt.gca().set_xlim(start_date, end_date)
plt.legend()
plt.tight_layout()

plt.show()

# -------------------------------------------------------------------------------------------------------------
# # # # # #
# Mortality per 1,000 livebirths due to ALRI

# get birth counts
birth_count = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="on_birth",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
    ),
    do_scaling=False
)

# mean_births_per_year = \
#     birth_count.groupby(axis=1, by=birth_count.columns.get_level_values('stat')).mean()

birth_count.to_csv(outputspath / ("batch_run_birth_results" + ".csv"))

fig2, ax2 = plt.subplots()

# McAllister et al. 2019 estimates
plt.plot(McAllister_data.Year, McAllister_data.Death_per1000_livebirths)  # no upper/lower

# model output
mort_per_livebirth = (alri_death_count / birth_count * 1000).dropna()

plt.plot(mort_per_livebirth.index, mort_per_livebirth, color="mediumseagreen")  # model
plt.title("ALRI Mortality per 1,000 livebirths")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Mortality (/100k)")
plt.gca().set_xlim(start_date, end_date)
plt.legend(["McAllister 2019", "Model"])
plt.tight_layout()
# plt.savefig(outputpath / ("ALRI_Mortality_model_comparison" + datestamp + ".png"), format='png')

plt.show()
