"""
This file uses the results of the batch file to get the incidence and mortality rates.
The results of the batchrun were put into the 'outputspath' results_folder
"""

import datetime
from pathlib import Path
import numpy as np

from tlo.analysis.utils import compare_number_of_deaths, parse_log_file

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

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract incident cases and death for all runs of each draw
# uses draw/run as column index by default
do_scaling = True  # if True, multiply by data-to-model scaling factor to correspond to the real population size

# incident count from model
alri_incident_count = extract_results(
        results_folder,
        module="tlo.methods.alri",
        key="event_counts",
        column='incident_cases',
        index="date",
        do_scaling=do_scaling)

# death count from model
alri_death_count = extract_results(
        results_folder,
        module="tlo.methods.alri",
        key="event_counts",
        column='deaths',
        index="date",
        do_scaling=do_scaling)

# set the index as year value (not date)
alri_incident_count.index = alri_incident_count.index.year
alri_death_count.index = alri_death_count.index.year

# store the output numbers of incident cases and death in each run of each draw
alri_incident_count.to_csv(outputspath / ("batch_run_incident_count" + ".csv"))
alri_death_count.to_csv(outputspath / ("batch_run_death_count" + ".csv"))


# --------------------------------------- PERSON-YEARS (DENOMINATOR) ---------------------------------------
# call this for each run and then take the mean to use as denominator for mortality / incidence etc.
def get_under5_person_years(py_):
    # get person-years of under 5 year-olds
    years = pd.to_datetime(py_['date']).dt.year
    py = pd.Series(dtype="int64", index=years)
    for year in years:
        tot_py = (
            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series) +
            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
        ).transpose()
        tot_py.index = tot_py.index.astype(int)
        py[year] = tot_py.loc[0:4].sum().values[0]

    return py


# person-years for under 5 years of age from model
person_years = extract_results(
    results_folder=results_folder,
    module='tlo.methods.demography',
    key='person_years',
    custom_generate_series=get_under5_person_years,
    do_scaling=do_scaling)


# ---------------------------------------------------------------------------------------------------
# # # # # # # # # # INCIDENCE & MORTALITY RATE (CASES/ DENOMINATOR) # # # # # # # # # #
# ---------- Incidence ----------
# calculate the incidence rate (cases per 100 child-years) in each run of each draw
incidence_per_run_per_draw = ((alri_incident_count / person_years) * 100).dropna()
# get mean / upper/ lower statistics
incidence_summary_per_draw = summarize(incidence_per_run_per_draw)

# store the incidence summary statistics of each draw in each run of each draw
incidence_summary_per_draw.to_csv(outputspath / ("batch_run_incidence_100cy_results" + ".csv"))

# ---------- Mortality ----------
# calculate the mortality rate (deaths per 100,000 under-5 population or child-years) in each run of each draw
mortality_per_run_per_draw = ((alri_death_count / person_years) * 100000).dropna()
# get mean / upper/ lower statistics
mortality_summary_per_draw = summarize(mortality_per_run_per_draw)

# store the mortality summary statistics of each draw in each run of each draw
mortality_summary_per_draw.to_csv(outputspath / ("batch_run_mortality_100000pop_results" + ".csv"))

# ----------------------------------- CREATE PLOTS - SINGLE RUN FIGURES -----------------------------------
# INCIDENCE & MORTALITY RATE - OUTPUT OVERTIME
start_date = 2010
end_date = 2031
draw = 0

# import GBD data for Malawi's ALRI burden estimates
GBD_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_Alri.xlsx",
    sheet_name="GBD_Malawi_estimates",
    )
# import McAllister estimates for Malawi's ALRI incidence
McAllister_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_Alri.xlsx",
    sheet_name="McAllister_2019",
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
plt.plot(incidence_summary_per_draw.index, incidence_summary_per_draw.loc[:, (draw, 'mean')].values,
         color='teal', label='Model')
plt.fill_between(
    incidence_summary_per_draw.index,
    incidence_summary_per_draw.loc[:, (draw, 'mean')].values,
    incidence_summary_per_draw.loc[:, (draw, 'mean')].values,
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
plt.plot(mortality_summary_per_draw.index, mortality_summary_per_draw.loc[:, (draw, 'mean')].values,
         color='teal', label='Model')  # model
plt.fill_between(
    mortality_summary_per_draw.index,
    mortality_summary_per_draw.loc[:, (draw, 'lower')].values,
    mortality_summary_per_draw.loc[:, (draw, 'upper')].values,
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

plt.plot(counts.index, mort_per_livebirth, color="mediumseagreen")  # model
plt.title("ALRI Mortality per 1,000 livebirths")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Mortality (/100k)")
plt.gca().set_xlim(start_date, end_date)
plt.legend(["McAllister 2019", "Model"])
plt.tight_layout()
# plt.savefig(outputpath / ("ALRI_Mortality_model_comparison" + datestamp + ".png"), format='png')

plt.show()
