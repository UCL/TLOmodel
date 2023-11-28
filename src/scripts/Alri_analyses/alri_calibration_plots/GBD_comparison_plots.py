"""
This is the analysis script for the calibration of the ALRI model
"""
# %% Import Statements and initial declarations
import datetime
import os
import random
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

log_filename = 'none'
# log_filename = outputpath / 'GBD_lri_comparison_15k_pop__2023-09-11T163954.log'
# <-- insert name of log file to avoid re-running the simulation

if not os.path.exists(log_filename):
    # If logfile does not exists, re-run the simulation:
    # Do not run this cell if you already have a logfile from a simulation:

    start_date = Date(2010, 1, 1)
    end_date = Date(2025, 12, 31)
    popsize = 15000

    log_config = {
        "filename": "GBD_lri_comparison_15k_pop",
        "directory": "./outputs",
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.alri": logging.DEBUG,
            "tlo.methods.demography": logging.INFO,
            # "tlo.methods.healthburden": logging.INFO,
        }
    }

    seed = random.randint(0, 50000)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)

    # run the simulation
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                      force_any_symptom_to_lead_to_healthcareseeking=True),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  service_availability=['*'], cons_availability='default', disable=True),
        alri.Alri(resourcefilepath=resourcefilepath),
        alri.AlriPropertiesOfOtherModules()
    )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # display filename
    log_filename = sim.log_filepath
    print(f"log_filename: {log_filename}")  # outputs\GBD_lri_comparison_15k_pop__2023-09-11T163954.log

output = parse_log_file(log_filename)

# ------------------------------------------------------------------
# Calculate the "incidence rate" and "mortality rate" from the output of event counts
counts = output['tlo.methods.alri']['event_counts']
counts['year'] = pd.to_datetime(counts['date']).dt.year
counts.drop(columns='date', inplace=True)
counts.set_index(
    'year',
    drop=True,
    inplace=True
)

# get person-years of < 5 year-olds
py_ = output['tlo.methods.demography']['person_years']
years = pd.to_datetime(py_['date']).dt.year
py = pd.DataFrame(index=years, columns=['<5y'])
for year in years:
    tot_py = (
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
    ).transpose()
    tot_py.index = tot_py.index.astype(int)
    py.loc[year, '<5y'] = tot_py.loc[0:4].sum().values[0]


# get all live births
births = output['tlo.methods.demography']['on_birth']
births['year'] = pd.to_datetime(births['date']).dt.year
births.drop(columns='date', inplace=True)
births.set_index(
    'year',
    drop=True,
    inplace=True
)
births_per_year = births.groupby('year').size()

# # get population size to make a comparison
# pop = output['tlo.methods.demography']['num_children']
# pop['year'] = pd.to_datetime(pop['date']).dt.year
# pop.drop(columns='date', inplace=True)
# pop.set_index(
#     'year',
#     drop=True,
#     inplace=True
# )
# pop.columns = pop.columns.astype(int)
# pop['<5y'] = pop[0] + pop[1] + pop[1] + pop[3] + pop[4]
# pop.drop(columns=[x for x in range(5)], inplace=True)

# Incidence rate outputted from the ALRI model - using the tracker to get the number of cases per year
inc_rate = (counts.incident_cases.div(py['<5y'], axis=0).dropna()) * 100

# Mortality rate outputted from the ALRI model - using the tracker to get the number of deaths per year
mort_rate = (counts.deaths.div(py['<5y'], axis=0).dropna()) * 100000

# ----------------------------------- CREATE PLOTS - SINGLE RUN FIGURES -----------------------------------
# INCIDENCE & MORTALITY RATE - OUTPUT OVERTIME
start_date = 2010
end_date = 2026

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
plt.figure(1, figsize=(10, 10))

# INCIDENCE RATE
# # # # # ALRI incidence per 100 child-years # # # # #
fig, ax = plt.subplots()

# GBD estimates
plt.plot(GBD_data.Year, GBD_data.Incidence_per100_children)
plt.fill_between(
    GBD_data.Year,
    GBD_data.Incidence_per100_lower,
    GBD_data.Incidence_per100_upper,
    alpha=0.5,
)
# McAllister et al 2019 estimates
years_with_data = McAllister_data.dropna(axis=0)
plt.plot(years_with_data.Year, years_with_data.Incidence_per100_children)
plt.fill_between(
    years_with_data.Year,
    years_with_data.Incidence_per100_lower,
    years_with_data.Incidence_per100_upper,
    alpha=0.5,
)
# model output
plt.plot(counts.index, inc_rate, color="mediumseagreen")
plt.title("ALRI incidence per 100 child-years")
plt.xlabel("Year")
plt.ylabel("Incidence (/100cy)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["GBD", "McAllister 2019", "Model"])
plt.tight_layout()
# plt.savefig(outputpath / ("ALRI_Incidence_model_comparison" + datestamp + ".png"), format='png')

plt.show()

# MORTALITY RATE
# # # # # ALRI mortality per 100,000 children # # # # #

fig1, ax1 = plt.subplots()

# GBD estimates
plt.plot(GBD_data.Year, GBD_data.Death_per100k_children)  # GBD data
plt.fill_between(
    GBD_data.Year,
    GBD_data.Death_per100k_lower,
    GBD_data.Death_per100k_upper,
    alpha=0.5,
)

# # McAllister et al 2019 estimates
# plt.plot(McAllister_data.Year, McAllister_data.Death_per1000_livebirths * 100)  # no upper/lower

# model output
plt.plot(counts.index, mort_rate, color="mediumseagreen")  # model
plt.title("ALRI Mortality per 100,000 children")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Mortality (/100k)")
plt.gca().set_xlim(start_date, end_date)
plt.legend(["GBD", "Model"])
plt.tight_layout()
# plt.savefig(outputpath / ("ALRI_Mortality_model_comparison" + datestamp + ".png"), format='png')

plt.show()

# # # # # #
# Mortality per 1,000 livebirths due to ALRI

fig2, ax2 = plt.subplots()

# McAllister et al. 2019 estimates
plt.plot(McAllister_data.Year, McAllister_data.Death_per1000_livebirths)  # no upper/lower

# model output
mort_per_livebirth = (counts.deaths / births_per_year * 1000).dropna()

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

# -------------------------------------------------------------------------------------------------------------
# still on mortality, use the compare_number_of_deaths function # # # # # #
# Get comparison function from utils.py
comparison = compare_number_of_deaths(logfile=log_filename, resourcefilepath=resourcefilepath)

# get only the estimates for Lower respiratory infections for 0-4 yo
lri_comparison = comparison.loc[(['2010-2014', '2015-2019'], slice(None), '0-4', 'Lower respiratory infections')]
lri_join_gender = lri_comparison.groupby('period').sum()

# Make a simple bar chart
plt.style.use("default")
plt.figure(1, figsize=(10, 10))
fig3, ax3 = plt.subplots()

lri_join_gender['model'].plot.bar(color='#ADD8E6', label='Model')
ax3.errorbar(x=lri_join_gender['model'].index, y=lri_join_gender.GBD_mean,
             yerr=[lri_join_gender.GBD_lower, lri_join_gender.GBD_upper],
             fmt='o', color='#23395d', label="GBD")
plt.title('Mean annual deaths due to ALRI, 2010-2019')
plt.xlabel("Time period")
plt.ylabel("Number of deaths")
plt.xticks(rotation=0)
plt.legend(loc=2)
plt.tight_layout()
# plt.savefig(outputpath / ("ALRI_death_calibration_plot" + datestamp + ".png"), format='png')
plt.show()

# -------------------------------------------------------------------------------------------------------------
# # # # # # # # # # ALRI DALYs # # # # # # # # # #
# ------------------------------------------------------------------
# # Get the total DALYs from the output of health burden
# dalys = output['tlo.methods.healthburden']['dalys']
# dalys.drop(columns='Other', inplace=True)
# dalys.drop(columns='date', inplace=True)
# dalys.drop(dalys.loc[dalys['age_range'] != '0-4'].index, inplace=True)
# dalys.set_index(
#     'year',
#     drop=True,
#     inplace=True
# )
# sf = output['tlo.methods.population']['scaling_factor']['scaling_factor'].values[0]
# dalys = dalys.groupby('year').sum() * sf
#
# plt.style.use("ggplot")
# plt.figure(1, figsize=(10, 10))
# fig4, ax4 = plt.subplots()
#
# # GBD estimates
# plt.plot(GBD_data.Year, GBD_data.DALYs)  # GBD data
# plt.fill_between(
#     GBD_data.Year,
#     GBD_data.DALYs_lower,
#     GBD_data.DALYs_upper,
#     alpha=0.5,
# )
# # model output
# plt.plot(dalys, color="mediumseagreen")  # model
# plt.title("ALRI DALYs")
# plt.xlabel("Year")
# plt.xticks(rotation=90)
# plt.ylabel("DALYs")
# plt.gca().set_xlim(start_date, end_date)
# plt.legend(["GBD", "Model"])
# plt.tight_layout()
# # plt.savefig(outputpath / ("ALRI_DALYs_model_comparison" + datestamp + ".png"), format='png')
#
# plt.show()

# -----------------------------------------------------------------------------------------------
# check the case fatality rates (CFR)

# using the tracker to get the number of cases per year
number_of_cases = counts.incident_cases

# using the tracker to get the number of deaths per year
number_of_deaths = counts.deaths

# calculate CFR
CFR_in_percentage = (number_of_deaths / number_of_cases) * 100

fig5, ax5 = plt.subplots()

# model output
plt.plot(CFR_in_percentage, color="mediumseagreen")  # model
plt.title("ALRI CFR")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("CRF (%)")
plt.legend(["Model"])
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------------------
# check the care-seeking proportion

# using the tracker to get the number of cases per year
number_of_cases = counts.incident_cases

# using the tracker to get the number of deaths per year
number_of_sought_care = counts.seeking_care

# calculate the proportion of those who sought care
sought_care_proportion = (number_of_sought_care / number_of_cases)

fig6, ax6 = plt.subplots()

# DHS care-seeking estimates
dhs_years = [2010, 2015]
dhs_care_seeking = [0.7, 0.78]
plt.plot(dhs_years, dhs_care_seeking)  # GBD data

# model output
plt.plot(sought_care_proportion, color="mediumseagreen")  # model
plt.title("ALRI care-seeking")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("sought care (%)")
plt.gca().set_ylim(0.0, 1.0)
plt.legend(["DHS", "Model"])
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------------------------
# check the complicated cases proportion

# using the tracker to get the number of cases per year
number_of_cases = counts.incident_cases

# using the tracker to get the number of hypoxaemic cases per year
number_of_hypoxaemic_cases = counts.hypoxaemic_cases
number_of_pulmonary_complications = counts.pulmonary_complication_cases
number_of_systemic_complications = counts.systemic_complication_cases

# calculate the proportion of those who are hypoxaemia SpO2<93%
hypoxaemic_cases_proportion = (number_of_hypoxaemic_cases / number_of_cases)
pulmonary_complications_proportion = (number_of_pulmonary_complications / number_of_cases)
systemic_complications_proportion = (number_of_systemic_complications / number_of_cases)

fig7, ax7 = plt.subplots()

# DHS care-seeking estimates
years = [2010, 2020]
target_data = [0.31, 0.31]
plt.plot(years, target_data)  # Rahman data

# model output
plt.plot(hypoxaemic_cases_proportion, color="mediumseagreen")  # model
plt.plot(pulmonary_complications_proportion)  # model
plt.plot(systemic_complications_proportion)  # model
plt.title("% Complications - Target data vs model output")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Proportion of hypoxaemic cases")
plt.gca().set_ylim(0.0, 1.0)
plt.legend(["Target hypoxaemia", "Model hypoxaemia", "Model pulmonary complications", "Model sepsis"])
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------------
# counts = output['tlo.methods.alri']['classification']
# counts['year'] = pd.to_datetime(counts['date']).dt.year
# counts.drop(columns='date', inplace=True)
# counts.set_index(
#     'year',
#     drop=True,
#     inplace=True
# )
#
# print(counts)
