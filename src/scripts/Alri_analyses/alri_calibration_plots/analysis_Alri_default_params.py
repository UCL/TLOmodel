"""
This will run the Alri Module and plot the incidence rate of each pathogen by each age group.
This will then be compared with:
    * The input incidence rate for each pathogen
    * The desired incidence rate for each pathogen
There is treatment.
"""
# %% Import Statements and initial declarations
import datetime
from pathlib import Path

import pandas as pd
import random
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

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=1)
popsize = 10_000

log_config = {
    "filename": "alri_with_treatment",
    "directory": "./outputs",
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.alri": logging.INFO,
        "tlo.methods.demography": logging.INFO,
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
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
    alri.Alri(resourcefilepath=resourcefilepath),
    alri.AlriPropertiesOfOtherModules()
)

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Read the output:
output = parse_log_file(sim.log_filepath)

# # Save the output for a single individual to a csv (if needed)
# one_person = output['tlo.methods.alri']['log_individual'].to_csv(r'./outputs/one_person2.csv', index=False)


# %% ----------------------------  INCIDENCE RATE OF Alri BY PATHOGEN  ----------------------------
# Calculate the "incidence rate" from the output counts of incidence
counts = output['tlo.methods.alri']['incidence_count_by_age_and_pathogen']
counts['year'] = pd.to_datetime(counts['date']).dt.year
counts.drop(columns='date', inplace=True)
counts.set_index(
    'year',
    drop=True,
    inplace=True
)
counts = counts.drop(columns='5+').rename(columns={'0': '0y', '1': '1y', '2-4': '2-4y'})  # for consistency

# get person-years of 0 year-old, 1 year-olds and 2-4 year-old
py_ = output['tlo.methods.demography']['person_years']
years = pd.to_datetime(py_['date']).dt.year
py = pd.DataFrame(index=years, columns=['0y', '1y', '2-4y'])
for year in years:
    tot_py = (
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
    ).transpose()
    tot_py.index = tot_py.index.astype(int)
    py.loc[year, '0y'] = tot_py.loc[0].values[0]
    py.loc[year, '1y'] = tot_py.loc[1].values[0]
    py.loc[year, '2-4y'] = tot_py.loc[2:4].sum().values[0]

# # get population size to make a comparison
pop = output['tlo.methods.demography']['num_children']
pop.set_index(
    'date',
    drop=True,
    inplace=True
)
pop.columns = pop.columns.astype(int)
pop['0y'] = pop[0]
pop['1y'] = pop[1]
pop['2-4y'] = pop[2] + pop[3] + pop[4]
pop.drop(columns=[x for x in range(5)], inplace=True)

# Incidence rate among 0, 1, 2-4 year-olds
inc_rate = dict()
for age_grp in ['0y', '1y', '2-4y']:
    inc_rate[age_grp] = counts[age_grp].apply(pd.Series).div(py[age_grp], axis=0).dropna()

# Load the incidence rate data to which we calibrate
calibration_incidence_rate_0_year_olds = {
    'RSV': 0.20894235414769,
    'Rhinovirus': 0.0274244465366988,
    'HMPV': 0.0406998098142096,
    'Parainfluenza': 0.0485328018094284,
    'Strep_pneumoniae_PCV13': 0.0103565835007875,
    'Strep_pneumoniae_non_PCV13': 0.0128974025600678,
    'Hib': 0.00621106728372586,
    'H.influenzae_non_type_b': 0.022609258909881,
    'Staph_aureus': 0.0270478905463932,
    'Enterobacteriaceae': 0.00752599421733598,
    'other_Strepto_Enterococci': 0.010697841813081,
    'Influenza': 0.018049324485616,
    'P.jirovecii': 0.00763324058146232,
    'other_viral_pathogens': 0.0330816176887697,
    'other_bacterial_pathogens': 0.0401399384246373,
    'other_pathogens_NoS': 0.0142557529486722,
}

calibration_incidence_rate_1_year_olds = {
    'RSV': 0.0479711003028054,
    'Rhinovirus': 0.0649126035406354,
    'HMPV': 0.0153809183965049,
    'Parainfluenza': 0.0172975975307687,
    'Strep_pneumoniae_PCV13': 0.01458202824,
    'Strep_pneumoniae_non_PCV13': 0.00721850608427674,
    'Hib': 0.00308118142035609,
    'H.influenzae_non_type_b': 0.011194369738496,
    'Staph_aureus': 0.00237032063281197,
    'Enterobacteriaceae': 0.00952216931330798,
    'other_Strepto_Enterococci': 0.00074779632,
    'Influenza': 0.00743425365659338,
    'P.jirovecii': 0.00578687175110399,
    'other_viral_pathogens': 0.0391308640365976,
    'other_bacterial_pathogens': 0.0343208917584349,
    'other_pathogens_NoS': 0.0273278079099261,
}

calibration_incidence_rate_2_to_4_year_olds = {
    'RSV': 0.0161586219207671,
    'Rhinovirus': 0.0237558962721129,
    'HMPV': 0.00495994272302193,
    'Parainfluenza': 0.00562173145001561,
    'Strep_pneumoniae_PCV13': 0.00415184952,
    'Strep_pneumoniae_non_PCV13': 0.00245410175998914,
    'Hib': 0.000999599939671829,
    'H.influenzae_non_type_b': 0.00351153299381416,
    'Staph_aureus': 0.000748005760516924,
    'Enterobacteriaceae': 0.00333715991289632,
    'other_Strepto_Enterococci': 0.00021291536,
    'Influenza': 0.00251973468285803,
    'P.jirovecii': 0.00183746840601445,
    'other_viral_pathogens': 0.014316113233013,
    'other_bacterial_pathogens': 0.0118895426334933,
    'other_pathogens_NoS': 0.0102148807957972,
}

# Produce a set of line plot comparing to the calibration data
fig, axes = plt.subplots(ncols=4, nrows=4, sharey=True, sharex=True, figsize=(20, 20))
for ax_num, pathogen in enumerate(sim.modules['Alri'].all_pathogens):
    ax = fig.axes[ax_num]
    inc_rate['0y'][pathogen].plot(ax=ax, label='Model output')
    ax.hlines(y=calibration_incidence_rate_0_year_olds[pathogen],  # axhlines is to plot horizontal lines at each y
              xmin=min(inc_rate['0y'].index),
              xmax=max(inc_rate['0y'].index),
              label='calibrating_data',
              color='r'
              )
    ax.set_title(f'{pathogen}')
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidence Rate <1 year-olds")
    ax.legend()
plt.savefig(outputpath / ("ALRI_inc_rate_by_pathogen_and_time_0_year_olds" + datestamp + ".png"), format='png')
plt.tight_layout()
plt.show()

# Produce a bar plot for means of incidence rate during the simulation:
inc_mean = pd.DataFrame()
inc_mean['0y_model_output'] = inc_rate['0y'].mean()
inc_mean['1y_model_output'] = inc_rate['1y'].mean()
inc_mean['2-4y_model_output'] = inc_rate['2-4y'].mean()

# put in the inputs:
inc_mean['0y_calibrating_data'] = pd.Series(data=calibration_incidence_rate_0_year_olds)
inc_mean['1y_calibrating_data'] = pd.Series(data=calibration_incidence_rate_1_year_olds)
inc_mean['2-4y_calibrating_data'] = pd.Series(data=calibration_incidence_rate_2_to_4_year_olds)

# 0 year-olds
inc_mean.plot.bar(y=['0y_model_output', '0y_calibrating_data'])
plt.title('Incidence Rate: 0 year-olds')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing\n Alri per year')
plt.tight_layout()
plt.savefig(outputpath / ("ALRI_inc_rate_calibration_0_year_olds" + datestamp + ".png"), format='png')
plt.show()

# 1 year-olds
inc_mean.plot.bar(y=['1y_model_output', '1y_calibrating_data'])
plt.title('Incidence Rate: 1 year-olds')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing Alri per year')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing\n Alri per year')
plt.tight_layout()
plt.savefig(outputpath / ("ALRI_inc_rate_calibration_1_year_olds" + datestamp + ".png"), format='png')
plt.show()

# 2-4 year-olds
inc_mean.plot.bar(y=['2-4y_model_output', '2-4y_calibrating_data'])
plt.title('Incidence Rate: 2-4 year-olds')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing\n Alri per year')
plt.tight_layout()
plt.savefig(outputpath / ("ALRI_inc_rate_calibration_2-4_year_olds" + datestamp + ".png"), format='png')
plt.show()

# %% ----------------------------  MEAN DEATH RATE BY PATHOGEN  ----------------------------

# Get comparison
comparison = compare_number_of_deaths(logfile=sim.log_filepath, resourcefilepath=resourcefilepath)

# Make a simple bar chart
comparison.loc[('2010-2014', slice(None), '0-4', 'Lower respiratory infections')].sum().plot.bar()
plt.title('Deaths per year due to ALRI, 2010-2014')
plt.tight_layout()
plt.savefig(outputpath / ("ALRI_death_calibration_plot" + datestamp + ".png"), format='png')
plt.show()
