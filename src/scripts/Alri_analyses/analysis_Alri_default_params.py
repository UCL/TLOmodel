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
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file
from tlo.methods import (
    alri,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
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
end_date = Date(2019, 12, 31)
popsize = 10000

log_config = {
    "filename": "alri_with_treatment",
    "directory": "./outputs",
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.alri": logging.INFO,
        "tlo.methods.demography": logging.INFO,
    }
}

# Establish the simulation object
sim = Simulation(start_date=start_date, log_config=log_config, show_progress_bar=True)

# run the simulation
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),

    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),

    alri.Alri(resourcefilepath=resourcefilepath, log_indivdual=22),  # choose to log an individual
    alri.PropertiesOfOtherModules()
)

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Read the output:
output = parse_log_file(sim.log_filepath)

# Save the output for a single individual to a csv (if needed)
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
    'RSV': 0.2749486927,
    'Rhinovirus': 0.021661326125,
    'HMPV': 0.038206021625,
    'Parainfluenza': 0.036674105375,
    'Strep_pneumoniae_PCV13': 0.007200006375,
    'Strep_pneumoniae_non_PCV13': 0.007200006375,
    'Hib': 0.009038305875,
    'H.influenzae_non_type_b': 0.009038305875,
    'Staph_aureus': 0.01133618025,
    'Enterobacteriaceae': 0.010263838875,
    'other_Strepto_Enterococci': 0.010263838875,
    'Influenza': 0.017678343875,
    'P.jirovecii': 0.0091914975,
    'Bocavirus': 0.012776211875,
    'Adenovirus': 0.012776211875,
    'other_viral_pathogens': 0.0001,
    'other_bacterial_pathogens': 0.042801770375,
}

calibration_incidence_rate_1_year_olds = {
    'RSV': 0.11019461832,
    'Rhinovirus': 0.03504417426,
    'HMPV': 0.01840570614,
    'Parainfluenza': 0.01915350246,
    'Strep_pneumoniae_PCV13': 0.00990830124,
    'Strep_pneumoniae_non_PCV13': 0.00990830124,
    'Hib': 0.00626279418,
    'H.influenzae_non_type_b': 0.00626279418,
    'Staph_aureus': 0.00168254172,
    'Enterobacteriaceae': 0.00364550706,
    'other_Strepto_Enterococci': 0.00364550706,
    'Influenza': 0.01204943742,
    'P.jirovecii': 0.00074779632,
    'Bocavirus': 0.0066279141,
    'Adenovirus': 0.0066279141,
    'other_viral_pathogens': 0.0001,
    'other_bacterial_pathogens': 0.05785196202,
}

calibration_incidence_rate_2_to_4_year_olds = {
    'RSV': 0.0087827586,
    'Rhinovirus': 0.00846338556,
    'HMPV': 0.0031405015,
    'Parainfluenza': 0.00361956112,
    'Strep_pneumoniae_PCV13': 0.00244852664,
    'Strep_pneumoniae_non_PCV13': 0.00244852664,
    'Hib': 0.00117103448,
    'H.influenzae_non_type_b': 0.00117103448,
    'Staph_aureus': 0.00063874608,
    'Enterobacteriaceae': 0.00157025078,
    'other_Strepto_Enterococci': 0.00157025078,
    'Influenza': 0.00170332288,
    'P.jirovecii': 0.00010645768,
    'Bocavirus': 0,
    'Adenovirus': 0,
    'other_viral_pathogens': 0.0001,
    'other_bacterial_pathogens': 0.01501053288
}

# Produce a set of line plot comparing to the calibration data
fig, axes = plt.subplots(ncols=4, nrows=5, sharey=True, sharex=True, figsize=(10, 20))
for ax_num, pathogen in enumerate(sim.modules['Alri'].all_pathogens):
    ax = fig.axes[ax_num]
    inc_rate['0y'][pathogen].plot(ax=ax, label='Model output')
    ax.hlines(y=calibration_incidence_rate_0_year_olds[pathogen],  # axhlines is to plot horizontal lines at each y
              xmin=min(inc_rate['0y'].index),
              xmax=max(inc_rate['0y'].index),
              label='calibrating_data'
              )
    ax.set_title(f'{pathogen}')
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidence Rate <1 year-olds")
plt.savefig(outputpath / ("ALRI_inc_rate_by_pathogen_and_time_0_year_olds" + datestamp + ".png"), format='png')
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
plt.ylabel('Risk of pathogen causing Alri per year')
plt.tight_layout()
plt.savefig(outputpath / ("ALRI_inc_rate_calibration_0_year_olds" + datestamp + ".png"), format='png')
plt.show()

# 1 year-olds
inc_mean.plot.bar(y=['1y_model_output', '1y_calibrating_data'])
plt.title('Incidence Rate: 1 year-olds')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing Alri per year')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing Alri per year')
plt.tight_layout()
plt.savefig(outputpath / ("ALRI_inc_rate_calibration_1_year_olds" + datestamp + ".png"), format='png')
plt.show()

# 2-4 year-olds
inc_mean.plot.bar(y=['2-4y_model_output', '2-4y_calibrating_data'])
plt.title('Incidence Rate: 2-4 year-olds')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing Alri per year')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing Alri per year')
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
