"""
This will run the ALRI Module and plot the rate of death for pneumonia overall and compare with data.
There is treatment.
"""

# %% Import Statements and initial declarations
import datetime
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    pneumonia,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
)

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 20000

log_config = {
    'filename': 'LogFile',
    'custom_levels': {
        '*': logging.WARNING,
        'tlo.methods.demography': logging.INFO,
        'tlo.methods.pneumonia': logging.INFO
    }
}

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             pneumonia.ALRI(resourcefilepath=resourcefilepath),
             dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
             )

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Get the output from the logfile
output = parse_log_file(sim.log_filepath)


# %% ----------------------------  INCIDENCE RATE OF DIARRHOEA BY PATHOGEN  ----------------------------

#  Calculate the "incidence rate" from the output counts of incidence
counts = output['tlo.methods.pneumonia']['incidence_count_by_pathogen']
counts['year'] = pd.to_datetime(counts['date']).dt.year
counts.drop(columns='date', inplace=True)
counts.set_index(
    'year',
    drop=True,
    inplace=True
)

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
    # py.loc[year, '<5y'] = tot_py.loc[0:4].sum().values[0]

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
# pop['<5y'] = pop[0] + pop[1] + pop[2] + pop[3] + pop[4]
pop.drop(columns=[x for x in range(5)], inplace=True)

# Incidence rate among 0, 1, 2-4 year-olds
inc_rate = dict()
for age_grp in ['0y', '1y', '2-4y']:
    inc_rate[age_grp] = counts[age_grp].apply(pd.Series).div(py[age_grp], axis=0).dropna()

# Load the incidence rate data to which we calibrate
calibration_incidence_rate_0_year_olds = {
    'RSV': 9.7007598 / 100.0,
    'Rhinovirus': 9.7007598 / 100.0,
    'HMPV': 9.7007598 / 100.0,
    'Parainfluenza': 9.7007598 / 100.0,
    'Strep_pneumoniae_PCV13': 9.7007598 / 100.0,
    'Strep_pneumoniae_non_PCV13': 9.7007598 / 100.0,
    'Hib': 9.7007598 / 100.0,
    'H.influenzae_non_type_b': 9.7007598 / 100.0,
    'Staph_aureus': 9.7007598 / 100.0,
    'Enterobacteriaceae': 9.7007598 / 100.0,
    'other_Strepto_Enterococci': 9.7007598 / 100.0,
    'Influenza': 9.7007598 / 100.0,
    'P.jirovecii': 9.7007598 / 100.0,
    'Bocavirus': 9.7007598 / 100.0,
    'Adenovirus': 9.7007598 / 100.0,
    'other_viral_pathogens': 9.7007598 / 100.0,
    'other_bacterial_pathogens': 9.7007598 / 100.0
}

calibration_incidence_rate_1_year_olds = {
    'RSV': 9.7007598 / 100.0,
    'Rhinovirus': 9.7007598 / 100.0,
    'HMPV': 9.7007598 / 100.0,
    'Parainfluenza': 9.7007598 / 100.0,
    'Strep_pneumoniae_PCV13': 9.7007598 / 100.0,
    'Strep_pneumoniae_non_PCV13': 9.7007598 / 100.0,
    'Hib': 9.7007598 / 100.0,
    'H.influenzae_non_type_b': 9.7007598 / 100.0,
    'Staph_aureus': 9.7007598 / 100.0,
    'Enterobacteriaceae': 9.7007598 / 100.0,
    'other_Strepto_Enterococci': 9.7007598 / 100.0,
    'Influenza': 9.7007598 / 100.0,
    'P.jirovecii': 9.7007598 / 100.0,
    'Bocavirus': 9.7007598 / 100.0,
    'Adenovirus': 9.7007598 / 100.0,
    'other_viral_pathogens': 9.7007598 / 100.0,
    'other_bacterial_pathogens': 9.7007598 / 100.0
}

calibration_incidence_rate_2_to_4_year_olds = {
    'RSV': 9.7007598 / 100.0,
    'Rhinovirus': 9.7007598 / 100.0,
    'HMPV': 9.7007598 / 100.0,
    'Parainfluenza': 9.7007598 / 100.0,
    'Strep_pneumoniae_PCV13': 9.7007598 / 100.0,
    'Strep_pneumoniae_non_PCV13': 9.7007598 / 100.0,
    'Hib': 9.7007598 / 100.0,
    'H.influenzae_non_type_b': 9.7007598 / 100.0,
    'Staph_aureus': 9.7007598 / 100.0,
    'Enterobacteriaceae': 9.7007598 / 100.0,
    'other_Strepto_Enterococci': 9.7007598 / 100.0,
    'Influenza': 9.7007598 / 100.0,
    'P.jirovecii': 9.7007598 / 100.0,
    'Bocavirus': 9.7007598 / 100.0,
    'Adenovirus': 9.7007598 / 100.0,
    'other_viral_pathogens': 9.7007598 / 100.0,
    'other_bacterial_pathogens': 9.7007598 / 100.0
}

# Produce a set of line plot comparing to the calibration data
fig, axes = plt.subplots(ncols=2, nrows=5, sharey=True, figsize=(10, 20))
for ax_num, pathogen in enumerate(sim.modules['ALRI'].pathogens):
    ax = fig.axes[ax_num]
    inc_rate['0y'][pathogen].plot(ax=ax, label='Model output')
    ax.hlines(y=calibration_incidence_rate_0_year_olds[pathogen],  # axhlines is to plot horizontal lines at each y
              xmin=min(inc_rate['0y'].index),
              xmax=max(inc_rate['0y'].index),
              label='calibrating_data'
              )
    ax.set_title(f'{pathogen}')
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidence Rate")
    ax.legend()
plt.title('Incidence Rate among <1 year old')
plt.savefig(outputpath / ("ALRI_inc_rate_by_pathogen_0_year_olds" + datestamp + ".pdf"), format='pdf')
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
plt.ylabel('Risk of pathogen causing pneumonia per year')
plt.savefig(outputpath / ("ALRI_inc_rate_calibration_0_year_olds" + datestamp + ".pdf"), format='pdf')
plt.tight_layout()
plt.show()

# 1 year-olds
inc_mean.plot.bar(y=['1y_model_output', '1y_calibrating_data'])
plt.title('Incidence Rate: 1 year-olds')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing pneumonia per year')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing pneumonia per year')
plt.savefig(outputpath / ("ALRI_inc_rate_calibration_1_year_olds" + datestamp + ".pdf"), format='pdf')
plt.tight_layout()
plt.show()

# 2-4 year-olds
inc_mean.plot.bar(y=['2-4y_model_output', '2-4y_calibrating_data'])
plt.title('Incidence Rate: 2-4 year-olds')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing pneumonia per year')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing pneumonia per year')
plt.savefig(outputpath / ("ALRI_inc_rate_calibration_2-4_year_olds" + datestamp + ".pdf"), format='pdf')
plt.tight_layout()
plt.show()


# %% ----------------------------  MEAN DEATH RATE BY PATHOGEN  ----------------------------
# # TODO: this set of graphs
# Load the death data to which we calibrate:
# IHME (www.healthdata.org) / GBD project --> total deaths due to pneumonia in Malawi,
# per 100,000 child-years (under 5's) https://vizhub.healthdata.org/gbd-compare/
# http://ghdx.healthdata.org/gbd-results-tool?params=gbd-api-2017-permalink/9dd202e225b13cc2df7557a5759a0aca

# http://ghdx.healthdata.org/gbd-results-tool
calibration_death_rate_per_year_under_5s = {
    2010: 193.68 / 100000,
    2011: 188.13 / 100000,
    2012: 166.88 / 100000,
    2013: 147.95 / 100000,
    2014: 137.38 / 100000,
    2015: 132.82 / 100000,
    2016: 125.80 / 100000,
    2017: 122.06 / 100000,
    2018: 113.38 / 100000,
    2019: 103.60 / 100000
}

all_deaths = output['tlo.methods.demography']['death']
all_deaths['year'] = pd.to_datetime(all_deaths['date']).dt.year
all_deaths = all_deaths.loc[all_deaths['age'] < 5].copy()
all_deaths['age_grp'] = all_deaths['age'].map(
    {0: '0y',
     1: '1y',
     2: '2-4y',
     3: '2-4y',
     4: '2-4y'}
)
deaths = all_deaths.groupby(by=['year', 'age_grp', 'cause']).size().reset_index()
deaths['cause_simplified'] = [x[0] for x in deaths['cause'].str.split('_')]
deaths = deaths.drop(deaths.loc[deaths['cause_simplified'] != 'ALRI'].index)
deaths = deaths.groupby(by=['age_grp', 'year']).size().reset_index()
deaths.rename(columns={0: 'count'}, inplace=True)
deaths.drop(deaths.index[deaths['year'] > 2010.0], inplace=True)
deaths = deaths.pivot(values='count', columns='age_grp', index='year')

# Death Rate = death count (by year, by age-group) / person-years
death_rate = deaths.div(py)

list_tuples = sorted(calibration_death_rate_per_year_under_5s.items())
x, y = zip(*list_tuples)  # unpack a list of pairs into two tuples
data_df = pd.DataFrame.from_dict(data=calibration_death_rate_per_year_under_5s, orient='index', columns=['GBD_data'])
data_df = data_df.rename_axis('year')
print(data_df)
# produce plot comparison in 2010 (<5s):
death_rate['under_5'] = death_rate.sum(axis=1)
death_rate = death_rate.drop(['0y', '1y', '2-4y'], axis=1)
death_rate = death_rate.rename_axis('year')

death_dict = death_rate.T.to_dict('list')
print(death_dict)
list_tuples1 = sorted(death_dict.items())
x1, y1 = zip(*list_tuples1)

joint_df = pd.concat([data_df, death_rate], axis=1)

# plot death rate comparison
plt.plot(x, y, color='tab:red', label='GBD_data')
plt.plot(x1, y1, color='tab:blue', label='Model output')
plt.xlabel('Year')
plt.ylabel('Death rate')
axes = plt.gca()
axes.set_ylim(ymin=0)
# plt.legend('GBD data', 'Model output')
plt.title('Death rate from 2010 and 2019 - GBD data vs model output')
plt.savefig(outputpath / ("ALRI_death_rate_GBD_vs_model" + datestamp + ".pdf"), format='pdf')
plt.show()

# fig, ax1 = plt.subplots()
# ax1.set_xlabel('Year')
# ax1.set_ylabel('Death rate')
# ax1.plot(x, y, color='tab:red', marker="^", label='GBD_data')
# ax1.plot(x1, y1, color='tab:blue', marker="o", label='model_output')
#
# plt.show()

#
# death_rate_comparison.plot.bar()
# plt.title('Death Rate to ALRI in Under 5s')
# plt.savefig(outputpath / ("ALRI_death_rate_0-5_year_olds" + datestamp + ".pdf"), format='pdf')
# plt.show()
