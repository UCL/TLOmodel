"""
This will run the Diarrhoea Module and plot the incidence rate of each pathogen by each age group.
This will then be compared with:
    * The input incidence rate for each pathogen
    * The desired incidence rate for each pathogen
There is no treatment.
"""

# %% Import Statements and initial declarations
import datetime
import logging
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tlo import Date, Simulation
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import contraception, demography, diarrhoea, childhood_management, healthsystem, enhanced_lifestyle, \
    symptommanager, healthburden

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath / ('LogFile' + datestamp + '.log')

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 2)
popsize = 5000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# this block of code is to capture the outputs to file

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)
logging.getLogger("tlo.methods.demography").setLevel(logging.INFO)
logging.getLogger("tlo.methods.contraception").setLevel(logging.INFO)
logging.getLogger("tlo.methods.diarrhoea").setLevel(logging.INFO)


# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath))
sim.register(childhood_management.ChildhoodManagement(resourcefilepath=resourcefilepath))

sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# this will make sure that the logging file is complete
fh.flush()
output = parse_log_file(logfile)

# %%
# Calculate the "incidence rate" from the output counts of incidence

counts = output['tlo.methods.diarrhoea']['incidence_count_by_pathogen']
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
        (py_.loc[pd.to_datetime(py_['date']).dt.year==year]['M']).apply(pd.Series) + \
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
    ).transpose()

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
pop['0y']=pop[0]
pop['1y']=pop[1]
pop['2-4y']=pop[2]+pop[3]+pop[4]
pop.drop(columns=[x for x in range(5)], inplace=True)

# Incidence rate among 0, 1, 2-4 year-olds
inc_rate = dict()
for age_grp in ['0y', '1y', '2-4y']:
    inc_rate[age_grp] = counts[age_grp].apply(pd.Series).div(py[age_grp], axis=0).dropna()

# Get the 3 month risk parameters that were input:
base_riskper3mo = dict()
for pathogen in sim.modules['Diarrhoea'].pathogens:
    base_riskper3mo[pathogen] = \
        sim.modules['Diarrhoea'].parameters[f'base_inc_rate_diarrhoea_by_{pathogen}']

# Load the incidence rate data to which we calibrate
calibration_incidence_rate_0_year_olds = {
        'rotavirus': 17.5245863/100.0,
        'shigella': 11.7936462/100.0,
        'adenovirus': 5.866180/100.0,
        'cryptosporidium': 3.0886699/100.0,
        'campylobacter': 9.8663257/100.0,
        'ST-ETEC': 27.925146/100.0,
        'sapovirus': 10.0972179/100.0,
        'norovirus': 20.4864004/100.0,
        'astrovirus': 5.4208352/100.0,
        'tEPEC': 6.0822457/100.0
}

calibration_incidence_rate_1_year_olds = {
        'rotavirus': 9.7007598/100.0,
        'shigella': 7.8794104/100.0,
        'adenovirus': 5.8661803/100.0,
        'cryptosporidium': 1.1792363/100.0,
        'campylobacter': 2.7915478/100.0,
        'ST-ETEC': 17.0477152/100.0,
        'sapovirus': 13.2603114/100.0,
        'norovirus': 6.6146727/100.0,
        'astrovirus': 3.5974076/100.0,
        'tEPEC': 2.2716889/100.0
}

calibration_incidence_rate_2_to_4_year_olds = {
        'rotavirus': 0.9324/100.0,
        'shigella': 9.3018/100.0,
        'adenovirus': 0.6438/100.0,
        'cryptosporidium': 0.4662/100.0,
        'campylobacter': 0.4884/100.0,
        'ST-ETEC': 1.9758/100.0,
        'sapovirus': 0.555/100.0,
        'norovirus': 0.0888/100.0,
        'astrovirus': 0.1332/100.0,
        'tEPEC': 0.1998/100.0
}

# Produce a set of line plot comparing to the calibration data
fig, axes = plt.subplots(ncols=2, nrows=5, sharey=True)
for ax_num, pathogen in enumerate(sim.modules['Diarrhoea'].pathogens):
    ax = fig.axes[ax_num]
    inc_rate['0y'][pathogen].plot(ax=ax, label='Model output')
    ax.hlines(y=calibration_incidence_rate_0_year_olds[pathogen],
                     xmin=min(inc_rate['0y'].index),
                     xmax=max(inc_rate['0y'].index),
                     label='calibrating_data'
            )
    ax.set_title(f'{pathogen}')
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidence Rate")
    ax.legend()
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
plt.title('0 year-olds')
plt.show()

# 1 year-olds
inc_mean.plot.bar(y=['1y_model_output', '1y_calibrating_data'])
plt.title('1 year-olds')
plt.show()

# 2-4 year-olds
inc_mean.plot.bar(y=['2-4y_model_output', '2-4y_calibrating_data'])
plt.title('2-4 year-olds')
plt.show()

# %%
# Look at deaths arising? Or anything else?

# 3 month check - running
# 1 mont check - running - good.
# 6 moth - good.
