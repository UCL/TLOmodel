"""
This will run the Diarrhoea Module and plot the rate of death for diarrhoea overall and compare with data.
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
    diarrhoea,
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
end_date = Date(2016, 1, 1)
popsize = 20_000

log_config = {
    'filename': 'LogFile',
    'custom_levels': {
        '*': logging.WARNING,
        'tlo.methods.demography': logging.INFO,
        'tlo.methods.diarrhoea': logging.INFO
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
             diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
             dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
             )

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Get the output from the logfile
output = parse_log_file(sim.log_filepath)


# %% ----------------------------  INCIDENCE RATE OF DIARRHOEA BY PATHOGEN  ----------------------------

#  Calculate the "incidence rate" from the output counts of incidence
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
    'rotavirus': 17.5245863 / 100.0,
    'shigella': 11.7936462 / 100.0,
    'adenovirus': 5.866180 / 100.0,
    'cryptosporidium': 3.0886699 / 100.0,
    'campylobacter': 9.8663257 / 100.0,
    'ST-ETEC': 27.925146 / 100.0,
    'sapovirus': 10.0972179 / 100.0,
    'norovirus': 20.4864004 / 100.0,
    'astrovirus': 5.4208352 / 100.0,
    'tEPEC': 6.0822457 / 100.0
}

calibration_incidence_rate_1_year_olds = {
    'rotavirus': 9.7007598 / 100.0,
    'shigella': 7.8794104 / 100.0,
    'adenovirus': 5.8661803 / 100.0,
    'cryptosporidium': 1.1792363 / 100.0,
    'campylobacter': 2.7915478 / 100.0,
    'ST-ETEC': 17.0477152 / 100.0,
    'sapovirus': 13.2603114 / 100.0,
    'norovirus': 6.6146727 / 100.0,
    'astrovirus': 3.5974076 / 100.0,
    'tEPEC': 2.2716889 / 100.0
}

calibration_incidence_rate_2_to_4_year_olds = {
    'rotavirus': 0.9324 / 100.0,
    'shigella': 9.3018 / 100.0,
    'adenovirus': 0.6438 / 100.0,
    'cryptosporidium': 0.4662 / 100.0,
    'campylobacter': 0.4884 / 100.0,
    'ST-ETEC': 1.9758 / 100.0,
    'sapovirus': 0.555 / 100.0,
    'norovirus': 0.0888 / 100.0,
    'astrovirus': 0.1332 / 100.0,
    'tEPEC': 0.1998 / 100.0
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
plt.title('Incidence Rate: 0 year-olds')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing diarrhoea per year')
plt.savefig(outputpath / ("Diarrhoea_inc_rate_calibration_0_year_olds" + datestamp + ".pdf"), format='pdf')
plt.tight_layout()
plt.show()

# 1 year-olds
inc_mean.plot.bar(y=['1y_model_output', '1y_calibrating_data'])
plt.title('Incidence Rate: 1 year-olds')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing diarrhoea per year')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing diarrhoea per year')
plt.savefig(outputpath / ("Diarrhoea_inc_rate_calibration_1_year_olds" + datestamp + ".pdf"), format='pdf')
plt.tight_layout()
plt.show()

# 2-4 year-olds
inc_mean.plot.bar(y=['2-4y_model_output', '2-4y_calibrating_data'])
plt.title('Incidence Rate: 2-4 year-olds')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing diarrhoea per year')
plt.xlabel('Pathogen')
plt.ylabel('Risk of pathogen causing diarrhoea per year')
plt.savefig(outputpath / ("Diarrhoea_inc_rate_calibration_2-4_year_olds" + datestamp + ".pdf"), format='pdf')
plt.tight_layout()
plt.show()


# %% ----------------------------  MEAN DEATH RATE BY PATHOGEN  ----------------------------
# Load the death data to which we calibrate:
# IHME (www.healthdata.org) / GBD project --> total deaths due to diarrhoea in Malawi,
# per 100,000 child-years (under 5's) https://vizhub.healthdata.org/gbd-compare/
# http://ghdx.healthdata.org/gbd-results-tool?params=gbd-api-2017-permalink/9dd202e225b13cc2df7557a5759a0aca

calibration_death_rate_per_year_under_5s = {
    '2010': 148 / 100000,   # CI: 111-190
    '2017': 93 / 100000     # CI: 61-135
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
deaths = deaths.drop(deaths.loc[deaths['cause_simplified'] != 'Diarrhoea'].index)
deaths = deaths.groupby(by=['age_grp', 'year']).size().reset_index()
deaths.rename(columns={0: 'count'}, inplace=True)
# deaths.drop(deaths.index[deaths['year'] > 2010.0], inplace=True)
deaths = deaths.pivot(values='count', columns='age_grp', index='year')

# Death Rate = death count (by year, by age-group) / person-years
death_rate = deaths.div(py)

# produce plot comparison in 2010 (<5s):
death_rate_comparison = pd.Series(
    data={
        'data (2010)': calibration_death_rate_per_year_under_5s['2010'],
        'data (2017)': calibration_death_rate_per_year_under_5s['2017'],
        'model': death_rate.loc[2010].mean()
    }
)

death_rate_comparison.plot.bar()
plt.title('Death Rate to Diarrhoea in Under 5s')
plt.savefig(outputpath / ("Diarrhoea_death_rate_0-5_year_olds" + datestamp + ".pdf"), format='pdf')
plt.show()


# %% Plot total numbers of death against comparable estimate from GBD
from scripts.utils.helper_funcs_for_processing_data_files import load_gbd_deaths_and_dalys_data

scaling_factor = demography.get_scaling_factor(output, resourcefilepath)

gbd = load_gbd_deaths_and_dalys_data(output)

diarrhoea_deaths_gbd = (gbd.loc[
    (gbd.measure_name == 'Deaths') &
    (gbd.unified_cause == 'Childhood Diarrhoea') &
    (gbd.age_range == '0-4') &
    gbd.year.isin([2010, 2011, 2012, 2013, 2014, 2015, 2016])
    ]).groupby(by='year')[['val', 'upper', 'lower']].sum()

deaths_under_5s_scaled = deaths.sum(axis=1) * scaling_factor

fig, ax = plt.subplots()
ax.plot(deaths_under_5s_scaled.index, deaths_under_5s_scaled.values, 'r--', label='Model: With Treatment')
ax.plot(diarrhoea_deaths_gbd.index, diarrhoea_deaths_gbd.val, 'b', label='GBD Diarrhoea deaths <5s')
ax.fill_between(
    diarrhoea_deaths_gbd.index,
    diarrhoea_deaths_gbd.lower,
    diarrhoea_deaths_gbd.upper, color='b', alpha=0.5)
ax.legend()
ax.set_ylabel('Number of deaths')
ax.set_title('Comparison Between Model and GBD')
plt.show()

#%% Look at Case Fatality Rate
cfr = dict()
for age_grp in ['0y', '1y', '2-4y']:
    cfr[age_grp] = deaths[age_grp] / counts[age_grp].apply(pd.Series).sum(axis=1)
cfr = pd.DataFrame(cfr).drop(index=2015).mean() * 100_000

cfr.plot.bar()
plt.title('Case Fatality Rate for Diarrhoea')
plt.ylabel('Deaths per 100k Cases')
plt.xlabel('Age-Group')
plt.show()
