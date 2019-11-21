"""
Plot to demonstrate correspondence between model and data output wrt births, population size and total deaths
In the combination of both the codes from Tim C in Contraception and Tim H in Demography
"""

# %% Import Statements
import datetime
import logging
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import *
from tlo.methods import contraception, demography
from tlo.methods.demography import make_age_range_lookup


# Where will output go - by default, wherever this script is run
outputpath = ""

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# assume Python console is started in the top-leve TLOModel directory
resourcefilepath = Path("./resources")

# %% Run the Simulation
logfile = outputpath + 'LogFile' + datestamp + '.log'

start_date = Date(2010, 1, 1)
end_date = Date(2070, 1, 2)
popsize = 1000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# this block of code is to capture the output to file

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# this will make sure that the logging file is complete
fh.flush()

# %% read the results

# FOR STORED RESULTS
logfile = 'LogFile__2019_11_21.log'

parsed_output = parse_log_file(logfile)

scaled_output = scale_to_population(parsed_output, resourcefilepath)

#%% Population
# Trend in Number Over Time

# Population Growth Over Time:
# Load and format model results
model_df = scaled_output["tlo.methods.demography"]["population"]
model_df['year']= pd.to_datetime(model_df.date).dt.year

# Load Data: WPP_Annual
wpp_ann = pd.read_csv(Path(resourcefilepath) / "ResourceFile_Pop_Annual_WPP.csv")
wpp_ann_total = wpp_ann.groupby(['Year']).sum().sum(axis=1)

# Load Data: Census
cens = pd.read_csv(Path(resourcefilepath) / "ResourceFile_PopulationSize_2018Census.csv")
cens_2018 = cens.groupby('Sex')['Count'].sum()

# Plot population size over time
plt.plot(model_df['year'], model_df['total'])
plt.plot(wpp_ann_total.index, wpp_ann_total)
plt.plot(2018,cens_2018.sum(),'*')
plt.title("Population Size")
plt.xlabel("Year")
plt.ylabel("Population Size")
plt.gca().set_xlim(2010, 2050)
plt.legend(["Model", "WPP","Census 2018"])
plt.show()


# Population Size in 2018

# Census vs WPP vs Model
wpp_2018 = wpp_ann.groupby(['Year','Sex'])['Count'].sum()[2018]

# Get Model totals for males and females in 2018
model_2018 = model_df.loc[model_df['year']==2018,['male','female']].reset_index(drop=True).transpose().rename(index={'male':'M', 'female':'F'})

popsize = pd.concat([cens_2018, wpp_2018,model_2018], keys=['Census_2018','WPP','Model']).unstack()
popsize.transpose().plot(kind='bar')
plt.title('Population Size (2018)')
plt.show()

# TODO; Why the discrepancy between WPP and Census?
# TODO; GBD population data in here too


#%% Population
# Population Pyramid at two time points

# Import and format male population data:
model_m= scaled_output["tlo.methods.demography"]["age_range_m"]
model_m=model_m.loc[pd.to_datetime(model_m['date']).dt.year==2018].drop(columns=['date','Year']).melt(value_name='Model',var_name='Age_Grp')
model_m.index= model_m['Age_Grp'].astype(make_age_grp_types())
model_m = model_m.loc[model_m.index.dropna(),'Model']

# Import and format Census data:
cens = pd.read_csv(Path(resourcefilepath) / "ResourceFile_PopulationSize_2018Census.csv")
cens_m = cens.loc[cens['Sex']=='M'].groupby(by='Age_Grp')['Count'].sum()
cens_m.index=cens_m.index.astype(make_age_grp_types())
cens_m = cens_m.loc[cens_m.index.dropna()]

# Import and format WPP data:
wpp_ann = pd.read_csv(Path(resourcefilepath) / "ResourceFile_Pop_Annual_WPP.csv")
wpp_ann = wpp_ann.loc[wpp_ann['Year']==2018]
wpp_m = wpp_ann.loc[wpp_ann['Sex']=='M',['Count','Age_Grp']].groupby('Age_Grp')['Count'].sum()
wpp_m.index=wpp_m.index.astype(make_age_grp_types())
wpp_m =wpp_m.loc[wpp_m.index.dropna()]

# Make into dataframe:
pop_m=pd.DataFrame(model_m)
pop_m['Census']=cens_m
pop_m['WPP']=wpp_m

pop_m.plot.bar()
plt.show()

# TODO; Check WPP calcs
# TODO: repeat for women



#%% Births

# TH ************** GOT TO HERE **************

# Number over time
births = scaled_output['tlo.methods.demography']['birth_groupby_scaled']
births = births.reset_index()

# Births over time (Model)
# Aggregate the model output into five year periods:
(__tmp__, calendar_period_lookup) = make_calendar_period_lookup()
births["Period"] = births["year"].map(calendar_period_lookup)
births_model = births.groupby(by='Period')['count'].sum()
births_model.index = births_model.index.astype(make_calendar_period_type())

# Births over time (WPP)
wpp = pd.read_csv(Path(resourcefilepath) / "ResourceFile_TotalBirths_WPP.csv")
wpp = wpp.groupby(['Period','Variant'])['Total_Births'].sum().unstack()
wpp.index = wpp.index.astype(make_calendar_period_type())
wpp.columns ='WPP_' + wpp.columns

#TODO ADD CENSUS

# Merge in model results
wpp['Model']=births_model



ax = wpp.plot.line(y=['Model','WPP_Estimates','WPP_Medium variant'])
plt.xticks(np.arange(len(wpp.index)), wpp.index)
ax.fill_between(wpp.index, wpp['WPP_Low variant'], wpp['WPP_High variant'], facecolor='blue', alpha=0.5)
plt.xticks(rotation=90)
ax.set_title('Number of Births Per Calendar Period')
ax.legend(loc='upper left')
ax.set_xlabel('Calendar Period')
ax.set_ylabel('Number per period')
plt.show()



#%% Births
# Births to mothers by age
(__tmp__, age_grp_lookup) = make_age_range_lookup()
births["mother_age_grp"] = births["mother_age"].map(age_grp_lookup)
nbirths_byage = births.groupby(by=['year','mother_age_grp'])['child'].count().unstack(fill_value=0).stack()
nbirths_byage_2015 = nbirths_byage[2015]
nbirths_byage_2030 = nbirths_byage[2030]


#%% Deaths
# Number over time
deaths = scaled_output['tlo.methods.demography']['death_groupby_scaled']
deaths = deaths.reset_index()

# Aggregate the model output into five year periods for age and time:
(__tmp__, calendar_period_lookup) = make_calendar_period_lookup()
deaths["period"] = deaths["year"].map(calendar_period_lookup)
deaths["age_grp"] = deaths["age"].map(age_grp_lookup)

# Create time-series
ndeaths = deaths.groupby(by='period')['count'].sum()

# Get WPP data
wpp = pd.read_csv(Path(resourcefilepath) / "ResourceFile_TotalDeaths_WPP.csv")

# Adjust the labelling of the calendar periods in WPP so that it is inclusive years (2010-2014, 2015-2019 not 2010-2015, 2015-2020, etc)
wpp['t_lo'], wpp['t_hi']=wpp['Period'].str.split('-',1).str
wpp['t_hi'] = wpp['t_hi'].astype(int) - 1
wpp['period'] = wpp['t_lo'].astype(str) + '-' + wpp['t_hi'].astype(str)
# Make a total TODO: get rid of stings in WPP
wpp['total'] = wpp.iloc[:,2:21].astype(float).sum(axis=1)

# load the GBD data
gbd =  pd.read_csv(Path(resourcefilepath) / "ResourceFile_Deaths_And_Causes_DeathRates_GBD.csv")
gbd = gbd.loc[gbd['measure_name']=='Deaths'].copy()
gbd['period'] = gbd["year"].map(calendar_period_lookup)

# collapse by cause to give total
ndeaths_gbd = gbd.groupby(by=['period'])[['val','upper','lower']].sum()


fig, ax = plt.subplots(1)
ax.plot(ndeaths_gbd.index,ndeaths_gbd['val'],lw=2, label='GBD Estimate', color='blue')
ax.fill_between(ndeaths_gbd.index, ndeaths_gbd['lower'], ndeaths_gbd['upper'], facecolor='blue', alpha=0.5)
ax.plot(ndeaths.index,ndeaths,lw=2, label='Model', color='red')
ax.set_title('Number of Deaths Per Calendar Period')
ax.legend(loc='upper left')
plt.xticks(rotation=90)
ax.set_xlabel('Calendar Period')
ax.set_ylabel('Number per period')
plt.show()


#%% Deaths
# Number by age at two different time points
ndeaths_by_age_model = deaths.groupby(by=['period','age_grp'])['count'].sum().reset_index()

# ndeaths_by_age_model_2010 = ndeaths_by_age_model.loc[ndeaths_by_age_model['period']=='2010-2014']
# ndeaths_by_age_model_2030 = ndeaths_by_age_model.loc[ndeaths_by_age_model['period']=='2030-2034']

ndeaths_by_age_gbd = gbd.groupby(by=['period','age_name'])[['val','upper','lower']].sum().reset_index()




# SPECIFY ORDER

# Merge together
ndeaths = ndeaths_by_age_model.merge(ndeaths_by_age_gbd, on = ['period','age_grp'], suffixes=('Model','GBD'))




ndeaths_by_age_gbd_2010 = ndeaths_by_age_gbd.loc[ndeaths_by_age_gbd['period']=='2010-2014']
ndeaths_by_age_gbd_2030 = ndeaths_by_age_gbd.loc[ndeaths_by_age_gbd['period']=='2030-2034']


#TODO: Bring in WPP data here


# *****
# turn the age-grp into categories - this to go into utils
from pandas.api.types import CategoricalDtype

age_grp_cats = list()
for i in age_grp_lookup.values():
    if i not in age_grp_cats:
        age_grp_cats.append(i)


age_grp_type = CategoricalDtype(categories=age_grp_cats,ordered=True)


ndeaths['age_grp'] = ndeaths['age_grp'].astype(age_grp_type)
ndeaths = ndeaths.sort_values(by = ['age_grp'])
# ******

ndeaths = ndeaths.loc[ndeaths['period']=='2015-2019']
ndeaths.index=ndeaths['age_grp']
ndeaths.plot.line(x='age_grp',y=['count','val'])
plt.plot(ndeaths['lower'])
plt.fill_between(ndeaths['age_grp'],ndeaths['lower'],ndeaths['upper'], alpha=0.5)
plt.xticks(np.arange(len(ndeaths.index)), ndeaths.index)
ax[0].set_title('Number of Deaths by Age: 2010-2014')
ax[0].legend(loc='upper left')
ax[0].set_xlabel('Age Group')
ax[0].set_ylabel('Number per period')
plt.xticks(rotation=90)
plt.show()



fig, ax = plt.subplots(2)
ax[0].plot(ndeaths_by_age_gbd_2010['age_name'],ndeaths_by_age_gbd_2010['val'],lw=2, label='GBD Estimate', color='blue')
ax[0].fill_between(ndeaths_by_age_gbd_2010['age_name'],ndeaths_by_age_gbd_2010['lower'],ndeaths_by_age_gbd_2010['upper'], facecolor='blue', alpha=0.5)
ax[0].plot(ndeaths_by_age_model_2010['age_grp'],ndeaths_by_age_model_2010['count'],lw=2, label='Model', color='red')
ax[0].set_title('Number of Deaths by Age: 2010-2014')
ax[0].legend(loc='upper left')
ax[0].set_xlabel('Age Group')
ax[0].set_ylabel('Number per period')



ax = sns.lineplot(x='age_grp', y='value', \
                  data=ndeaths.melt(id_vars=['period','age_grp']), \
                  hue = 'variable')
plt.xticks(rotation=90)
plt.show()




#%% Desths
# Plot by region in the census

#%% Deaths
# Number by Cause by time

#%% Deaths
# Number by Cause by age and time


