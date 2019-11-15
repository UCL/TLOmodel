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
from tlo.analysis.utils import parse_log_file
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

start_date = Date(2010, 1, 1)
end_date = Date(2070, 1, 2)
popsize = 1000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# this block of code is to capture the output to file
logfile = outputpath + 'LogFile' + datestamp + '.log'

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
output = parse_log_file(logfile)

#%%



# get model population size by year for model and data for scaling the respective metrics of each

# Population Growth Over Time:
# Load Model Results
pop_df = output["tlo.methods.demography"]["population"]
pop_df['year']= pd.to_datetime(pop_df.date).dt.year
Model_Years = pop_df['year']
Model_Pop = pop_df.total
Model_Pop_Normalised = (np.asarray(Model_Pop) / np.asarray(Model_Pop[Model_Years == 2010]))

# Load Data
# 1) WPP_Annnual
wpp_ann = pd.read_csv(Path(resourcefilepath) / "ResourceFile_Pop_Annual_WPP.csv").groupby(['Year']).sum().sum(axis=1)
wpp_ann_norm = wpp_ann / wpp_ann[2010]
wpp_ann_norm.plot()
plt.show()


# Plot population size over time
plt.plot(np.asarray(Model_Years), Model_Pop_Normalised)
plt.plot(np.asarray(wpp_ann_norm.index), np.asarray(wpp_ann_norm))
plt.title("Population Size")
plt.xlabel("Year")
plt.ylabel("Population Size (Normalised to 2010)")
plt.gca().set_xlim(2010, 2050)
plt.legend(["Model (Normalised to 2010)", "WPP (Normalised to 2010)"])
plt.savefig(outputpath + "Pop_Size_Over_Time" + datestamp + ".pdf")
plt.show()



#%% Population Size

# Census vs WPP vs Model

cens = pd.read_csv(Path(resourcefilepath) / "ResourceFile_PopulationSize_2018Census.csv")
cens_2018 = cens.groupby('sex')['number'].sum()

wpp = (pd.read_csv(Path(resourcefilepath) / "ResourceFile_Pop_Annual_WPP.csv").groupby(['Year','Sex']).sum().sum(axis=1))
wpp_2018 = wpp[2018]

model = pop_df.melt(id_vars='year', value_vars=['male','female'],var_name='sex',value_name='number')
model['sex'] = model['sex'].replace({'male':'M', 'female':'F'})
model_2018 = model.loc[model['year']==2018].groupby(['sex'])['number'].sum()


cens_2018.plot.bar()
plt.show()

wpp_2018.plot.bar()
plt.show()

model_2018.plot.bar()
plt.show()

popsize = pd.concat([cens_2018, wpp_2018,model_2018], keys=['Census_2018','WPP','Model']).unstack()

popsize.transpose().plot(kind='bar')
plt.title('Population Size (2018)')
plt.show()

# TODO; Why the discrepancy between WPP and Census?
# TODO; GBD population data in here too

#%% Births

births = output['tlo.methods.demography']['on_birth']
births["date"] = pd.to_datetime(births["date"])
births["year"] = births["date"].dt.year



# Births over time (Model)
# Aggregate the model output into five year periods:
def make_calendar_period_lookup():
    """Returns a dictionary mapping calendar year (in years) to five year period
    i.e. { 0: '0-4', 1: '0-4', ..., 119: '100+', 120: '100+' }
    """

    def chunks(items, n):
        """Takes a list and divides it into parts of size n"""
        for index in range(0, len(items), n):
            yield items[index:index + n]

    # split all the ages from min to limit (100 years) into 5 year ranges
    parts = chunks(range(1950, 2100), 5)

    # any year >= 2100 are in the '2100+' category
    default_category = '%d+' % 2100
    lookup = defaultdict(lambda: default_category)

    # collect the possible ranges
    ranges = []

    # loop over each range and map all ages falling within the range to the range
    for part in parts:
        start = part.start
        end = part.stop - 1
        value = '%s-%s' % (start, end)
        ranges.append(value)
        for i in range(start, part.stop):
            lookup[i] = value

    ranges.append(default_category)
    return ranges, lookup


(__tmp__, calendar_period_lookup) = make_calendar_period_lookup()
births["period"] = births["year"].map(calendar_period_lookup)
nbirths = births.groupby(by='period')['child'].count()


# Births over time (WPP)
wpp = pd.read_csv(Path(resourcefilepath) / "ResourceFile_TotalBirths_WPP.csv")

# Adjust the labelling of the calendar periods in WPP so that it is inclusive years (2010-2014, 2015-2019 not 2010-2015, 2015-2020, etc)
wpp['t_lo'], wpp['t_hi']=wpp['Period'].str.split('-',1).str
wpp['t_hi'] = wpp['t_hi'].astype(int) - 1
wpp['period'] = wpp['t_lo'].astype(str) + '-' + wpp['t_hi'].astype(str)

wpp = wpp.groupby(['period','Variant'])['Total_Births'].sum()


fig, ax = plt.subplots(1)
ax.plot(wpp[:,'Estimates'].index,wpp[:,'Estimates'],lw=2, label='WPP Estimate', color='blue')
ax.plot(wpp[:,'Medium variant'].index, wpp[:,'Medium variant'], lw=2, label='WPP Projection', color='blue')
ax.fill_between(wpp[:,'Low variant'].index, wpp[:,'Low variant'], wpp[:,'High variant'], facecolor='blue', alpha=0.5)
ax.plot(nbirths.index,nbirths,lw=2, label='Model', color='red')
plt.xticks(rotation=90)
ax.set_title('Number of Births Per Calendar Period')
ax.legend(loc='upper left')
ax.set_xlabel('Calendar Period')
ax.set_ylabel('Number per period')
plt.show()


#%%

#TODO: make utilitiy function to scale-up the population size for particualr thing in the log

ax.plot(list(range(len(wpp[:,'Low variant']))), wpp[:,'Medium variant'], lw=2, label='WPP Projection', color='blue')
ax.fill_between(t, wpp[:,'Low variant'], wpp[:,'High variant'], facecolor='blue', alpha=0.5)

# Births to mothers by age
(__tmp__, age_grp_lookup) = make_age_range_lookup()
births["mother_age_grp"] = births["mother_age"].map(age_grp_lookup)
nbirths_byage = births.groupby(by=['year','mother_age_grp'])['child'].count().unstack(fill_value=0).stack()
nbirths_byage_2015 = nbirths_byage[2015]
nbirths_byage_2030 = nbirths_byage[2030]





##% Deaths
