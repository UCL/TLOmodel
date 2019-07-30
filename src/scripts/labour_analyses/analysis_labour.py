# %% Import Statements
import datetime
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, labour, lifestyle, newborn_outcomes

# Where will output go - by default, wherever this script is run
outputpath = ''

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# assume Python console is started in the top-leve TLOModel directory
resourcefile_demography = Path('./resources')


# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)
popsize = 100

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
sim.register(demography.Demography(resourcefilepath=resourcefile_demography))
sim.register(lifestyle.Lifestyle())
sim.register(labour.Labour())
sim.register(newborn_outcomes.NewbornOutcomes())
sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

#TODO; Need to think about integration with HealthBurden capture (DALY weights)

# this will make sure that the logging file is complete
fh.flush()

# %% read the results
output = parse_log_file(logfile)

# %% Plot Total Maternal Deaths Over time:

# https://stackoverflow.com/questions/38792122/how-to-group-and-count-rows-by-month-and-year-using-pandas

# Yearly Maternal Deaths
deaths_df = output['tlo.methods.labour']['maternal_death']
deaths_df['date'] = pd.to_datetime(deaths_df['date'])
deaths_df['year'] = deaths_df['date'].dt.year
death_by_cause = deaths_df.groupby(['year'])['person_id'].size()

death_by_cause = death_by_cause.reset_index()
death_by_cause.index = death_by_cause['year']
death_by_cause.drop(columns='year', inplace=True)
death_by_cause = death_by_cause.rename(columns={'person_id':'num_deaths'})

death_by_cause.plot.bar(stacked=True)
plt.title("Total Maternal Deaths per Year")
plt.show()

# Consider maternal mortality rate (deaths per 1000 population per anum

# %% Plot Still Births  Over time:

deaths_df = output['tlo.methods.labour']['still_birth']
deaths_df['date'] = pd.to_datetime(deaths_df['date'])
deaths_df['year'] = deaths_df['date'].dt.year
death_by_cause = deaths_df.groupby(['year'])['person_id'].size()

death_by_cause = death_by_cause.reset_index()
death_by_cause.index = death_by_cause['year']
death_by_cause.drop(columns='year', inplace=True)
death_by_cause = death_by_cause.rename(columns={'person_id':'num_deaths'})

death_by_cause.plot.bar(stacked=True)
plt.title(" Total Still Births per Year")
plt.show()

# %% Plot Maternal Mortality Ratio Over time:

maternal_deaths_df = output['tlo.methods.labour']['maternal_death']
maternal_deaths_df['date'] = pd.to_datetime(maternal_deaths_df['date'])
maternal_deaths_df['year'] = maternal_deaths_df['date'].dt.year
death_by_year_m = maternal_deaths_df.groupby(['year'])['person_id'].size()

live_births = output['tlo.methods.demography']['live_births']
live_births['date'] = pd.to_datetime(live_births['date'])
live_births['year'] = live_births['date'].dt.year
birth_by_year_n = live_births.groupby(['year'])['child'].size()

mmr_df = pd.concat((death_by_year_m, birth_by_year_n),axis=1)
mmr_df.columns = ['maternal_deaths', 'live_births']
mmr_df['MMR'] = mmr_df['maternal_deaths']/mmr_df['live_births'] * 100000

mmr_df.plot.bar(y='MMR', stacked=True)
plt.title("Yearly Maternal Mortality Rate")
plt.show()


# %% Plot  Still Birth Rate Over time (still births per 1000 births):

still_births_df = output['tlo.methods.labour']['still_birth']
still_births_df['date'] = pd.to_datetime(still_births_df['date'])
still_births_df['year'] = still_births_df['date'].dt.year
death_by_year_s = still_births_df.groupby(['year'])['person_id'].size()

all_births_df = output['tlo.methods.demography']['on_birth']
all_births_df['date'] = pd.to_datetime(all_births_df['date'])
all_births_df['year'] = all_births_df['date'].dt.year
birth_by_year = all_births_df.groupby(['year'])['child'].size()

sbr_df = pd.concat((death_by_year_s, birth_by_year), axis=1)
sbr_df.columns = ['still_births', 'all_births']
sbr_df['SBR'] = sbr_df['still_births']/sbr_df['all_births'] * 1000

sbr_df.plot.bar(y='SBR', stacked=True)
plt.title("Yearly Still Birth Rate")
plt.show()
