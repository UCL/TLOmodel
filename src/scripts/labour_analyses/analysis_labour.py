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
from tlo.methods import demography, labour, lifestyle, newborn_outcomes, healthsystem

# Where will output go - by default, wherever this script is run
#outputpath = './src/scripts/analyses_labour/'
outputpath = ""

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# assume Python console is started in the top-leve TLOModel directory
#resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
resourcefilepath = "./resources/"


# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 2000

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
sim.register(lifestyle.Lifestyle())
sim.register(labour.Labour(resourcefilepath=resourcefilepath))
sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))

sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

#TODO; Need to think about integration with HealthBurden capture (DALY weights)
# TODO: must capture indirect death of women during pregnancy

# this will make sure that the logging file is complete
fh.flush()

# %% read the results
output = parse_log_file(logfile)

# %% Plot Total Maternal Deaths Over time:

# https://stackoverflow.com/questions/38792122/how-to-group-and-count-rows-by-month-and-year-using-pandas

# Yearly Maternal Deaths
deaths_df = output['tlo.methods.labour']['maternal_death'] #N.B WONT RUN ON VERY SMALL POP- NEEDS SOME DEATHS
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


# %% Plot Still Births  Over time:

deaths_df = output['tlo.methods.labour']['still_birth']
deaths_df['date'] = pd.to_datetime(deaths_df['date'])
deaths_df['year'] = deaths_df['date'].dt.year
death_by_cause = deaths_df.groupby(['year'])['person_id'].size()
# TODO: n.b do we count maternal death where the baby dies as a still birth

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

# %% Plot Perinatal Mortality Rate Over time (stillbirths + deaths in first week of life per 1000 births ):

neonatal_deaths_df = output['tlo.methods.newborn_outcomes']['neonatal_death_48hrs']
neonatal_deaths_df['date'] = pd.to_datetime(neonatal_deaths_df['date'])
neonatal_deaths_df['year'] = neonatal_deaths_df['date'].dt.year
death_by_year_pmr = (neonatal_deaths_df.groupby(['year'])['person_id'].size()) + death_by_year_s

pmr_df = pd.concat((death_by_year_pmr, birth_by_year), axis=1)
pmr_df.columns = ['perinatal_deaths', 'all_births']
pmr_df['PMR'] = pmr_df['perinatal_deaths']/pmr_df['all_births'] * 1000

pmr_df.plot.bar(y='PMR', stacked=True)
plt.title("Yearly Perinatal Mortality Rate")
plt.show()

# ----------------------------------- Incidence of Complications in Labour --------------------------------------------
# Todo: Confirm standard metrics of incidence for these complications to allow comparison

# Maternal Sepsis
# Yearly Incidence(per 1000 births):
maternal_sepsis_df = output['tlo.methods.labour']['maternal_sepsis']
maternal_sepsis_df['date'] = pd.to_datetime(maternal_sepsis_df['date'])
maternal_sepsis_df['year'] = maternal_sepsis_df['date'].dt.year
sepsis_by_year = maternal_sepsis_df.groupby(['year'])['person_id'].size()

msi_df=pd.concat((sepsis_by_year, birth_by_year), axis=1)
msi_df.columns = ['maternal_sepsis_cases', 'all_births']
msi_df['sepsis_incidence'] = msi_df['maternal_sepsis_cases']/msi_df['all_births'] * 1000

msi_df.plot.bar(y='sepsis_incidence', stacked=True)
plt.title("Yearly Maternal Sepsis Incidence")
plt.show()

# Eclampsia
# Yearly Incidence(per 10000 (?) births):
eclampsia_df = output['tlo.methods.labour']['eclampsia']
eclampsia_df['date'] = pd.to_datetime(eclampsia_df['date'])
eclampsia_df['year'] = eclampsia_df['date'].dt.year
eclampsia_by_year = eclampsia_df.groupby(['year'])['person_id'].size()

ei_df=pd.concat((eclampsia_by_year, birth_by_year), axis=1)
ei_df.columns = ['eclampsia_cases', 'all_births']
ei_df['eclampsia_incidence'] = ei_df['eclampsia_cases']/ei_df['all_births'] * 10000

ei_df.plot.bar(y='eclampsia_incidence', stacked=True)
plt.title("Yearly Incidence of Eclampsia")
plt.show()

# Obstructed Labour (per 1000 births)
obstructed_labour_df = output['tlo.methods.labour']['obstructed_labour']
obstructed_labour_df['date'] = pd.to_datetime(obstructed_labour_df['date'])
obstructed_labour_df['year'] = obstructed_labour_df['date'].dt.year
obstructed_labour_year = obstructed_labour_df.groupby(['year'])['person_id'].size()

ol_df=pd.concat((obstructed_labour_year, birth_by_year), axis=1)
ol_df.columns = ['obstructed_labour_cases', 'all_births']
ol_df['obstructed_labour_incidence'] = ol_df['obstructed_labour_cases']/ol_df['all_births'] * 1000

ol_df.plot.bar(y='obstructed_labour_incidence', stacked=True)
plt.title("Yearly Incidence of Obstructed Labour")
plt.show()

# Antepartum Haemorrhage

aph_df = output['tlo.methods.labour']['antepartum_haemorrhage']
aph_df['date'] = pd.to_datetime(aph_df['date'])
aph_df['year'] = aph_df['date'].dt.year
aph_year = aph_df.groupby(['year'])['person_id'].size()

aph_n_df = pd.concat((aph_year, birth_by_year), axis=1)
aph_n_df.columns = ['aph_cases', 'all_births']
aph_n_df['aph_incidence'] = aph_n_df['aph_cases']/aph_n_df['all_births'] * 1000

aph_n_df.plot.bar(y='aph_incidence', stacked=True)
plt.title("Yearly Incidence of Antepartum Haemorrhage")
plt.show()

# Post Partum Haemorrhage

pph_df = output['tlo.methods.labour']['postpartum_haemorrhage']
pph_df['date'] = pd.to_datetime(pph_df['date'])
pph_df['year'] = pph_df['date'].dt.year
pph_year = pph_df.groupby(['year'])['person_id'].size()

pph_n_df = pd.concat((pph_year, birth_by_year), axis=1)
pph_n_df.columns = ['pph_cases', 'all_births']
pph_n_df['pph_incidence'] = pph_n_df['pph_cases']/pph_n_df['all_births'] * 1000

pph_n_df.plot.bar(y='pph_incidence', stacked=True)
plt.title("Yearly Incidence of Postpartum Haemorrhage")
plt.show()
