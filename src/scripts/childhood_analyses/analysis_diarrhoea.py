"""
This will run the Diarrhoea Module and plot the incidence rate of each pathogen by each age group.
This will then be compared with:
    * The input incidence rate for each pathogen
    * The desired incidence rate for each pathogen
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
end_date = Date(2015, 1, 2)
popsize = 100

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
# sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath)) ## removing this so remove any health care seeking so Ines can focus on the 'natural history' and 'epidemiology'
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

counts = output['tlo.methods.diarrhoea']['incidence_count_by_patho']
counts.set_index(
    'date',
    drop=True,
    inplace=True
)

# get population size to make a comparison
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

# Incidence rate among 0 year-olds
inc_rate = dict()
for age_grp in counts.columns:
    c = counts[age_grp].apply(pd.Series)
    inc_rate[age_grp] = c.div(pop[age_grp], axis=0).dropna()

# Get the incidence rates that were input:
base_incidence_rate = dict()
for pathogen in sim.modules['Diarrhoea'].pathogens:
    base_incidence_rate[pathogen] = \
        sim.modules['Diarrhoea'].parameters[f'base_incidence_diarrhoea_by_{pathogen}']
# TODO: finish this when can rely on the name of the parameters


# Compare this with the desried outputs.



# TODO: look at deaths








# Produce a plot:
inc_rate['0y'].plot()

plt.show()
