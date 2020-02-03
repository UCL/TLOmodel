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
end_date = Date(2015, 1, 2)
popsize = 10000

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
    i = c.div(pop[age_grp], axis=0).dropna()
    i['year'] = pd.to_datetime(i.index).year
    i.set_index('year', drop=True, inplace=True)
    inc_rate[age_grp] = i

# Get the incidence rates that were input:
base_incidence_rate = dict()
for pathogen in sim.modules['Diarrhoea'].pathogens:
    base_incidence_rate[pathogen] = \
        sim.modules['Diarrhoea'].parameters[f'base_incidence_diarrhoea_by_{pathogen}']

# Produce a set of line plot:
fig, axes = plt.subplots(ncols=2, nrows=5, sharey=True)
for ax_num, pathogen in enumerate(sim.modules['Diarrhoea'].pathogens):
    ax = fig.axes[ax_num]
    inc_rate['0y'][pathogen].plot(ax=ax, label='Model output')
    ax.hlines(y=base_incidence_rate[pathogen][0],
                     xmin=min(inc_rate['0y'].index),
                     xmax=max(inc_rate['0y'].index),
                     label='Parameter'
            )
    ax.set_title(f'{pathogen}')
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidence Rate")
    ax.legend()
plt.show()


# Produce a bar plot:


# %%
# Compare this with the desired incidence level.

# Get the incidence rates that were input:
calibration_incidence_rate_0_year_olds= {
        'rotavirus' : 5.0 ,
        'shigella' : 5.0 ,
        'adenovirus' : 5.0 ,
        'cryptosporidium' : 5.0 ,
        'campylobacter' : 5.0 ,
        'ST-ETEC' : 5.0 ,
        'sapovirus' : 5.0 ,
        'norovirus' : 5.0 ,
        'astrovirus' : 5.0 ,
        'tEPEC' : 5.0
}

# TODO: this plot!






