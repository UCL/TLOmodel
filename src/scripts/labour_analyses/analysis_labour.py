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
from tlo.methods import demography, labour, newborn_outcomes, eclampsia_treatment, haemorrhage_treatment, \
    sepsis_treatment, caesarean_section

# Where will output go - by default, wherever this script is run
outputpath = ''

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# assume Python console is started in the top-leve TLOModel directory
resourcefile_demography = Path('./resources')


# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2050, 1, 1)
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
sim.register(demography.Demography(resourcefilepath=resourcefile_demography))
sim.register(labour.Labour())
sim.register(eclampsia_treatment.EclampsiaTreatment())
sim.register(caesarean_section.CaesareanSection())
sim.register(sepsis_treatment.SepsisTreatment())
sim.register(newborn_outcomes.NewbornOutcomes())
sim.register(haemorrhage_treatment.HaemorrhageTreatment())

sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# this will make sure that the logging file is complete
fh.flush()

# %% read the results
output = parse_log_file(logfile)

# %% Plot Maternal Deaths Over time:

deaths_df = output['tlo.methods.labour']['maternal_death']

plt.plot_date(deaths_df['date'], deaths_df['age'])
plt.xlabel('Year')
plt.ylabel('Age at Death') #This should just be number of deaths
plt.savefig(outputpath + 'MaternalDeaths' + datestamp + '.pdf')
plt.show()


# %% Plot Still Births  Over time:


# %% Plot Maternal Mortality Ratio Over time:


# %% Plot Maternal Still Birth Ratio Over time:
