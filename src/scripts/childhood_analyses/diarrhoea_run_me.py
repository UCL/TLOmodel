"""
@ Ines - whilst we work on this, I think running this file will make it easier to do the debugging.
So let this me the file that we both use to run the model for now.

"""

import datetime
import logging
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, enhanced_lifestyle, diarrhoea, symptommanager, healthsystem, \
    childhood_management, contraception, healthseekingbehaviour

resourcefilepath = Path('./resources')
outputpath = Path('./outputs/')

# %% Run the Simulation
start_date = Date(2010, 1, 1)
end_date = Date(2019, 1, 1)
popsize = 1000

# Set up the logger:
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath / ("LogFile" + datestamp + ".log")
# Set up the logger:
# logfile = outputpath / ("LogFile.log")


if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

sim = Simulation(start_date=start_date)
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
# sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath)) ## removing this so remove any health care seeking so Ines can focus on the 'natural history' and 'epidemiology'
sim.register(diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath))
sim.register(childhood_management.ChildhoodManagement(resourcefilepath=resourcefilepath))

sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

fh.flush()
output = parse_log_file(logfile)

# %% -----------------------------------------------------------------------------------
# %% Plot Incidence of Diarrhoea Over time:
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results on clinical types of diarrhoea
clinical_type_df = output['tlo.methods.diarrhoea']['clinical_diarrhoea_type']
Model_Years = pd.to_datetime(clinical_type_df.date)
Model_AWD = clinical_type_df.AWD
Model_dysentery = clinical_type_df.dysentery
Model_persistent = clinical_type_df.persistent

fig1, ax = plt.subplots(figsize=(9, 7))
ax.plot(np.asarray(Model_Years), Model_AWD)
ax.plot(np.asarray(Model_Years), Model_dysentery)
ax.plot(np.asarray(Model_Years), Model_persistent)

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Total clinical diarrhoea")
plt.xlabel("Year")
plt.ylabel("Number of diarrhoea episodes")
plt.legend(['acute watery diarrhoea', 'dysentery', 'persistent diarrhoea'])
plt.savefig(outputpath / ('3 clinical diarrhoea types' + datestamp + '.pdf'))

plt.show()
