# %% Import Statements
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
from tlo.methods import contraception, demography

# Where will output go - by default, wherever this script is run
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource directory for modules
# by default, this script runs in the same directory as this file
resourcefilepath = Path("./resources")


# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2070, 1, 2)
popsize = 1000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# this block of code is to capture the output to file
logfile = outputpath / ('LogFile' + datestamp + '.log')

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

# %% Plot Contraception Use Over time:

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
co_df = output['tlo.methods.contraception']['contraception']
Model_Years = pd.to_datetime(co_df.date)
Model_total = co_df.total
Model_not_using = co_df.not_using
Model_using = co_df.using

fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_total)
ax.plot(np.asarray(Model_Years), Model_not_using)
ax.plot(np.asarray(Model_Years), Model_using)
# plt.plot(Data_Years, Data_Pop_Normalised)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Contraception Use")
plt.xlabel("Year")
plt.ylabel("Number of women")
# plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['Total women age 15-49 years', 'Not Using Contraception', 'Using Contraception'])
plt.savefig(outputpath / ('Contraception Use' + datestamp + '.pdf'),format='pdf')
plt.show()


# %% Plot Contraception Use By Method Over time:

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
com_df = output['tlo.methods.contraception']['contraception']
Model_Years = pd.to_datetime(com_df.date)
Model_pill = com_df.pill
Model_IUD = com_df.IUD
Model_injections = com_df.injections
Model_implant = com_df.implant
Model_male_condom = com_df.male_condom
Model_female_sterilization = com_df.female_sterilization
Model_other_modern = com_df.other_modern
Model_periodic_abstinence = com_df.periodic_abstinence
Model_withdrawal = com_df.withdrawal
Model_other_traditional = com_df.other_traditional

fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_pill)
ax.plot(np.asarray(Model_Years), Model_IUD)
ax.plot(np.asarray(Model_Years), Model_injections)
ax.plot(np.asarray(Model_Years), Model_implant)
ax.plot(np.asarray(Model_Years), Model_male_condom)
ax.plot(np.asarray(Model_Years), Model_female_sterilization)
ax.plot(np.asarray(Model_Years), Model_other_modern)
ax.plot(np.asarray(Model_Years), Model_periodic_abstinence)
ax.plot(np.asarray(Model_Years), Model_withdrawal)
ax.plot(np.asarray(Model_Years), Model_other_traditional)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Contraception Use By Method")
plt.xlabel("Year")
plt.ylabel("Number using method")
# plt.gca().set_ylim(0, 50)
# plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization',
            'other_modern', 'periodic_abstinence', 'withdrawal', 'other_traditional'])
plt.savefig(outputpath /  ('Contraception Use By Method' + datestamp + '.pdf'), format='pdf')
plt.show()

# %% Plot Pregnancies Over time:

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
preg_df = output['tlo.methods.contraception']['pregnancy']
Model_Years = pd.to_datetime(preg_df.date)
Model_pregnancy = preg_df.total
Model_pregnant = preg_df.pregnant
Model_not_pregnant = preg_df.not_pregnant

fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_total)
ax.plot(np.asarray(Model_Years), Model_pregnant)
ax.plot(np.asarray(Model_Years), Model_not_pregnant)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Pregnancies Over Time")
plt.xlabel("Year")
plt.ylabel("Number of pregnancies")
# plt.gca().set_ylim(0, 50)
# plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['total', 'pregnant', 'not_pregnant'])
plt.savefig(outputpath / ('Pregnancies Over Time' + datestamp + '.pdf'), format='pdf')
plt.show()
