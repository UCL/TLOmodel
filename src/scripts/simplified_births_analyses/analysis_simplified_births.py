# %% Import Statements
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    simplified_births
)

# Where will outputs go - by default, wherever this script is run
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource directory for modules
# by default, this script runs in the same directory as this file
resourcefilepath = Path("./resources")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 2)
popsize = 1000

sim = Simulation(start_date=start_date, seed=1, log_config={'filename': 'simple_births', 'directory': outputpath})

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             simplified_births.Simplifiedbirths(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# %% read the results
output = parse_log_file(sim.log_filepath)

# %% Plot Contraception Use Over time:
years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
si_df = output['tlo.methods.simplified_births']['total_births']
Model_Years = pd.to_datetime(si_df.date)
Model_total_births = si_df.total


fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_total_births)
# plt.plot(Data_Years, Data_Pop_Normalised)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Simple births")
plt.xlabel("Year")
plt.ylabel("Number of births")

plt.legend(['Total births'])
plt.savefig(outputpath / ('Simplified Births' + datestamp + '.pdf'), format='pdf')
plt.show()
