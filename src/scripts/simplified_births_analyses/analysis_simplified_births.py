"""** I think this file can be deleted: it's now superceded by the calibration files **"""

# %% Import Statements
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from tlo import Date, Simulation
from tlo.methods import demography, simplified_births

# Where will outputs go - by default, wherever this script is run
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource directory for modules
# by default, this script runs in the same directory as this file
resourcefilepath = Path("./resources")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 1000

sim = Simulation(start_date=start_date, seed=1, log_config={'filename': 'simplified_births', 'directory': outputpath})

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# define the dataframe
df = sim.population.props

# select babies born during simulation
number_of_ever_newborns = df.loc[df.date_of_birth.notna() & (df.mother_id >= 0)]

# getting total number of newborns per year
total_births_per_year = pd.DataFrame(data=number_of_ever_newborns['date_of_birth'].dt.year.value_counts())
total_births_per_year.sort_index(inplace=True)
total_births_per_year.rename(columns={'date_of_birth': 'total_births'}, inplace=True)
# print(total_births)

# %% Plot Births:
years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

Model_Years = pd.to_datetime(total_births_per_year.index, format='%Y')
Model_total_births = total_births_per_year.total_births

fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_total_births)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Total Births per Year")
plt.xlabel("Year")
plt.ylabel("Number of births")

plt.legend(['Total births'])
plt.savefig(outputpath / ('Simplified Births' + datestamp + '.pdf'), format='pdf')
plt.show()
