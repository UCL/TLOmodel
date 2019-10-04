import datetime
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, healthburden, healthsystem, lifestyle, hypertension #, t2dm, chronicsyndrome, mockitis,

# [NB. Working directory must be set to the root of TLO: TLOmodel/]
# TODO: adapt to NCD analysis
# TODO: add check that there is HTN >0
# TODO: add check that HTN is in right prevalence
# TODO: plot data versus model

# Where will output go
outputpath = ""

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")


# %% Run the Simulation

start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=2011, month=12, day=31)
popsize = 1000

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath + 'LogFile' + datestamp + '.log'

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)


# -----------------------------------------------------------------
# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ['*']


# -----------------------------------------------------------------
# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=service_availability))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
sim.register(hypertension.Hypertension(resourcefilepath=resourcefilepath))
# sim.register(t2dm.T2DM(resourcefilepath=resourcefilepath))


# -----------------------------------------------------------------
# Run the simulation and flush the logger
sim.seed_rngs(5)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()


# %% read the results
output = parse_log_file(logfile)

# -----------------------------------------------------------------
# Plot output

# Load Population and normalise
pop_df = output["tlo.methods.demography"]["population"]
Model_Years = pd.to_datetime(pop_df.date)
Model_Pop = pop_df.total
Model_Pop_Normalised = (
     100 * np.asarray(Model_Pop) / np.asarray(Model_Pop[Model_Years == "2010-01-01"])
)


plt.plot(np.asarray(Model_Years), Model_Pop_Normalised)


# Load hypertension data and plot it
prev_df = output["tlo.methods.hypertension"]["ht_prevalence"]
Model_Years = pd.to_datetime(prev_df.date)
Model_Prev_total = prev_df.total

plt.plot(np.asarray(Model_Years), Model_Prev_total)


# Load validation date
val_data_df = output["tlo.methods.hypertension"]["ht_prevalence_data_validation"]
val_model_df = output["tlo.methods.hypertension"]["ht_prevalence_model_validation"]
Plot_Years = pd.to_datetime(val_data_df.date)  # Dates from either will work, they are identical






prev_data_df = output["tlo.methods.hypertension"]["ht_prevalence_data_2"]

Model_Years = pd.to_datetime(prev_data_df.date)
Model_Prev_Data_total = prev_data_df.total

plt.plot(np.asarray(Model_Years), Model_Prev_Data_total)

Data_Prev_total = prev_data_df.total

plt.plot(np.asarray(Model_Years), Model_Prev_total)

#plt.plot(np.asarray(Model_Years), Model_Prev_total, Data_Prev_total)   #TODO: need to fix code to plot data versus code

plt.show()





