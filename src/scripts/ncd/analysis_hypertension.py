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
# Plot and check output

# Load overall prevalence model and prevalence data
val_data_df = output["tlo.methods.hypertension"]["ht_prevalence_data_validation"]   # Load the existing data
val_model_df = output["tlo.methods.hypertension"]["ht_prevalence_model_validation"] #Load the model data
Data_Years = pd.to_datetime(val_data_df.date)  # Pick out the information about time from data
Data_total = val_data_df.total  # Pick out overall prevalence from data
Data_min = val_data_df.total_min # Pick out min CI
Data_max = val_data_df.total_max # Pick out max CI
Model_Years = pd.to_datetime(val_model_df.date)  # Pick out the information about time from data
Model_total = val_model_df.total    # Pick out overall prevalence from model

# Check there is hypertension and the righ prevalence
#TODO: check still robust
print(Model_total>0)        #TODO: want to use assert but not workign with series
print(Data_min<Model_total)
print(Data_max<Model_total)

plt.plot(np.asarray(Data_Years), Data_total, label='Date')
plt.plot(np.asarray(Model_Years), Data_min, label='Min 95% CI')
plt.plot(np.asarray(Model_Years), Data_max, label='Max 95% CI')
plt.plot(np.asarray(Model_Years), Model_total, label='Model')
plt.title("Overall prevalence: data vs model")
plt.xlabel("Years")
plt.ylabel("Prevalence")
plt.gca().set_xlim(Date(2010,1,1), Date(2011,1,1))
plt.gca().set_ylim(0,100)
plt.gca().legend(loc='lower right', bbox_to_anchor=(1.4, 0.5))

plt.show()


# Repeat but as scatter graph
Data_Years = (2010, 2011)   #ToDo: can this be automated - start_date:end_date without time format (dsnt work scatter)

plt.scatter(Data_Years, Data_total, label='Date')
plt.scatter(Data_Years, Data_min, label='Min 95% CI')
plt.scatter(Data_Years, Data_max, label='Max 95% CI')
plt.scatter(Data_Years, Model_total, label='Model')
plt.title("Overall prevalence: data vs model")
plt.xlabel("Years")
plt.ylabel("Prevalence")
plt.xticks(np.arange(min(Data_Years), max(Data_Years)+1))
plt.gca().set_ylim(0,100)
plt.gca().legend(loc='lower right', bbox_to_anchor=(1.4, 0.5))
plt.show()


# Load and plot overall age-specific model vs data
#TODO: Write below
Data_total = val_data_df.total  # Pick out overall prevalence from data
Data_min = val_data_df.total_min # Pick out min CI
Data_max = val_data_df.total_max # Pick out max CI
Model_Years = pd.to_datetime(val_model_df.date)  # Pick out the information about time from data
Model_total = val_model_df.total    # Pick out overall prevalence from model

plt.plot(np.asarray(Data_Years), Data_total)
plt.plot(np.asarray(Model_Years), Data_min)
plt.plot(np.asarray(Model_Years), Data_max)
plt.plot(np.asarray(Model_Years), Model_total)

plt.show()





