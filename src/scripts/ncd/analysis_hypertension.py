import datetime
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, healthburden, healthsystem, lifestyle, hypertension


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
val_data_df = output["tlo.methods.hypertension"]["ht_prevalence_data_validation"]    # Load the existing data
val_model_df = output["tlo.methods.hypertension"]["ht_prevalence_model_validation"]  # Load the model data
Data_Years = pd.to_datetime(val_data_df.date)  # Pick out the information about time from data
Data_total = val_data_df.total    # Pick out overall prevalence from data
Data_min = val_data_df.total_min  # Pick out min CI
Data_max = val_data_df.total_max  # Pick out max CI
Model_Years = pd.to_datetime(val_model_df.date)  # Pick out the information about time from data
Model_total = val_model_df.total  # Pick out overall prevalence from model


# Check there is hypertension if it compares to data
# TODO: would be nice to have it as break or assert (didn't manage with those functions)
print("Is there hypertension in the model:", "\n", Model_total > 0)
print("Is the prevalence of hypertension above the min 95% CI of the data: ", "\n", Data_min < Model_total)
print("Is the prevalence of hypertension above the min 95% CI of the data: ", "\n", Data_max < Model_total)


# Scatter graph of overall prevalence data vs model
Plot_Years = (2010, 2011)   # ToDo: can this be automated - start_date:end_date without time format (dsnt work scatter)

plt.scatter(Plot_Years, Data_total, label='Data', color='k')
plt.scatter(Plot_Years, Data_min, label='Min 95% CI', color='grey')
plt.scatter(Plot_Years, Data_max, label='Max 95% CI', color='grey')
plt.scatter(Plot_Years, Model_total, label='Model', color='red')
plt.title("Overall prevalence: data vs model")
plt.xlabel("Years")
plt.ylabel("Prevalence")
plt.xticks(np.arange(min(Data_Years), max(Data_Years)+1))
plt.gca().set_ylim(0, 100)
plt.gca().legend(loc='lower right', bbox_to_anchor=(1.4, 0.5))
plt.show()


# Load and plot overall age-specific model vs data
# Generate a dataframe, clean index and populating it from outputAs above retrieve corresponding output and plot.
# TODO: see if there is a fast way of coding this
Plot_AgeGroups = ('25 to 35', '35 to 45', '45 to 55', '55 to 65')
df = val_data_df
df2 = val_model_df
df.index = Plot_Years
df2.index = Plot_Years

Data_Age = pd.DataFrame(index=Data_Years,
                        columns=['Age25to35', 'Age35to45', 'Age45to55', 'Age55to65'],
                        data=df[['age25to35', 'age35to45', 'age45to55', 'age55to65']].values)
Data_Age_min = pd.DataFrame(index=Data_Years,
                            columns=['Age25to35_min', 'Age35to45_min', 'Age45to55_min', 'Age55to65_min'],
                            data=df[['age25to35_min', 'age35to45_min', 'age45to55_min', 'age55to65_min']].values)
Data_Age_max = pd.DataFrame(index=Data_Years,
                            columns=['Age25to35_max', 'Age35to45_max', 'Age45to55_max', 'Age55to65_max'],
                            data=df[['age25to35_max', 'age35to45_max', 'age45to55_max', 'age55to65_max']].values)
Model_Age = pd.DataFrame(index=Data_Years,
                         columns=['Age25to35', 'Age35to45', 'Age45to55', 'Age55to65'],
                         data=df2[['25to35', '35to45', '45to55', '55to65']].values)

# Clean by year
# 2010
Data_Age_2010 = pd.DataFrame(Data_Age.loc[2010]).transpose()
Data_Age_min_2010 = pd.DataFrame(Data_Age_min.loc[2010]).transpose()
Data_Age_max_2010 = pd.DataFrame(Data_Age_max.loc[2010]).transpose()
Model_Age_2010 = pd.DataFrame(Model_Age.loc[2010]).transpose()

# 2011 onwards - only model needed
Model_Age_2011 = pd.DataFrame(Model_Age.loc[2011]).transpose()


# Plot the whole lot
plt.scatter(Plot_AgeGroups, Data_Age_2010, label='Data by age', color='k')
plt.scatter(Plot_AgeGroups, Data_Age_min_2010, label='Min 95% CI data by age', color='grey')
plt.scatter(Plot_AgeGroups, Data_Age_max_2010, label='Max 95% CI data by age', color='grey')
plt.scatter(Plot_AgeGroups, Model_Age_2010, label='Model by age - 2010', color='red')
plt.scatter(Plot_AgeGroups, Model_Age_2011, label='Model by age - 2011', color='orange')

plt.title("Age-specific prevalence: data vs model")
plt.xlabel("Age Groups")
plt.ylabel("Prevalence")
plt.gca().set_ylim(0, 100)
plt.gca().legend(loc='lower right', bbox_to_anchor=(0.45, 0.65))

plt.show()

print("We are half way!")
