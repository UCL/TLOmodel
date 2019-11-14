import datetime
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    hypertension,
)

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
resourcefilepath = Path(os.path.dirname(__file__)) / '../../../resources'

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
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
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
val_data_df = output["tlo.methods.hypertension"]["ht_prevalence_data_validation"]  # Load the existing data
val_model_df = output["tlo.methods.hypertension"]["ht_prevalence_model_validation"]
val_model_df2 = output["tlo.methods.hypertension"]["ht_prevalence_model_validation_2"]  # Load the model data
# TODO: I do not get the same numbers out for the model output using groupby. Need to check and redo plotting code
Data_Years = pd.to_datetime(val_data_df.date)  # Pick out the information about time from data
Data_total = val_data_df.prevalence_total  # Pick out overall prevalence from data
Data_min = val_data_df.min95ci_total  # Pick out min CI
Data_max = val_data_df.max95ci_total  # Pick out max CI
Model_Years = pd.to_datetime(val_model_df.date)  # Pick out the information about time from data
Model_total = val_model_df.total  # Pick out overall prevalence from model

# Check there is hypertension if it compares to data
# TODO: would be nice to have it as break or assert (didn't manage with those functions)
# TODO: will have to refit once all BMI cats are there so prevalence is actually in line with data - ignore for now
print("Is there hypertension in the model:", "\n", Model_total > 0)
print("Is the prevalence of hypertension above the min 95% CI of the data: ", "\n", Data_min < Model_total)
print("Is the prevalence of hypertension above the min 95% CI of the data: ", "\n", Data_max > Model_total)


# Scatter graph of overall prevalence data vs model
Plot_Years = (2010, 2011)  # ToDo: can this be automated - start_date:end_date without time format (dsnt work scatter)

plt.scatter(Plot_Years, Data_total, label='Data', color='k')
plt.scatter(Plot_Years, Data_min, label='Min 95% CI', color='grey')
plt.scatter(Plot_Years, Data_max, label='Max 95% CI', color='grey')
plt.scatter(Plot_Years, Model_total, label='Model', color='red')
plt.title("Overall prevalence: data vs model")
plt.xlabel("Years")
plt.ylabel("Prevalence")
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

Data_Age = pd.DataFrame(
    index=Plot_Years,
    columns=['Age25to35', 'Age35to45', 'Age45to55', 'Age55to65'],
    data=df[['prevalence_25_to_35', 'prevalence_35_to_45', 'prevalence_45_to_55', 'prevalence_55_to_65']].values,
)
Data_Age_min = pd.DataFrame(
    index=Plot_Years,
    columns=['Age25to35_min', 'Age35to45_min', 'Age45to55_min', 'Age55to65_min'],
    data=df[['min95ci_25_to_35', 'min95ci_35_to_45', 'min95ci_45_to_55', 'min95ci_55_to_65']].values,
)
Data_Age_max = pd.DataFrame(
    index=Plot_Years,
    columns=['Age25to35_max', 'Age35to45_max', 'Age45to55_max', 'Age55to65_max'],
    data=df[['max95ci_25_to_35', 'max95ci_35_to_45', 'max95ci_45_to_55', 'max95ci_55_to_65']].values,
)
Model_Age = pd.DataFrame(
    index=Plot_Years,
    columns=['Age25to35', 'Age35to45', 'Age45to55', 'Age55to65'],
    data=df2[['25to35', '35to45', '45to55', '55to65']].values,
)

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

# Load prevalence model and compare to data other than STEP data
val_data_df = output["tlo.methods.hypertension"]["ht_prevalence_data_extra"]  # Load the existing data
val_model_df = output["tlo.methods.hypertension"]["ht_prevalence_model_extra"]
Data_Years = pd.to_datetime(val_data_df.date)  # Pick out the information about time from data
Data_price = val_data_df.prevalence_price  # Pick out overall prevalence from price data
Data_price_min = val_data_df.min95ci_price  # Pick out min CI
Data_price_max = val_data_df.max95ci_price  # Pick out max CI

Data_divala = val_data_df.prevalence_divala  # Pick out overall prevalence from divala data
Data_divala_min = val_data_df.min95ci_divala  # Pick out min CI
Data_divala_max = val_data_df.max95ci_divala  # Pick out max CI

Data_ruecker = val_data_df.prevalence_ruecker  # Pick out overall prevalence from ruecker data
Data_ruecker_min = val_data_df.min95ci_ruecker  # Pick out min CI
Data_ruecker_max = val_data_df.max95ci_ruecker  # Pick out max CI

Data_ramirez = val_data_df.prevalence_ramirez  # Pick out overall prevalence from ramirez data
Data_ramirez_min = val_data_df.min95ci_ramirez  # Pick out min CI
Data_ramirez_max = val_data_df.max95ci_ramirez  # Pick out max CI

Model_Years = pd.to_datetime(val_model_df.date)  # Pick out the information about time from data
Model_price = val_model_df.divala_model  # Pick out overall prevalence from model
Model_divala = val_model_df.price_model  # Pick out overall prevalence from model
Model_ruecker = val_model_df.ruecker_model  # Pick out overall prevalence from model
Model_ramirez = val_model_df.ramirez_model  # Pick out overall prevalence from model


# Scatter graph of overall prevalence of non-STEP data vs model
Plot_Years = (2010, 2011)  # ToDo: can this be automated - start_date:end_date without time format (dsnt work scatter)

# Price et al 2018
plt.scatter(Plot_Years, Data_price, label='Data', color='k')
plt.scatter(Plot_Years, Data_price_min, label='Min 95% CI', color='grey')
plt.scatter(Plot_Years, Data_price_max, label='Max 95% CI', color='grey')
plt.scatter(Plot_Years, Model_price, label='Model', color='red')

plt.title("Overall prevalence: data vs model (Price et al 2018)")
plt.xlabel("Years")
plt.ylabel("Prevalence")
plt.gca().set_ylim(0, 100)
plt.gca().legend(loc='lower right', bbox_to_anchor=(1.4, 0.5))
plt.show()


# Divala et al 2017
plt.scatter(Plot_Years, Data_divala, label='Data', color='k')
plt.scatter(Plot_Years, Data_divala_min, label='Min 95% CI', color='grey')
plt.scatter(Plot_Years, Data_divala_max, label='Max 95% CI', color='grey')
plt.scatter(Plot_Years, Model_divala, label='Model', color='red')

plt.title("Overall prevalence: data vs model (Divala et al 2017)")
plt.xlabel("Years")
plt.ylabel("Prevalence")
plt.gca().set_ylim(0, 100)
plt.gca().legend(loc='lower right', bbox_to_anchor=(1.4, 0.5))
plt.show()

# Ruecker et al 2018
plt.scatter(Plot_Years, Data_ruecker, label='Data', color='k')
plt.scatter(Plot_Years, Data_ruecker_min, label='Min 95% CI', color='grey')
plt.scatter(Plot_Years, Data_ruecker_max, label='Max 95% CI', color='grey')
plt.scatter(Plot_Years, Model_ruecker, label='Model', color='red')

plt.title("Overall prevalence: data vs model (Ruecker et al 2018)")
plt.xlabel("Years")
plt.ylabel("Prevalence")
plt.gca().set_ylim(0, 100)
plt.gca().legend(loc='lower right', bbox_to_anchor=(1.4, 0.5))
plt.show()

# Ramirez et al 2010
plt.scatter(Plot_Years, Data_ramirez, label='Data', color='k')
plt.scatter(Plot_Years, Data_ramirez_min, label='Min 95% CI', color='grey')
plt.scatter(Plot_Years, Data_ramirez_max, label='Max 95% CI', color='grey')
plt.scatter(Plot_Years, Model_ramirez, label='Model', color='red')

plt.title("Overall prevalence: data vs model (Ramirez et al 2010)")
plt.xlabel("Years")
plt.ylabel("Prevalence")
plt.gca().set_ylim(0, 100)
plt.gca().legend(loc='lower right', bbox_to_anchor=(1.4, 0.5))
plt.show()

print("We are half way!")
