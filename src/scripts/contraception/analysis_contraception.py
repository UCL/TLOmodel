# %% Import Statements
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import contraception, demography, enhanced_lifestyle, healthsystem, symptommanager, \
    healthseekingbehaviour, labour, pregnancy_supervisor

# Where will outputs go - by default, wherever this script is run
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource directory for modules
# by default, this script runs in the same directory as this file
#resourcefilepath = Path("./resources")


# %% Run the Simulation

# To reproduce the results, you must set the seed for the Simulation instance. The Simulation
# will seed the random number generators for each module when they are registered.
# If a seed argument is not given, one is generated. It is output in the log and can be
# used to reproduce results of a run
seed = 123

log_config = {
    "filename": "contraception_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.contraception": logging.INFO
    }
}

# Basic arguments required for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 2)
pop_size = 1000

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Path to the resource files used by the disease and intervention methods
resources = "./resources"

# Used to configure health system behaviour
service_availability = ["*"]

# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
    demography.Demography(resourcefilepath=resources),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
    healthsystem.HealthSystem(resourcefilepath=resources, service_availability=service_availability,
                              ignore_cons_constraints=True),  # ignore constraints allows everyone to get contraception
    symptommanager.SymptomManager(resourcefilepath=resources),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
    contraception.Contraception(resourcefilepath=resources),
    labour.Labour(resourcefilepath=resources),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
)
# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df = parse_log_file(sim.log_filepath)


# %% Plot Contraception Use Over time:
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
co_df = log_df['tlo.methods.contraception']['contraception']
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
plt.savefig(outputpath / ('Contraception Use' + datestamp + '.pdf'), format='pdf')
plt.show()


# %% Plot Contraception Use By Method Over time:

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
com_df = log_df['tlo.methods.contraception']['contraception']
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
plt.savefig(outputpath / ('Contraception Use By Method' + datestamp + '.pdf'), format='pdf')
plt.show()

# %% Plot Pregnancies Over time:

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
preg_df = log_df['tlo.methods.contraception']['pregnancy']
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

# %% Plot Consumables Over time:

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
com_df = log_df['tlo.methods.contraception']['contraception_consumables_summary']
Model_Years = pd.to_datetime(com_df.date)
Model_pill = com_df.pills
Model_IUD = com_df.IUDs
Model_injections = com_df.injections
Model_implant = com_df.implants
Model_male_condom = com_df.male_condoms
Model_female_sterilization = com_df.female_sterilizations
Model_female_condom = com_df.female_condoms

fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_pill)
ax.plot(np.asarray(Model_Years), Model_IUD)
ax.plot(np.asarray(Model_Years), Model_injections)
ax.plot(np.asarray(Model_Years), Model_implant)
ax.plot(np.asarray(Model_Years), Model_male_condom)
ax.plot(np.asarray(Model_Years), Model_female_sterilization)
ax.plot(np.asarray(Model_Years), Model_female_condom)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Contraception Consumables By Method")
plt.xlabel("Year")
plt.ylabel("Consumables used (number using method")
# plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['pills', 'IUDs', 'injections', 'implants', 'male_condoms', 'female_sterilizations',
            'female condoms'])
plt.savefig(outputpath / ('Contraception Consumables By Method' + datestamp + '.pdf'), format='pdf')
plt.show()

# %% Plot Consumable Costs Over time:

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
com_df = log_df['tlo.methods.contraception']['contraception_consumables_summary']
Model_Years = pd.to_datetime(com_df.date)
Model_pill = com_df.pill_costs
Model_IUD = com_df.IUD_costs
Model_injections = com_df.injections_costs
Model_implant = com_df.implant_costs
Model_male_condom = com_df.male_condom_costs
Model_female_sterilization = com_df.female_sterilization_costs
Model_female_condom = com_df.female_condom_costs

fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_pill)
ax.plot(np.asarray(Model_Years), Model_IUD)
ax.plot(np.asarray(Model_Years), Model_injections)
ax.plot(np.asarray(Model_Years), Model_implant)
ax.plot(np.asarray(Model_Years), Model_male_condom)
ax.plot(np.asarray(Model_Years), Model_female_sterilization)
ax.plot(np.asarray(Model_Years), Model_female_condom)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Contraception Consumable Costs By Method")
plt.xlabel("Year")
plt.ylabel("Consumable Costs (Cumulative)")
# plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['pill costs', 'IUD costs', 'injection costs', 'implant costs', 'male condom costs',
            'female sterilization costs', 'female condom costs'])
plt.savefig(outputpath / ('Contraception Consumable Costs By Method' + datestamp + '.pdf'), format='pdf')
plt.show()

# %% Plot Public Health Costs Over time:

years = mdates.YearLocator()   # every year
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
com_df = log_df['tlo.methods.contraception']['contraception']
Model_Years = pd.to_datetime(com_df.date)
Model_public_health_costs1 = com_df.public_health_costs1
Model_public_health_costs2 = com_df.public_health_costs2

fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_public_health_costs1)
ax.plot(np.asarray(Model_Years), Model_public_health_costs2)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Public Health Costs for Contraception uptake")
plt.xlabel("Year")
plt.ylabel("Cost")
#plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['population scope campaign to increase contraception initiation',
            'post partum family planning (PPFP) campaign'])
plt.savefig(outputpath / ('Public Health Costs' + datestamp + '.pdf'), format='pdf')
plt.show()

