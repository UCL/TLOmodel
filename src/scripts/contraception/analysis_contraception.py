import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    symptommanager,
)

# Path to the resource files used by the disease and intervention methods
resources = Path("./resources")

# Where will outputs go - by default, wherever this script is run
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# %% Run the Simulation
log_config = {
    "filename": "contraception_analysis",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.contraception": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO
    }
}

# Basic arguments required for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2020, 12, 31)
pop_size = 20000

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# Used to configure health system behaviour
service_availability = ["*"]

# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
    demography.Demography(resourcefilepath=resources),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
    symptommanager.SymptomManager(resourcefilepath=resources),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
    healthsystem.HealthSystem(resourcefilepath=resources, disable=False),
    # <-- HealthSystem functioning

    contraception.Contraception(resourcefilepath=resources, use_healthsystem=True),  # <-- using HealthSystem
    contraception.SimplifiedPregnancyAndLabour(),

    hiv.DummyHivModule(),
)

"""N.B. To examine the usage of contraceptive consumables, we need to retrieve this from the HealthSystem log as in
 the below example. But, if we don't need this, we can run the model much more quickly using the a configuration of
```
sim.register(
    demography.Demography(resourcefilepath=resources),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
    symptommanager.SymptomManager(resourcefilepath=resources),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
    healthsystem.HealthSystem(resourcefilepath=resources, disable=True),  # <-- disable HealthSystem

    contraception.Contraception(resourcefilepath=resources, use_healthsystem=False),  # <-- do not use HealthSystem
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resources),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
    labour.Labour(resourcefilepath=resources),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resources),

    hiv.DummyHivModule(),
)
```
 """

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df = parse_log_file(sim.log_filepath)

# %% Plot Contraception Use Over time:
years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
co_df = log_df['tlo.methods.contraception']['contraception_use_summary'].set_index('date')
Model_Years = pd.to_datetime(co_df.index)
Model_total = co_df.sum(axis=1)
Model_not_using = co_df.not_using
Model_using = Model_total - Model_not_using

fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_total)
ax.plot(np.asarray(Model_Years), Model_not_using)
ax.plot(np.asarray(Model_Years), Model_using)
# plt.plot(Data_Years, Data_Pop_Normalised)

# format the ticks
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_major_formatter(years_fmt)

plt.title("Contraception Use")
plt.xlabel("Year")
plt.ylabel("Number of women")
# plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['Total women age 15-49 years', 'Not Using Contraception', 'Using Contraception'])
plt.savefig(outputpath / ('Contraception Use' + datestamp + '.png'), format='png')
plt.show()

# %% Plot proportion of women using each contraception method over time:
years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
co_df = log_df['tlo.methods.contraception']['contraception_use_summary'].set_index('date')
Model_Years = pd.to_datetime(co_df.index)
Model_total = co_df.sum(axis=1)
Model_pill = co_df.pill
Model_IUD = co_df.IUD
Model_injections = co_df.injections
Model_implant = co_df.implant
Model_male_condom = co_df.male_condom
Model_female_sterilization = co_df.female_sterilization
Model_other_modern = co_df.other_modern
Model_periodic_abstinence = co_df.periodic_abstinence
Model_withdrawal = co_df.withdrawal
Model_other_traditional = co_df.other_traditional

fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_pill/Model_total)
ax.plot(np.asarray(Model_Years), Model_IUD/Model_total)
ax.plot(np.asarray(Model_Years), Model_injections/Model_total)
ax.plot(np.asarray(Model_Years), Model_implant/Model_total)
ax.plot(np.asarray(Model_Years), Model_male_condom/Model_total)
ax.plot(np.asarray(Model_Years), Model_female_sterilization/Model_total)
ax.plot(np.asarray(Model_Years), Model_other_modern/Model_total)
ax.plot(np.asarray(Model_Years), Model_periodic_abstinence/Model_total)
ax.plot(np.asarray(Model_Years), Model_withdrawal/Model_total)
ax.plot(np.asarray(Model_Years), Model_other_traditional/Model_total)
# plt.plot(Data_Years, Data_Pop_Normalised)

# format the ticks
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_major_formatter(years_fmt)

plt.title("Proportion of women using each contraception method over time:")
plt.xlabel("Year")
plt.ylabel("% of women")
# plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization',
            'other_modern', 'periodic_abstinence', 'withdrawal', 'other_traditional'])
plt.savefig(outputpath / ('Contraception Use Proportion' + datestamp + '.png'), format='png')
plt.show()

# %% Plot proportion of women using each contraception method over time:
years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
co_df = log_df['tlo.methods.contraception']['contraception_use_summary'].set_index('date')
Model_Years = pd.to_datetime(co_df.index)
Model_total = co_df.sum(axis=1)
Model_pill = co_df.pill
Model_IUD = co_df.IUD
Model_implant = co_df.implant
Model_male_condom = co_df.male_condom
Model_other_modern = co_df.other_modern
Model_periodic_abstinence = co_df.periodic_abstinence
Model_withdrawal = co_df.withdrawal
Model_other_traditional = co_df.other_traditional

fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_pill/Model_total)
ax.plot(np.asarray(Model_Years), Model_IUD/Model_total)
ax.plot(np.asarray(Model_Years), Model_implant/Model_total)
ax.plot(np.asarray(Model_Years), Model_male_condom/Model_total)
ax.plot(np.asarray(Model_Years), Model_other_modern/Model_total)
ax.plot(np.asarray(Model_Years), Model_periodic_abstinence/Model_total)
ax.plot(np.asarray(Model_Years), Model_withdrawal/Model_total)
ax.plot(np.asarray(Model_Years), Model_other_traditional/Model_total)
# plt.plot(Data_Years, Data_Pop_Normalised)

# format the ticks
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_major_formatter(years_fmt)

plt.title("Proportion of women using each contraception method over time:")
plt.xlabel("Year")
plt.ylabel("% of women")
# plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['pill', 'IUD', 'implant', 'male_condom',
            'other_modern', 'periodic_abstinence', 'withdrawal', 'other_traditional'])
plt.savefig(outputpath / ('Contraception Use Proportion magnified' + datestamp + '.png'), format='png')
plt.show()

# %% Plot Contraception Use By Method Over time:

years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
com_df = log_df['tlo.methods.contraception']['contraception_use_summary']
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
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_major_formatter(years_fmt)

plt.title("Contraception Use By Method")
plt.xlabel("Year")
plt.ylabel("Number using method")
# plt.gca().set_ylim(0, 50)
# plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization',
            'other_modern', 'periodic_abstinence', 'withdrawal', 'other_traditional'])
plt.savefig(outputpath / ('Contraception Use By Method' + datestamp + '.png'), format='png')
plt.show()

# %% Plot Pregnancies Over time:

years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results
preg_df = log_df['tlo.methods.contraception']['pregnancy'].set_index('date')
preg_df.index = pd.to_datetime(preg_df.index).year
num_pregs_by_year = preg_df.groupby(by=preg_df.index).size()
Model_Years = num_pregs_by_year.index
Model_pregnancy = num_pregs_by_year.values


fig, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_pregnancy)


# format the ticks
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_major_formatter(years_fmt)

plt.title("Pregnancies Over Time")
plt.xlabel("Year")
plt.ylabel("Number of pregnancies")
# plt.gca().set_ylim(0, 50)
# plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
plt.legend(['total', 'pregnant', 'not_pregnant'])
plt.savefig(outputpath / ('Pregnancies Over Time' + datestamp + '.png'), format='png')
plt.show()


# %% Plot Consumables Over time:
"""
If has the simulation has been with Contraception module using the HealthSystem and the HealthSystem not disabled, then
 use ...

cons = log_df['tlo.methods.healthsystem']['Consumables'].copy()

# Load consumables log and put the date as the index and provide
cons['date'] = pd.to_datetime(cons['date'])
cons['year'] = cons['date'].dt.year
cons = cons.set_index('year')

# Drop any entry that is not related to Contraception
cons = cons.loc[cons.TREATMENT_ID.str.startswith('Contraception')]

# Make counts of every type of package that is actually used (i.e. was available when requested) each year.
def unpack(in_dict_as_string):
    in_dict = eval(in_dict_as_string)
    l = list()
    for k, v in in_dict.items():
        for _v in range(v):
            l.append(k)
    return l

pkg_counts = cons['Package_Available'].apply(unpack).apply(pd.Series).dropna().astype(int)[0].value_counts()
"""

# What follows is TimC's original code for this section:
# ...
# ...
# years = mdates.YearLocator()  # every year
# months = mdates.MonthLocator()  # every month
# years_fmt = mdates.DateFormatter('%Y')
#
# # Load Model Results
# com_df = log_df['tlo.methods.contraception']['contraception_use_yearly_summary']
# Model_Years = pd.to_datetime(com_df.date)
# Model_pill = com_df.pill
# Model_IUD = com_df.IUD
# Model_injections = com_df.injections
# Model_implant = com_df.implant
# Model_male_condom = com_df.male_condom
# Model_female_sterilization = com_df.female_sterilization
# Model_other_modern = com_df.other_modern
#
# fig, ax = plt.subplots()
# ax.plot(np.asarray(Model_Years), Model_pill)
# ax.plot(np.asarray(Model_Years), Model_IUD)
# ax.plot(np.asarray(Model_Years), Model_injections)
# ax.plot(np.asarray(Model_Years), Model_implant)
# ax.plot(np.asarray(Model_Years), Model_male_condom)
# ax.plot(np.asarray(Model_Years), Model_female_sterilization)
# ax.plot(np.asarray(Model_Years), Model_other_modern)
#
# # format the ticks
# # ax.xaxis.set_major_locator(years)
# # ax.xaxis.set_major_formatter(years_fmt)
#
# plt.title("Contraception Consumables By Method")
# plt.xlabel("Year")
# plt.ylabel("Consumables used (number using method")
# # plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
# plt.legend(['pills', 'IUDs', 'injections', 'implants', 'male_condoms', 'female_sterilizations',
#             'other modern'])
# plt.savefig(outputpath / ('Contraception Consumables By Method' + datestamp + '.png'), format='png')
# plt.show()
#
# # %% Plot Consumable Costs Over time:
#
# years = mdates.YearLocator()  # every year
# months = mdates.MonthLocator()  # every month
# years_fmt = mdates.DateFormatter('%Y')
#
# # Load Model Results
# com_df = log_df['tlo.methods.contraception']['contraception_costs_yearly_summary']
# Model_Years = pd.to_datetime(com_df.date)
# Model_pill = com_df.pill_annual_cost
# Model_IUD = com_df.IUD_annual_cost
# Model_injections = com_df.injections_annual_cost
# Model_implant = com_df.implant_annual_cost
# Model_male_condom = com_df.male_condom_annual_cost
# Model_female_sterilization = com_df.female_sterilization_annual_cost
# Model_other_modern = com_df.other_modern_annual_cost
#
# fig, ax = plt.subplots()
# ax.plot(np.asarray(Model_Years), Model_pill)
# ax.plot(np.asarray(Model_Years), Model_IUD)
# ax.plot(np.asarray(Model_Years), Model_injections)
# ax.plot(np.asarray(Model_Years), Model_implant)
# ax.plot(np.asarray(Model_Years), Model_male_condom)
# ax.plot(np.asarray(Model_Years), Model_female_sterilization)
# ax.plot(np.asarray(Model_Years), Model_other_modern)
#
# # format the ticks
# # ax.xaxis.set_major_locator(years)
# # ax.xaxis.set_major_formatter(years_fmt)
#
# plt.title("Contraception Consumable Costs By Method")
# plt.xlabel("Year")
# plt.ylabel("Consumable Costs (Cumulative)")
# # plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
# plt.legend(['pill costs', 'IUD costs', 'injection costs', 'implant costs', 'male condom costs',
#             'female sterilization costs', 'other modern method costs'])
# plt.savefig(outputpath / ('Contraception Consumable Costs By Method' + datestamp + '.png'), format='png')
# plt.show()
#
# # %% Plot Public Health Costs Over time:
#
# years = mdates.YearLocator()  # every year
# years_fmt = mdates.DateFormatter('%Y')
#
# # Load Model Results
# com_df = log_df['tlo.methods.contraception']['contraception_costs_yearly_summary']
# Model_Years = pd.to_datetime(com_df.date)
# Model_public_health_costs1 = com_df.public_health_costs1
# Model_public_health_costs2 = com_df.public_health_costs2
#
# fig, ax = plt.subplots()
# ax.plot(np.asarray(Model_Years), Model_public_health_costs1)
# ax.plot(np.asarray(Model_Years), Model_public_health_costs2)
#
# # format the ticks
# # ax.xaxis.set_major_locator(years)
# # ax.xaxis.set_major_formatter(years_fmt)
#
# plt.title("Public Health Costs for Contraception uptake")
# plt.xlabel("Year")
# plt.ylabel("Cost")
# # plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
# plt.legend(['population scope campaign to increase contraception initiation',
#             'post partum family planning (PPFP) campaign'])
# plt.savefig(outputpath / ('Public Health Costs' + datestamp + '.png'), format='png')
# plt.show()
