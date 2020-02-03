"""
Plot to demonstrate correspondence between model and data outputs wrt births, population size and total deaths
In the combination of both the codes from Tim C in Contraception and Tim H in Demography
"""

# %% Import Statements and initial declarations
import datetime
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import (
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    parse_log_file,
)
from tlo.methods import (
    contraception,
    demography,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    enhanced_lifestyle,
    symptommanager,
)
from tlo.util import create_age_range_lookup

outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# assume Python console is started in the top-leve TLOModel directory
# resourcefilepath = Path(os.path.dirname(__file__)) / '../../../resources'
resourcefilepath = Path("./resources")

logfile = outputpath / ('LogFile' + datestamp + '.log')

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 2)
popsize = 200

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# this block of code is to capture the outputs to file

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# this will make sure that the logging file is complete
fh.flush()

# %% read the results

# FOR STORED RESULTS
# logfile = 'LogFile__2020_01_07.log'

parsed_output = parse_log_file(logfile)

scale_to_population = demography.scale_to_population
scaled_output = scale_to_population(parsed_output, resourcefilepath)

# %% Population Size
# Trend in Number Over Time

# Population Growth Over Time:
# Load and format model results
model_df = scaled_output["tlo.methods.demography"]["population"]
model_df['year'] = pd.to_datetime(model_df.date).dt.year

# Load Data: WPP_Annual
wpp_ann = pd.read_csv(Path(resourcefilepath) / "ResourceFile_Pop_Annual_WPP.csv")
wpp_ann_total = wpp_ann.groupby(['Year']).sum().sum(axis=1)

# Load Data: Census
cens = pd.read_csv(Path(resourcefilepath) / "ResourceFile_PopulationSize_2018Census.csv")
cens_2018 = cens.groupby('Sex')['Count'].sum()

# Plot population size over time
plt.plot(model_df['year'], model_df['total'])
plt.plot(wpp_ann_total.index, wpp_ann_total)
plt.plot(2018, cens_2018.sum(), '*')
plt.title("Population Size")
plt.xlabel("Year")
plt.ylabel("Population Size")
plt.gca().set_xlim(2010, 2050)
plt.legend(["Model", "WPP", "Census 2018"])
plt.savefig(outputpath / ("Pop_Over_Time" + datestamp + ".pdf"), format='pdf')
plt.show()

# Population Size in 2018

# Census vs WPP vs Model
wpp_2018 = wpp_ann.groupby(['Year', 'Sex'])['Count'].sum()[2018]

# Get Model totals for males and females in 2018
model_2018 = model_df.loc[model_df['year'] == 2018, ['male', 'female']].reset_index(drop=True).transpose().rename(
    index={'male': 'M', 'female': 'F'})

popsize = pd.concat([cens_2018, wpp_2018, model_2018], keys=['Census_2018', 'WPP', 'Model']).unstack()
popsize.columns = ['Females', 'Males']

# Plot:
popsize.transpose().plot(kind='bar')
plt.title('Population Size (2018)')
plt.xticks(rotation=0)
plt.savefig(outputpath / ("Pop_Size_2018" + datestamp + ".pdf"), format='pdf')
plt.show()

# %% Population Pyramid
# Population Pyramid at two time points

for year in [2018, 2030]:

    # Import and model:
    model_m = scaled_output["tlo.methods.demography"]["age_range_m"]
    model_m = model_m.loc[pd.to_datetime(model_m['date']).dt.year == year].drop(columns=['date']).melt(
        value_name='Model', var_name='Age_Grp')
    model_m.index = model_m['Age_Grp'].astype(make_age_grp_types())
    model_m = model_m.loc[model_m.index.dropna(), 'Model']

    model_f = scaled_output["tlo.methods.demography"]["age_range_f"]
    model_f = model_f.loc[pd.to_datetime(model_f['date']).dt.year == year].drop(columns=['date']).melt(
        value_name='Model', var_name='Age_Grp')
    model_f.index = model_f['Age_Grp'].astype(make_age_grp_types())
    model_f = model_f.loc[model_f.index.dropna(), 'Model']

    # Import and format WPP data:
    wpp_ann = pd.read_csv(Path(resourcefilepath) / "ResourceFile_Pop_Annual_WPP.csv")
    wpp_ann = wpp_ann.loc[wpp_ann['Year'] == year]
    wpp_m = wpp_ann.loc[wpp_ann['Sex'] == 'M', ['Count', 'Age_Grp']].groupby('Age_Grp')['Count'].sum()
    wpp_m.index = wpp_m.index.astype(make_age_grp_types())
    wpp_m = wpp_m.loc[wpp_m.index.dropna()]

    wpp_f = wpp_ann.loc[wpp_ann['Sex'] == 'F', ['Count', 'Age_Grp']].groupby('Age_Grp')['Count'].sum()
    wpp_f.index = wpp_f.index.astype(make_age_grp_types())
    wpp_f = wpp_f.loc[wpp_f.index.dropna()]

    # Make into dataframes for plotting:
    pop_m = pd.DataFrame(model_m)
    pop_m['WPP'] = wpp_m

    pop_f = pd.DataFrame(model_f)
    pop_f['WPP'] = wpp_m

    if year == 2018:
        # Import and format Census data, and add to the comparison if the year is 2018 (year of census)
        cens = pd.read_csv(Path(resourcefilepath) / "ResourceFile_PopulationSize_2018Census.csv")
        cens_m = cens.loc[cens['Sex'] == 'M'].groupby(by='Age_Grp')['Count'].sum()
        cens_m.index = cens_m.index.astype(make_age_grp_types())
        cens_m = cens_m.loc[cens_m.index.dropna()]

        pop_m['Census'] = cens_m
        cens_f = cens.loc[cens['Sex'] == 'F'].groupby(by='Age_Grp')['Count'].sum()
        cens_f.index = cens_f.index.astype(make_age_grp_types())
        cens_f = cens_f.loc[cens_f.index.dropna()]
        pop_f['Census'] = cens_f

    # Simple plot of population pyramid
    fig, axes = plt.subplots(ncols=1, nrows=2, sharey=True)
    pop_m.plot.bar(ax=axes[0], align="center")
    axes[0].set_xlabel('Age Group')
    axes[0].set_title('Males: ' + str(year))
    pop_f.plot.bar(ax=axes[1], align="center")
    axes[1].set_xlabel('Age Group')
    axes[1].set_title('Females: ' + str(year))
    plt.savefig(outputpath / ("Pop_Size_" + str(year) + datestamp + ".pdf"), format='pdf')
    plt.show()

# %% Births: Number over time

# Births over time (Model)
births_model = scaled_output['tlo.methods.demography']['birth_groupby_scaled'].reset_index()
# Aggregate the model outputs into five year periods:
(__tmp__, calendar_period_lookup) = make_calendar_period_lookup()
births_model["Period"] = births_model["year"].map(calendar_period_lookup)
births_model = births_model.groupby(by='Period')['count'].sum()
births_model.index = births_model.index.astype(make_calendar_period_type())

# Births over time (WPP)
wpp = pd.read_csv(Path(resourcefilepath) / "ResourceFile_TotalBirths_WPP.csv")
wpp = wpp.groupby(['Period', 'Variant'])['Total_Births'].sum().unstack()
wpp.index = wpp.index.astype(make_calendar_period_type())
wpp.columns = 'WPP_' + wpp.columns

# Births in 2018 Census
cens = pd.read_csv(Path(resourcefilepath) / "ResourceFile_Births_2018Census.csv")
cens_per_5y_per = cens['Count'].sum() * 5

# Merge in model results
births = wpp.copy()
births['Model'] = births_model
births['Census'] = np.nan
births.at[cens['Period'][0], 'Census'] = cens_per_5y_per

# Plot:
ax = births.plot.line(y=['Model', 'Census', 'WPP_Estimates', 'WPP_Medium variant'])
plt.scatter(x=np.arange(len(births.index))[births.index == [cens['Period'][0]]],
            y=births.loc[cens['Period'][0], 'Census'], marker='^', color='orange')
plt.xticks(np.arange(len(births.index)), births.index)
ax.fill_between(births.index, births['WPP_Low variant'], births['WPP_High variant'], facecolor='red', alpha=0.2)
plt.xticks(rotation=90)
ax.set_title('Number of Births Per Calendar Period')
ax.legend(loc='upper left')
ax.set_xlabel('Calendar Period')
ax.set_ylabel('Number per period')
plt.savefig(outputpath / ("Births_Over_Time_" + datestamp + ".pdf"), format='pdf')
plt.show()


# %% Deaths

# Get Model ouput
deaths_model = scaled_output['tlo.methods.demography']['death_groupby_scaled'].reset_index()

# Aggregate the model outputs into five year periods for age and time:
(__tmp__, calendar_period_lookup) = make_calendar_period_lookup()
deaths_model["Period"] = deaths_model["year"].map(calendar_period_lookup)
(__tmp__, age_grp_lookup) = create_age_range_lookup(min_age=0, max_age=100, range_size=5)
deaths_model["Age_Grp"] = deaths_model["age"].map(age_grp_lookup)

deaths_model = deaths_model.rename(columns={'count': 'Count', 'sex': 'Sex'})
deaths_model = pd.DataFrame(deaths_model.groupby(['Period', 'Sex', 'Age_Grp'])['Count'].sum()).reset_index()
deaths_model['Variant'] = 'Model'

# Load WPP data
wpp = pd.read_csv(Path(resourcefilepath) / "ResourceFile_TotalDeaths_WPP.csv")

# Load GBD
# TODO: Deaths among 0-1 for GBD?
gbd = pd.read_csv(Path(resourcefilepath) / "ResourceFile_TotalDeaths_GBD.csv")
gbd = pd.DataFrame(gbd.drop(columns=['Year']).groupby(by=['Period', 'Sex', 'Age_Grp', 'Variant']).sum()).reset_index()

# Combine into one large dataframe
deaths = pd.concat([deaths_model, wpp, gbd], ignore_index=True, sort=False)
deaths['Age_Grp'] = deaths['Age_Grp'].astype(make_age_grp_types())
deaths['Period'] = deaths['Period'].astype(make_calendar_period_type())

# Total of deaths over time
tot_deaths = pd.DataFrame(deaths.groupby(by=['Period', 'Variant']).sum()).unstack()
tot_deaths.columns = pd.Index([label[1] for label in tot_deaths.columns.tolist()])

# Plot:
ax = tot_deaths.plot(y=['WPP_Estimates', 'WPP_Medium variant', 'GBD_Est', 'Model'])
plt.xticks(np.arange(len(tot_deaths.index)), tot_deaths.index)
ax.fill_between(tot_deaths.index, tot_deaths['WPP_Low variant'], tot_deaths['WPP_High variant'], facecolor='orange',
                alpha=0.5)
ax.fill_between(tot_deaths.index, tot_deaths['GBD_Lower'], tot_deaths['GBD_Upper'], facecolor='green', alpha=0.2)
plt.xticks(rotation=90)
ax.set_title('Number of Deaths Per Calendar Period')
ax.legend(loc='upper left')
ax.set_xlabel('Calendar Period')
ax.set_ylabel('Number per period')
plt.savefig(outputpath / ("Deaths_Over_Time_" + datestamp + ".pdf"), format='pdf')
plt.show()
# NB. Its' expected that WPP range is very narrow (to narrow to see)

# Deaths by age in 2010-2014
period = '2010-2014'

tot_deaths_byage = pd.DataFrame(
    deaths.loc[deaths['Period'] == period].groupby(by=['Variant', 'Age_Grp']).sum()).unstack()
tot_deaths_byage.columns = pd.Index([label[1] for label in tot_deaths_byage.columns.tolist()])
tot_deaths_byage = tot_deaths_byage.transpose()
ax = tot_deaths_byage.plot(y=['WPP_Estimates', 'GBD_Est', 'Model'])
plt.xticks(np.arange(len(tot_deaths_byage.index)), tot_deaths_byage.index)
ax.fill_between(tot_deaths_byage.index, tot_deaths_byage['GBD_Lower'], tot_deaths_byage['GBD_Upper'],
                facecolor='orange', alpha=0.2)
plt.xticks(rotation=90)
ax.set_title('Number of Deaths Per Calendar Period: ' + str(period))
ax.legend(loc='upper left')
ax.set_xlabel('Age Group')
ax.set_ylabel('Number per period')
plt.savefig(outputpath / ("Deaths_By_Age_" + datestamp + ".pdf"), format='pdf')
plt.show()
