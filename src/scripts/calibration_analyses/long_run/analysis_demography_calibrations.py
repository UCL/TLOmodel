"""
Plot to demonstrate correspondence between model and data outputs wrt births, population size and total deaths.

This uses Scenario file: src/scripts/long_run/long_run.py

"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    parse_log_file,
)
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize
)

from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)


from tlo.methods import demography
from tlo.util import create_age_range_lookup

outputspath = Path('./outputs')
results_folder = get_scenario_outputs('long_run.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder, draw=0, run=3)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# TODO -- When the long-runs work, finish off converting this script to use the results from the batchrun system

#     # We register all modules in a single call to the register method, calling once with multiple
#     # objects. This is preferred to registering each module in multiple calls because we will be
#     # able to handle dependencies if modules are registered together
#     sim.register(
#         demography.Demography(resourcefilepath=resources),
#         enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
#         healthsystem.HealthSystem(resourcefilepath=resources, disable=True),
#         symptommanager.SymptomManager(resourcefilepath=resources),
#         healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
#         healthburden.HealthBurden(resourcefilepath=resources),
#         contraception.Contraception(resourcefilepath=resources),
#         care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resources),
#         pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
#         labour.Labour(resourcefilepath=resources),
#         newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
#         postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resources),
#     )
# # 2) Extract a specific log series for all runs:
# extracted = extract_results(results_folder,
#                             module="tlo.methods.mockitis",
#                             key="summary",
#                             column="PropInf",
#                             index="date")


# %% Filename etc

# Resource file path
rfp = Path("./resources")

# Where will outputs be found
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / '2020_11_25_long_run.pickle'

with open(results_filename, 'rb') as f:
    output = pickle.load(f)['output']

make_file_name = lambda stub: outputpath / f"{datetime.today().strftime('%Y_%m_%d''')}_{stub}.png"

# %% read the results and do the scaling:
scaled_output = demography.scale_to_population(output, rfp)

# %% Population Size
# Trend in Number Over Time

# Population Growth Over Time:
# Load and format model results
model_df = scaled_output["tlo.methods.demography"]["population"]
model_df['year'] = pd.to_datetime(model_df.date).dt.year

# Load Data: WPP_Annual
wpp_ann = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_Pop_Annual_WPP.csv")
wpp_ann_total = wpp_ann.groupby(['Year']).sum().sum(axis=1)

# Load Data: Census
cens = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_PopulationSize_2018Census.csv")
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
plt.savefig(make_file_name("Pop_Over_Time"))
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
plt.savefig(make_file_name("Pop_Size_2018"))
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
    wpp_ann = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_Pop_Annual_WPP.csv")
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
        cens = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_PopulationSize_2018Census.csv")
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
    plt.savefig(make_file_name(f"Pop_Size_{year}"))
    plt.show()

# %% Births: Number over time

# Births over time (Model)
births_model = scaled_output['tlo.methods.demography']['birth_groupby_scaled'].reset_index()
# Aggregate the model outputs into five year periods:
(__tmp__, calendar_period_lookup) = make_calendar_period_lookup()
births_model["Period"] = births_model["year"].map(calendar_period_lookup)
births_model = births_model.loc[births_model.year < 2030].groupby(by='Period')['count'].sum()
births_model.index = births_model.index.astype(make_calendar_period_type())

# Births over time (WPP)
wpp = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_TotalBirths_WPP.csv")
wpp = wpp.groupby(['Period', 'Variant'])['Total_Births'].sum().unstack()
wpp.index = wpp.index.astype(make_calendar_period_type())
wpp.columns = 'WPP_' + wpp.columns

# Births in 2018 Census
cens = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_Births_2018Census.csv")
cens_per_5y_per = cens['Count'].sum() * 5

# Merge in model results
births = wpp.copy()
births['Model'] = births_model
births['Census'] = np.nan
births.at[cens['Period'][0], 'Census'] = cens_per_5y_per

# Plot:
cens_period = cens['Period'][0]
ax = births.plot.line(y=['Model',  'WPP_Estimates', 'WPP_Medium variant'])
births.plot.line(
    y=['Census'],
    marker='^',
    color='red',
    ax=ax
)
plt.xticks(np.arange(len(births.index)), births.index)
ax.fill_between(births.index, births['WPP_Low variant'], births['WPP_High variant'], facecolor='green', alpha=0.2)
plt.xticks(rotation=90)
ax.set_title('Number of Births Per Calendar Period')
ax.legend(loc='upper left')
ax.set_xlabel('Calendar Period')
ax.set_ylabel('Number per period')
plt.savefig(make_file_name("Births_Over_Time"))
plt.tight_layout()
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
deaths_model = pd.DataFrame(deaths_model.loc[deaths_model.year < 2030].groupby(['Period', 'Sex', 'Age_Grp'])['Count'].sum()).reset_index()
deaths_model['Variant'] = 'Model'

# Load WPP data
wpp = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_TotalDeaths_WPP.csv")

# Load GBD
gbd = pd.read_csv(rfp / "demography" / "ResourceFile_TotalDeaths_GBD.csv")
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
plt.savefig(make_file_name("Deaths_OverTime"))
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
plt.savefig(make_file_name("Deaths_By_Age"))
plt.show()
