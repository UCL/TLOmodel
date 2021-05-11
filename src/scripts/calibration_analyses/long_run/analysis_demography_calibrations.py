"""
Plot to demonstrate correspondence between model and data outputs wrt births, population size and total deaths.

This uses Scenario file: src/scripts/long_run/long_run.py

"""

# TODO -- When the long-runs work, finish off converting this script to use the results from the batchrun system

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

from tlo.methods import demography
from tlo.util import create_age_range_lookup

# Declare usual paths:
outputspath = Path('./outputs/tbh03@ic.ac.uk')
rfp = Path('./resources')

# ** Declare the results folder ***
results_folder = get_scenario_outputs('long_run.py', outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: outputspath / f"{datetime.today().strftime('%Y_%m_%d''')}_{stub}.png"

# %% Examine the results folder:

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder, draw=0, run=0)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
# params = extract_params(results_folder)



# %% Population Size
# Trend in Number Over Time

# 1) Population Growth Over Time:

# Load and format model results (with year as integer:
pop_model = summarize(extract_results(results_folder,
                            module="tlo.methods.demography",
                            key="population",
                            column="total",
                            index="date"),
                      collapse_columns=True
                      )
pop_model.index = pop_model.index.year

# Load Data: WPP_Annual
wpp_ann = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_Pop_Annual_WPP.csv")
wpp_ann['Age_Grp'] = wpp_ann['Age_Grp'].astype(make_age_grp_types())

wpp_ann_total = wpp_ann.groupby(['Year']).sum().sum(axis=1)

# Load Data: Census
cens = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_PopulationSize_2018Census.csv")
cens['Age_Grp'] = cens['Age_Grp'].astype(make_age_grp_types())

cens_2018 = cens.groupby('Sex')['Count'].sum()

# Work out the scaling-factor (using mean in case sampling is ever greater than once per year):
mean_pop_2018 = pop_model.loc[pop_model.index == 2018, 'mean'].mean()
sf = cens_2018.sum() / mean_pop_2018

# Update the model results to incorporate the scaling factor
pop_model *= sf

# Plot population size over time
plt.plot(pop_model.index, pop_model['mean'])
plt.fill_between(pop_model.index,
                 pop_model['lower'],
                 pop_model['upper'],
                 color='blue',
                 alpha=0.5
                 )
plt.plot(wpp_ann_total.index, wpp_ann_total)
plt.plot(2018, cens_2018.sum(), '*')
plt.title("Population Size")
plt.xlabel("Year")
plt.ylabel("Population Size")
plt.gca().set_xlim(2010, 2050)
plt.legend(["Model", "WPP", "Census 2018"])
plt.savefig(make_graph_file_name("Pop_Over_Time"))
plt.show()


# 2) Population Size in 2018 (broken down by Male and Female)

# Census vs WPP vs Model
wpp_2018 = wpp_ann.groupby(['Year', 'Sex'])['Count'].sum()[2018]

# Get Model totals for males and females in 2018 (with scaling factor)
pop_model_male = summarize(extract_results(results_folder,
                            module="tlo.methods.demography",
                            key="population",
                            column="male",
                            index="date"),
                      collapse_columns=True
                      ).mul(sf)
pop_model_male.index = pop_model_male.index.year

pop_model_female = summarize(extract_results(results_folder,
                            module="tlo.methods.demography",
                            key="population",
                            column="female",
                            index="date"),
                      collapse_columns=True
                      ).mul(sf)
pop_model_female.index = pop_model_female.index.year

pop_2018 = {
    'Census': cens_2018,
    'WPP': wpp_2018,
    'Model': {
        'F': pop_model_female.loc[2018, 'mean'],
        'M': pop_model_male.loc[2018, 'mean']
    }
}

pop_2018 = pd.DataFrame(pop_2018)

# Plot:
pop_2018.plot(kind='bar')
plt.title('Population Size (2018)')
plt.xticks(rotation=0)
plt.savefig(make_graph_file_name("Pop_Size_2018"))
plt.show()

# %% Population Pyramid
# Population Pyramid at two time points

# Get Age/Sex Breakdown of population (with scaling)

calperiods, calperiodlookup = make_calendar_period_lookup()

def get_mean_pop_by_age_for_sex_and_year(sex, year):
    if sex == 'F':
        key = "age_range_f"
    else:
        key = "age_range_f"

    agegroups = list(make_age_grp_types().categories)
    output = dict()
    for agegroup in agegroups:
        num = summarize(extract_results(results_folder,
                                module="tlo.methods.demography",
                                key=key,
                                column=agegroup,
                                index="date"),
                  collapse_columns=True,
                  only_mean=True
                  )
        output[agegroup] = num.loc[num.index.year == year].values.mean()
    return pd.Series(output)



for year in [2018, 2030]:

    # Get WPP data:
    wpp_thisyr = wpp_ann.loc[wpp_ann['Year'] == year].groupby(['Sex', 'Age_Grp'])['Count'].sum()

    pops = dict()
    for sex in ['M', 'F']:
        # Import model results and scale:
        model = get_mean_pop_by_age_for_sex_and_year(sex, year).mul(sf)

        # Make into dataframes for plotting:
        pops[sex] = {
            'Model': model,
            'WPP':  wpp_thisyr.loc[sex]
        }

        if year == 2018:
            # Import and format Census data, and add to the comparison if the year is 2018 (year of census)
            pops[sex]['2018 Census'] = cens.loc[cens['Sex'] == sex].groupby(by='Age_Grp')['Count'].sum()


    # Simple plot of population pyramid
    fig, axes = plt.subplots(ncols=1, nrows=2, sharey=True)
    pd.DataFrame(pops['M']).plot.bar(ax=axes[0], align="center")
    axes[0].set_xlabel('Age Group')
    axes[0].set_title('Males: ' + str(year))
    pd.DataFrame(pops['F']).plot.bar(ax=axes[1], align="center")
    axes[1].set_xlabel('Age Group')
    axes[1].set_title('Females: ' + str(year))
    plt.savefig(make_graph_file_name(f"Pop_Size_{year}"))
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
