"""
Plot to demonstrate correspondence between model and data outputs wrt births, population size and total deaths.

This uses Scenario file: src/scripts/long_run/long_run.py

"""
# TODO GET SCALING FACTOR FROM INSIDE SIM LOG

# TODO -- Coding -- ordering of each element on the plot to get the consistent pattern of overlay;


import pickle
from datetime import datetime
from pathlib import Path
from pytest import approx
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
    format_gbd,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize
)

from tlo.methods import demography
from tlo.util import create_age_range_lookup
from matplotlib.ticker import FormatStrFormatter

# Declare usual paths:
outputspath = Path('./outputs/tbh03@ic.ac.uk')
rfp = Path('./resources')

# ** Declare the results folder ***
results_folder = get_scenario_outputs('long_run.py', outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: outputspath / f"{datetime.today().strftime('%Y_%m_%d''')}_{stub}.png"

# Define colo(u)rs to use:
colors = {
    'Model': 'royalblue',
    'Census': 'darkred',
    'WPP': 'forestgreen',
    'GBD': 'plum'
}

# %% Examine the results folder:

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations (will report that no parameters changed)
params = extract_params(results_folder)

# %% Population Size
# Trend in Number Over Time

# 1) Population Growth Over Time:

# Load and format model results (with year as integer):
pop_model = summarize(extract_results(results_folder,
                                      module="tlo.methods.demography",
                                      key="population",
                                      column="total",
                                      index="date",
                                      do_scaling=True
                                      ),
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

# Plot population size over time
fig, ax = plt.subplots()
ax.plot(wpp_ann_total.index, wpp_ann_total/1e6,
        label='WPP', color=colors['WPP'])
ax.plot(2018.5, cens_2018.sum()/1e6,
        marker='o', markersize=10, linestyle='none', label='Census', zorder=10, color=colors['Census'])
ax.plot(pop_model.index, pop_model['mean']/1e6,
        label='Model (mean)', color=colors['Model'])
ax.fill_between(pop_model.index,
                 pop_model['lower']/1e6,
                 pop_model['upper']/1e6,
                 color=colors['Model'],
                 alpha=0.2,
                 zorder=5
                 )
ax.set_title("Population Size 2010-2030")
ax.set_xlabel("Year")
ax.set_ylabel("Population Size (millions)")
ax.set_xlim(2010, 2030)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_ylim(0, 30)
ax.legend()
fig.tight_layout()
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
                            index="date",
                                           do_scaling=True),
                      collapse_columns=True
                      )
pop_model_male.index = pop_model_male.index.year

pop_model_female = summarize(extract_results(results_folder,
                            module="tlo.methods.demography",
                            key="population",
                            column="female",
                            index="date",
                                             do_scaling=True),
                      collapse_columns=True
                      )
pop_model_female.index = pop_model_female.index.year

pop_2018 = {
    'Census': cens_2018,
    'WPP': wpp_2018,
    'Model': pd.Series({
        'F': pop_model_female.loc[2018, 'mean'],
        'M': pop_model_male.loc[2018, 'mean']
    })
}


# Plot:
labels = ['F', 'M']
sexname = lambda x: 'Females' if x=='F' else 'Males'
width = 0.2
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()
for i, key in enumerate(pop_2018):
    ax.bar(x=x + (i-1)*width*1.05, height=[pop_2018[key][sex]/1e6 for sex in labels],
           width=width,
           label=key,
           color=colors[key]
           )
ax.set_xticks(x)
ax.set_xticklabels([sexname(sex) for sex in labels])
ax.set_ylabel('Sex')
ax.set_ylabel('Population Size (millions)')
ax.set_ylim(0, 10)
ax.set_title('Population Size 2018')
ax.legend()
fig.tight_layout()
plt.savefig(make_graph_file_name("Pop_Over_Time"))
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
                                index="date",
                                        do_scaling=True),
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
        model = get_mean_pop_by_age_for_sex_and_year(sex, year)

        # Make into dataframes for plotting:
        pops[sex] = {
            'Model': model,
            'WPP':  wpp_thisyr.loc[sex]
        }

        if year == 2018:
            # Import and format Census data, and add to the comparison if the year is 2018 (year of census)
            pops[sex]['Census'] = cens.loc[cens['Sex'] == sex].groupby(by='Age_Grp')['Count'].sum()


    # Simple plot of population pyramid
    fig, axes = plt.subplots(ncols=1, nrows=2, sharey=True, sharex=True)
    labels = ['F', 'M']
    width = 0.2
    x = np.arange(len(list(make_age_grp_types().categories)))  # the label locations
    for i, sex in enumerate(labels):
        for t, key in enumerate(pops[sex]):
            axes[i].bar(x=x + (t-1)*width*1.2, label=key, height=pops[sex][key]/1e6, color=colors[key], width=width)
        axes[i].set_title(f"{sexname(sex)}: {str(year)}")
        axes[i].set_ylabel('Population Size (millions)')

    axes[1].set_xlabel('Age Group')
    plt.xticks(x, list(make_age_grp_types().categories), rotation=90)
    axes[1].legend()
    fig.tight_layout()
    plt.savefig(make_graph_file_name(f"Pop_Size_{year}"))
    plt.show()




# %% Births: Number over time

# Births over time (Model)
births_by_year = summarize(extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="on_birth",
    custom_generate_series="assign(year = lambda x: x['date'].dt.year)"
                           ".groupby(['year'])['year'].count()",
    do_scaling=True
),
    collapse_columns=True
)

# Aggregate the model outputs into five year periods:
calperiods, calperiodlookup = make_calendar_period_lookup()
births_by_year['Period'] = births_by_year.index.map(calperiodlookup)
births_model = births_by_year.loc[births_by_year.index < 2030].groupby(by='Period').sum()
births_model.index = births_model.index.astype(make_calendar_period_type())
births_model.columns = ['Model_' + col for col in births_model.columns]

# Births over time (WPP)
wpp_births = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_TotalBirths_WPP.csv")
wpp_births = wpp_births.groupby(['Period', 'Variant'])['Total_Births'].sum().unstack()
wpp_births.index = wpp_births.index.astype(make_calendar_period_type())
wpp_births.columns = 'WPP_' + wpp_births.columns

# Make the WPP line connect to the 'medium variant' to make the lines look like they join up
wpp_births['WPP_continuous'] = wpp_births['WPP_Estimates'].combine_first(wpp_births['WPP_Medium variant'])


# Births in 2018 Census
cens_births = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_Births_2018Census.csv")
cens_births_per_5y_per = cens_births['Count'].sum() * 5

# Merge in model results
births = wpp_births.merge(births_model, right_index=True, left_index=True, how='left')
births['Census'] = np.nan
births.at[cens['Period'][0], 'Census'] = cens_births_per_5y_per


# Limit births.index between 2010 and 2030
time_period = {
    '1950-2099': births.index,
    '2010-2030':  [(2010 <= int(x[0])) &  (int(x[1]) < 2030) for x in births.index.str.split('-')]
}

# Plot:
for tp in time_period:
    births_loc = births.loc[time_period[tp]]
    fig, ax = plt.subplots()
    ax.plot()
    ax.plot(
        births_loc.index,
        births_loc['Census']/1e6,
        linestyle='none', marker='o', markersize=10, label='Census', zorder=10, color=colors['Census'])
    ax.plot(
        births_loc.index,
        births_loc['Model_mean']/1e6,
        label='Model',
        color=colors['Model']
    )
    ax.fill_between(births_loc.index,
                    births_loc['Model_lower']/1e6,
                    births_loc['Model_upper']/1e6,
                    facecolor=colors['Model'], alpha=0.2)
    ax.plot(
        births_loc.index,
        births_loc['WPP_continuous']/1e6,
        color=colors['WPP'],
        label='WPP'
    )
    ax.fill_between(births_loc.index,
                    births_loc['WPP_Low variant']/1e6,
                    births_loc['WPP_High variant']/1e6,
                    facecolor=colors['WPP'], alpha=0.2)
    ax.legend(loc='upper left')
    plt.xticks(rotation=90)
    ax.set_title(f"Number of Births {tp}")
    ax.set_xlabel('Calendar Period')
    ax.set_ylabel('Births per period (millions)')
    ax.set_ylim(0, 8.0)
    plt.xticks(np.arange(len(births_loc.index)), births_loc.index)
    plt.tight_layout()
    plt.savefig(make_graph_file_name(f"Births_Over_Time_{tp}"))
    plt.show()

# %% All-Cause Deaths

# Get Model ouput (aggregating by year before doing the summarize)
deaths_by_age_and_year = summarize(extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series="assign(year = lambda x: x['date'].dt.year)"
                           ".groupby(['sex', 'year', 'age'])['person_id'].count()",
    do_scaling=True
),
    collapse_columns=True
).reset_index()

# Aggregate the model outputs into five year periods for age and time:
calperiods, calperiodlookup = make_calendar_period_lookup()
deaths_by_age_and_year["Period"] = deaths_by_age_and_year["year"].map(calperiodlookup)

(__tmp__, age_grp_lookup) = create_age_range_lookup(min_age=0, max_age=100, range_size=5)
deaths_by_age_and_year["Age_Grp"] = deaths_by_age_and_year["age"].map(age_grp_lookup)

deaths_by_age_and_year =deaths_by_age_and_year.rename(columns={'sex': 'Sex'})

deaths_model = pd.DataFrame(deaths_by_age_and_year.loc[deaths_by_age_and_year["year"] < 2030].groupby(['Period', 'Sex', 'Age_Grp']).sum()).reset_index()
deaths_model = deaths_model.melt(
    id_vars=['Period', 'Sex', 'Age_Grp'], value_vars=['mean', 'lower', 'upper'], var_name='Variant' ,value_name='Count')
deaths_model['Variant'] = 'Model_' + deaths_model['Variant']

# Load WPP data
wpp_deaths = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_TotalDeaths_WPP.csv")

# Load GBD
gbd = format_gbd(pd.read_csv(rfp / "demography" / "ResourceFile_TotalDeaths_GBD.csv"))

# Compute sums by period
gbd = pd.DataFrame(gbd.drop(columns=['Year']).groupby(by=['Period', 'Sex', 'Age_Grp', 'Variant']).sum()).reset_index()

# Combine into one large dataframe
deaths = pd.concat([deaths_model, wpp_deaths, gbd], ignore_index=True, sort=False)
deaths['Age_Grp'] = deaths['Age_Grp'].astype(make_age_grp_types())
deaths['Period'] = deaths['Period'].astype(make_calendar_period_type())

# Total of deaths over time (summing age/sex)
tot_deaths = pd.DataFrame(deaths.groupby(by=['Period', 'Variant']).sum()).unstack()
tot_deaths.columns = pd.Index([label[1] for label in tot_deaths.columns.tolist()])
tot_deaths = tot_deaths.replace(0, np.nan)

# Make the WPP line connect to the 'medium variant' to make the lines look like they join up
tot_deaths['WPP_continuous'] = tot_deaths['WPP_Estimates'].combine_first(tot_deaths['WPP_Medium variant'])

# Plot:
fig, ax = plt.subplots()
ax.plot(
    tot_deaths.index,
    tot_deaths['WPP_continuous']/1e6,
    label='WPP',
    color=colors['WPP'])
ax.fill_between(
    tot_deaths.index,
    tot_deaths['WPP_Low variant']/1e6,
    tot_deaths['WPP_High variant']/1e6,
    facecolor=colors['WPP'], alpha=0.2)
ax.plot(
    tot_deaths.index,
    tot_deaths['GBD_Est']/1e6,
    label='GBD',
    color=colors['GBD']
)
ax.fill_between(
    tot_deaths.index,
    tot_deaths['GBD_Lower']/1e6,
    tot_deaths['GBD_Upper']/1e6,
    facecolor=colors['GBD'], alpha=0.2)
ax.plot(
    tot_deaths.index,
    tot_deaths['Model_mean']/1e6,
    label='Model',
    color=colors['Model']
)
ax.fill_between(
    tot_deaths.index,
    tot_deaths['Model_lower']/1e6,
    tot_deaths['Model_upper']/1e6,
    facecolor=colors['Model'], alpha=0.2)

ax.set_title('Number of Deaths')
ax.legend(loc='upper left')
ax.set_xlabel('Calendar Period')
ax.set_ylabel('Number per period (millions)')
plt.xticks(np.arange(len(tot_deaths.index)), tot_deaths.index, rotation=90)
fig.tight_layout()
plt.savefig(make_graph_file_name("Deaths_OverTime"))
plt.show()


# Deaths by age, during in selection periods
calperiods_selected = list()
for cal in calperiods:
    if cal != '2100+':
        if (2010 <= int(cal.split('-')[0])) and (int(cal.split('-')[1]) < 2030):
            calperiods_selected.append(cal)

for period in calperiods_selected:

    fig, ax = plt.subplots(ncols=1, nrows=2, sharey=True, sharex=True)
    for i, sex in enumerate(['F', 'M']):

        tot_deaths_byage = pd.DataFrame(
            deaths.loc[(deaths['Period'] == period) & (deaths['Sex'] == sex)].groupby(by=['Variant', 'Age_Grp']).sum()).unstack()
        tot_deaths_byage.columns = pd.Index([label[1] for label in tot_deaths_byage.columns.tolist()])
        tot_deaths_byage = tot_deaths_byage.transpose()

        if 'WPP_Medium variant' in tot_deaths_byage.columns:
            ax[i].plot(
                tot_deaths_byage.index,
                tot_deaths_byage['WPP_Medium variant']/1e3,
                label='WPP',
                color=colors['WPP'])
            ax[i].fill_between(
                tot_deaths_byage.index,
                tot_deaths_byage['WPP_Low variant']/1e3,
                tot_deaths_byage['WPP_High variant']/1e3,
                facecolor=colors['WPP'], alpha=0.2)
        else:
            ax[i].plot(
                tot_deaths_byage.index,
                tot_deaths_byage['WPP_Estimates']/1e3,
                label='WPP',
                color=colors['WPP'])

        if 'GBD_Est' in tot_deaths_byage.columns:
            ax[i].plot(
                tot_deaths_byage.index,
                tot_deaths_byage['GBD_Est']/1e3,
                label='GBD',
                color=colors['GBD'])
            ax[i].fill_between(
                tot_deaths_byage.index,
                tot_deaths_byage['GBD_Lower']/1e3,
                tot_deaths_byage['GBD_Upper']/1e3,
                facecolor=colors['GBD'], alpha=0.2)

        ax[i].plot(
            tot_deaths_byage.index,
            tot_deaths_byage['Model_mean']/1e3,
            label='Model',
            color=colors['Model'])
        ax[i].fill_between(
            tot_deaths_byage.index,
            tot_deaths_byage['Model_lower']/1e3,
            tot_deaths_byage['Model_upper']/1e3,
            facecolor=colors['Model'], alpha=0.2)

        ax[i].set_xticks(np.arange(len(tot_deaths_byage.index)))
        ax[i].set_xticklabels(tot_deaths_byage.index, rotation=90)
        ax[i].set_title(f"Number of Deaths {period}: {sexname(sex)}")
        ax[i].legend(loc='upper left')
        ax[i].set_xlabel('Age Group')
        ax[i].set_ylabel('Deaths per period (thousands)')
        ax[i].set_ylim(0, 80)

    fig.tight_layout()
    plt.savefig(make_graph_file_name(f"Deaths_By_Age_{period}"))
    plt.show()
