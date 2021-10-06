"""
Plot to demonstrate correspondence between model and data outputs wrt births, population size and total deaths.

This uses the results of the Scenario defined in: src/scripts/long_run/long_run.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    format_gbd,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    summarize,
)

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'long_run_no_diseases.py'  # <-- update this to look at other results

# %% Declare usual paths:
outputspath = Path('./outputs/tbh03@ic.ac.uk')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
print(f"Results folder is: {results_folder}")

# If needed -- in the case that pickles were not created remotely during batch
# create_pickles_locally(results_folder)

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# Define colo(u)rs to use:
colors = {
    'Model': 'royalblue',
    'Census': 'darkred',
    'WPP': 'forestgreen',
    'GBD': 'plum'
}

# Define how to call the sexes:
sexname = lambda x: 'Females' if x == 'F' else 'Males'  # noqa: E731

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
ax.plot(wpp_ann_total.index, wpp_ann_total / 1e6,
        label='WPP', color=colors['WPP'])
ax.plot(2018.5, cens_2018.sum() / 1e6,
        marker='o', markersize=10, linestyle='none', label='Census', zorder=10, color=colors['Census'])
ax.plot(pop_model.index, pop_model['mean'] / 1e6,
        label='Model (mean)', color=colors['Model'])
ax.fill_between(pop_model.index,
                pop_model['lower'] / 1e6,
                pop_model['upper'] / 1e6,
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

width = 0.2
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()
for i, key in enumerate(pop_2018):
    ax.bar(x=x + (i - 1) * width * 1.05, height=[pop_2018[key][sex] / 1e6 for sex in labels],
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

df = log['tlo.methods.demography']['age_range_f']
y = df.loc[lambda x: pd.to_datetime(x.date).dt.year == 2010].drop(columns=['date']).melt(var_name='age_grp').set_index(
    'age_grp')['value']


def get_mean_pop_by_age_for_sex_and_year(sex, year):
    if sex == 'F':
        key = "age_range_f"
    else:
        key = "age_range_f"

    num_by_age = summarize(
        extract_results(results_folder,
                        module="tlo.methods.demography",
                        key=key,
                        custom_generate_series=(
                            lambda df_: df_.loc[pd.to_datetime(df_.date).dt.year == 2010].drop(
                                columns=['date']
                            ).melt(
                                var_name='age_grp'
                            ).set_index('age_grp')['value']
                        ),
                        do_scaling=True
                        ),
        collapse_columns=True,
        only_mean=True
    )
    return num_by_age


for year in [2018, 2029]:

    # Get WPP data:
    wpp_thisyr = wpp_ann.loc[wpp_ann['Year'] == year].groupby(['Sex', 'Age_Grp'])['Count'].sum()

    pops = dict()
    for sex in ['M', 'F']:
        # Import model results and scale:
        model = get_mean_pop_by_age_for_sex_and_year(sex, year)

        # Make into dataframes for plotting:
        pops[sex] = {
            'Model': model,
            'WPP': wpp_thisyr.loc[sex]
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
            axes[i].bar(x=x + (t - 1) * width * 1.2, label=key, height=pops[sex][key] / 1e6, color=colors[key],
                        width=width)
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
births_results = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="on_birth",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
    ),
    do_scaling=True
)

# zero-out the valyes for 2030 (because the current mdoel run includes 2030 but this skews 5 years averages)
births_results.loc[2030] = 0

# Aggregate the model outputs into five year periods:
calperiods, calperiodlookup = make_calendar_period_lookup()
births_results.index = births_results.index.map(calperiodlookup).astype(make_calendar_period_type())
births_results = births_results.groupby(by=births_results.index).sum()
births_results = births_results.replace({0: np.nan})

# Produce summary of results:
births_model = summarize(births_results, collapse_columns=True)
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
    '2010-2030': [(2010 <= int(x[0])) & (int(x[1]) < 2030) for x in births.index.str.split('-')]
}

# Plot:
for tp in time_period:
    births_loc = births.loc[time_period[tp]]
    fig, ax = plt.subplots()
    ax.plot()
    ax.plot(
        births_loc.index,
        births_loc['Census'] / 1e6,
        linestyle='none', marker='o', markersize=10, label='Census', zorder=10, color=colors['Census'])
    ax.plot(
        births_loc.index,
        births_loc['Model_mean'] / 1e6,
        label='Model',
        color=colors['Model']
    )
    ax.fill_between(births_loc.index,
                    births_loc['Model_lower'] / 1e6,
                    births_loc['Model_upper'] / 1e6,
                    facecolor=colors['Model'], alpha=0.2)
    ax.plot(
        births_loc.index,
        births_loc['WPP_continuous'] / 1e6,
        color=colors['WPP'],
        label='WPP'
    )
    ax.fill_between(births_loc.index,
                    births_loc['WPP_Low variant'] / 1e6,
                    births_loc['WPP_High variant'] / 1e6,
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


# %% Describe patterns of contraceptive usage over time

def get_annual_mean_usage(_df):
    _x = _df \
        .assign(year=df['date'].dt.year) \
        .set_index('year') \
        .drop(columns=['date']) \
        .apply(lambda row: row / row.sum(),
               axis=1
               )
    return _x.groupby(_x.index).mean().stack()


mean_usage = summarize(extract_results(results_folder,
                                       module="tlo.methods.contraception",
                                       key="contraception_use_summary",
                                       custom_generate_series=get_annual_mean_usage,
                                       do_scaling=False),
                       collapse_columns=True
                       )

# Plot just the means:
mean_usage_mean = mean_usage['mean'].unstack()
fig, ax = plt.subplots()
spacing = (np.arange(len(mean_usage_mean)) % 5) == 0
mean_usage_mean.loc[spacing].plot.bar(stacked=True, ax=ax, legend=False)
plt.title('Proportion Females 15-49 Using Contraceptive Methods')
plt.xlabel('Date')
plt.ylabel('Proportion')

fig.legend(loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.65)
plt.savefig(make_graph_file_name("Contraception"))
plt.show()

# %% All-Cause Deaths
#  todo - fix this ;only do summarize after the groupbys
#  Get Model ouput (aggregating by year before doing the summarize)

results_deaths = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['sex', 'year', 'age'])['person_id'].count()
    ),
    do_scaling=True
)

# Aggregate the model outputs into five year periods for age and time:
agegrps, agegrplookup = make_age_grp_lookup()
calperiods, calperiodlookup = make_calendar_period_lookup()

results_deaths = results_deaths.reset_index()
results_deaths['Age_Grp'] = results_deaths['age'].map(agegrplookup).astype(make_age_grp_types())
results_deaths['Period'] = results_deaths['year'].map(calperiodlookup).astype(make_calendar_period_type())
results_deaths = results_deaths.rename(columns={'sex': 'Sex'})
results_deaths = results_deaths.drop(columns=['age', 'year']).groupby(['Period', 'Sex', 'Age_Grp']).sum()

# Load WPP data
wpp_deaths = pd.read_csv(Path(rfp) / "demography" / "ResourceFile_TotalDeaths_WPP.csv")
wpp_deaths['Period'] = wpp_deaths['Period'].astype(make_calendar_period_type())
wpp_deaths['Age_Grp'] = wpp_deaths['Age_Grp'].astype(make_age_grp_types())

# Load GBD
gbd_deaths = format_gbd(pd.read_csv(rfp / "gbd" / "ResourceFile_TotalDeaths_GBD2019.csv"))

# For GBD, compute sums by period
gbd_deaths = pd.DataFrame(
    gbd_deaths.drop(columns=['Year']).groupby(by=['Period', 'Sex', 'Age_Grp', 'Variant']).sum()).reset_index()

# 1) Plot deaths over time (all ages)

# Summarize model results (for all ages) and process into desired format:
deaths_model_by_period = summarize(results_deaths.sum(level=0), collapse_columns=True).reset_index()
deaths_model_by_period = deaths_model_by_period.melt(
    id_vars=['Period'], value_vars=['mean', 'lower', 'upper'], var_name='Variant', value_name='Count')
deaths_model_by_period['Variant'] = 'Model_' + deaths_model_by_period['Variant']

# Sum WPP and GBD to give total deaths by period
wpp_deaths_byperiod = wpp_deaths.groupby(by=['Variant', 'Period'])['Count'].sum().reset_index()
gbd_deaths_by_period = gbd_deaths.groupby(by=['Variant', 'Period'])['Count'].sum().reset_index()

# Combine into one large dataframe
deaths_by_period = pd.concat(
    [deaths_model_by_period,
     wpp_deaths_byperiod,
     gbd_deaths_by_period
     ],
    ignore_index=True, sort=False
)

deaths_by_period['Period'] = deaths_by_period['Period'].astype(make_calendar_period_type())

# Total of deaths over time (summing age/sex)
deaths_by_period = pd.DataFrame(deaths_by_period.groupby(by=['Period', 'Variant']).sum()).unstack()
deaths_by_period.columns = pd.Index([label[1] for label in deaths_by_period.columns.tolist()])
deaths_by_period = deaths_by_period.replace(0, np.nan)

# Make the WPP line connect to the 'medium variant' to make the lines look like they join up
deaths_by_period['WPP_continuous'] = deaths_by_period['WPP_Estimates'].combine_first(
    deaths_by_period['WPP_Medium variant'])

# Plot:
fig, ax = plt.subplots()
ax.plot(
    deaths_by_period.index,
    deaths_by_period['WPP_continuous'] / 1e6,
    label='WPP',
    color=colors['WPP'])
ax.fill_between(
    deaths_by_period.index,
    deaths_by_period['WPP_Low variant'] / 1e6,
    deaths_by_period['WPP_High variant'] / 1e6,
    facecolor=colors['WPP'], alpha=0.2)
ax.plot(
    deaths_by_period.index,
    deaths_by_period['GBD_Est'] / 1e6,
    label='GBD',
    color=colors['GBD']
)
ax.fill_between(
    deaths_by_period.index,
    deaths_by_period['GBD_Lower'] / 1e6,
    deaths_by_period['GBD_Upper'] / 1e6,
    facecolor=colors['GBD'], alpha=0.2)
ax.plot(
    deaths_by_period.index,
    deaths_by_period['Model_mean'] / 1e6,
    label='Model',
    color=colors['Model']
)
ax.fill_between(
    deaths_by_period.index,
    deaths_by_period['Model_lower'] / 1e6,
    deaths_by_period['Model_upper'] / 1e6,
    facecolor=colors['Model'], alpha=0.2)

ax.set_title('Number of Deaths')
ax.legend(loc='upper left')
ax.set_xlabel('Calendar Period')
ax.set_ylabel('Number per period (millions)')
plt.xticks(np.arange(len(deaths_by_period.index)), deaths_by_period.index, rotation=90)
fig.tight_layout()
plt.savefig(make_graph_file_name("Deaths_OverTime"))
plt.show()

# 2) Plots by sex and age-group for selected period:

# Summarize model results (with breakdown by age/sex/period) and process into desired format:
deaths_model_by_agesexperiod = summarize(results_deaths, collapse_columns=True).reset_index()
deaths_model_by_agesexperiod = deaths_model_by_agesexperiod.melt(
    id_vars=['Period', 'Sex', 'Age_Grp'], value_vars=['mean', 'lower', 'upper'], var_name='Variant', value_name='Count')
deaths_model_by_agesexperiod['Variant'] = 'Model_' + deaths_model_by_agesexperiod['Variant']

# Combine into one large dataframe
deaths_by_agesexperiod = pd.concat(
    [deaths_model_by_agesexperiod,
     wpp_deaths,
     gbd_deaths
     ],
    ignore_index=True, sort=False
)

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
            deaths_by_agesexperiod.loc[
                (deaths_by_agesexperiod['Period'] == period) & (deaths_by_agesexperiod['Sex'] == sex)].groupby(
                by=['Variant', 'Age_Grp']).sum()).unstack()
        tot_deaths_byage.columns = pd.Index([label[1] for label in tot_deaths_byage.columns.tolist()])
        tot_deaths_byage = tot_deaths_byage.transpose()

        if 'WPP_Medium variant' in tot_deaths_byage.columns:
            ax[i].plot(
                tot_deaths_byage.index,
                tot_deaths_byage['WPP_Medium variant'] / 1e3,
                label='WPP',
                color=colors['WPP'])
            ax[i].fill_between(
                tot_deaths_byage.index,
                tot_deaths_byage['WPP_Low variant'] / 1e3,
                tot_deaths_byage['WPP_High variant'] / 1e3,
                facecolor=colors['WPP'], alpha=0.2)
        else:
            ax[i].plot(
                tot_deaths_byage.index,
                tot_deaths_byage['WPP_Estimates'] / 1e3,
                label='WPP',
                color=colors['WPP'])

        if 'GBD_Est' in tot_deaths_byage.columns:
            ax[i].plot(
                tot_deaths_byage.index,
                tot_deaths_byage['GBD_Est'] / 1e3,
                label='GBD',
                color=colors['GBD'])
            ax[i].fill_between(
                tot_deaths_byage.index,
                tot_deaths_byage['GBD_Lower'] / 1e3,
                tot_deaths_byage['GBD_Upper'] / 1e3,
                facecolor=colors['GBD'], alpha=0.2)

        ax[i].plot(
            tot_deaths_byage.index,
            tot_deaths_byage['Model_mean'] / 1e3,
            label='Model',
            color=colors['Model'])
        ax[i].fill_between(
            tot_deaths_byage.index,
            tot_deaths_byage['Model_lower'] / 1e3,
            tot_deaths_byage['Model_upper'] / 1e3,
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
