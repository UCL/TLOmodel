"""Produce comparisons between model and GBD of deaths by cause in a particular period."""

# todo - unify the labelling of causes in the HealthBurden module to simplify processing
# todo - do all the same for DALYS

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from tlo.analysis.utils import (
    make_age_grp_types,
    make_age_grp_lookup,
    make_calendar_period_lookup,
    make_calendar_period_type,
    extract_params,
    extract_results,
    get_scaling_factor,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    format_gbd,
    get_causes_mappers
)

# %% Declare usual paths:
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

# %% Set the period for the analysis (comparison is made of the average annual number of deaths in this period)
period = '2010-2014'

# %% Load and process the GBD data
gbd = format_gbd(pd.read_csv(rfp / 'demography' / 'ResourceFile_Deaths_And_DALYS_GBD2019.csv'))


# %% Load modelling results:

# get the scaling_factor for the population run: todo - put this in the 'tlo.population' log
sf = get_scaling_factor(results_folder=results_folder, resourcefilepath=rfp)

# Extract results, summing by year
deaths = summarize(extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series="assign(year = lambda x: x['date'].dt.year)"
                           ".groupby(['sex', 'date', 'age', 'cause'])['person_id'].count()"
),
    collapse_columns=True
).mul(sf).reset_index()

# Sum by year/sex/age-group
agegrps, agegrplookup = make_age_grp_lookup()
deaths['year'] = deaths['date'].dt.year
deaths['age_grp'] = deaths['age'].map(agegrplookup)
deaths['age_grp'] = deaths['age_grp'].astype(make_age_grp_types())
deaths_model = deaths.drop(columns=['age']).groupby(by=['sex', 'age_grp', 'cause', 'year']).sum().reset_index()

# Define period:
calperiods, calperiodlookup = make_calendar_period_lookup()
deaths_model['period'] = deaths_model['year'].map(calperiodlookup).astype(make_calendar_period_type())

# %% Load the cause-mappers (checking that it's up-to-date) and use it to define 'unified_cause' for model and gbd
#  outputs
mapper_from_tlo_strings, mapper_from_gbd_strings = get_causes_mappers(
    gbd_causes=pd.unique(gbd['cause_name']),
    tlo_causes=pd.unique(deaths_model['cause'])
)

deaths_model['unified_cause'] = deaths_model['cause'].map(mapper_from_tlo_strings)
assert not deaths_model['unified_cause'].isna().any()

gbd['unified_cause'] = gbd['cause_name'].map(mapper_from_gbd_strings)
assert not gbd['unified_cause'].isna().any()


# %% Make comparable pivot-tables of the GBD and Model Outputs:
# Find the average deaths (per unified cause) per year within the five-year period of interest
# (index=sex/age, columns=unified_cause) (for particular period specified)

deaths_pt = dict()

# - GBD (making some unifying name changes)
deaths_pt['GBD'] = gbd.loc[(gbd.Period == period) & (gbd.measure_name == 'Deaths')]\
    .rename(columns={
                    'Sex': 'sex',
                    'Age_Grp':'age_grp',
                    'GBD_Est': 'mean',
                    'GBD_Lower': 'lower',
                    'GBD_Upper': 'upper'
    })\
    .groupby(['sex', 'age_grp', 'unified_cause'])[['mean', 'lower', 'upper']].sum().unstack().div(5.0)

# - TLO Model:
deaths_pt['Model'] = deaths_model.loc[(deaths_model.period == period)].groupby(
    by=['sex', 'age_grp', 'unified_cause']
)[['mean', 'lower', 'upper']].sum().unstack(fill_value=0.0).div(5.0)


# %% Make figures of overall summaries of deaths by cause
# todo - improve formatting

dats = ['GBD', 'Model']
sexes = ['F', 'M']
sexname = lambda x: 'Females' if x=='F' else 'Males'

fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True, figsize=(40, 40))

for col, sex in enumerate(sexes):
    for row, dat in enumerate(dats):

        ax = axes[row][col]
        df = deaths_pt[dat].loc[sex].loc[:, pd.IndexSlice['mean']] / 1e3

        xs = np.arange(len(df.index))
        df.plot.bar(stacked=True, ax=ax, fontsize=30)
        ax.set_xlabel('Age Group', fontsize=40)
        ax.set_title(f"{sexname(sex)}: {dat}", fontsize=60)
        ax.get_legend().remove()

# add a big axis, hide frame
bigax = fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
bigax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
bigax.set_ylabel("Deaths per year (thousands)", fontsize=40)
fig.legend(loc="center right", fontsize=15)
fig.tight_layout()
plt.savefig(make_graph_file_name(f"Deaths_StackedBars_ModelvsGBD_{period}"))
plt.show()


# %% Plots comparing between model and actual across all ages and sex:

causes = list(deaths_pt['Model'].loc[:, pd.IndexSlice['mean']].columns)

# Get total number of deaths (all ages) from each source
tot_deaths_by_cause = pd.concat({dat: deaths_pt[dat].sum() for dat in deaths_pt.keys()}, axis=1).fillna(0.0)

fig, ax = plt.subplots()
xylim = 250
all_causes = tot_deaths_by_cause.index.levels[1]
select_labels = ['AIDS', 'Childhood Diarrhoea', 'Other']

for cause in all_causes:

    vals = tot_deaths_by_cause.loc[(slice(None), cause),] / 1e3

    x = vals.at[('mean', cause), 'GBD']
    xerr = np.array([
               x - vals.at[('lower', cause), 'GBD'],
               vals.at[('upper', cause), 'GBD'] - x
    ]).reshape(2, 1)
    y = vals.at[('mean', cause), 'Model']
    yerr = np.array([
        y - vals.at[('lower', cause), 'Model'],
        vals.at[('upper', cause), 'Model'] - y
    ]).reshape(2,1)

    ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, label=cause)

    if cause in select_labels:
        ax.annotate(cause,
                (x,y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
                )


line_x = np.linspace(0, xylim)
ax.plot(line_x, line_x, 'r')
ax.set(xlim=(0, xylim), ylim=(0, xylim))
ax.set_xlabel('GBD')
ax.set_ylabel('Model')
ax.set_title(f'Deaths by Cause {period}')
plt.savefig(make_graph_file_name(f"Deaths_Scatter_Plot_{period}"))
plt.show()


# %% Plots of deaths patten for each cause:

sexes = ['F', 'M']
dats = ['GBD', 'Model']

reformat_cause = lambda x: x.replace(' / ', '_')

for cause in all_causes:
    try:
        deaths_this_cause = pd.concat(
            {dat: deaths_pt[dat].loc[:,(slice(None), cause)] for dat in deaths_pt.keys()}, axis=1
        ).fillna(0.0) / 1e3

        x = list(deaths_this_cause.index.levels[1])
        xs = np.arange(len(x))

        fig, ax = plt.subplots(ncols=1, nrows=2, sharey=True, sharex=True)
        for row, sex in enumerate(sexes):
            for dat in dats:
                ax[row].plot(
                    xs,
                    deaths_this_cause.loc[(sex,),(dat, 'mean', cause)].values,
                    label=dat,
                    color=colors[dat]
                )
                ax[row].fill_between(
                    xs,
                    deaths_this_cause.loc[(sex,),(dat, 'upper', cause)].values,
                    deaths_this_cause.loc[(sex,),(dat, 'lower', cause)].values,
                    facecolor=colors[dat], alpha=0.2
                )
            ax[row].legend()
            ax[row].set_xticks(xs)
            ax[row].set_xticklabels(x, rotation=90)
            ax[row].set_xlabel('Age Group')
            ax[row].set_xlabel('Deaths (thousands)')
            ax[row].set_title(f"{cause}: {sexname(sex)}, {period}")
            ax[row].legend()

        fig.tight_layout()
        plt.savefig(make_graph_file_name(f"Deaths_Scatter_Plot_{period}_{reformat_cause(cause)}"))
        plt.show()

    except KeyError:
        print(f"Could not produce plot for {reformat_cause(cause)}")
