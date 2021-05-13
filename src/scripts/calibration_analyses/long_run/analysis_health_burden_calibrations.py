"""Produce comparisons between model and GBD of deaths by cause in a particular period."""

# todo - get these plots working when the output comes from the batch system
# todo - investigate causes of _extreme_ variation in deaths
# todo - unify the labelling of causes in the HealthBurden module to simplify processing
# todo - same for DALYS

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

# rename the uncertainity range to align with the model
gbd = gbd.rename(columns={'val': 'mean'})


# %% Load modelling results:

# get the scaling_factor for the population run: todo - put this in the 'tlo.population' log
sf = get_scaling_factor(results_folder=results_folder, resourcefilepath=rfp)

deaths = summarize(extract_results(results_folder,
                                         module="tlo.methods.demography",
                                         key="death",
                                         custom_generate_series="groupby(['sex', 'date', 'age', 'cause'])['person_id'].count()"
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

# - GBD:
deaths_pt['GBD'] = gbd.loc[(gbd.period == period) & (gbd.measure_name == 'Deaths')]\
    .groupby(['sex', 'age_grp', 'unified_cause'])[['mean', 'lower', 'upper']].sum().unstack().div(5.0)

# - TLO Model:
deaths_pt['Model'] = deaths_model.loc[(deaths_model.period == period)].groupby(
    by=['sex', 'age_grp', 'unified_cause']
)[['mean', 'lower', 'upper']].sum().unstack(fill_value=0.0).div(5.0)


# %% Make figures of overall summaries of deaths by cause
# todo - improve formatting!!! ---- this should be a stacked bar chart (like the below)

dats = ['GBD', 'Model']
sex = ['F', 'M']
sexname = lambda x: 'Females' if x=='F' else 'Males'

fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True)

for col, sex in enumerate(sex):
    for row, dat in enumerate(dats):

        ax = axes[row][col]
        df = deaths_pt[dat].loc[sex].loc[:, pd.IndexSlice['mean']] / 1e3

        xs = np.arange(len(df.index))
        df.plot.bar(ax=ax)
        ax.set_xlabel('Age Group')
        ax.set_ylabel(f"Deaths per year (thousands)")
        ax.set_title(f"{sexname(sex)}: {dat}")

plt.show()



# todo - got to here!
# %% Plots comparing between model and actual across all ages and sex:

def make_scatter_graph(gbd_totals, model_totals, title, show_labels=None):
    scatter = pd.concat({
        'GBD': gbd_totals,
        'Model': model_totals}, axis=1, sort=True
        )

    xylim = np.round(scatter.max().max(), -5)

    # plot points:
    fig, ax = plt.subplots()
    ax.plot(scatter['GBD'], scatter['Model'], 'bo')

    # label selected points:
    if show_labels:
        for label in show_labels:
            row = scatter.loc[label]
            ax.annotate(label,
                        (row['GBD'], row['Model']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        )

    # X=Y line
    line_x = np.linspace(0, xylim)
    ax.plot(line_x, line_x, 'r')

    ax.set(xlim=(0, xylim), ylim=(0, xylim))
    ax.set_xlabel('GBD')
    ax.set_ylabel('Model')
    ax.set_title(f'{title} in {year}')
    plt.savefig(make_file_name("Deaths_Scatter_Plot"))
    plt.show()

# - Make scatter graph for deaths
make_scatter_graph(gbd_totals=gbd_deaths_pt.sum(),
                   model_totals=deaths_pt.sum(),
                   title="Total Deaths",
                   show_labels=['AIDS', 'Childhood Diarrhoea', 'Malaria', 'Other'])

# - Make scatter graph for DALYS
make_scatter_graph(gbd_totals=gbd_dalys_pt.sum(),
                   model_totals=dalys_pt.sum(),
                   title="Total DALYS",
                   show_labels=['AIDS', 'Childhood Diarrhoea', 'Malaria', 'Other'])

# %% Make stacked bar charts breaking-out causes by age/sex

def make_stacked_bar_comparison(gbd_pt, model_pt, ylabel):

    def plot_stacked_bar_chart_of_deaths_by_cause(ax, pt, title, ylabel):
        pt.plot.bar(stacked=True, ax=ax)
        ax.set_title(title)
        ax.set_ylabel(f"{ylabel}")
        ax.set_label('Age Group')
        ax.get_legend().remove()

    sex_as_string = lambda x: 'Females' if x == 'F' else 'Males'

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, sex in enumerate(['F', 'M']):
         plot_stacked_bar_chart_of_deaths_by_cause(
             ax=axs[i, 0],
             pt=model_pt.loc[(sex,), ],
             ylabel=ylabel,
             title=f"Model: {sex_as_string(sex)}")
         plot_stacked_bar_chart_of_deaths_by_cause(
             ax=axs[i, 1],
             pt=gbd_pt.loc[(sex,), ],
             ylabel=ylabel,
             title=f"GBD: {sex_as_string(sex)}")
    plt.show()
    plt.savefig(outputpath / f"StackedBars_{ylabel}.png")

# Make stacked bar comparison plot of deaths
make_stacked_bar_comparison(
    model_pt=deaths_pt,
    gbd_pt=gbd_deaths_pt,
    ylabel='Deaths')

# Make stacked bar comparison plot of DALYS
make_stacked_bar_comparison(
    model_pt=dalys_pt,
    gbd_pt=gbd_dalys_pt,
    ylabel='DALYS')



# # %% make figure for Malaria:
# plt.bar(
#     ['GBD', 'TLO'],
#     [gbd_deaths_pt['Malaria'].sum(), deaths_pt['Malaria'].sum()]
# )
# plt.show()
