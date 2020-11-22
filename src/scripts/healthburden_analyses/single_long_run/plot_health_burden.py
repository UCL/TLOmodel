"""Produce comparisons between model and GBD of deaths by cause in a particular year."""

# todo - update to newest data
# todo - deaths, to include 0 year-olds.
# todo - look to see why diarrhoea causes so many deaths
# todo - get GBD dalys database


import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scripts.healthburden_analyses.single_long_run.helper_funcs_for_processing_data_files import *

from tlo.methods import demography

# %% Set file paths etc
# Define the particular year for the focus of this analysis
year = 2010

# Resource file path
rfp = Path("./resources")

# Where will outputs be found
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / 'long_run.pickle'

with open(results_filename, 'rb') as f:
    output = pickle.load(f)['output']

datestamp = datetime.today().strftime("__%Y_%m_%d")

# %% Load and process the GBD data

gbd = load_gbd_deaths_and_dalys_data(output)

# Make pivot-tables of the form (index=sex/age, columns=unified_cause) (in a particular year)
#  - Limit to relevant year and make the pivot table for deaths
gbd_deaths_pt = gbd.loc[(gbd.year == year) & (gbd.measure_name == 'Deaths')]\
    .pivot_table(index=['sex', 'age_range'], columns='unified_cause', values='val', fill_value=0)

#  - Limit to relevant year and make the pivot table for deaths
gbd_dalys_pt = gbd.loc[(gbd.year == year) & (gbd.measure_name == 'DALYs (Disability-Adjusted Life Years)')]\
    .pivot_table(index=['sex', 'age_range'], columns='unified_cause', values='val', fill_value=0)


# %% Get the scaling so that the population size matches that of the real population of Malawi
scaling_factor = get_scaling_factor(output)


# %% Get deaths from Model into a pivot-table (index=sex/age, columns=unified_cause) (in the particular year)
deaths_df = output["tlo.methods.demography"]["death"]
deaths_df["date"] = pd.to_datetime(deaths_df["date"])
deaths_df["year"] = deaths_df["date"].dt.year
deaths_df['age_range'] = age_cats(deaths_df.age)

mapper_from_tlo_strings, _ = get_causes_mappers(output)
deaths_df["unified_cause"] = deaths_df["cause"].map(mapper_from_tlo_strings)
assert not pd.isnull(deaths_df["cause"]).any()

df = deaths_df.loc[(deaths_df.year == year)].groupby(
    ['sex', 'age_range', 'unified_cause']).size().reset_index()
df = df.rename(columns={0: 'count'})
df['count'] *= scaling_factor

deaths_pt = df.pivot_table(index=['sex', 'age_range'], columns='unified_cause', values='count', fill_value=0)

# todo - guaranteee that all values of age_range are represented

# %% Get DALYS from Model into a pivot-table (index=sex/age, columns=unified_cause) (in the particular year)
dalys = output['tlo.methods.healthburden']['dalys'].copy()

# drop date because this is just the date of logging and not the date to which the results pertain.
dalys = dalys.drop(columns=['date'])

# re-clasify the causes of DALYS
dalys_melt = dalys.melt(id_vars=['sex', 'age_range', 'year'], var_name='cause', value_name='value')
dalys_melt['cause'] = dalys_melt.cause\
    .replace({'YLL_Demography_Other': '_Other'})\
    .str.split('_').apply(lambda x: x[1]).map(mapper_from_tlo_strings)
assert not dalys_melt['cause'].isna().any()

# limit to the year (so the output wil be new year's day of the next year)
dalys_melt = dalys_melt.loc[dalys_melt.year == (year + 1)]
dalys_melt = dalys_melt.drop(columns=['year'])

# format age-groups and set index to be sex/age_range:
age_range_categories, _ = get_age_range_categories()
dalys_melt['age_range'] = pd.Categorical(dalys_melt['age_range'].replace({
    '95-99': '95+',
    '100+': '95+'}
), categories=age_range_categories, ordered=True)

# scale the total dalys
dalys_melt['value'] *= scaling_factor

# Make pivot-table
dalys_pt = dalys_melt.pivot_table(index=['sex', 'age_range'], columns='cause', values='value', fill_value=0)


# %% Make figures of overall summaries of deaths by cause
deaths_pt.sum().plot.bar()
plt.xlabel('Cause')
plt.ylabel(f"Total deaths in {year}")
plt.savefig(outputpath / f"Deaths by cause {datestamp}.png")
plt.show()

dalys_pt.sum().plot.bar()
plt.xlabel('Cause')
plt.ylabel(f"Total DALYS in {year}")
plt.savefig(outputpath / f"DALYS by cause {datestamp}.png")
plt.show()


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
    plt.savefig(outputpath / f"ScatterPlot_{title}.png")
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

