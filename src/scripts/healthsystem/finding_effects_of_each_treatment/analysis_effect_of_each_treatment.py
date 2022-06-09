"""Produce plots to show the impact of removing each set of Treatments from the healthcare system"""

from pathlib import Path

import numpy as np
import pandas as pd
import squarify
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import extract_results, get_scenario_outputs, load_pickled_dataframes, make_age_grp_lookup, \
    make_age_grp_types, summarize

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'scenario_effect_of_each_treatment.py'

# %% Declare usual paths:
outputspath = Path('./outputs/tbh03@ic.ac.uk')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

TARGET_PERIOD = (Date(2010, 1, 1), Date(2019, 12, 31))

_, age_grp_lookup = make_age_grp_lookup()


def get_parameter_names_from_scenario_file() -> tuple:
    """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
    from scripts.healthsystem.finding_effects_of_each_treatment.scenario_effect_of_each_treatment import (
        EffectOfEachTreatment,
    )
    e = EffectOfEachTreatment()
    return tuple(e._scenarios.keys())


def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


def get_counts_of_hsi_by_treatment_id(_df):
    return _df\
        .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID']\
        .apply(pd.Series)\
        .sum().\
        astype(int)


def get_colors(x):
    cmap = plt.cm.get_cmap('jet')
    return [cmap(i) for i in np.arange(0, 1, 1.0 / len(x))]


# %% Define parameter names
param_names = get_parameter_names_from_scenario_file()

# Find the difference in the number of deaths between each draw and a comparison, comparing with the run
comparator_draw_num = [i for i, x in enumerate(param_names) if x == 'Everything'][0]


# %% Quantify the health associated with each TREATMENT_ID (short) (The difference in deaths and DALYS between each
# scenario and the 'Everything' scenario.)

def set_param_names_as_column_index_level_0(_df):
    """Set the columns index (level 0) as the param_names. Suppress the 'No' prefix and '*' suffix."""
    param_names_no_prefix = [x.lstrip("No ").rstrip("*") for x in param_names]
    _df.columns = _df.columns.set_levels(param_names_no_prefix, level=0)
    return _df


def num_deaths_by_age_group(_df):
    """Return total number of deaths (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]\
        .groupby(_df['age'].map(age_grp_lookup).astype(make_age_grp_types()))\
        .size()

def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'year']) \
        .sum()

num_deaths = extract_results(
    results_folder,
    module='tlo.methods.demography',
    key='death',
    custom_generate_series=num_deaths_by_age_group,
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0).sum()  # (Summing across age-groups)

num_dalys = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0).sum()  # (Summing across causes)


def find_percentage_difference_relative_to_comparison(_ser: pd.Series, comparison: str):
    """Find the percentage difference in the values in a pd.Series with a multi-index, between the draws (level 0)
    within the runs (level 1). Drop the comparison entries."""
    return _ser\
        .unstack()\
        .apply(lambda x: 100.0 * (x[comparison] - x) / x[comparison], axis=0)\
        .drop(index=[comparison])\
        .stack()

fig, ax = plt.subplots()
name_of_plot = 'Deaths Averted by Each TREATMENT_ID (Short)'
pc_deaths_averted = summarize(
    pd.DataFrame(
        find_percentage_difference_relative_to_comparison(num_deaths, comparison='Everything')).T
).iloc[0].unstack().sort_values(by='mean', ascending=False)
pc_deaths_averted['mean'].plot.barh(ax=ax)
ax.set_title(name_of_plot)
ax.set_ylabel('TREATMENT_ID (Short)')
ax.set_xlabel('Percent of Deaths Averted')
ax.yaxis.set_tick_params(labelsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
fig.show()


fig, ax = plt.subplots()
name_of_plot = 'DALYS Averted by Each TREATMENT_ID (Short)'
pc_dalys_averted = summarize(
    pd.DataFrame(
        find_percentage_difference_relative_to_comparison(num_dalys, comparison='Everything')).T
).iloc[0].unstack().sort_values(by='mean', ascending=False)
pc_dalys_averted['mean'].plot.barh(ax=ax)
ax.set_title(name_of_plot)
ax.set_ylabel('TREATMENT_ID (Short)')
ax.set_xlabel('Percent of DALYS Averted')
ax.yaxis.set_tick_params(labelsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
fig.show()



# %% Quantify the healthcare system resources used with each TREATMENT_ID (short) (The difference in the number of
# appointments between each scenario and the 'Everything' scenario.)

counts_of_hsi_by_treatment_id = extract_results(
    results_folder,
    module='tlo.methods.healthsystem.summary',
    key='HSI_Event',
    custom_generate_series=get_counts_of_hsi_by_treatment_id,
    do_scaling=True
)

# 1) Examine the appointments that are occuring
for i, scenario_name in enumerate(param_names):
    average_num_hsi = counts_of_hsi_by_treatment_id.loc[:, (i, slice(None))].mean(axis=1)
    average_num_hsi = average_num_hsi.loc[average_num_hsi > 0]

    # todo: find a way to fail gracefully here when one draw (=20) fails....
    # todo: find why nan's are appearing anywhere in `counts_of_hsi_treatment_id`
    fig, ax = plt.subplots()
    name_of_plot = f'HSI Events Occurring With Service Availability = {scenario_name}'
    squarify.plot(
        sizes=average_num_hsi.values,
        label=average_num_hsi.index,
        color=get_colors(average_num_hsi.values),
        alpha=1,
        pad=True,
        ax=ax,
        text_kwargs={'color': 'black', 'size': 8},
    )
    ax.set_axis_off()
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()


# 2) Look at differences...

# todo ....

