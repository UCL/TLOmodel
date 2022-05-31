"""Produce plots to show the impact of removing each set of Treatments from the healthsystem"""

from pathlib import Path

import numpy as np
import squarify
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import extract_results, get_scenario_outputs

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'scenario_effect_of_each_treatment.py'

# %% Declare usual paths:
outputspath = Path('./outputs/tbh03@ic.ac.uk')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

TARGET_PERIOD = (Date(2010, 1, 1), Date(2010, 12, 31))


def get_parameter_names_from_scenario_file() -> tuple:
    """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
    from scripts.healthsystem.finding_effects_of_each_treatment.scenario_effect_of_each_treatment import (
        EffectOfEachTreatment,
    )
    e = EffectOfEachTreatment()
    return tuple(e._scenarios.keys())


param_names = get_parameter_names_from_scenario_file()


# %% Examine the HSI that occurred under each scenario....

def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


def get_counts_of_hsi_by_treatment_id(_df):
    return _df.groupby(by='TREATMENT_ID').size()


def get_colors(x):
    cmap = plt.cm.get_cmap('jet')
    return [cmap(i) for i in np.arange(0, 1, 1.0 / len(x))]


counts_of_hsi_by_treatment_id = extract_results(
    results_folder,
    module='tlo.methods.healthsystem',  # todo <-- change this to use the summary logger only (and update the scenario file)
    key='HSI_Event',
    custom_generate_series=get_counts_of_hsi_by_treatment_id,
    do_scaling=True
)

for i, scenario_name in enumerate(param_names):
    average_num_hsi = counts_of_hsi_by_treatment_id.loc[:, (i, slice(None))].mean(axis=1)
    average_num_hsi = average_num_hsi.loc[average_num_hsi > 0]

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

# %% Quantify the health difference between each scenario and the 'Everything' scenario.
# todo....

#
# import pickle
# from datetime import datetime
# from pathlib import Path
#
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Define paths and filenames
# rfp = Path("./resources")
# outputpath = Path("./outputs")  # folder for convenience of storing outputs
# results_filename = outputpath / '2020_11_23_health_system_systematic_run.pickle'
# make_file_name = lambda stub: outputpath / f"{datetime.today().strftime('%Y_%m_%d''')}_{stub}.png"  # noqa: E731
#
# with open(results_filename, 'rb') as f:
#     results = pickle.load(f)['results']
#
# # %% Make summary plots:
# # Get total deaths in the duration of each simulation:
# deaths = dict()
# for key in results.keys():
#     deaths[key] = len(results[key]['tlo.methods.demography']['death'])
#
# deaths = pd.Series(deaths)
#
# # compute the excess deaths compared to the No Treatments
# excess_deaths = deaths['Nothing'] - deaths[~(deaths.index == 'Nothing')]
#
# excess_deaths.plot.barh()
# plt.title('The Impact of Each set of Treatment_IDs')
# plt.ylabel('Deaths Averted by treatment_id, 2010-2014')
# plt.savefig(make_file_name('Impact_of_each_treatment_id'))
# plt.tight_layout()
# plt.show()
