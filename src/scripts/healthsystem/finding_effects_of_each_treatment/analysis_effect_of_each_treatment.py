"""Produce plots to show the impact of removing each set of Treatments from the healthsystem"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import squarify
from tlo import Date

from tlo.analysis.utils import (
    extract_results,
    format_gbd,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    summarize,
)

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'scenario_effect_of_each_treatment.py'

# %% Declare usual paths:
outputspath = Path('./outputs/tbh03@ic.ac.uk')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

a = load_pickled_dataframes(results_folder, 0, 0)['tlo.methods.healthsystem']
b = load_pickled_dataframes(results_folder, 1, 0)['tlo.methods.healthsystem']
c = load_pickled_dataframes(results_folder, 2, 0)['tlo.methods.healthsystem']
d = load_pickled_dataframes(results_folder, 3, 0)['tlo.methods.healthsystem']

TARGET_PERIOD = (Date(2010, 1, 1), Date(2010, 12, 31))


# Get parameter names




# %% Examine the HSI's occurring under each scenario....

def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


def formatting_hsi_df(_df):
    """Standard formatting for the HSI_Event log."""

    # Remove entries for those HSI that did not run
    _df = drop_outside_period(_df) \
        .drop(_df.index[~_df.did_run]) \
        .reset_index(drop=True) \
        .drop(columns=['Person_ID', 'Squeeze_Factor', 'Facility_ID', 'did_run'])

    # Unpack the dictionary in `Number_By_Appt_Type_Code`.
    _df = _df.join(_df['Number_By_Appt_Type_Code'].apply(pd.Series).fillna(0.0)).drop(
        columns='Number_By_Appt_Type_Code')

    # Produce course version of TREATMENT_ID (just first level, which is the module)
    _df['TREATMENT_ID_SHORT'] = _df['TREATMENT_ID'].str.split('_').apply(lambda x: x[0])

    return _df

def get_counts_of_hsi_by_treatment_id(_df):
    return formatting_hsi_df(_df).groupby(by='TREATMENT_ID').size()


def get_counts_of_hsi_by_treatment_id_short(_df):
    return formatting_hsi_df(_df).groupby(by='TREATMENT_ID_SHORT').size()


def get_colors(x):
    cmap = plt.cm.get_cmap('jet')
    return [cmap(i) for i in np.arange(0, 1, 1.0 / len(x))]


counts_of_hsi_by_treatment_id = extract_results(
    results_folder,
    module='tlo.methods.healthsystem',
    key='HSI_Event',
    custom_generate_series=get_counts_of_hsi_by_treatment_id,
    do_scaling=True
)

counts_of_hsi_by_treatment_id_short = extract_results(
    results_folder,
    module='tlo.methods.healthsystem',
    key='HSI_Event',
    custom_generate_series=get_counts_of_hsi_by_treatment_id_short,
    do_scaling=True
)

fig, ax = plt.subplots()
name_of_plot = 'Proportion of HSI Events by TREATMENT_ID'
squarify.plot(
    sizes=counts_of_hsi_by_treatment_id.values,
    label=counts_of_hsi_by_treatment_id.index,
    color=get_colors(counts_of_hsi_by_treatment_id_short.values),
    alpha=1,
    pad=True,
    ax=ax,
    text_kwargs={'color': 'black', 'size': 8},
)
ax.set_axis_off()
ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
fig.show()

fig, ax = plt.subplots()
name_of_plot = 'HSI Events by TREATMENT_ID (Short)'
squarify.plot(
    sizes=counts_of_hsi_by_treatment_id_short.values,
    label=counts_of_hsi_by_treatment_id_short.index,
    color=get_colors(counts_of_hsi_by_treatment_id_short.values),
    alpha=1,
    pad=True,
    ax=ax,
    text_kwargs={'color': 'black', 'size': 8}
)
ax.set_axis_off()
ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
fig.show()














import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Define paths and filenames
rfp = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / '2020_11_23_health_system_systematic_run.pickle'
make_file_name = lambda stub: outputpath / f"{datetime.today().strftime('%Y_%m_%d''')}_{stub}.png"  # noqa: E731

with open(results_filename, 'rb') as f:
    results = pickle.load(f)['results']

# %% Make summary plots:
# Get total deaths in the duration of each simulation:
deaths = dict()
for key in results.keys():
    deaths[key] = len(results[key]['tlo.methods.demography']['death'])

deaths = pd.Series(deaths)

# compute the excess deaths compared to the No Treatments
excess_deaths = deaths['Nothing'] - deaths[~(deaths.index == 'Nothing')]

excess_deaths.plot.barh()
plt.title('The Impact of Each set of Treatment_IDs')
plt.ylabel('Deaths Averted by treatment_id, 2010-2014')
plt.savefig(make_file_name('Impact_of_each_treatment_id'))
plt.tight_layout()
plt.show()
