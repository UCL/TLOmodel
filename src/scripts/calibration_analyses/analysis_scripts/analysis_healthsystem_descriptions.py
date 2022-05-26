"""Description of the HSI that run.
N.B. This script uses the package `squarify`: so run, `pip install squarify` first.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import squarify
from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

# todo - define a colormap for TREATMENT_ID short and use this in Figure 1 and 3
# todo - Selective labelling of only the biggest blocks.

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'long_run_all_diseases.py'  # <-- update this to look at other results

# %% Declare usual paths.
outputspath = Path('./outputs/tbh03@ic.ac.uk')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# Define colours to use:
colors = {
    'Model': 'royalblue',
    'Census': 'darkred',
    'WPP': 'forestgreen',
    'GBD': 'plum'
}

# %% Declare helper functions
def formatting_hsi_df(_df):
    """Standard formatting for the HSI_Event log."""

    # Remove entries for those HSI that did not run
    _df = _df.drop(_df.index[~_df.did_run])\
             .reset_index(drop=True)\
             .drop(columns=['Person_ID', 'Squeeze_Factor', 'Facility_ID', 'did_run'])

    # todo: Limit the record to a particular date_range: currently no limitation on date-range

    # Unpack the dictionary in `Number_By_Appt_Type_Code`.
    _df = _df.join(_df['Number_By_Appt_Type_Code'].apply(pd.Series).fillna(0.0)).drop(columns='Number_By_Appt_Type_Code')

    # Produce course version of TREATMENT_ID (just first level, which is the module)
    _df['TREATMENT_ID_SHORT'] = _df['TREATMENT_ID'].str.split('_').apply(lambda x: x[0])

    return _df

def get_colors(x):
    cmap = plt.cm.get_cmap('jet')
    return [cmap(i) for i in np.arange(0, 1, 1.0 / len(x))]


#%% "Figure 1": The Distribution of HSI_Events that occur by TREATMENT_ID

def get_counts_of_hsi_by_treatment_id(_df):
    return formatting_hsi_df(_df).groupby(by='TREATMENT_ID').size()


def get_counts_of_hsi_by_treatment_id_short(_df):
    return formatting_hsi_df(_df).groupby(by='TREATMENT_ID_SHORT').size()


counts_of_hsi_by_treatment_id = summarize(
    extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_treatment_id,
            do_scaling=True
    ),
    only_mean=True
)

counts_of_hsi_by_treatment_id_short = summarize(
    extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_treatment_id_short,
            do_scaling=True
    ),
    only_mean=True
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


#%% "Figure 2": The Appointments Used

def get_counts_of_appt_type_by_treatment_id_short(_df):
    return formatting_hsi_df(_df)\
            .drop(columns=['date', 'TREATMENT_ID', 'Facility_Level'])\
            .melt(id_vars=['TREATMENT_ID_SHORT'], var_name='Appt_Type', value_name='Num')\
            .groupby(by=['TREATMENT_ID_SHORT', 'Appt_Type'])['Num'].sum()

counts_of_appt_by_treatment_id_short = summarize(
    extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='HSI_Event',
            custom_generate_series=get_counts_of_appt_type_by_treatment_id_short,
            do_scaling=True
    ),
    only_mean=True,
    collapse_columns=True,
)

fig, ax = plt.subplots()
name_of_plot = 'Appointment Types Used'
(counts_of_appt_by_treatment_id_short / 1e6).unstack().plot.bar(ax=ax, stacked=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(ncol=3, prop={'size': 6}, loc='upper left')
ax.set_ylabel('Number of appointments (millions)')
ax.set_xlabel('TREATMENT_ID (Short)')
ax.set_ylim(0, 80)
ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
fig.tight_layout()
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
fig.show()


#%% "Figure 3": The Fraction of the time of each HCW used by each TREATMENT_ID (Short)

def get_share_of_time_for_hw_by_short_treatment_id(_df):
    appts = formatting_hsi_df(_df)\
        .drop(columns=['date', 'TREATMENT_ID'])\
        .melt(id_vars=['Facility_Level', 'TREATMENT_ID_SHORT'], var_name='Appt_Type', value_name='Num')\
        .groupby(by=['TREATMENT_ID_SHORT', 'Facility_Level', 'Appt_Type'])['Num'].sum()\
        .reset_index()

    # Find the time of each HealthCareWorker (HCW) for each appointment at eahc level
    att = pd.pivot_table(
        pd.read_csv(rfp / 'healthsystem' / 'human_resources' / 'definitions' / 'ResourceFile_Appt_Time_Table.csv'),
        index=['Appt_Type_Code', 'Facility_Level'],
        columns='Officer_Category',
        values='Time_Taken_Mins',
        fill_value=0.0
    ).reset_index()

    m = appts.merge(att,
                    left_on=['Appt_Type', 'Facility_Level'],
                    right_on=['Appt_Type_Code', 'Facility_Level'],
                    how='left')\
             .drop(columns=['Facility_Level', 'Appt_Type', 'Appt_Type_Code'])\
             .set_index('TREATMENT_ID_SHORT')

    return m.apply(lambda row: row * row['Num'], axis=1) \
            .drop(columns='Num') \
            .groupby(level=0).sum()\
            .apply(lambda col: col / col.sum(), axis=0)\
            .stack()


share_of_time_for_hw_by_short_treatment_id = summarize(
    extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='HSI_Event',
            custom_generate_series=get_share_of_time_for_hw_by_short_treatment_id,
            do_scaling=True
    ),
    only_mean=True,
    collapse_columns=True
)

def drop_zero_rows(ser):
    return ser.drop(ser[ser == 0].index)

all_cadres = share_of_time_for_hw_by_short_treatment_id.index.levels[1]
cadres_to_plot = ['DCSA', 'Nursing_and_Midwifery', 'Clinical', 'Pharmacy']

fig, ax = plt.subplots(nrows=2, ncols=2)
name_of_plot = 'Proportion of Time Used For Selected Cadre by TREATMENT_ID (Short)'
for _cadre, _ax in zip(cadres_to_plot, ax.reshape(-1)):
    _x = drop_zero_rows(share_of_time_for_hw_by_short_treatment_id.loc[(slice(None), _cadre)])
    squarify.plot(
        sizes=_x.values,
        label=_x.index,
        color=get_colors(_x),
        alpha=1,
        pad=True,
        ax=_ax,
        text_kwargs={'color': 'black', 'size': 8},
    )
    _ax.set_axis_off()
    _ax.set_title(f'{_cadre}', {'size': 10, 'color': 'black'})
fig.suptitle(name_of_plot, fontproperties={'size': 12})
fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
fig.show()


#%% "Figure 4": The level of usage of the HealthSystem HR Resources

log = load_pickled_dataframes(results_folder, 0, 0)['tlo.methods.healthsystem']['Capacity'].set_index('date')

df = log['Frac_Time_Used_By_Facility_ID'].apply(pd.Series)
df.columns = df.columns.astype(int)
df = df.reindex(columns=sorted(df.columns))

fig, ax = plt.subplots()
df.plot(ax=ax)

fig.tight_layout()
fig.show()







#%% "Figure 5": The level of usage of the Beds in the HealthSystem
# todo ...


#%% "Figure 6": Usage of consumables in the HealthSystem
# todo ...


