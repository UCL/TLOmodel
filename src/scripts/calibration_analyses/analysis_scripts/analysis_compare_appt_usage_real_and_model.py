"""
Compare appointment usage from model output with real appointment usage.

The real appointment usage is collected from DHIS2 system and HIV Dept.

N.B. This script uses the package `squarify`: so run, `pip install squarify` first.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import squarify
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
    summarize,
    unflatten_flattened_multi_index_in_logging,
    create_pickles_locally,
)

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'long_run_all_diseases.py'

# %% Declare usual paths.
# path of model output
outputspath = Path('./outputs/bshe@ic.ac.uk')
# path of real appointment usage
# todo: store the real data in the right folder
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

# create_pickles_locally(results_folder, compressed_file_name_prefix="long_run")  # <-- sometimes needed after download

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2010, 1, 1), Date(2010, 12, 31))

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"


# %% Declare helper functions

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


def get_counts_of_appt_type_by_treatment_id_short(_df):
    return formatting_hsi_df(_df) \
        .drop(columns=['date', 'TREATMENT_ID', 'Facility_Level']) \
        .melt(id_vars=['TREATMENT_ID_SHORT'], var_name='Appt_Type', value_name='Num') \
        .groupby(by=['TREATMENT_ID_SHORT', 'Appt_Type'])['Num'].sum()


# counts of hsi events
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

# counts of appointments
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


# todo: For model output, get the monthly usage per appointment between 2012 and 2019

# todo: Compare the model output with the real usage
