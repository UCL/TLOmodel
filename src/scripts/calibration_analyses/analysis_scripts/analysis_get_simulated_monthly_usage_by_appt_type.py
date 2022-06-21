"""
Obtain the simulated monthly usage by appt type, to be compared with real usage.

N.B. This script takes hours to run due to the big data.

N.B. This script uses the package `squarify`: so run, `pip install squarify` first.
"""
# from collections import defaultdict
from pathlib import Path

# import numpy as np
import pandas as pd
# import squarify
# from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    # get_scenario_outputs,
    summarize,
    # unflatten_flattened_multi_index_in_logging,
    # create_pickles_locally,
    # load_pickled_dataframes,
)


def apply(results_folder: Path, output_folder: Path):
    # %% Declare the name of the file that specified the scenarios used in this run.
    # scenario_filename = 'long_run_all_diseases.py'

    # %% Declare usual paths.
    # path of model output
    # outputspath = Path('./outputs/bshe@ic.ac.uk')
    # path of real appointment usage and relevant data
    # rfp = Path('./resources')

    # Find results folder (most recent run generated using that scenario_filename)
    # results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    # create_pickles_locally(results_folder, compressed_file_name_prefix="long_run")
    # <-- sometimes needed after download
    # log = load_pickled_dataframes(results_folder, 0, 4)['tlo.methods.healthsystem.summary']
    #
    # hsi = log['HSI_Event']
    #
    # appt_usage = hsi.join(hsi['Number_By_Appt_Type_Code'].apply(pd.Series).fillna(0.0)).drop(
    #     columns='Number_By_Appt_Type_Code')

    # Declare period for which the results will be generated (defined inclusively)
    TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))

    # Declare path for output graphs from this script
    # make_graph_file_name = lambda stub: results_folder / f"{stub}.png"

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
            .drop(columns=['Person_ID', 'Squeeze_Factor', 'did_run', 'TREATMENT_ID', 'Facility_Level'])

        # Unpack the dictionary in `Number_By_Appt_Type_Code`.
        _df = _df.join(_df['Number_By_Appt_Type_Code'].apply(pd.Series).fillna(0.0)).drop(
            columns='Number_By_Appt_Type_Code')

        return _df

    # hsi = log['HSI_Event']
    # hsi = formatting_hsi_df(hsi)

    def get_counts_of_appt_type(_df):
        # long format
        _df = formatting_hsi_df(_df).melt(id_vars=['date', 'Facility_ID'], var_name='Appt_Type', value_name='Num')

        # creat month and year
        _df['Year'] = _df['date'].dt.year
        _df['Month'] = _df['date'].dt.month

        # group by Year, Month, Appt_Type, Facility_ID
        _df = _df.drop(columns='date').groupby(by=['Year', 'Month', 'Facility_ID', 'Appt_Type'])['Num'].sum()

        return _df

    # hsi = log['HSI_Event']
    # hsi = get_counts_of_appt_type(hsi)

    # counts of appointments
    counts_of_appt_type = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='HSI_Event',
            custom_generate_series=get_counts_of_appt_type,
            do_scaling=True
        ),
        only_mean=True,  # the mean of different runs; if False, uncertainty will be considered
        collapse_columns=True,  # squeeze dimensions of draws and runs
    ).reset_index()

    counts_of_appt_type_2015_2019 = counts_of_appt_type.copy()
    counts_of_appt_type_2015_2019.to_csv(
        output_folder / 'Simulated appt usage between 2015 and 2019.csv', index=False)
