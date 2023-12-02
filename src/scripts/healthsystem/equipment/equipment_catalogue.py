import argparse
import warnings
from pathlib import Path

import pandas as pd

from tlo.analysis.utils import extract_results

# %%% TO SET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Declare whether to scale the counts to Malawi population size
# (True/False)
do_scaling = True
# Declare as a list by which hsi event details you want the equipment be grouped in the catalogue (choose one or more)
# (event details: 'event_name', 'module_name', 'treatment_id', 'facility_level', 'appt_footprint', 'beddays_footprint')
catalog_by_details = ['treatment_id', 'facility_level']
# Declare which time period you want the equipment be grouped in the catalogue (choose only one)
# (periods: 'monthly', 'annual')
catalog_by_time = 'annual'
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# TODO: verify inputs are as expected

# %%% Output file names
# detailed CSV name
output_detailed_file_name = 'equipment_monthly_counts__all_event_details.csv'
# requested details only CSV name
time_index = 'year' if catalog_by_time == 'annual' else 'date'
output_file_name = \
    'equipment_' + catalog_by_time + '_counts__by_' + time_index + '_' + '_'.join(catalog_by_details) + '.csv'


def get_monthly_hsi_event_counts(results_folder: Path) -> pd.DataFrame:
    """Returned pd.DataFrame gives the monthly counts of all the hsi event details logged (details as keys)
    for each simulated month.
    NB. 'healthsystem.summary' logger required to have been set at the level INFO or higher."""

    def get_hsi_event_counts(_df):
        """Get the counts of all the hsi event details logged."""

        def unpack_dict_in_series(_raw: pd.Series):
            # Create an empty DataFrame to store the data
            df = pd.DataFrame()

            # Iterate through the dictionary items
            for _, mydict in _raw.items():
                for date, inner_dict in mydict.items():
                    # Convert the inner_dict to a list of dictionaries with 'date'
                    data = [{'date': date, 'event_details_key': inner_dict_key, 'count': inner_dict_set} for
                            inner_dict_key, inner_dict_set in inner_dict.items()]
                    # Create a DataFrame from the list with date & fac_level as indexes
                    temp_df = pd.DataFrame(data)
                    temp_df.set_index(['date', 'event_details_key'], inplace=True)
                    temp_df.columns = [None]

                    # Concatenate the temporary DataFrame to the result DataFrame
                    df = pd.concat([df, temp_df])

            df.columns = [None]

            return df

        return _df \
            .set_index('date') \
            .pipe(unpack_dict_in_series) \
            .stack() \
            .droplevel(level=2)

    return extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='hsi_event_counts',
        custom_generate_series=get_hsi_event_counts,
        do_scaling=do_scaling
        )


def get_hsi_event_keys_all_runs(results_folder: Path) -> pd.DataFrame:
    """Returned pd.DataFrame gives the dictionaries of hsi_event_details for each draw and run.
    NB. 'healthsystem.summary' logger required to have been set at the level INFO or higher."""

    def get_hsi_event_keys(_df):
        """Get the hsi_event_keys for one particular run."""
        return _df['hsi_event_key_to_event_details']

    return extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='hsi_event_details',
        custom_generate_series=get_hsi_event_keys
        )


def create_equipment_catalogues(results_folder: Path, output_folder: Path):

    # %% Catalog equipment by all HSI event details
    sim_equipment = get_monthly_hsi_event_counts(results_folder)
    sim_equipment_df = pd.DataFrame(sim_equipment)
    hsi_event_keys = get_hsi_event_keys_all_runs(results_folder)

    final_df = pd.DataFrame()

    def details_col_to_str(details_col):
        return details_col.apply(lambda x: ', '.join(map(str, x)))

    for col in hsi_event_keys.columns:
        df_col = sim_equipment_df[col].dropna()
        decoded_keys = df_col.index.get_level_values(1).astype(str).map(hsi_event_keys.at[0, col])

        # %%% Verify the keys in dictionary and dataframe for the run 'col' are same
        # Check if all keys in hsi_event_keys_set are in the 'event_details_key' of df_col
        hsi_event_keys_set = set(hsi_event_keys.at[0, col].keys())
        missing_keys_df =\
            [key for key in hsi_event_keys_set if key not in df_col.index.get_level_values('event_details_key')]

        # Check if all keys in the 'event_details_key' of df_col are in hsi_event_keys_set
        missing_keys_dict =\
            [key for key in df_col.index.get_level_values('event_details_key') if key not in hsi_event_keys_set]

        # Warn if some keys are missing
        if missing_keys_df:
            warnings.warn(UserWarning(f"Keys missing in sim_equipment_df for the run {col}: {missing_keys_df}"))

        if missing_keys_dict:
            warnings.warn(UserWarning(f"Keys missing in hsi_event_keys for the run {col}: {missing_keys_dict}"))
        # %%%

        df_col = pd.concat([df_col, pd.DataFrame(decoded_keys.tolist(), index=df_col.index)], axis=1)
        # Make values in 'appt_footprint', 'beddays_footprint' columns to be string
        df_col['appt_footprint'] = details_col_to_str(df_col['appt_footprint'])
        df_col['beddays_footprint'] = details_col_to_str(df_col['beddays_footprint'])
        # Explode the 'equipment' column
        exploded_df = df_col.explode('equipment')
        # Remove the 'event_details_key' and replace the index with hsi event details as indexes
        exploded_df = exploded_df.droplevel(level=1)
        exploded_df = exploded_df.set_index(
            ['event_name', 'module_name', 'treatment_id', 'facility_level', 'appt_footprint', 'beddays_footprint',
             'equipment'], append=True
        )
        # Sum values with the same multi-index (keep also empty indexes)
        exploded_df = exploded_df.groupby(level=exploded_df.index.names, dropna=False).sum()
        # Add the results for the run 'col' to final_df
        final_df = pd.concat([final_df, exploded_df], axis=1)

    # Replace NaN with 0
    final_df.fillna(0, inplace=True)
    # Save the detailed equipment catalogue
    final_df.to_csv(output_folder / output_detailed_file_name)
    print(f'{output_detailed_file_name} saved.')
    # ---

    # %% Catalog equipment by requested details
    equipment_counts_by_time_and_requested_details = final_df.copy()

    # Sum counts for each equipment with the same date, treatment id, and facility level (remaining indexes removed),
    # keeping only non-empty 'equipment' indexes
    to_be_grouped_by = ['date'] + catalog_by_details + ['equipment']
    equipment_counts_by_time_and_requested_details = equipment_counts_by_time_and_requested_details.groupby(
        to_be_grouped_by,
        dropna=True
    ).sum()

    if catalog_by_time == 'annual':
        # Sum counts annually
        equipment_counts_by_time_and_requested_details['year'] = \
            equipment_counts_by_time_and_requested_details.index.get_level_values('date').year
        equipment_counts_by_time_and_requested_details.set_index('year', append=True, inplace=True)
        equipment_counts_by_time_and_requested_details.index.droplevel('date')
        to_be_grouped_by = ['year'] + catalog_by_details + ['equipment']
        equipment_counts_by_time_and_requested_details = equipment_counts_by_time_and_requested_details.groupby(
            to_be_grouped_by
        ).sum()

    # Save the equipment counts CSV
    equipment_counts_by_time_and_requested_details.to_csv(output_folder / output_file_name)
    print(f'{output_file_name} saved.')
    # # ---

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    create_equipment_catalogues(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
    )
# NB. Edit run configuration, the Parameters: "./outputs/sejjej5@ucl.ac.uk/long_run_all_diseases-2023-09-04T233551Z"
