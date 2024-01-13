import argparse
import warnings
from pathlib import Path

import pandas as pd

from tlo.analysis.utils import extract_results

# TODO: make these to be arguments of called fnc
# %%% TO SET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Declare whether to scale the counts to Malawi population size
# (True/False)
do_scaling = True
# Declare as a list by which hsi event details you want the equipment be grouped in the catalogue (choose any number)
# (event details: 'event_name', 'module_name', 'treatment_id', 'facility_level', 'appt_footprint', 'beddays_footprint')
catalog_by_details = ['treatment_id', 'facility_level']
# Declare which time period you want the equipment be grouped in the catalogue (choose only one)
# (periods: 'monthly', 'annual')
catalog_by_time = 'annual'
# Suffix for output file names
suffix_file_names = '__5y_20Kpop_10runs'
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# TODO: Could I have use the bin_hsi_event_details from src/tlo/analysis/utils.py instead? If so, how?
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
    # %%% Verify inputs are as expected
    assert isinstance(do_scaling, bool), "The input parameter 'do_scaling' must be a boolean (True or False)"
    assert isinstance(catalog_by_details, list), "The input parameter 'catalog_by_details' must be a list"
    event_details = \
        {'event_name', 'module_name', 'treatment_id', 'facility_level', 'appt_footprint', 'beddays_footprint'}
    for item in catalog_by_details:
        assert isinstance(item, str) and item in event_details, \
            f"Each element in the input list 'catalog_by_details' must be a string and be one of the details:\n" \
            f"{event_details}"
    assert catalog_by_time in {'monthly', 'annual'}, \
        "The input parameter 'catalog_by_time' must be one of the strings ('monthly' or 'annual')"
    # ---

    # %%% Set output file names
    # detailed CSV name
    output_detailed_file_name = 'equipment_monthly_counts__all_event_details' + suffix_file_names + '.csv'
    # requested details only CSV name
    time_index = 'year' if catalog_by_time == 'annual' else 'date'
    output_focused_file_name = \
        'equipment_' + catalog_by_time + '_counts__by_' + time_index + '_' + '_'.join(catalog_by_details) + \
        suffix_file_names + '.csv'
    output_summary_file_name = 'equipment_summary__module_name_event_name_treatment_id' + suffix_file_names + '.csv'
    # ---

    # %% Catalog equipment by all HSI event details
    sim_equipment = get_monthly_hsi_event_counts(results_folder)
    sim_equipment_df = pd.DataFrame(sim_equipment)
    hsi_event_keys = get_hsi_event_keys_all_runs(results_folder)

    final_df = pd.DataFrame()

    def details_col_to_str(details_col):
        return details_col.apply(lambda x: ', '.join(map(str, x)))

    def lists_of_strings_to_strings_of_list(list_of_strings_col):
        return list_of_strings_col.apply(lambda x: "['" + "', '".join(map(str, x)) + "']")

    def strings_of_list_to_lists_of_strings(strings_of_list_col):
        return strings_of_list_col.apply(lambda x: x.strip('][').split(', '))

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
        # Make values in 'appt_footprint', 'beddays_footprint', and 'equipment' columns to be string
        df_col['appt_footprint'] = details_col_to_str(df_col['appt_footprint'])
        df_col['beddays_footprint'] = details_col_to_str(df_col['beddays_footprint'])
        df_col['equipment'] = lists_of_strings_to_strings_of_list(df_col['equipment'])
        df_col = (df_col.droplevel(level=1)
                  .set_index(['module_name', 'event_name', 'treatment_id', 'facility_level', 'appt_footprint',
                              'beddays_footprint', 'equipment'], append=True))
        final_df = pd.concat([final_df, df_col], axis=1)

    # Replace NaN with 0
    final_df.fillna(0, inplace=True)
    final_df.sort_index(inplace=True)
    # Save the detailed equipment catalogue
    final_df.to_csv(output_folder / output_detailed_file_name)
    print(f'{output_detailed_file_name} saved.')
    # ---

    # %% Catalog equipment summary
    equipment_summary = final_df.copy()
    equipment_summary = equipment_summary.groupby(['module_name', 'event_name', 'treatment_id', 'equipment']).sum()
    equipment_summary = \
        equipment_summary.reset_index().set_index(['module_name', 'event_name', 'treatment_id', 'equipment'])
    # Save the summary equipment catalogue
    equipment_summary.index.to_frame().to_csv(output_folder / output_summary_file_name, index=False)
    print(f'{output_summary_file_name} saved.')
    # ---

    # %% Catalog equipment by requested details
    equipment_counts_by_time_and_requested_details = final_df.copy()

    # Sum counts for each equipment set with the same date, treatment id, and facility level
    # (remaining indexes removed), keeping only non-empty 'equipment' indexes
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

    # Remove rows with no equipment used
    equipment_counts_by_time_and_requested_details.drop("['']", level='equipment', axis=0, inplace=True)
    # Split the equipment by an item per row
    equipment_counts_by_time_and_requested_details['equipment'] = \
        equipment_counts_by_time_and_requested_details.index.get_level_values('equipment')
    equipment_counts_by_time_and_requested_details.index = \
        equipment_counts_by_time_and_requested_details.index.droplevel('equipment')
    equipment_counts_by_time_and_requested_details['equipment'] = strings_of_list_to_lists_of_strings(
        equipment_counts_by_time_and_requested_details['equipment']
    )
    exploded_df = equipment_counts_by_time_and_requested_details.explode('equipment')
    exploded_df = exploded_df.set_index(['equipment'], append=True)
    # Sum values with the same multi-index
    exploded_df = exploded_df.groupby(level=exploded_df.index.names).sum()

    # Save the equipment counts CSV
    exploded_df.to_csv(output_folder / output_focused_file_name)
    print(f'{output_focused_file_name} saved.')
    # ---

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
