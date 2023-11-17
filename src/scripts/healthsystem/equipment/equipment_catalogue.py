import argparse
from pathlib import Path

import pandas as pd

from tlo.analysis.utils import extract_results, load_pickled_dataframes


def get_annual_hsi_event_counts(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the simulated annual counts of all the hsi event details logged (details as keys)
    for each simulated year.
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
        custom_generate_series=get_hsi_event_counts
        )


def get_hsi_event_keys(results_folder: Path):
    return load_pickled_dataframes(results_folder, 0, 0, "tlo.methods.healthsystem.summary")[
        "tlo.methods.healthsystem.summary"
    ]["hsi_event_details"]["hsi_event_key_to_event_details"][0]


def create_equipment_catalogues(results_folder: Path, output_folder: Path):

    # Declare output file names
    output_detailed_file_name = 'equipment_annual_counts__all_event_details.csv'
    output_file_name = 'equipment_annual_counts__by_Date_EventName_FacLevel.csv'

    # %% Catalogue equipment by all HSI event details
    sim_equipment = get_annual_hsi_event_counts(results_folder)
    sim_equipment_df = pd.DataFrame(sim_equipment)
    sim_equipment_df.fillna(0, inplace=True)

    hsi_event_keys = get_hsi_event_keys(results_folder)

    decoded_keys = sim_equipment_df.index.get_level_values(1).astype(str).map(hsi_event_keys)
    sim_equipment_df = pd.concat([sim_equipment_df, pd.DataFrame(decoded_keys.tolist(), index=sim_equipment_df.index)], axis=1)
    # Make values in 'appt_footprint', and 'beddays_footprint' columns to be string
    sim_equipment_df['appt_footprint'] = sim_equipment_df['appt_footprint'].apply(lambda x: ', '.join(map(str, x)))
    sim_equipment_df['beddays_footprint'] = sim_equipment_df['beddays_footprint'].apply(lambda x: ', '.join(map(str, x)))
    # Explode the 'equipment' column
    exploded_df = sim_equipment_df.explode('equipment')
    # Remove the 'event_details_key' and replace the index with hsi event details as indexes
    exploded_df = exploded_df.droplevel(level=1)
    exploded_df = exploded_df.set_index(
        ['event_name', 'module_name', 'treatment_id', 'facility_level', 'appt_footprint', 'beddays_footprint',
         'equipment'], append=True
    )
    # Sum values with the same multi-index
    exploded_df = exploded_df.groupby(exploded_df.index).sum()
    # Convert the index back to a MultiIndex
    exploded_df.index = pd.MultiIndex.from_tuples(exploded_df.index)
    exploded_df.index.names = \
        ['date', 'event_name', 'module_name', 'treatment_id', 'facility_level', 'appt_footprint', 'beddays_footprint',
         'equipment']

    # Save the detailed equipment catalogue by levels
    exploded_df.to_csv(output_folder / output_detailed_file_name)
    print(f'{output_detailed_file_name} saved.')
    # ---

    # %% Catalogue equipment by Facility Levels and HSI Event Names

    equipment_counts_by_date_hsi_name_level_df = exploded_df.copy()

    # Sum counts for each equipment with the same date, hsi event name, and level (remaining indexes removed)
    # equipment_counts_by_date_hsi_name_level_df
    equipment_counts_by_date_hsi_name_level_df = \
        equipment_counts_by_date_hsi_name_level_df.groupby(['date', 'event_name', 'facility_level', 'equipment']).sum()

    # Save the CSV equipment counts
    equipment_counts_by_date_hsi_name_level_df.to_csv(output_folder / output_file_name)
    print(f'{output_file_name} saved.')
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
