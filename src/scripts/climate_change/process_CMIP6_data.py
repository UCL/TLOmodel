import glob
import os
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset


def unzip_all_in_directory(directory):
    """
    Unzips all .zip files in the specified directory, extracting each into a separate folder.

    Parameters:
        directory (str): The path to the folder containing the .zip files.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.zip'):
            file_path = os.path.join(directory, filename)
            extract_dir = os.path.join(directory, filename[:-4])
            os.makedirs(extract_dir, exist_ok=True)

            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile:
                print(f"Skipped {filename}: not a valid zip file.")


def extract_nc_files_from_unzipped_folders(directory):
    """
    Searches for .nc files in the specified directory and all its subfolders,
    and copies them to the output directory, maintaining the folder structure.

    Parameters:
        directory (str): The path to the folder containing the unzipped folders.
    """
    output_directory = os.path.join(directory, 'nc_files')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for root, _, files in os.walk(directory):
        # Skip the output directory to prevent recursive copying
        if root == output_directory:
            continue

        for filename in files:
            if filename.endswith('.nc'):
                source_file_path = os.path.join(root, filename)
                destination_file_path = os.path.join(output_directory, filename)

                # Only copy if the file does not already exist in the output directory
                if not os.path.exists(destination_file_path):
                    shutil.copy2(source_file_path, output_directory)


# unzip files and extract the netCDF files

base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/"

scenarios = ["ssp1_1_9", "ssp2_4_5"]
for scenario in scenarios:
    scenario_directory = os.path.join(base_dir, scenario)
    unzip_all_in_directory(scenario_directory)
    extract_nc_files_from_unzipped_folders(scenario_directory)


# Put all into one csv file
base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/"

file_list = glob.glob(os.path.join(base_dir, "*.nc"))
data_by_model_and_grid = {}

for scenario in scenarios:
    print(scenario)
    scenario_directory = os.path.join(base_dir, scenario)
    nc_file_directory = os.path.join(scenario_directory, 'nc_files')

    for file in glob.glob(os.path.join(nc_file_directory, "*.nc")):
        model = re.search(r'pr_day_(.*?)_' + scenario.replace('_', ''), file).group(1)
        data_per_model  = Dataset(file, mode='r')
        pr_data = data_per_model.variables['pr'][:]  # in kg m-2 s-1 = mm s-1 x 86400 to get to day
        lat_data = data_per_model.variables['lat'][:]
        long_data = data_per_model.variables['lon'][:]
        grid_dictionary = {}
        grid = 0
        for i in range(len(long_data)):
            for j in range(len(lat_data)):
                precip_data_for_grid = pr_data[:,j,i] # across all time points
                precip_data_for_grid = precip_data_for_grid * 86400 # to get from per second to per day
                grid_dictionary[grid] = precip_data_for_grid
                grid += 1
        data_by_model_and_grid[model] = grid_dictionary
    data_by_model_and_grid = pd.DataFrame.from_dict(data_by_model_and_grid)
    data_by_model_and_grid.to_csv(Path(scenario_directory)/"data_by_model_and_grid.csv")

    # now find the modal length of data for each model in the dictionary - this tells us the resolution, and most of the models are at different resolutions
    non_na_lengths = data_by_model_and_grid.count()
    # now drop all columns that are not that length, so the rest can be aggregated over
    modal_non_na_length = non_na_lengths.mode()[0]
    data_by_model_and_grid_same_length = data_by_model_and_grid.loc[:, non_na_lengths == modal_non_na_length]
    data_by_model_and_grid_same_length = data_by_model_and_grid_same_length.dropna(axis=0)
    data_by_model_and_grid_same_length.to_csv(Path(scenario_directory)/"data_by_model_and_grid_modal_resolution.csv")

    # Now average across each time point for each grid square.
    precip_by_timepoint = {}
    for grid in range(len(data_by_model_and_grid_same_length)):
        timepoint_values = []
        for model in data_by_model_and_grid_same_length.columns:
            model_data = data_by_model_and_grid_same_length.loc[grid, model]
            if not timepoint_values:
                timepoint_values = [[] for _ in range(len(model_data))]
            for i, value in enumerate(model_data):
                if not np.ma.is_masked(value):
                    timepoint_values[i].append(value)

        precip_by_timepoint[grid] = [np.mean(tp_values) if tp_values else np.nan for tp_values in timepoint_values]

    precip_df = pd.DataFrame.from_dict(precip_by_timepoint, orient='index')
    average_projected_precip = precip_df.mean(axis=0)
    average_projected_precip.to_csv(Path(scenario_directory)/"mean_projected_precip_by_timepoint_modal_resolution.csv")




