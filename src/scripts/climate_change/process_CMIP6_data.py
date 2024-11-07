import glob
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from netCDF4 import Dataset


def unzip_all_in_directory(directory):
    """
    Unzips all .zip files in the specified directory, extracting each into a separate folder.

    Parameters:
        directory (str): The path to the folder containing the .zip files.
    """
    for filename in os.listdir(directory):
        print(f"Processing {filename}")
        if filename.endswith('.zip'):
            file_path = os.path.join(directory, filename)
            extract_dir = os.path.join(directory, filename[:-4])
            os.makedirs(extract_dir, exist_ok=True)

            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Extracted {filename} to {extract_dir}")
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

scenarios = ["ssp1_1_9"] #,"ssp1_2_6", "ssp4_3_4", "ssp5_3_4OS", "ssp2_4_5", "ssp4_6_0", "ssp3_7_0", "ssp5_8_5"]
scenarios = ["ssp2_4_5"]
for scenario in scenarios:
    scenario_directory = os.path.join(base_dir, scenario)
    unzip_all_in_directory(scenario_directory)
    extract_nc_files_from_unzipped_folders(scenario_directory)


# Put all into one csv file
base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/"

scenario = scenarios[0]
scenario_directory = os.path.join(base_dir, scenario)
nc_file_directory = os.path.join(scenario_directory, 'nc_files')
#### Multiple files
file_list = glob.glob(os.path.join(nc_file_directory, "*.nc"))
data_by_model_and_grid = {}
if scenario == "ssp1_1_9":
    models = ["cams_csm1_0", "ipsl_cm6a_lr", "miroc6", "miroc_es2l", "mri_esm2_0", "canesm5", "cnrm_esm2_1",
              "ec_earth3", "ec_earth3_veg_lr", "fgoals_g3", "gfdl_esm4", "ukesm1_0_ll"]
elif scenario == "ssp2_4_5":
    models = ["access_cm2", "awi_cm_1_1_mr", "bcc_csm2_mr", "cams_csm1_0", "cmcc_esm2", "hadgem3_gc31_ll",
                     "iitm_esm", "inm_cm5_0", "ipsl_cm6a_lr", "kiost_esm", "miroc6", "miroc_es2l",
                     "mri_esm2_0", "noresm2_mm", "canesm5", "cesm2", "cmcc_cm2_sr5", "cnrm_cm6_1", "cnrm_esm2_1",
                     "ec_earth3_cc", "ec_earth3_veg_lr", "fgoals_g3",
                     "gfdl_esm4", "inm_cm4_8", "kace_1_0_g", "mpi_esm1_2_lr", "nesm3", "noresm2_lm", "ukesm1_0_ll"]
model = 0
for file in file_list:
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
    data_by_model_and_grid[models[model]] = grid_dictionary
    model += 1
data_by_model_and_grid = pd.DataFrame.from_dict(data_by_model_and_grid)

data_by_model_and_grid.to_csv(Path(scenario_directory)/"data_by_model_and_grid.csv")





