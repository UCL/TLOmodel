# Put all into one csv file

import glob
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from netCDF4 import Dataset

base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/"

#### Multiple files
file_list = glob.glob(os.path.join(base_dir, "*.nc"))
data_by_model_and_grid = {}
models = ["cams_csm1_0", "ipsl_cm6a_lr", "miroc6","miroc_es2l", "mri_esm2_0", "canesm5", "cnrm_esm2_1", "ec_earth3", "ec_earth3_veg_lr", "fgoals_g3", "gfdl_esm4", "ukesm1_0_ll"]
model = 0
for file in glob.glob(os.path.join(base_dir, "*.nc")):
    data_per_model  = Dataset(file, mode='r')
    pr_data = data_per_model.variables['pr'][:]  # in kg m-2 s-1 = mm s-1 x 86400 to get to day
    lat_data = data_per_model.variables['lat'][:]
    long_data = data_per_model.variables['lon'][:]
    grid_dictionary = {}
    grid = 0
    for i in range(len(long_data)):
        for j in range(len(lat_data)):
            precip_data_for_grid = pr_data[:,i,j] # across all time points
            precip_data_for_grid = precip_data_for_grid * 86400 # to get from per second to per day
            grid_dictionary[grid] = precip_data_for_grid
            grid += 1
    data_by_model_and_grid[models[model]] = grid_dictionary
    model += 1
print(data_by_model_and_grid)
data_by_model_and_grid = pd.DataFrame.from_dict(data_by_model_and_grid)
data_by_model_and_grid.to_csv(Path(scenario_directory)/"data_by_model_and_grid.csv")
