import glob
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from netCDF4 import Dataset



base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_maximum"

years = range(2011, 2025)

# access each nc file, and calculate a 5-day cumulative maximum for each month

for year in years:
    year_directory = os.path.join(base_dir, str(year))
    precip_datafile = glob.glob(os.path.join(year_directory, "*.nc")) # only one reanalysis dataset, do not need to loop over files
    data_per_model  = Dataset(precip_datafile[0], mode='r')
    print(data_per_model.variables.keys())
    pr_data = data_per_model.variables['tp'][:]  # in kg m-2 s-1 = mm s-1 x 86400 to get to day
    lat_data = data_per_model.variables['latitude'][:]
    long_data = data_per_model.variables['longitude'][:]
    time_days = data_per_model.variables['valid_time'][:]
    for j in len(long_data):
        for i in len(lat_data):
            pr_data_for_square = pr_data[:,i,j]
    #### Multiple files
    #file_list = glob.glob(os.path.join(nc_file_directory, "*.nc"))
