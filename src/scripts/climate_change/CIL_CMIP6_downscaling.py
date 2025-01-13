#!/usr/bin/env python
# coding: utf-8

# From https://planetarycomputer.microsoft.com/dataset/cil-gdpcir-cc0#Ensemble-example

# In[1]:


import planetary_computer
import pystac_client

import xarray as xr
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

import os
import re
import glob
import shutil
import zipfile
from pathlib import Path

import difflib
from scipy.spatial import KDTree

import matplotlib.pyplot as plt
import geopandas as gpd
import regionmask
import cartopy.crs as ccrs

from netCDF4 import Dataset

from carbonplan import styles  # noqa: F401
import intake
import cmip6_downscaling


# Load and organise data

# In[52]:


import xarray as xr
import pandas as pd
from pystac_client import Client
from planetary_computer import sign_inplace
from tqdm import tqdm



ANC = True
Inpatient = False
multiplier = 1 # no need for multiplier
years = range(2025, 2071) # final date is 1st Jan 2100
month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] * len(years)
window_size = 5

if ANC:
    reporting_data = pd.read_csv(
        "/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_ANC_by_smaller_facility_lm.csv")
elif Inpatient:
    reporting_data = pd.read_csv(
        "/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_Inpatient_by_smaller_facility_lm.csv")
general_facilities = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.shp")

facilities_with_lat_long = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_lat_long_region.csv")


# In[3]:


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

def get_facility_lat_long(reporting_facility, facilities_df, cutoff=0.90, n_matches=3):
    """
    Function to find the closest matching facility name and return its latitude and longitude.

    Parameters:
    - reporting_facility: The facility name for which latitude and longitude are needed.
    - facilities_df : DataFrame containing facility names ('Fname') and their corresponding latitudes ('A109__Latitude') and longitudes ('A109__Longitude').
    - cutoff: The minimum similarity score for a match. Default is 0.90.
    - n_matches: The maximum number of matches to consider. Default is 3.

    Returns: match_name, lat_for_facility, long_for_facility

    """
    matching_facility_name = difflib.get_close_matches(reporting_facility, facilities_df['Fname'], n=n_matches,
                                                       cutoff=cutoff)

    if matching_facility_name:
        match_name = matching_facility_name[0]  # Access the string directly
        lat_for_facility = facilities_df.loc[facilities_df['Fname'] == match_name, "A109__Latitude"].iloc[0]
        long_for_facility = facilities_df.loc[facilities_df['Fname'] == match_name, "A109__Longitude"].iloc[0]
        return match_name, lat_for_facility, long_for_facility
    else:
        return np.nan, np.nan, np.nan

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


# In[ ]:


base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/"
nc_file_directory = os.path.join(base_dir, 'nc_files')
# NB these are daily
scenarios = ["ssp245", "ssp585"]

data_by_model_and_grid = {}
for scenario in scenarios:
    print(scenario)
    scenario_directory = os.path.join(base_dir, scenario)

    grid_centroids = {}
    cumulative_sum_by_models = {}
    file_path_downscaled = f"/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/"
    output_file = f"CIL_combined_{scenario}_2025_2070.nc"
    file_pattern = os.path.join(file_path_downscaled, "CIL_subset_ssp245_*.nc")
    data_all_models = xr.open_mfdataset(file_pattern, combine='nested', concat_dim="time")
    data_all_models.compute()

    data_all_models.to_netcdf(output_file)
    #data_all_models = xr.open_dataset(file_path_downscaled)

    ## Get models of interest - min, med, max
    # Assuming 'pr' is the variable representing precipitation in the dataset
    pr_aggregated = data_all_models.mean(dim=["lat", "lon", "time"], skipna=True)  # Work with the 'pr' DataArray

    # Find the model with the lowest value
    min_model_object = pr_aggregated['pr'].idxmin(dim="model")
    min_model = min_model_object.values.item()
    # Find the model with the median value
    sorted_models = pr_aggregated.sortby("model")
    n_models = len(pr_aggregated.model)
    median_index = n_models // 2
    median_model_object = sorted_models["model"][median_index]
    print(median_model_object)
    median_model = median_model_object.values.item()
    print(median_model)
    # Find the model with the highest value
    max_model_object = pr_aggregated['pr'].idxmax(dim="model")
    max_model = max_model_object.values.item()

    models_of_interest = [min_model, median_model, max_model]
    #models_of_interest = [median_model]

    print("Models of interest", models_of_interest)
    # see which facilities have reporting data and data on latitude and longitude
    weather_df_lowest_window = pd.DataFrame()
    weather_df_median_window = pd.DataFrame()
    weather_df_highest_window = pd.DataFrame()

    weather_df_lowest_monthly = pd.DataFrame()
    weather_df_median_monthly = pd.DataFrame()
    weather_df_highest_monthly = pd.DataFrame()
    for model in models_of_interest:
        data_per_model = data_all_models.sel(model=model)
        pr_data = data_per_model.variables['pr'][:]  # in kg m-2 s-1 = mm s-1 x 86400 to get to day
        lat_data = data_per_model.variables['lat'][:]
        lon_data = data_per_model.variables['lon'][:]
        lon_grid, lat_grid = np.meshgrid(lon_data, lat_data)
        centroids = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))

        # Store centroids
        grid_centroids[model] = centroids
        grid_dictionary = {}
        grid = 0
        for i in lat_data:
            for j in lon_data:
                precip_data_for_grid = data_per_model.sel(lat=i, lon=j, method="nearest")  # across all time points
                grid_dictionary[grid] = precip_data_for_grid.pr.data
                grid += 1
        data_by_model_and_grid[model] = grid_dictionary

    for reporting_facility in reporting_data.columns:
        print(reporting_facility)
        grid_precipitation_for_facility = {}
        match_name, lat_for_facility, long_for_facility = get_facility_lat_long(reporting_facility, facilities_with_lat_long)
        if not np.isnan(long_for_facility) and not np.isnan(lat_for_facility):
            facility_location = np.array([lat_for_facility, long_for_facility])
            kd_trees_by_model = {}

            # Loop over each model of interest
            for model in models_of_interest:
                centroids = grid_centroids[model]
                kd_tree = KDTree(centroids)
                distance, closest_grid_index = kd_tree.query(facility_location)
                grid_precipitation_for_facility[model] = data_by_model_and_grid[model][closest_grid_index]

                cumulative_sum_monthly = []
                cumulative_sum_window = []

                begin_day = 0
                # Calculate monthly cumulative sums
                for month_idx, month_length in enumerate(month_lengths):
                    days_for_grid_monthly = grid_precipitation_for_facility[model][begin_day:begin_day + month_length]
                    cumulative_sums_monthly = [
                        sum(days_for_grid_monthly)
                    ]
                    max_cumulative_sums_monthly = max(cumulative_sums_monthly)
                    cumulative_sum_monthly.append(max_cumulative_sums_monthly)
                    begin_day += month_length

                begin_day = 0
                # Calculate windowed cumulative sums
                for month_idx, month_length in enumerate(month_lengths):
                    days_for_grid_window = grid_precipitation_for_facility[model][begin_day:begin_day + month_length]

                    cumulative_sums_window = [
                        sum(days_for_grid_window[day:day + window_size])
                        for day in range(month_length - window_size + 1)
                    ]

                    max_cumulative_sums_window = max(cumulative_sums_window)
                    cumulative_sum_window.append(max_cumulative_sums_window)
                    begin_day += month_length

                # Assign the calculated data to the correct dataframe based on the model
                if model == min_model:
                    weather_df_lowest_monthly[reporting_facility] = cumulative_sum_monthly
                    weather_df_lowest_window[reporting_facility] = cumulative_sum_window
                elif model == median_model:
                    weather_df_median_monthly[reporting_facility] = cumulative_sum_monthly
                    weather_df_median_window[reporting_facility] = cumulative_sum_window
                elif model == max_model:
                    weather_df_highest_monthly[reporting_facility] = cumulative_sum_monthly
                    weather_df_highest_window[reporting_facility] = cumulative_sum_window

    if ANC:
            weather_df_lowest_window.to_csv(Path(scenario_directory) / f"lowest_model_daily_prediction_weather_by_facility_KDBall_ANC_downscaled_CIL_{scenario}.csv", index=False)
            weather_df_median_window.to_csv(Path(scenario_directory) / f"median_model_daily_prediction_weather_by_facility_KDBall_ANC_downscaled_CIL_{scenario}.csv", index=False)
            weather_df_highest_window.to_csv(Path(scenario_directory) / f"highest_model_daily_prediction_weather_by_facility_KDBall_ANC_downscaled_CIL_{scenario}.csv", index=False)

            weather_df_lowest_monthly.to_csv(Path(scenario_directory) / f"lowest_model_monthly_prediction_weather_by_facility_KDBall_ANC_downscaled_CIL_{scenario}.csv", index=False)
            weather_df_median_monthly.to_csv(Path(scenario_directory) / f"median_model_monthly_prediction_weather_by_facility_KDBall_ANC_downscaled_CIL_{scenario}.csv", index=False)
            weather_df_highest_monthly.to_csv(Path(scenario_directory) / f"highest_model_monthly_prediction_weather_by_facility_KDBall_ANC_downscaled_CIL_{scenario}.csv", index=False)


# In[27]:


data_all_models.mean(dim=["lat", "lon", "time"], skipna=True)


# In[ ]:




