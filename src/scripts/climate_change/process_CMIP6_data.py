import glob
import os
import re
import shutil
import zipfile
from pathlib import Path
import difflib

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import geopandas as gpd

five_day = False
monthly_cumulative = True
multiplier = 86400
years = range(2015, 2100)
reporting_data = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_ANC_by_smaller_facility_lm.csv")

general_facilities = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.shp")

facilities_with_lat_long = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_lat_long_region.csv")


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


# unzip files and extract the netCDF files

base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/"

scenarios = ["ssp1_1_9", "ssp2_4_5"]
for scenario in scenarios:
    scenario_directory = os.path.join(base_dir, scenario)
    unzip_all_in_directory(scenario_directory)
    extract_nc_files_from_unzipped_folders(scenario_directory)


# Put all into one csv file
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
        #time_data = data_per_model.variables['time'] 31046 days
        grid_dictionary = {}
        grid = 0
        for i in range(len(long_data)):
            for j in range(len(lat_data)):
                precip_data_for_grid = pr_data[:,j,i] # across all time points
                precip_data_for_grid = precip_data_for_grid * multiplier # to get from per second to per day
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

    mean_precip_by_timepoint = {}
    median_precip_by_timepoint = {}
    percentile_25_by_timepoint = {}
    percentile_75_by_timepoint = {}

    # Calculate the statistics for each grid
    for grid in range(len(data_by_model_and_grid_same_length)):
        timepoint_values = []
        for model in data_by_model_and_grid_same_length.columns:
            model_data = data_by_model_and_grid_same_length.loc[grid, model]
            if not timepoint_values:
                timepoint_values = [[] for _ in range(len(model_data))]
            for i, value in enumerate(model_data):
                if not np.ma.is_masked(value):
                    timepoint_values[i].append(value)
        # Calculate and store statistics for each grid and timepoint
        mean_precip_by_timepoint[grid] = [np.mean(tp_values) if tp_values else np.nan for tp_values in timepoint_values]
        median_precip_by_timepoint[grid] = [np.median(tp_values) if tp_values else np.nan for tp_values in
                                            timepoint_values]
        percentile_25_by_timepoint[grid] = [np.percentile(tp_values, 25) if tp_values else np.nan for tp_values in
                                            timepoint_values]
        percentile_75_by_timepoint[grid] = [np.percentile(tp_values, 75) if tp_values else np.nan for tp_values in
                                            timepoint_values]

    # Convert each dictionary to a DataFrame and save to CSV
    mean_df = pd.DataFrame.from_dict(mean_precip_by_timepoint, orient='index')
    mean_df.to_csv(Path(scenario_directory) / "mean_projected_precip_by_timepoint_modal_resolution.csv")

    median_df = pd.DataFrame.from_dict(median_precip_by_timepoint, orient='index')
    median_df.to_csv(Path(scenario_directory) / "median_projected_precip_by_timepoint_modal_resolution.csv")

    percentile_25_df = pd.DataFrame.from_dict(percentile_25_by_timepoint, orient='index')
    percentile_25_df.to_csv(
        Path(scenario_directory) / "percentile_25_projected_precip_by_timepoint_modal_resolution.csv")

    percentile_75_df = pd.DataFrame.from_dict(percentile_75_by_timepoint, orient='index')
    percentile_75_df.to_csv(
        Path(scenario_directory) / "percentile_75_projected_precip_by_timepoint_modal_resolution.csv")

    ## now do monthly 5-day max

    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]*len(years)

    if five_day:
        window_size = 5  # Set the window size to 5 days
        cumulative_sum_by_grid = {}
        mean_precip_by_timepoint = pd.DataFrame.from_dict(mean_precip_by_timepoint)
        for grid in range(mean_precip_by_timepoint.shape[1]): # columns are grids
                    pr_data_for_grid = mean_precip_by_timepoint[grid]
                    if grid not in cumulative_sum_by_grid:
                        cumulative_sum_by_grid[grid] = []
                    begin_day = 0
                    for month_idx, month_length in enumerate(month_lengths):
                        days_for_grid = pr_data_for_grid[begin_day:begin_day + month_length]
                        cumulative_sums = []
                        for day in range(month_length - window_size + 1):
                            window_sum = sum(days_for_grid[day:day + window_size])
                            cumulative_sums.append(window_sum)
                        max_cumulative_sums = max(cumulative_sums)
                        cumulative_sum_by_grid[grid].append(max_cumulative_sums)
                        begin_day += month_length
        df_cumulative_sum = pd.DataFrame.from_dict(cumulative_sum_by_grid, orient='index')
        df_cumulative_sum.astype(float)
        df_cumulative_sum = df_cumulative_sum.T
        df_cumulative_sum.to_csv(Path(scenario_directory) / "five_day_cumulative_sum_by_grid.csv")


        ############### NOW HAVE LAT/LONG OF FACILITIES #####################
        facilities_with_location = []
        cumulative_sum_by_facility = {}
        for reporting_facility in reporting_data.columns:
            match_name, lat_for_facility, long_for_facility = get_facility_lat_long(reporting_facility, facilities_with_lat_long)

            index_for_x = ((long_data - long_for_facility)**2).argmin()
            index_for_y= ((lat_data - lat_for_facility)**2).argmin()
                # which grid number is it
            grid = index_for_x * index_for_y + 1
            cumulative_sum_by_facility[reporting_facility] = df_cumulative_sum[grid]  # across all time points

            ## below are not in facilities file?
            if reporting_facility == "Central East Zone":
                grid = general_facilities[general_facilities["District"] == "Nkhotakota"]["Grid_Index"].iloc[
                    0]  # furtherst east zone
                cumulative_sum_by_facility[reporting_facility] = df_cumulative_sum[grid]
            elif (reporting_facility == "Central Hospital"):
                grid = general_facilities[general_facilities["District"] == "Lilongwe City"]["Grid_Index"].iloc[
                    0]  # all labelled X City will be in the same grid
                cumulative_sum_by_facility[reporting_facility] = df_cumulative_sum[grid]
            else:
                continue


        ### Get data ready for linear regression between reporting and weather data
        weather_df = pd.DataFrame.from_dict(cumulative_sum_by_facility, orient='index').T
        weather_df.columns = facilities_with_location
        weather_df.to_csv(Path(scenario_directory)/"prediction_weather_by_smaller_facilities_with_ANC_lm.csv")

    elif monthly_cumulative:
        cumulative_sum_by_grid = {}
        mean_precip_by_timepoint = pd.DataFrame.from_dict(mean_precip_by_timepoint)
        cumulative_sum_by_facility = {}
        for grid in range(mean_precip_by_timepoint.shape[1]):  # columns are grids
                pr_data_for_grid = mean_precip_by_timepoint[grid]
                if grid not in cumulative_sum_by_grid:
                    cumulative_sum_by_grid[grid] = []
                begin_day = 0
                for month_idx, month_length in enumerate(month_lengths):
                    window_size = month_length  # Set the window size to 5 days
                    days_for_grid = pr_data_for_grid[begin_day:begin_day + month_length]
                    cumulative_sums = []
                    for day in range(month_length - window_size + 1):
                        window_sum = sum(days_for_grid[day:day + window_size])
                        cumulative_sums.append(window_sum)
                    max_cumulative_sums = max(cumulative_sums)
                    cumulative_sum_by_grid[grid].append(max_cumulative_sums)
                    begin_day += month_length

        df_cumulative_sum = pd.DataFrame.from_dict(cumulative_sum_by_grid, orient='index')
        df_cumulative_sum.astype(float)
        df_cumulative_sum = df_cumulative_sum.T
        df_cumulative_sum.to_csv(Path(scenario_directory) / "monthly_cumulative_sum_by_grid.csv")

        ##

        ############### NOW HAVE LAT/LONG OF FACILITIES #####################
        facilities_with_location = []
        for reporting_facility in reporting_data.columns:
            match_name, lat_for_facility, long_for_facility = get_facility_lat_long(reporting_facility, facilities_with_lat_long)

            index_for_x = ((long_data - long_for_facility) ** 2).argmin()
            index_for_y = ((lat_data - lat_for_facility) ** 2).argmin()
                # which grid number is it
            grid = index_for_x*index_for_y + 1
            cumulative_sum_by_facility[reporting_facility] = df_cumulative_sum[grid]  # across all time points

            ## below are not in facilities file?
            if reporting_facility == "Central East Zone":
                grid = general_facilities[general_facilities["District"] == "Nkhotakota"]["Grid_Index"].iloc[
                    0]  # furtherst east zone
                cumulative_sum_by_facility[reporting_facility] = df_cumulative_sum[grid]
            elif (reporting_facility == "Central Hospital"):
                grid = general_facilities[general_facilities["District"] == "Lilongwe City"]["Grid_Index"].iloc[
                    0]  # all labelled X City will be in the same grid
                cumulative_sum_by_facility[reporting_facility] = df_cumulative_sum[grid]
            else:
                continue

        ### Get data ready for linear regression between reporting and weather data
        weather_df = pd.DataFrame.from_dict(cumulative_sum_by_facility, orient='index').T
        weather_df.columns = facilities_with_location
        weather_df.to_csv(Path(scenario_directory) / "prediction_weather_monthly_by_smaller_facilities_with_ANC_lm.csv")




