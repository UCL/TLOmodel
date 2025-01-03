import glob
import os
import re
import shutil
import zipfile
from pathlib import Path
import difflib
from scipy.spatial import KDTree

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import geopandas as gpd

ANC = True
Inpatient = False
monthly_cumulative = False
multiplier = 86400
years = range(2015, 2100)
month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] * len(years)
if monthly_cumulative:
    window_size = np.nan
else:
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
    grid_centroids = {}
    for file in glob.glob(os.path.join(nc_file_directory, "*.nc")):
        model = re.search(r'pr_day_(.*?)_' + scenario.replace('_', ''), file).group(1)
        data_per_model  = Dataset(file, mode='r')
        pr_data = data_per_model.variables['pr'][:]  # in kg m-2 s-1 = mm s-1 x 86400 to get to day
        lat_data = data_per_model.variables['lat'][:]
        long_data = data_per_model.variables['lon'][:]
        lon_grid, lat_grid = np.meshgrid(long_data, lat_data)
        centroids = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))

        # Store centroids
        grid_centroids[model] = centroids
        grid_dictionary = {}
        grid = 0
        for i in range(len(long_data)):
            for j in range(len(lat_data)):
                precip_data_for_grid = pr_data[:,j,i] # across all time points
                precip_data_for_grid = precip_data_for_grid * multiplier # to get from per second to per day
                grid_dictionary[grid] = precip_data_for_grid.data.values
                grid += 1
        data_by_model_and_grid[model] = grid_dictionary
    data_by_model_and_grid = pd.DataFrame.from_dict(data_by_model_and_grid)
    data_by_model_and_grid.to_csv(Path(scenario_directory)/"data_by_model_and_grid.csv")
    # now find the modal length of data for each model in the dictionary - this tells us the resolution, and most of the models are at different resolutions
    non_na_lengths = data_by_model_and_grid.count()
    # now drop all columns that are not that length, so the rest can be aggregated over
    modal_non_na_length = non_na_lengths.mode()[0]
    data_by_model_and_grid_same_length = data_by_model_and_grid.loc[:, non_na_lengths == modal_non_na_length]
    na_values = data_by_model_and_grid_same_length.isna()
    na_values_count = na_values.sum()
    data_by_model_and_grid_same_length = data_by_model_and_grid_same_length.dropna(axis=0)
    data_by_model_and_grid_same_length.to_csv(Path(scenario_directory)/"data_by_model_and_grid_modal_resolution.csv", index = True)

    # # Now loop over facilities to locate each one in a grid cell for each model
    # facilities_with_location = []
    # # see which facilities have reporting data and data on latitude and longitude
    # median_precipitation_by_facility = {}
    # percentiles_25_by_facility = {}
    # percentiles_75_by_facility = {}
    # cumulative_sum_window = {}
    # for reporting_facility in reporting_data.columns:
    #         grid_precipitation_for_facility = {}
    #         match_name, lat_for_facility, long_for_facility = get_facility_lat_long(reporting_facility, facilities_with_lat_long)
    #         if not np.isnan(long_for_facility) and not np.isnan(lat_for_facility):
    #                 facility_location = np.array([lat_for_facility, long_for_facility])
    #                 kd_trees_by_model = {}
    #                 for model in grid_centroids.keys():
    #                         if model in data_by_model_and_grid_same_length.columns:
    #                             centroids = grid_centroids[model]
    #                             kd_tree = KDTree(centroids)
    #                             distance, closest_grid_index = kd_tree.query(facility_location)
    #                             grid_precipitation_for_facility[model] = data_by_model_and_grid_same_length[model][closest_grid_index].data
    #                 first_model = next(iter(grid_precipitation_for_facility))
    #                 median_all_timepoints_for_facility = []
    #                 p25_all_timepoints_for_facility = []
    #                 p75_all_timepoints_for_facility = []
    #
    #                 for t in range(len(grid_precipitation_for_facility[first_model])): #all should be the same length
    #                     per_time_point_by_model = []
    #                     for precip_data in grid_precipitation_for_facility.values():
    #                         if len(precip_data) == len(grid_precipitation_for_facility[first_model]): # ensure same time resolution
    #                             per_time_point_by_model.append(precip_data[t])
    #                     p25_all_timepoints_for_facility.append(np.percentile(per_time_point_by_model, 25))
    #                     p75_all_timepoints_for_facility.append(np.percentile(per_time_point_by_model, 75))
    #                     median_all_timepoints_for_facility.append(np.median(per_time_point_by_model))
    #                     cumulative_sum_window[reporting_facility] = []
    #                     begin_day = 0
    #                     for month_idx, month_length in enumerate(month_lengths):
    #                         if monthly_cumulative:
    #                             window_size = month_length
    #                         days_for_grid = median_all_timepoints_for_facility[begin_day:begin_day + month_length]
    #                         cumulative_sums = [
    #                             sum(days_for_grid[day:day + window_size])
    #                             for day in range(month_length - window_size + 1)
    #                         ]
    #                         max_cumulative_sums = max(cumulative_sums)
    #                         cumulative_sum_window[reporting_facility].append(max_cumulative_sums)
    #                         begin_day += month_length
    #                 median_precipitation_by_facility[reporting_facility] = median_all_timepoints_for_facility
    #                 percentiles_25_by_facility[reporting_facility] = p25_all_timepoints_for_facility
    #                 percentiles_75_by_facility[reporting_facility] = p75_all_timepoints_for_facility
    #
    # weather_df_median = pd.DataFrame.from_dict(median_precipitation_by_facility, orient='index').T
    # weather_df_p25 = pd.DataFrame.from_dict(percentiles_25_by_facility, orient='index').T
    # weather_df_p75 = pd.DataFrame.from_dict(percentiles_75_by_facility, orient='index').T
    # df_cumulative_sum = pd.DataFrame.from_dict(cumulative_sum_window, orient='index').T
    # df_cumulative_sum.astype(float)
    # if ANC:
    #     weather_df_median.to_csv(Path(scenario_directory) / "median_daily_prediction_weather_by_facility_KDBall.csv", index=False)
    #     weather_df_p25.to_csv(Path(scenario_directory) / "p25_daily_prediction_weather_by_facility_KDBall.csv", index=False)
    #     weather_df_p75.to_csv(Path(scenario_directory) / "p75_daily_prediction_weather_by_facility_KDBall.csv", index=False)
    #     if monthly_cumulative:
    #         df_cumulative_sum.to_csv(Path(scenario_directory) / "monthly_cumulative_sum_by_facility_KDBall.csv")
    #
    #     else:
    #         df_cumulative_sum.to_csv(Path(scenario_directory) / f"{window_size}day__cumulative_sum_by_facility_KDBall.csv")
    # elif Inpatient:
    #     weather_df_median.to_csv(Path(scenario_directory) / "median_daily_prediction_weather_by_facility_with_inpatient_KDBall.csv",
    #                              index=False)
    #     weather_df_p25.to_csv(Path(scenario_directory) / "p25_daily_prediction_weather_by_facility_with_inpatient__KDBall.csv",
    #                           index=False)
    #     weather_df_p75.to_csv(Path(scenario_directory) / "p75_daily_prediction_weather_by_facility_with_inpatient__KDBall.csv",
    #                           index=False)
    #     if monthly_cumulative:
    #         df_cumulative_sum.to_csv(Path(scenario_directory) / "monthly_cumulative_sum_by_facility_with_inpatient__KDBall.csv")
    #
    #     else:
    #         df_cumulative_sum.to_csv(
    #             Path(scenario_directory) / f"{window_size}day__cumulative_sum_by_facility_with_inpatient__KDBall.csv")
    #
