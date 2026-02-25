import glob
import os
import re
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import KDTree

monthly_cumulative = False
multiplier = 86400
years = range(2025, 2041)
month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] * len(years)
if monthly_cumulative:
    window_size = np.nan
else:
    window_size = 5

# ── Load ALL facilities directly — no fuzzy matching needed ───────────────────
facilities_with_lat_long = pd.read_csv(
    "/Users/rem76/Desktop/Climate_Change_Health/Data/facilities_with_lat_long_region.csv",
    low_memory=False
)
facilities_with_lat_long = facilities_with_lat_long.drop_duplicates(
    subset="Fname", keep="first"
).reset_index(drop=True)

# Drop rows with no coordinates
facilities_with_lat_long = facilities_with_lat_long.dropna(
    subset=["A109__Latitude", "A109__Longitude"]
).reset_index(drop=True)

print(f"Total facilities with coordinates: {len(facilities_with_lat_long)}")


def unzip_all_in_directory(directory):
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
    output_directory = os.path.join(directory, 'nc_files')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for root, _, files in os.walk(directory):
        if root == output_directory:
            continue
        for filename in files:
            if filename.endswith('.nc'):
                source_file_path = os.path.join(root, filename)
                destination_file_path = os.path.join(output_directory, filename)
                if not os.path.exists(destination_file_path):
                    shutil.copy2(source_file_path, output_directory)


# ── Unzip and extract NetCDF files ────────────────────────────────────────────
base_dir = "/Users/rem76/Desktop/Climate_Change_Health/Data/Precipitation_data/"
scenarios = ["ssp2_4_5"]

for scenario in scenarios:
    scenario_directory = os.path.join(base_dir, scenario)
    unzip_all_in_directory(scenario_directory)
    extract_nc_files_from_unzipped_folders(scenario_directory)

# ── Main extraction loop ───────────────────────────────────────────────────────
for scenario in scenarios:
    print(f"\nProcessing scenario: {scenario}")
    scenario_directory = os.path.join(base_dir, scenario)
    nc_file_directory = os.path.join(scenario_directory, 'nc_files')

    grid_centroids = {}
    data_by_model_and_grid = {}

    for file in glob.glob(os.path.join(nc_file_directory, "*.nc")):
        model = re.search(r'pr_day_(.*?)_' + scenario.replace('_', ''), file).group(1)
        data_per_model = Dataset(file, mode='r')
        pr_data = data_per_model.variables['pr'][:]
        lat_data = data_per_model.variables['lat'][:]
        long_data = data_per_model.variables['lon'][:]

        lon_grid, lat_grid = np.meshgrid(long_data, lat_data)
        centroids = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
        grid_centroids[model] = centroids

        grid_dictionary = {}
        grid = 0
        for i in range(len(long_data)):
            for j in range(len(lat_data)):
                precip_data_for_grid = pr_data[:, j, i] * multiplier
                grid_dictionary[grid] = precip_data_for_grid.data
                grid += 1
        data_by_model_and_grid[model] = grid_dictionary

    data_by_model_and_grid = pd.DataFrame.from_dict(data_by_model_and_grid)
    data_by_model_and_grid.to_csv(Path(scenario_directory) / "data_by_model_and_grid.csv")

    # Keep only models at the modal (most common) resolution
    non_na_lengths = data_by_model_and_grid.count()
    modal_non_na_length = non_na_lengths.mode()[0]
    data_by_model_and_grid_same_length = data_by_model_and_grid.loc[
                                         :, non_na_lengths == modal_non_na_length
                                         ].dropna(axis=0)
    data_by_model_and_grid_same_length.to_csv(
        Path(scenario_directory) / f"data_by_model_and_grid_modal_resolution_all_facilities.csv",
        index=True
    )

    # ── Loop over ALL facilities in the lat/long file ─────────────────────────
    # CHANGED: three dicts instead of one, for lowest/mean/highest
    cumulative_sum_lowest = {}  # p25 across models
    cumulative_sum_mean = {}  # median across models
    cumulative_sum_highest = {}  # p75 across models
    skipped = []

    # Pre-build KDTrees once per model (outside facility loop for efficiency)
    kd_trees_by_model = {
        model: KDTree(grid_centroids[model])
        for model in grid_centroids
        if model in data_by_model_and_grid_same_length.columns
    }

    for _, fac_row in facilities_with_lat_long.iterrows():
        fname = fac_row["Fname"]
        lat_for_fac = fac_row["A109__Latitude"]
        long_for_fac = fac_row["A109__Longitude"]

        if pd.isna(lat_for_fac) or pd.isna(long_for_fac):
            skipped.append(fname)
            continue

        facility_location = np.array([lat_for_fac, long_for_fac])

        # Find the closest grid cell for each model
        grid_precipitation_for_facility = {}
        for model, kd_tree in kd_trees_by_model.items():
            _, closest_grid_index = kd_tree.query(facility_location)
            grid_precipitation_for_facility[model] = (
                data_by_model_and_grid_same_length[model][closest_grid_index].data
            )

        if not grid_precipitation_for_facility:
            skipped.append(fname)
            continue

        first_model = next(iter(grid_precipitation_for_facility))
        n_timepoints = len(grid_precipitation_for_facility[first_model])

        # Stack all models into array of shape (n_models, n_timepoints)
        model_array = np.array([
            precip_data
            for precip_data in grid_precipitation_for_facility.values()
            if len(precip_data) == n_timepoints
        ])
        p25_all = np.percentile(model_array, 25, axis=0).tolist()
        median_all = np.percentile(model_array, 50, axis=0).tolist()
        p75_all = np.percentile(model_array, 75, axis=0).tolist()
        # CHANGED: compute rolling window for each of the three variants
        for daily_values, store in [
            (p25_all, cumulative_sum_lowest),
            (median_all, cumulative_sum_mean),
            (p75_all, cumulative_sum_highest),
        ]:
            store[fname] = []
            begin_day = 0
            for month_idx, month_length in enumerate(month_lengths):
                if monthly_cumulative:
                    window_size = month_length
                days_for_grid = daily_values[begin_day: begin_day + month_length]
                cumulative_sums = [
                    sum(days_for_grid[day: day + window_size])
                    for day in range(month_length - window_size + 1)
                ]
                store[fname].append(max(cumulative_sums))
                begin_day += month_length

    if skipped:
        print(f"  Skipped {len(skipped)} facilities (missing coordinates or data): {skipped[:5]}")
    print(f"  Extracted data for {len(cumulative_sum_mean)} facilities")

    # CHANGED: save three files matching lowest/mean/highest naming convention
    suffix = f"all_facilities"
    for label, data in [("lowest", cumulative_sum_lowest),
                        ("mean", cumulative_sum_mean),
                        ("highest", cumulative_sum_highest)]:
        df = pd.DataFrame.from_dict(data, orient='index').T.astype(float)
        if monthly_cumulative:
            df.to_csv(Path(scenario_directory) / f"{label}_monthly_prediction_weather_by_{suffix}.csv")
        else:
            df.to_csv(Path(scenario_directory) / f"{label}_window_prediction_weather_by_{suffix}.csv")

    print(f"  Saved lowest/mean/highest outputs to {scenario_directory}")
