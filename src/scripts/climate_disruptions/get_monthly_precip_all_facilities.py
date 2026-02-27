"""
Extract downscaled CMIP6 precipitation metrics for all Malawi facilities.

For each scenario and representative model (lowest/median/highest precipitation),
computes per-facility monthly totals and 5-day rolling maximums, then saves to CSV.

Inputs:
  - Local .nc files in base_dir/{scenario}/*.nc (bias-corrected, mm/s units)
  - facilities_with_lat_long_region.csv
  - monthly_reporting_ANC_by_smaller_facility_lm.csv (if ANC=True)

Outputs:
  - base_dir/{scenario}/{lowest|median|highest}_monthly_prediction_weather_by_facility.csv
  - base_dir/{scenario}/{lowest|median|highest}_window_prediction_weather_by_facility.csv
"""

import os
import re
import glob
import shutil
import zipfile
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree

# ── Config ─────────────────────────────────────────────────────────────────────
ANC = True
Inpatient = False
multiplier = 86400  # kg m-2 s-1 → mm/day
window_size = 5
years = range(2025, 2041)

base_dir = "/Users/rem76/Desktop/Climate_Change_Health/Data/Precipitation_Data/Downscaled_CMIP6_data/"
scenarios = ["ssp2_4_5"]

FACILITIES_PATH = "/Users/rem76/Desktop/Climate_Change_Health/Data/facilities_with_lat_long_region.csv"

# Month lengths (non-leap years; extend or adjust if needed)
month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] * len(years)
year_lengths = [365, 366, 365, 366] * max(1, len(years) // 4)
year_lengths = year_lengths[:len(years)]

# ── Load static data ───────────────────────────────────────────────────────────
facilities_with_lat_long = (
    pd.read_csv(FACILITIES_PATH, low_memory=False)
    .drop_duplicates(subset="Fname", keep="first")
    .dropna(subset=["A109__Latitude", "A109__Longitude"])
    .reset_index(drop=True)
)
print(f"Total facilities with coordinates: {len(facilities_with_lat_long)}")


# ── Helper functions ───────────────────────────────────────────────────────────
def get_facility_lat_long(reporting_facility, facilities_df, cutoff=0.90, n_matches=3):
    """Return (match_name, lat, lon) for the closest-matching facility name, or (nan, nan, nan)."""
    matches = difflib.get_close_matches(
        reporting_facility, facilities_df["Fname"], n=n_matches, cutoff=cutoff
    )
    if matches:
        match_name = matches[0]
        lat = facilities_df.loc[facilities_df["Fname"] == match_name, "A109__Latitude"].iloc[0]
        lon = facilities_df.loc[facilities_df["Fname"] == match_name, "A109__Longitude"].iloc[0]
        return match_name, lat, lon
    return np.nan, np.nan, np.nan


def calculate_monthly_metrics(precip_data, month_lengths, window_size):
    """
    Given a 1-D array of daily precipitation values and a list of month lengths,
    return (monthly_totals, monthly_max_window) each of length len(month_lengths).
    """
    monthly_totals = []
    monthly_max_window = []
    begin_day = 0
    for month_length in month_lengths:
        days = precip_data[begin_day: begin_day + month_length]
        monthly_totals.append(float(np.sum(days)))
        if len(days) >= window_size:
            rolling = np.convolve(days, np.ones(window_size, dtype=float), mode="valid")
            monthly_max_window.append(float(np.max(rolling)))
        else:
            monthly_max_window.append(float(np.sum(days)))
        begin_day += month_length
    return monthly_totals, monthly_max_window


def unzip_all_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            file_path = os.path.join(directory, filename)
            extract_dir = os.path.join(directory, filename[:-4])
            os.makedirs(extract_dir, exist_ok=True)
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile:
                print(f"Skipped {filename}: not a valid zip file.")


def extract_nc_files_from_unzipped_folders(directory):
    output_directory = os.path.join(directory, "nc_files")
    os.makedirs(output_directory, exist_ok=True)
    for root, _, files in os.walk(directory):
        if root == output_directory:
            continue
        for filename in files:
            if filename.endswith(".nc"):
                src = os.path.join(root, filename)
                dst = os.path.join(output_directory, filename)
                if not os.path.exists(dst):
                    shutil.copy2(src, output_directory)


# ── Main processing ────────────────────────────────────────────────────────────
for scenario in scenarios:
    print(f"\nProcessing scenario: {scenario}")
    scenario_directory = Path(base_dir) / scenario

    # ── Load all models and select representatives ────────────────────────────
    data_by_model_and_grid = {}
    grid_centroids = {}
    cumulative_sum_by_model = {}

    for file in sorted(glob.glob(str(scenario_directory / "*.nc"))):
        model = re.search(r".*/(.*?)_ssp\d+", file).group(1)
        print(f"  Loading model: {model}")

        with xr.open_dataset(file) as data_per_model:
            lat_data = data_per_model["lat"].values
            lon_data = data_per_model["lon"].values
            lon_grid, lat_grid = np.meshgrid(lon_data, lat_data)
            centroids = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
            grid_centroids[model] = centroids

            # Build grid → daily precip array mapping
            grid_dictionary = {}
            for grid, (i, j) in enumerate(np.ndindex(len(lat_data), len(lon_data))):
                precip = data_per_model.sel(lat=lat_data[i], lon=lon_data[j], method="nearest")
                grid_dictionary[grid] = precip["pr"].values * multiplier  # → mm/day
            data_by_model_and_grid[model] = grid_dictionary

            # Compute mean annual total across the grid for model selection
            pr_avg = data_per_model["pr"].mean(dim=["lat", "lon"]).values * multiplier
            begin_day = 0
            annual_totals = []
            for year_length in year_lengths:
                annual_totals.append(np.sum(pr_avg[begin_day: begin_day + year_length]))
                begin_day += year_length
            cumulative_sum_by_model[model] = float(np.mean(annual_totals))

    sorted_models = sorted(cumulative_sum_by_model, key=cumulative_sum_by_model.get)
    lowest_model = sorted_models[0]
    highest_model = sorted_models[-1]
    median_model = sorted_models[len(sorted_models) // 2]
    models_of_interest = [lowest_model, median_model, highest_model]
    model_categories = {lowest_model: "lowest", median_model: "median", highest_model: "highest"}
    print(f"  Models of interest: {models_of_interest}")

    # ── Compute per-facility metrics for each representative model ────────────
    results = {
        model: {"monthly": {}, "window": {}}
        for model in models_of_interest
    }

    for _, fac_row in facilities_with_lat_long.iterrows():
        fname = fac_row["Fname"]
        lat = fac_row["A109__Latitude"]
        lon = fac_row["A109__Longitude"]
        facility_location = np.array([lat, lon])

        for model in models_of_interest:
            kd_tree = KDTree(grid_centroids[model])
            _, closest_grid_index = kd_tree.query(facility_location)
            precip_data = data_by_model_and_grid[model][int(closest_grid_index)]

            monthly_totals, monthly_max_window = calculate_monthly_metrics(
                precip_data, month_lengths, window_size
            )
            results[model]["monthly"][fname] = monthly_totals
            results[model]["window"][fname] = monthly_max_window

    # ── Save outputs ──────────────────────────────────────────────────────────
    for model in models_of_interest:
        category = model_categories[model]
        for metric_type in ("monthly", "window"):
            df = pd.DataFrame.from_dict(results[model][metric_type], orient="columns")
            output_file = scenario_directory / f"{category}_{metric_type}_prediction_weather_by_facility.csv"
            df.to_csv(output_file, index=True)
            print(f"  Saved {metric_type} data for {category} ({model}) → {output_file}")

print("\nDone.")
