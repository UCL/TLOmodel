import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

# Load ERA5 data
era5_data_xr = xr.open_dataset('/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/monthly_data/724bab97773bb7ba4e1635356ad0d12.nc')
era5_data_xr_until_2024 = era5_data_xr.sel(date=slice('20110101', '20250101'))
era5_data_xr_until_2024['date'] = pd.to_datetime(era5_data_xr_until_2024['date'], format='%Y%m%d')
era5_data_xr_until_2024 = era5_data_xr_until_2024.mean(dim=["latitude", "longitude"])
dates = pd.to_datetime(era5_data_xr_until_2024['date'].values)
days_in_month = dates.to_series().dt.days_in_month
era5_data_xr_until_2024 = era5_data_xr_until_2024 * days_in_month.values
era5_precipitation_data = era5_data_xr_until_2024['tp'].resample(date='Y').sum('date')
# print(era5_precipitation_data)
# # Plot ERA5 annual precipitation data
# plt.plot(era5_precipitation_data * 1000, color="blue", linewidth=2, linestyle='--', marker='s', markersize=4, label='ERA5')
# plt.show()
# #
# Load CMIP6 downscaled data
base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data"
scenarios = ["ssp126", "ssp245", "ssp585"]

for scenario in scenarios:
    scenario_directory = os.path.join(base_dir, scenario)
    file_path_downscaled = f"/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/CIL_combined_{scenario}_2025_2070.nc"
    data_all_models = xr.open_dataset(file_path_downscaled)

    plt.figure(figsize=(12, 8))

    for model in data_all_models['model'].values:
        model_data = data_all_models.sel(model=model)
        pr_aggregated = model_data.mean(dim=['lat', 'lon'], skipna=True)

        model_annual_precip = pr_aggregated['pr'].resample(time='YE').sum('time')
        plt.plot(range(len(model_annual_precip)), model_annual_precip, alpha=0.5, label=f'{model}')
    plt.legend()
    plt.show()
#
# pr_aggregated = data_all_models.mean(dim=['lat', 'lon','model'], skipna=True)
# pr_aggregated_annual = pr_aggregated['pr'].resample(time='YE').sum('time')
# std_pr = data_all_models.std(dim=['lat', 'lon',"model"], skipna=True)
# std_pr_annual = std_pr['pr'].resample(time='YE').sum('time')
#
#
# # Print results
# upper_bound = pr_aggregated_annual + std_pr_annual
# lower_bound = pr_aggregated_annual - std_pr_annual
#
# # Plot annual precipitation with confidence interval
# plt.plot(pr_aggregated_annual, color="black", linewidth=2, linestyle='--', marker='s', markersize=4, label='Annual Precipitation')
# #plt.fill_between(range(0,len(pr_aggregated_annual)), lower_bound.values, upper_bound.values, color="blue", alpha=0.2, label="95% Confidence Interval")
#
# # Labels and title
# plt.xlabel("Time")
# plt.ylabel("Annual Precipitation (mm)")
# plt.title(f"Annual Precipitation for {scenario} (with 95% Confidence Interval)")
# plt.legend()
# plt.show()
#
# ## Plot together
# plt.figure(figsize=(12, 7))
#
# plt.plot(
#     range(0, len(era5_precipitation_data)),
#     era5_precipitation_data * 1000,
#     color="#1C6E8C",
#     linewidth=2,
#     linestyle='--',
#     marker='o',
#     markersize=6,
#     label='ERA5 Precipitation',
# )
#
# plt.plot(
#     range(len(era5_precipitation_data) , len(era5_precipitation_data) + len(pr_aggregated_annual) - 1),
#     pr_aggregated_annual[:-1],
#     color="#9AC4F8",
#     linewidth=2,
#     linestyle='-',
#     marker='s',
#     markersize=6,
#     label=f'CMIP6 Scenario {scenario}',
# )
#
# plt.fill_between(
#     range(len(era5_precipitation_data), len(era5_precipitation_data) + len(pr_aggregated_annual)),
#     lower_bound.values,
#     upper_bound.values,
#     color="#9AC4F8",
#     alpha=0.3,
# )
# years = np.arange(2011, 2071, 5)  # Array of years from 2011 to 2070
# plt.xticks(np.linspace(0, len(era5_precipitation_data) + len(pr_aggregated_annual) - 1, len(years)), years)
# plt.xlabel("Time (Years)", fontsize=14, labelpad=10)
# plt.ylabel("Annual Precipitation (mm)", fontsize=14, labelpad=10)
# plt.legend(loc="upper left", fontsize=12, frameon=True, shadow=True, fancybox=True)
# plt.tight_layout()
# plt.show()
#
#
# #### Now plot all
#
# base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/"
#
# scenarios = ["ssp129", "ssp245", "ssp585"]
# # Put all into one csv file
# file_list = glob.glob(os.path.join(base_dir, "*.nc"))
# data_by_model_and_grid = {}
#
# for scenario in scenarios:
#     print(scenario)
#     scenario_directory = os.path.join(base_dir, scenario)
#     nc_file_directory = os.path.join(scenario_directory, 'nc_files')
#     grid_centroids = {}
#     for file in glob.glob(os.path.join(nc_file_directory, "*.nc")):
#         model = re.search(r'pr_day_(.*?)_' + scenario.replace('_', ''), file).group(1)
#         data_per_model  = Dataset(file, mode='r')
#         pr_aggregated = data_all_models.mean(dim=['lat', 'lon', 'model'], skipna=True)
#         pr_aggregated_annual = pr_aggregated['pr'].resample(time='YE').sum('time')
#         std_pr = data_all_models.std(dim=['lat', 'lon', "model"], skipna=True)
#         std_pr_annual = std_pr['pr'].resample(time='Y').sum('time')
#
#         # Print results
#         upper_bound = pr_aggregated_annual + std_pr_annual
#         lower_bound = pr_aggregated_annual - std_pr_annual
#
#         # Plot annual precipitation with confidence interval
#         plt.figure(figsize=(10, 6))
#         plt.plot(pr_aggregated_annual, color="blue", linewidth=2, linestyle='--', marker='s', markersize=4,
#                  label='Annual Precipitation')
#         plt.fill_between(range(0, len(pr_aggregated_annual)), lower_bound.values, upper_bound.values, color="blue",
#                          alpha=0.2, label="95% Confidence Interval")
#
#         # Labels and title
#         plt.xlabel("Time")
#         plt.ylabel("Annual Precipitation (mm)")
#         plt.title(f"Annual Precipitation for {scenario} (with 95% Confidence Interval)")
#         plt.legend()
#         plt.show()
#
#         ## Plot together
#         plt.figure(figsize=(12, 7))
#
#         plt.plot(
#             range(0, len(era5_precipitation_data)),
#             era5_precipitation_data * 1000,
#             color="#1C6E8C",
#             linewidth=2,
#             linestyle='--',
#             marker='o',
#             markersize=6,
#             label='ERA5 Precipitation',
#         )
#
#         plt.plot(
#             range(len(era5_precipitation_data), len(era5_precipitation_data) + len(pr_aggregated_annual) - 1),
#             pr_aggregated_annual[:-1],
#             color="#9AC4F8",
#             linewidth=2,
#             linestyle='-',
#             marker='s',
#             markersize=6,
#             label=f'CMIP6 Scenario {scenario}',
#         )
#
#         plt.fill_between(
#             range(len(era5_precipitation_data), len(era5_precipitation_data) + len(pr_aggregated_annual)),
#             lower_bound.values,
#             upper_bound.values,
#             color="#9AC4F8",
#             alpha=0.3,
#         )
#         years = np.arange(2011, 2071, 5)  # Array of years from 2011 to 2070
#         plt.xticks(np.linspace(0, len(era5_precipitation_data) + len(pr_aggregated_annual) - 1, len(years)), years)
#         plt.xlabel("Time (Years)", fontsize=14, labelpad=10)
#         plt.ylabel("Annual Precipitation (mm)", fontsize=14, labelpad=10)
#         plt.legend(loc="upper left", fontsize=12, frameon=True, shadow=True, fancybox=True)
#         plt.tight_layout()
#         plt.show()
#
