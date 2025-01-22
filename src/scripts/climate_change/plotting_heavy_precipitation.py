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
#
# for scenario in scenarios:
#     scenario_directory = os.path.join(base_dir, scenario)
#     file_path_downscaled = f"/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/CIL_combined_{scenario}_2024_2070.nc"
#     data_all_models = xr.open_dataset(file_path_downscaled)
#
#     plt.figure(figsize=(12, 8))
#
#     for model in data_all_models['model'].values:
#         model_data = data_all_models.sel(model=model)
#         pr_aggregated = model_data.mean(dim=['lat', 'lon'], skipna=True)
#         model_annual_precip = pr_aggregated['pr'].resample(time='YE').sum('time')
#         plt.plot(range(len(model_annual_precip)), model_annual_precip, alpha=0.5, label=f'{model}')
#     plt.legend()
#     plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
axes = axes.flatten()

for i, scenario in enumerate(scenarios):
    scenario_directory = os.path.join(base_dir, scenario)
    file_path_downscaled = f"/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/CIL_combined_{scenario}_2024_2070.nc"
    data_all_models = xr.open_dataset(file_path_downscaled)
    pr_aggregated_mean = data_all_models.mean(dim=['lat', 'lon', 'model'], skipna=True)
    pr_aggregated_median = data_all_models.median(dim='model', skipna=True)
    pr_aggregated_median = pr_aggregated_median.mean(dim=['lat', 'lon'], skipna=True)

    model_annual_precip = pr_aggregated_mean['pr'].resample(time='YE').sum('time')
    #model_annual_precip_median = pr_aggregated_median['pr'].resample(time='YE').sum('time')

    for model in data_all_models['model'].values:
        model_data = data_all_models.sel(model=model)
        pr_aggregated = model_data.mean(dim=['lat', 'lon'], skipna=True)
        model_annual_precip = pr_aggregated['pr'].resample(time='YE').sum('time')
        axes[i].plot(
            range(len(era5_precipitation_data), len(era5_precipitation_data) + len(model_annual_precip)),
            model_annual_precip,
            alpha=0.5,
            #label=f'{model}'
        )
    axes[i].plot(range(len(era5_precipitation_data)), era5_precipitation_data * 1000, color="#312F2F", linewidth=2, linestyle='--', label='ERA5')

    axes[i].plot(
        range(len(era5_precipitation_data), len(era5_precipitation_data) + len(model_annual_precip)),
        model_annual_precip,
        label="Mean of CMIP6",
        color = 'black'
    )
    # axes[i].plot(
    #     range(len(era5_precipitation_data), len(era5_precipitation_data) + len(model_annual_precip)),
    #     model_annual_precip_median,
    #     label="Median",
    #     color='black'
    # )
    axes[i].set_xticklabels(range(2025,2071))
    axes[i].set_title(scenario.upper())
    axes[i].set_xlabel('Year')
    if i == 0:
        axes[i].set_ylabel('Annual Precipitation (mm)')
        axes[i].legend()

plt.tight_layout()
plt.show()
plt.savefig('/Users/rem76/Desktop/Climate_change_health/Results/ANC_disruptions/histroical_future_precip_annual.png')

## Plot 95% CI
#
# fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
# axes = axes.flatten()
#
# for i, scenario in enumerate(scenarios):
#     scenario_directory = os.path.join(base_dir, scenario)
#     file_path_downscaled = f"/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/CIL_combined_{scenario}_2024_2070.nc"
#     data_all_models = xr.open_dataset(file_path_downscaled)
#     model_annual_precip = data_all_models.mean(dim=['lat', 'lon', 'model'], skipna=True)
#     #model_annual_precip = pr_aggregated_mean['pr'].resample(time='YE').sum('time')
#     std_pr_annual = data_all_models.std(dim=['lat', 'lon', "model"], skipna=True)
#     #std_pr_annual = std_pr['pr'].resample(time='YE').sum('time')
#     print(model_annual_precip)
#     print(len(model_annual_precip))
#     upper_bound = model_annual_precip['pr'] + std_pr_annual['pr']
#     lower_bound = model_annual_precip['pr'] - std_pr_annual['pr']
#     axes[i].plot(
#         range(0, len(era5_precipitation_data)),
#         era5_precipitation_data * 1000,
#         color="#1C6E8C",
#         linewidth=2,
#         linestyle='--',
#         marker='o',
#         markersize=6,
#         label='ERA5 Precipitation',
#     )
#
#     axes[i].plot(
#         range(len(era5_precipitation_data),len(era5_precipitation_data) +len(model_annual_precip)),
#         model_annual_precip.to_dataarray(),
#         label="Ensemble mean",
#         color = 'black'
#     )
#     axes[i].fill_between(
#     range(len(era5_precipitation_data), len(era5_precipitation_data) + len(model_annual_precip)),
#     lower_bound,
#     upper_bound,
#     color="#9AC4F8",
#     alpha=0.3,
# )
#     axes[i].set_title(scenario.upper())
#     axes[i].set_xlabel('Year')
#     #axes[i].legend(fontsize='small')
#
# fig.text(0.04, 0.5, 'Annual Precipitation (mm)', va='center', rotation='vertical')
#
# plt.tight_layout()
# plt.show()
