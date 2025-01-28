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
    axes[i].set_xticks(range(0, len(era5_precipitation_data) + len(model_annual_precip), 10))
    axes[i].set_xticklabels(range(2010, 2071, 10))
    axes[i].set_title(scenario.upper())
    axes[i].set_xlabel('Year')
    if i == 0:
        axes[i].set_ylabel('Annual Precipitation (mm)')
        axes[i].legend()

plt.tight_layout()
plt.savefig('/Users/rem76/Desktop/Climate_change_health/Results/ANC_disruptions/histroical_future_precip_annual.png')

## now do model ensembles

model_types = ['lowest', 'mean', 'highest']
# Configuration and constants
min_year_for_analysis = 2025
absolute_min_year = 2024
max_year_for_analysis = 2071
data_path = "/Users/rem76/Desktop/Climate_change_health/Data/"

# Define SSP scenario
ssp_scenarios = ["ssp126", "ssp245", "ssp585"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
axes = axes.flatten()
historical_weather = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv",
    index_col=0)
historical_weather = historical_weather.mean(axis = 1)
historical_weather = historical_weather.to_frame(name='mean_precipitation')
historical_weather.reset_index()
historical_weather_sum = historical_weather.groupby(historical_weather.index // 12).sum()
for i, ssp_scenario in enumerate(ssp_scenarios):
    axes[i].plot(
        range(len(historical_weather_sum)),
        historical_weather_sum,
        color="#312F2F",
        linewidth=2,
        linestyle='--',
        label='ERA5'
    )
    for model in model_types:
        weather_data_prediction_monthly_original = pd.read_csv(
            f"{data_path}Precipitation_data/Downscaled_CMIP6_data_CIL/{ssp_scenario}/{model}_monthly_prediction_weather_by_facility.csv",
            dtype={'column_name': 'float64'}, index_col=0
        )

        y_data = weather_data_prediction_monthly_original.mean(axis = 1)
        y_data = y_data.to_frame(name='mean_precipitation')
        y_data.reset_index(inplace=True)
        y_data = y_data.groupby(
            y_data.index // 12
        ).sum()
        axes[i].plot(
            range(len(historical_weather_sum), len(historical_weather_sum) + len(y_data)),
            y_data['mean_precipitation'],
            label=f"{model}",
        )

        # Fix xticks and labels
        axes[i].set_xticks(range(0,len(historical_weather_sum) + len(y_data), 10))
        axes[i].set_xticklabels(range(2010, 2071, 10))
        axes[i].set_title(ssp_scenario.upper())
        axes[i].set_xlabel('Year')
        if i == 0:
            axes[i].set_ylabel('Annual Precipitation (mm)')
            axes[i].legend()

plt.tight_layout()
plt.savefig('/Users/rem76/Desktop/Climate_change_health/Results/ANC_disruptions/histroical_future_precip_annual_selected_models.png')
