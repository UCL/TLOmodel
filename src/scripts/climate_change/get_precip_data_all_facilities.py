import glob
import os
from pathlib import Path

import pandas as pd
from netCDF4 import Dataset

# facility data
multiplier = 1000
five_day = True
cumulative = True

facilities_with_lat_long = pd.read_csv(
    "/Users/rem76/Desktop/Climate_Change_Health/Data/facilities_with_lat_long_region.csv")
facilities_with_lat_long = facilities_with_lat_long.dropna(subset=["A109__Latitude", "A109__Longitude"])
print(os.listdir("/Users/rem76/Desktop/Climate_Change_Health/Data/Precipitation_data/Historical/"))
if five_day:
    window_size = 5
    if cumulative:
        window_size_for_average = 1
    else:
        window_size_for_average = 5
else:
    window_size = 1
    window_size_for_average = 1

# historical weather directory
base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total"

years = range(2011, 2025)
# years = range(1940, 1980)

month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
max_average_by_grid = {}

for year in years:
    year_directory = os.path.join(base_dir, str(year))
    precip_datafile = glob.glob(os.path.join(year_directory, "*.nc"))
    if not precip_datafile:
        print(f"No .nc file found for year {year} in {year_directory}")

    data_per_model = Dataset(precip_datafile[0], mode='r')
    data_per_model = Dataset(precip_datafile[0], mode='r')
    print(data_per_model.variables['tp'].shape)  # what is the time dimension?
    print(len(data_per_model.variables['tp'][:]))
    pr_data = data_per_model.variables['tp'][:]
    lat_data = data_per_model.variables['latitude'][:]
    long_data = data_per_model.variables['longitude'][:]
    grid = 0
    for j in range(len(long_data)):
        for i in range(len(lat_data)):
            pr_data_for_square = pr_data[:, i, j]
            if grid not in max_average_by_grid:
                max_average_by_grid[grid] = []
            daily_totals = [
                sum(pr_data_for_square[day * 24:(day + 1) * 24]) for day in range(len(pr_data_for_square) // 24)
            ]
            begin_day = 0
            for month_idx, month_length in enumerate(month_lengths):
                days_for_grid = daily_totals[begin_day:begin_day + month_length]
                moving_averages = []
                for day in range(month_length - window_size + 1):
                    window_average = sum(days_for_grid[day:day + window_size]) / window_size_for_average
                    moving_averages.append(window_average)
                max_moving_average = max(moving_averages)
                max_average_by_grid[grid].append(max_moving_average * multiplier)
                begin_day += month_length
            grid += 1

df = pd.DataFrame.from_dict(max_average_by_grid, orient='index')
df = df.T
if max(years) < 2000:
    df.to_csv(Path(base_dir) / f"historical_{min(years)}_{max(years)}_daily_total_by_grid.csv")
else:
    df.to_csv(Path(base_dir) / "historical_daily_total_by_grid.csv")

########## add in facility data ##################

max_average_by_facility = {}
monthly_total_by_facility = {}

for year in years:
    year_directory = os.path.join(base_dir, str(year))
    precip_datafile = glob.glob(os.path.join(year_directory, "*.nc"))
    data_per_model = Dataset(precip_datafile[0], mode='r')
    pr_data = data_per_model.variables['tp'][:]
    lat_data = data_per_model.variables['latitude'][:]
    long_data = data_per_model.variables['longitude'][:]

    for _, row in facilities_with_lat_long.iterrows():
        facility = row["Fname"]
        lat_for_facility = row["A109__Latitude"]
        long_for_facility = row["A109__Longitude"]

        if facility not in max_average_by_facility:
            max_average_by_facility[facility] = []
        if facility not in monthly_total_by_facility:
            monthly_total_by_facility[facility] = []

        index_for_x = ((long_data - long_for_facility) ** 2).argmin()
        index_for_y = ((lat_data - lat_for_facility) ** 2).argmin()
        pr_data_for_square = pr_data[:, index_for_y, index_for_x]
        daily_totals = [
            sum(pr_data_for_square[day * 24:(day + 1) * 24]) for day in range(len(pr_data_for_square) // 24)
        ]
        begin_day = 0
        for month_idx, month_length in enumerate(month_lengths):
            days_for_grid = daily_totals[begin_day:begin_day + month_length]

            # Five-day rolling max
            moving_averages = []
            for day in range(month_length - window_size + 1):
                window_average = sum(days_for_grid[day:day + window_size]) / window_size_for_average
                moving_averages.append(window_average)
            max_moving_average = max(moving_averages)
            max_average_by_facility[facility].append(max_moving_average * multiplier)

            # Monthly total
            monthly_total = sum(days_for_grid) * multiplier
            monthly_total_by_facility[facility].append(monthly_total)

            begin_day += month_length

# Five-day output
df_of_facilities = pd.DataFrame.from_dict(max_average_by_facility, orient='index').T

# Monthly total output
df_monthly = pd.DataFrame.from_dict(monthly_total_by_facility, orient='index').T

if max(years) > 2000:
    if five_day:
        if cumulative:
            df_of_facilities.to_csv(Path(base_dir) / "historical_daily_total_by_all_facilities_five_day_cumulative.csv")
        else:
            df_of_facilities.to_csv(Path(base_dir) / "historical_daily_total_by_all_facilities_five_day_average.csv")
    else:
        df_of_facilities.to_csv(Path(base_dir) / "historical_daily_total_by_all_facilities.csv")
    df_monthly.to_csv(Path(base_dir) / "historical_monthly_total_by_all_facilities.csv")
else:
    if five_day:
        if cumulative:
            df_of_facilities.to_csv(Path(
                base_dir) / f"historical_{min(years)}_{max(years)}_daily_total_by_all_facilities_five_day_cumulative.csv")
        else:
            df_of_facilities.to_csv(Path(
                base_dir) / f"historical_{min(years)}_{max(years)}_daily_total_by_all_facilities_five_day_average.csv")
    else:
        df_of_facilities.to_csv(
            Path(base_dir) / f"historical_{min(years)}_{max(years)}_daily_total_by_all_facilities.csv")
    df_monthly.to_csv(Path(base_dir) / f"historical_{min(years)}_{max(years)}_monthly_total_by_all_facilities.csv")
