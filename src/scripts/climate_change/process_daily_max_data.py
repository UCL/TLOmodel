import glob
import os
from netCDF4 import Dataset
from pathlib import Path
import pandas as pd

base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_maximum"
years = range(2011, 2025)

# month lengths, account for leap years for February in 2012, 2016, and 2020
month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
leap_year_month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
window_size = 5
max_average_by_grid = {}

for year in years:
    year_directory = os.path.join(base_dir, str(year))
    precip_datafile = glob.glob(os.path.join(year_directory, "*.nc"))
    data_per_model = Dataset(precip_datafile[0], mode='r')
    pr_data = data_per_model.variables['tp'][:]  # precipitation data in kg m-2 s-1
    lat_data = data_per_model.variables['latitude'][:]
    long_data = data_per_model.variables['longitude'][:]

    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        current_month_lengths = leap_year_month_lengths
    else:
        current_month_lengths = month_lengths

    grid = 0
    for j in range(len(long_data)):
        for i in range(len(lat_data)):
            pr_data_for_square = pr_data[:, i, j]
            if grid not in max_average_by_grid:
                max_average_by_grid[grid] = []

            begin_day = 0
            for month_idx, month_length in enumerate(current_month_lengths):
                days_for_grid = pr_data_for_square[begin_day:begin_day + month_length]
                moving_averages = []
                for day in range(month_length - window_size + 1):
                    window_average = sum(days_for_grid[day:day + window_size]) / window_size
                    moving_averages.append(window_average)

                max_moving_average = max(moving_averages)
                max_average_by_grid[grid].append(max_moving_average* 86400)

                begin_day += month_length
            print(len(max_average_by_grid))
            grid += 1

df = pd.DataFrame.from_dict(max_average_by_grid, orient='index')
df = df.T
df.to_csv(Path(base_dir)/"historical_daily_max_by_grid.csv")
