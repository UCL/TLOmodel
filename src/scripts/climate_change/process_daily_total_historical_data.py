import difflib
import glob
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from netCDF4 import Dataset

ANC = True
# facility data
multiplier = 1000
five_day = True
cumulative = True
general_facilities = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.shp")

facilities_with_lat_long = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_lat_long_region.csv")

if five_day:
    window_size = 5
    if cumulative:
        window_size_for_average = 1
    else:
        window_size_for_average = 5
else:
    window_size = 1
    window_size_for_average = 1

# Data accessed from https://dhis2.health.gov.mw/dhis-web-data-visualizer/#/YiQK65skxjz
# Reporting rate is expected reporting vs actual reporting
if ANC:
    reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_change_health/Data/ANC_data/ANC_data_2011_2024.csv') #January 2011 - January 2024
else:
    reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_change_health/Data/Inpatient_Data/HMIS_Total_Number_Admissions.csv') #January 2011 - January 2024
# ANALYSIS DONE IN OCTOBER 2024 - so drop October, November, December 2024
columns_to_drop = reporting_data.columns[reporting_data.columns.str.endswith(('October 2024', 'November 2024', 'December 2024'))]
reporting_data = reporting_data.drop(columns=columns_to_drop)
# drop NAs
reporting_data = reporting_data.dropna(subset = reporting_data.columns[3:], how='all') # drops 90 clinics
### now aggregate over months
monthly_reporting_data_by_facility =  {}
months = set(col.split(" - Reporting rate ")[1] for col in reporting_data.columns if " - Reporting rate " in col)

# put in order
months = [date.strip() for date in months] # extra spaces??
dates = pd.to_datetime(months, format='%B %Y', errors='coerce')
months = dates.sort_values().strftime('%B %Y').tolist() # puts them in ascending order
for month in months:
    columns_of_interest_all_metrics = [reporting_data.columns[1]] + reporting_data.columns[reporting_data.columns.str.endswith(month)].tolist()
    data_of_interest_by_month = reporting_data[columns_of_interest_all_metrics]
    numeric_data = data_of_interest_by_month.select_dtypes(include='number')
    monthly_mean_by_facility = numeric_data.mean(axis=1)
    monthly_reporting_data_by_facility[month] = monthly_mean_by_facility
monthly_reporting_by_facility = pd.DataFrame(monthly_reporting_data_by_facility)
monthly_reporting_by_facility["facility"] = reporting_data["organisationunitname"].values

# historical weather directory
base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total"

#years = range(2011, 2025)
years = range(1940, 1980)

month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
max_average_by_grid = {}

for year in years:
    year_directory = os.path.join(base_dir, str(year))
    precip_datafile = glob.glob(os.path.join(year_directory, "*.nc"))
    data_per_model = Dataset(precip_datafile[0], mode='r')
    pr_data = data_per_model.variables['tp'][:]  # precipitation data in m for daily reanalysis
    lat_data = data_per_model.variables['latitude'][:]
    long_data = data_per_model.variables['longitude'][:]
    grid = 0
    for j in range(len(long_data)):
        for i in range(len(lat_data)):
            pr_data_for_square = pr_data[:, i, j]
            if grid not in max_average_by_grid:
                max_average_by_grid[grid] = []
            # Aggregate hourly data to daily totals
            daily_totals = [
                sum(pr_data_for_square[day*24:(day+1)*24]) for day in range(len(pr_data_for_square) // 24)
            ]
            begin_day = 0
            for month_idx, month_length in enumerate(month_lengths):
                days_for_grid = daily_totals[begin_day:begin_day + month_length]
                moving_averages = []
                for day in range(month_length - window_size + 1):
                    window_average = sum(days_for_grid[day:day + window_size]) / window_size_for_average
                    moving_averages.append(window_average)

                max_moving_average = max(moving_averages)
                max_average_by_grid[grid].append(max_moving_average* multiplier)

                begin_day += month_length
            grid += 1

df = pd.DataFrame.from_dict(max_average_by_grid, orient='index')
df = df.T
if max(years) < 2000:
    df.to_csv(Path(base_dir)/f"historical_{min(years)}_{max(years)}_daily_total_by_grid.csv")
else:
    df.to_csv(Path(base_dir)/"historical_daily_total_by_grid.csv")


########## add in reporting data ##################

max_average_by_facility = {}
for year in years:
    year_directory = os.path.join(base_dir, str(year))
    precip_datafile = glob.glob(os.path.join(year_directory, "*.nc"))
    data_per_model = Dataset(precip_datafile[0], mode='r')
    pr_data = data_per_model.variables['tp'][:]  # precipitation data in kg m-2 s-1
    lat_data = data_per_model.variables['latitude'][:]
    long_data = data_per_model.variables['longitude'][:]
    # loop over clinics
    for reporting_facility in monthly_reporting_by_facility["facility"]:
        matching_facility_name = difflib.get_close_matches(reporting_facility, facilities_with_lat_long['Fname'], n=3,
                                                           cutoff=0.90)

        if (reporting_facility == "Central Hospital"):
            matching_facility_name = "Lilongwe City Assembly Chinsapo"

        if matching_facility_name:
            match_name = matching_facility_name[0]  # Access the string directly
            # Initialize facility key if not already

            lat_for_facility = facilities_with_lat_long.loc[
                facilities_with_lat_long['Fname'] == match_name, "A109__Latitude"].iloc[0]
            long_for_facility = facilities_with_lat_long.loc[
                facilities_with_lat_long['Fname'] == match_name, "A109__Longitude"].iloc[0]
            if pd.isna(lat_for_facility):
                continue
            if reporting_facility not in max_average_by_facility:
                max_average_by_facility[reporting_facility] = []
            index_for_x = ((long_data - long_for_facility) ** 2).argmin()
            index_for_y = ((lat_data - lat_for_facility) ** 2).argmin()
            pr_data_for_square = pr_data[:, index_for_y, index_for_x]
            daily_totals = [
                sum(pr_data_for_square[day*24:(day+1)*24]) for day in range(len(pr_data_for_square) // 24)
            ]
            begin_day = 0
            for month_idx, month_length in enumerate(month_lengths):
                days_for_grid = daily_totals[begin_day:begin_day + month_length]
                moving_averages = []
                for day in range(month_length - window_size + 1):
                    window_average = sum(days_for_grid[day:day + window_size]) / window_size_for_average
                    moving_averages.append(window_average)
                max_moving_average = max(moving_averages)
                max_average_by_facility[reporting_facility].append(max_moving_average * multiplier)

                begin_day += month_length



#
df_of_facilities = pd.DataFrame.from_dict(max_average_by_facility, orient='index')
df_of_facilities = df_of_facilities.iloc[:, :-3] ## THESE ARE OCT/NOV/DEC OF 2024, and for moment don't have that reporting data
df_of_facilities = df_of_facilities.T

if max(years) > 2000:
    if five_day:
        if cumulative:
            if ANC:
                df_of_facilities.to_csv(Path(base_dir) / "historical_daily_total_by_facilities_with_ANC_five_day_cumulative.csv")
            else:
                df_of_facilities.to_csv(Path(base_dir) / "historical_daily_total_by_facility_five_day_cumulative_inpatient.csv")
        else:
            if ANC:
                df_of_facilities.to_csv(Path(base_dir) / "historical_daily_total_by_facilities_with_ANC_five_day_average.csv")
            else:
                df_of_facilities.to_csv(Path(base_dir) / "historical_daily_total_by_facility_five_day_average_inpatient.csv")
    else:
            if ANC:
                df_of_facilities.to_csv(Path(base_dir) / "historical_daily_total_by_facilities_with_ANC.csv")
            else:
                df_of_facilities.to_csv(Path(base_dir) / "historical_daily_total_by_facility_inpatient.csv")
else:
    if five_day:
        if cumulative:
            if ANC:
                df_of_facilities.to_csv(Path(base_dir) / f"historical_{min(years)}_{max(years)}_daily_total_by_facilities_with_ANC_five_day_cumulative.csv")
            else:
                df_of_facilities.to_csv(Path(base_dir) / f"historical_{min(years)}_{max(years)}_daily_total_by_facility_five_day_cumulative_inpatient.csv")
        else:
            if ANC:
                df_of_facilities.to_csv(Path(base_dir) / f"historical_{min(years)}_{max(years)}_daily_total_by_facilities_with_ANC_five_day_average.csv")
            else:
                df_of_facilities.to_csv(Path(base_dir) / f"historical_{min(years)}_{max(years)}_daily_total_by_facility_five_day_average_inpatient.csv")
    else:
            if ANC:
                df_of_facilities.to_csv(Path(base_dir) / f"historical_{min(years)}_{max(years)}_daily_total_by_facilities_with_ANC.csv")
            else:
                df_of_facilities.to_csv(Path(base_dir) / f"historical_{min(years)}_{max(years)}_daily_total_by_facility_inpatient.csv")
