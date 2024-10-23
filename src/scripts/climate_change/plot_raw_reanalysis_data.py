import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# Load the dataset and the variable
file_path = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/139ef85ab4df0a12fc01854395fc9a6d.nc"
dataset = Dataset(file_path, mode='r')
pr_data = dataset.variables['tp'][:] # in kg m-2 s-1 = mm s-1 x 86400 to get to day
time_data = dataset.variables['date'][:]
lat_data = dataset.variables['latitude'][:]
long_data = dataset.variables['longitude'][:]

## Initial plot
for i in range(len(lat_data)):
    for j in range(len(long_data)):
        pr_data_time_series_grid_1 = pr_data[:, i, j]
        pr_data_time_series_grid_1 *= 86400 # to get to days
        plt.plot(pr_data_time_series_grid_1)

plt.title('Average Precipitation Over Time - Grid ')
plt.ylabel('Precip (mm)')
plt.xlabel('Time')
# plt.show()


weather_data_historical = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facility_lm.csv", index_col=0)

for i in range(len(weather_data_historical.columns)):
    plt.plot(weather_data_historical.iloc[:, i], label = weather_data_historical.columns[i])

plt.title('Average Precipitation Over Time - Facility ')
plt.ylabel('Precip (mm)')
plt.xlabel('Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()



monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_by_smaller_facility_lm.csv", index_col=0)

for i in range(len(monthly_reporting_by_facility.columns)):
    plt.plot(monthly_reporting_by_facility.iloc[:, i], label = monthly_reporting_by_facility.columns[i])

plt.title('Average Reprting Over Time - Facility ')
plt.ylabel('Reporting (%)')
plt.xlabel('Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()


for i in range(len(monthly_reporting_by_facility.columns)):
    for j in range(len(monthly_reporting_by_facility.iloc[:, i])):
        if weather_data_historical.iloc[j, i] > 1000:
            plt.scatter(weather_data_historical.iloc[j, i], monthly_reporting_by_facility.iloc[j, i])

plt.title('Average Reprting Over Time - Facility ')
plt.ylabel('Reporting(%)')
plt.xlabel('Precip (mm)')
plt.xlim(1000,4000)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()





