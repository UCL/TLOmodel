import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# data is from 2011 - 2024
monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_by_facility_lm.csv", index_col=0)
weather_data_historical = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_facility_lm.csv", index_col=0)
#
# Plot each facility's reporting data against weather data
plt.figure(figsize=(12, 6))

for facility in weather_data_historical.columns:
    plt.plot(weather_data_historical.index, monthly_reporting_by_facility[facility], label=facility)
months = weather_data_historical.index
year_labels = range(2011, 2025, 1)
year_ticks = range(0, len(months), 12)
plt.xticks(year_ticks, year_labels, rotation=90)
plt.xlabel('Weather Data')
plt.ylabel('Reporting')
plt.title('Reporting vs. Weather Data by Facility')
plt.legend(title='Facilities', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
#plt.show()

## Drop Mental Hospital - bad reporting generally
weather_data_historical = weather_data_historical.drop("Zomba Mental Hospital", axis=1)
monthly_reporting_by_facility = monthly_reporting_by_facility.drop("Zomba Mental Hospital", axis=1)
## Drop September 2024
weather_data_historical = weather_data_historical.drop(weather_data_historical.index[-1])
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[-1])
## Drop before 2014?  12*4
weather_data_historical = weather_data_historical.drop(weather_data_historical.index[0:48]).reset_index(drop=True)
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[0:48]).reset_index(drop=True)

# Plot each facility's reporting data against weather data
plt.figure(figsize=(12, 6))

for facility in weather_data_historical.columns:
    plt.plot(weather_data_historical.index, monthly_reporting_by_facility[facility], label=facility)
months = weather_data_historical.index
year_labels = range(2015, 2025, 1)
year_ticks = range(0, len(months), 12)
plt.xticks(year_ticks, year_labels, rotation=90)
plt.xlabel('Weather Data')
plt.ylabel('Reporting')
plt.title('Reporting vs. Weather Data by Facility')
plt.legend(title='Facilities', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
#plt.show()



## Linear regression - flattened
# year
year = range(2014, 2024, 1) # year as a fixed effect
year_repeated = [y for y in year for _ in range(12)]
year = year_repeated[:-4]
year_flattened = year*len(weather_data_historical.columns) # to get flattened data

# month
month = range(12)
month_repeated = [m for m in month for _ in range(2014, 2024, 1)]
month = month_repeated[:-4]
month_flattened = month*len(weather_data_historical.columns)

# facility as fixed effect
facility_flattened = list(range(len(weather_data_historical.columns))) * len(month)

# linear regression - flatten for more data points
weather_data = weather_data_historical.values.flatten()
y = monthly_reporting_by_facility.values.flatten()
X = pd.DataFrame({
    'weather_data': weather_data,
    'year': year_flattened,
    'month': month_flattened,
    'facility': facility_flattened
})

# One-hot encode the 'facility' column for a fixed effect
facility_encoded = pd.get_dummies(X['facility'])

X = np.column_stack((X[['weather_data', 'year', 'month']], facility_encoded))

# # Perform linear regression
mod = sm.OLS(y,X)
results = mod.fit()

print(results.summary())

# # ## Linear regression - by facility
results_list = []
#
for facility in monthly_reporting_by_facility.columns:
    y = monthly_reporting_by_facility[facility].values
    weather = weather_data_historical[facility].values
    X = np.column_stack((weather, year, month))

    mod = sm.OLS(y, X)
    results = mod.fit()

    print(results.summary())




