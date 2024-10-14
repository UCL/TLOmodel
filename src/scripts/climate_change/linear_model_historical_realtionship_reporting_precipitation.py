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
## Drop Queen Elizabeth - bad reporting generally, also have one represented for Lilongwe
weather_data_historical = weather_data_historical.drop("Queen Elizabeth Central Hospital", axis=1)
monthly_reporting_by_facility = monthly_reporting_by_facility.drop("Queen Elizabeth Central Hospital", axis=1)
## Drop September 2024
weather_data_historical = weather_data_historical.drop(weather_data_historical.index[-1])
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[-1])
## Drop before 2016?  12*6
weather_data_historical = weather_data_historical.drop(weather_data_historical.index[0:72]).reset_index(drop=True)
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[0:72]).reset_index(drop=True)

# Plot each facility's reporting data against weather data
plt.figure(figsize=(12, 6))

for facility in weather_data_historical.columns:
    plt.plot(weather_data_historical.index, monthly_reporting_by_facility[facility], label=facility)
months = weather_data_historical.index
year_labels = range(2017, 2025, 1)
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
year = range(2016, 2024, 1) # year as a fixed effect
year_repeated = [y for y in year for _ in range(12)]
year = year_repeated[:-4]
year_flattened = year*len(weather_data_historical.columns) # to get flattened data

# month
month = range(12)
month_repeated = [m for m in month for _ in range(2016, 2024, 1)]
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

model = sm.OLS(y,X)
results = model.fit()

print(results.summary())

# # ## Linear regression - by facility
results_list = []
#
for facility in monthly_reporting_by_facility.columns:
    y = monthly_reporting_by_facility[facility].values
    weather = weather_data_historical[facility].values
    X = np.column_stack((weather, year, month))

    model = sm.OLS(y, X)
    results = model.fit()

    #print(results.summary())


## Binary above/below average for that month
X_df = pd.DataFrame({
    'weather_data': weather_data,
    'year': year_flattened,
    'month': month_flattened,
    'facility': facility_flattened
})

grouped_data = X_df.groupby(['facility', 'month'])['weather_data'].mean().reset_index()
above_below_average = []
for facility in range(len(monthly_reporting_by_facility.columns)):
    for month in range(12):
        average_for_month = grouped_data[(grouped_data["facility"] == facility) & (grouped_data["month"] == month)][
            "weather_data"]
        X_data = X_df[(X_df["month"] == month) & (X_df["facility"] == facility)]
        for value in X_data["weather_data"]:
            above_below_average.append(1 if value > average_for_month.values[0] else 0)


# Add the binary variable to the predictors
X = pd.DataFrame({
    'weather_data': weather_data,
    'year': year_flattened,
    'month': month_flattened,
    'facility': facility_flattened,
    'precip_above_average': above_below_average
})
# One-hot encode the 'facility' column for a fixed effect
facility_encoded = pd.get_dummies(X['facility'])

X = np.column_stack((X[['weather_data', 'year', 'month', 'precip_above_average']], facility_encoded))
y = monthly_reporting_by_facility.values.flatten()

print(len(y))
model = sm.OLS(y,X)
results = model.fit()

print(results.summary())


### Top 80 percentile?
X_df = pd.DataFrame({
    'weather_data': weather_data,
    'year': year_flattened,
    'month': month_flattened,
    'facility': facility_flattened
})

grouped_data_based_on_percentiles = X_df.groupby(['facility', 'month'])['weather_data'].quantile(0.8).reset_index()

above_80_percentile = []
for facility in range(len(monthly_reporting_by_facility.columns)):
    for month in range(12):
        percentile_for_month = grouped_data_based_on_percentiles[(grouped_data_based_on_percentiles["facility"] == facility) & (grouped_data_based_on_percentiles["month"] == month)][
            "weather_data"]
        X_data = X_df[(X_df["month"] == month) & (X_df["facility"] == facility)]
        for value in X_data["weather_data"]:
            above_80_percentile.append(1 if value > percentile_for_month.values[0] else 0)


# Add the binary variable to the predictors
X = pd.DataFrame({
    'weather_data': weather_data,
    'year': year_flattened,
    'month': month_flattened,
    'facility': facility_flattened,
    'precip_above_average': above_below_average,
    'above_80_percentile': above_80_percentile
})
# One-hot encode the 'facility' column for a fixed effect
facility_encoded = pd.get_dummies(X['facility'])

X = np.column_stack((X[['weather_data', 'year', 'month', 'precip_above_average', 'above_80_percentile']], facility_encoded))
y = monthly_reporting_by_facility.values.flatten()

print(len(y))
model = sm.OLS(y,X)
results = model.fit()

print(results.summary())


