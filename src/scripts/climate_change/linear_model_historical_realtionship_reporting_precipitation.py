import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# # data is from 2011 - 2024 - for facility
monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_by_smaller_facility_lm.csv", index_col=0)
weather_data_historical = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facility_lm.csv", index_col=0)
print(len(monthly_reporting_by_facility.columns))
print(len(weather_data_historical.columns))
# monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_by_DHO_lm.csv", index_col=0)
# weather_data_historical = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_DHO_lm.csv", index_col=0)

# Plot each facility's reporting data against weather data
# plt.figure(figsize=(12, 6))
# for facility in weather_data_historical.columns:
#     plt.plot(weather_data_historical.index, monthly_reporting_by_facility, label=facility)
# months = weather_data_historical.index
# year_labels = range(2011, 2025, 1)
# year_ticks = range(0, len(months), 12)
# plt.xticks(year_ticks, year_labels, rotation=90)
# plt.xlabel('Year')
# plt.ylabel('Reporting %')
# plt.title('Reporting by Facility')
# plt.legend(title='Facilities', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid()
# plt.tight_layout()
# #plt.show()

## Drop September 2024
weather_data_historical = weather_data_historical.drop(weather_data_historical.index[1])
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[1])

## Drop 2011-2017 7*12
# weather_data_historical = weather_data_historical.drop(weather_data_historical.index[0:84]).reset_index(drop=True)
# monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[0:84]).reset_index(drop=True)

# Plot each facility's reporting data against weather data
# plt.figure(figsize=(12, 6))
#
# for facility in weather_data_historical.columns:
#     plt.plot(weather_data_historical.index, monthly_reporting_by_facility, label=facility)
# months = weather_data_historical.index
# year_labels = range(2015, 2025, 1)
# year_ticks = range(0, len(months), 12)
# plt.xticks(year_ticks, year_labels, rotation=90)
# plt.xlabel('Weather Data')
# plt.ylabel('Reporting')
# plt.title('Reporting vs. Weather Data by Facility')
# plt.legend(title='Facilities', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid()
# plt.tight_layout()
# #plt.show()



## Linear regression - flattened
# year
year_range = range(2011, 2025, 1) # year as a fixed effect
year_repeated = [y for y in year_range for _ in range(12)]
year = year_repeated[:-4]
year_flattened = year*len(weather_data_historical.columns) # to get flattened data

# month
month = range(12)
month_repeated = [m for m in month for _ in year_range]
month = month_repeated[4:]
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
mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
model = sm.OLS(y[mask],X[mask])
results = model.fit()

print(results.summary())


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

mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
model = sm.OLS(y[mask],X[mask])
results = model.fit()
print(results.summary())




## Binary above/below 300mm for that month (very heavy rain)
X_df = pd.DataFrame({
    'weather_data': weather_data,
    'year': year_flattened,
    'month': month_flattened,
    'facility': facility_flattened
})

grouped_data = X_df.groupby(['facility', 'month'])['weather_data'].mean().reset_index()
above_below_700 = []
for facility in range(len(monthly_reporting_by_facility.columns)):
    for month in range(12):
        X_data = X_df[(X_df["month"] == month) & (X_df["facility"] == facility)]
        for value in X_data["weather_data"]:

            above_below_700.append(1 if value > 700 else 0)


# Add the binary variable to the predictors
X = pd.DataFrame({
    'weather_data': weather_data,
    'year': year_flattened,
    'month': month_flattened,
    'facility': facility_flattened,
    'precip_above_700': above_below_700
})
# One-hot encode the 'facility' column for a fixed effect
facility_encoded = pd.get_dummies(X['facility'])

X = np.column_stack((X[['weather_data', 'year', 'month', 'precip_above_700']], facility_encoded))
y = monthly_reporting_by_facility.values.flatten()

mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
model = sm.OLS(y[mask],X[mask])
results = model.fit()
print(results.summary())

######### Now has it exceeded in previous 12 months


exceeds_700_last_12_weather = []

for facility in weather_data_historical.columns:
    facility_data = weather_data_historical[facility]

    for i in range(len(facility_data)):
        if i >= 12:
            last_12_values = facility_data[i - 12:i]
            exceeds_700_last_12_weather.append(1 if (last_12_values > 700).any() else 0)
        else:
            exceeds_700_last_12_weather.append(np.nan)

X = pd.DataFrame({
    'weather_data': weather_data,
    'year': year_flattened,
    'month': month_flattened,
    'facility': facility_flattened,
    'precip_above_700': above_below_700,
    'exceeds_700_last_12_weather': exceeds_700_last_12_weather
})
facility_encoded = pd.get_dummies(X['facility'])

X = np.column_stack(
    (X[['weather_data', 'year', 'month', 'precip_above_700', 'exceeds_700_last_12_weather']], facility_encoded))
y = monthly_reporting_by_facility.values.flatten()

mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
model = sm.OLS(y[mask], X[mask])
results = model.fit()

print(results.summary())

########### Add other covariates

expanded_facility_info = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm.csv", index_col=0)
expanded_facility_info = expanded_facility_info.T.reindex(columns=expanded_facility_info.index)
print(len(expanded_facility_info))
zone_info = expanded_facility_info["Zonename"]
zone_info_each_month = [z for z in zone_info for _ in range(12) for _ in year_range]
zone_info_each_month = zone_info_each_month[4*len(monthly_reporting_by_facility.columns):] # first four months, no data (Sept - Dec 2024)

zone_encoded = pd.get_dummies(zone_info_each_month)
resid_info = expanded_facility_info['Resid']
resid_info_each_month = [r for r in resid_info for _ in range(12) for _ in year_range]
resid_info_each_month = resid_info_each_month[4*len(monthly_reporting_by_facility.columns):] # first four months, no data (Sept - Dec 2024)
resid_encoded = pd.get_dummies(resid_info_each_month)
owner_info = expanded_facility_info['A105']
owner_info_each_month = [o for o in owner_info for _ in range(12) for _ in year_range]
owner_info_each_month = owner_info_each_month[4*len(monthly_reporting_by_facility.columns):] # first four months, no data (Sept - Dec 2024)

owner_encoded = pd.get_dummies(owner_info_each_month)
X = pd.DataFrame({
    'weather_data': weather_data,
    'year': year_flattened,
    'month': month_flattened,
    'facility': facility_flattened,
    'precip_above_700': above_below_700,
    'exceeds_700_last_12_weather': exceeds_700_last_12_weather
})
print(X)
facility_encoded = pd.get_dummies(X['facility'])

X = np.column_stack(
    (X[['weather_data', 'year', 'month', 'precip_above_700', 'exceeds_700_last_12_weather']], facility_encoded, resid_encoded,zone_encoded, owner_encoded))
y = monthly_reporting_by_facility.values.flatten()

mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
model = sm.OLS(y[mask], X[mask])
results = model.fit()

print(results.summary())
