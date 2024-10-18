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
weather_data_historical = weather_data_historical.drop(weather_data_historical.index[-1])
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[-1])

# ## Drop 2011-2017 7*12
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
month_range = range(12)
num_facilities = len(weather_data_historical.columns)
year_range = range(2011, 2025, 1) # year as a fixed effect
year_repeated = [y for y in year_range for _ in range(12)]
year = year_repeated[:-4]
year_flattened = year*len(weather_data_historical.columns) # to get flattened data

# month
month = range(12)
month_repeated = [m for m in month for _ in year_range]
month = month_repeated[:-4]
month_flattened = month*len(weather_data_historical.columns)

# facility as fixed effect
facility_flattened = list(range(len(weather_data_historical.columns))) * len(month)

# Flatten data
weather_data = weather_data_historical.values.flatten()
y = monthly_reporting_by_facility.values.flatten()

# Function to build model
def build_model(predictors, dependent_var, scale_y=True, binomial=True):
    X = np.column_stack(predictors)
    y_scaled = (dependent_var / 100) if scale_y else dependent_var
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_scaled)
    model = sm.GLM(y_scaled[mask], X[mask], family=sm.families.Binomial()) if binomial else sm.OLS(y_scaled[mask], X[mask])
    return model.fit()

# One-hot encode facilities
facility_encoded = pd.get_dummies(facility_flattened)

print(len(facility_flattened))
print(len(month_flattened))
print(len(weather_data))

# Above/below average for each month
grouped_data = pd.DataFrame({
    'facility': facility_flattened,
    'month': month_flattened,
    'weather_data': weather_data
}).groupby(['facility', 'month'])['weather_data'].mean().reset_index()


def create_binary_feature(threshold, weather_data_df, recent_months):
    binary_feature_list = []
    for facility in weather_data_df.columns:
        facility_data = weather_data_df[facility]

        for i in range(len(facility_data)):
            if hasattr(threshold, "__len__"):  # Check if threshold is iterable
                facility_threshold = threshold[i]
            else:
                facility_threshold = threshold  # Use scalar threshold if it's not iterable

            if i >= recent_months:
                last_x_values = facility_data[i - recent_months:i]
                binary_feature_list.append(1 if (last_x_values > facility_threshold).any() else 0)
            else:
                binary_feature_list.append(np.nan)

    return binary_feature_list
above_below_average = create_binary_feature(
    grouped_data.groupby(['facility', 'month'])['weather_data'].transform('mean'), weather_data_historical, 0
)

above_below_700 = create_binary_feature(700, weather_data_historical, 12)

# Build models
X_base = pd.DataFrame({
    'weather_data': weather_data,
    'year': year_flattened,
    'month': month_flattened,
    'precip_above_average': above_below_average,
    'precip_above_700': above_below_700
})

X = np.column_stack([X_base[['weather_data', 'year', 'month', 'precip_above_average', 'precip_above_700']], facility_encoded])
results = build_model([X], y)
print(results.summary())

# Exceeds threshold in last 12 months
exceeds_700_last_12 = [
    1 if (weather_data_historical[facility][i-12:i] > 700).any() else 0
    if i >= 12 else np.nan
    for facility in weather_data_historical.columns for i in range(len(weather_data_historical[facility]))
]
print(exceeds_700_last_12)
print(above_below_700)
X_base['exceeds_700_last_12_weather'] = exceeds_700_last_12
X = np.column_stack([X_base, facility_encoded])
results = build_model([X], y)
print(results.summary())

# Add additional covariates (zone, resid, etc.)
expanded_facility_info = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm.csv", index_col=0).T

expanded_facility_info = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm.csv", index_col=0)
expanded_facility_info = expanded_facility_info.T.reindex(columns=expanded_facility_info.index)
def repeat_info(info, num_facilities, year_range):
    repeated_info = [i for i in info for _ in range(12) for _ in year_range]
    return repeated_info[4 * num_facilities:]  # Exclude first 4 months (Sept - Dec 2024)

# Zone information encoding
zone_info_each_month = repeat_info(expanded_facility_info["Zonename"], num_facilities, year_range)
zone_encoded = pd.get_dummies(zone_info_each_month)

# Resid information encoding
resid_info_each_month = repeat_info(expanded_facility_info['Resid'], num_facilities, year_range)
resid_encoded = pd.get_dummies(resid_info_each_month)

# Owner information encoding
owner_info_each_month = repeat_info(expanded_facility_info['A105'], num_facilities, year_range)
owner_encoded = pd.get_dummies(owner_info_each_month)

X = np.column_stack([X_base, facility_encoded, resid_encoded, resid_encoded, owner_encoded])
results = build_model([X], y)
print(results.summary())

# Add lagged model
lag_1_month = weather_data_historical.shift(1).values.flatten()
lag_3_month = weather_data_historical.shift(3).values.flatten()

X_lagged = np.column_stack([X_base[['weather_data', 'year', 'month', 'precip_above_700', 'exceeds_700_last_12_weather']], lag_1_month, lag_3_month, facility_encoded])
results = build_model([X_lagged], y)
print(results.summary())
