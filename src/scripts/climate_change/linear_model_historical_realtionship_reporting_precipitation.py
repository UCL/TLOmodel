import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
from collections import defaultdict

ANC = True
daily_max = False
daily_total = False
min_year_for_analyis = 2011
absolute_min_year = 2011
mask_threshold = 0
five_day = True
cumulative = True
# # data is from 2011 - 2024 - for facility
if ANC:
    monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_ANC_by_smaller_facility_lm.csv", index_col=0)
    if daily_max:
        if five_day:
            if cumulative:
                weather_data_historical = pd.read_csv(
                    "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_maximum/historical_daily_total_by_facilities_with_ANC_five_day_cumulative.csv",
                    index_col=0)
            else:
                weather_data_historical = pd.read_csv(
                    "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_maximum/historical_daily_total_by_facilities_with_ANC_five_day_average.csv",
                    index_col=0)
        else:
            weather_data_historical = pd.read_csv(
                "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_maximum/historical_daily_max_by_facilities_with_ANC.csv",
                index_col=0)
    elif daily_total:
        weather_data_historical = pd.read_csv(
                "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facilities_with_ANC.csv",
                index_col=0)
    else:
        weather_data_historical = pd.read_csv(
            "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv",
            index_col=0)

else:
    monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_by_smaller_facility_lm.csv", index_col=0)
    if daily_max:
        weather_data_historical = pd.read_csv(
            "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_maximum/historical_daily_max_by_facility.csv",
            index_col=0)
    elif daily_total:
        weather_data_historical = pd.read_csv(
            "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facility.csv",
            index_col=0)
    else:
        weather_data_historical = pd.read_csv(
            "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facility_lm.csv",
            index_col=0)


def build_model(X, y, scale_y=False, beta=False, X_mask_mm=0):
    epsilon = 1e-5
    if scale_y:
        y_scaled = np.clip(y / 100, epsilon, 1 - epsilon)
    else:
        y_scaled = y
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_scaled) & (X[:, 0] >= X_mask_mm)
    model = BetaModel(y_scaled[mask], X[mask]) if beta else sm.OLS(y_scaled[mask], X[mask])
    model_fit = model.fit()
    predictions = model_fit.predict(X[mask])
    residuals = y_scaled[mask] - predictions
    return model_fit, predictions, residuals, mask

def create_binary_feature(threshold, weather_data_df, recent_months):
    binary_feature_list = []
    for facility in weather_data_df.columns:
        facility_data = weather_data_df[facility]

        for i in range(len(facility_data)):
            facility_threshold = threshold[i] if hasattr(threshold, "__len__") else threshold

            if i >= recent_months:
                last_x_values = facility_data[i - recent_months:i]
                binary_feature_list.append(1 if (last_x_values > facility_threshold).any() else 0)
            else:
                binary_feature_list.append(np.nan)

    return binary_feature_list

## Drop September 2024 -
weather_data_historical = weather_data_historical.drop(weather_data_historical.index[-1])
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[-1])

## Drop before 2017
weather_data_historical = weather_data_historical.iloc[(min_year_for_analyis-absolute_min_year)*12 :]
monthly_reporting_by_facility = monthly_reporting_by_facility.iloc[(min_year_for_analyis-absolute_min_year)*12:]
## Linear regression
month_range = range(12)
num_facilities = len(weather_data_historical.columns)
year_range = range(min_year_for_analyis, 2025, 1) # year as a fixed effect
year_repeated = [y for y in year_range for _ in range(12)]
year = year_repeated[:-4]
year_flattened = year*len(weather_data_historical.columns) # to get flattened data
month = range(12)
month_repeated = []
for _ in year_range:
    month_repeated.extend(range(1, 13))
month = month_repeated[:-4]
month_flattened = month*len(weather_data_historical.columns)

facility_flattened = list(range(len(weather_data_historical.columns))) * len(month)

# Flatten data
weather_data = weather_data_historical.values.flatten()
y = monthly_reporting_by_facility.values.flatten()
# One-hot encode facilities
facility_encoded = pd.get_dummies(facility_flattened, drop_first=True)

# Above/below average for each month
grouped_data = pd.DataFrame({
    'facility': facility_flattened,
    'month': month_flattened,
    'weather_data': weather_data
}).groupby(['facility', 'month'])['weather_data'].mean().reset_index()

above_below_average = create_binary_feature(
    grouped_data.groupby(['facility', 'month'])['weather_data'].transform('mean'), weather_data_historical, 0
)
above_below_X = create_binary_feature(1000, weather_data_historical, 12)

# Prepare additional facility info
if ANC:
    expanded_facility_info = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm_with_ANC.csv", index_col=0)

else:
    expanded_facility_info = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm.csv", index_col=0)
expanded_facility_info = expanded_facility_info.T.reindex(columns=expanded_facility_info.index)

def repeat_info(info, num_facilities, year_range):
    repeated_info = [i for i in info for _ in range(12) for _ in year_range]
    return repeated_info[:-4 * num_facilities]  # Exclude first final months (Sept - Dec 2024)

zone_info_each_month = repeat_info(expanded_facility_info["Zonename"], num_facilities, year_range)
zone_encoded = pd.get_dummies(zone_info_each_month, drop_first=True)
resid_info_each_month = repeat_info(expanded_facility_info['Resid'], num_facilities, year_range)
resid_encoded = pd.get_dummies(resid_info_each_month, drop_first=True)
owner_info_each_month = repeat_info(expanded_facility_info['A105'], num_facilities, year_range)
owner_encoded = pd.get_dummies(owner_info_each_month, drop_first=True)
ftype_info_each_month = repeat_info(expanded_facility_info['Ftype'], num_facilities, year_range)
ftype_encoded = pd.get_dummies(ftype_info_each_month, drop_first=True)
altitude = [float(x) for x in repeat_info(expanded_facility_info['A109__Altitude'], num_facilities, year_range)]
# Lagged weather
lag_1_month = weather_data_historical.shift(1).values.flatten()
lag_3_month = weather_data_historical.shift(3).values.flatten()

altitude = np.array(altitude)
altitude = np.where(altitude < 0, np.nan, altitude)
altitude = list(altitude)

#### STEP 1: GENERATE PREDICTIONS OF ANC DATA ###########
X = np.column_stack([
    year_flattened,
    month_flattened,
    resid_encoded,
    zone_encoded,
    owner_encoded,
    ftype_encoded,
    facility_encoded,
    altitude
])
results, y_pred, residuals, mask  = build_model(X, np.log(y) , X_mask_mm = mask_threshold)

print(results)
year_month_labels = np.array([f"{y}-{m}" for y, m in zip(year_flattened, month_flattened)])
X_filtered = X[mask]
y_filtered = y[mask]
year_month_labels_filtered = year_month_labels[mask]
# residuals are the next "y" variable, as they are the defecit in cases
plt.scatter(year_month_labels_filtered, np.log(y_filtered), color='#1C6E8C', alpha=0.5, label='Actual data')
plt.scatter(year_month_labels_filtered, y_pred, color='#9AC4F8', alpha=0.7, label='Predicted data')
#plt.xticks(ticks=, labels=sorted_years, rotation=45, ha='right')

plt.xticks(rotation=45, ha='right')
plt.xlabel('Year')
plt.ylabel('Log(Number of ANC visits)')
plt.title('Monthly ANC Visits vs. Precipitation')
plt.ylim(0, 10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()

### STEP 2 - USE THESE IN PREDICTIONS###

y = residuals

X = np.column_stack([
    weather_data,
    year_flattened,
    month_flattened,
    resid_encoded,
    zone_encoded,
    owner_encoded,
    ftype_encoded,
    lag_1_month,
    lag_3_month,
    facility_encoded,
    altitude
])
results, y_pred, residuals, mask  = build_model(X, np.log(y) , X_mask_mm = mask_threshold)

if np.nanmin(y) < 1:
    y += 1e-6  # Shift to ensure positivity

### Now include only significant predictors
X = np.column_stack([
    weather_data,
    year_flattened,
    month_flattened,
    resid_encoded,
    zone_encoded,
    owner_encoded,
    ftype_encoded,
    lag_1_month,
    lag_3_month,
    altitude,
    above_below_X
])
results, y_pred, residuals, mask  = build_model(X, np.log(y) , X_mask_mm = mask_threshold)

print(results.summary())


##### Plot y_predic

X_filtered = X[mask]
if ANC:
    plt.scatter(X_filtered[:, 0], np.log(y) [mask], color='red', alpha=0.5)
    plt.scatter(X_filtered[:, 0], y_pred)
    plt.title(' ')
    plt.ylabel('Number of ANC visits')
    plt.xlabel('Precip (mm)')
    plt.ylim(0,10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
else:

    plt.scatter(X_filtered[:, 0], np.log(y)[mask], color='red', alpha=0.5)
    plt.scatter(X_filtered[:, 0], y_pred)
    plt.title(' ')
    plt.ylabel('Reporting (%)')
    plt.xlabel('Precip (mm)')
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #plt.show()

# save model
#
# # Save the model using pickle
# with open('linear_model_ANC_daily_max.pkl', 'wb') as file:
#     pickle.dump(results, file)
#
# # Now you can load the model and use it for predictions
# with open('saved_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)


year_month_labels = np.array([f"{y}-{m}" for y, m in zip(year_flattened, month_flattened)])
X_filtered = X[mask]
y_filtered = y[mask]
year_month_labels_filtered = year_month_labels[mask]
# first_index_by_year = {}
# years_in_labels = [label[:4] for label in year_month_labels_filtered]
# year_counts = defaultdict(int)
# for year in years_in_labels:
#     year_counts[year] += 1
# print(year_counts)
# sorted_years = sorted(year_counts.keys())
# print(sorted_years)
# cumulative_counts = []
# cumulative_sum = 0
# for year in sorted_years:
#     cumulative_sum += year_counts[year]
#     cumulative_counts.append(cumulative_sum)
# cumulative_counts = [first_index_by_year[year] for year in sorted_years]
#
print(year_month_labels_filtered)
plt.figure(figsize=(12, 6))
plt.scatter(year_month_labels_filtered, np.log(y_filtered), color='#1C6E8C', alpha=0.5, label='Actual data')
plt.scatter(year_month_labels_filtered, y_pred, color='#9AC4F8', alpha=0.7, label='Predicted data')
#plt.xticks(ticks=, labels=sorted_years, rotation=45, ha='right')

plt.xticks(rotation=45, ha='right')
plt.xlabel('Year')
plt.ylabel('Log(Number of ANC visits)')
plt.title('Monthly ANC Visits vs. Precipitation')
plt.ylim(0, 10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()
