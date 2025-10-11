import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM

min_year_for_analyis = 2015
absolute_min_year = 2011
mask_threshold = 0
use_percentile_mask_threshold = False
log_y = False # will use a binary outcome

covid_months = range((2020 - min_year_for_analyis)* 12 + 4, (2020 - min_year_for_analyis)* 12 + 4 + 20) # Bingling's paper: disruption between April 2020 and Dec 2021, a period of 20 months
cyclone_freddy_months_phalombe = range((2023 - min_year_for_analyis)* 12 + 4, (2020 - min_year_for_analyis)* 12 + 4 + 14) # From news report and DHIS2, see disruption from April 2023 - June 2024, 14 months
cyclone_freddy_months_thumbwe = range((2023 - min_year_for_analyis)* 12 + 3, (2020 - min_year_for_analyis)* 12 + 3 + 12) # From news report and DHIS2, see disruption from March 2023 - March 2024, 12 months

############# Read in data ###########
# # data is from 2011 - 2024 - for facility
monthly_reporting_by_facility = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_ANC_by_smaller_facility_lm.csv", index_col=0)
### Combine weather variables ##
weather_data_monthly = pd.read_csv(
                "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv",
                index_col=0)

weather_data_five_day_cumulative = pd.read_csv(
                        "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facilities_with_ANC_five_day_cumulative.csv",
                        index_col=0)


def build_model(X, y, X_mask_mm=0):

    mask = (~np.isnan(X).any(axis=1) & ~np.isnan(y) & (X[:, 0] >= X_mask_mm))
    model = sm.GLM(y[mask], X[mask], family=Binomial())
    model_fit = model.fit()
    return model_fit, model_fit.predict(X[mask]), mask

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



##############################################################################################
########################## STEP 0: Tidy data ##########################
##############################################################################################
## Remove any columns that sum to 0 in the monthly reporting data (e.g. for inpatient data, may mean they don't have the facility)
zero_sum_columns = monthly_reporting_by_facility.columns[(monthly_reporting_by_facility.sum(axis=0) == 0)]
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(columns=zero_sum_columns)

# Prep weather data
weather_data_monthly = weather_data_monthly.drop(columns=zero_sum_columns, errors='ignore')
weather_data_five_day_cumulative = weather_data_five_day_cumulative.drop(columns=zero_sum_columns, errors='ignore')

weather_data_monthly = weather_data_monthly.drop(weather_data_monthly.index[-2:])
weather_data_five_day_cumulative = weather_data_five_day_cumulative.drop(weather_data_five_day_cumulative.index[-1:])

    # code if years need to be dropped
weather_data_monthly = weather_data_monthly.iloc[(min_year_for_analyis - absolute_min_year) * 12:]
weather_data_five_day_cumulative = weather_data_five_day_cumulative.iloc[(min_year_for_analyis - absolute_min_year) * 12:]
weather_data_monthly_flattened = weather_data_monthly.values.flatten()
weather_data_five_day_cumulative_flattened = weather_data_five_day_cumulative.values.flatten()
weather_data = np.vstack((weather_data_monthly_flattened,weather_data_five_day_cumulative_flattened)).T


# Drop September 2024 in ANC/reporting data
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[-1])
# code if years need to be dropped
monthly_reporting_by_facility = monthly_reporting_by_facility.iloc[(min_year_for_analyis-absolute_min_year)*12:]
# Linear regression
month_range = range(12)
num_facilities = len(monthly_reporting_by_facility.columns)
year_range = range(min_year_for_analyis, 2025, 1) # year as a fixed effect
year_repeated = [y for y in year_range for _ in range(12)]
year = year_repeated[:-4]
year_flattened = year*len(monthly_reporting_by_facility.columns) # to get flattened data
month = range(12)
month_repeated = []
for _ in year_range:
    month_repeated.extend(range(1, 13))
month = month_repeated[:-4]
month_flattened = month*len(monthly_reporting_by_facility.columns)

facility_flattened = list(range(len(monthly_reporting_by_facility.columns))) * len(month)

# Flatten data
y = monthly_reporting_by_facility.values.flatten()
#y[np.isnan(y)] = 0 # if all of these are expected to report, then can I assume all 0?
y[~np.isnan(y)] = 0
y[np.isnan(y)] = 1 # create binary outcome
print(y)

if use_percentile_mask_threshold:
    mask_threshold = np.nanpercentile(weather_data, 90)
    print(mask_threshold)

# One-hot encode facilities
facility_encoded = pd.get_dummies(facility_flattened, drop_first=True)

    # Above/below average for each month
grouped_data = pd.DataFrame({
        'facility': facility_flattened,
        'month': month_flattened,
        'weather_data': weather_data_monthly_flattened
    }).groupby(['facility', 'month'])['weather_data'].mean().reset_index()

above_below_X = create_binary_feature(np.nanpercentile(weather_data_monthly_flattened, 90), weather_data_monthly, 12)

# Prepare additional facility info
expanded_facility_info = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm_with_ANC.csv", index_col=0)
expanded_facility_info = expanded_facility_info.drop(columns=zero_sum_columns)

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
minimum_distance = [float(x) for x in repeat_info(expanded_facility_info['minimum_distance'], num_facilities, year_range)]

# Lagged weather
lag_1_month = weather_data_monthly.shift(1).values.flatten()
lag_2_month = weather_data_monthly.shift(2).values.flatten()
lag_3_month = weather_data_monthly.shift(3).values.flatten()
lag_4_month = weather_data_monthly.shift(4).values.flatten()


altitude = np.array(altitude)
altitude = np.where(altitude < 0, np.nan, altitude)
altitude = list(altitude)


# ##############################################################################################
# ########################## STEP 1: GENERATE PREDICTIONS OF ANC DATA ##########################
# ##############################################################################################
#
# X = np.column_stack([
#     year_flattened,
#     month_flattened,
#     resid_encoded,
#     zone_encoded,
#     owner_encoded,
#     ftype_encoded,
#     facility_encoded,
#     altitude,
#     minimum_distance
# ])
#
# results, y_pred, mask_ANC_data = build_model(X , y, X_mask_mm=mask_threshold)
#
#
#
# print("ANC prediction", results.summary())
#
# # plot
# year_month_labels = np.array([f"{y}-{m}" for y, m in zip(year_flattened, month_flattened)])
# y_filtered = y[mask_ANC_data]
# year_month_labels_filtered = year_month_labels[mask_ANC_data]
# data_ANC_predictions = pd.DataFrame({
#             'Year_Month': year_month_labels_filtered,
#             'y_filtered': y_filtered,
#             'y_pred': y_pred,
#              'residuals': y_filtered - y_pred
#     })
#
# data_ANC_predictions = data_ANC_predictions.sort_values(by='Year_Month').reset_index(drop=True)
# x_labels = data_ANC_predictions['Year_Month'][::num_facilities*12]
#
# # Set the xticks at corresponding positions
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))
# step = num_facilities * 12
# data_ANC_predictions_grouped = data_ANC_predictions.groupby('Year_Month').mean().reset_index()
#
# xticks = data_ANC_predictions['Year_Month'][::len(year_range)*num_facilities]
# # Panel A: Actual data and predictions
# axs[0].scatter(data_ANC_predictions['Year_Month'], data_ANC_predictions['y_filtered'], color='#1C6E8C', alpha=0.5, label='Actual data')
# axs[0].scatter(data_ANC_predictions['Year_Month'], data_ANC_predictions['y_pred'], color='#9AC4F8', alpha=0.7, label='Predictions')
# axs[0].scatter(data_ANC_predictions_grouped['Year_Month'], data_ANC_predictions_grouped['y_filtered'], color='red', alpha=0.5, label='Mean Actual data')
# axs[0].scatter(data_ANC_predictions_grouped['Year_Month'], data_ANC_predictions_grouped['y_pred'], color='yellow', alpha=0.7, label='Mean Predictions')
#
# axs[0].set_xticks(xticks)
# axs[0].set_xticklabels(xticks, rotation=45, ha='right')
# axs[0].set_xlabel('Year')
# axs[0].set_ylabel('Number of ANC visits')
# axs[0].set_title('A: Monthly ANC Visits vs. Precipitation')
# axs[0].legend(loc='upper left')
#
# plt.tight_layout()
# plt.show()
#
# ########### Add in weather data ############
#
#
# X_weather = np.column_stack([
#         weather_data,
#         np.array(year_flattened),
#         np.array(month_flattened),
#         resid_encoded,
#         zone_encoded,
#         owner_encoded,
#         ftype_encoded,
#         lag_1_month,
#         lag_2_month,
#         lag_3_month,
#         lag_4_month,
#         facility_encoded,
#         np.array(altitude),
#         np.array(minimum_distance),
#         above_below_X
#     ])
#
# results_of_weather_model, y_pred_weather, mask_all_data = build_model(X_weather, y,
#                                                                  X_mask_mm=mask_threshold)
# print("All predictors", results_of_weather_model.summary())
# #
# X_filtered = X_weather[mask_all_data]
#
# # Effect size
#
# y_mean = np.mean(y[mask_all_data])
# SS_total = np.sum((y[mask_all_data] - y_mean) ** 2)
#
# predictor_variances = np.var(X_filtered, axis=0, ddof=1)
# coefficients = results_of_weather_model.params
# SS_effect = coefficients**2 * predictor_variances
# eta_squared = SS_effect / SS_total
# effect_size_summary = pd.DataFrame({
#     'Coefficient': coefficients,
#     'SS_effect': SS_effect,
#     'Eta-squared': eta_squared
# }).sort_values(by='Eta-squared', ascending=False)
#
# print(effect_size_summary)
#
#
# fig, axs = plt.subplots(1, 2, figsize=(10, 6))
#
# indices_ANC_data = np.where(mask_ANC_data)[0]
# indices_all_data = np.where(mask_all_data)[0]
# common_indices = np.intersect1d(indices_ANC_data, indices_all_data)
# matched_y_pred = y_pred[np.isin(indices_ANC_data, common_indices)]
# matched_y_pred_weather = y_pred_weather[np.isin(indices_all_data, common_indices)]
#
# axs[0].scatter(X_filtered[:, 0], y[mask_all_data], color='red', alpha=0.5, label = 'Non weather model')
# axs[0].hlines(y = 0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], color = 'black', linestyle = '--')
# axs[0].scatter(X_filtered[:, 0], matched_y_pred_weather, label='Weather model')
# axs[0].hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], color='black', linestyle='--')
# axs[0].set_ylabel('ANC visits')
#
# plt.show()
#
# ## See impact on reporting
#
# predicted_missingness = np.zeros(len(matched_y_pred))
# predicted_missingness[matched_y_pred_weather > 0.5 ] = 1
#
#
# fig, axs = plt.subplots(1, 2, figsize=(10, 6))
#
# axs[0].scatter(X_filtered[:, 0], predicted_missingness, color='red', alpha=0.5)
# axs[0].hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], color='black', linestyle='--')
# axs[0].set_ylabel('Missing data presence/absence')
# axs[0].set_ylabel('Monthly total precipitation (mm)')
#
# axs[1].scatter(X_filtered[:, 1], predicted_missingness, color='red', alpha=0.5)
# axs[1].hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], color='black', linestyle='--')
# axs[1].set_ylabel('Missing data presence/absence')
# axs[1].set_ylabel('Five day cumulative total precipitation (mm)')
#
#
# plt.show()
#


### Difference in weather data ####
########### Add in weather data ############

print(weather_data[:,0])
X_weather_1 = np.column_stack([
        weather_data[:,0],
        np.array(year_flattened),
        np.array(month_flattened),
        resid_encoded,
        zone_encoded,
        owner_encoded,
        ftype_encoded,
        lag_1_month,
        lag_2_month,
        lag_3_month,
        lag_4_month,
        facility_encoded,
        np.array(altitude),
        np.array(minimum_distance),
        above_below_X
    ])

results_of_weather_model_1, y_pred_weather_1, mask_all_data_1 = build_model(X_weather_1, y,
                                                                 X_mask_mm=mask_threshold)



X_weather_2 = np.column_stack([
        weather_data[:,1],
        np.array(year_flattened),
        np.array(month_flattened),
        resid_encoded,
        zone_encoded,
        owner_encoded,
        ftype_encoded,
        lag_1_month,
        lag_2_month,
        lag_3_month,
        lag_4_month,
        facility_encoded,
        np.array(altitude),
        np.array(minimum_distance),
        above_below_X
    ])

results_of_weather_model_2, y_pred_weather_2, mask_all_data_2 = build_model(X_weather_2, y,
                                                                 X_mask_mm=mask_threshold)
print("All predictors", results_of_weather_model_1.summary())
print("All predictors", results_of_weather_model_2.summary())

#
X_filtered_1 = X_weather_1[mask_all_data_1]
X_filtered_2 = X_weather_2[mask_all_data_2]

## See impact on reporting

predicted_missingness_1 = np.zeros(len(y_pred_weather_1))
predicted_missingness_1[y_pred_weather_1 > 0.5 ] = 1

predicted_missingness_2 = np.zeros(len(y_pred_weather_2))
predicted_missingness_2[y_pred_weather_2 > 0.5 ] = 1

print(sum(predicted_missingness_1) - sum(predicted_missingness_2))
