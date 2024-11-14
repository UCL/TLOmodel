import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
import joblib


ANC = True
daily_max = False
daily_total = True
min_year_for_analyis = 2011
absolute_min_year = 2011
mask_threshold = 0
five_day = True
cumulative = True
model_fit_ANC_data = False
model_fit_weather_data = True
model_filename = (
    f"best_model_{'ANC' if ANC else 'Reporting'}_prediction_"
    f"{'5_day' if five_day else 'monthly'}_"
    f"{'cumulative' if cumulative else ('max' if daily_max else 'total')}_precip.pkl"
)

model_filename_weather_model = (
    f"best_model_weather_"
    f"{'5_day' if five_day else 'monthly'}_"
    f"{'cumulative' if cumulative else ('max' if daily_max else 'total')}_precip.pkl"
)

log_y = True
# # data is from 2011 - 2024 - for facility
if ANC:
    monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_ANC_by_smaller_facility_lm.csv", index_col=0)
    if daily_max:
        if five_day:
            if cumulative:
                weather_data_historical = pd.read_csv(
                    "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facilities_with_ANC_five_day_cumulative.csv",
                    index_col=0)
            else:
                weather_data_historical = pd.read_csv(
                    "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facilities_with_ANC_five_day_average.csv",
                    index_col=0)
        else:
            weather_data_historical = pd.read_csv(
                "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facility_five_day_cumulative.csv",
                index_col=0)
    elif daily_total:
        if five_day:
            if cumulative:
                weather_data_historical = pd.read_csv(
                        "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facilities_with_ANC_five_day_cumulative.csv",
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

def build_model(X, y, scale_y=False, beta=False, log_y=False, X_mask_mm=0):
    epsilon = 1e-5

    if log_y:
        y = np.log(np.clip(y, epsilon, None))  # Log-transform y with clipping for positivity
    if scale_y:
        y_scaled = np.clip(y / 100, epsilon, 1 - epsilon)
    else:
        y_scaled = y
    mask = (~np.isnan(X).any(axis=1) & ~np.isnan(y_scaled) & (X[:, 0] >= X_mask_mm) & (y <= 1e4))
    model = BetaModel(y_scaled[mask], X[mask]) if beta else sm.OLS(y_scaled[mask], X[mask])
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

def stepwise_selection(X, y, log_y, p_value_threshold=0.05):
    included = []
    current_aic = np.inf

    while True:
        changed = False

        # Step 1: Try adding each excluded predictor and select the best one by AIC if significant
        excluded = list(set(range(X.shape[1])) - set(included))
        new_aic = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            subset_X = X[:, included + [new_column]]
            results, _, _ = build_model(subset_X, y, log_y=log_y, X_mask_mm=mask_threshold)
            if results.pvalues[-1] < p_value_threshold:
                new_aic[new_column] = results.aic

        # Add the predictor with the best AIC if it's better than the current model's AIC
        if not new_aic.empty and new_aic.min() < current_aic:
            best_feature = new_aic.idxmin()
            included.append(best_feature)
            current_aic = new_aic.min()
            changed = True

        # Step 2: Check significance of each included predictor and remove non-significant ones
        results, _, _ = build_model(X[:, included], y, log_y=log_y, X_mask_mm=mask_threshold)
        pvalues = pd.Series(results.pvalues, index=included)
        for feature in pvalues[pvalues > p_value_threshold].index:
            included.remove(feature)
            changed = True

        # Exit if no changes were made in this iteration
        if not changed:
            break

    return included
#
#
# def stepwise_selection(X, y, log_y, p_value_threshold=0.05):
#     included = []  # Start with an empty set of features
#     current_r2 = 0  # Start with a baseline of 0 for R^2
#     while True:
#         changed = False
#         excluded = list(set(range(X.shape[1])) - set(included))
#         new_r2 = pd.Series(index=excluded, dtype=float)
#         for new_column in excluded:
#             subset_X = X[:, included + [new_column]]
#             results, _, _ = build_model(subset_X, y, log_y=log_y)
#             new_r2[new_column] = results.rsquared  # Evaluate R^2 with the new feature
#
#         best_r2 = new_r2.max()
#         print(best_r2)
#         if best_r2 - current_r2 > 1e-4:
#             best_feature = new_r2.idxmax()
#             included.append(best_feature)
#             current_r2 = best_r2
#             changed = True
#
#         # Backward elimination: remove features with high p-values
#         results, _, _ = build_model(X[:, included], y, log_y=log_y)
#         pvalues = pd.Series(results.pvalues, index=included)
#         worst_pval = pvalues.max()
#         if worst_pval > p_value_threshold:
#             worst_feature = pvalues.idxmax()
#             included.remove(worst_feature)
#             changed = True
#
#         if not changed:
#             break
#     return included


##############################################################################################
########################## STEP 0: Tidy data ##########################
##############################################################################################
# Drop September 2024 -
weather_data_historical = weather_data_historical.drop(weather_data_historical.index[-1])
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[-1])

# code if years need to be dropped
weather_data_historical = weather_data_historical.iloc[(min_year_for_analyis-absolute_min_year)*12 :]
monthly_reporting_by_facility = monthly_reporting_by_facility.iloc[(min_year_for_analyis-absolute_min_year)*12:]
# Linear regression
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
if np.nanmin(y) < 1:
     y += 1e-6  # Shift to ensure positivity as taking log
y[y > 1e5] = np.nan
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
lag_2_month = weather_data_historical.shift(2).values.flatten()
lag_3_month = weather_data_historical.shift(3).values.flatten()
lag_4_month = weather_data_historical.shift(4).values.flatten()

altitude = np.array(altitude)
altitude = np.where(altitude < 0, np.nan, altitude)
altitude = list(altitude)

#print(np.nanmax(y) ) ## one y value is > 167,000. So should remove

##############################################################################################
########################## STEP 1: GENERATE PREDICTIONS OF ANC DATA ##########################
##############################################################################################

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

if model_fit_ANC_data:
    best_predictors = stepwise_selection(X, y, log_y=log_y)
    X_best = X[:, best_predictors]
    results, y_pred, mask_ANC_data = build_model(X_best, y, log_y=log_y, X_mask_mm=mask_threshold)
    model_data = {
        'model': results,
        'mask': mask_ANC_data,
        'best_predictors': best_predictors  # Save best predictors
    }
    joblib.dump(model_data, model_filename)
else:
    model_data = joblib.load(model_filename)
    results = model_data['model']
    mask_ANC_data = model_data['mask']
    best_predictors = model_data['best_predictors']
    X = X[:, best_predictors]
    y_pred = results.predict(X[mask_ANC_data])

if log_y:
    residuals_percentage = (y[mask_ANC_data] - np.exp(y_pred))
else:
    residuals_percentage = (y[mask_ANC_data] - y_pred)

print("ANC prediction", results.summary())

# plot
year_month_labels = np.array([f"{y}-{m}" for y, m in zip(year_flattened, month_flattened)])
y_filtered = y[mask_ANC_data]
year_month_labels_filtered = year_month_labels[mask_ANC_data]
if log_y:
    data_ANC_predictions = pd.DataFrame({
        'Year_Month': year_month_labels_filtered,
        'y_filtered': y_filtered,
        'y_pred': np.exp(y_pred),
    })
else:
    data_ANC_predictions = pd.DataFrame({
            'Year_Month': year_month_labels_filtered,
            'y_filtered': y_filtered,
            'y_pred': y_pred,
        })

data_ANC_predictions = data_ANC_predictions.sort_values(by='Year_Month').reset_index(drop=True)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Actual data and predictions
axs[0].scatter(data_ANC_predictions['Year_Month'], data_ANC_predictions['y_filtered'], color='#1C6E8C', alpha=0.5, label='Actual data')
axs[0].scatter(data_ANC_predictions['Year_Month'], data_ANC_predictions['y_pred'], color='#9AC4F8', alpha=0.7, label='Predictions')
axs[0].set_xticklabels(data_ANC_predictions['Year_Month'], rotation=45, ha='right')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Number of ANC visits')
axs[0].set_title('A: Monthly ANC Visits vs. Precipitation')
axs[0].legend(loc='upper left')

# Panel B: Residuals (in percentage)
axs[1].scatter(data_ANC_predictions['Year_Month'], residuals_percentage, color='#9AC4F8', alpha=0.7, label='Residuals')
axs[1].set_xticklabels(data_ANC_predictions['Year_Month'], rotation=45, ha='right')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Residuals')
axs[1].set_title('B: Residuals')
axs[1].legend(loc='upper left')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Actual data and predictions
axs[0].scatter(data_ANC_predictions['Year_Month'], np.log(data_ANC_predictions['y_filtered']), color='#1C6E8C', alpha=0.5, label='Actual data')
axs[0].scatter(data_ANC_predictions['Year_Month'], np.log(data_ANC_predictions['y_pred']), color='#9AC4F8', alpha=0.7, label='Predictions')
axs[0].set_xticklabels(data_ANC_predictions['Year_Month'], rotation=45, ha='right')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Log(Number of ANC visits)')
axs[0].set_title('A: Monthly ANC Visits vs. Precipitation')
axs[0].legend(loc='upper left')

# Panel B: Residuals (in percentage)
axs[1].scatter(data_ANC_predictions['Year_Month'], np.log(residuals_percentage), color='#9AC4F8', alpha=0.7, label='Residuals')
axs[1].set_xticklabels(data_ANC_predictions['Year_Month'], rotation=45, ha='right')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Log(Residual ANC cases)')
axs[1].set_title('B: Residuals cases')
axs[1].legend(loc='upper left')
plt.tight_layout()
plt.show()

##############################################################################################
########################## STEP 2 - USE THESE IN PREDICTIONS ##########################
##############################################################################################
X = np.column_stack([
    weather_data[mask_ANC_data],
    np.array(year_flattened)[mask_ANC_data],
    np.array(month_flattened)[mask_ANC_data],
    resid_encoded[mask_ANC_data],
    zone_encoded[mask_ANC_data],
    owner_encoded[mask_ANC_data],
    ftype_encoded[mask_ANC_data],
    lag_1_month[mask_ANC_data],
    lag_2_month[mask_ANC_data],
    lag_3_month[mask_ANC_data],
    lag_4_month[mask_ANC_data],
    facility_encoded[mask_ANC_data],
    np.array(altitude)[mask_ANC_data]
])

log_y = False
if model_fit_weather_data:
    best_predictors = stepwise_selection(X, residuals_percentage, log_y=log_y)
    print(best_predictors)
    X_best = X[:, best_predictors]
    results_of_weather_model, y_pred, mask_all_data = build_model(X_best, residuals_percentage, log_y=log_y, X_mask_mm=mask_threshold)
    model_data = {
        'model': results_of_weather_model,
        'mask': mask_all_data,
        'best_predictors': best_predictors  # Save best predictors
    }
    joblib.dump(model_data, model_filename_weather_model)
else:
    model_data = joblib.load(model_filename_weather_model)
    results_of_weather_model = model_data['model']
    mask_all_data = model_data['mask']
    best_predictors = model_data['best_predictors']
results_of_weather_model, y_pred, mask_all_data = build_model(X_best, residuals_percentage, log_y=log_y,
                                                              X_mask_mm=mask_threshold)
print(results_of_weather_model.summary())

X = np.column_stack([
    weather_data[mask_ANC_data],
    np.array(year_flattened)[mask_ANC_data],
    np.array(month_flattened)[mask_ANC_data],
    resid_encoded[mask_ANC_data],
    zone_encoded[mask_ANC_data],
    owner_encoded[mask_ANC_data],
    ftype_encoded[mask_ANC_data],
    lag_1_month[mask_ANC_data],
    lag_2_month[mask_ANC_data],
    lag_3_month[mask_ANC_data],
    lag_4_month[mask_ANC_data],
    facility_encoded[mask_ANC_data],
    np.array(altitude)[mask_ANC_data],

])

X_weather = np.column_stack([
    weather_data[mask_ANC_data],
    np.array(month_flattened)[mask_ANC_data],
    X[:,best_predictors],
    lag_1_month[mask_ANC_data],
    lag_2_month[mask_ANC_data],
    lag_3_month[mask_ANC_data],
    lag_4_month[mask_ANC_data]
])
results_of_weather_model, y_pred, mask_all_data = build_model(X_weather, residuals_percentage, log_y=log_y, X_mask_mm=mask_threshold)

print(results_of_weather_model.summary())

##### Plot y_predic

X_filtered = X[mask_all_data]

plt.scatter(X_filtered[:, 0], (residuals_percentage[mask_all_data]), color='red', alpha=0.5)
plt.scatter(X_filtered[:, 0],y_pred)
plt.title(' ')
plt.ylabel('Change in ANC visits')
plt.xlabel('Precip (mm)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()



X_weather = np.column_stack([
    weather_data[mask_ANC_data],
    X[:,best_predictors],
    np.array(month_flattened)[mask_ANC_data],
    lag_1_month[mask_ANC_data],
    lag_2_month[mask_ANC_data],
    lag_3_month[mask_ANC_data],
    lag_4_month[mask_ANC_data]
])
results_of_weather_model, y_pred, mask_all_data = build_model(X_weather, residuals_percentage, log_y=log_y, X_mask_mm=mask_threshold)

print(results_of_weather_model.summary())

##### Plot y_predic

X_filtered = X[mask_all_data]

plt.scatter(X_filtered[:, 0], (residuals_percentage[mask_all_data]), color='red', alpha=0.5)
plt.scatter(X_filtered[:, 0],y_pred)
plt.title(' ')
plt.ylabel('Change in ANC visits')
plt.xlabel('Precip (mm)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
