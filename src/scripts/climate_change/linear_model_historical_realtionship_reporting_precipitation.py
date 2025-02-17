import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial, Poisson
from statsmodels.genmod.generalized_linear_model import GLM
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, f_regression
import scipy.stats as stats
ANC = True
Inpatient = False
if ANC:
    service = 'ANC'
if Inpatient:
    service = 'Inpatient'
daily_max = False
daily_total = False
if daily_total:
    five_day = True
    cumulative = True
else:
    five_day = False
    cumulative = False
feature_selection = False
use_all_weather = True
min_year_for_analysis = 2012
absolute_min_year = 2011
mask_threshold = -np.inf # accounts for scaling
#mask_threshold = 50
use_percentile_mask_threshold = True
baseline_years = range(1940, 1979)
min_year_for_analysis_baseline = min(baseline_years)
absolute_min_year_baseline = min_year_for_analysis_baseline - 1
max_year_for_analysis_baseline = max(baseline_years) + 1

poisson = False
log_y = True

covid_months = range((2020 - min_year_for_analysis)* 12 + 4, (2020 - min_year_for_analysis)* 12 + 4 + 20) # Bingling's paper: disruption between April 2020 and Dec 2021, a period of 20 months
cyclone_freddy_months_phalombe = range((2023 - min_year_for_analysis)* 12 + 4, (2020 - min_year_for_analysis)* 12 + 4 + 14) # From news report and DHIS2, see disruption from April 2023 - June 2024, 14 months

cyclone_freddy_months_thumbwe = range((2023 - min_year_for_analysis)* 12 + 3, (2020 - min_year_for_analysis)* 12 + 3 + 12) # From news report and DHIS2, see disruption from March 2023 - March 2024, 12 months

model_filename = (
    f"best_model_{'ANC' if ANC else 'Reporting'}_prediction_"
    f"{'5_day' if five_day else 'monthly'}_"
    f"{'cumulative' if cumulative else ('max' if daily_max else 'total')}_"
    f"{'poisson' if poisson else 'linear'}_precip.pkl"
)
print(model_filename)
model_filename_weather_model = (
    f"best_model_weather_"
    f"{'5_day' if five_day else 'monthly'}_"
    f"{'cumulative' if cumulative else ('max' if daily_max else 'total')}_"
    f"{'poisson' if poisson else 'linear'}_precip.pkl"
)
print(model_filename_weather_model)
# # data is from 2011 - 2024 - for facility
if ANC:
    monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_ANC_by_smaller_facility_lm.csv", index_col=0)
elif Inpatient:
    monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_Inpatient_by_smaller_facility_lm.csv", index_col=0)

def build_model(X, y, poisson=False, log_y=False, feature_selection=False, k_best=None):
    epsilon = 1

    # Log-transform y with clipping for positivity
    if log_y:
        y = np.log(np.clip(y, epsilon, None))

        # Apply mask to filter valid data
    mask = (~np.isnan(X).any(axis=1) & ~np.isnan(y) & (y <= 1e4))
    X_filtered, y_filtered = X[mask], y[mask]

    # Feature selection step (optional)
    if feature_selection:
        if poisson:
            raise ValueError("Feature selection using f_regression is only compatible with OLS regression.")
        selector = SelectKBest(score_func=f_regression, k=k_best or 'all')
        X_filtered = selector.fit_transform(X_filtered, y_filtered)
        selected_features = selector.get_support()
    else:
        selected_features = np.ones(X.shape[1], dtype=bool)  # Keep all features if no selection

    # Build the model
    model = GLM(y_filtered, X_filtered, family=NegativeBinomial(), method='nm') if poisson else sm.OLS(y_filtered,
                                                                                                       X_filtered)
    model_fit = model.fit()

    return model_fit, model_fit.predict(X_filtered), mask, selected_features


def create_binary_feature(threshold, weather_data_df, recent_months):
    binary_feature_list = []
    for facility in weather_data_df.columns:
        facility_data = weather_data_df[facility]
        for i in range(len(facility_data)):
            facility_threshold = threshold[i] if hasattr(threshold, "__len__") else threshold

            if i >= recent_months: # only count for recent months, and have to discount the data kept in for this purpose. Also, first 12 months have no data to check back to
                last_x_values = facility_data[i - recent_months:i]
                binary_feature_list.append(1 if (last_x_values > facility_threshold).any() else 0)

    return binary_feature_list

def stepwise_selection(X, y, log_y, poisson, p_value_threshold=0.05):
    included = []
    current_aic = np.inf

    while True:
        changed = False

        # Step 1: Try adding each excluded predictor and select the best one by AIC if significant
        excluded = list(set(range(X.shape[1])) - set(included))
        new_aic = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            subset_X = X[:, included + [new_column]]
            results, y_pred, mask_ANC_data, _ = build_model(subset_X, y, poisson, log_y=log_y)
            new_aic[new_column] = results.aic

        # Add the predictor with the best AIC if it's better than the current model's AIC
        if not new_aic.empty and new_aic.min() < current_aic:
            best_feature = new_aic.idxmin()
            included.append(best_feature)
            current_aic = new_aic.min()
            changed = True
        print(current_aic)


        # Exit if no changes were made in this iteration
        if not changed:
            break
    included.sort()
    results, y_pred, mask_ANC_data, _ = build_model(X[:, included], y, poisson, log_y=log_y)

    return included, results, y_pred, mask_ANC_data

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def repeat_info(info, num_facilities, year_range, historical):
    # Repeat facilities in alternating order for each month and year
    repeated_info = [info[i % len(info)] for i in range(len(year_range) * 12 * num_facilities)]

    if historical:
        return repeated_info[:-4 * num_facilities]  # Exclude final 4 months for all facilities
    else:
        return repeated_info
#

def process_weather_data(weather_df, zero_sum_columns, min_year_for_analysis, absolute_min_year, lags=[1, 2, 3, 4, 9, 12]):
    """
    Processes weather data by dropping zero-sum columns, computing lags, and flattening.
    """
    weather_df = weather_df.drop(columns=zero_sum_columns, errors='ignore').iloc[:-(2 if 'monthly' in weather_df.columns else 1)]
    lags_data = {lag: weather_df.shift(lag).values[(min_year_for_analysis - absolute_min_year) * 12:].flatten() for lag in lags}
    return weather_df.iloc[(min_year_for_analysis - absolute_min_year) * 12:], lags_data

### Try combine weather variables ##
if use_all_weather:
    if ANC:
        weather_data_monthly_original = pd.read_csv(
                    "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv",
                    index_col=0)

        weather_data_five_day_cumulative_original = pd.read_csv(
                            "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facilities_with_ANC_five_day_cumulative.csv",
                            index_col=0)
    if Inpatient:
        weather_data_monthly_original = pd.read_csv(
            "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_Inpatient_lm.csv",
            index_col=0)

        weather_data_five_day_cumulative_original = pd.read_csv(
            "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facility_five_day_cumulative_inpatient.csv",
            index_col=0)
##############################################################################################
########################## STEP 0: Tidy data ##########################
##############################################################################################
## Remove any columns that sum to 0 in the monthly reporting data (e.g. for inpatient data, may mean they don't have the facility)
zero_sum_columns = monthly_reporting_by_facility.columns[(monthly_reporting_by_facility.sum(axis=0) == 0)]
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(columns=zero_sum_columns)

if use_all_weather:
    weather_data_monthly_df = weather_data_monthly_original.drop(columns=zero_sum_columns, errors='ignore')
    nan_indices = np.isnan(weather_data_monthly_df)

    weather_data_five_day_cumulative_df = weather_data_five_day_cumulative_original.drop(columns=zero_sum_columns, errors='ignore')

    weather_data_monthly_df = weather_data_monthly_df.drop(weather_data_monthly_df.index[-2:])
    weather_data_five_day_cumulative_df = weather_data_five_day_cumulative_df.drop(weather_data_five_day_cumulative_df.index[-1:])
    lag_1_month = weather_data_monthly_df.shift(1).values
    lag_2_month = weather_data_monthly_df.shift(2).values
    lag_3_month = weather_data_monthly_df.shift(3).values
    lag_4_month = weather_data_monthly_df.shift(4).values
    lag_9_month = weather_data_monthly_df.shift(9).values

    lag_1_month = lag_1_month[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
    lag_2_month = lag_2_month[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
    lag_3_month = lag_3_month[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
    lag_4_month = lag_4_month[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
    lag_9_month = lag_9_month[(min_year_for_analysis - absolute_min_year) * 12:].flatten()

    lag_1_5_day = weather_data_five_day_cumulative_df.shift(1).values
    lag_2_5_day = weather_data_five_day_cumulative_df.shift(2).values
    lag_3_5_day = weather_data_five_day_cumulative_df.shift(3).values
    lag_4_5_day = weather_data_five_day_cumulative_df.shift(4).values
    lag_9_5_day = weather_data_five_day_cumulative_df.shift(9).values

    lag_1_5_day = lag_1_5_day[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
    lag_2_5_day = lag_2_5_day[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
    lag_3_5_day = lag_3_5_day[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
    lag_4_5_day = lag_4_5_day[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
    lag_9_5_day = lag_9_5_day[(min_year_for_analysis - absolute_min_year) * 12:].flatten()

    # need for binary
    lag_12_month = weather_data_monthly_df.shift(12).values
    lag_12_month = lag_12_month[(min_year_for_analysis - absolute_min_year) * 12:].flatten()

    # mask covid months - don't need to do on lagged data, because the removal of these entries in the model will remove all rows
    weather_data_monthly = weather_data_monthly_df # need to keep these seperate for the binary values later
    weather_data_five_day_cumulative = weather_data_five_day_cumulative_df

    #weather_data_monthly.loc[covid_months, :] = np.nan
    #weather_data_five_day_cumulative.loc[covid_months, :] = np.nan
    # code if years need to be dropped
    weather_data_monthly = weather_data_monthly.iloc[(min_year_for_analysis - absolute_min_year) * 12:]
    weather_data_five_day_cumulative = weather_data_five_day_cumulative.iloc[(min_year_for_analysis - absolute_min_year) * 12:]
    weather_data_monthly_flattened = weather_data_monthly.values.flatten()
    weather_data_five_day_cumulative_flattened = weather_data_five_day_cumulative.values.flatten()
    print(len(weather_data_five_day_cumulative_flattened))

    weather_data = np.vstack((weather_data_monthly_flattened,weather_data_five_day_cumulative_flattened)).T

# # Mask COVID-19 months for reporting
monthly_reporting_by_facility.iloc[covid_months, :] = np.nan
# Mask for missing data with Cyclone Freddy
monthly_reporting_by_facility.loc[cyclone_freddy_months_phalombe, 'Phalombe Health Centre'] = 0
monthly_reporting_by_facility.loc[cyclone_freddy_months_thumbwe, 'Thumbwe Health Centre'] = 0

# Drop September 2024 in ANC/reporting data
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[-1])
# code if years need to be dropped
monthly_reporting_by_facility = monthly_reporting_by_facility.iloc[(min_year_for_analysis-absolute_min_year)*12:]
# Linear regression
month_range = range(12)
num_facilities = len(monthly_reporting_by_facility.columns)
year_range = range(min_year_for_analysis, 2025, 1) # year as a fixed effect
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
if np.nanmin(y) < 1:
     y += 1  # Shift to ensure positivity as taking log
y[y > 4e3] = np.nan

# One-hot encode facilities
facility_encoded = pd.get_dummies(facility_flattened, drop_first=True)
# above below
weather_data_monthly_subsetted = weather_data_monthly_df.iloc[
                                            (min_year_for_analysis - absolute_min_year - 1) * 12:]
weather_data_monthly_original_flattened = weather_data_monthly_subsetted.values.flatten()
percentile_90 = np.nanpercentile(weather_data_monthly_original_flattened, 90)
above_below_X = lag_12_month > percentile_90
# Prepare additional facility info
if ANC:
    expanded_facility_info = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm_with_ANC.csv", index_col=0)
elif Inpatient:
    expanded_facility_info = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/expanded_facility_info_by_smaller_facility_lm_with_inpatient_days.csv", index_col=0)

expanded_facility_info = expanded_facility_info.drop(columns=zero_sum_columns)

expanded_facility_info = expanded_facility_info.T.reindex(columns=expanded_facility_info.index)

zone_info_each_month = repeat_info(expanded_facility_info["Zonename"], num_facilities, year_range, historical = True)
zone_encoded = pd.get_dummies(zone_info_each_month, drop_first=True)
dist_info_each_month = repeat_info(expanded_facility_info["Dist"], num_facilities, year_range, historical = True)
dist_encoded = pd.get_dummies(dist_info_each_month, drop_first=True)
resid_info_each_month = repeat_info(expanded_facility_info['Resid'], num_facilities, year_range, historical = True)
resid_encoded = pd.get_dummies(resid_info_each_month, drop_first=True)
owner_info_each_month = repeat_info(expanded_facility_info['A105'], num_facilities, year_range, historical = True)
owner_encoded = pd.get_dummies(owner_info_each_month, drop_first=True)
ftype_info_each_month = repeat_info(expanded_facility_info['Ftype'], num_facilities, year_range, historical = True)
ftype_encoded = pd.get_dummies(ftype_info_each_month, drop_first=True)
altitude = [float(x) for x in repeat_info(expanded_facility_info['A109__Altitude'], num_facilities, year_range, historical = True)]
minimum_distance = [float(x) for x in repeat_info(expanded_facility_info['minimum_distance'], num_facilities, year_range, historical = True)]

altitude = np.array(altitude)
altitude = np.where(altitude < 0, np.nan, altitude)
mean_altitude = round(np.nanmean(altitude))
altitude = np.where(np.isnan(altitude), float(mean_altitude), altitude)
altitude = np.nan_to_num(altitude, nan=mean_altitude, posinf=mean_altitude, neginf=mean_altitude)
altitude = list(altitude)

minimum_distance = np.nan_to_num(minimum_distance, nan=np.nan, posinf=np.nan, neginf=np.nan) # just in case

########################## STEP 1: GENERATE PREDICTIONS OF ANC DATA ##########################

##############################################################################################

#    Continuous columns that need to be standardized (weather_data, lag variables, altitude, minimum_distance)
X_continuous = np.column_stack([
    year_flattened,
    month_flattened,
    altitude,
    np.array(minimum_distance)
])

X_categorical = np.column_stack([
    resid_encoded,
    zone_encoded,
    dist_encoded,
    owner_encoded,
    #ftype_encoded,
    #facility_encoded,
])
scaler = StandardScaler()
X_continuous_scaled = scaler.fit_transform(X_continuous)
X_continuous_scaled = X_continuous
X_ANC_standardized = np.column_stack([X_continuous_scaled, X_categorical])
# Create column names
# continuous_columns = ['Year', 'Month', 'Altitude', 'Minimum_Distance']
# categorical_columns = [
#     f'Resid_{i}' for i in range(resid_encoded.shape[1])
# ] + [
#     f'Zone_{i}' for i in range(zone_encoded.shape[1])
# ] + [
#     f'Dist_{i}' for i in range(dist_encoded.shape[1])
# ] + [
#     f'Owner_{i}' for i in range(owner_encoded.shape[1])
# ] + [
#     f'Ftype_{i}' for i in range(ftype_encoded.shape[1])
# ] + [
#     f'Facility_{i}' for i in range(facility_encoded.shape[1])
# ]
#
# # Combine into a DataFrame
# columns = continuous_columns + categorical_columns
# df_combined = pd.DataFrame(X_ANC_standardized, columns=columns)
#
# # Standardize the continuous variables
# df_combined[continuous_columns] = (df_combined[continuous_columns] - df_combined[continuous_columns].mean()) / df_combined[continuous_columns].std()
#
# # Compute the correlation matrix
# correlation_matrix = df_combined.corr()
# correlation_matrix.to_csv('/Users/rem76/Desktop/Climate_change_health/Data/correlation_matrix_of_predictors.csv')

# Display the correlation matrix

#results, y_pred, mask_ANC_data, selected_features = build_model(X_ANC_standardized , y, poisson = poisson, log_y=log_y, X_mask_mm=mask_threshold, feature_selection = feature_selection)

included, results, y_pred, mask_ANC_data = stepwise_selection(X_ANC_standardized , y, poisson = poisson, log_y=log_y,)
coefficients = results.params

coefficient_names = ["year", "month", "altitude", "minimum_distance"] + list(resid_encoded.columns) + list(zone_encoded.columns) + \
                     list(owner_encoded.columns)
coefficient_names = pd.Series(coefficient_names)
coefficient_names = coefficient_names[included]
coefficients_df = pd.DataFrame(coefficients, columns=['coefficients'])
continuous_coefficients = coefficients[:len(X_continuous_scaled[0])]
categorical_coefficients = coefficients[len(X_continuous_scaled[0]):]
means = scaler.mean_
scales = scaler.scale_
rescaled_continuous_coefficients = continuous_coefficients * scales
rescaled_coefficients = np.concatenate([rescaled_continuous_coefficients, categorical_coefficients])
rescaled_coefficients_df = pd.DataFrame(rescaled_coefficients, columns=['rescaled coefficients'])
p_values = results.pvalues
p_values_df = pd.DataFrame(p_values, columns=['p_values'])
results_df = pd.concat([coefficient_names, coefficients_df, p_values_df], axis=1)

results_df.to_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_model_historical_{service}.csv')


y_weather = np.exp(y_pred)

print("ANC prediction", results.summary())

# plot
year_month_labels = np.array([f"{y}-{m}" for y, m in zip(year_flattened, month_flattened)])
y_filtered = y[mask_ANC_data]
year_month_labels_filtered = year_month_labels[mask_ANC_data]
data_ANC_predictions = pd.DataFrame({
        'Year_Month': year_month_labels_filtered,
        'y_filtered': y_filtered,
        'y_pred': np.exp(y_pred),
        'residuals': y_filtered - np.exp(y_pred)
    })


data_ANC_predictions = data_ANC_predictions.sort_values(by='Year_Month').reset_index(drop=True)
x_labels = data_ANC_predictions['Year_Month'][::num_facilities*12]

# Set the xticks at corresponding positions
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
step = num_facilities * 12
data_ANC_predictions_grouped = data_ANC_predictions.groupby('Year_Month').mean().reset_index()

xticks = data_ANC_predictions['Year_Month'][::len(year_range)*num_facilities]
# Panel A: Actual data and predictions
axs[0].scatter(data_ANC_predictions['Year_Month'], data_ANC_predictions['y_filtered'], color='#1C6E8C', alpha=0.5, label='Actual data')
axs[0].scatter(data_ANC_predictions['Year_Month'], data_ANC_predictions['y_pred'], color='#9AC4F8', alpha=0.7, label='Predictions')
axs[0].scatter(data_ANC_predictions_grouped['Year_Month'], data_ANC_predictions_grouped['y_filtered'], color='red', alpha=0.5, label='Mean Actual data')
axs[0].scatter(data_ANC_predictions_grouped['Year_Month'], data_ANC_predictions_grouped['y_pred'], color='yellow', alpha=0.7, label='Mean Predictions')

axs[0].set_xticks(xticks)
axs[0].set_xticklabels(xticks, rotation=45, ha='right')
axs[0].set_xlabel('Year')
axs[0].set_ylabel(f'Number of {service}  visits')
axs[0].set_title(f'A: Monthly {service}  Visits vs. Precipitation')






axs[0].legend(loc='upper left')

# Panel B: Residuals

axs[1].scatter(data_ANC_predictions['Year_Month'], (data_ANC_predictions['y_filtered'] - data_ANC_predictions['y_pred']), color='#9AC4F8', alpha=0.7, label='Residuals')
axs[1].scatter(data_ANC_predictions_grouped['Year_Month'], data_ANC_predictions_grouped['residuals'],
                       color='red', alpha=0.7, label='Mean Residuals')

axs[1].set_xticks(xticks)
axs[1].set_xticklabels(xticks, rotation=45, ha='right')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Residuals')
axs[1].set_title('B: Residuals')
axs[1].legend(loc='upper left')
axs[1].set_ylim(top = 3000)
plt.tight_layout()
#plt.show()


##############################################################################################
########################## STEP 2 - USE THESE IN PREDICTIONS ##########################
##############################################################################################


#    Continuous columns that need to be standardized (weather_data, lag variables, altitude, minimum_distance)
X_continuous = np.column_stack([
        weather_data,
        weather_data[:,0]*weather_data[:,0],
        weather_data[:,1] * weather_data[:,1],
        weather_data[:, 0] * weather_data[:, 0] * weather_data[:, 0],
        weather_data[:, 1] * weather_data[:, 1] * weather_data[:, 1],
        weather_data[:, 1] * weather_data[:,0],
        np.array(year_flattened),
        np.array(month_flattened),
        lag_1_month,
        lag_2_month,
        lag_3_month,
        lag_4_month,
        lag_9_month,
        lag_1_5_day,
        lag_2_5_day,
        lag_3_5_day,
        lag_4_5_day,
        lag_9_5_day,
        np.array(altitude),
        np.array(minimum_distance)]
)

X_categorical = np.column_stack([
        resid_encoded,
        zone_encoded,
        dist_encoded,
        owner_encoded,
        #ftype_encoded,
        #facility_encoded,
        #np.array(above_below_X)[mask_ANC_data],
    ])

scaler = StandardScaler()
X_continuous_scaled = scaler.fit_transform(X_continuous)
X_continuous_scaled = X_continuous

X_weather_standardized = np.column_stack([X_continuous_scaled, X_categorical])
if use_percentile_mask_threshold:
    mask_threshold = np.nanpercentile(X_weather_standardized[:,0], 0)
    print(mask_threshold)
    X_weather_standardized[:, 0] = np.where(
        X_weather_standardized[:, 0] < mask_threshold, np.nan, X_weather_standardized[:, 0]
    )
    print(X_weather_standardized[:, 0] )

# results_of_weather_model, y_pred_weather, mask_all_data, selected_features = build_model(X_weather_standardized, y, poisson = poisson, log_y=log_y,
#                                                                  X_mask_mm=mask_threshold, feature_selection =  feature_selection)
included_weather, results_of_weather_model, y_pred_weather, mask_all_data = stepwise_selection(X_weather_standardized , y, poisson = poisson, log_y=log_y,)

coefficient_names_weather = ["precip_monthly_total", "precip_5_day_max", "precip_monthly_total_2", "precip_5_day_max_2",
                             "precip_monthly_total_3", "precip_5_day_max_3", "5_day_monthly", "year", "month",
                             "lag_1_month", "lag_2_month", "lag_3_month", "lag_4_month", "lag_9_month",
                             "lag_1_5_day", "lag_2_5_day", "lag_3_5_day", "lag_4_5_day", "lag_9_month",
                             "altitude", "minimum_distance"] + \
                            list(resid_encoded.columns) + list(zone_encoded.columns) + \
                            list(owner_encoded.columns)
coefficient_names_weather = pd.Series(coefficient_names_weather)
coefficient_names_weather = coefficient_names_weather[included_weather]
print(coefficient_names_weather)
coefficients_weather = results_of_weather_model.params
coefficients_weather_df = pd.DataFrame(coefficients_weather, columns=['coefficients'])

p_values_weather = results_of_weather_model.pvalues
p_values_weather_df = pd.DataFrame(p_values_weather, columns=['p_values'])
results_weather_df = pd.concat([coefficient_names_weather, coefficients_weather_df, p_values_weather_df, rescaled_coefficients_df], axis=1)
results_weather_df.to_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_weather_model_historical_{service}.csv')

print("All predictors", results_of_weather_model.summary())
#
X_filtered = X_weather_standardized[mask_all_data]

fig, axs = plt.subplots(1, 2, figsize=(10, 6))


indices_ANC_data = np.where(mask_ANC_data)[0]
indices_all_data = np.where(mask_all_data)[0]
common_indices = np.intersect1d(indices_ANC_data, indices_all_data)
matched_y_pred = y_pred[np.isin(indices_ANC_data, common_indices)]
matched_y_pred_weather = y_pred_weather[np.isin(indices_all_data, common_indices)]
monthly_weather_predictions = X_filtered[:, 0][np.isin(indices_all_data, common_indices)]


axs[0].scatter(X_filtered[:, 0], y[mask_all_data], color='red', alpha=0.5, label = 'Non weather model')
axs[0].hlines(y = 0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], color = 'black', linestyle = '--')
axs[0].scatter(X_filtered[:, 0], np.exp(y_pred_weather), label='Weather model', color="blue", alpha = 0.5)
axs[0].hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], color='black', linestyle='--')
axs[0].set_ylabel(f'{service}  visits')



axs[1].scatter(monthly_weather_predictions, np.exp(matched_y_pred_weather) - np.exp(matched_y_pred), color='red', alpha=0.5, label = 'Residuals')
axs[1].hlines(y = 0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], color = 'black', linestyle = '--')
axs[1].set_ylabel('Difference between weather and non-weather model')

axs[0].set_xlabel('Monthly precipitation (mm)')
axs[1].set_xlabel('Monthly precipitation (mm)')

axs[0].legend(loc='upper left', borderaxespad=0.)


plt.show()
## average of predictions
data_weather_predictions = pd.DataFrame({
    'Year': np.array(year_flattened)[mask_all_data],
    'Month': np.array(month_flattened)[mask_all_data],
    'Year_Month': year_month_labels_filtered,  # Ensure this is properly formatted
    'y_pred_weather': np.exp(matched_y_pred_weather),
    'y_pred_no_weather': np.exp(matched_y_pred),
    'difference': np.exp(matched_y_pred) - np.exp(matched_y_pred_weather)
})

data_weather_predictions_grouped = data_weather_predictions.groupby('Year_Month', as_index=False).sum()

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(data_weather_predictions_grouped['Year_Month'],
           data_weather_predictions_grouped['difference'],
           color='#823038', alpha=0.7)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

y_max = max(abs(data_weather_predictions_grouped['difference'])) + 50
ax.set_ylim(-y_max, y_max)
# Separate positive and negative values
positive_mask = data_weather_predictions_grouped['difference'] >= 0
negative_mask = ~positive_mask

ax.stem(data_weather_predictions_grouped['Year_Month'][positive_mask],
        data_weather_predictions_grouped['difference'][positive_mask],
        linefmt='#1C6E8C', markerfmt='o', basefmt="black")
ax.stem(data_weather_predictions_grouped['Year_Month'][negative_mask],
        data_weather_predictions_grouped['difference'][negative_mask],
        linefmt='#823038', markerfmt='o', basefmt="black")
ax.set_xlabel('Year-Month')
ax.set_ylabel(f'Difference in Predicted {service} Services (Without vs. With Precipitation)')
january_ticks = data_weather_predictions_grouped[data_weather_predictions_grouped['Year_Month'].str.endswith('-1')]
print(january_ticks)
ax.set_xticks(january_ticks['Year_Month'])
ax.set_xticklabels(january_ticks['Year_Month'].str[:4], rotation=45, ha='right')
ax.axvline(x='2023-3', color='#CDC6AE', linestyle='--', linewidth=1,  alpha=0.3, label="Cyclone Freddy")
ax.axvline(x='2023-2', color='#CDC6AE', linestyle='--', linewidth=1, alpha=0.3)
ax.axvspan('2023-2', '2023-3', color='#CDC6AE', alpha=0.3)
plt.tight_layout()
plt.savefig( f'/Users/rem76/Desktop/Climate_change_health/Results/{service}_disruptions/{service}_disruptions_difference_historical_models.png')
plt.show()

## save historical predictions
full_data_weather_predictions_historical = pd.DataFrame({
    'Year': np.array(year_flattened)[mask_all_data],
    'Month': np.array(month_flattened)[mask_all_data],
    'Facility_ID': np.array(facility_flattened)[mask_all_data],
    'Altitude': np.array(altitude)[mask_all_data],
    'Zone': np.array(zone_info_each_month)[mask_all_data],
    'District': np.array(dist_info_each_month)[mask_all_data],
    'Resid': np.array(resid_info_each_month)[mask_all_data],
    'Owner': np.array(owner_info_each_month)[mask_all_data],
    'Facility_Type': np.array(ftype_info_each_month)[mask_all_data],
    'Precipitation': X_weather_standardized[mask_all_data,0],
    'Lag_1_Precipitation': np.array(lag_1_month)[mask_all_data],
    'Lag_2_Precipitation': np.array(lag_2_month)[mask_all_data],
    'Lag_3_Precipitation': np.array(lag_3_month)[mask_all_data],
    'Lag_4_Precipitation': np.array(lag_4_month)[mask_all_data],
    'Predicted_Weather_Model': np.exp(matched_y_pred_weather),
    'Predicted_No_Weather_Model': np.exp(matched_y_pred),
    'Difference_in_Expectation': np.exp(matched_y_pred_weather) - np.exp(matched_y_pred),
})
full_data_weather_predictions_historical.to_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_model_historical_predictions_{service}.csv')

############## LR #########################################

# Extract log-likelihood values
log_likelihood_null = results.llf  # Null model
log_likelihood_full = results_of_weather_model.llf  # Full model

LR_stat = -2 * (log_likelihood_null - log_likelihood_full)
df = len(results_of_weather_model.params) - len(results.params)
p_value = 1 - stats.chi2.cdf(LR_stat, df)

# Print results
print(f"Likelihood Ratio Test Statistic: {LR_stat:.4f}")
print(f"Degrees of Freedom: {df}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("The full model is significantly better than the null model (p < 0.05).")
else:
    print("No significant improvement by adding weather variables.")

############### ADD IN CMIP DATA ###########################
def get_weather_data(ssp_scenario, model_type):
    weather_data_prediction_five_day_cumulative_original = pd.read_csv(
        f"{data_path}Precipitation_data/Downscaled_CMIP6_data_CIL/{ssp_scenario}/{model_type}_window_prediction_weather_by_facility_{service}.csv",
        dtype={'column_name': 'float64'}
    )
    weather_data_prediction_five_day_cumulative_original = weather_data_prediction_five_day_cumulative_original.drop(
        weather_data_prediction_five_day_cumulative_original.columns[0], axis=1
    ) # first column are date/months
    weather_data_prediction_monthly_original = pd.read_csv(
        f"{data_path}Precipitation_data/Downscaled_CMIP6_data_CIL/{ssp_scenario}/{model_type}_monthly_prediction_weather_by_facility_{service}.csv",
        dtype={'column_name': 'float64'}
    )
    weather_data_prediction_monthly_original = weather_data_prediction_monthly_original.drop(
        weather_data_prediction_monthly_original.columns[0], axis=1
    ) # first column are date/months
    weather_data_prediction_monthly_df = weather_data_prediction_monthly_original.drop(columns=zero_sum_columns)
    weather_data_prediction_five_day_cumulative_df = weather_data_prediction_five_day_cumulative_original.drop(
        columns=zero_sum_columns)

    return weather_data_prediction_five_day_cumulative_df, weather_data_prediction_monthly_df
model_types = ['lowest', 'mean', 'highest']
# Configuration and constants
min_year_for_analysis = 2025
absolute_min_year = 2024
max_year_for_analysis = 2071
data_path = "/Users/rem76/Desktop/Climate_change_health/Data/"

# Define SSP scenario
ssp_scenarios = ["ssp126","ssp245", "ssp585"]

# Load and preprocess weather data
for ssp_scenario in ssp_scenarios:
    for model_type in model_types:
        print(ssp_scenario, model_type)
        weather_data_prediction_five_day_cumulative_df, weather_data_prediction_monthly_df = get_weather_data(ssp_scenario,
                                                                                                              model_type)
        lag_1_month_prediction = weather_data_prediction_monthly_df.shift(1).values
        lag_2_month_prediction = weather_data_prediction_monthly_df.shift(2).values
        lag_3_month_prediction = weather_data_prediction_monthly_df.shift(3).values
        lag_4_month_prediction = weather_data_prediction_monthly_df.shift(4).values
        lag_9_month_prediction = weather_data_prediction_monthly_df.shift(9).values

        lag_1_month_prediction = lag_1_month_prediction[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
        lag_2_month_prediction = lag_2_month_prediction[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
        lag_3_month_prediction = lag_3_month_prediction[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
        lag_4_month_prediction = lag_4_month_prediction[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
        lag_9_month_prediction = lag_9_month_prediction[(min_year_for_analysis - absolute_min_year) * 12:].flatten()

        lag_1_5_day_prediction = weather_data_prediction_five_day_cumulative_df.shift(1).values
        lag_2_5_day_prediction = weather_data_prediction_five_day_cumulative_df.shift(2).values
        lag_3_5_day_prediction = weather_data_prediction_five_day_cumulative_df.shift(3).values
        lag_4_5_day_prediction = weather_data_prediction_five_day_cumulative_df.shift(4).values
        lag_9_5_day_prediction = weather_data_prediction_five_day_cumulative_df.shift(9).values

        lag_1_5_day_prediction = lag_1_5_day_prediction[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
        lag_2_5_day_prediction = lag_2_5_day_prediction[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
        lag_3_5_day_prediction = lag_3_5_day_prediction[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
        lag_4_5_day_prediction = lag_4_5_day_prediction[(min_year_for_analysis - absolute_min_year) * 12:].flatten()
        lag_9_5_day_prediction = lag_9_5_day_prediction[(min_year_for_analysis - absolute_min_year) * 12:].flatten()

        weather_data_prediction_five_day_cumulative = weather_data_prediction_five_day_cumulative_df # keep these seperate for binary features

                # need for binary comparison
        lag_12_month = weather_data_prediction_monthly_df.shift(12).values
        lag_12_month = lag_12_month[(min_year_for_analysis - absolute_min_year) * 12:].flatten()

        weather_data_prediction_monthly = weather_data_prediction_monthly_df # keep these seperate for binary features

        weather_data_prediction_five_day_cumulative = weather_data_prediction_five_day_cumulative.iloc[(min_year_for_analysis - absolute_min_year) * 12:]
        weather_data_prediction_monthly = weather_data_prediction_monthly.iloc[(min_year_for_analysis - absolute_min_year) * 12:]
        weather_data_prediction_monthly_flattened = weather_data_prediction_monthly.values.flatten()
        weather_data_prediction_five_day_cumulative_flattened = weather_data_prediction_five_day_cumulative.values.flatten()
        weather_data_prediction_flatten = np.vstack((weather_data_prediction_monthly_flattened, weather_data_prediction_five_day_cumulative_flattened)).T
        num_facilities = len(weather_data_prediction_monthly.columns)

        missing_facility = [col for col in expanded_facility_info.index if col not in weather_data_prediction_monthly.columns]
        expanded_facility_info = expanded_facility_info.drop(missing_facility)
        year_range_prediction = range(min_year_for_analysis, max_year_for_analysis)
        month_repeated_prediction = [m for _ in year_range_prediction for m in range(1, 13)]
        year_flattened_prediction = np.repeat(year_range_prediction, 12 * num_facilities)
        month_flattened_prediction = month_repeated_prediction * num_facilities
        facility_flattened_prediction = np.tile(range(num_facilities), len(year_flattened_prediction) // num_facilities)
        # Encode facilities and create above/below average weather data
        facility_encoded_prediction = pd.get_dummies(facility_flattened_prediction, drop_first=True)

        # Load and preprocess facility information
        zone_info_prediction = repeat_info(expanded_facility_info["Zonename"], num_facilities, year_range_prediction, historical = False)
        zone_encoded_prediction = pd.get_dummies(zone_info_prediction, drop_first=True)
        dist_info_prediction = repeat_info(expanded_facility_info["Dist"], num_facilities, year_range_prediction, historical=False)
        dist_encoded_prediction = pd.get_dummies(dist_info_prediction, drop_first=True)
        resid_info_prediction = repeat_info(expanded_facility_info['Resid'], num_facilities, year_range_prediction, historical = False)
        resid_encoded_prediction = pd.get_dummies(resid_info_prediction, drop_first=True)
        owner_info_prediction = repeat_info(expanded_facility_info['A105'], num_facilities, year_range_prediction, historical = False)
        owner_encoded_prediction = pd.get_dummies(owner_info_prediction, drop_first=True)
        ftype_info_prediction = repeat_info(expanded_facility_info['Ftype'], num_facilities, year_range_prediction, historical = False)
        ftype_encoded_prediction = pd.get_dummies(ftype_info_prediction, drop_first=True)
        altitude_prediction = [float(x) for x in repeat_info(expanded_facility_info['A109__Altitude'],num_facilities, year_range_prediction, historical = False)]
        minimum_distance_prediction = [float(x) for x in repeat_info(expanded_facility_info['minimum_distance'],num_facilities, year_range_prediction, historical = False)]
        # minimum_distance_prediction = np.nan_to_num(minimum_distance_prediction, nan=np.nan, posinf=np.nan, neginf=np.nan) # just in case

        altitude_prediction = np.array(altitude_prediction)
        altitude_prediction = np.where(altitude_prediction < 0, np.nan, altitude_prediction)
        mean_altitude_prediction = round(np.nanmean(altitude_prediction))
        altitude_prediction = np.where(np.isnan(altitude_prediction), float(mean_altitude), altitude_prediction)
        altitude_prediction = np.nan_to_num(altitude_prediction, nan=mean_altitude_prediction, posinf=mean_altitude_prediction, neginf=mean_altitude_prediction)
        altitude_prediction = list(altitude_prediction)

        minimum_distance_prediction = np.nan_to_num(minimum_distance_prediction, nan=np.nan, posinf=np.nan, neginf=np.nan) # just in case
        # Weather data

        X_continuous_weather = np.column_stack([
            weather_data_prediction_flatten,
            weather_data_prediction_flatten[:,0]*weather_data_prediction_flatten[:,0],
            weather_data_prediction_flatten[:,1] * weather_data_prediction_flatten[:,1],
            weather_data_prediction_flatten[:, 0] * weather_data_prediction_flatten[:, 0] * weather_data_prediction_flatten[:, 0],
            weather_data_prediction_flatten[:, 1] * weather_data_prediction_flatten[:, 1] * weather_data_prediction_flatten[:, 1],
            weather_data_prediction_flatten[:, 1] * weather_data_prediction_flatten[:,0],
            np.array(year_flattened_prediction),
            np.array(month_flattened_prediction),
            lag_1_month_prediction,
            lag_2_month_prediction,
            lag_3_month_prediction,
            lag_4_month_prediction,
            lag_9_month_prediction,
            lag_1_5_day_prediction,
            lag_2_5_day_prediction,
            lag_3_5_day_prediction,
            lag_4_5_day_prediction,
            lag_9_5_day_prediction,
            altitude_prediction,
            minimum_distance_prediction
        ])

        X_categorical_weather = np.column_stack([
            resid_encoded_prediction,
            zone_encoded_prediction,
            #dist_encoded_prediction,
            owner_encoded_prediction,
            #ftype_encoded_prediction,
            #facility_encoded_prediction
        ])

        scaler_weather = StandardScaler()
        X_continuous_weather_scaled = scaler_weather.fit_transform(X_continuous_weather)
        X_continuous_weather_scaled = X_continuous_weather
        X_basis_weather = np.column_stack([X_continuous_weather_scaled, X_categorical_weather])

        X_basis_weather_filtered = X_basis_weather[X_basis_weather[:, 0] > mask_threshold]

        X_basis_weather_filtered = X_basis_weather_filtered[:,included_weather] # account for model selection in previous steps
        # format output
        year_month_labels = np.array([f"{y}-{m}" for y, m in zip(X_basis_weather_filtered[:, 2], X_basis_weather[:, 3])])
        predictions_weather = results_of_weather_model.predict(X_basis_weather_filtered)

        data_weather_predictions = pd.DataFrame({
                'Year_Month': year_month_labels,
                'y_pred_weather': np.exp(predictions_weather)
            })


        X_continuous_ANC = np.column_stack([
                np.array(year_flattened_prediction),
                np.array(month_flattened_prediction),
                altitude_prediction,
                minimum_distance_prediction
            ])

        X_categorical_ANC = np.column_stack([
                resid_encoded_prediction,
                zone_encoded_prediction,
                #dist_encoded_prediction,
                owner_encoded_prediction,
                #ftype_encoded_prediction,
                #facility_encoded_prediction
            ])

        scaler_ANC = StandardScaler()
        X_continuous_ANC_scaled = scaler_ANC.fit_transform(X_continuous_ANC)
        X_continuous_ANC_scaled = X_continuous_ANC

        X_bases_ANC_standardized = np.column_stack([X_continuous_ANC_scaled, X_categorical_ANC])
        X_bases_ANC_standardized = X_bases_ANC_standardized[:,included] # account for model selection in previous steps
        y_pred_ANC = results.predict(X_bases_ANC_standardized)
        predictions = np.exp(predictions_weather) - np.exp(y_pred_ANC[X_basis_weather[:, 0] > mask_threshold])
        data_weather_predictions['y_pred_no_weather'] = np.exp(y_pred_ANC[X_basis_weather[:, 0] > mask_threshold])

        data_weather_predictions['difference_in_expectation'] = predictions
        data_weather_predictions['weather'] = X_basis_weather[X_basis_weather[:, 0] > mask_threshold, 0]
        data_weather_predictions_grouped = data_weather_predictions.groupby('Year_Month').mean().reset_index()

        # Plotting results
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        #axs[0].scatter(data_weather_predictions['Year_Month'], data_weather_predictions['difference_in_expectation'], color='#9AC4F8', alpha=0.1, label ='Predictions from weather model')
        axs[0].scatter(data_weather_predictions_grouped['Year_Month'], data_weather_predictions_grouped['difference_in_expectation'], color='red', alpha=0.7, label='Mean of predictions')
        axs[0].set_xlabel('Year/Month')
        xticks = data_weather_predictions['Year_Month'][::len(year_range) * 12 * num_facilities]
        axs[0].set_xticks(xticks)
        axs[0].set_xticklabels(xticks, rotation=45, ha='right')
        axs[0].set_ylabel(f'Difference Predicted {service} visits due to rainfall')
        axs[0].legend(loc='upper left')
        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].scatter(data_weather_predictions['weather'],data_weather_predictions['difference_in_expectation'], color='#9AC4F8', alpha=0.1,
                           label='Predictions')

        axs[0].set_xlabel('Precipitation (mm)')
        axs[0].set_ylabel(f'Difference in of {service} visits between weather and non-weather model')

        plt.tight_layout()
        plt.show()
        # Format output: Add all relevant X variables
        full_data_weather_predictions = pd.DataFrame({
            'Year': year_flattened_prediction[X_basis_weather[:, 0] > mask_threshold],
            'Month': np.array(month_flattened_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'Facility_ID': facility_flattened_prediction[X_basis_weather[:, 0] > mask_threshold],
            'Altitude': np.array(altitude_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'Zone': np.array(zone_info_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'District':np.array(dist_info_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'Resid': np.array(resid_info_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'Owner': np.array(owner_info_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'Facility_Type': np.array(ftype_info_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'Precipitation': X_basis_weather[X_basis_weather[:, 0] > mask_threshold, 0],
            'Lag_1_Precipitation': np.array(lag_1_month_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'Lag_2_Precipitation': np.array(lag_2_month_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'Lag_3_Precipitation': np.array(lag_3_month_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'Lag_4_Precipitation': np.array(lag_4_month_prediction)[X_basis_weather[:, 0] > mask_threshold],
            'Predicted_Weather_Model': np.exp(predictions_weather),
            'Predicted_No_Weather_Model': np.exp(y_pred_ANC[X_basis_weather[:, 0] > mask_threshold]),
            'Difference_in_Expectation': predictions,
        })

        #Save the results
        full_data_weather_predictions.to_csv(f"{data_path}weather_predictions_with_X_{ssp_scenario}_{model_type}_{service}.csv", index=False)

        X_basis_weather_filtered = pd.DataFrame(X_basis_weather_filtered)

        # Save to CSV
        X_basis_weather_filtered.to_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/X_basis_weather_filtered_predictions_{ssp_scenario}_{model_type}_{service}.csv', index=False)



################# Semi-post industrial, pre-2000 data #################
if ANC:
    # Load data
    weather_data_monthly_20th_century = pd.read_csv(
        "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm_baseline.csv",
        index_col=0
    )

    weather_data_five_day_cumulative_20th_century = pd.read_csv(
        f"/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_{min(baseline_years)}_{max(baseline_years)}_daily_total_by_facilities_with_ANC_five_day_cumulative.csv",
        index_col=0
    )

    # Drop zero-sum columns
    weather_data_monthly_20th_century = weather_data_monthly_20th_century.drop(columns=zero_sum_columns,
                                                                               errors='ignore')
    weather_data_five_day_cumulative_20th_century = weather_data_five_day_cumulative_20th_century.drop(
        columns=zero_sum_columns, errors='ignore')

    # lags
    lags_monthly_baseline = {i: weather_data_monthly_20th_century.shift(i).values for i in [1, 2, 3, 4, 9, 12]}
    for key in lags_monthly_baseline:
        lags_monthly_baseline[key] = lags_monthly_baseline[key][(min_year_for_analysis_baseline - absolute_min_year_baseline) * 12:].flatten()

    lags_five_day_baseline = {i: weather_data_five_day_cumulative_20th_century.shift(i).values for i in [1, 2, 3, 4, 9]}
    for key in lags_five_day_baseline:
        lags_five_day_baseline[key] = lags_five_day_baseline[key][(min_year_for_analysis_baseline - absolute_min_year_baseline) * 12:].flatten()
    # process
    weather_data_monthly_20th_century = weather_data_monthly_20th_century.iloc[
                                        (min_year_for_analysis_baseline - absolute_min_year_baseline) * 12:]
    weather_data_five_day_cumulative_20th_century = weather_data_five_day_cumulative_20th_century.iloc[
                                                    (min_year_for_analysis_baseline - absolute_min_year_baseline) * 12:]

    weather_data_monthly_flattened = weather_data_monthly_20th_century.values.flatten()
    weather_data_five_day_cumulative_flattened = weather_data_five_day_cumulative_20th_century.values.flatten()
    print(len(weather_data_five_day_cumulative_flattened))
    weather_data_baseline = np.vstack(
        (weather_data_monthly_flattened, weather_data_five_day_cumulative_flattened)).T

# covariates
year_range_baseline = range(min_year_for_analysis_baseline, max_year_for_analysis_baseline, 1)
year_repeated_baseline = [y for y in year_range_baseline for _ in range(12)]
year_flattened_baseline = year_repeated_baseline*len(monthly_reporting_by_facility.columns) # to get flattened data
month = range(12)
month_repeated = []
for _ in year_range:
    month_repeated.extend(range(1, 13))
month_flattened_baseline = month*len(monthly_reporting_by_facility.columns)

zone_info_each_month_baseline = repeat_info(expanded_facility_info["Zonename"], num_facilities, year_range_baseline, historical = False)
zone_encoded_baseline = pd.get_dummies(zone_info_each_month_baseline, drop_first=True)
dist_info_each_month_baseline = repeat_info(expanded_facility_info["Dist"], num_facilities, year_range_baseline, historical = False)
dist_encoded_baseline = pd.get_dummies(dist_info_each_month_baseline, drop_first=True)
resid_info_each_month_baseline = repeat_info(expanded_facility_info['Resid'], num_facilities, year_range_baseline, historical = False)
resid_encoded_baseline = pd.get_dummies(resid_info_each_month_baseline, drop_first=True)
owner_info_each_month_baseline = repeat_info(expanded_facility_info['A105'], num_facilities, year_range_baseline, historical = False)
owner_encoded_baseline = pd.get_dummies(owner_info_each_month_baseline, drop_first=True)
ftype_info_each_month_baseline = repeat_info(expanded_facility_info['Ftype'], num_facilities, year_range_baseline, historical = False)
ftype_encoded_baseline = pd.get_dummies(ftype_info_each_month_baseline, drop_first=True)
altitude_baseline = [float(x) for x in repeat_info(expanded_facility_info['A109__Altitude'], num_facilities, year_range_baseline, historical = False)]
minimum_distance_baseline = [float(x) for x in repeat_info(expanded_facility_info['minimum_distance'], num_facilities, year_range_baseline, historical = False)]

altitude_baseline = np.array(altitude_baseline)
altitude_baseline = np.where(altitude_baseline < 0, np.nan, altitude_baseline)
mean_altitude = round(np.nanmean(altitude_baseline))
altitude_baseline = np.where(np.isnan(altitude_baseline), float(mean_altitude), altitude_baseline)
altitude_baseline = np.nan_to_num(altitude_baseline, nan=mean_altitude, posinf=mean_altitude, neginf=mean_altitude)
altitude_baseline = list(altitude_baseline)

minimum_distance_baseline = np.nan_to_num(minimum_distance_baseline, nan=np.nan, posinf=np.nan, neginf=np.nan) # just in case


X_continuous_baseline = np.column_stack([
    year_flattened_baseline,
    month_flattened_baseline,
    altitude_baseline,
    np.array(minimum_distance_baseline)
])

X_categorical_baseline = np.column_stack([
    resid_encoded_baseline,
    zone_encoded_baseline,
    dist_encoded_baseline,
    owner_encoded_baseline,
    #ftype_encoded,
    #facility_encoded,
])
scaler = StandardScaler()
X_continuous_scaled_baseline = scaler.fit_transform(X_continuous_baseline)
X_continuous_scaled_baseline = X_continuous_baseline
X_ANC_standardized_baseline = np.column_stack([X_continuous_scaled_baseline, X_categorical_baseline])
X_basis_weather_filtered_baseline = X_ANC_standardized_baseline[X_ANC_standardized_baseline[:, 0] > mask_threshold]

X_continuous_baseline = np.column_stack([
        weather_data_baseline,
        weather_data_baseline[:,0]*weather_data_baseline[:,0],
        weather_data_baseline[:,1] * weather_data_baseline[:,1],
        weather_data_baseline[:, 0] * weather_data_baseline[:, 0] * weather_data_baseline[:, 0],
        weather_data_baseline[:, 1] * weather_data_baseline[:, 1] * weather_data_baseline[:, 1],
        weather_data_baseline[:, 1] * weather_data_baseline[:,0],
        np.array(year_flattened_baseline),
        np.array(month_flattened_baseline),
        lags_monthly_baseline,
        lags_monthly_baseline,
        np.array(altitude_baseline),
        np.array(minimum_distance_baseline)]
)

X_categorical_baseline = np.column_stack([
        resid_encoded_baseline,
        zone_encoded_baseline,
        dist_encoded_baseline,
        owner_encoded_baseline,
    ])

scaler = StandardScaler()
X_continuous_scaled_baseline = scaler.fit_transform(X_continuous_baseline)
X_continuous_scaled_baseline = X_continuous_baseline
X_bases_ANC_standardized_baseline = np.column_stack([X_continuous_scaled_baseline, X_categorical_baseline])
X_bases_ANC_standardized_baseline = X_bases_ANC_standardized_baseline[:, included]
X_basis_weather_filtered_baseline = X_basis_weather_filtered_baseline[:,included_weather]

predictions_weather_baseline = results_of_weather_model.predict(X_basis_weather_filtered_baseline )
y_pred_ANC_baseline = results.predict(X_bases_ANC_standardized_baseline)

predictions_baseline = np.exp(predictions_weather_baseline) - np.exp(y_pred_ANC_baseline[X_basis_weather[:, 0] > mask_threshold])
year_month_labels_baseline = np.array([f"{y}-{m}" for y, m in zip(X_basis_weather_filtered_baseline[:, 2], X_basis_weather_filtered_baseline[:, 3])])

data_weather_predictions_baseline = pd.DataFrame({
    'Year_Month': year_month_labels_baseline,
    'y_pred_weather': np.exp(predictions_weather_baseline)
})

data_weather_predictions_baseline['y_pred_no_weather'] = np.exp(y_pred_ANC_baseline[X_basis_weather[:, 0] > mask_threshold])

data_weather_predictions_baseline['difference_in_expectation'] = predictions_baseline
data_weather_predictions_baseline['weather'] = X_basis_weather[X_basis_weather[:, 0] > mask_threshold, 0]
data_weather_predictions_grouped_baseline = data_weather_predictions.groupby('Year_Month').mean().reset_index()

# Plotting results
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
# axs[0].scatter(data_weather_predictions['Year_Month'], data_weather_predictions['difference_in_expectation'], color='#9AC4F8', alpha=0.1, label ='Predictions from weather model')
axs[0].scatter(data_weather_predictions_grouped_baseline['Year_Month'],
               data_weather_predictions_grouped_baseline['difference_in_expectation'], color='red', alpha=0.7,
               label='Mean of predictions')
axs[0].set_xlabel('Year/Month')
xticks = data_weather_predictions_baseline['Year_Month'][::len(year_range) * 12 * num_facilities]
axs[0].set_xticks(xticks)
axs[0].set_xticklabels(xticks, rotation=45, ha='right')
axs[0].set_ylabel(f'Difference Predicted {service} visits due to rainfall')
axs[0].legend(loc='upper left')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].scatter(data_weather_predictions_baseline['weather'], data_weather_predictions_baseline['difference_in_expectation'],
               color='#9AC4F8', alpha=0.1,
               label='Predictions')

axs[0].set_xlabel('Precipitation (mm)')
axs[0].set_ylabel(f'Difference in of {service} visits between weather and non-weather model')

plt.tight_layout()
plt.show()
# Format output: Add all relevant X variables
full_data_weather_predictions = pd.DataFrame({
    'Year': year_flattened_baseline[X_basis_weather[:, 0] > mask_threshold],
    'Month': np.array(month_flattened_baseline)[X_basis_weather[:, 0] > mask_threshold],
    'Facility_ID': facility_flattened_prediction[X_basis_weather[:, 0] > mask_threshold],
    'Altitude': np.array(altitude_prediction)[X_basis_weather[:, 0] > mask_threshold],
    'Zone': np.array(zone_info_prediction)[X_basis_weather[:, 0] > mask_threshold],
    'District': np.array(dist_info_prediction)[X_basis_weather[:, 0] > mask_threshold],
    'Resid': np.array(resid_info_prediction)[X_basis_weather[:, 0] > mask_threshold],
    'Owner': np.array(owner_info_prediction)[X_basis_weather[:, 0] > mask_threshold],
    'Facility_Type': np.array(ftype_info_prediction)[X_basis_weather[:, 0] > mask_threshold],
    'Precipitation': X_basis_weather[X_basis_weather[:, 0] > mask_threshold, 0],
    'Lag_1_Precipitation': np.array(lag_1_month_prediction)[X_basis_weather[:, 0] > mask_threshold],
    'Lag_2_Precipitation': np.array(lag_2_month_prediction)[X_basis_weather[:, 0] > mask_threshold],
    'Lag_3_Precipitation': np.array(lag_3_month_prediction)[X_basis_weather[:, 0] > mask_threshold],
    'Lag_4_Precipitation': np.array(lag_4_month_prediction)[X_basis_weather[:, 0] > mask_threshold],
    'Predicted_Weather_Model': np.exp(predictions_weather),
    'Predicted_No_Weather_Model': np.exp(y_pred_ANC[X_basis_weather[:, 0] > mask_threshold]),
    'Difference_in_Expectation': predictions,
})

# Save the results
full_data_weather_predictions.to_csv(f"{data_path}weather_predictions_with_X_baseline_{service}.csv",
                                     index=False)

X_basis_weather_filtered = pd.DataFrame(X_basis_weather_filtered)

# Save to CSV
full_data_weather_predictions.to_csv(f"{data_path}weather_predictions_with_X_baseline_{service}.csv", index=False)
