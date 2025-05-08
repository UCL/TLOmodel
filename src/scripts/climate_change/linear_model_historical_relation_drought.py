import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.families import NegativeBinomial, Poisson
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
from functions_for_data_cleaning_lm import build_model, stepwise_selection, repeat_info

ANC = True
Inpatient = False
if ANC:
    service = 'ANC'
if Inpatient:
    service = 'Inpatient'

feature_selection = True
min_year_for_analysis = 2012
absolute_min_year = 2011
mask_threshold = -np.inf # accounts for scaling
#mask_threshold = 50
use_percentile_mask_threshold = False
year_range = range(min_year_for_analysis, 2025, 1) # year as a fixed effect

poisson = False
log_y = True

covid_months = range((2020 - min_year_for_analysis)* 12 + 4, (2020 - min_year_for_analysis)* 12 + 4 + 20) # Bingling's paper: disruption between April 2020 and Dec 2021, a period of 20 months
cyclone_freddy_months_phalombe = range((2023 - min_year_for_analysis)* 12 + 4, (2020 - min_year_for_analysis)* 12 + 4 + 14) # From news report and DHIS2, see disruption from April 2023 - June 2024, 14 months

cyclone_freddy_months_thumbwe = range((2023 - min_year_for_analysis)* 12 + 3, (2020 - min_year_for_analysis)* 12 + 3 + 12) # From news report and DHIS2, see disruption from March 2023 - March 2024, 12 months

# data is from 2011 - 2024 - for facility
if ANC:
    monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_ANC_by_smaller_facility_lm.csv", index_col=0)
elif Inpatient:
    monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_Inpatient_by_smaller_facility_lm.csv", index_col=0)


### Read in three-month drought SPI ##
weather_data_monthly_original = pd.read_csv(f"/Users/rem76/Desktop/Climate_change_health/Data/Drought_data/historical_drought_data_2010_2024.csv", index_col=0)
weather_data_monthly_original = (weather_data_monthly_original < -1).astype(int)

##############################################################################################
########################## STEP 0: Tidy data ##########################
##############################################################################################
## Remove any columns that sum to 0 in the monthly reporting data (e.g. for inpatient data, may mean they don't have the facility)
zero_sum_columns = monthly_reporting_by_facility.columns[(monthly_reporting_by_facility.sum(axis=0) == 0)]
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(columns=zero_sum_columns)

weather_data_monthly_df = weather_data_monthly_original.drop(columns=zero_sum_columns, errors='ignore')

nan_indices = np.isnan(weather_data_monthly_df)
weather_data_monthly_df = weather_data_monthly_df.drop(weather_data_monthly_df.index[-2:])
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

    # need for binary
lag_12_month = weather_data_monthly_df.shift(12).values
lag_12_month = lag_12_month[(min_year_for_analysis - absolute_min_year) * 12:].flatten()

    # mask covid months - don't need to do on lagged data, because the removal of these entries in the model will remove all rows
weather_data_monthly = weather_data_monthly_df # need to keep these seperate for the binary values later

    #weather_data_monthly.loc[covid_months, :] = np.nan
    #weather_data_five_day_cumulative.loc[covid_months, :] = np.nan
    # code if years need to be dropped
weather_data_monthly = weather_data_monthly.iloc[(min_year_for_analysis - absolute_min_year) * 12:]
weather_data_monthly_flattened = weather_data_monthly.values.flatten()

weather_data = (weather_data_monthly_flattened).T
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
year_repeated = [y for y in year_range for _ in range(12)]
year = year_repeated[:-4]
month = range(1, 13)
year_flattened = year*len(monthly_reporting_by_facility.columns) # to get flattened data
month_repeated = [m for m in range(1,9) for _ in range(len(monthly_reporting_by_facility.columns))]
month_repeated = month_repeated*len(year_range)
month_repeated_abbreviated = [m for m in range(9,13) for _ in range(len(monthly_reporting_by_facility.columns))]
month_repeated_abbreviated = month_repeated_abbreviated*(len(year_range) - 1)
month_repeated.extend(month_repeated_abbreviated)
month_flattened = month_repeated


facility_flattened = list(monthly_reporting_by_facility.columns) * len(month_repeated)
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
    #dist_encoded,
    owner_encoded,
    #ftype_encoded,
    #facility_encoded,
])
scaler = StandardScaler()
X_continuous_scaled = scaler.fit_transform(X_continuous)
#X_continuous_scaled = X_continuous
X_ANC_standardized = np.column_stack([X_continuous_scaled, X_categorical])
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

#results_df.to_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_model_historical_{service}.csv')


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
axs[0].set_title(f'A: Monthly {service}  Visits vs. SPI')



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
        np.array(year_flattened),
        np.array(month_flattened),
        lag_1_month,
        lag_2_month,
        lag_3_month,
        lag_4_month,
        lag_9_month,
        np.array(altitude),
        np.array(minimum_distance)
    ]
)
print(weather_data)
X_categorical = np.column_stack([
        resid_encoded,
        zone_encoded,
        #dist_encoded,
        owner_encoded,
        #ftype_encoded,
        #facility_encoded,
        #np.array(above_below_X)[mask_ANC_data],
     ])

scaler = StandardScaler()
X_continuous_scaled = scaler.fit_transform(X_continuous)
#X_continuous_scaled = X_continuous

X_weather_standardized = np.column_stack([X_continuous_scaled, X_categorical])
if use_percentile_mask_threshold:
    mask_threshold = np.nanpercentile(X_weather_standardized[:,0], 0)
    print(mask_threshold)
    X_weather_standardized[:, 0] = np.where(
        X_weather_standardized[:, 0] < mask_threshold, np.nan, X_weather_standardized[:, 0]
    )

# results_of_weather_model, y_pred_weather, mask_all_data, selected_features = build_model(X_weather_standardized, y, poisson = poisson, log_y=log_y,
#                                                                  X_mask_mm=mask_threshold, feature_selection =  feature_selection)
included_weather, results_of_weather_model, y_pred_weather, mask_all_data = stepwise_selection(X_weather_standardized , y, poisson = poisson, log_y=log_y,)

coefficient_names_weather = ["SPI", "year", "month",
                             "lag_1_month", "lag_2_month", "lag_3_month", "lag_4_month", "lag_9_month",
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
#results_weather_df.to_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_weather_model_historical_{service}.csv')

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

axs[0].set_xlabel('Monthly SPI')
axs[1].set_xlabel('Monthly SPI')

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
        linefmt='#1C6E8C', markerfmt='o', basefmt="black", label="More appointments projected due to lower SPI")
ax.stem(data_weather_predictions_grouped['Year_Month'][negative_mask],
        data_weather_predictions_grouped['difference'][negative_mask],
        linefmt='#823038', markerfmt='o', basefmt="black", label="Fewer appointments projected due to lower SPI")

ax.set_xlabel('Year-Month')
ax.set_ylabel(f'Difference in Predicted {service} Services (Without vs. With SPI)')
january_ticks = data_weather_predictions_grouped[data_weather_predictions_grouped['Year_Month'].str.endswith('-1')]
ax.set_xticks(january_ticks['Year_Month'])
ax.set_xticklabels(january_ticks['Year_Month'].str[:4], rotation=45, ha='right')
ax.axvline(x='2023-3', color='#CDC6AE', linestyle='--', linewidth=1,  alpha=0.3, label="Cyclone Freddy")
ax.axvline(x='2023-2', color='#CDC6AE', linestyle='--', linewidth=1, alpha=0.3)
ax.axvspan('2023-2', '2023-3', color='#CDC6AE', alpha=0.3)
ax.legend(loc='upper left')

plt.tight_layout()
#plt.savefig( f'/Users/rem76/Desktop/Climate_change_health/Results/{service}_disruptions/{service}_disruptions_difference_historical_models.png')
#plt.show()

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
    'SPI': X_weather_standardized[mask_all_data,0],
    'Lag_1_SPI': np.array(lag_1_month)[mask_all_data],
    'Lag_2_SPI': np.array(lag_2_month)[mask_all_data],
    'Lag_3_SPI': np.array(lag_3_month)[mask_all_data],
    'Lag_4_SPI': np.array(lag_4_month)[mask_all_data],
    'Predicted_Weather_Model': np.exp(matched_y_pred_weather),
    'Predicted_No_Weather_Model': np.exp(matched_y_pred),
    'Difference_in_Expectation': np.exp(matched_y_pred_weather) - np.exp(matched_y_pred),
})
#full_data_weather_predictions_historical.to_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_model_historical_predictions_{service}.csv')

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
