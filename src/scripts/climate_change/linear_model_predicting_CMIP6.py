import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

# Configuration and constants
ANC = True
min_year_for_analysis = 2015
absolute_min_year = 2015
max_year_for_analysis = 2099
five_day, cumulative, model_fit_ANC_data, model_fit_weather_data = False, False, True, True
data_path = "/Users/rem76/Desktop/Climate_change_health/Data/"

# Load and preprocess weather data
weather_data_prediction = pd.read_csv(f"{data_path}Precipitation_data/ssp2_4_5/prediction_weather_by_smaller_facilities_with_ANC_lm.csv", index_col=0, dtype={'column_name': 'float64'})
weather_data_prediction = pd.read_csv(f"{data_path}Precipitation_data/ssp2_4_5/prediction_weather_monthly_by_smaller_facilities_with_ANC_lm.csv", index_col=0, dtype={'column_name': 'float64'})
print(weather_data_prediction)
#weather_data_prediction = weather_data_prediction.iloc[(min_year_for_analysis - absolute_min_year) * 12:]
# Flatten data and prepare for regression
num_facilities = len(weather_data_prediction.columns)
year_range = range(min_year_for_analysis, max_year_for_analysis + 1)
month_repeated = [m for _ in year_range for m in range(1, 13)]
year_flattened = np.repeat(year_range, 12 * num_facilities)
month_flattened = month_repeated * num_facilities
facility_flattened = np.tile(range(num_facilities), len(year_flattened) // num_facilities)

# Encode facilities and create above/below average weather data
weather_data = weather_data_prediction.values.flatten()
facility_encoded = pd.get_dummies(facility_flattened, drop_first=True)

grouped_data = pd.DataFrame({
    'facility': facility_flattened,
    'month': month_flattened,
    'weather_data': weather_data
}).groupby(['facility', 'month'])['weather_data'].mean().reset_index()

# Load and preprocess facility information
info_file = "expanded_facility_info_by_smaller_facility_lm_with_ANC.csv" if ANC else "expanded_facility_info_by_smaller_facility_lm.csv"
expanded_facility_info = pd.read_csv(f"{data_path}{info_file}", index_col=0).T

def repeat_info(info, year_range):
    repeated_info = [i for i in info for _ in range(12) for _ in year_range]
    return repeated_info

zone_info = repeat_info(expanded_facility_info["Zonename"], year_range)
zone_encoded = pd.get_dummies(zone_info, drop_first=True)
resid_info = repeat_info(expanded_facility_info['Resid'], year_range)
resid_encoded = pd.get_dummies(resid_info, drop_first=True)
owner_info = repeat_info(expanded_facility_info['A105'], year_range)
owner_encoded = pd.get_dummies(owner_info, drop_first=True)
ftype_info = repeat_info(expanded_facility_info['Ftype'], year_range)
ftype_encoded = pd.get_dummies(ftype_info, drop_first=True)
#altitude = np.where(np.array(repeat_info(expanded_facility_info['A109__Altitude'], num_facilities, year_range)) < 0, np.nan, altitude).tolist()

# Lagged weather data
lag_1_month = weather_data_prediction.shift(1).values.flatten()
lag_2_month = weather_data_prediction.shift(2).values.flatten()
lag_3_month = weather_data_prediction.shift(3).values.flatten()
lag_4_month = weather_data_prediction.shift(4).values.flatten()
# Load

# Load and prepare model
model_data_ANC =joblib.load('/Users/rem76/PycharmProjects/TLOmodel/best_model_ANC_prediction_5_day_cumulative_linear_precip.pkl') # don't need mask
best_params_ANC_pred = model_data_ANC['best_predictors']

# Assemble predictors

X_bases = X = np.column_stack([
    year_flattened,
    month_flattened,
    resid_encoded,
    zone_encoded,
    owner_encoded,
    ftype_encoded,
    facility_encoded,
])
X_from_best_models = X_bases[:,best_params_ANC_pred]

model_data = joblib.load('/Users/rem76/PycharmProjects/TLOmodel/best_model_weather_5_day_cumulative_linear_precip.pkl') # don't need mask
results_of_weather_model = model_data['model']
best_params_weather_pred = model_data['best_predictors']
print(best_params_weather_pred)
X_basis_weather = np.column_stack([
    weather_data,
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
    facility_encoded])

X = X_basis_weather[:,best_params_weather_pred]
# Predictions and formatting output
predictions = results_of_weather_model.predict(X)
year_month_labels = np.array([f"{y}-{m}" for y, m in zip(year_flattened, month_flattened)])
data_ANC_predictions = pd.DataFrame({
    'Year_Month': year_month_labels,
    'y_pred': predictions
})

# Plotting results
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(data_ANC_predictions['Year_Month'], data_ANC_predictions['y_pred'], color='#9AC4F8', alpha=0.7, label='Predictions')
axs[0].set_xticklabels(data_ANC_predictions['Year_Month'], rotation=45, ha='right')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Change in ANC visits due to 5-day monthly maximum precipitation')
axs[0].set_title('Change in Monthly ANC Visits vs. Precipitation')
axs[0].legend(loc='upper left')

axs[1].scatter(X[:,0], data_ANC_predictions['y_pred'], color='#9AC4F8', alpha=0.7, label='Predictions')
plt.tight_layout()
plt.show()

