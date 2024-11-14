import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

# Configuration and constants
ANC = True
min_year_for_analysis = 2015
absolute_min_year = 2015
max_year_for_analysis = 2099
five_day, cumulative, model_fit_ANC_data, model_fit_weather_data = True, True, True, True
data_path = "/Users/rem76/Desktop/Climate_change_health/Data/"

# Load and preprocess weather data
weather_data_prediction = pd.read_csv(f"{data_path}Precipitation_data/ssp2_4_5/prediction_weather_by_smaller_facilities_with_ANC_lm.csv")
weather_data_prediction = weather_data_prediction.iloc[(min_year_for_analysis - absolute_min_year) * 12:]

# Flatten data and prepare for regression
num_facilities = len(weather_data_prediction.columns)
year_range = range(min_year_for_analysis, max_year_for_analysis)
month_repeated = [m for _ in year_range for m in range(1, 13)][:-4]
year_flattened = np.repeat(year_range, 12 * num_facilities)[:-4 * num_facilities]
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

def repeat_info(info, num_facilities, year_range):
    return [i for i in info for _ in range(12 * len(year_range))][:-4 * num_facilities]

zone_info = repeat_info(expanded_facility_info["Zonename"], num_facilities, year_range)
zone_encoded = pd.get_dummies(zone_info, drop_first=True)
resid_info = repeat_info(expanded_facility_info['Resid'], num_facilities, year_range)
resid_encoded = pd.get_dummies(resid_info, drop_first=True)
owner_info = repeat_info(expanded_facility_info['A105'], num_facilities, year_range)
owner_encoded = pd.get_dummies(owner_info, drop_first=True)
ftype_info = repeat_info(expanded_facility_info['Ftype'], num_facilities, year_range)
ftype_encoded = pd.get_dummies(ftype_info, drop_first=True)
altitude = np.where(np.array(repeat_info(expanded_facility_info['A109__Altitude'], num_facilities, year_range)) < 0, np.nan, altitude).tolist()

# Lagged weather data
lags = [weather_data_prediction.shift(i).values.flatten() for i in range(1, 5)]

# Load and prepare model
model_data = joblib.load('best_model_weather_5_day_cumulative_precip.pkl')
results_of_weather_model = model_data['model']
mask_all_data = model_data['mask']

# Assemble predictors
X = np.column_stack([
    weather_data[mask_all_data],
    year_flattened[mask_all_data],
    month_flattened[mask_all_data],
    resid_encoded[mask_all_data],
    zone_encoded[mask_all_data],
    owner_encoded[mask_all_data],
    ftype_encoded[mask_all_data],
    *[lag[mask_all_data] for lag in lags],
    facility_encoded[mask_all_data],
    np.array(altitude)[mask_all_data]
])

# Predictions and formatting output
predictions = results_of_weather_model.predict(X)
year_month_labels = np.array([f"{y}-{m}" for y, m in zip(year_flattened, month_flattened)])
data_ANC_predictions = pd.DataFrame({
    'Year_Month': year_month_labels[mask_all_data],
    'y_pred': np.exp(predictions)
})

# Plotting results
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(data_ANC_predictions['Year_Month'], data_ANC_predictions['y_pred'], color='#9AC4F8', alpha=0.7, label='Predictions')
axs[0].set_xticklabels(data_ANC_predictions['Year_Month'], rotation=45, ha='right')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Number of ANC visits')
axs[0].set_title('Change in Monthly ANC Visits vs. Precipitation')
axs[0].legend(loc='upper left')
plt.tight_layout()
plt.show()
