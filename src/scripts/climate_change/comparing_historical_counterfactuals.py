import matplotlib.pyplot as plt
import pandas as pd

# Load data
service = 'ANC'
historical_predictions = pd.read_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_model_historical_predictions_{service}.csv')
baseline_predictions = pd.read_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/weather_predictions_with_X_baseline_{service}.csv')

# Plot the 'Precipitation' column from both datasets
plt.figure(figsize=(10, 6))
plt.plot(historical_predictions['Precipitation'], label='Historical Predictions', color='#1C6E8C')
plt.plot(baseline_predictions['Precipitation'], label='Baseline Predictions', color='#9AC4F8')
plt.title('Precipitation: Historical vs Baseline Predictions')
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(True)
plt.show()


weather_data_monthly_original = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv",
    index_col=0
)
weather_data_monthly_20th_century = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm_baseline.csv",
    index_col=0
)

# Plot the 'Precipitation' column from both datasets
plt.figure(figsize=(10, 6))
print(weather_data_monthly_original)
plt.plot(weather_data_monthly_original, label='Original Monthly Weather', color='#1C6E8C')
plt.plot(weather_data_monthly_20th_century, label='20th Century Monthly Weather', color='#9AC4F8')
plt.title('Precipitation: Original vs 20th Century Monthly Weather')
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(True)
plt.show()

weather_data_monthly_original = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv",
    index_col=0
)
weather_data_monthly_20th_century = pd.read_csv(
    "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm_baseline.csv",
    index_col=0
)

# Calculate the difference between the two DataFrames
difference = weather_data_monthly_original - weather_data_monthly_20th_century

# Calculate the average difference for each column
average_difference = difference.mean()

# Display the result
print(average_difference)



