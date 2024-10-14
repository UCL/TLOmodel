from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# data is from 2011 - 2024
monthly_reporting_by_facility = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_by_facility_lm.csv", index_col=0)
weather_data_historical = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_facility_lm.csv", index_col=0)
#
# Plot each facility's reporting data against weather data
plt.figure(figsize=(12, 6))

for facility in weather_data_historical.columns:
    plt.plot(weather_data_historical.index, monthly_reporting_by_facility[facility], label=facility)
months = weather_data_historical.index
year_labels = range(2011, 2025, 1)
year_ticks = range(0, len(months), 12)
plt.xticks(year_ticks, year_labels, rotation=90)
plt.xlabel('Weather Data')
plt.ylabel('Reporting')
plt.title('Reporting vs. Weather Data by Facility')
plt.legend(title='Facilities', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
#plt.show()

## Drop Mental Hospital - bad reporting generally
weather_data_historical = weather_data_historical.drop("Zomba Mental Hospital", axis=1)
monthly_reporting_by_facility = monthly_reporting_by_facility.drop("Zomba Mental Hospital", axis=1)
## Drop September 2024
weather_data_historical = weather_data_historical.drop(weather_data_historical.index[-1])
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[-1])
## Drop before 2014?  12*4
weather_data_historical = weather_data_historical.drop(weather_data_historical.index[0:48]).reset_index(drop=True)
monthly_reporting_by_facility = monthly_reporting_by_facility.drop(monthly_reporting_by_facility.index[0:48]).reset_index(drop=True)

# Plot each facility's reporting data against weather data
plt.figure(figsize=(12, 6))

for facility in weather_data_historical.columns:
    plt.plot(weather_data_historical.index, monthly_reporting_by_facility[facility], label=facility)
months = weather_data_historical.index
year_labels = range(2015, 2025, 1)
year_ticks = range(0, len(months), 12)
plt.xticks(year_ticks, year_labels, rotation=90)
plt.xlabel('Weather Data')
plt.ylabel('Reporting')
plt.title('Reporting vs. Weather Data by Facility')
plt.legend(title='Facilities', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
#plt.show()


## Linear regression
print(monthly_reporting_by_facility)
year = range(2014, 2024, 1) # year as a fixed effect
year_repeated = [y for y in year for _ in range(12)]
year = year_repeated[:-4]
X = weather_data_historical.values
y = monthly_reporting_by_facility.values
if X.ndim == 1:
    X = X.reshape(-1, 1)
if y.ndim == 1:
    y = y.reshape(-1, 1)
print(len(year))
print(len(X))
X = np.column_stack((X, year))

# Perform linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Evaluate the model
r2 = r2_score(y, y_pred)
print(f'R-squared: {r2:.2f}')
print(model.coef_)
print(model.intercept_)
