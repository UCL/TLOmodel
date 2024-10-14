import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

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



## Linear regression - flattened
# year as a fixed effect
year = range(2014, 2024, 1) # year as a fixed effect
year_repeated = [y for y in year for _ in range(12)]
year = year_repeated[:-4]
year_flattened = year*len(weather_data_historical.columns) # to get flattened data

# add month as a fixed effect
month = range(12)
month_repeated = [m for m in month for _ in range(2014, 2024, 1)]
month = month_repeated[:-4]
month_flattened = month*len(weather_data_historical.columns)

# facility as fixed effect
facility_flattened = list(range(len(weather_data_historical.columns))) * len(month)
# location as a fixed effect

# linear regression - flatten for more data points
X = weather_data_historical.values.flatten().reshape(-1, 1)
y = monthly_reporting_by_facility.values.flatten()

#X = np.column_stack((X)) #, year_flattened, month_flattened, facility_flattened))
print("X shape:", X.shape)
print("y shape:", y.shape)
# # Perform linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
print(r2 )
print(model.coef_)


# ## Linear regression - by facility
results_list = []

for facility in monthly_reporting_by_facility.columns:
    y = monthly_reporting_by_facility[facility].values
    X = weather_data_historical.values
    X = np.column_stack((X, year, month))

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    coefficients = model.coef_

    results_list.append({'Facility': facility, 'R2': r2, 'Coefficients': coefficients})

results = pd.DataFrame(results_list)
print(results)
