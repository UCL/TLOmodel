import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

monthly_reporting_by_facility = pd.read_csv_csv("/Users/rem76/Desktop/Climate_change_health/Data/monthly_reporting_by_facility_lm.csv")
weather_data_historical = pd.read_csv_csv("/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_facility_lm.csv")
plt.figure(figsize=(12, 6))

# Plot each facility's reporting data against weather data
for facility in weather_data_historical.columns:
    plt.plot(weather_data_historical.index, monthly_reporting_by_facility[facility], label=facility)

plt.xlabel('Weather Data')
plt.ylabel('Reporting')
plt.title('Reporting vs. Weather Data by Facility')
plt.legend(title='Facilities', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

X = weather_data_historical.values
y = monthly_reporting_by_facility.values
if X.ndim == 1:
    X = X.reshape(-1, 1)
if y.ndim == 1:
    y = y.reshape(-1, 1)


# Perform linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print(X)
print(y)
# Evaluate the model
r2 = r2_score(y, y_pred)
print(f'R-squared: {r2:.2f}')
print(model.coef_)
print(model.intercept_)
print(model.fit)
