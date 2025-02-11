from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon, shapiro
from scipy.stats import ttest_1samp

import geopandas as gpd
from netCDF4 import Dataset
from shapely.geometry import Polygon
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import os

from tlo.analysis.utils import (
    extract_results,
    make_age_grp_lookup,
    make_calendar_period_lookup,
    summarize,
)

min_year = 2025
max_year = 2061
scenarios = ['ssp126', 'ssp245', 'ssp585']
model_types = ['lowest', 'mean', 'highest']
year_range = range(min_year, max_year)
# global min for all heatmaps for same scale
global_min = -5
global_max = 0
# service
Inpatient = True

if Inpatient:
    service = 'Inpatient'
results_folder_to_save = Path(f'/Users/rem76/Desktop/Climate_change_health/Results/{service}_disruptions')
resourcefilepath = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-25T110820Z")
historical_predictions = pd.read_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_model_historical_predictions_{service}.csv')
precipitation_threshold = historical_predictions['Precipitation'].quantile(0.9)
print(precipitation_threshold)

# Load map of Malawi for later
file_path_historical_data = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/2011/60ab007aa16d679a32f9c3e186d2f744.nc"
dataset = Dataset(file_path_historical_data, mode='r')
pr_data = dataset.variables['tp'][:]
lat_data = dataset.variables['latitude'][:]
long_data = dataset.variables['longitude'][:]
meshgrid_from_netCDF = np.meshgrid(long_data, lat_data)

malawi = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm0_nso_20181016.shp")
malawi_admin2 = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
#
# change names of some districts for consistency
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Blantyre City', 'Blantyre')
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Mzuzu City', 'Mzuzu')
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Lilongwe City', 'Lilongwe')
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Zomba City', 'Zomba')

difference_lat = lat_data[1] - lat_data[0]
difference_long = long_data[1] - long_data[0]
#
# # Get expected disturbance from the model
#
results_list = []

#Loop through scenarios and model types
for scenario in scenarios:
    for model_type in model_types:
        predictions_from_cmip = pd.read_csv(
            f'/Users/rem76/Desktop/Climate_change_health/Data/weather_predictions_with_X_{scenario}_{model_type}.csv'
        )
        predictions_from_cmip =  predictions_from_cmip.loc[predictions_from_cmip['Difference_in_Expectation'] < 0]
        predictions_from_cmip = predictions_from_cmip[predictions_from_cmip['Year'] <= 2061]
        # total disruptions
        predictions_from_cmip_sum = predictions_from_cmip.groupby('Year').sum().reset_index()
        predictions_from_cmip_sum['Percentage_Difference'] = (
                predictions_from_cmip_sum['Difference_in_Expectation'] / predictions_from_cmip_sum[
                'Predicted_No_Weather_Model'])
        percentage_difference = predictions_from_cmip_sum[
            'Percentage_Difference'].values

        # Check for negative values (missed cases?)
        negative_sum = np.sum(percentage_difference[percentage_difference < 0])

        # now do extreme precipitation by district and year, use original dataframe to get monthly top 10% precip
        filtered_predictions = predictions_from_cmip[predictions_from_cmip['Precipitation'] >= precipitation_threshold]
        filtered_predictions_sum = filtered_predictions.groupby('Year').sum().reset_index()
        percent_due_to_extreme = filtered_predictions_sum['Difference_in_Expectation'] / predictions_from_cmip_sum['Predicted_No_Weather_Model']
        values_extreme_precip =  percent_due_to_extreme.values
        negative_sum_extreme_precip = np.sum(values_extreme_precip[values_extreme_precip < 0])
        result_df = pd.DataFrame({
            "Scenario": [scenario],
            "Model_Type": [model_type],
            "Negative_Sum": [negative_sum],
            "Negative_Percentage": [negative_sum ],
            "Extreme_Precip": [negative_sum_extreme_precip],
            "Extreme_Precip_Percentage": [(negative_sum_extreme_precip  / negative_sum) * 100]
        })

        results_list.append(result_df)

        # Save multiplied values by model and scenario
        # multiplied_values_df = pd.DataFrame({
        #     'Year': year_range,
        #     'Scenario': scenario,
        #     'Model_Type': model_type,
        #     'Multiplied_Values': multiplied_values,
        #     'Multiplied_Values_extreme_precip': multiplied_values_extreme_precip
        #
        # })
        # multiplied_values_df.to_csv(results_folder_to_save/f'multiplied_values_{scenario}_{model_type}.csv', index=False)

final_results = pd.concat(results_list, ignore_index=True)
print(final_results)
final_results.to_csv(f'/Users/rem76/Desktop/Climate_change_health/Results/{service}_disruptions/negative_sums_and_percentages.csv', index=False)




## now all grids
fig, axes = plt.subplots(3, 3, figsize=(18, 18),)


for scenario in scenarios:
    for model_type in model_types:
        predictions_from_cmip = pd.read_csv(
            f'/Users/rem76/Desktop/Climate_change_health/Data/weather_predictions_with_X_{scenario}_{model_type}.csv'
        )
        predictions_from_cmip =  predictions_from_cmip.loc[predictions_from_cmip['Difference_in_Expectation'] < 0]
        predictions_from_cmip_sum = predictions_from_cmip_sum[predictions_from_cmip_sum['Year'] <= 2061]

        predictions_from_cmip_sum = predictions_from_cmip.groupby('District').sum().reset_index()
        predictions_from_cmip_sum['Percentage_Difference'] = (
            predictions_from_cmip_sum['Difference_in_Expectation'] / predictions_from_cmip_sum['Predicted_No_Weather_Model']
        ) * 100

        predictions_from_cmip_sum['District'] = predictions_from_cmip_sum['District'].replace(
            {"Mzimba North": "Mzimba", "Mzimba South": "Mzimba"}
        )
        percentage_diff_by_district = predictions_from_cmip_sum.groupby('District')['Percentage_Difference'].mean()
        malawi_admin2['Percentage_Difference'] = malawi_admin2['ADM2_EN'].map(percentage_diff_by_district)
        malawi_admin2.loc[malawi_admin2['Percentage_Difference'] > 0, 'Percentage_Difference'] = 0


for i, scenario in enumerate(scenarios):
    for j, model_type in enumerate(model_types):
        predictions_from_cmip = pd.read_csv(
            f'/Users/rem76/Desktop/Climate_change_health/Data/weather_predictions_with_X_{scenario}_{model_type}.csv'
        )
        predictions_from_cmip =  predictions_from_cmip.loc[predictions_from_cmip['Difference_in_Expectation'] < 0]
        predictions_from_cmip_sum = predictions_from_cmip_sum[predictions_from_cmip_sum['Year'] <= 2061]

        predictions_from_cmip_sum = predictions_from_cmip.groupby('District').sum().reset_index()
        predictions_from_cmip_sum['Percentage_Difference'] = (
            predictions_from_cmip_sum['Difference_in_Expectation'] / predictions_from_cmip_sum['Predicted_No_Weather_Model']
        ) * 100

        predictions_from_cmip_sum['District'] = predictions_from_cmip_sum['District'].replace(
            {"Mzimba North": "Mzimba", "Mzimba South": "Mzimba"}
        )
        percentage_diff_by_district = predictions_from_cmip_sum.groupby('District')['Percentage_Difference'].mean()
        malawi_admin2['Percentage_Difference'] = malawi_admin2['ADM2_EN'].map(percentage_diff_by_district)
        malawi_admin2.loc[malawi_admin2['Percentage_Difference'] > 0, 'Percentage_Difference'] = 0
        ax = axes[i, j]
        malawi_admin2.dropna(subset=['Percentage_Difference']).plot(
            ax=ax,
            column='Percentage_Difference',
            cmap='Blues_r',
            edgecolor='black',
            alpha=1,
            legend=False,
            vmin=global_min,
            vmax=global_max
        )

        ax.set_title(f"{scenario}: {model_type}", fontsize=14)

        if i != 1:
            ax.set_xlabel("")
        if j != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Latitude", fontsize=10)

        if i == 1:
            ax.set_xlabel("Longitude", fontsize=10)

sm = plt.cm.ScalarMappable(
    cmap='Blues_r',
    norm=mcolors.Normalize(vmin=global_min, vmax=global_max)
)
sm.set_array([])
fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.8, label="Percentage Difference (%)")
plt.suptitle("Percentage Difference Maps by Scenario and Model Type", fontsize=16, y=1.02)
plt.savefig(results_folder_to_save / 'percentage_difference_maps_grid.png')
plt.show()



significant_results_year = []
#
# Assuming 'district' is a column in your data
for scenario in scenarios:
    for model_type in model_types:
        predictions_from_cmip = pd.read_csv(
            f'/Users/rem76/Desktop/Climate_change_health/Data/weather_predictions_with_X_{scenario}_{model_type}.csv'
        )
        predictions_from_cmip =  predictions_from_cmip.loc[predictions_from_cmip['Difference_in_Expectation'] < 0]
        predictions_from_cmip_sum = predictions_from_cmip_sum[predictions_from_cmip_sum['Year'] <= 2061]

        predictions_from_cmip_sum = predictions_from_cmip.groupby(['District', 'Year']).sum().reset_index()
        for district in predictions_from_cmip_sum['District'].unique():
                district_values = predictions_from_cmip_sum[predictions_from_cmip_sum['District'] == district]
                no_weather_model = district_values['Predicted_No_Weather_Model'].values
                weather_model = district_values['Predicted_Weather_Model'].values

                # Calculate the difference
                difference = no_weather_model - weather_model

                # Perform a one-sample t-test assuming 0 as the null hypothesis mean
                t_stat, p_value = ttest_1samp(difference, popmean=0)
                # Print results if p-value is below 0.05 (statistically significant)
                if p_value < 0.05:
                    print(f"Scenario: {scenario}, Model Type: {model_type}, District: {district}, "
                          f"t-stat: {t_stat:.2f}, p-value: {p_value:.4f}")
## now all grids



####### Historical disruptions ##########


historical_predictions = pd.read_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_model_historical_predictions_{service}.csv')
historical_predictions = historical_predictions.loc[historical_predictions['Difference_in_Expectation'] < 0]

historical_predictions_sum = historical_predictions.groupby('District').sum().reset_index()
historical_predictions_sum['Percentage_Difference'] = (
    historical_predictions_sum['Difference_in_Expectation'] / historical_predictions_sum['Predicted_No_Weather_Model']
) * 100

historical_predictions_sum['District'] = historical_predictions_sum['District'].replace(
    {"Mzimba North": "Mzimba", "Mzimba South": "Mzimba"}
)

percentage_diff_by_district_historical = historical_predictions_sum.groupby('District')['Percentage_Difference'].mean()
malawi_admin2['Percentage_Difference_historical'] = malawi_admin2['ADM2_EN'].map(percentage_diff_by_district_historical)
malawi_admin2.loc[malawi_admin2['Percentage_Difference_historical'] > 0, 'Percentage_Difference_historical'] = 0
percentage_diff_by_district_historical_average = historical_predictions_sum['Percentage_Difference'].mean()
print(malawi_admin2)
filtered_predictions = historical_predictions[historical_predictions['Precipitation'] >= precipitation_threshold]
filtered_predictions_sum = filtered_predictions.groupby('Year').sum().reset_index()
percent_due_to_extreme = filtered_predictions_sum['Difference_in_Expectation'].sum()
percent_due_to_extreme = percent_due_to_extreme/historical_predictions['Difference_in_Expectation'].sum()
print(percent_due_to_extreme)
fig, ax = plt.subplots(figsize=(10, 10))

malawi_admin2.dropna(subset=['Percentage_Difference_historical']).plot(
    ax=ax,
    column='Percentage_Difference_historical',
    cmap='Blues_r',
    edgecolor='black',
    alpha=1,
    legend=False,
    vmin=global_min,
    vmax=0
)

ax.set_ylabel("Latitude", fontsize=10)
ax.set_xlabel("Longitude", fontsize=10)

sm = plt.cm.ScalarMappable(
    cmap='Blues_r',
    norm=mcolors.Normalize(vmin=global_min, vmax=global_max)
)
sm.set_array([])
fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, label="Percentage Difference (%)")

plt.title("", fontsize=16)
plt.savefig(results_folder_to_save / 'percentage_difference_map_historical.png')
plt.show()

## stacked bar chart

# Create the figure and axes grid
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
y_min = float('inf')
y_max = float('-inf')
x_min = float('inf')
x_max = float('-inf')
percentage_diff_by_year_district_all = {}
percentage_diff_by_year_district_scenario = {}
year_groupings = range(2025, 2060, 5)
for i, scenario in enumerate(scenarios):
    percentage_diff_by_year_district_all[scenario] = {}
    percentage_diff_by_year_district_scenario[scenario] = {}

    for j, model_type in enumerate(model_types):
        percentage_diff_by_year_district_all[scenario][model_type] = {}
        percentage_diff_by_year_district_scenario[scenario][model_type] = {}
        percentage_diff_by_year_district_scenario[scenario][model_type] = 0

        percentage_diff_by_year_district = {}

        predictions_from_cmip = pd.read_csv(
            f'/Users/rem76/Desktop/Climate_change_health/Data/weather_predictions_with_X_{scenario}_{model_type}.csv'
        )
        predictions_from_cmip =  predictions_from_cmip.loc[predictions_from_cmip['Difference_in_Expectation'] < 0]
        predictions_from_cmip_sum = predictions_from_cmip_sum[predictions_from_cmip_sum['Year'] <= 2061]

        predictions_from_cmip_sum = predictions_from_cmip.groupby(['Year', 'District']).sum().reset_index()
        predictions_from_cmip_sum = predictions_from_cmip_sum[predictions_from_cmip_sum['Year'] <= 2060]
        predictions_from_cmip_sum['Percentage_Difference'] = (
                                                                 predictions_from_cmip_sum[
                                                                     'Difference_in_Expectation'] /
                                                                 predictions_from_cmip_sum['Predicted_No_Weather_Model']
                                                             )
        predictions_from_cmip_sum.loc[
            predictions_from_cmip_sum['Percentage_Difference'] > 0, 'Percentage_Difference'] = 0
        predictions_from_cmip_sum['District'] = predictions_from_cmip_sum['District'].replace(
            {"Mzimba North": "Mzimba", "Mzimba South": "Mzimba"}
        )

        for year in year_groupings:
            subset = predictions_from_cmip_sum[
                (predictions_from_cmip_sum['Year'] >= year) &
                (predictions_from_cmip_sum['Year'] <= year + 5)
                ]
            for _, row in subset.iterrows():
                district = row['District']
                percentage_diff = row['Percentage_Difference']
                if year not in percentage_diff_by_year_district:
                    percentage_diff_by_year_district[year] = {}
                if district not in percentage_diff_by_year_district[year]:
                    percentage_diff_by_year_district[year][district] = 0
                percentage_diff_by_year_district[year][district] += (percentage_diff) # 1.4 is conversion factor between births and pregancies
                percentage_diff_by_year_district_scenario[scenario][model_type] = (percentage_diff )

        data_for_plot = pd.DataFrame.from_dict(percentage_diff_by_year_district, orient='index').fillna(0)
        y_min = min(y_min, data_for_plot.min().min())
        y_max = max(y_max, data_for_plot.max().max())
        x_min = min(x_min, data_for_plot.index.min())
        x_max = max(x_max, data_for_plot.index.max())

        ax = axes[i, j]
        data_for_plot.plot(kind='bar', stacked=True, ax=ax, cmap='tab20', legend=False)
        ax.set_title(f"{scenario}: {model_type}", fontsize=10)
        if i == len(scenarios) - 1:
            ax.set_xlabel('Year', fontsize=12)
        if j == 0:
            ax.set_ylabel(f'Deficit of {service} services', fontsize=12)
        if (i == 0) & (j == 2):
           ax.legend(title="Districts", fontsize=10, title_fontsize=10, bbox_to_anchor=(1., 1))
        percentage_diff_by_year_district_all[scenario][model_type] = percentage_diff_by_year_district
for ax in axes.flatten():
    ax.set_ylim(y_min*15, y_max)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels,  bbox_to_anchor=(1, -10), loc = "center right", fontsize=10, title="Districts")
plt.tight_layout()
plt.savefig(results_folder_to_save / 'stacked_bar_percentage_difference_5_years_grid_single_legend.png')
#plt.show()
#

# Define file paths
monthly_reporting_file = "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv"
five_day_max_file = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facilities_with_ANC_five_day_cumulative.csv"
precipitation_path = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL"

# Define scenarios, models, and services
scenarios = ["ssp126", "ssp245", "ssp585"]
model_types = ["lowest", "mean", "highest"]
services = ["ANC", "Inpatient"]

# Load the monthly reporting data
monthly_reporting_df = pd.read_csv(monthly_reporting_file)
monthly_reporting_values = monthly_reporting_df.iloc[:, 1:].values.flatten()

# Load the 5-day max reporting data
five_day_max_df = pd.read_csv(five_day_max_file)
five_day_max_values = five_day_max_df.iloc[:, 1:].values.flatten()

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Monthly Reporting Distribution
sns.kdeplot(monthly_reporting_values, label="Monthly Reporting ANC", color='#1C6E8C', ax=axes[0])
axes[0].set_title("Distribution Comparisons - Monthly")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Density")
axes[0].legend()

# 5-Day Max Reporting Distribution
sns.kdeplot(five_day_max_values, label="5-Day Max Reporting ANC", color='#1C6E8C', ax=axes[1])
axes[1].set_title("Distribution Comparisons - 5 Day Max")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Density")
axes[1].legend()

# Loop through scenarios, models, and services to plot distributions
for scenario in scenarios:
    for model in model_types:
        for service in services:
            # Monthly Prediction Data
            monthly_file_path = os.path.join(
                precipitation_path, scenario, f"{model}_monthly_prediction_weather_by_facility_{service}.csv"
            )
            if os.path.exists(monthly_file_path):
                df = pd.read_csv(monthly_file_path)
                values = df.iloc[:, 1:].values.flatten()
                sns.kdeplot(values, label=f"{scenario} - {model} - {service}", alpha=0.7, ax=axes[0])

            # 5-Day Max Prediction Data
            five_day_file_path = os.path.join(
                precipitation_path, scenario, f"{model}_window_prediction_weather_by_facility_{service}.csv"
            )
            if os.path.exists(five_day_file_path):
                df = pd.read_csv(five_day_file_path)
                values = df.iloc[:, 1:].values.flatten()
                sns.kdeplot(values, label=f"{scenario} - {model} - {service}", alpha=0.7, ax=axes[1])

plt.tight_layout()
plt.show()



# Define file paths
monthly_reporting_file = "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv"
five_day_max_file = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facilities_with_ANC_five_day_cumulative.csv"
precipitation_path = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL"

# Define scenarios, models, and services
scenarios = ["ssp126", "ssp245", "ssp585"]
model_types = ["lowest", "mean", "highest"]
services = ["Inpatient"]

def filter_top_20_percent(values):
    """Filter values to keep only those in the top 80th percentile."""
    threshold = np.percentile(values, 80)
    return values[values >= threshold]

# Load and filter monthly reporting data
monthly_reporting_df = pd.read_csv(monthly_reporting_file)
monthly_reporting_values = monthly_reporting_df.iloc[:, 1:].values.flatten()
monthly_reporting_values = filter_top_20_percent(monthly_reporting_values)

# Load and filter 5-day max reporting data
five_day_max_df = pd.read_csv(five_day_max_file)
five_day_max_values = five_day_max_df.iloc[:, 1:].values.flatten()
five_day_max_values = filter_top_20_percent(five_day_max_values)

# Create a figure with 3 rows (one per scenario) and 2 columns (Monthly, 5-Day Max)
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for i, scenario in enumerate(scenarios):
    # Monthly Reporting Distribution
    sns.kdeplot(monthly_reporting_values, label="Monthly Reporting ANC", color='black', ax=axes[i, 0], alpha=1)
    axes[i, 0].set_title(f"Top 20% Distribution - Monthly ({scenario})")
    axes[i, 0].set_xlabel("Precipitation (mm)")
    axes[i, 0].set_ylabel("Density")

    # 5-Day Max Reporting Distribution
    sns.kdeplot(five_day_max_values, label="5-Day Max Reporting ANC", color='black', ax=axes[i, 1], alpha=1)
    axes[i, 1].set_title(f"Top 20% Distribution - 5 Day Max ({scenario})")
    axes[i, 1].set_xlabel("Precipitation (mm)")
    axes[i, 1].set_ylabel("Density")

    # Loop through model types and services to plot distributions
    for model in model_types:
        for service in services:
            # Monthly Prediction Data
            monthly_file_path = os.path.join(
                precipitation_path, scenario, f"{model}_monthly_prediction_weather_by_facility_{service}.csv"
            )
            if os.path.exists(monthly_file_path):
                df = pd.read_csv(monthly_file_path)
                values = df.iloc[:, 1:].values.flatten()
                values = filter_top_20_percent(values)
                if len(values) > 0:
                    sns.kdeplot(values, label=f"{model} - {service}", alpha=0.4, ax=axes[i, 0])

            # 5-Day Max Prediction Data
            five_day_file_path = os.path.join(
                precipitation_path, scenario, f"{model}_window_prediction_weather_by_facility_{service}.csv"
            )
            if os.path.exists(five_day_file_path):
                df = pd.read_csv(five_day_file_path)
                values = df.iloc[:, 1:].values.flatten()
                values = filter_top_20_percent(values)
                num_events_above_X = (values > 300).sum()
                print(f"Number of 5-day max precipitation events above X mm: {num_events_above_X}")

                if len(values) > 0:
                    sns.kdeplot(values, label=f"{model} - {service}", alpha=0.3, ax=axes[i, 1])

    axes[0, 0].legend()
    axes[0, 1].legend()

plt.tight_layout()
plt.savefig(f"/Users/rem76/Desktop/Climate_change_health/Data/historical_vs_future_precipitation_{service}.png")
