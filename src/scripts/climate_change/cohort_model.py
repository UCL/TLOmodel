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
global_min = 0
global_max = 5
# service
ANC =True
if ANC:
    service = 'ANC'

## Get birth results
results_folder_to_save = Path(f'/Users/rem76/Desktop/Climate_change_health/Results/{service}_disruptions')
results_folder_for_births = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-25T110820Z")
resourcefilepath = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-25T110820Z")
historical_predictions = pd.read_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_model_historical_predictions_{service}.csv')
precipitation_threshold = historical_predictions['Precipitation'].quantile(0.9)
print(precipitation_threshold)
agegrps, agegrplookup = make_age_grp_lookup()
calperiods, calperiodlookup = make_calendar_period_lookup()
births_results = extract_results(
        results_folder_for_births,
        module="tlo.methods.demography",
        key="on_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
        do_scaling=True
    )

print(births_results)
births_results = births_results.groupby(by=births_results.index).sum()
births_results = births_results.replace({0: np.nan})

births_model = summarize(births_results, collapse_columns=True)
births_model.columns = ['Model_' + col for col in births_model.columns]
births_model_subset = births_model.iloc[15:].copy() # don't want 2010-2024

print("historical:", births_model.iloc[2:15, 1].sum())
births_model_subset_historical = births_model.iloc[2:15, 1]
# Load map of Malawi for later
file_path_historical_data = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/2011/60ab007aa16d679a32f9c3e186d2f744.nc"
dataset = Dataset(file_path_historical_data, mode='r')
pr_data = dataset.variables['tp'][:]
lat_data = dataset.variables['latitude'][:]
long_data = dataset.variables['longitude'][:]
meshgrid_from_netCDF = np.meshgrid(long_data, lat_data)

malawi = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm0_nso_20181016.shp")
malawi_admin2 = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
water_bodies = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/Water_Supply_Control-Rivers-shp/Water_Supply_Control-Rivers.shp")
#
# change names of some districts for consistency
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Blantyre City', 'Blantyre')
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Mzuzu City', 'Mzuzu')
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Lilongwe City', 'Lilongwe')
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Zomba City', 'Zomba')

difference_lat = lat_data[1] - lat_data[0]
difference_long = long_data[1] - long_data[0]

# # # Get expected disturbance from the model

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
        # Match birth results and predictions
        matching_rows = min(len(births_model_subset), len(predictions_from_cmip_sum))
        multiplied_values = births_model_subset.head(matching_rows).iloc[:, 1].values * predictions_from_cmip_sum[
            'Percentage_Difference'].head(matching_rows).values * 1.4 # 1.4 is conversion from births to pregnacnies

        # Check for negative values (missed cases?)
        negative_sum = np.sum(multiplied_values[multiplied_values < 0])

        # now do extreme precipitation by district and year, use original dataframe to get monthly top 10% precip
        filtered_predictions = predictions_from_cmip[predictions_from_cmip['Precipitation'] >= precipitation_threshold]
        filtered_predictions_sum = filtered_predictions.groupby('Year').sum().reset_index()
        percent_due_to_extreme = filtered_predictions_sum['Difference_in_Expectation'] / predictions_from_cmip_sum['Predicted_No_Weather_Model']
        print(percent_due_to_extreme)
        multiplied_values_extreme_precip = births_model_subset.head(matching_rows).iloc[:, 1].values * percent_due_to_extreme.head(matching_rows).values * 1.4
        negative_sum_extreme_precip = np.sum(multiplied_values_extreme_precip[multiplied_values_extreme_precip < 0])
        result_df = pd.DataFrame({
            "Scenario": [scenario],
            "Model_Type": [model_type],
            "Negative_Sum": [negative_sum],
            "Negative_Percentage": [negative_sum / (births_model_subset['Model_mean'].sum() * 1.4) * 100],
            "Extreme_Precip": [negative_sum_extreme_precip],
            "Extreme_Precip_Percentage": [(negative_sum_extreme_precip  / negative_sum) * 100]
        })

        results_list.append(result_df)

        # Save multiplied values by model and scenario
        multiplied_values_df = pd.DataFrame({
            'Year': year_range[:matching_rows],
            'Scenario': scenario,
            'Model_Type': model_type,
            'Multiplied_Values': multiplied_values,
            'Multiplied_Values_extreme_precip': multiplied_values_extreme_precip

        })
        multiplied_values_df.to_csv(results_folder_to_save/f'multiplied_values_{scenario}_{model_type}.csv', index=False)

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
        malawi_admin2['Percentage_Difference'] = malawi_admin2['Percentage_Difference'].abs() # for mapping, to show %


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
        malawi_admin2['Percentage_Difference'] = malawi_admin2['Percentage_Difference'].abs() # for mapping, to show %

        ax = axes[i, j]
        water_bodies.plot(ax=ax, facecolor="none", edgecolor="#999999", linewidth=0.5, hatch="xxx")
        water_bodies.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

        malawi_admin2.dropna(subset=['Percentage_Difference']).plot(
            ax=ax,
            column='Percentage_Difference',
            cmap='Blues',
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
    cmap='Blues',
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

# #### Now do number of births based on the TLO model and 2018 census
population_file = "/Users/rem76/PycharmProjects/TLOmodel/resources/demography/ResourceFile_PopulationSize_2018Census.csv"
population_data = pd.read_csv(population_file)
population_data = population_data.merge(
    predictions_from_cmip[['District', 'Zone']], # need for larger aggregation, equivalent to zones. also invariant across scenarios
    on='District',
    how='left'
)
population_data_grouped = population_data.groupby("District")["Count"].sum()
total_population = population_data_grouped.sum()
population_proportion = population_data_grouped / total_population

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
        predictions_from_cmip_sum['Percentage_Difference'] = predictions_from_cmip_sum['Percentage_Difference'].abs() # for mapping, to show %

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
                row_index = births_model_subset.index.get_loc(year)

                number_of_births = population_proportion[district] * births_model_subset.iloc[row_index]["Model_mean"]
                if year not in percentage_diff_by_year_district:
                    percentage_diff_by_year_district[year] = {}
                if district not in percentage_diff_by_year_district[year]:
                    percentage_diff_by_year_district[year][district] = 0
                percentage_diff_by_year_district[year][district] += (percentage_diff * number_of_births) * 1.4 # 1.4 is conversion factor between births and pregancies
                percentage_diff_by_year_district_scenario[scenario][model_type] = (percentage_diff * number_of_births) * 1.4

        data_for_plot = pd.DataFrame.from_dict(percentage_diff_by_year_district, orient='index').fillna(0)
        y_min = min(y_min, data_for_plot.min().min())
        y_max = max(y_max, data_for_plot.max().max())
        x_min = min(x_min, data_for_plot.index.min())
        x_max = max(x_max, data_for_plot.index.max())

        ax = axes[i, j]
        data_for_plot.plot(kind='bar', stacked=True, ax=ax, cmap='tab20', legend=False)
        #ax.set_title(f"{scenario}: {model_type}", fontsize=10)
        if i == len(scenarios) - 1:
            ax.set_xlabel('Year', fontsize=12)
        if j == 0:
            ax.set_ylabel(f'Distruption of {service} services', fontsize=10)
        #if (i == 0) & (j == 2):
        #    ax.legend(title="Districts", fontsize=10, title_fontsize=10, bbox_to_anchor=(1., 1))
        percentage_diff_by_year_district_all[scenario][model_type] = percentage_diff_by_year_district
for ax in axes.flatten():
    ax.set_ylim(y_min, y_max*19)

for j, model_type in enumerate(["Lowest", "Mean", "Highest"]):
    axes[0, j].set_title(model_type, fontsize=14, fontweight='bold')
for i, scenario in enumerate(scenarios):
    axes[i, 0].annotate(scenario, xy=(-0.3, 0.5), xycoords="axes fraction",
                         fontsize=14, fontweight='bold', ha='center', va='center', rotation=90)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels,  bbox_to_anchor=(1, -10), loc = "center right", fontsize=10, title="Districts")
#plt.tight_layout()
plt.savefig(results_folder_to_save / 'stacked_bar_percentage_difference_5_years_grid_single_legend_with_births.png')
#plt.show()
# #
#

## % of cases that occur due to being in the top 10 percet?
# #### Now do number of births based on the TLO model and 2018 census
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
        predictions_from_cmip_sum['Percentage_Difference'] = predictions_from_cmip_sum['Percentage_Difference'].abs() # for mapping, to show %
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
                row_index = births_model_subset.index.get_loc(year)

                number_of_births = population_proportion[district] * births_model_subset.iloc[row_index]["Model_mean"]
                if year not in percentage_diff_by_year_district:
                    percentage_diff_by_year_district[year] = {}
                if district not in percentage_diff_by_year_district[year]:
                    percentage_diff_by_year_district[year][district] = 0
                percentage_diff_by_year_district[year][district] += ((percentage_diff * number_of_births) * 1.4)/(number_of_births*1.4) * 1000 # 1.4 is conversion factor between births and pregancies
                percentage_diff_by_year_district_scenario[scenario][model_type] = (percentage_diff * number_of_births) * 1.4

        data_for_plot = pd.DataFrame.from_dict(percentage_diff_by_year_district, orient='index').fillna(0)
        y_min = min(y_min, data_for_plot.min().min())
        y_max = max(y_max, data_for_plot.max().max())
        x_min = min(x_min, data_for_plot.index.min())
        x_max = max(x_max, data_for_plot.index.max())

        ax = axes[i, j]
        data_for_plot.plot(kind='bar', stacked=True, ax=ax, cmap='tab20', legend=False)
        #ax.set_title(f"{scenario}: {model_type}", fontsize=10)
        if i == len(scenarios) - 1:
            ax.set_xlabel('Year', fontsize=10)
        if j == 0:
            ax.set_ylabel(f'Distruption of {service} services per 1,000 pregnancies', fontsize=12)
        #if (i == 0) & (j == 2):
        #    ax.legend(title="Districts", fontsize=10, title_fontsize=10, bbox_to_anchor=(1., 1))
        percentage_diff_by_year_district_all[scenario][model_type] = percentage_diff_by_year_district
for ax in axes.flatten():
    ax.set_ylim(y_min, y_max*19)
handles, labels = ax.get_legend_handles_labels()

for j, model_type in enumerate(["Lowest", "Mean", "Highest"]):
    axes[0, j].set_title(model_type, fontsize=14, fontweight='bold')
for i, scenario in enumerate(scenarios):
    axes[i, 0].annotate(scenario, xy=(-0.3, 0.5), xycoords="axes fraction",
                         fontsize=14, fontweight='bold', ha='center', va='center', rotation=90)
#plt.tight_layout()
plt.savefig(results_folder_to_save / 'stacked_bar_percentage_difference_5_years_grid_single_legend_with_births_per_1000.png')
#plt.show()
#
#

# #### By zone

population_data_grouped_zone = population_data.groupby("Zone")["Count"].sum()
population_proportion_zone = population_data_grouped_zone / total_population

# Create the figure and axes grid
fig, axes = plt.subplots(3, 3, figsize=(18.5, 18))
y_min = float('inf')
y_max = float('-inf')
x_min = float('inf')
x_max = float('-inf')
percentage_diff_by_year_zone_all = {}
percentage_diff_by_year_zone_scenario = {}
year_groupings = range(2025, 2060, 5)
for i, scenario in enumerate(scenarios):
    percentage_diff_by_year_zone_all[scenario] = {}
    percentage_diff_by_year_zone_scenario[scenario] = {}

    for j, model_type in enumerate(model_types):
        percentage_diff_by_year_zone_all[scenario][model_type] = {}
        percentage_diff_by_year_zone_scenario[scenario][model_type] = {}
        percentage_diff_by_year_zone_scenario[scenario][model_type] = 0

        percentage_diff_by_year_zone = {}

        predictions_from_cmip =  predictions_from_cmip.loc[predictions_from_cmip['Difference_in_Expectation'] < 0]
        predictions_from_cmip_sum = predictions_from_cmip_sum[predictions_from_cmip_sum['Year'] <= 2061]

        predictions_from_cmip_sum = predictions_from_cmip.groupby(['Year', 'Zone']).sum().reset_index()
        predictions_from_cmip_sum = predictions_from_cmip_sum[predictions_from_cmip_sum['Year'] <= 2060]
        predictions_from_cmip_sum['Percentage_Difference'] = (
                                                                 predictions_from_cmip_sum[
                                                                     'Difference_in_Expectation'] /
                                                                 predictions_from_cmip_sum['Predicted_No_Weather_Model'])
        predictions_from_cmip_sum.loc[
            predictions_from_cmip_sum['Percentage_Difference'] > 0, 'Percentage_Difference'] = 0
        predictions_from_cmip_sum['Percentage_Difference'] = predictions_from_cmip_sum['Percentage_Difference'].abs() # for mapping, to show %

        predictions_from_cmip_sum['District'] = predictions_from_cmip_sum['District'].replace(
            {"Mzimba North": "Mzimba", "Mzimba South": "Mzimba"}
        )

        for year in year_groupings:
            subset = predictions_from_cmip_sum[
                (predictions_from_cmip_sum['Year'] >= year) &
                (predictions_from_cmip_sum['Year'] <= year + 5)
                ]
            for _, row in subset.iterrows():
                zone = row['Zone']
                percentage_diff = row['Percentage_Difference']
                row_index = births_model_subset.index.get_loc(year)

                number_of_births = population_data_grouped_zone[zone] * births_model_subset.iloc[row_index]["Model_mean"]
                if year not in percentage_diff_by_year_zone:
                    percentage_diff_by_year_zone[year] = {}
                if zone not in percentage_diff_by_year_zone[year]:
                    percentage_diff_by_year_zone[year][zone] = 0
                percentage_diff_by_year_zone[year][zone] += ((percentage_diff * number_of_births) * 1.4)/(number_of_births*1.4) * 1000 # 1.4 is conversion factor between births and pregancies
                percentage_diff_by_year_zone_scenario[scenario][model_type] = (percentage_diff * number_of_births) * 1.4

        data_for_plot = pd.DataFrame.from_dict(percentage_diff_by_year_zone, orient='index').fillna(0)
        y_min = min(y_min, data_for_plot.min().min())
        y_max = max(y_max, data_for_plot.max().max())
        x_min = min(x_min, data_for_plot.index.min())
        x_max = max(x_max, data_for_plot.index.max())

        ax = axes[i, j]
        data_for_plot.plot(kind='bar', stacked=True, ax=ax, cmap='tab20', legend=False)
        #ax.set_title(f"{scenario}: {model_type}", fontsize=10)
        if i == len(scenarios) - 1:
            ax.set_xlabel('Year', fontsize=12)
        if j == 0:
            ax.set_ylabel(f'Deficit of {service} services per 1,000 pregnancies', fontsize=10,  labelpad=10)
        if (i == 0) & (j == 2):
           ax.legend(title="Zones", fontsize=10, title_fontsize=10, ncol = 2)
        percentage_diff_by_year_zone_all[scenario][model_type] = percentage_diff_by_year_zone
for ax in axes.flatten():
    ax.set_ylim(y_min, y_max*6)

for j, model_type in enumerate(["Lowest", "Mean", "Highest"]):
    axes[0, j].set_title(model_type, fontsize=14, fontweight='bold')
for i, scenario in enumerate(scenarios):
    axes[i, 0].annotate(scenario, xy=(-0.3, 0.5), xycoords="axes fraction",
                         fontsize=13, fontweight='bold', ha='center', va='center', rotation=90)
plt.subplots_adjust(left=0.12)
#plt.tight_layout()
plt.savefig(results_folder_to_save / 'stacked_bar_percentage_difference_5_years_grid_single_legend_with_births_per_1000_ZONE.png')
#plt.show()
#

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
malawi_admin2['Percentage_Difference_historical'] = malawi_admin2['Percentage_Difference_historical'].abs()  # for mapping, to show %

percentage_diff_by_district_historical_average = historical_predictions_sum['Percentage_Difference'].mean()
filtered_predictions = historical_predictions[historical_predictions['Precipitation'] >= precipitation_threshold]
filtered_predictions_sum = filtered_predictions.groupby('Year').sum().reset_index()
percent_due_to_extreme = filtered_predictions_sum['Difference_in_Expectation'].sum()
percent_due_to_extreme = percent_due_to_extreme/historical_predictions['Difference_in_Expectation'].sum()
print(percent_due_to_extreme)
fig, ax = plt.subplots(figsize=(10, 10))
water_bodies.plot(ax=ax, facecolor="none", edgecolor="#999999", linewidth=0.5, hatch="xxx")
water_bodies.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

malawi_admin2.dropna(subset=['Percentage_Difference_historical']).plot(
    ax=ax,
    column='Percentage_Difference_historical',
    cmap='Blues',
    edgecolor='black',
    alpha=1,
    legend=False,
    vmin=global_min,
    vmax=global_max
)

ax.set_ylabel("Latitude", fontsize=10)
ax.set_xlabel("Longitude", fontsize=10)

sm = plt.cm.ScalarMappable(
    cmap='Blues',
    norm=mcolors.Normalize(vmin=global_min, vmax=global_max)
)
sm.set_array([])
fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, label="Percentage Difference (%)")

plt.title("", fontsize=16)
plt.savefig(results_folder_to_save / 'percentage_difference_map_historical.png')
plt.show()



# #### Now do number of births based on the TLO model (2010 - 2024) and 2018 census

historical_predictions_negative =  historical_predictions.loc[historical_predictions['Difference_in_Expectation'] < 0]
historical_predictions_negative = historical_predictions_negative[historical_predictions_negative['Year'] <= 2024]
        # total disruptions
historical_predictions_negative_sum = historical_predictions_negative.groupby('Year').sum().reset_index()
historical_predictions_negative_sum['Percentage_Difference'] = (
                historical_predictions_negative_sum['Difference_in_Expectation'] / historical_predictions_negative_sum[
                'Predicted_No_Weather_Model'])
        # Match birth results and predictions
multiplied_values_historical = births_model_subset_historical.values * historical_predictions_negative_sum[
            'Percentage_Difference'].values * 1.4 # 1.4 is conversion from births to pregnacnies

# Check for negative values (missed cases?)
negative_sum_historical = np.sum(multiplied_values_historical[multiplied_values_historical < 0])

# now do extreme precipitation by district and year, use original dataframe to get monthly top 10% precip
filtered_predictions_historical = historical_predictions_negative[historical_predictions_negative['Precipitation'] >= precipitation_threshold]
filtered_predictions_sum_historical = filtered_predictions_historical.groupby('Year').sum().reset_index()
percent_due_to_extreme_historical = filtered_predictions_sum_historical['Difference_in_Expectation'] / historical_predictions_negative_sum['Predicted_No_Weather_Model']
print(percent_due_to_extreme_historical)
multiplied_values_extreme_precip_historical = births_model_subset_historical.values * percent_due_to_extreme_historical.values * 1.4
negative_sum_extreme_precip_historical = np.sum(multiplied_values_extreme_precip_historical[multiplied_values_extreme_precip_historical < 0])
result_df_historical = pd.DataFrame({
            "Negative_Sum": [negative_sum_historical],
            "Negative_Percentage": [negative_sum_historical / (births_model_subset_historical.sum() * 1.4) * 100],
            "Extreme_Precip": [negative_sum_extreme_precip_historical],
            "Extreme_Precip_Percentage": [(negative_sum_extreme_precip_historical  / negative_sum_historical) * 100]
        })

# Save multiplied values by model and scenario
multiplied_values_df_historical = pd.DataFrame({
            'Year': range(2012, 2025),
            'Multiplied_Values': multiplied_values_historical,
            'Multiplied_Values_extreme_precip': multiplied_values_extreme_precip_historical

        })
multiplied_values_df_historical.to_csv(results_folder_to_save/f'multiplied_values_historical.csv', index=False)

print(result_df_historical)
result_df_historical.to_csv(f'/Users/rem76/Desktop/Climate_change_health/Results/{service}_disruptions/negative_sums_and_percentages_historical.csv', index=False)


###### Effect of CYCLONE FREDDY #######

historical_predictions = pd.read_csv(f'/Users/rem76/Desktop/Climate_change_health/Data/results_of_model_historical_predictions_{service}.csv')


def calculate_percentage_difference(historical_predictions, year, zone=None):
    months = [1, 2, 3]
    filtered_data = {
        month: historical_predictions[
            (historical_predictions['Year'] == year) & (historical_predictions['Month'] == month)
            & ((historical_predictions['Zone'] == zone) if zone else True)
            ]
        for month in months
    }

    for month in months[1:]:
        if not filtered_data[1].empty and not filtered_data[month].empty:
            total_1 = filtered_data[1]['Predicted_Weather_Model'].sum()
            total_n = filtered_data[month]['Predicted_Weather_Model'].sum()
            percent_difference = ((total_1 - total_n) / total_1) * 100
            print(f"Percentage Difference January and {month} for {zone}: {percent_difference:.2f}%")
        else:
            print(f"One of the datasets for January or {month} is empty, cannot compute percentage difference.")


# Run the function for different zones
calculate_percentage_difference(historical_predictions, 2023, 'South East')
calculate_percentage_difference(historical_predictions, 2023, 'South West')
calculate_percentage_difference(historical_predictions, 2023)  # Without zone filter

###
historical_predictions_negative_freddy = historical_predictions.loc[historical_predictions['Difference_in_Expectation'] < 0]
historical_predictions_negative_freddy = historical_predictions_negative_freddy[historical_predictions_negative_freddy['Year'] == 2023]
historical_predictions_negative_freddy = historical_predictions_negative_freddy[historical_predictions_negative_freddy['Zone'] == "South West"]
historical_predictions_negative_freddy_feb = historical_predictions_negative_freddy[
    historical_predictions_negative_freddy['Month'].isin([2])
]
print("February mean:", historical_predictions_negative_freddy_feb['Difference_in_Expectation'].mean())
print("February max:", historical_predictions_negative_freddy_feb['Difference_in_Expectation'].min())

historical_predictions_negative_freddy_march = historical_predictions_negative_freddy[
    historical_predictions_negative_freddy['Month'].isin([3])
]
print("March mean:", historical_predictions_negative_freddy_march['Difference_in_Expectation'].mean(skipna=True))
print("March max:", historical_predictions_negative_freddy_march['Difference_in_Expectation'].min(skipna=True))


##################
# Define file paths
monthly_reporting_file = "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv"
five_day_max_file = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/historical_daily_total_by_facilities_with_ANC_five_day_cumulative.csv"
precipitation_path = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL"

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
                if len(values) > 0:
                    sns.kdeplot(values, label=f"{model} - {service}", alpha=0.3, ax=axes[i, 1])

    axes[0, 0].legend()
    axes[0, 1].legend()

#plt.tight_layout()
plt.show()
plt.savefig(f"/Users/rem76/Desktop/Climate_change_health/Data/historical_vs_future_precipitation_ANC.png")



