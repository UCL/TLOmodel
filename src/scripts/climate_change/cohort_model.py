from pathlib import Path

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

## Get birth results
results_folder_to_save = Path('/Users/rem76/Desktop/Climate_change_health/Results/ANC_disruptions')
results_folder_for_births = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-25T110820Z")
resourcefilepath = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-25T110820Z")
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
births_results = births_results.groupby(by=births_results.index).sum()
births_results = births_results.replace({0: np.nan})

births_model = summarize(births_results, collapse_columns=True)
births_model.columns = ['Model_' + col for col in births_model.columns]
births_model_subset = births_model.iloc[15:].copy() # don't want 2010-2024

# Load map of Malawi for later
file_path_historical_data = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_total/2011/60ab007aa16d679a32f9c3e186d2f744.nc"
dataset = Dataset(file_path_historical_data, mode='r')
pr_data = dataset.variables['tp'][:]
lat_data = dataset.variables['latitude'][:]
long_data = dataset.variables['longitude'][:]
meshgrid_from_netCDF = np.meshgrid(long_data, lat_data)

malawi = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm0_nso_20181016.shp")
malawi_admin2 = gpd.read_file("/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")

# change names of some districts for consistency
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Blantyre City', 'Blantyre')
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Mzuzu City', 'Mzuzu')
malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Lilongwe City', 'Lilongwe')

difference_lat = lat_data[1] - lat_data[0]
difference_long = long_data[1] - long_data[0]

# Get expected disturbance from the model
scenarios = ['ssp245', 'ssp585']
model_types = ['lowest', 'median', 'highest']
year_range = range(min_year, max_year)

# Loop through scenarios and model types
for scenario in scenarios:
    for model_type in model_types:
        predictions_from_cmip = pd.read_csv(
            f'/Users/rem76/Desktop/Climate_change_health/Data/weather_predictions_with_X_{scenario}_{model_type}.csv'
        )
        predictions_from_cmip_sum = predictions_from_cmip.groupby('Year').sum().reset_index()
        predictions_from_cmip_sum['Percentage_Difference'] = (
                predictions_from_cmip_sum['Difference_in_Expectation'] / predictions_from_cmip_sum[
                'Predicted_No_Weather_Model'])
        # Match birth results and predictions
        matching_rows = min(len(births_model_subset), len(predictions_from_cmip_sum))
        print(matching_rows)
        multiplied_values = births_model_subset.head(matching_rows).iloc[:, 1].values * predictions_from_cmip_sum[
            'Percentage_Difference'].head(matching_rows).values
        births_model_subset['Multiplied_Values'] = multiplied_values

        # Plot the results
        plt.plot(year_range, multiplied_values)
        plt.ylabel("Change ANC cases due to weather")
        plt.axhline(y=0, color='black', linestyle='--')
        plt.plot(year_range, predictions_from_cmip_sum.iloc[0:36, -1] * 100)
        plt.xlabel("Percentage Change in ANC cases due to weather")
        plt.axhline(y=0, color='black', linestyle='--')

        # Check for negative values (missed cases?)
        negative_sum = np.sum(multiplied_values[multiplied_values < 0])
        print("Sum of values < 0:", negative_sum)
        print(negative_sum / births_model_subset['Model_mean'].sum() * 100)

        # Plot by zone
        predictions_from_cmip_sum = predictions_from_cmip.groupby(['Year', 'Zone']).sum().reset_index()
        plt.figure(figsize=(10, 6))
        for zone in predictions_from_cmip_sum['Zone'].unique():
            zone_data = predictions_from_cmip_sum[predictions_from_cmip_sum['Zone'] == zone]
            zone_data['Percentage_Difference'] = (zone_data['Difference_in_Expectation'] / zone_data[
                'Predicted_No_Weather_Model']) * 100
            plt.plot(zone_data['Year'], zone_data['Percentage_Difference'], label=f'Zone {zone}')
        plt.xlabel("Year")
        plt.ylabel("Change ANC cases due to weather")
        plt.axhline(y=0, color='black', linestyle='--')
        plt.ylim(-1.5, 0)
        plt.legend(title='Zones')

        # Plot by district
        predictions_from_cmip_sum = predictions_from_cmip.groupby(['Year', 'District']).sum().reset_index()
        plt.figure(figsize=(10, 6))
        for district in predictions_from_cmip_sum['District'].unique():
            district_data = predictions_from_cmip_sum[predictions_from_cmip_sum['District'] == district]
            district_data['Percentage_Difference'] = (district_data['Difference_in_Expectation'] / district_data[
                'Predicted_No_Weather_Model']) * 100
            plt.plot(district_data['Year'], district_data['Percentage_Difference'], label=f'{district}')
        plt.xlabel("Year")
        plt.ylabel("Change ANC cases due to weather")
        plt.axhline(y=0, color='black', linestyle='--')
        plt.ylim(-2.5, 0)
        plt.legend(title='Districts')

        # Generate district map visualization
        predictions_from_cmip_sum['District'] = predictions_from_cmip_sum['District'].replace(
            {"Mzimba North": "Mzimba", "Mzimba South": "Mzimba"})
        polygons = [
            Polygon(
                [(x, y), (x + difference_long, y), (x + difference_long, y + difference_lat), (x, y + difference_lat)])
            for x in long_data for y in lat_data
        ]
        grid = gpd.GeoDataFrame({'geometry': polygons}, crs=malawi.crs)
        grid_clipped_ADM2 = gpd.overlay(grid, malawi_admin2, how='intersection')
        predictions_from_cmip_sum['Percentage_Difference'] = (predictions_from_cmip_sum['Difference_in_Expectation'] /
                                                              predictions_from_cmip_sum[
                                                                  'Predicted_No_Weather_Model']) * 100
        percentage_diff_by_district = predictions_from_cmip_sum.groupby('District')['Percentage_Difference'].mean()
        grid_clipped_ADM2['Percentage_Difference'] = grid_clipped_ADM2['ADM2_EN'].map(percentage_diff_by_district)
        grid_clipped_ADM2.loc[grid_clipped_ADM2['Percentage_Difference'] > 0, 'Percentage_Difference'] = 0

        # Plot map
        fig, ax = plt.subplots(figsize=(12, 12))
        malawi_admin2.plot(ax=ax, edgecolor='white', color='white')
        grid_clipped_ADM2.dropna(subset=['Percentage_Difference']).plot(
            ax=ax,
            column='Percentage_Difference',
            cmap='Blues_r',
            edgecolor='black',
            alpha=1,
            legend=False
        )
        sm = plt.cm.ScalarMappable(cmap='Blues_r',
                                   norm=mcolors.Normalize(vmin=grid_clipped_ADM2['Percentage_Difference'].min(),
                                                          vmax=grid_clipped_ADM2['Percentage_Difference'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
        cbar.set_label("Percentage Difference (%)", fontsize=12)
        plt.xlabel("Longitude", fontsize=14)
        plt.ylabel("Latitude", fontsize=14)
        plt.title(f"{scenario}: {model_type}", fontsize=16)
        plt.tight_layout()
        plt.savefig(results_folder_to_save / f'{scenario}_{model_type}_map_Malawi_cumulative_difference.png')
        # Save multiplied values by model and scenario
        multiplied_values_df = pd.DataFrame({
            'Year': year_range[:matching_rows],
            'Scenario': scenario,
            'Model_Type': model_type,
            'Multiplied_Values': multiplied_values
        })
        multiplied_values_df.to_csv(results_folder_to_save/f'multiplied_values_{scenario}_{model_type}.csv', index=False)

#
# # Get unique districts from both sources
# adm2_districts = set(grid_clipped_ADM2['ADM2_EN'].unique())
# prediction_districts = set(predictions_from_cmip_sum['District'].unique())
#
# # Districts in ADM2 but not in predictions
# missing_in_predictions = adm2_districts - prediction_districts
# print("Districts in ADM2 but not in predictions:", missing_in_predictions)
#
# # Districts in predictions but not in ADM2
# missing_in_adm2 = prediction_districts - adm2_districts
# print("Districts in predictions but not in ADM2:", missing_in_adm2)
#




