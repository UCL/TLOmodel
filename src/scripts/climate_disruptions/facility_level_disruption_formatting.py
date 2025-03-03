import pandas as pd
import numpy as np
from pathlib import Path

resourcefilepath = '/Users/rem76/PycharmProjects/TLOmodel/resources'
path_to_resourcefiles_for_healthsystem = Path(resourcefilepath) / 'healthsystem'

climatefilepath = '/Users/rem76/Desktop/Climate_change_health/Data'
services = ["ANC"]
scenarios = ['ssp126', 'ssp245', 'ssp585']
ensemble_types = ['lowest', 'mean', 'highest']
# read in reference files
Master_Facilities_List = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')

# read in climate files - all will have the same facilities
sample_climate_file = Path(climatefilepath)/'weather_predictions_with_X_ssp585_mean_ANC.csv'

Climate_Projection_Facilities_List = pd.read_csv(sample_climate_file)

# Calculate proportion
matched_facilities = Master_Facilities_List['Facility_ID'].isin(Climate_Projection_Facilities_List['Facility_ID']).mean()

print(f"Proportion of Facility_IDs in Master_Facilities_List that are in Climate_Projection_Facilities_List: {matched_facilities:.2%}")

# all facilities are represented

## So create a dataframe/files of disruptions
for scenario in scenarios:
    for model in ensemble_types:

        for service in services:
            climate_file = pd.read_csv(Path(climatefilepath) / f'weather_predictions_with_X_{scenario}_{model}_{service}.csv')
            projected_precip_disruptions = pd.DataFrame(
                columns=['Facility_ID', 'year', 'month', 'service', 'disruption', 'mean_all_service'])
            projected_precip_disruptions['Facility_ID'] = climate_file['Facility_ID']
            projected_precip_disruptions['year'] = climate_file['Year']
            projected_precip_disruptions['month'] = climate_file['Month']
            for service in services:
                projected_precip_disruptions['service'] = [service] * len(climate_file['Month'])
            projected_precip_disruptions['disruption'] = climate_file['Difference_in_Expectation']
            projected_precip_disruptions['disruption'] = np.where(
                projected_precip_disruptions['disruption'] < 0,
                projected_precip_disruptions['disruption'],
                0
            )
        projected_precip_disruptions['mean_all_service'] = \
        projected_precip_disruptions.groupby(['Facility_ID', 'month', 'year'])['disruption'].transform('mean')
        projected_precip_disruptions.to_csv(f'/Users/rem76/Desktop/ResourceFiles_Disruptions/ResourceFile_Precipitation_Disruptions_{scenario}_{model}.csv')

