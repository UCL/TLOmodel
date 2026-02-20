import pandas as pd
import numpy as np
from pathlib import Path

resourcefilepath = "/Users/rem76/PycharmProjects/TLOmodel/resources"
path_to_resourcefiles_for_healthsystem = Path(resourcefilepath) / "healthsystem"

climatefilepath = "/Users/rem76/Desktop/Climate_Change_Health/Data"
services = ["ANC"]
scenarios = ["ssp126", "ssp245", "ssp585"]
ensemble_types = ["lowest", "mean", "highest"]
# read in climate files - all will have the same facilities
sample_climate_file = Path(climatefilepath) / "weather_predictions_with_X_ssp585_mean_ANC.csv"

Climate_Projection_Facilities_List = pd.read_csv(sample_climate_file)

## So create a dataframe/files of disruptions
for scenario in scenarios:
    for model in ensemble_types:
        for service in services:
            climate_file = pd.read_csv(
                Path(climatefilepath) / f"weather_predictions_with_X_{scenario}_{model}_{service}.csv"
            )
            projected_precip_disruptions = pd.DataFrame(
                columns=["RealFacility_ID", "year", "month", "service", "disruption", "mean_all_service"]
            )
            projected_precip_disruptions["RealFacility_ID"] = climate_file["Facility_ID"]
            projected_precip_disruptions["year"] = climate_file["Year"]
            projected_precip_disruptions["month"] = climate_file["Month"]
            for service in services:
                # projected_precip_disruptions['service'] = ['ANC'] * len(climate_file['Month'])
                projected_precip_disruptions["service"] = ["all"] * len(
                    climate_file["Month"]
                )  # initially assuming all are disrupted as ANC is

            projected_precip_disruptions["disruption"] = (
                climate_file["Difference_in_Expectation"] / climate_file["Predicted_No_Weather_Model"]
            )
            projected_precip_disruptions["disruption"] = np.where(
                projected_precip_disruptions["disruption"] < 0, projected_precip_disruptions["disruption"], 0
            )
            projected_precip_disruptions["disruption"] = abs(
                projected_precip_disruptions["disruption"]
            )  # for sampling later
        projected_precip_disruptions["mean_all_service"] = projected_precip_disruptions.groupby(
            ["RealFacility_ID", "month", "year"]
        )["disruption"].transform("mean")
        projected_precip_disruptions.to_csv(
            f"/Users/rem76/Desktop/Climate_Change_Health/Disruption_data_for_model/ResourceFile_Precipitation_Disruptions_{scenario}_{model}.csv"
        )
