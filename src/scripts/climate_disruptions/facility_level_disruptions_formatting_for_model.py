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

# ═══════════════════════════════════════════════════════════════════════════
# Filter facilities CSV to only include facilities with ANC disruptions
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("Filtering facilities to only those with ANC disruptions")
print("=" * 80)

# Read the full facilities file
facilities_csv_path = "/Users/rem76/PycharmProjects/TLOmodel/resources/climate_change_impacts/facilities_with_lat_long_region.csv"
facilities_df = pd.read_csv(facilities_csv_path, low_memory=False)

print(f"\nTotal facilities in original file: {len(facilities_df)}")
print(f"Columns in facilities file: {list(facilities_df.columns)[:10]}...")  # Show first 10 columns

# Get unique facility names from the climate/disruption data
disrupted_facility_names = Climate_Projection_Facilities_List["Facility_ID"].unique()
print(f"Facilities with ANC disruptions in climate data: {len(disrupted_facility_names)}")

# Try to find the facility name column - could be 'Fname', 'Facility_ID', or something else
facility_name_col = None
for possible_col in ['Fname', 'Facility_ID', 'facility_name', 'FacilityName', 'Facility_Name', 'name', 'Name']:
    if possible_col in facilities_df.columns:
        facility_name_col = possible_col
        print(f"Found facility name column: '{facility_name_col}'")
        break

if facility_name_col is None:
    print("\nERROR: Could not find facility name column!")
    print(f"Available columns: {list(facilities_df.columns)}")
    print("\nPlease specify which column contains facility names.")
    # Try to guess based on first few rows
    print("\nFirst few rows of the dataframe:")
    print(facilities_df.head())
else:
    # Filter the facilities dataframe to only include disrupted facilities
    filtered_df = facilities_df[facilities_df[facility_name_col].isin(disrupted_facility_names)]

    print(f"Facilities found in master list: {len(filtered_df)}")
    print(f"Facilities NOT found in master list: {len(disrupted_facility_names) - len(filtered_df)}")

    # Check which ones weren't found (for debugging)
    found_names = set(filtered_df[facility_name_col].values)
    missing = [f for f in disrupted_facility_names if f not in found_names]
    if missing:
        print(f"\nFacilities with disruptions but not in master facilities list:")
        for m in missing[:10]:  # Show first 10
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Save the filtered file
    output_path = Path(climatefilepath) / "facilities_with_anc_disruptions.csv"
    filtered_df.to_csv(output_path, index=False)

    print(f"\n✓ Filtered facilities saved to: {output_path}")
    print(f"  Total rows: {len(filtered_df)}")

print("=" * 80 + "\n")
