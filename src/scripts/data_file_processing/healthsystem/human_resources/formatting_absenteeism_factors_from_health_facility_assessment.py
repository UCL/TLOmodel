import pandas as pd

dict = { "1a" : "L1a_Av_Mins_Per_Day", "1b":"L1b_Av_Mins_Per_Day", "2":"L2_Av_Mins_Per_Day", "0":"L0_Av_Mins_Per_Day", "3": "L3_Av_Mins_Per_Day", "4": "L4_Av_Mins_Per_Day", "5": "L5_Av_Mins_Per_Day"}

# Specify the file paths
file_path1 = "resources/healthsystem/human_resources/actual/ResourceFile_Daily_Capabilities.csv"
file_path2 = "resources/healthsystem/human_resources/definitions/ResourceFile_Officer_Types_Table.csv"
file_path3 = "resources/healthsystem/absenteeism/HHFA_amended_ResourceFile_patient_facing_time.xlsx"

# Load Excel files into DataFrames
daily_capabilities = pd.read_csv(file_path1)
officer_types = pd.read_csv(file_path2)
survey_daily_capabilities = pd.read_excel(file_path3, sheet_name="Scenario 2")

# Clean survey_daily_capabilities by replacing officer codes with category, and calculating mean within category
merged_df = pd.merge(survey_daily_capabilities, officer_types, on="Officer_Type_Code", how="left")
survey_daily_capabilities["Officer_Category"] = merged_df["Officer_Category"]
del survey_daily_capabilities["Officer_Type_Code"]
del survey_daily_capabilities["Total_Av_Working_Days"]
survey_daily_capabilities = survey_daily_capabilities.groupby("Officer_Category").mean().reset_index()

# Obtain average mins per day
daily_capabilities["Av_mins_per_day"] = (daily_capabilities["Total_Mins_Per_Day"]/daily_capabilities["Staff_Count"]).fillna(0)

# Obtain officers types
officers = daily_capabilities["Officer_Category"].drop_duplicates()

# Obtain mean daily capabilities for given facility level and officer category across all facilities
summarise_daily_capabilities = pd.DataFrame(columns=survey_daily_capabilities.columns)
summarise_daily_capabilities["Officer_Category"] = survey_daily_capabilities["Officer_Category"]

for level in ["0", "1a", "1b", "2"]:
    dc_at_level = daily_capabilities[daily_capabilities["Facility_Level"]==level]
    for officer in officers:
        dc_at_level_officer = dc_at_level[dc_at_level["Officer_Category"]==officer]
        mean_val = dc_at_level_officer["Av_mins_per_day"].mean()
        summarise_daily_capabilities.loc[summarise_daily_capabilities["Officer_Category"] == officer, dict[level]] = mean_val

survey_daily_capabilities = survey_daily_capabilities.set_index("Officer_Category")
summarise_daily_capabilities = summarise_daily_capabilities.set_index("Officer_Category")

# If not data is available, assume scaling factor of 1
absenteeism_factor = (survey_daily_capabilities/summarise_daily_capabilities).fillna(1.)

# Output absenteeism file
absenteeism_factor.to_excel("absenteeism_factor.xlsx")

print(absenteeism_factor)
