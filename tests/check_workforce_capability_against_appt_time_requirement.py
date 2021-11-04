from pathlib import Path

import pandas as pd

resourcefilepath = Path('./resources')

# The path of necessary ResourceFile tables
outputlocation = resourcefilepath / 'healthsystem'

# Get districts and regions info
pop = pd.read_csv(outputlocation / 'organisation' / 'ResourceFile_District_Population_Data.csv')
pop_districts = pop['District'].values
pop_regions = pd.unique(pop['Region'])

# Get appt time table (coarse)
appt_time_table_coarse = pd.read_csv(
    outputlocation / 'human_resources' / 'definitions' / 'ResourceFile_Appt_Time_Table_Coarse.csv')

# Get funded capabilities (coarse)
fund_daily_capability_coarse = pd.read_csv(
    outputlocation / 'human_resources' / 'funded' / 'ResourceFile_Daily_Capabilities_Coarse.csv')

# Get actual capabilities (coarse)
curr_daily_capability_coarse = pd.read_csv(
    outputlocation / 'human_resources' / 'actual' / 'ResourceFile_Daily_Capabilities_Coarse.csv')

# Facility_Levels
Facility_Levels = ['0', '1a', '1b', '2', '3', '4', '5']


# test for an appointment that has time requirements at a particular level (in Appt_Time_Table), \
# then indeed, the staff capabilities are available to satisfy that, for a person in any district \
# (including the regional and national facilities)

# Define the test function
def all_appts_can_run(capability):
    # Creat a table storing whether the appts have consistent time requirements and capabilities
    appt_have_or_miss_capability = appt_time_table_coarse.copy()
    # Delete the column of minutes
    appt_have_or_miss_capability.drop(columns=['Time_Taken_Mins'], inplace=True)
    # Store the info of district (including central hospital, ZMH) that fails
    appt_have_or_miss_capability.loc[:, 'fail_district'] = ''

    for i in appt_have_or_miss_capability.index:  # Loop through all appts
        # Get the info of app, level and officer_category
        # the_appt = appt_have_or_miss_capability.loc[i, 'Appt_Type_Code']
        the_level = appt_have_or_miss_capability.loc[i, 'Facility_Level']
        the_officer_category = appt_have_or_miss_capability.loc[i, 'Officer_Category']

        # Check in daily_capabilities that the required officer_category at a level is there or not, for every district
        # Store the info of district (including central hospital, ZMH) that fails
        if the_level in Facility_Levels[0:4]:  # Levels 0, 1a, 1b, 2
            k = 0  # Record how many districts fail
            for district in pop_districts:
                idx = capability[
                    (capability['District'] == district) &
                    (capability['Facility_Level'] == the_level) &
                    (capability['Officer_Category'] == the_officer_category)].index
                if idx.size == 0:
                    # Store the district that fails to provide required officer_category
                    appt_have_or_miss_capability.loc[i, 'fail_district'] = \
                        appt_have_or_miss_capability.loc[i, 'fail_district'] + district + ','
                    k += 1
            if k == 0:
                appt_have_or_miss_capability.loc[i, 'fail_district'] = 'All districts pass'
        elif the_level == '3':  # Levels 3 and 4 (Level 5 has no required service times)
            m = 0  # Record how many regions fail
            for region in pop_regions:
                idx1 = capability[
                    (capability['Region'] == region) &
                    (capability['Facility_Level'] == the_level) &
                    (capability['Officer_Category'] == the_officer_category)].index
                if idx1.size == 0:
                    # Store the regional hospital that fails
                    appt_have_or_miss_capability.loc[i, 'fail_district'] = \
                        appt_have_or_miss_capability.loc[i, 'fail_district'] + 'Referral Hospital_' + region + ','
                    m += 1
            if m == 0:
                appt_have_or_miss_capability.loc[i, 'fail_district'] = 'All districts pass'
        elif the_level == '4':  # Zomba Mental Hospital
            n = 0  # Record is ZMH failed
            idx2 = capability[
                (capability['Facility_Level'] == the_level) &
                (capability['Officer_Category'] == the_officer_category)].index
            if idx2.size == 0:
                appt_have_or_miss_capability.loc[i, 'fail_district'] = \
                    appt_have_or_miss_capability.loc[i, 'fail_district'] + 'Zomba Mental Hospital,'
                n += 1
            if n == 0:
                appt_have_or_miss_capability.loc[i, 'fail_district'] = 'All districts pass'
        else:
            assert 0 == 1  # There should be no 'else'; otherwise, the generated tables above is incorrect

    return appt_have_or_miss_capability


# Save results for funded
appt_have_or_miss_capability_funded = all_appts_can_run(fund_daily_capability_coarse)
appt_have_or_miss_capability_funded.to_csv(
    outputlocation / 'human_resources' / 'funded' / 'appt_have_or_miss_capability_funded.csv', index=False)

# Save results for actual
appt_have_or_miss_capability_actual = all_appts_can_run(curr_daily_capability_coarse)
appt_have_or_miss_capability_actual.to_csv(
    outputlocation / 'human_resources' / 'actual' / 'appt_have_or_miss_capability_actual.csv', index=False)
