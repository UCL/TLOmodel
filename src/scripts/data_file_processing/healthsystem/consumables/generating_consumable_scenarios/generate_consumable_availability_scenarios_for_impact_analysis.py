"""
This script adds estimates of availability of consumables under different scenarios to the base Resource File:

Outputs:
* Updated version of ResourceFile_Consumables_availability_small.csv (estimate of consumable available - smaller file for use in the
 simulation) which includes new consumable availability estimates for all scenarios

Inputs:
* outputs/regression_analysis/predictions/predicted_consumable_availability_regression_scenarios.csv - This file is hosted
locally after running consumable_resource_analyses_with_hhfa/regression_analysis/main.R
* ResourceFile_Consumables_availability_small.csv` - This file contains the original consumable availability estimates
from OpenLMIS 2018 data
* `ResourceFile_Consumables_matched.csv` - This file contains the crosswalk of HHFA consumables to consumables in the
TLO model

It creates one row for each consumable for availability at a specific facility and month when the data is extracted from
the OpenLMIS dataset and one row for each consumable for availability aggregated across all facilities when the data is
extracted from the Harmonised Health Facility Assessment 2018/19.

Consumable availability is measured as probability of consumable being available at any point in time.
"""
import calendar
import datetime
from collections import defaultdict
from pathlib import Path
import os

import matplotlib.pyplot as plt
from plotnine import * # ggplot, aes, geom_point for ggplots from R
import numpy as np
import pandas as pd

from tlo.methods.consumables import check_format_of_consumables_file

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    '/Users/sm2511/Dropbox/Thanzi la Onse'
)

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define a pathway to the data folder (note: currently outside the TLO model directory)
# remember to set working directory to TLOmodel/
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"

# 1. Import and clean data files
#**********************************
# 1.1 Import TLO model availability data
#------------------------------------------------------
tlo_availability_df = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv")
# Drop any scenario data previously included in the resourcefile
tlo_availability_df = tlo_availability_df[['Facility_ID', 'month', 'item_code', 'available_prop']]

# 1.1.1 Attach district, facility level, program to this dataset
#----------------------------------------------------------------
# Get TLO Facility_ID for each district and facility level
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}
tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    on = ['Facility_ID'], how='left')

# 1.1.2 Attach programs
programs = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")[['category', 'item_code', 'module_name']]
# TODO See if programs can be extracted from a different location as ResourceFile_Consumables_availability_and_usage.csv is now deprecated in master
programs = programs.drop_duplicates('item_code')
# manually add category for the two consumables for which it is missing
tlo_availability_df = tlo_availability_df.merge(programs, on = ['item_code'], how = 'left')

# 1.2 Import scenario data
#------------------------------------------------------
scenario_availability_df = pd.read_csv(outputfilepath / "regression_analysis/predictions/predicted_consumable_availability_regression_scenarios.csv")
scenario_availability_df = scenario_availability_df.drop(['Unnamed: 0'], axis=1)
scenario_availability_df = scenario_availability_df.rename({'item': 'item_hhfa'}, axis=1)

# Prepare scenario data to be merged to TLO model availability based on TLO model features
# 1.2.1 Level of care
#------------------------------------------------------
scenario_availability_df['fac_type'] = scenario_availability_df['fac_type_original'].str.replace("Facility_level_", "")

# 1.2.2 District
#------------------------------------------------------
# Do some mapping to make the Districts line-up with the definition of Districts in the model
rename_and_collapse_to_model_districts = {
    'Mzimba South': 'Mzimba',
    'Mzimba North': 'Mzimba',
}
scenario_availability_df = scenario_availability_df.rename({'district': 'district_original'}, axis=1)
scenario_availability_df['District'] = scenario_availability_df['district_original'].replace(rename_and_collapse_to_model_districts)

# Cities to get same results as their respective regions
copy_source_to_destination = {
    'Mzimba': 'Mzuzu City',
    'Lilongwe': 'Lilongwe City',
    'Zomba': 'Zomba City',
    'Blantyre': 'Blantyre City',
    'Nkhata Bay': 'Likoma' # based on anecdotal evidence, assume that they experience the same change in avaiability as a result of interventions based on regression results
}
for source, destination in copy_source_to_destination.items():
    new_rows = scenario_availability_df.loc[scenario_availability_df.District == source].copy() # standardised district names
    new_rows.District = destination
    scenario_availability_df = pd.concat([scenario_availability_df, new_rows], ignore_index = True)

assert sorted(set(districts)) == sorted(set(pd.unique(scenario_availability_df.District)))

# 1.2.3 Facility_ID
# #------------------------------------------------------
# Merge-in facility_id
scenario_availability_df = scenario_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['District', 'fac_type'],
                    right_on=['District', 'Facility_Level'], how='left', indicator=True)
scenario_availability_df = scenario_availability_df.rename({'_merge': 'merge_facid'}, axis=1)

# Extract list of District X Facility Level combinations for which there is no HHFA data
scenario_availability_df_test = scenario_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['District', 'Facility_Level'],
                    right_on=['District', 'Facility_Level'], how='right', indicator=True)
cond_no_1b = (scenario_availability_df_test['Facility_Level'].isin(['1b'])) & (scenario_availability_df_test['_merge'] == 'right_only')
cond_no_1a = (scenario_availability_df_test['Facility_Level'].isin(['1a'])) & (scenario_availability_df_test['_merge'] == 'right_only')
districts_with_no_scenario_data_for_1b = scenario_availability_df_test[cond_no_1b]['District'].unique()
districts_with_no_scenario_data_for_1a = scenario_availability_df_test[cond_no_1a]['District'].unique()
districts_with_no_scenario_data_for_1b_only = np.setdiff1d(districts_with_no_scenario_data_for_1b, districts_with_no_scenario_data_for_1a)

# According to HHFA data, Balaka, Machinga, Mwanza, Ntchisi and Salima do not have level 1b facilities
# Likoma was not included in the regression because of the limited variation within the district - only 4 facilities

# 1.2.4 Program
#------------------------------------------------------
map_model_programs_to_hhfa = {
    'contraception': 'contraception',
    'general': 'general',
    'reproductive_health': 'obs&newb',
    'road_traffic_injuries': 'surgical',
    'epi': 'epi',
    'neonatal_health': 'obs&newb',
    'other_childhood_illnesses': 'alri',
    'malaria': 'malaria',
    'tb': 'tb',
    'hiv': 'hiv',
    'undernutrition': 'child',
    'ncds': 'ncds',
    'cardiometabolicdisorders': 'ncds',
    'cancer': 'ncds',
}
# TODO Check if the above mapping is correct
# TODO collapse infection_prev and general in the HHFA-based predicted dataframe

scenario_availability_df['category_tlo'] = scenario_availability_df['program_plot'].replace(map_model_programs_to_hhfa)

# 1.2.5 Consumable/Item code and Category
#------------------------------------------------------
# Load TLO - HHFA consumable name crosswalk
consumable_crosswalk_df = pd.read_csv(path_for_new_resourcefiles / 'ResourceFile_consumables_matched.csv', encoding='ISO-8859-1')[['module_name', 'item_code', 'consumable_name_tlo',
'item_code_hhfa', 'item_hhfa', 'regression_application', 'notes_on_regression_application']]
# TODO Check that this crosswalk is complete
# TODO is module_name used?
# TODO add new consumables Rifapentine to this?

# Now merge in TLO item codes
scenario_availability_df = scenario_availability_df.reset_index(drop = True)
scenario_availability_df = scenario_availability_df.merge(consumable_crosswalk_df[['item_code', 'item_hhfa', 'regression_application', 'module_name']],
                    on = ['item_hhfa'], how='right', indicator=True, validate = "m:m")
scenario_availability_df = scenario_availability_df.drop_duplicates(['Facility_ID', 'item_code'])
scenario_availability_df = scenario_availability_df.rename({'_merge': 'merge_itemcode'}, axis=1)
print("Number of item codes from the TLO model for which no match was found in the regression-based scenario data = ", scenario_availability_df.merge_itemcode.value_counts()[1])

# Before merging the above dataframe with tlo_availability_df, and apply a general interpolation rule to fill any gaps,
# we need to make sure that any specific interpolation rules are applied to the scenario dataframe

# Further a row needs to be added for 1b level under Balaka, Machinga, Mwanza, Ntchisi and Salima
print("Number of unique facility IDs: \n - TLO consumable data = ", tlo_availability_df.Facility_ID.nunique(),
      "\n - Scenario based on regression = ", scenario_availability_df.Facility_ID.nunique(),
      "\nNumber of unique item codes: \n - TLO consumable availability data = ", tlo_availability_df.item_code.nunique(),
      "\n - TLO consumable availability repository = ", consumable_crosswalk_df.item_code.nunique(),
      "\n - Scenario based on regression = ", scenario_availability_df.item_code.nunique())

# Extract list of TLO consumables which weren't matched with the availability prediction dataframe
items_not_matched = scenario_availability_df['merge_itemcode'] == 'right_only'
scenario_availability_df[items_not_matched][['item_code', 'regression_application']].to_csv(outputfilepath / 'temp_items_not_matched.csv')

# Get average  availability_change_prop value by facility_ID and category_tlo
scenario_availability_df = scenario_availability_df.merge(programs[['category', 'item_code']],
                                            on = ['item_code'], validate = "m:1",
                                            how = "left")

# check that all consumables have a category assigned to them
map_items_with_missing_category_to_category= {77:'reproductive_health',
301:'alri',
63: 'neonatal_health',
258: 'cancer',
1735: 'general'}

# Update the category column based on item_code
scenario_availability_df['category'] = scenario_availability_df.apply(lambda row: map_items_with_missing_category_to_category[row['item_code']]
                          if row['item_code'] in map_items_with_missing_category_to_category
                          else row['category'], axis=1)

# 1.3 Initial interpolation
#------------------------------------------------------
# 1.3.1 Items not relevant to the regression analysis
items_not_relevant_to_regression = (items_not_matched) & (scenario_availability_df['regression_application'] == 'not relevant to logistic regression analysis')
# For category 3, replace availability_change_prop with 1, since we assume that the system-level intervention does not change availability
list_of_scenario_variables = ['change_proportion_scenario1', 'change_proportion_scenario2',
       'change_proportion_scenario3', 'change_proportion_scenario4', 'change_proportion_scenario5']
for var in list_of_scenario_variables:
    scenario_availability_df.loc[items_not_relevant_to_regression,var] = 1

# 1.3.2 For level 1b for the districts where this level was not present in the regression analysis/HHFA dataset, assume
# that the change is equal to the product of the (ratio of average change across districts for level 1b to
# average change across districts for level 1a) and change for each item_code for level 1a for that district
#------------------------------------------------------------------------------------------------------------
average_change_across_districts = scenario_availability_df.groupby(['Facility_Level','item_code'])[list_of_scenario_variables].mean().reset_index()

# Generate the ratio of the proportional changes to availability of level 1b to 1a in the districts for which level 1b data is available
new_colnames_1a = {col: col + '_1a' if col in list_of_scenario_variables else col for col in average_change_across_districts.columns}
new_colnames_1b = {col: col + '_1b' if col in list_of_scenario_variables else col for col in average_change_across_districts.columns}
average_change_across_districts_for_1a = average_change_across_districts[average_change_across_districts.Facility_Level == "1a"].rename(new_colnames_1a, axis = 1).drop('Facility_Level', axis = 1)
average_change_across_districts_for_1b = average_change_across_districts[average_change_across_districts.Facility_Level == "1b"].rename(new_colnames_1b, axis = 1).drop('Facility_Level', axis = 1)
ratio_of_change_across_districts_1b_to_1a = average_change_across_districts_for_1a.merge(average_change_across_districts_for_1b,
                                                                                         how = "left", on = ['item_code'])
for var in list_of_scenario_variables:
    var_ratio = 'ratio_' + var
    var_1a = var + '_1a'
    var_1b = var + '_1b'
    ratio_of_change_across_districts_1b_to_1a[var_ratio] = (ratio_of_change_across_districts_1b_to_1a[var_1b])/(ratio_of_change_across_districts_1b_to_1a[var_1a])
ratio_of_change_across_districts_1b_to_1a.reset_index(drop = True)
# TODO check if this ratio should be of the proportions minus 1

# Use the above for those districts with no level 1b facilities recorded in the HHFA data
scenario_availability_df = scenario_availability_df.reset_index(drop = True)
cond_1b_missing_districts = scenario_availability_df.District.isin(districts_with_no_scenario_data_for_1b_only)
cond_1a = scenario_availability_df.Facility_Level == '1a'
cond_1b = scenario_availability_df.Facility_Level == '1b'
df_1a = scenario_availability_df[cond_1b_missing_districts & cond_1a]

ratio_vars = ['ratio_' + item for item in list_of_scenario_variables]

item_var = ['item_code']
df_missing_1b_imputed = df_1a.merge(ratio_of_change_across_districts_1b_to_1a[item_var + ratio_vars],
                               on = ['item_code'],
                               how = 'left', validate = "m:1")

for var in list_of_scenario_variables:
    df_missing_1b_imputed[var] = df_missing_1b_imputed[var] * df_missing_1b_imputed['ratio_' + var]

df_missing_1b_imputed.Facility_Level = '1b' # Update facility level to 1
# Replace Facility_IDs
df_missing_1b_imputed = df_missing_1b_imputed.drop('Facility_ID', axis = 1).merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                                                                                     on =['District', 'Facility_Level'],
                                                                                     how = 'left')

df_without_districts_with_no_1b_facilities = scenario_availability_df[~(cond_1b_missing_districts & cond_1b)]
scenario_availability_df = pd.concat([df_without_districts_with_no_1b_facilities, df_missing_1b_imputed], ignore_index = True)

# 2. Merge TLO model availability data with scenario data using crosswalk
#*************************************************************************
# 2.1 Merge the two datasets
#------------------------------------------------------
id_variables = ['item_code','Facility_ID']

full_scenario_df = tlo_availability_df.merge(scenario_availability_df[id_variables + list_of_scenario_variables],
                               how='left', on=['Facility_ID', 'item_code'], indicator = True)
full_scenario_df = full_scenario_df.rename({'_merge': 'merge_scenario'}, axis=1)
full_scenario_df = full_scenario_df.drop_duplicates(['Facility_ID', 'item_code', 'month'])

# Check that level 1b values are currently imputed
# full_scenario_df[full_scenario_df.District == 'Balaka'].groupby(['District', 'Facility_Level'])['change_proportion_scenario1'].mean()

# 2.2 Further imputation
#------------------------------------------------------
# 2.2.1 For all levels other than 1a and 1b, there will be no change in consumable availability
#------------------------------------------------------------------------------------------------------------
fac_levels_not_relevant_to_regression = full_scenario_df.Facility_Level.isin(['0', '2', '3', '4'])

for var in list_of_scenario_variables:
    full_scenario_df.loc[fac_levels_not_relevant_to_regression, var] = 1

# 2.3 Final checks
#------------------------------------------------------
# 2.3.1 Check that the merged dataframe has the same number of unique items, facility IDs, and total
# number of rows as the original small availability resource file
#---------------------------------------------------------------------------------------------------------
assert(full_scenario_df.item_code.nunique() == tlo_availability_df.item_code.nunique())
assert(full_scenario_df.Facility_ID.nunique() == tlo_availability_df.Facility_ID.nunique())
assert(len(full_scenario_df) == len(tlo_availability_df))

# 2.3.2 Construct dataset that conforms to the principles expected by the simulation: i.e. that there is an entry for every
# facility_id and for every month for every item_code.
#-----------------------------------------------------------------------------------------------------------------------
# Generate the dataframe that has the desired size and shape
fac_ids = set(mfl.loc[mfl.Facility_Level != '5'].Facility_ID)
item_codes = set(tlo_availability_df.item_code.unique())
months = range(1, 13)
all_availability_columns = ['available_prop'] + list_of_scenario_variables

# Create a MultiIndex from the product of fac_ids, months, and item_codes
index = pd.MultiIndex.from_product([fac_ids, months, item_codes], names=['Facility_ID', 'month', 'item_code'])

# Initialize a DataFrame with the MultiIndex and columns, filled with NaN
full_set = pd.DataFrame(index=index, columns=all_availability_columns)
full_set = full_set.astype(float)  # Ensure all columns are float type and filled with NaN

# Insert the data, where it is available.
full_set = full_set.combine_first(full_scenario_df.set_index(['Facility_ID', 'month', 'item_code'])[all_availability_columns])

# Fill in the blanks with rules for interpolation.
facilities_by_level = defaultdict(set)
for ix, row in mfl.iterrows():
    facilities_by_level[row['Facility_Level']].add(row['Facility_ID'])

items_by_category = defaultdict(set)
for ix, row in programs.iterrows():
    items_by_category[row['category']].add(row['item_code'])

def get_other_facilities_of_same_level(_fac_id):
    """Return a set of facility_id for other facilities that are of the same level as that provided."""
    for v in facilities_by_level.values():
        if _fac_id in v:
            return v - {_fac_id}

def get_other_items_of_same_category(_item_code):
    """Return a set of item_codes for other items that are in the same category/program as that provided."""
    for v in items_by_category.values():
        if _item_code in v:
            return v - {_item_code}
def interpolate_missing_with_mean(_ser):
    """Return a series in which any values that are null are replaced with the mean of the non-missing."""
    if pd.isnull(_ser).all():
        raise ValueError
    return _ser.fillna(_ser.mean())

# Create new dataset that include the interpolations (The operation is not done "in place", because the logic is based
# on what results are missing before the interpolations in other facilities).
full_set_interpolated = full_set * np.nan
full_set_interpolated.available_prop = full_set.available_prop

for fac in fac_ids:
    for item in item_codes:
        for col in list_of_scenario_variables:
            print(f"Now doing: fac={fac}, item={item}, column={col}")

            # Get records of the availability of this item in this facility.
            _monthly_records = full_set.loc[(fac, slice(None), item), col].copy()

            if pd.notnull(_monthly_records).any():
                # If there is at least one record of this item at this facility, then interpolate the missing months from
                # the months for there are data on this item in this facility. (If none are missing, this has no effect).
                _monthly_records = interpolate_missing_with_mean(_monthly_records)

            else:
                # If there is no record of this item at this facility, check to see if it's available at other facilities
                # of the same level
                # Or if there is no record of item at other facilities at this level, check to see if other items of this category
                # are available at this facility level
                facilities = list(get_other_facilities_of_same_level(fac))

                other_items = get_other_items_of_same_category(item)
                items = list(other_items) if other_items else other_items

                recorded_at_other_facilities_of_same_level = pd.notnull(
                    full_set.loc[(facilities, slice(None), item), col]
                ).any()

                if not items:
                    category_recorded_at_other_facilities_of_same_level = False
                else:
                    category_recorded_at_other_facilities_of_same_level = pd.notnull(
                        full_set.loc[(fac, slice(None), items), col]
                    ).any()

                if recorded_at_other_facilities_of_same_level:
                    # If it recorded at other facilities of same level, find the average availability of the item at other
                    # facilities of the same level.
                    print("Data for facility ", fac, " extrapolated from other facilities within level - ", facilities)
                    facilities = list(get_other_facilities_of_same_level(fac))
                    _monthly_records = interpolate_missing_with_mean(
                        full_set.loc[(facilities, slice(None), item), col].groupby(level=1).mean()
                    )

                elif category_recorded_at_other_facilities_of_same_level:
                    # If it recorded at other facilities of same level, find the average availability of the item at other
                    # facilities of the same level.
                    print("Data for item ", item, " extrapolated from other items within category - ", items)
                    _monthly_records = interpolate_missing_with_mean(
                        full_set.loc[(fac, slice(None), items), col].groupby(level=1).mean()
                    )

                else:
                    # If it is not recorded at other facilities of same level, then assume it is never available at the
                    # facility.
                    print("No interpolation worked")
                    _monthly_records = _monthly_records.fillna(1.0)
# TODO this should be available_prop

            # Insert values (including corrections) into the resulting dataset.
            full_set_interpolated.loc[(fac, slice(None), item), col] = _monthly_records.values
            # temporary code
            assert full_set_interpolated.loc[(fac, slice(None), item), col].mean() >= 0

# 3. Generate regression-based scenario data on consumable availablity
#*************************************************************************
# Create new consumable availability estimates for TLO model consumables using
# estimates of proportional change from the regression analysis based on HHFA data
#------------------------------------------------------
prefix = 'change_proportion_'
list_of_scenario_suffixes = [s.replace(prefix, '') for s in list_of_scenario_variables]

for scenario in list_of_scenario_suffixes:
    full_set_interpolated['available_prop_' + scenario] = full_set_interpolated['available_prop'] * full_set_interpolated['change_proportion_' + scenario]
    availability_greater_than_1 = full_set_interpolated['available_prop_' + scenario] > 1
    full_set_interpolated.loc[availability_greater_than_1, 'available_prop_' + scenario] = 1

    assert(sum(full_set_interpolated['available_prop_' + scenario].isna()) ==
           sum(full_set_interpolated['change_proportion_' + scenario].isna()))

# 4. Generate best performing facility-based scenario data on consumable availablity
#*********************************************************************************************
df = full_set_interpolated.reset_index().copy()

# Try updating the avaiability to represent the 75th percentile by consumable
facility_levels = ['1a', '1b']
target_percentiles = [75, 90, 99]

# Populate the dictionary
for level in facility_levels:
    # Create an empty dictionary for the current level
    best_performing_facilities[level] = {}

    for item in item_codes:
        best_performing_facilities[level][item] = {}
        # Get the mean availability by Facility for the current level
        mean_consumable_availability = pd.DataFrame(df[(df.Facility_ID.isin(facilities_by_level[level])) & (df.item_code == item)]
                                                    .groupby('Facility_ID')['available_prop'].mean()).reset_index()

        # Calculate the percentile rank of each row for 'available_prop'
        mean_consumable_availability['percentile_rank'] = mean_consumable_availability['available_prop'].rank(pct=True) * 100

        # Find the row which is closest to the nth percentile rank for each target percentile
        for target_perc in target_percentiles:
            # Calculate the difference to target percentile
            mean_consumable_availability['diff_to_target_' + str(target_perc)] = np.abs(
                mean_consumable_availability['percentile_rank'] - target_perc)

            # Find the row with the minimum difference to the target percentile
            closest_row = mean_consumable_availability.loc[
                mean_consumable_availability['diff_to_target_' + str(target_perc)].idxmin()]

            # Store the Facility_ID of the closest row in the dictionary for the current level
            best_performing_facilities[level][item][str(target_perc) + 'th percentile'] = closest_row['Facility_ID']

print("Reference facilities at each level for each item: ", best_performing_facilities)
# TODO Flip the nesting order above for percentile to go before item?

# Obtain the updated availability estimates for level 1a for scenarios 6-8
updated_availability_1a = df[['item_code', 'month']].drop_duplicates()
updated_availability_1b = df[['item_code', 'month']].drop_duplicates()
temporary_df = pd.DataFrame([])
availability_dataframes = [updated_availability_1a, updated_availability_1b]

i = 6 # start scenario counter
j = 0 # start level counter
for level in facility_levels:
    for target_perc in target_percentiles:
        for item in item_codes:

            print("Running level ", level, "; Running scenario ", str(i), "; Running item ", item)
            reference_facility = df['Facility_ID'] == best_performing_facilities[level][item][str(target_perc) + 'th percentile']
            current_item = df['item_code'] == item
            availability_at_reference_facility = df[reference_facility & current_item][['item_code', 'month', 'available_prop']]

            if temporary_df.empty:
                temporary_df = availability_at_reference_facility
            else:
                temporary_df = pd.concat([temporary_df,availability_at_reference_facility], ignore_index = True)

        column_name = 'available_prop_scenario' + str(i)
        temporary_df = temporary_df.rename(columns = {'available_prop': column_name })
        availability_dataframes[j] = availability_dataframes[j].merge(temporary_df, on =  ['item_code', 'month'], how = 'left', validate = '1:1')
        temporary_df = pd.DataFrame([])
        i = i + 1
    i = 6 # restart scenario counter
    j = j + 1 # move to the next level

# Merge the above scenario data to the full availability scenario dataframe
# Scenario 6-8 availability data for level 1a
df_new_1a = df[df['Facility_ID'].isin(facilities_by_level['1a'])].merge(availability_dataframes[0],on = ['item_code', 'month'],
                                      how = 'left',
                                      validate = "m:1")
# Scenario 6-8 availability data for level 1b
df_new_1b = df[df['Facility_ID'].isin(facilities_by_level['1b'])].merge(availability_dataframes[1],on = ['item_code', 'month'],
                                      how = 'left',
                                      validate = "m:1")
# Scenario 6-8 availability data for other levels
df_new_otherlevels = df[~df['Facility_ID'].isin(facilities_by_level['1a']|facilities_by_level['1b'])]
new_scenario_columns = ['available_prop_scenario6', 'available_prop_scenario7', 'available_prop_scenario8']
for col in new_scenario_columns:
    df_new_otherlevels[col] = df_new_otherlevels['available_prop']

# Append the above dataframes
df_new = pd.concat([df_new_1a, df_new_1b, df_new_otherlevels], ignore_index = True)

# Save dataframe
#------------------------------------------------------
list_of_scenario_suffixes = list_of_scenario_suffixes + ['scenario6', 'scenario7', 'scenario8']
final_list_of_scenario_vars = ['available_prop_' + item for item in list_of_scenario_suffixes]
old_vars = ['Facility_ID', 'month', 'item_code', 'available_prop']
full_df_with_scenario = df_new[old_vars + final_list_of_scenario_vars].reset_index().drop('index', axis = 1)

# --- Check that the exported file has the properties required of it by the model code. --- #
check_format_of_consumables_file(df=full_df_with_scenario, fac_ids=fac_ids)

# Save updated consumable availability resource file with scenario data
full_df_with_scenario.to_csv(
    path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv",
    index=False
)
# TODO: Create a column providing the source of scenario data

# 8. Plot new availability estimates by scenario
#*********************************************************************************************
# Creating the line plot with ggplot
df_for_plots = full_df_with_scenario.merge(programs[['category', 'item_code']], on = 'item_code', how = "left", validate = "m:1")
df_for_plots = df_for_plots.merge(mfl[['Facility_ID', 'Facility_Level']], on = 'Facility_ID', how = 'left', validate = "m:1")
def generate_barplot_of_scenarios(_df, _x_axis_var, _filename):
    df_for_line_plot = _df.groupby([_x_axis_var])[['available_prop'] + final_list_of_scenario_vars].mean()
    df_for_line_plot = df_for_line_plot.reset_index().melt(id_vars=[_x_axis_var], value_vars=['available_prop'] + final_list_of_scenario_vars,
                        var_name='Scenario', value_name='Value')
    plot = (ggplot(df_for_line_plot.reset_index(), aes(x=_x_axis_var, y='Value', fill = 'Scenario'))
            + geom_bar(stat='identity', position='dodge')
            + ylim(0, 1)
            + labs(title = "Probability of availability across scenarios",
                   x=_x_axis_var,
                   y='Probability of availability')
            + theme(axis_text_x=element_text(angle=45, hjust=1))
           )
    # Create the directory if it doesn't exist
    directory = outputfilepath / 'consumable_scenario_analysis'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plot.save(filename= directory / _filename, dpi=300, width=10, height=8, units='in')
generate_barplot_of_scenarios(_df = df_for_plots, _x_axis_var = 'category', _filename = 'availability_by_category.png')
generate_barplot_of_scenarios(_df = df_for_plots, _x_axis_var = 'Facility_Level', _filename = 'availability_by_level.png')

# Scenario on the X axis, level on the Y axis
# Scenario on the X axis, program on the Y axis
# TODO add heat maps

'''
# 2.3.2. Browse missingness in the availability_change_prop variable
#------------------------------------------------------
pivot_table = pd.pivot_table(scenario_availability_df,
                             values=list_of_scenario_variables,
                             index=['category'],
                             columns=['Facility_Level'],
                             aggfunc=lambda x: sum(pd.isna(x))/len(x)*100)
pivot_table.to_csv(outputfilepath / "temp.csv")
print(pivot_table[('change_proportion_scenario5', '0')])

a = availability_dataframes[1].reset_index()
print(best_performing_facilities['1b'][5][str(75) + 'th percentile'])
print(best_performing_facilities['1b'][222][str(90) + 'th percentile'])
print(best_performing_facilities['1b'][222][str(99) + 'th percentile'])
a[a.item_code == 222][['month', 'available_prop_scenario8']]
item_chosen = 222
fac_chosen = 110
print(df[(df.item_code == item_chosen) & (df.Facility_ID == fac_chosen)][['month', 'available_prop']])

'''

