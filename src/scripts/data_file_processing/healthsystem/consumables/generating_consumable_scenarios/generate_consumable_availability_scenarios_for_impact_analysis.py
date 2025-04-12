"""
This script adds estimates of availability of consumables under different scenarios to the base Resource File:

Outputs:
* Updated version of ResourceFile_Consumables_availability_small.csv (estimate of consumable available - smaller file for use in the
 simulation) which includes new consumable availability estimates for all scenarios. The following scenarios are generated
1. 'default' : this is the benchmark scenario with 2018 levels of consumable availability
2. 'scenario1' : [Level 1a + 1b] All items perform as well as consumables other than drugs/diagnostic tests
3. 'scenario2' : [Level 1a + 1b] 1 + All items perform as well as consumables classified as 'Vital' in the Essential Medicines List
4. 'scenario3' : [Level 1a + 1b] 2 + All facilities perform as well as those in which consumables stock is managed by pharmacists
5. 'scenario4' : [Level 1a + 1b] 3 + Level 1a facilities perform as well as level 1b
6. 'scenario5' : [Level 1a + 1b] 4 + All facilities perform as well as CHAM facilities
7. 'scenario6' : [Level 1a + 1b] All facilities have the same probability of consumable availability as the 75th percentile best performing facility for each individual item
8. 'scenario7' : [Level 1a + 1b] All facilities have the same probability of consumable availability as the 90th percentile best performing facility for each individual item
9. 'scenario8' : [Level 1a + 1b] All facilities have the same probability of consumable availability as the 99th percentile best performing facility for each individual item
10. 'scenario9' : [Level 1a + 1b + 2] All facilities have the same probability of consumable availability as the 99th percentile best performing facility for each individual item
11. 'scenario10' : [Level 1a + 1b] All programs perform as well as HIV
12. 'scenario11' : [Level 1a + 1b] All programs perform as well as EPI
13. 'scenario12' : [Level 1a + 1b] HIV performs as well as other programs (excluding EPI)
14. 'all': all consumable are always available - provides the theoretical maximum health gains which can be made through improving consumable supply

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
import seaborn as sns
import numpy as np
import pandas as pd

from tlo.methods.consumables import check_format_of_consumables_file

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
tlo_availability_df = tlo_availability_df[['Facility_ID', 'month','item_code', 'available_prop']]

# Import item_category
program_item_mapping = pd.read_csv(path_for_new_resourcefiles  / 'ResourceFile_Consumables_Item_Designations.csv')[['Item_Code', 'item_category']]
program_item_mapping = program_item_mapping.rename(columns ={'Item_Code': 'item_code'})[program_item_mapping.item_category.notna()]

# 1.1.1 Attach district,  facility level and item_category to this dataset
#----------------------------------------------------------------
# Get TLO Facility_ID for each district and facility level
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}
tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    on = ['Facility_ID'], how='left')

tlo_availability_df = tlo_availability_df.merge(program_item_mapping,
                    on = ['item_code'], how='left')

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
# Do some mapping to make the Districts in the scenario file line-up with the definition of Districts in the model
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
df_to_check_prediction_completeness = scenario_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['District', 'Facility_Level'],
                    right_on=['District', 'Facility_Level'], how='right', indicator=True)
cond_no_1b = (df_to_check_prediction_completeness['Facility_Level'].isin(['1b'])) & (df_to_check_prediction_completeness['_merge'] == 'right_only')
cond_no_1a = (df_to_check_prediction_completeness['Facility_Level'].isin(['1a'])) & (df_to_check_prediction_completeness['_merge'] == 'right_only')
districts_with_no_scenario_data_for_1b = df_to_check_prediction_completeness[cond_no_1b]['District'].unique()
districts_with_no_scenario_data_for_1a = df_to_check_prediction_completeness[cond_no_1a]['District'].unique()
districts_with_no_scenario_data_for_1b_only = np.setdiff1d(districts_with_no_scenario_data_for_1b, districts_with_no_scenario_data_for_1a)

# According to HHFA data, Balaka, Machinga, Mwanza, Ntchisi and Salima do not have level 1b facilities
# Likoma was not included in the regression because of the limited variation within the district - only 4 facilities - we have assumed that the change of consumable
# availability in Likoma is equal to that predicted for Nkhata Bay

# 1.2.4 Program
#------------------------------------------------------
scenario_availability_df.loc[scenario_availability_df.program_plot == 'infection_prev', 'program_plot'] = 'general' # there is no separate infection_prevention category in the TLO availability data
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
# Reverse the map_model_programs_to_hhfa dictionary
hhfa_to_model_programs = {v: k for k, v in map_model_programs_to_hhfa.items()}

scenario_availability_df['category_tlo'] = scenario_availability_df['program_plot'].replace(hhfa_to_model_programs) # TODO this might not be relevant

# 1.2.5 Consumable/Item code and Category
#------------------------------------------------------
# Load TLO - HHFA consumable name crosswalk
consumable_crosswalk_df = pd.read_csv(path_for_new_resourcefiles / 'ResourceFile_consumables_matched.csv', encoding='ISO-8859-1')[['item_code', 'consumable_name_tlo',
'item_hhfa_for_scenario_generation', 'hhfa_mapping_rationale_for_scenario_generation']]

# Keep only item_codes in the availability dataframe
consumable_crosswalk_df = consumable_crosswalk_df.merge(tlo_availability_df[['item_code']], how = 'right', on = 'item_code')
# TODO is module_name used?
# TODO add new consumables Rifapentine to this?

# Now merge in TLO item codes
scenario_availability_df = scenario_availability_df.reset_index(drop = True)
scenario_availability_df = scenario_availability_df.merge(consumable_crosswalk_df[['item_code', 'item_hhfa_for_scenario_generation', 'hhfa_mapping_rationale_for_scenario_generation', 'consumable_name_tlo']],
                    left_on = ['item_hhfa'], right_on = ['item_hhfa_for_scenario_generation'], how='right', indicator=True, validate = "m:m")
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

# Get average  availability_change_prop value by facility_ID and category_tlo
scenario_availability_df = scenario_availability_df.merge(program_item_mapping,
                                            on = ['item_code'], validate = "m:1",
                                            how = "left")

# 1.3 Initial interpolation
#------------------------------------------------------
# 1.3.1 Items not relevant to the regression analysis
items_not_relevant_to_regression = (items_not_matched) & (scenario_availability_df['hhfa_mapping_rationale_for_scenario_generation'] == 'not relevant to logistic regression analysis')
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

# For districts with no level 1b data in the HHFA, use the ratio of change in level 1b facilities to level 1a facilities to generate the expected proportional change in availability
# for level 1b facilities in those districts
scenario_availability_df = scenario_availability_df.reset_index(drop = True)
cond_districts_with_1b_missing = scenario_availability_df.District.isin(districts_with_no_scenario_data_for_1b_only)
cond_1a = scenario_availability_df.Facility_Level == '1a'
cond_1b = scenario_availability_df.Facility_Level == '1b'
df_1a = scenario_availability_df[cond_districts_with_1b_missing & cond_1a]

ratio_vars = ['ratio_' + item for item in list_of_scenario_variables] # create columns to represent the ratio of change in 1b facilities to level 1a facilities

item_var = ['item_code']

# First merge the dataframe with changes at level 1a with the ratio of 1b to 1a
df_missing_1b_imputed = df_1a.merge(ratio_of_change_across_districts_1b_to_1a[item_var + ratio_vars],
                               on = item_var,
                               how = 'left', validate = "m:1")

# Then multiply the ratio of 1b to 1a with the change at level 1a to get the expected change at level 1b
for var in list_of_scenario_variables:
    df_missing_1b_imputed[var] = df_missing_1b_imputed[var] * df_missing_1b_imputed['ratio_' + var]
# Update columns so the dataframe in fact refers to level 1b facilities
df_missing_1b_imputed.Facility_Level = '1b' # Update facility level to 1
# Replace Facility_IDs
df_missing_1b_imputed = df_missing_1b_imputed.drop('Facility_ID', axis = 1).merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                                                                                     on =['District', 'Facility_Level'],
                                                                                     how = 'left')
# Append the new imputed level 1b dataframe to the original dataframe
df_without_districts_with_no_1b_facilities = scenario_availability_df[~(cond_districts_with_1b_missing & cond_1b)]
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
#full_scenario_df[full_scenario_df.District == 'Balaka'].groupby(['District', 'Facility_Level'])['change_proportion_scenario1'].mean()

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
assert(full_scenario_df.item_code.nunique() == tlo_availability_df.item_code.nunique()) # the number of items in the new dataframe is the same at the original availability dataframe
assert(full_scenario_df.Facility_ID.nunique() == tlo_availability_df.Facility_ID.nunique()) # the number of Facility IDs in the new dataframe is the same at the original availability dataframe
assert(len(full_scenario_df) == len(tlo_availability_df)) # the number of rows in the new dataframe is the same at the original availability dataframe

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
for ix, row in program_item_mapping.iterrows():
    items_by_category[row['item_category']].add(row['item_code'])

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
                    # If it is not recorded at other facilities of same level, then assume that there is no change
                    print("No interpolation worked")
                    _monthly_records = _monthly_records.fillna(1.0)

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
           sum(full_set_interpolated['change_proportion_' + scenario].isna())) # make sure that there is an entry for every row in which there was previously data

# 4. Generate best performing facility-based scenario data on consumable availability
#***************************************************************************************
df = full_set_interpolated.reset_index().copy()

# Try updating the avaiability to represent the 75th percentile by consumable
facility_levels = ['1a', '1b', '2']
target_percentiles = [75, 90, 99]

best_performing_facilities = {}
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

# Obtain the updated availability estimates for level 1a for scenarios 6-8
updated_availability_1a = df[['item_code', 'month']].drop_duplicates()
updated_availability_1b = df[['item_code', 'month']].drop_duplicates()
updated_availability_2 = df[['item_code', 'month']].drop_duplicates()
temporary_df = pd.DataFrame([])
availability_dataframes = [updated_availability_1a, updated_availability_1b, updated_availability_2]

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
# 75, 90 and 99th percentile availability data for level 1a
df_new_1a = df[df['Facility_ID'].isin(facilities_by_level['1a'])].merge(availability_dataframes[0],on = ['item_code', 'month'],
                                      how = 'left',
                                      validate = "m:1")
# 75, 90 and 99th percentile availability data for level 1b
df_new_1b = df[df['Facility_ID'].isin(facilities_by_level['1b'])].merge(availability_dataframes[1],on = ['item_code', 'month'],
                                      how = 'left',
                                      validate = "m:1")
# 75, 90 and 99th percentile availability data for level 2
df_new_2 = df[df['Facility_ID'].isin(facilities_by_level['2'])].merge(availability_dataframes[2],on = ['item_code', 'month'],
                                      how = 'left',
                                      validate = "m:1")

# Generate scenarios 6-8
#------------------------
# scenario 6: only levels 1a and 1b changed to availability at 75th percentile for the corresponding level
# scenario 7: only levels 1a and 1b changed to availability at 90th percentile for the corresponding level
# scenario 8: only levels 1a and 1b changed to availability at 99th percentile for the corresponding level
# Scenario 6-8 availability data for other levels
df_new_otherlevels = df[~df['Facility_ID'].isin(facilities_by_level['1a']|facilities_by_level['1b'])]
new_scenario_columns = ['available_prop_scenario6', 'available_prop_scenario7', 'available_prop_scenario8']
for col in new_scenario_columns:
    df_new_otherlevels[col] = df_new_otherlevels['available_prop']
# Append the above dataframes
df_new_scenarios6to8 = pd.concat([df_new_1a, df_new_1b, df_new_otherlevels], ignore_index = True)


# Generate scenario 9
#------------------------
# scenario 9: levels 1a, 1b and 2 changed to availability at 99th percentile for the corresponding level
df_new_otherlevels = df_new_scenarios6to8[~df_new_scenarios6to8['Facility_ID'].isin(facilities_by_level['1a']|facilities_by_level['1b']|facilities_by_level['2'])].reset_index(drop  = True)
df_new_1a_scenario9 =  df_new_scenarios6to8[df_new_scenarios6to8['Facility_ID'].isin(facilities_by_level['1a'])].reset_index(drop  = True)
df_new_1b_scenario9 =  df_new_scenarios6to8[df_new_scenarios6to8['Facility_ID'].isin(facilities_by_level['1b'])].reset_index(drop  = True)
df_new_2_scenario9 =  df_new_2[df_new_2['Facility_ID'].isin(facilities_by_level['2'])].reset_index(drop  = True)
new_scenario_columns = ['available_prop_scenario9']
for col in new_scenario_columns:
    df_new_otherlevels[col] = df_new_otherlevels['available_prop']
    df_new_1a_scenario9[col] = df_new_1a_scenario9['available_prop_scenario8']
    df_new_1b_scenario9[col] = df_new_1b_scenario9['available_prop_scenario8']
    df_new_2_scenario9[col] = df_new_2_scenario9['available_prop_scenario8']
# Append the above dataframes
df_new_scenarios9 = pd.concat([df_new_1a_scenario9, df_new_1b_scenario9, df_new_2_scenario9, df_new_otherlevels], ignore_index = True)

# 6. Generate scenarios based on the performance of vertical programs
#***************************************************************************************
cond_levels1a1b = (tlo_availability_df.Facility_Level == '1a') |(tlo_availability_df.Facility_Level == '1b')
cond_hiv = tlo_availability_df.item_category == 'hiv'
cond_epi = tlo_availability_df.item_category == 'epi'
#cond_not_hivorepi = (tlo_availability_df.item_category != 'hiv') & (tlo_availability_df.item_category != 'epi')
nonhivepi_availability_df = tlo_availability_df[(~cond_hiv & ~cond_epi) & cond_levels1a1b]
hivepi_availability_df = tlo_availability_df[(cond_hiv| cond_epi) & cond_levels1a1b]
irrelevant_levels_availability_df = tlo_availability_df[~cond_levels1a1b]

hiv_availability_df = tlo_availability_df[cond_hiv & cond_levels1a1b].groupby(['Facility_ID', 'month', 'item_category'])['available_prop'].mean().reset_index()
hiv_availability_df = hiv_availability_df.rename(columns = {'available_prop': 'available_prop_scenario10'})
hivepi_availability_df['available_prop_scenario10'] = hivepi_availability_df['available_prop']
irrelevant_levels_availability_df['available_prop_scenario10'] = irrelevant_levels_availability_df['available_prop']
minimum_scenario_varlist = ['Facility_ID', 'month', 'item_code', 'available_prop_scenario10']
hiv_scenario_df = nonhivepi_availability_df.merge(hiv_availability_df, on = ['Facility_ID', 'month'] , how = 'left', validate = 'm:1')
hiv_scenario_df = pd.concat([hiv_scenario_df[minimum_scenario_varlist], hivepi_availability_df[minimum_scenario_varlist], irrelevant_levels_availability_df[minimum_scenario_varlist]], ignore_index = True)

epi_availability_df = tlo_availability_df[cond_epi].groupby(['Facility_ID', 'month', 'item_category'])['available_prop'].mean().reset_index()
epi_availability_df = epi_availability_df.rename(columns = {'available_prop': 'available_prop_scenario11'})
hivepi_availability_df['available_prop_scenario11'] = hivepi_availability_df['available_prop']
irrelevant_levels_availability_df['available_prop_scenario11'] = irrelevant_levels_availability_df['available_prop']
epi_scenario_df = nonhivepi_availability_df.merge(epi_availability_df, on = ['Facility_ID', 'month'] , how = 'left', validate = 'm:1')
minimum_scenario_varlist = ['Facility_ID', 'month', 'item_code', 'available_prop_scenario11']
epi_scenario_df = nonhivepi_availability_df.merge(epi_availability_df, on = ['Facility_ID', 'month'] , how = 'left', validate = 'm:1')
epi_scenario_df = pd.concat([epi_scenario_df[minimum_scenario_varlist], hivepi_availability_df[minimum_scenario_varlist], irrelevant_levels_availability_df[minimum_scenario_varlist]], ignore_index = True)

# 7. Generate a scenario to represent HIV availability falling to that of other programs
#***************************************************************************************
nonhivepi_availability_average = tlo_availability_df[(~cond_hiv & ~cond_epi)].groupby(['Facility_ID', 'month'])['available_prop'].mean().reset_index()
nonhiv_availability_df = tlo_availability_df[~cond_hiv]
non_vertical_hiv_availability_df = tlo_availability_df[cond_hiv]
nonhivepi_availability_average = nonhivepi_availability_average.rename(columns = {'available_prop':'available_prop_scenario12'})
nonhiv_availability_df['available_prop_scenario12'] = nonhiv_availability_df['available_prop']
non_vertical_hiv_availability_df = non_vertical_hiv_availability_df.merge(nonhivepi_availability_average, on = ['Facility_ID', 'month'],  how = 'left', validate = 'm:1')
minimum_scenario_varlist = ['Facility_ID', 'month', 'item_code', 'available_prop_scenario12']
non_vertical_hiv_scenario_df = pd.concat([non_vertical_hiv_availability_df[minimum_scenario_varlist], nonhiv_availability_df[minimum_scenario_varlist]], ignore_index = True)

# Add scenarios 6 to 11 to the original dataframe
#------------------------------------------------------
list_of_scenario_suffixes_first_stage = list_of_scenario_suffixes + ['scenario6', 'scenario7', 'scenario8', 'scenario9']
list_of_scenario_variables_first_stage = ['available_prop_' + item for item in list_of_scenario_suffixes_first_stage]
old_vars = ['Facility_ID', 'month', 'item_code']
full_df_with_scenario = df_new_scenarios6to8[old_vars + ['available_prop'] + [col for col in list_of_scenario_variables_first_stage if col != 'available_prop_scenario9']].reset_index().drop('index', axis = 1)
full_df_with_scenario = full_df_with_scenario.merge(df_new_scenarios9[old_vars + ['available_prop_scenario9']], on = old_vars, how = 'left', validate = "1:1")

list_of_scenario_suffixes_second_stage = list_of_scenario_suffixes_first_stage + ['scenario10', 'scenario11', 'scenario12']
final_list_of_scenario_vars = ['available_prop_' + item for item in list_of_scenario_suffixes_second_stage]
full_df_with_scenario = full_df_with_scenario.merge(hiv_scenario_df[old_vars + ['available_prop_scenario10']], on = old_vars, how = 'left', validate = "1:1")
full_df_with_scenario = full_df_with_scenario.merge(epi_scenario_df[old_vars + ['available_prop_scenario11']], on = old_vars, how = 'left', validate = "1:1")
full_df_with_scenario = full_df_with_scenario.merge(non_vertical_hiv_scenario_df[old_vars + ['available_prop_scenario12']], on = old_vars, how = 'left', validate = "1:1")

#full_df_with_scenario = full_df_with_scenario.merge(program_item_mapping, on = 'item_code', validate = 'm:1', how = 'left')

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
# Create the directory if it doesn't exist
figurespath = outputfilepath / 'consumable_scenario_analysis'
if not os.path.exists(figurespath):
    os.makedirs(figurespath)

# Creating the line plot with ggplot
df_for_plots = full_df_with_scenario.merge(mfl[['Facility_ID', 'Facility_Level']], on = 'Facility_ID', how = 'left', validate = "m:1")
df_for_plots = df_for_plots.merge(program_item_mapping, on = 'item_code', how = 'left', validate = "m:1")
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

    plot.save(filename= figurespath / _filename, dpi=300, width=10, height=8, units='in')
generate_barplot_of_scenarios(_df = df_for_plots, _x_axis_var = 'item_category', _filename = 'availability_by_category.png')
generate_barplot_of_scenarios(_df = df_for_plots, _x_axis_var = 'Facility_Level', _filename = 'availability_by_level.png')

# Create heatmaps by Facility_Level of average availability by item_category across chosen scenarios
number_of_scenarios = 12
availability_columns = ['available_prop'] + [f'available_prop_scenario{i}' for i in
                                             range(1, number_of_scenarios + 1)]

for level in fac_levels:
    # Generate a heatmap
    # Pivot the DataFrame
    aggregated_df = df_for_plots.groupby(['item_category', 'Facility_Level'])[availability_columns].mean().reset_index()
    aggregated_df = aggregated_df[aggregated_df.Facility_Level.isin([level])]
    heatmap_data = aggregated_df.set_index('item_category').drop(columns = 'Facility_Level')

    # Calculate the aggregate row and column
    aggregate_col= aggregated_df[availability_columns].mean()
    #overall_aggregate = aggregate_col.mean()

    # Add aggregate row and column
    #heatmap_data['Average'] = aggregate_row
    #aggregate_col['Average'] = overall_aggregate
    heatmap_data.loc['Average'] = aggregate_col

    # Generate the heatmap
    sns.set(font_scale=0.5)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', cbar_kws={'label': 'Proportion of days on which consumable is available'})

    # Customize the plot
    plt.title(f'Facility Level {level}')
    plt.xlabel('Scenarios')
    plt.ylabel(f'Disease/Public health \n program')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.savefig(figurespath /f'consumable_availability_heatmap_{level}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Create heatmap of average availability by Facility_Level across chosen scenarios
scenario_list = [1,2,3,6,7,8,9,10,11]
chosen_availability_columns = ['available_prop'] + [f'available_prop_scenario{i}' for i in
                                             scenario_list]
# Pivot the DataFrame
aggregated_df = df_for_plots.groupby(['Facility_Level'])[chosen_availability_columns].mean().reset_index()
heatmap_data = aggregated_df.set_index('Facility_Level')

# Calculate the aggregate row and column
aggregate_col= aggregated_df[chosen_availability_columns].mean()
#overall_aggregate = aggregate_col.mean()

# Add aggregate row and column
#heatmap_data['Average'] = aggregate_row
#aggregate_col['Average'] = overall_aggregate
heatmap_data.loc['Average'] = aggregate_col

# Generate the heatmap
sns.set(font_scale=0.5)
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', cbar_kws={'label': 'Proportion of days on which consumable is available'})

# Customize the plot
plt.title(f'Availability across scenarios')
plt.xlabel('Scenarios')
plt.ylabel(f'Facility Level')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.savefig(figurespath /f'consumable_availability_heatmap_alllevels.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Create heatmap of average availability by Facility_Level just showing scenario 12
scenario_list = [12]
chosen_availability_columns = ['available_prop'] + [f'available_prop_scenario{i}' for i in
                                             scenario_list]
# Pivot the DataFrame
df_for_hiv_plot = df_for_plots
df_for_hiv_plot['hiv_or_other'] = np.where(df_for_hiv_plot['item_category'] == 'hiv', 'hiv', 'other programs')

aggregated_df = df_for_hiv_plot.groupby(['Facility_Level', 'hiv_or_other'])[chosen_availability_columns].mean().reset_index()
aggregated_df = aggregated_df.rename(columns = {'available_prop': 'Actual', 'available_prop_scenario12': 'HIV moved to Govt supply chain'})
heatmap_data = aggregated_df.pivot_table(index=['Facility_Level'],  # Keep other relevant columns in the index
                                      columns='hiv_or_other',
                                      values=['Actual', 'HIV moved to Govt supply chain'])
# Generate the heatmap
sns.set(font_scale=1)
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', cbar_kws={'label': 'Proportion of days on which consumable is available'})

# Customize the plot
plt.title(f'Availability across scenarios')
plt.xlabel('Scenarios')
plt.ylabel(f'Facility Level')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.savefig(figurespath /f'consumable_availability_heatmap_hiv_v_other.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# Scenario on the X axis, level on the Y axis
# Scenario on the X axis, program on the Y axis
# TODO add heat maps i. heatmap by item_category across the sceanrios

'''
# 2.3.2. Browse missingness in the availability_change_prop variable
#------------------------------------------------------
pivot_table = pd.pivot_table(scenario_availability_df,
                             values=list_of_scenario_variables,
                             index=['item_category'],
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

