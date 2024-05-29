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
scenario_availability_df['district'] = scenario_availability_df['district_original'].replace(rename_and_collapse_to_model_districts)

# Cities to get same results as their respective regions
copy_source_to_destination = {
    'Mzimba': 'Mzuzu City',
    'Lilongwe': 'Lilongwe City',
    'Zomba': 'Zomba City',
    'Blantyre': 'Blantyre City',
    'Nkhata Bay': 'Likoma' # based on anecdotal evidence, assume that they experience the same change in avaiability as a result of interventions based on regression results
}
for source, destination in copy_source_to_destination.items():
    new_rows = scenario_availability_df.loc[scenario_availability_df.district == source].copy() # standardised district names
    new_rows.district = destination
    scenario_availability_df = pd.concat([scenario_availability_df, new_rows], ignore_index = True)

# 1.2.3 Facility_ID
# #------------------------------------------------------
# Merge-in facility_id
scenario_availability_facid_merge = scenario_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['district', 'fac_type'],
                    right_on=['District', 'Facility_Level'], how='left', indicator=True)
scenario_availability_facid_merge = scenario_availability_facid_merge.rename({'_merge': 'merge_facid'}, axis=1)

# Extract list of District X Facility Level combinations for which there is no HHFA data
scenario_availability_df_test = scenario_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['district', 'fac_type'],
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
# TODO check how infection_prev should be mapped

scenario_availability_facid_merge['category_tlo'] = scenario_availability_facid_merge['program_plot'].replace(map_model_programs_to_hhfa)

# 1.2.5 Consumable/Item code
#------------------------------------------------------
# Load TLO - HHFA consumable name crosswalk
consumable_crosswalk_df = pd.read_csv(path_for_new_resourcefiles / 'ResourceFile_consumables_matched.csv', encoding='ISO-8859-1')[['module_name', 'item_code', 'consumable_name_tlo',
'item_code_hhfa', 'item_hhfa', 'regression_application', 'notes_on_regression_application']]
# TODO Check that this crosswalk is complete
# TODO is module_name used?
# TODO add new consumables Rifapentine to this?

# Now merge in TLO item codes
scenario_availability_facid_merge = scenario_availability_facid_merge.reset_index().drop(['index'], axis=1)
scenario_availability_facid_itemcode_merge = scenario_availability_facid_merge.merge(consumable_crosswalk_df[['item_code', 'item_hhfa', 'regression_application', 'module_name']],
                    on = ['item_hhfa'], how='right', indicator=True, validate = "m:m")
scenario_availability_facid_itemcode_merge = scenario_availability_facid_itemcode_merge.drop_duplicates(['Facility_ID', 'item_code'])
#scenario_availability_facid_itemcode_merge.to_csv(outputfilepath / "temp_merge_status.csv")
scenario_availability_facid_itemcode_merge = scenario_availability_facid_itemcode_merge.rename({'_merge': 'merge_itemcode'}, axis=1)
print("Number of item codes from the TLO model for which no match was found in the regression-based scenario data = ", scenario_availability_facid_itemcode_merge.merge_itemcode.value_counts()[1])

# Before merging the above dataframe with tlo_availability_df, we need to duplicate rows for each of the unmatched items
# to represent all facility IDs

# Further a row needs to be added for 1b level under Balaka, Machinga, Mwanza, Ntchisi and Salima
print("Number of unique facility IDs: \n - TLO consumable data = ", tlo_availability_df.Facility_ID.nunique(),
      "\n - Scenario based on regression = ", scenario_availability_facid_itemcode_merge.Facility_ID.nunique(),
      "\nNumber of unique item codes: \n - TLO consumable availability data = ", tlo_availability_df.item_code.nunique(),
      "\n - TLO consumable availability repository = ", consumable_crosswalk_df.item_code.nunique(),
      "\n - Scenario based on regression = ", scenario_availability_facid_itemcode_merge.item_code.nunique())

# 1.2.6 Interpolation/Imputation where data is missing
#------------------------------------------------------
# 1.2.6.1 Facility IDs not matched
#------------------------------------------------------
# Before merging the scenario dataframe with tlo_availability_df, generate rows with all 57 relevant facility IDs for item_codes
# which are not matched
df = scenario_availability_facid_itemcode_merge
df_missing_facids = df.loc[df['Facility_ID'].isna()].reset_index()
df_missing_facids = df_missing_facids.drop_duplicates('item_code') # These item_codes don't have separate rows by
# Facility_ID because they were not found in the HHFA regression analysis

df_with_facids = df.loc[~df['Facility_ID'].isna()] # these are rows with Facility_ID

# Create a cartesian product of unique values in columns 'item_code' and 'Facility_ID' from both dataframes
df_with_all_facids = pd.DataFrame({'Facility_ID': np.repeat(df_with_facids['Facility_ID'].unique(), df_missing_facids['item_code'].nunique()),
                       'item_code': np.tile(df_missing_facids['item_code'].unique(), df_with_facids['Facility_ID'].nunique())})
# Merge df_new with df_missing on columns 'item_code' and 'Facility_ID'
df_with_all_facids = df_with_all_facids[['Facility_ID', 'item_code']].merge(df_missing_facids.drop('Facility_ID', axis = 1), on=['item_code'], how='left', validate = "m:1")
# df_new = df_new.append(df_existing)
df_with_all_facids = pd.concat([df_with_all_facids, df_with_facids], ignore_index = True)
# Now all item_codes have all Facility_IDs included in the HHFA regression analysis but no values for scenarios

scenario_final_df = df_with_all_facids
#len(scenario_final_df[scenario_final_df['change_proportion_scenario1'].isna()])

# Now provide scenario data for rows where this data is not available
# 1.2.6.1 Extract list of TLO consumables which weren't matched with the availability prediction dataframe
#scenario_availability_facid_itemcode_merge[items_not_matched][['item_code', 'regression_application']].to_csv(outputfilepath / 'temp_items_not_matched.csv')

# Get average  availability_change_prop value by facility_ID and category_tlo
scenario_final_df = scenario_final_df.merge(programs[['category', 'item_code']],
                                            on = ['item_code'], validate = "m:1",
                                            how = "left")
list_of_scenario_variables = ['change_proportion_scenario1', 'change_proportion_scenario2',
       'change_proportion_scenario3', 'change_proportion_scenario4', 'change_proportion_scenario5']
scenario_averages_by_program_and_facid = scenario_final_df.groupby(['Facility_ID','category'])[list_of_scenario_variables].mean().reset_index()

# check that all consumables have a category assigned to them
map_items_with_missing_category_to_category= {77:'reproductive_health',
301:'alri',
63: 'neonatal_health',
258: 'cancer',
1735: 'general'}
# Update the category column based on item_code
scenario_final_df['category'] = scenario_final_df.apply(lambda row: map_items_with_missing_category_to_category[row['item_code']]
                          if row['item_code'] in map_items_with_missing_category_to_category
                          else row['category'], axis=1)

# a. Some consumables which were matched with a corresponding HHFA item do not appear in the above dataset because
# they were excluded from the regression analysis due to missing availability information on them
# from most facilities. We will assume that their availability changes as per the average overall change
# (eg. diazepam, morphine, atropine)
items_not_matched = scenario_final_df['merge_itemcode'] == 'right_only'
scenario_final_df[items_not_matched]['regression_application'].unique()

unmatched_items_category1 = (items_not_matched) & ((scenario_final_df['regression_application'].isna()) | \
                                                   (scenario_final_df['regression_application'] == 'proxy'))

# b.'assume average'
unmatched_items_category2 = (items_not_matched) & (scenario_final_df['regression_application'] == 'assume average')

# Replace missing instances with the above average values for categories 1 and 2
scenario_cat1_and_cat2 = scenario_final_df[(unmatched_items_category1|unmatched_items_category2)]
scenario_cat1_and_cat2 = scenario_cat1_and_cat2.drop(list_of_scenario_variables, axis = 1)

scenario_cat1_and_cat2 = scenario_cat1_and_cat2.merge(scenario_averages_by_program_and_facid,
                                                      on = ['Facility_ID','category'], validate = "m:1",
                                                      how = "left")
for var in list_of_scenario_variables:
    var_imputed = var + '_imputed'
    scenario_cat1_and_cat2[var_imputed] = scenario_cat1_and_cat2[var]
    scenario_final_df = scenario_final_df.merge(scenario_cat1_and_cat2[['Facility_ID','item_code', var_imputed]],
                                                          on = ['Facility_ID','item_code'], validate = "1:1",
                                                          how = "left")
    scenario_final_df.loc[(unmatched_items_category1|unmatched_items_category2), var] = scenario_final_df[var_imputed]

# c. if category is missing, take average across items
cond_missing_category = scenario_final_df.category.isna()
cond_missing_scenario_data = scenario_final_df.change_proportion_scenario1.isna()
print("The following items don't have an appropriate category assigned for imputation - ", scenario_final_df[cond_missing_category & cond_missing_scenario_data]['item_code'].unique())
for var in list_of_scenario_variables:
    scenario_final_df.loc[cond_missing_category & cond_missing_scenario_data, var] = scenario_final_df[var].mean()

# d. 'not relevant to logistic regression analysis'
unmatched_items_category3 = (items_not_matched) & (scenario_final_df['regression_application'] == 'not relevant to logistic regression analysis')
# For category 3, replace availability_change_prop with 1, since we assume that the system-level intervention does not change availability
for var in list_of_scenario_variables:
    scenario_final_df.loc[unmatched_items_category3,var] = 1

# e. any other categories of unmatched consumables
unmatched_items_category4 = ~unmatched_items_category1 & ~unmatched_items_category2 & ~unmatched_items_category3 & items_not_matched
assert(sum(unmatched_items_category4) == 0) # check that we haven't missed any consumables

# 1.2.6.2 Inf values
#------------------------------------------------------
# Where the values are Inf because the availability changed from 0 to X, replace with the average for the category and Facility_ID
for var in list_of_scenario_variables:
    print(f"Running scenario {var}")
    change_value_is_infinite = scenario_final_df[var].isin([np.inf])
    print("Number of Inf values changed to average = ", sum(change_value_is_infinite))
'''
    # None of the values is infinite so we don't have to run the code below
    df_inf = scenario_final_df[change_value_is_infinite].reset_index()

    average_change_across_category_and_facid = scenario_final_df[~change_value_is_infinite].groupby(['Facility_ID','category'])[var].mean().reset_index()

    df_inf_replaced = df_inf.drop(var, axis = 1).merge(average_change_across_category_and_facid,
                                   on = ['Facility_ID','category'],
                                   how = 'left', validate = "m:1")
    scenario_final_df = pd.concat([scenario_final_df[~change_value_is_infinite], df_inf_replaced], ignore_index = True)
'''

'''
# 1.2.6.3 Assume that the availability at level 1b in districts (Balaka, Salima, Ntchisi, Mwanza, Machinga) changes in the same proportion as the average across all
# districts at level 1b
cond1 = ~scenario_final_df.District.isin(districts_with_no_scenario_data_for_1b_only)
cond2 = scenario_final_df.Facility_Level == '1b'
scenario_averages_by_item_for_level1b = scenario_final_df[cond1 & cond2].groupby(['item_code'])[list_of_scenario_variables].mean().reset_index()
dfs = []
for dist in districts_with_no_scenario_data_for_1b_only:
    df_copy = scenario_averages_by_item_for_level1b.copy()
    df_copy['district'] = dist
    dfs.append(df_copy)

result_df = pd.concat(dfs, ignore_index=True)
result_df['Facility_Level'] = '1b'
result_df = result_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']], left_on = ['district', 'Facility_Level'], right_on = ['District', 'Facility_Level'], how = 'left')
scenario_final_df = pd.concat([scenario_final_df, result_df], ignore_index=True)

'''
# TODO There are still some items missing for some facility IDs

# 2. Merge TLO model availability data with scenario data using crosswalk
#*************************************************************************
# 2.1 Merge the two datasets
#------------------------------------------------------
id_variables = ['item_code','Facility_ID','regression_application']

new_availability_df = tlo_availability_df.merge(scenario_final_df[id_variables + list_of_scenario_variables],
                               how='left', on=['Facility_ID', 'item_code'], indicator = True)
new_availability_df = new_availability_df.rename({'_merge': 'merge_scenario'}, axis=1)
new_availability_df = new_availability_df.drop_duplicates(['Facility_ID', 'item_code', 'month'])

# 2.2 Further imputation
#------------------------------------------------------
# 2.2.1 For level 1b for the districts where this level was not present in the regression analysis/HHFA dataset, assume
# that the change is equal to the product of the (ratio of average change across districts for level 1b to
# average change across districts for level 1a) and change for each item_code for level 1a for that district
#------------------------------------------------------
average_change_across_districts = scenario_final_df.groupby(['Facility_Level','item_code'])[list_of_scenario_variables].mean().reset_index()

#average_change_across_districts = scenario_final_df.groupby(['Facility_Level','item_code']).agg({'availability_change_prop': lambda x: x.mean(skipna = False)}).reset_index()
new_colnames_1a = {col: col + '_1a' if col in list_of_scenario_variables else col for col in average_change_across_districts.columns}
new_colnames_1b = {col: col + '_1b' if col in list_of_scenario_variables else col for col in average_change_across_districts.columns}
average_change_across_districts_for_1a = average_change_across_districts[average_change_across_districts.Facility_Level == "1a"].rename(new_colnames_1a, axis = 1).drop('Facility_Level', axis = 1)
average_change_across_districts_for_1b = average_change_across_districts[average_change_across_districts.Facility_Level == "1b"].rename(new_colnames_1b, axis = 1).drop('Facility_Level', axis = 1)
ratio_of_change_across_districts_1b_to_1a = average_change_across_districts_for_1a.merge(average_change_across_districts_for_1b,
                                                                                         how = "left", on = ['item_code'])
# START HERE
for var in list_of_scenario_variables:
    var_ratio = 'ratio_' + var
    var_1a = var + '_1a'
    var_1b = var + '_1b'
    ratio_of_change_across_districts_1b_to_1a[var_ratio] = (ratio_of_change_across_districts_1b_to_1a[var_1b]-1)/(ratio_of_change_across_districts_1b_to_1a[var_1a] - 1)
ratio_of_change_across_districts_1b_to_1a.reset_index()

# Use the above for those districts no level 1b facilities recorded in the HHFA data
cond_1b_missing_district = new_availability_df.District.isin(districts_with_no_scenario_data_for_1b_only)
cond_1b = new_availability_df.Facility_Level == '1b'
cond_1a = new_availability_df.Facility_Level == '1a'
df_missing_1b = new_availability_df[cond_1b_missing_district & cond_1b]
df_1a = new_availability_df[cond_1b_missing_district & cond_1a]

ratio_vars = ['ratio_' + item for item in list_of_scenario_variables]
item_var = ['item_code']
df_missing_1b_imputed = df_missing_1b.merge(ratio_of_change_across_districts_1b_to_1a[item_var + ratio_vars],
                               on = ['item_code'],
                               how = 'left', validate = "m:1")
for var in list_of_scenario_variables:
    # check that the values we are replacing are in fact missing
    assert np.isnan(df_missing_1b_imputed[var].unique()).all()

id_vars_level1a = ['District', 'item_code', 'month']
df_missing_1b_imputed = df_missing_1b_imputed.drop(list_of_scenario_variables, axis = 1).merge(df_1a[id_vars_level1a + list_of_scenario_variables],
                               on = id_vars_level1a,
                               how = 'left', validate = "1:1", indicator = True)

for var in list_of_scenario_variables:
    df_missing_1b_imputed[var] = ((df_missing_1b_imputed[var]-1) * df_missing_1b_imputed['ratio_' + var]) + 1

new_availability_df_imputed = pd.concat([new_availability_df[~(cond_1b_missing_district & cond_1b)], df_missing_1b_imputed], ignore_index = True)

# 2.2.2 For all levels other than 1a and 1b, there will be no change in consumable availability
#------------------------------------------------------
fac_levels_not_relevant_to_regression = new_availability_df_imputed.Facility_Level.isin(['0', '2', '3', '4'])
new_availability_df_imputed.loc[fac_levels_not_relevant_to_regression, 'availability_change_prop'] = 1

# 2.3 Final checks
#------------------------------------------------------
# 2.3.1 Check that the merged dataframe has the same number of unique items, facility IDs, and total
# number of rows as the original small availability resource file
#------------------------------------------------------
assert(new_availability_df_imputed.item_code.nunique() == tlo_availability_df.item_code.nunique())
assert(new_availability_df_imputed.Facility_ID.nunique() == tlo_availability_df.Facility_ID.nunique())
assert(len(new_availability_df_imputed) == len(tlo_availability_df))

# 2.3.2. Browse missingness in the availability_change_prop variable
#------------------------------------------------------
pivot_table = pd.pivot_table(new_availability_df_imputed,
                             values=list_of_scenario_variables,
                             index=['category'],
                             columns=['Facility_Level'],
                             aggfunc=lambda x: sum(pd.isna(x))/len(x)*100)
pivot_table.to_csv(outputfilepath / "temp.csv")
print(pivot_table[('change_proportion_scenario5', '1b')])
'''
Cases which are still missing data:
1. For the 5 districts without 1b facilities in HHFA (Balaka, Machinga, Mwanza, Ntchisi, Salima), data on 54 items
for 1b is missing.
2. Chitipa 1b is missing data on 5 consumables - 176, 177,178,179,181,192, 2675
3. 184	187 are missing for nearly all districts

Previously, Likoma did not have data from the regression analysis - I have now used values from Nkhata bay as proxy
'''

# 2.2.4 PLaceholder code to replace all other missing values
# TODO Check why there are still instances of missing data when regression_application is assume average or proxy
#------------------------------------------------------
# For all other cases, assume no change
for var in list_of_scenario_variables:
    missing_change_data = new_availability_df_imputed[var].isna()
    new_availability_df_imputed.loc[missing_change_data, var] = new_availability_df_imputed[var].mean()

# Where the merge_scenario == "left_only", we need to provide data on "availability_change_prop"
# new_availability_df_imputed.to_csv(outputfilepath / 'current_status_of_scenario_merge.csv')
# TODO Check why the city and district both have an instance of facility level 2

# 3. Generate scenario data on consumable availablity
#------------------------------------------------------
# Create new consumable availability estimates for TLO model consumables using
# estimates of proportional change from the regression analysis based on HHFA data
#------------------------------------------------------
prefix = 'change_proportion_'
list_of_scenario_suffixes = [s.replace(prefix, '') for s in list_of_scenario_variables]

for scenario in list_of_scenario_suffixes:
    new_availability_df_imputed['available_prop_scenario_' + scenario] = new_availability_df_imputed['available_prop'] * new_availability_df_imputed['change_proportion_' + scenario]
    availability_greater_than_1 = new_availability_df_imputed['available_prop_scenario_' + scenario] > 1
    new_availability_df_imputed.loc[availability_greater_than_1, 'available_prop_scenario_' + scenario] = 1

    assert(sum(new_availability_df_imputed['available_prop_scenario_' + scenario].isna()) ==
           sum(new_availability_df_imputed['change_proportion_' + scenario].isna()))

# Save
#------------------------------------------------------
final_list_of_scenario_vars = ['available_prop_scenario_' + item for item in list_of_scenario_suffixes]
old_vars = ['Facility_ID', 'month', 'item_code', 'available_prop']
full_df_with_scenario = new_availability_df_imputed[old_vars + final_list_of_scenario_vars].reset_index().drop('index', axis = 1)
# Remove suffix for column names ending with '_cat' to mean 'categorised'
full_df_with_scenario.columns = [col.replace('_cat', '') if col.endswith('_cat') else col for col in full_df_with_scenario.columns]

# Save updated consumable availability resource file with scenario data
full_df_with_scenario.to_csv(
    path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv",
    index=False
)
# TODO: What about cases which do not exist in HHFA data?
# TODO: Create a column providing the source of scenario data
