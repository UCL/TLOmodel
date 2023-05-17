"""
This script adds estimates of availability of consumables under different scenarios to the base Resource File:

Outputs:
* Updated version of ResourceFile_Consumables_availability_small.csv (estimate of consumable available - smaller file for use in the
 simulation) which includes new consumable availability estimates for policy evaluation scenarios

Inputs:
* outputs/regression_analysis/predictions/predicted_consumable_availability_computers_scenario.csv - This file is hosted
locally after running consumable_resource_analyses_with_hhfa/regression_analysis/8_predict.R
* ResourceFile_Consumables_availability_small.csv` - This file contains the original consumable availability estimates
from OpenLMIS 2018 data
* `ResourceFile_Consumables_matched.csv` - This file contains the crosswalk of HHFA consumables to consumables in the
TLO model

It creates one row for each consumable for availability at a specific facility and month when the data is extracted from
the OpenLMIS dataset and one row for each consumable for availability aggregated across all facilities when the data is
extracted from the Harmonised Health Facility Assessment 2018/19.

Consumable availability is measured as probability of stockout at any point in time.
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
    'C:/Users/sm2511/Dropbox/Thanzi la Onse'
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

# 1.1.1 Attach district and facility level to this dataset
#------------------------------------------------------
# Get TLO Facility_ID for each district and facility level
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}
tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    on = ['Facility_ID'], how='left')

# 1.1.2 Attach programs
programs = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")[['category', 'item_code', 'module_name']]
programs = programs.drop_duplicates('item_code')
tlo_availability_df = tlo_availability_df.merge(programs, on = ['item_code'], how = 'left')

# 1.2 Import scenario data
#------------------------------------------------------
scenario_availability_df = pd.read_csv(outputfilepath / "regression_analysis/predictions/predicted_consumable_availability_pharmacists_scenario.csv")
scenario_availability_df = scenario_availability_df.drop(['Unnamed: 0'], axis=1)
scenario_availability_df = scenario_availability_df.rename({'item': 'item_hhfa'}, axis=1)

# Get Scenario data ready to be merged based on TLO model features
# 1.2.1 Level of care
#------------------------------------------------------
scenario_availability_df['fac_type'] = scenario_availability_df['fac_type'].str.replace("Facility_level_", "")

# 1.2.2 District
#------------------------------------------------------
# Do some mapping to make the Districts line-up with the definition of Districts in the model
rename_and_collapse_to_model_districts = {
    'Mzimba South': 'Mzimba',
    'Mzimba North': 'Mzimba',
}
scenario_availability_df['district_std'] = scenario_availability_df['district'].replace(rename_and_collapse_to_model_districts)

# Cities to get same results as their respective regions
copy_source_to_destination = {
    'Mzimba': 'Mzuzu City',
    'Lilongwe': 'Lilongwe City',
    'Zomba': 'Zomba City',
    'Blantyre': 'Blantyre City',
    'Nkhata Bay': 'Likoma' # based on anecdotal evidence, assume that they experience the same change in avaiability as a result of interventions based on regression results
}
for source, destination in copy_source_to_destination.items():
    new_rows = scenario_availability_df.loc[scenario_availability_df.district_std == source].copy()
    new_rows.district_std = destination
    #scenario_availability_df = scenario_availability_df.append(new_rows)
    # The above append method is throwing an Attribute Error so I've replaced it with the concat method below
    scenario_availability_df = pd.concat([scenario_availability_df, new_rows], ignore_index = True)

# 1.2.3 Facility_ID
# #------------------------------------------------------
# Merge-in facility_id
scenario_availability_facid_merge = scenario_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['district_std', 'fac_type'],
                    right_on=['District', 'Facility_Level'], how='left', indicator=True)
scenario_availability_facid_merge = scenario_availability_facid_merge.rename({'_merge': 'merge_facid'}, axis=1)

# Extract list of District X Facility Level combinations for which there is no HHFA data
scenario_availability_df_test = scenario_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['district_std', 'fac_type'],
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
consumable_crosswalk_df = pd.read_csv(path_for_new_resourcefiles / 'ResourceFile_consumables_matched.csv', encoding='ISO-8859-1')

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
print("Number of unique facility IDs: \n TLO consumable data = ", tlo_availability_df.Facility_ID.nunique(),
      "\n Scenario based on regression = ", scenario_availability_facid_itemcode_merge.Facility_ID.nunique(),
      "\n Number of unique item codes: \n TLO consumable data = ", tlo_availability_df.item_code.nunique(),
      "\n  Scenario based on regression = ", scenario_availability_facid_itemcode_merge.item_code.nunique())

# 1.2.6 Interpolation/Imputation where data is missing
#------------------------------------------------------
# 1.2.6.1 Items not matched
#------------------------------------------------------
# Before merging the scenario dataframe with tlo_availability_df, generate rows with all 57 relevant facility IDs for item_codes
# which are not matched
df = scenario_availability_facid_itemcode_merge
df_missing = df.loc[df['Facility_ID'].isna()].reset_index()
df_missing = df_missing.drop_duplicates('item_code')

df_existing = df.loc[~df['Facility_ID'].isna()]

# Create a cartesian product of unique values in columns 'item_code' and 'Facility_ID' from both dataframes
df_new = pd.DataFrame({'Facility_ID': np.repeat(df_existing['Facility_ID'].unique(), df_missing['item_code'].nunique()),
                       'item_code': np.tile(df_missing['item_code'].unique(), df_existing['Facility_ID'].nunique())})
# Merge df_new with df_missing on columns 'item_code' and 'Facility_ID'
df_new = df_new[['Facility_ID', 'item_code']].merge(df_missing.drop('Facility_ID', axis = 1), on=['item_code'], how='left', validate = "m:1")
# df_new = df_new.append(df_existing)
df_new = pd.concat([df_new, df_existing], ignore_index = True)

#df_new.to_csv(outputfilepath / 'temp_scenario_file_with_adequate_rows.csv')
scenario_final_df = df_new
len(scenario_final_df[~scenario_final_df['availability_change_prop'].isna()])

# Now provide availability_change_prop for rows where this data is not available

# 1.2.6.1 Extract list of TLO consumables which weren't matched with the availability prediction dataframe
#scenario_availability_facid_itemcode_merge[items_not_matched][['item_code', 'regression_application']].to_csv(outputfilepath / 'temp_items_not_matched.csv')

# Get average  availability_change_prop value by facility_ID and category_tlo
scenario_final_df = scenario_final_df.drop('module_name', axis = 1).merge(programs,
                                            on = ['item_code'], validate = "m:1",
                                            how = "left")
scenario_averages_by_program_and_facid = scenario_final_df.groupby(['Facility_ID','category'])['availability_change_prop'].mean().reset_index()

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
scenario_cat1_and_cat2 = scenario_cat1_and_cat2.drop('availability_change_prop', axis = 1)

scenario_cat1_and_cat2 = scenario_cat1_and_cat2.merge(scenario_averages_by_program_and_facid,
                                                      on = ['Facility_ID','category'], validate = "m:1",
                                                      how = "left")
scenario_cat1_and_cat2['availability_imputed'] = scenario_cat1_and_cat2['availability_change_prop']
scenario_final_df = scenario_final_df.merge(scenario_cat1_and_cat2[['Facility_ID','item_code', 'availability_imputed']],
                                                      on = ['Facility_ID','item_code'], validate = "1:1",
                                                      how = "left")
scenario_final_df.loc[(unmatched_items_category1|unmatched_items_category2), 'availability_change_prop'] = scenario_final_df['availability_imputed']

# c. 'not relevant to logistic regression analysis'
unmatched_items_category3 = (items_not_matched) & (scenario_final_df['regression_application'] == 'not relevant to logistic regression analysis')
# For category 3, replace availability_change_prop with 1, since we assume that the system-level intervention does not change availability
scenario_final_df.loc[unmatched_items_category3,'availability_change_prop'] = 1

# d. any other categories of unmatched consumables
unmatched_items_category4 = ~unmatched_items_category1 & ~unmatched_items_category2 & ~unmatched_items_category3 & items_not_matched
assert(sum(unmatched_items_category4) == 0) # check that we haven't missed any consumables

# 1.2.6.2 Inf values
#------------------------------------------------------
# Where the values are Inf because the availability changed from 0 to X, replace with the average for the category and Facility_ID
change_value_is_infinite = scenario_final_df.availability_change_prop.isin([np.inf])
df_inf = scenario_final_df[change_value_is_infinite].reset_index()

average_change_across_category_and_facid = scenario_final_df[~change_value_is_infinite].groupby(['Facility_ID','category'])['availability_change_prop'].mean().reset_index()

df_inf_replaced = df_inf.drop('availability_change_prop', axis = 1).merge(average_change_across_category_and_facid,
                               on = ['Facility_ID','category'],
                               how = 'left', validate = "m:1")
scenario_final_df = pd.concat([scenario_final_df[~change_value_is_infinite], df_inf_replaced], ignore_index = True)

#cond1 = scenario_final_df.District.isin(districts_with_no_scenario_data_for_1b_only)
#cond2 = scenario_final_df.Facility_Level == '1b'
#scenario_final_df[cond1 & cond2]

# 2. Merge TLO model availability data with scenario data using crosswalk
#*************************************************************************
# 2.1 Merge the two datasets
#------------------------------------------------------
new_availability_df = tlo_availability_df.merge(scenario_final_df[['item_code','Facility_ID','availability_change_prop', 'regression_application']],
                               how='left', on=['Facility_ID', 'item_code'], indicator = True)
new_availability_df = new_availability_df.rename({'_merge': 'merge_scenario'}, axis=1)
new_availability_df = new_availability_df.drop_duplicates(['Facility_ID', 'item_code', 'month'])

# 2.2 Further imputation
#------------------------------------------------------
# 2.2.1 For level 1b for the districts where this level was not present in the regression analysis/HHFA dataset, assume
# that the change is equal to the product of the (ratio of average change across districts for level 1b to
# average change across districts for level 1a) and change for each item_code for level 1a for that district
#------------------------------------------------------
average_change_across_districts = scenario_final_df.groupby(['Facility_Level','item_code'])['availability_change_prop'].mean().reset_index()
#average_change_across_districts = scenario_final_df.groupby(['Facility_Level','item_code']).agg({'availability_change_prop': lambda x: x.mean(skipna = False)}).reset_index()
average_change_across_districts_for_1a = average_change_across_districts[average_change_across_districts.Facility_Level == "1a"].rename({'availability_change_prop' : 'availability_change_prop_1a'}, axis = 1).drop('Facility_Level', axis = 1)
average_change_across_districts_for_1b = average_change_across_districts[average_change_across_districts.Facility_Level == "1b"].rename({'availability_change_prop' : 'availability_change_prop_1b'}, axis = 1).drop('Facility_Level', axis = 1)
ratio_of_change_across_districts_1b_to_1a = average_change_across_districts_for_1a.merge(average_change_across_districts_for_1b,
                                                                                         how = "left", on = ['item_code'])
ratio_of_change_across_districts_1b_to_1a['ratio'] = (ratio_of_change_across_districts_1b_to_1a.availability_change_prop_1b-1)/(ratio_of_change_across_districts_1b_to_1a.availability_change_prop_1a - 1)
ratio_of_change_across_districts_1b_to_1a.reset_index()

# Use the above for those districts no level 1b facilities recorded in the HHFA data
cond_1b_missing_district = new_availability_df.District.isin(districts_with_no_scenario_data_for_1b_only)
cond_1b = new_availability_df.Facility_Level == '1b'
cond_1a = new_availability_df.Facility_Level == '1a'
df_missing_1b = new_availability_df[cond_1b_missing_district & cond_1b]
df_1a = new_availability_df[cond_1b_missing_district & cond_1a]
df_missing_1b_imputed = df_missing_1b.merge(ratio_of_change_across_districts_1b_to_1a[['item_code', 'ratio']],
                               on = ['item_code'],
                               how = 'left', validate = "m:1")
assert np.isnan(df_missing_1b_imputed['availability_change_prop'].unique()).all()
df_missing_1b_imputed = df_missing_1b_imputed.drop('availability_change_prop', axis = 1).merge(df_1a[['availability_change_prop', 'District', 'item_code', 'month']],
                               on = ['District', 'item_code', 'month'],
                               how = 'left', validate = "1:1", indicator = True)


df_missing_1b_imputed['availability_change_prop'] = ((df_missing_1b_imputed['availability_change_prop']-1) * df_missing_1b_imputed['ratio']) + 1

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
                             values=['availability_change_prop'],
                             index=['category'],
                             columns=['Facility_Level'],
                             aggfunc=lambda x: sum(pd.isna(x))/len(x)*100)

print(pivot_table[('availability_change_prop', '1b')])
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
missing_change_data = new_availability_df_imputed.availability_change_prop.isna()
new_availability_df_imputed.loc[missing_change_data, 'availability_change_prop'] = new_availability_df_imputed['availability_change_prop'].mean()

# Where the merge_scenario == "left_only", we need to provide data on "availability_change_prop"
# new_availability_df_imputed.to_csv(outputfilepath / 'current_status_of_scenario_merge.csv')
# TODO Check why the city and district both have an instance of facility level 2

# 3. Generate scenario data on consumable availablity
#------------------------------------------------------
# Create new consumable availability estimates for TLO model consumables using
# estimates of proportional change from the regression analysis based on HHFA data
#------------------------------------------------------
new_availability_df_imputed['available_prop_scenario1'] = new_availability_df_imputed['available_prop'] * new_availability_df_imputed['availability_change_prop']

availability_greater_than_1 = new_availability_df_imputed['available_prop_scenario1'] > 1
new_availability_df_imputed.loc[availability_greater_than_1, 'available_prop_scenario1'] = 1

assert(sum(new_availability_df_imputed.available_prop_scenario1.isna()) ==
       sum(new_availability_df_imputed.availability_change_prop.isna()))

# Save
#------------------------------------------------------
full_df_with_scenario = new_availability_df_imputed[['Facility_ID', 'month', 'item_code', 'available_prop', 'available_prop_scenario1']].reset_index().drop('index', axis = 1)
# full_df_with_scenario.to_csv(outputfilepath / "temp_consumable_resourcefile.csv", index=False)
# Save updated consumable availability resource file with scenario data
full_df_with_scenario.to_csv(
    path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv",
    index=False
)
# TODO: What about cases which do not exist in HHFA data?
# TODO: Create a column providing the source of scenario data
'''
scenario_availability_facid_itemcode_merge['source_of_prop_change'] = np.nan
scenario_availability_facid_itemcode_merge.loc[~items_not_matched, 'source_of_prop_change'] = 'direct output from regression analysis'

scenario_availability_facid_itemcode_merge.loc[unmatched_items_category1|unmatched_items_category2] = 'average output for relevant program and facility level from regression analysis'
scenario_availability_facid_itemcode_merge.loc[unmatched_items_category3] = 'assume no change'

'''

# 3. Plot the data
"""
ggplot(new_availability_df) + aes(x="category", y="availability_change_prop") + geom_point()

ggplot(new_availability_df, aes(x='category', y='availability_change_prop')) + \
    geom_point() + geom_jitter() + \
    geom_hline(yintercept = 1, linetype="dashed") + \
    labs(x="Category", y="Availability Change Proportion") + theme_bw()

ggplot(new_availability_df, aes(x='category', y='availability_change_prop')) +  geom_point(position = position_jitter(h = 0.2, seed = 123))  + geom_hline(yintercept = 1, linetype="dashed")
   # labs(x="Category", y="Availability Change Proportion") + theme_bw()
"""

