"""
This will generate estimates of availability of consumables used by disease modules
"""
# import copy
import datetime
# Import Statements and initial declarations
import os
# import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from matplotlib import pyplot  # for figures

# from tlo import Date, Simulation
# from tlo.analysis.utils import parse_log_file

# Set working directory
os.chdir('C:/Users/sm2511/PycharmProjects/TLOmodel')

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define a pathway to the data folder (note: currently outside the TLO model directory)
# remember to set working directory to TLOmodel/
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
datafilepath = Path("./../../Documents/health_systems/data")


# Define necessary functions
def change_colnames(df, NameChangeList):  # Change column names
    ColNames = df.columns
    ColNames2 = ColNames
    for (a, b) in NameChangeList:
        print(a, '-->', b)
        ColNames2 = [col.replace(a, b) for col in ColNames2]
    df.columns = ColNames2
    return df


# In[557]:


# 1. DATA IMPORT AND CLEANING ##
#########################################################################################

# Import 2018 data
lmis_df = pd.read_csv(resourcefilepath / 'ResourceFile_LMIS_2018.csv', low_memory=False)

# 1. BASIC CLEANING ##
# Rename columns
NameChangeList = [('RYear', 'year'), ('RMonth', 'month'), ('Name (Geographic Zones)', 'district'),
                  ('Name (Facility Operators)', 'fac_owner'), ('Name', 'fac_name'), ('fac_name (Programs)', 'program'),
                  ('fac_name (Facility Types)', 'fac_type'), ('Fullproductname', 'item'),
                  ('Closing Bal', 'closing_bal'), ('Dispensed', 'dispensed'), ('AMC', 'amc'), ('Received', 'received'),
                  ('Stockout days', 'stkout_days'), ]

change_colnames(lmis_df, NameChangeList)

# Remove Private Health facilities from the data
cond_pvt1 = lmis_df['fac_owner'] == 'Private'
cond_pvt2 = lmis_df['fac_type'] == 'Private Hospital'
lmis_df = lmis_df[~cond_pvt1 & ~cond_pvt2]

# Clean facility types to match with types in the TLO model
# See link: https://docs.google.com/spreadsheets/d/1fcp2-smCwbo0xQDh7bRUnMunCKguBzOIZjPFZKlHh5Y/edit#gid=0
cond_level0 = (lmis_df['fac_name'].str.contains('Health Post'))
cond_level1a = (lmis_df['fac_type'] == 'Clinic') | (lmis_df['fac_type'] == 'Health Centre') | (
    lmis_df['fac_name'].str.contains('Clinic')) | (lmis_df['fac_name'].str.contains('Health Centre')) | (
                   lmis_df['fac_name'].str.contains('Maternity')) | (lmis_df['fac_name'] == 'Chilaweni') | (
                   lmis_df['fac_name'] == 'Chimwawa') | (lmis_df['fac_name'].str.contains('Dispensary'))
cond_level1b = (lmis_df['fac_name'].str.contains('Community Hospital')) | (
    lmis_df['fac_type'] == 'Rural/Community Hospital') | (lmis_df['fac_name'] == 'St Peters Hospital') | (
                   lmis_df['fac_name'] == 'Police College Hospital') | (lmis_df['fac_type'] == 'CHAM') | (
                   lmis_df['fac_owner'] == 'CHAM')
cond_level2 = (lmis_df['fac_type'] == 'District Health Office') | (lmis_df['fac_name'].str.contains('DHO'))
cond_level3 = (lmis_df['fac_type'] == 'Central Hospital') | (lmis_df['fac_name'].str.contains('Central Hospital'))
cond_level4 = (lmis_df['fac_name'] == 'Zomba Mental Hospital')
lmis_df.loc[cond_level0, 'fac_type_tlo'] = 'Facility_level_0'
lmis_df.loc[cond_level1a, 'fac_type_tlo'] = 'Facility_level_1a'
lmis_df.loc[cond_level1b, 'fac_type_tlo'] = 'Facility_level_1b'
lmis_df.loc[cond_level2, 'fac_type_tlo'] = 'Facility_level_2'
lmis_df.loc[cond_level3, 'fac_type_tlo'] = 'Facility_level_3'
lmis_df.loc[cond_level4, 'fac_type_tlo'] = 'Facility_level_4'

print('Data import complete and ready for analysis')

# Convert values to numeric format
num_cols = ['year',
            'closing_bal',
            'dispensed',
            'stkout_days',
            'amc',
            'received']
lmis_df[num_cols] = lmis_df[num_cols].replace(to_replace=',', value='', regex=True)
lmis_df[num_cols] = lmis_df[num_cols].astype(float)

# Define relevant dictionaries for analysis
months_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
               9: 'September', 10: 'October', 11: 'November', 12: 'December'}
fac_types_dict = {1: 'Facility_level_0', 2: 'Facility_level_1a', 3: 'Facility_level_1b', 4: 'Facility_level_2',
                  5: 'Facility_level_3'}
districts_dict = {1: 'Balaka', 2: 'Blantyre', 3: 'Chikwawa', 4: 'Chiradzulu', 5: 'Chitipa', 6: 'Dedza',
                  7: 'Dowa', 8: 'Karonga', 9: 'Kasungu', 10: 'Lilongwe', 11: 'Machinga', 12: 'Mangochi',
                  13: 'Mchinji', 14: 'Mulanje', 15: 'Mwanza', 16: 'Mzimba North', 17: 'Mzimba South',
                  18: 'Neno', 19: 'Nkhata bay', 20: 'Nkhota Kota', 21: 'Nsanje', 22: 'Ntcheu', 23: 'Ntchisi',
                  24: 'Phalombe', 25: 'Rumphi', 26: 'Salima', 27: 'Thyolo', 28: 'Zomba'}
programs_lmis_dict = {1: 'Essential Meds', 2: 'HIV', 3: 'Malaria', 4: 'Nutrition', 5: 'RH',
                      6: 'TB'}  # programs listed in the OpenLMIS data
months_withdata = ['January', 'February', 'April', 'October', 'November']
months_interpolated = ['March', 'May', 'June', 'July', 'August', 'September', 'December']

# 2. RESHAPE AND REORDER ##
#########################################################################################

# Reshape dataframe so that each row represent a unique consumable and facility
lmis_df_wide = lmis_df.pivot_table(index=['district', 'fac_type_tlo', 'fac_name', 'program', 'item'], columns='month',
                                   values=['closing_bal', 'dispensed', 'received', 'stkout_days', 'amc'],
                                   fill_value=-99)

# Reorder columns in chronological order
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
          'December']
lmis_df_wide = lmis_df_wide.reindex(months, axis=1, level=1)

# Replace all the -99s created in the pivoting process above to NaNs
num = lmis_df_wide._get_numeric_data()
lmis_df_wide[num < 0] = np.nan

# 3. INTERPOLATE MISSING VALUES ##
#########################################################################################
# When stkout_days is empty but closing balance, dispensed and received entries are available
lmis_df_wide_flat = lmis_df_wide.reset_index()
count_stkout_entries = lmis_df_wide_flat['stkout_days'].count(axis=1).sum()
print(count_stkout_entries, "stockout entries before first interpolation")

# Define lists of months with the same number of days
months_dict31 = ['January', 'March', 'May', 'July', 'August', 'October', 'December']
months_dict30 = ['April', 'June', 'September', 'November']

for m in range(1, 13):
    # Identify datapoints which come from the original data source (not interpolation)
    lmis_df_wide_flat[('data_source', months_dict[m])] = np.nan  # empty column
    cond1 = lmis_df_wide_flat['stkout_days', months_dict[m]].notna()
    lmis_df_wide_flat.loc[cond1, [('data_source', months_dict[m])]] = 'original_lmis_data'

    # First update received to zero if other columns are avaialble
    cond1 = lmis_df_wide_flat['closing_bal', months_dict[m]].notna() & lmis_df_wide_flat['amc', months_dict[m]].notna()
    cond2 = lmis_df_wide_flat['received', months_dict[m]].notna()
    lmis_df_wide_flat.loc[cond1 & ~cond2, [('received', months_dict[m])]] = 0

# --- 3.1 RULE: 1.If i) stockout is missing, ii) closing_bal, amc and received are not missing , and iii) amc !=0 and,
#          then stkout_days[m] = (amc[m] - closing_bal[m-1] - received)/amc * number of days in the month ---
# (Note that the number of entries for closing balance, dispensed and received is always the same)
for m in range(2, 13):
    # Now update stkout_days if other columns are available
    cond1 = lmis_df_wide_flat['closing_bal', months_dict[m - 1]].notna() & lmis_df_wide_flat[
        'amc', months_dict[m]].notna() & lmis_df_wide_flat['received', months_dict[m]].notna()
    cond2 = lmis_df_wide_flat['stkout_days', months_dict[m]].notna()
    cond3 = lmis_df_wide_flat['amc', months_dict[m]] != 0
    lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('data_source', months_dict[m])]] = 'lmis_interpolation_rule1'

    if months_dict[m] in months_dict31:
        lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('stkout_days', months_dict[m])]] = (lmis_df_wide_flat[(
            'amc', months_dict[m])] - lmis_df_wide_flat[('closing_bal', months_dict[m - 1])] - lmis_df_wide_flat[(
                'received', months_dict[m])]) / lmis_df_wide_flat[('amc', months_dict[m])] * 31
    elif months_dict[m] in months_dict30:
        lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('stkout_days', months_dict[m])]] = (lmis_df_wide_flat[(
            'amc', months_dict[m])] - lmis_df_wide_flat[('closing_bal', months_dict[m - 1])] - lmis_df_wide_flat[(
                'received', months_dict[m])]) / lmis_df_wide_flat[('amc', months_dict[m])] * 30
    else:
        lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('stkout_days', months_dict[m])]] = (lmis_df_wide_flat[(
            'amc', months_dict[m])] - lmis_df_wide_flat[('closing_bal', months_dict[m - 1])] - lmis_df_wide_flat[(
                'received', months_dict[m])]) / lmis_df_wide_flat[('amc', months_dict[m])] * 28

count_stkout_entries = lmis_df_wide_flat['stkout_days'].count(axis=1).sum()
print(count_stkout_entries, "stockout entries after first interpolation")

# 3.2 --- If any stockout_days < 0 after the above interpolation, update to stockout_days = 0 ---
# RULE: If closing balance[previous month] - dispensed[this month] + received[this month] > 0, stockout == 0
for m in range(1, 13):
    cond1 = lmis_df_wide_flat['stkout_days', months_dict[m]] < 0
    # print(
    # "Negative stockout days ", lmis_df_wide_flat.loc[cond1,[('stkout_days', months_dict[m])]].count(axis = 1).sum()
    # )
    lmis_df_wide_flat.loc[cond1, [('data_source', months_dict[m])]] = 'lmis_interpolation_rule2'
    lmis_df_wide_flat.loc[cond1, [('stkout_days', months_dict[m])]] = 0

count_stkout_entries = lmis_df_wide_flat['stkout_days'].count(axis=1).sum()
print(count_stkout_entries, "stockout entries after second interpolation")

lmis_df_wide_flat['consumable_reported_horiz'] = lmis_df_wide_flat['closing_bal'].count(
    axis=1)  # generate a column which reports the number of entries of closing balance

# Flatten multilevel columns
lmis_df_wide_flat.columns = [' '.join(col).strip() for col in lmis_df_wide_flat.columns.values]

# 3.3 --- If the consumable was previously reported and during a given month, if any consumable was reported, assume
# 100% days of stckout ---
# RULE: If the balance on a consumable is ever reported and if any consumables are reported during the month,
# stkout_days = number of days of the month
for m in range(1, 13):
    month = 'closing_bal ' + months_dict[m]
    var_name = 'consumable_reported_vert ' + months_dict[m]
    lmis_df_wide_flat[var_name] = lmis_df_wide_flat.groupby("fac_name")[month].transform('count')

for m in range(1, 13):
    cond1 = lmis_df_wide_flat['consumable_reported_horiz'] > 0
    cond2 = lmis_df_wide_flat['consumable_reported_vert ' + months_dict[m]] > 0
    cond3 = lmis_df_wide_flat['stkout_days ' + months_dict[m]].notna()
    lmis_df_wide_flat.loc[cond1 & cond2 & ~cond3, [('data_source ' + months_dict[m])]] = 'lmis_interpolation_rule3'

    if months_dict[m] in months_dict31:
        lmis_df_wide_flat.loc[cond1 & cond2 & ~cond3, [('stkout_days ' + months_dict[m])]] = 31
    elif months_dict[m] in months_dict30:
        lmis_df_wide_flat.loc[cond1 & cond2 & ~cond3, [('stkout_days ' + months_dict[m])]] = 30
    else:
        lmis_df_wide_flat.loc[cond1 & cond2 & ~cond3, [('stkout_days ' + months_dict[m])]] = 28

count_stkout_entries = 0
for m in range(1, 13):
    count_stkout_entries = count_stkout_entries + lmis_df_wide_flat['stkout_days ' + months_dict[m]].count().sum()
print(count_stkout_entries, "stockout entries after third interpolation")

# In[560]:


# 4. CALCULATE STOCK OUT RATES BY MONTH and FACILITY ##
#########################################################################################

lmis = lmis_df_wide_flat  # choose dataframe

# Generate variables denoting the stockout proportion in each month
for m in range(1, 13):
    if months_dict[m] in months_dict31:
        lmis['stkout_prop ' + months_dict[m]] = lmis['stkout_days ' + months_dict[m]] / 31
    elif months_dict[m] in months_dict30:
        lmis['stkout_prop ' + months_dict[m]] = lmis['stkout_days ' + months_dict[m]] / 30
    else:
        lmis['stkout_prop ' + months_dict[m]] = lmis['stkout_days ' + months_dict[m]] / 28

# Reshape data
lmis = pd.wide_to_long(lmis, stubnames=['closing_bal', 'received', 'amc', 'dispensed', 'stkout_days', 'stkout_prop',
                                        'data_source', 'consumable_reported_vert'],
                       i=['district', 'fac_type_tlo', 'fac_name', 'program', 'item'], j='month',
                       sep=' ', suffix=r'\w+')
lmis = lmis.reset_index()

# In[561]:


# 5. LOAD CLEANED MATCHED CONSUMABLE LIST FROM TLO MODEL AND MERGE WITH LMIS DATA ##
#########################################################################################

# 5.1 --- Load and clean data ---
# Import matched list of consumanbles
consumables_df = pd.read_csv(resourcefilepath / 'ResourceFile_consumables_matched.csv', low_memory=False)
cond = consumables_df['matching_status'] == 'Remove'
consumables_df = consumables_df[~cond]  # Remove items which were removed due to updates or the existence of duplicates

# Keep only the correctly matched consumables for stockout analysis based on OpenLMIS
cond1 = consumables_df['matching_status'] == 'Matched'
cond2 = consumables_df['verified_by_DM_lead'] != 'Incorrect'
matched_consumables = consumables_df[cond1 & cond2]

# Rename columns
NameChangeList = [('consumable_name_lmis', 'item'), ]
change_colnames(consumables_df, NameChangeList)
change_colnames(matched_consumables, NameChangeList)

# 5.2 --- Merge data with LMIS data ---
lmis_matched_df = pd.merge(lmis, matched_consumables, how='inner', on=['item'])
lmis_matched_df = lmis_matched_df.sort_values('data_source')

# 5.2.i. For substitable drugs (within drug category), collapse by taking the product of stkout_prop (OR condition)
# This represents Pr(all substitutes with the item code are stocked out)
stkout_df = lmis_matched_df.groupby(
    ['module_name', 'district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'consumable_name_tlo', 'match_level1',
     'match_level2'],
    as_index=False).agg({'stkout_prop': 'prod',
                         'closing_bal': 'sum',
                         'amc': 'sum',
                         'dispensed': 'sum',
                         'received': 'sum',
                         'data_source': 'first',
                         'consumable_reported_horiz': 'first',
                         'consumable_reported_vert': 'first'})

# 5.2.ii. For complementary drugs, collapse by taking the product of (1-stkout_prob)
# This represents Pr(All drugs within item code (in different match_group's) are available)
stkout_df['available_prop'] = 1 - stkout_df['stkout_prop']
stkout_df = stkout_df.groupby(
    ['module_name', 'district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'consumable_name_tlo',
     'match_level2'],
    as_index=False).agg({'available_prop': 'prod',
                         'closing_bal': 'sum',  # could be min
                         'amc': 'sum',  # could be max
                         'dispensed': 'sum',  # could be max
                         'received': 'sum',  # could be min
                         'data_source': 'first',
                         'consumable_reported_horiz': 'first',
                         'consumable_reported_vert': 'first'})

# 5.2.iii. For substitable drugs (within consumable_name_tlo), collapse by taking the product of stkout_prop (OR
# condition)
# This represents Pr(all substitutes with the item code are stocked out)
stkout_df['stkout_prop'] = 1 - stkout_df['available_prop']
stkout_df = stkout_df.groupby(
    ['module_name', 'district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'consumable_name_tlo'],
    as_index=False).agg({'stkout_prop': 'prod',
                         'closing_bal': 'sum',
                         'amc': 'sum',
                         'dispensed': 'sum',
                         'received': 'sum',
                         'data_source': 'first',
                         'consumable_reported_horiz': 'first',
                         'consumable_reported_vert': 'first'})
stkout_df['available_prop'] = 1 - stkout_df['stkout_prop']

# Some missing values change to 100% stockouts during the aggregation above. Fix this manually
for var in ['stkout_prop', 'available_prop', 'closing_bal', 'amc', 'dispensed', 'received']:
    cond = stkout_df['data_source'].isna()
    stkout_df.loc[cond, var] = np.nan

stkout_df = stkout_df.reset_index()
stkout_df = stkout_df[
    ['module_name', 'district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'consumable_name_tlo',
     'stkout_prop', 'available_prop', 'closing_bal', 'amc', 'dispensed', 'received',
     'data_source', 'consumable_reported_horiz', 'consumable_reported_vert']]

# In[562]:


# 6. ADD STOCKOUT DATA FROM OTHER SOURCES TO COMPLETE STOCKOUT DATAFRAME ##
#########################################################################################

# --- 6.1. Generate a dataframe of stock availability for consumables which were not found in the OpenLMIS data but
# available in the HHFA 2018/19 --- #
# Save the list of items for which a match was not found in the OpenLMIS data
unmatched_consumables = consumables_df.drop_duplicates(['item_code'])
unmatched_consumables = pd.merge(unmatched_consumables, matched_consumables[['item', 'item_code']], how='left',
                                 on='item_code')
unmatched_consumables = unmatched_consumables[unmatched_consumables['item_y'].isna()]

# ** Extract stock availability data from HHFA and clean data **
hhfa_df = pd.read_excel(open(resourcefilepath / 'ResourceFile_hhfa_consumables.xlsx', 'rb'), sheet_name='hhfa_data')

# Use the ratio of availability rates between levels 1b on one hand and levels 2 and 3 on the other to extrapolate
# availability rates for levels 2 and 3 from the HHFA data
cond1b = stkout_df['fac_type_tlo'] == 'Facility_level_1b'
cond2 = stkout_df['fac_type_tlo'] == 'Facility_level_2'
cond3 = stkout_df['fac_type_tlo'] == 'Facility_level_3'
availratio_2to1b = stkout_df[cond2]['available_prop'].mean() / stkout_df[cond1b]['available_prop'].mean()
availratio_3to1b = stkout_df[cond3]['available_prop'].mean() / stkout_df[cond1b]['available_prop'].mean()

# To disaggregate the avalability rates for levels 1b, 2, and 3 from the HHFA, assume that the ratio of availability
# across the three levels is the same as that based on OpenLMIS
# There are a total of 101 "hospitals" surveyed for commodities in the HHFA (we assume that of these 28 are district
# hospitals and 4 are central hospitals -> 69 1b health facilities). The availability rates for the three levels can be
# dereived as follows
scaleparam = 101 / (69 + 28 * availratio_2to1b + 4 * availratio_3to1b)
hhfa_df['available_prop_hhfa_Facility_level_1b'] = scaleparam * hhfa_df['available_prop_hhfa_Facility_level_1b']
hhfa_df['available_prop_hhfa_Facility_level_2'] = availratio_2to1b * hhfa_df['available_prop_hhfa_Facility_level_1b']
hhfa_df['available_prop_hhfa_Facility_level_3'] = availratio_3to1b * hhfa_df['available_prop_hhfa_Facility_level_1b']

for var in ['available_prop_hhfa_Facility_level_2', 'available_prop_hhfa_Facility_level_3']:
    cond = hhfa_df[var] > 1
    hhfa_df.loc[cond, var] = 1

# Merge HHFA data with the list of unmatched consumables from the TLO model
unmatched_consumables_df = pd.merge(unmatched_consumables, hhfa_df, how='left', on='item_code')
unmatched_consumables_df = unmatched_consumables_df[
    ['module_name', 'item_code', 'consumable_name_tlo_x', 'available_prop_hhfa_Facility_level_0',
     'available_prop_hhfa_Facility_level_1a', 'available_prop_hhfa_Facility_level_1b',
     'available_prop_hhfa_Facility_level_2', 'available_prop_hhfa_Facility_level_3',
     'fac_count_Facility_level_0', 'fac_count_Facility_level_1a', 'fac_count_Facility_level_1b']]

# ** Need to edit this part of the code so the levels 2 and 3 don't take on all districts **
# Reshape dataframe of consumable availability taken from the HHFA in the same format as the stockout dataframe based
# on OpenLMIS
unmatched_consumables_df = pd.wide_to_long(unmatched_consumables_df, stubnames=['available_prop_hhfa', 'fac_count'],
                                           i=['item_code', 'consumable_name_tlo_x'], j='fac_type_tlo',
                                           sep='_', suffix=r'\w+')
unmatched_consumables_df['stkout_prop'] = 1 - unmatched_consumables_df['available_prop_hhfa']

unmatched_consumables_df = unmatched_consumables_df.reset_index()
n = len(unmatched_consumables_df)

# Final cleaning
NameChangeList = [('consumable_name_tlo_x', 'consumable_name_tlo'), ('available_prop_hhfa', 'available_prop'), ]
change_colnames(unmatched_consumables_df, NameChangeList)

# --- 6.2 Append OpenLMIS stockout dataframe with HHFA stockout dataframe and Extract in .csv format --- #
# Append common consumables stockout dataframe with the main dataframe
cond = unmatched_consumables_df['available_prop'].notna()
unmatched_consumables_df.loc[cond, 'data_source'] = 'hhfa_2018-19'
unmatched_consumables_df.loc[~cond, 'data_source'] = 'Not available'
stkout_df = stkout_df.append(unmatched_consumables_df)

# --- 6.3 Append stockout rate for facility level 0 from HHFA --- #
cond = hhfa_df['item_code'].notna()
hhfa_fac0 = hhfa_df[cond][
    ['item_code', 'consumable_name_tlo', 'fac_count_Facility_level_0', 'available_prop_hhfa_Facility_level_0']]
NameChangeList = [('fac_count_Facility_level_0', 'fac_count'),
                  ('available_prop_hhfa_Facility_level_0', 'available_prop'), ]
change_colnames(hhfa_fac0, NameChangeList)
hhfa_fac0['fac_type_tlo'] = 'Facility_level_0'
hhfa_fac0['stkout_prop'] = 1 - hhfa_fac0['available_prop']
hhfa_fac0['data_source'] = 'hhfa_2018-19'

hhfa_fac0 = pd.merge(hhfa_fac0, consumables_df[['item_code', 'module_name']], on='item_code', how='inner')
hhfa_fac0 = hhfa_fac0.drop_duplicates()

cond = stkout_df['fac_type_tlo'] == 'Facility_level_0'
stkout_df = stkout_df[~cond]
stkout_df = stkout_df.append(hhfa_fac0)

# --- 6.4 Generate new category variable for analysis --- #
cond_RH = (stkout_df['module_name'].str.contains('care_of_women_during_pregnancy')) | (
    stkout_df['module_name'].str.contains('labour'))
cond_newborn = (stkout_df['module_name'].str.contains('newborn'))
cond_ari = stkout_df['module_name'] == 'acute lower respiratory infections'
cond_rti = stkout_df['module_name'] == 'Road traffic injuries'
stkout_df['category'] = stkout_df['module_name']
stkout_df.loc[cond_RH, 'category'] = 'reproductive_health'
stkout_df.loc[cond_newborn, 'category'] = 'neonatal_health'
stkout_df.loc[cond_ari, 'category'] = 'ari'
stkout_df.loc[cond_rti, 'category'] = 'road_traffic_injuries'
stkout_df['category'] = stkout_df['category'].str.lower()

cond_condom = stkout_df['item_code'] == 2
stkout_df.loc[cond_condom, 'category'] = 'contraception'

# Create a general consumables category
general_cons_list = [300, 33, 57, 58, 141, 5, 6, 10, 21, 23, 127, 24, 80, 93, 144, 149, 154, 40, 67, 73, 76,
                     82, 101, 103, 88, 126, 135, 71, 98, 171, 133, 134, 244, 247]
diagnostics_cons_list = [41, 50, 128, 216, 2008, 47, 190, 191, 196, 206, 207, 163, 175, 184,
                         187]  # for now these have not been applied because most diagnostics are program specific

cond_general = stkout_df['item_code'].isin(general_cons_list)
stkout_df.loc[cond_general, 'category'] = 'general'

# --- 6.5 Replace district/fac_name/month entries where missing --- #
for var in ['district', 'fac_name', 'month']:
    cond = stkout_df[var].isna()
    stkout_df.loc[cond, var] = 'NA'

# --- 6.6 Export final stockout dataframe --- #
stkout_df.to_csv(resourcefilepath / "ResourceFile_consumable_availability.csv")

# In[564]:


# 8. CALIBRATION TO HHFA DATA, 2018/19 ##
#########################################################################################
# --- 8.1 Prepare calibratino dataframe --- ##
# i. Prepare calibration data from HHFA
hhfa_calibration_df = hhfa_df[['item_code', 'consumable_name_tlo', 'item_hhfa', 'available_prop_hhfa_Facility_level_0',
                               'available_prop_hhfa_Facility_level_1a', 'available_prop_hhfa_Facility_level_1b',
                               'available_prop_hhfa_Facility_level_2', 'available_prop_hhfa_Facility_level_3']]
hhfa_calibration_df = pd.wide_to_long(hhfa_calibration_df.dropna(), stubnames='available_prop_hhfa',
                                      i=['consumable_name_tlo', 'item_code', 'item_hhfa'], j='fac_type_tlo',
                                      sep='_', suffix=r'\w+')
hhfa_calibration_df = hhfa_calibration_df.reset_index()

# ii. Collapse district level data in stkout_df and exclude data extracted from HHFA
cond1 = stkout_df['data_source'] == 'hhfa_2018-19'
lmis_calibration_df = stkout_df[~cond1].groupby(['module_name', 'fac_type_tlo', 'item_code']).mean().reset_index()

# iii. Merge HHFA with stkout_df
calibration_df = pd.merge(lmis_calibration_df, hhfa_calibration_df, how='inner', on=['item_code', 'fac_type_tlo'])
calibration_df['difference'] = (calibration_df['available_prop_hhfa'] - calibration_df['available_prop'])

# --- 8.2 Compare OpenLMIS estimates with HHFA estimates --- ##
# Summary results by level of care
calibration_df.groupby(['fac_type_tlo'])[['available_prop', 'available_prop_hhfa', 'difference']].mean()

# Plots by consumable
size = 10
calibration_df['item_code'] = calibration_df['item_code'].astype(str)
calibration_df['labels'] = calibration_df['consumable_name_tlo'].str[:5]

cond = calibration_df['fac_type_tlo'] == 'Facility_level_1a'
ax = calibration_df[cond].plot.line(x='labels', y=['available_prop', 'available_prop_hhfa'])
ax.set_xticks(np.arange(len(calibration_df[cond]['labels'])))
ax.set_xticklabels(calibration_df[cond]['labels'], rotation=90, fontsize=7)
plt.title('Level 1a', fontsize=size, weight="bold")
# plt.savefig(outputfilepath / 'calibration_level1a.png')

cond = calibration_df['fac_type_tlo'] == 'Facility_level_1b'
ax = calibration_df[cond].plot.line(x='labels', y=['available_prop', 'available_prop_hhfa'])
ax.set_xticks(np.arange(len(calibration_df[cond]['labels'])))
ax.set_xticklabels(calibration_df[cond]['labels'], rotation=90, fontsize=7)
plt.title('Level 1b', fontsize=size, weight="bold")
# plt.savefig(outputfilepath / 'calibration_level1b.png')

cond = calibration_df['fac_type_tlo'] == 'Facility_level_2'
ax = calibration_df[cond].plot.line(x='labels', y=['available_prop', 'available_prop_hhfa'])
ax.set_xticks(np.arange(len(calibration_df[cond]['labels'])))
ax.set_xticklabels(calibration_df[cond]['labels'], rotation=90, fontsize=7)
plt.title('Level 2', fontsize=size, weight="bold")
# plt.savefig(outputfilepath / 'calibration_level2.png')

cond = calibration_df['fac_type_tlo'] == 'Facility_level_3'
ax = calibration_df[cond].plot.line(x='labels', y=['available_prop', 'available_prop_hhfa'])
ax.set_xticks(np.arange(len(calibration_df[cond]['labels'])))
ax.set_xticklabels(calibration_df[cond]['labels'], rotation=90, fontsize=7)
plt.title('Level 3', fontsize=size, weight="bold")
# plt.savefig(outputfilepath / 'calibration_level3.png')
