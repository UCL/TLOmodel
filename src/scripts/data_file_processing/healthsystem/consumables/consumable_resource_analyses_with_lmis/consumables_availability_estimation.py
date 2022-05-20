"""
This script generates estimates of availability of consumables used by disease modules:

* ResourceFile_Consumables_availability_and_usage.csv (a large file that gives consumable availability and usage).
* ResourceFile_Consumables_availability_small.csv (estimate of consumable available - smaller file for use in the
 simulation).

N.B. The file uses `ResourceFile_Consumables_matched.csv` as an input.

It creates one row for each consumable for availability at a specific facility and month when the data is extracted from
the OpenLMIS dataset and one row for each consumable for availability aggregated across all facilities when the data is
extracted from the Harmonised Health Facility Assessment 2018/19.

Consumable availability is measured as probability of stockout at any point in time.

Data from OpenLMIS includes closing balance, quantity received, quantity dispensed, and average monthly consumption
for each month by facility.

"""

import calendar
import datetime
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.methods.consumables import check_format_of_consumables_file

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    # 'C:/Users/sm2511/Dropbox/Thanzi la Onse'
    # '/Users/sejjj49/Dropbox/Thanzi la Onse'
    # 'C:/Users/tmangal/Dropbox/Thanzi la Onse'
)

path_to_files_in_the_tlo_dropbox = path_to_dropbox / "05 - Resources/Module-healthsystem/consumables raw files/"

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define a pathway to the data folder (note: currently outside the TLO model directory)
# remember to set working directory to TLOmodel/
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"


# Define necessary functions
def change_colnames(df, NameChangeList):  # Change column names
    ColNames = df.columns
    ColNames2 = ColNames
    for (a, b) in NameChangeList:
        print(a, '-->', b)
        ColNames2 = [col.replace(a, b) for col in ColNames2]
    df.columns = ColNames2
    return df


# %%
# 1. DATA IMPORT AND CLEANING ##
#########################################################################################

# Import 2018 data
lmis_df = pd.read_csv(path_to_files_in_the_tlo_dropbox / 'ResourceFile_LMIS_2018.csv', low_memory=False)

# 1. BASIC CLEANING ##
# Rename columns
NameChangeList = [('RYear', 'year'),
                  ('RMonth', 'month'),
                  ('Name (Geographic Zones)', 'district'),
                  ('Name (Facility Operators)', 'fac_owner'),
                  ('Name', 'fac_name'),
                  ('fac_name (Programs)', 'program'),
                  ('fac_name (Facility Types)', 'fac_type'),
                  ('Fullproductname', 'item'),
                  ('Closing Bal', 'closing_bal'),
                  ('Dispensed', 'dispensed'),
                  ('AMC', 'amc'),
                  ('Received', 'received'),
                  ('Stockout days', 'stkout_days')]

change_colnames(lmis_df, NameChangeList)

# Remove Private Health facilities from the data
cond_pvt1 = lmis_df['fac_owner'] == 'Private'
cond_pvt2 = lmis_df['fac_type'] == 'Private Hospital'
lmis_df = lmis_df[~cond_pvt1 & ~cond_pvt2]

# Clean facility types to match with types in the TLO model
# See link: https://docs.google.com/spreadsheets/d/1fcp2-smCwbo0xQDh7bRUnMunCKguBzOIZjPFZKlHh5Y/edit#gid=0
cond_level0 = (lmis_df['fac_name'].str.contains('Health Post'))
cond_level1a = (lmis_df['fac_type'] == 'Clinic') | (lmis_df['fac_type'] == 'Health Centre') | \
               (lmis_df['fac_name'].str.contains('Clinic')) | (lmis_df['fac_name'].str.contains('Health Centre')) | \
               (lmis_df['fac_name'].str.contains('Maternity')) | (lmis_df['fac_name'] == 'Chilaweni') | \
               (lmis_df['fac_name'] == 'Chimwawa') | (lmis_df['fac_name'].str.contains('Dispensary'))
cond_level1b = (lmis_df['fac_name'].str.contains('Community Hospital')) | \
               (lmis_df['fac_type'] == 'Rural/Community Hospital') | \
               (lmis_df['fac_name'] == 'St Peters Hospital') | \
               (lmis_df['fac_name'] == 'Police College Hospital') | \
               (lmis_df['fac_type'] == 'CHAM') | (lmis_df['fac_owner'] == 'CHAM')
cond_level2 = (lmis_df['fac_type'] == 'District Health Office') | (lmis_df['fac_name'].str.contains('DHO'))
cond_level3 = (lmis_df['fac_type'] == 'Central Hospital') | \
              (lmis_df['fac_name'].str.contains('Central Hospital'))
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
        lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('stkout_days', months_dict[m])]] = \
            (
                lmis_df_wide_flat[('amc', months_dict[m])]
                - lmis_df_wide_flat[('closing_bal', months_dict[m - 1])]
                - lmis_df_wide_flat[('received', months_dict[m])]
            ) / lmis_df_wide_flat[('amc', months_dict[m])] * 31
    elif months_dict[m] in months_dict30:
        lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('stkout_days', months_dict[m])]] = \
            (
                lmis_df_wide_flat[('amc', months_dict[m])]
                - lmis_df_wide_flat[('closing_bal', months_dict[m - 1])]
                - lmis_df_wide_flat[('received', months_dict[m])]
            ) / lmis_df_wide_flat[('amc', months_dict[m])] * 30
    else:
        lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('stkout_days', months_dict[m])]] = \
            (
                lmis_df_wide_flat[('amc', months_dict[m])]
                - lmis_df_wide_flat[('closing_bal', months_dict[m - 1])]
                - lmis_df_wide_flat[('received', months_dict[m])]
            ) / lmis_df_wide_flat[('amc', months_dict[m])] * 28

count_stkout_entries = lmis_df_wide_flat['stkout_days'].count(axis=1).sum()
print(count_stkout_entries, "stockout entries after first interpolation")

# 3.2 --- If any stockout_days < 0 after the above interpolation, update to stockout_days = 0 ---
# RULE: If closing balance[previous month] - dispensed[this month] + received[this month] > 0, stockout == 0
for m in range(1, 13):
    cond1 = lmis_df_wide_flat['stkout_days', months_dict[m]] < 0
    # print("Negative stockout days ", lmis_df_wide_flat.loc[cond1,[('stkout_days', months_dict[m])]].count(axis = 1
    # ).sum())
    lmis_df_wide_flat.loc[cond1, [('data_source', months_dict[m])]] = 'lmis_interpolation_rule2'
    lmis_df_wide_flat.loc[cond1, [('stkout_days', months_dict[m])]] = 0

count_stkout_entries = lmis_df_wide_flat['stkout_days'].count(axis=1).sum()
print(count_stkout_entries, "stockout entries after second interpolation")

lmis_df_wide_flat['consumable_reporting_freq'] = lmis_df_wide_flat['closing_bal'].count(axis=1)  # generate a column
# which reports the number of entries of closing balance for a specific consumable by a facility during the year

# Flatten multilevel columns
lmis_df_wide_flat.columns = [' '.join(col).strip() for col in lmis_df_wide_flat.columns.values]

# 3.3 --- If the consumable was previously reported and during a given month, if any consumable was reported, assume
# 100% days of stckout ---
# RULE: If the balance on a consumable is ever reported and if any consumables are reported during the month, stkout_
# days = number of days of the month
for m in range(1, 13):
    month = 'closing_bal ' + months_dict[m]
    var_name = 'consumables_reported_in_mth ' + months_dict[m]
    lmis_df_wide_flat[var_name] = lmis_df_wide_flat.groupby("fac_name")[month].transform('count')  # generate a column
# which reports the number of consumables for which closing balance was recorded by a facility in a given month

for m in range(1, 13):
    cond1 = lmis_df_wide_flat['consumable_reporting_freq'] > 0
    cond2 = lmis_df_wide_flat['consumables_reported_in_mth ' + months_dict[m]] > 0
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
                                        'data_source', 'consumables_reported_in_mth'],
                       i=['district', 'fac_type_tlo', 'fac_name', 'program', 'item'], j='month',
                       sep=' ', suffix=r'\w+')
lmis = lmis.reset_index()

# 5. LOAD CLEANED MATCHED CONSUMABLE LIST FROM TLO MODEL AND MERGE WITH LMIS DATA ##
#########################################################################################

# 5.1 --- Load and clean data ---
# Import matched list of consumanbles
consumables_df = pd.read_csv(path_for_new_resourcefiles / 'ResourceFile_consumables_matched.csv', low_memory=False)
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

# 2.i. For substitable drugs (within drug category), collapse by taking the product of stkout_prop (OR condition)
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
                         'consumable_reporting_freq': 'first',
                         'consumables_reported_in_mth': 'first'})

# 2.ii. For complementary drugs, collapse by taking the product of (1-stkout_prob)
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
                         'consumable_reporting_freq': 'first',
                         'consumables_reported_in_mth': 'first'})

# 2.iii. For substitutable drugs (within consumable_name_tlo), collapse by taking the product of stkout_prop (OR
# condition).
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
                         'consumable_reporting_freq': 'first',
                         'consumables_reported_in_mth': 'first'})

# Update impossible stockout values (This happens due to some stockout days figures being higher than the number of
#  days in the month)
stkout_df.loc[stkout_df['stkout_prop'] < 0, 'stkout_prop'] = 0
stkout_df.loc[stkout_df['stkout_prop'] > 1, 'stkout_prop'] = 1
# Eliminate duplicates
collapse_dict = {
    'stkout_prop': 'mean', 'closing_bal': 'mean', 'amc': 'mean', 'dispensed': 'mean', 'received': 'mean',
    'consumable_reporting_freq': 'mean', 'consumables_reported_in_mth': 'mean',
    'module_name': 'first', 'consumable_name_tlo': 'first', 'data_source': 'first'
}
stkout_df = stkout_df.groupby(['fac_type_tlo', 'fac_name', 'district', 'month', 'item_code'], as_index=False).agg(
    collapse_dict).reset_index()

stkout_df['available_prop'] = 1 - stkout_df['stkout_prop']

# Some missing values change to 100% stockouts during the aggregation above. Fix this manually
for var in ['stkout_prop', 'available_prop', 'closing_bal', 'amc', 'dispensed', 'received']:
    cond = stkout_df['data_source'].isna()
    stkout_df.loc[cond, var] = np.nan

stkout_df = stkout_df.reset_index()
stkout_df = stkout_df[
    ['module_name', 'district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'consumable_name_tlo',
     'available_prop', 'closing_bal', 'amc', 'dispensed', 'received',
     'data_source', 'consumable_reporting_freq', 'consumables_reported_in_mth']]

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
hhfa_df = pd.read_excel(path_to_files_in_the_tlo_dropbox / 'ResourceFile_hhfa_consumables.xlsx', sheet_name='hhfa_data')

# Use the ratio of availability rates between levels 1b on one hand and levels 2 and 3 on the other to extrapolate
# availability rates for levels 2 and 3 from the HHFA data
cond1b = stkout_df['fac_type_tlo'] == 'Facility_level_1b'
cond2 = stkout_df['fac_type_tlo'] == 'Facility_level_2'
cond3 = stkout_df['fac_type_tlo'] == 'Facility_level_3'
availratio_2to1b = stkout_df[cond2]['available_prop'].mean() / stkout_df[cond1b]['available_prop'].mean()
availratio_3to1b = stkout_df[cond3]['available_prop'].mean() / stkout_df[cond1b]['available_prop'].mean()

# To disaggregate the avalability rates for levels 1b, 2, and 3 from the HHFA, assume that the ratio of availability
# across the three
# levels is the same as that based on OpenLMIS
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

# Add further assumptions on consumable availability from other sources
assumptions_df = pd.read_excel(open(path_to_files_in_the_tlo_dropbox / 'ResourceFile_hhfa_consumables.xlsx', 'rb'),
                               sheet_name='availability_assumptions')
assumptions_df = assumptions_df[['item_code', 'available_prop_Facility_level_0',
                                 'available_prop_Facility_level_1a', 'available_prop_Facility_level_1b',
                                 'available_prop_Facility_level_2', 'available_prop_Facility_level_3']]

# Merge HHFA data with the list of unmatched consumables from the TLO model
unmatched_consumables_df = pd.merge(unmatched_consumables, hhfa_df, how='left', on='item_code')
unmatched_consumables_df = pd.merge(unmatched_consumables_df, assumptions_df, how='left', on='item_code')
# when not missing, replace with assumption

for level in ['0', '1a', '1b', '2', '3']:
    cond = unmatched_consumables_df['available_prop_hhfa_Facility_level_' + level].notna()
    unmatched_consumables_df.loc[cond, 'data_source'] = 'hhfa_2018-19'

    cond = unmatched_consumables_df['available_prop_Facility_level_' + level].notna()
    unmatched_consumables_df.loc[cond, 'data_source'] = 'other'
    unmatched_consumables_df.loc[cond, 'available_prop_hhfa_Facility_level_' + level] = unmatched_consumables_df[
        'available_prop_Facility_level_' + level]

unmatched_consumables_df = unmatched_consumables_df[
    ['module_name', 'item_code', 'consumable_name_tlo_x', 'available_prop_hhfa_Facility_level_0',
     'available_prop_hhfa_Facility_level_1a', 'available_prop_hhfa_Facility_level_1b',
     'available_prop_hhfa_Facility_level_2', 'available_prop_hhfa_Facility_level_3',
     'fac_count_Facility_level_0', 'fac_count_Facility_level_1a', 'fac_count_Facility_level_1b',
     'data_source']]

# Reshape dataframe of consumable availability taken from the HHFA in the same format as the stockout dataframe based
# on OpenLMIS
unmatched_consumables_df = pd.wide_to_long(unmatched_consumables_df, stubnames=['available_prop_hhfa', 'fac_count'],
                                           i=['item_code', 'consumable_name_tlo_x'], j='fac_type_tlo',
                                           sep='_', suffix=r'\w+')

unmatched_consumables_df = unmatched_consumables_df.reset_index()
n = len(unmatched_consumables_df)

# Final cleaning
NameChangeList = [('consumable_name_tlo_x', 'consumable_name_tlo'),
                  ('available_prop_hhfa', 'available_prop')]
change_colnames(unmatched_consumables_df, NameChangeList)

# --- 6.2 Append OpenLMIS stockout dataframe with HHFA stockout dataframe and Extract in .csv format --- #
# Append common consumables stockout dataframe with the main dataframe
cond = unmatched_consumables_df['available_prop'].notna()
unmatched_consumables_df.loc[~cond, 'data_source'] = 'Not available'
stkout_df = stkout_df.append(unmatched_consumables_df)

# --- 6.3 Append stockout rate for facility level 0 from HHFA --- #
cond = hhfa_df['item_code'].notna()
hhfa_fac0 = hhfa_df[cond][
    ['item_code', 'consumable_name_tlo', 'fac_count_Facility_level_0', 'available_prop_hhfa_Facility_level_0']]
NameChangeList = [('fac_count_Facility_level_0', 'fac_count'),
                  ('available_prop_hhfa_Facility_level_0', 'available_prop')]
change_colnames(hhfa_fac0, NameChangeList)
hhfa_fac0['fac_type_tlo'] = 'Facility_level_0'
hhfa_fac0['data_source'] = 'hhfa_2018-19'

hhfa_fac0 = pd.merge(hhfa_fac0, consumables_df[['item_code', 'module_name']], on='item_code', how='inner')
hhfa_fac0 = hhfa_fac0.drop_duplicates()

cond = stkout_df['fac_type_tlo'] == 'Facility_level_0'
stkout_df = stkout_df[~cond]
stkout_df = stkout_df.append(hhfa_fac0)

# --- 6.4 Generate new category variable for analysis --- #
stkout_df['category'] = stkout_df['module_name'].str.lower()
cond_RH = (stkout_df['category'].str.contains('care_of_women_during_pregnancy')) | \
          (stkout_df['category'].str.contains('labour'))
cond_newborn = (stkout_df['category'].str.contains('newborn'))
cond_childhood = (stkout_df['category'] == 'acute lower respiratory infections') | \
                 (stkout_df['category'] == 'measles') | \
                 (stkout_df['category'] == 'diarrhoea')
cond_rti = stkout_df['category'] == 'road traffic injuries'
cond_cancer = stkout_df['category'].str.contains('cancer')
cond_ncds = (stkout_df['category'] == 'epilepsy') | \
            (stkout_df['category'] == 'depression')
stkout_df.loc[cond_RH, 'category'] = 'reproductive_health'
stkout_df.loc[cond_cancer, 'category'] = 'cancer'
stkout_df.loc[cond_newborn, 'category'] = 'neonatal_health'
stkout_df.loc[cond_childhood, 'category'] = 'other_childhood_illnesses'
stkout_df.loc[cond_rti, 'category'] = 'road_traffic_injuries'
stkout_df.loc[cond_ncds, 'category'] = 'ncds'

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
    stkout_df.loc[cond, var] = 'Aggregate'

# --- 6.6 Export final stockout dataframe --- #
stkout_df.to_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")

# Final checks
stkout_df = stkout_df.drop(index=stkout_df.index[pd.isnull(stkout_df.available_prop)])
assert (stkout_df.available_prop >= 0.0).all(), "No Negative values"
assert (stkout_df.available_prop <= 1.0).all(), "No Values greater than 1.0 "
print(stkout_df.loc[(~(stkout_df.available_prop >= 0.0)) | (~(stkout_df.available_prop <= 1.0))].available_prop)
assert not stkout_df.duplicated(['fac_type_tlo', 'fac_name', 'district', 'month', 'item_code']).any(), "No duplicates"

# --- 6.7 Generate file for use in model run --- #
# 1) Smaller file size
# 2) Indexed by the 'Facility_ID' used in the model (which is an amalgmation of district and facility_level, defined in
#  the Master Facilities List.

# unify the set within each facility_id

mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}

sf = stkout_df[['item_code', 'month', 'district', 'fac_type_tlo', 'available_prop']].dropna()
sf.loc[sf.month == 'Aggregate', 'month'] = 'January'  # Assign arbitrary month to data only available at aggregate level
sf.loc[sf.district == 'Aggregate', 'district'] = 'Balaka'  \
    # Assign arbitrary district to data only available at # aggregate level
sf = sf.drop(index=sf.index[(sf.month == 'NA') | (sf.district == 'NA')])
sf.month = sf.month.map(dict(zip(calendar.month_name[1:13], range(1, 13))))
sf.item_code = sf.item_code.astype(int)
sf['fac_type_tlo'] = sf['fac_type_tlo'].str.replace("Facility_level_", "")

# Do some mapping to make the Districts line-up with the definition of Districts in the model
rename_and_collapse_to_model_districts = {
    'Nkhota Kota': 'Nkhotakota',
    'Mzimba South': 'Mzimba',
    'Mzimba North': 'Mzimba',
    'Nkhata bay': 'Nkhata Bay',
}

sf['district_std'] = sf['district'].replace(rename_and_collapse_to_model_districts)
# Take averages (now that 'Mzimba' is mapped-to by both 'Mzimba South' and 'Mzimba North'.)
sf = sf.groupby(by=['district_std', 'fac_type_tlo', 'month', 'item_code'])['available_prop'].mean().reset_index()

# Fill in missing data:
# 1) Cities to get same results as their respective regions
copy_source_to_destination = {
    'Mzimba': 'Mzuzu City',
    'Lilongwe': 'Lilongwe City',
    'Zomba': 'Zomba City',
    'Blantyre': 'Blantyre City'
}

for source, destination in copy_source_to_destination.items():
    new_rows = sf.loc[sf.district_std == source].copy()
    new_rows.district_std = destination
    sf = sf.append(new_rows)

# 2) Fill in Likoma (for which no data) with the means
means = sf.loc[sf.fac_type_tlo.isin(['1a', '1b', '2'])].groupby(by=['fac_type_tlo', 'month', 'item_code'])[
    'available_prop'].mean().reset_index()
new_rows = means.copy()
new_rows['district_std'] = 'Likoma'
sf = sf.append(new_rows)

assert sorted(set(districts)) == sorted(set(pd.unique(sf.district_std)))

# 3) copy the results for 'Mwanza/1b' to be equal to 'Mwanza/1a'.
mwanza_1a = sf.loc[(sf.district_std == 'Mwanza') & (sf.fac_type_tlo == '1a')]
mwanza_1b = sf.loc[(sf.district_std == 'Mwanza') & (sf.fac_type_tlo == '1a')].copy().assign(fac_type_tlo='1b')
sf = sf.append(mwanza_1b)

# 4) Copy all the results to create a level 0 with an availability equal to half that in the respective 1a
all_1a = sf.loc[sf.fac_type_tlo == '1a']
all_0 = sf.loc[sf.fac_type_tlo == '1a'].copy().assign(fac_type_tlo='0')
all_0.available_prop *= 0.5
sf = sf.append(all_0)

# Now, merge-in facility_id
sf_merge = sf.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['district_std', 'fac_type_tlo'],
                    right_on=['District', 'Facility_Level'], how='left', indicator=True)

# 5) Assign the Facility_IDs for those facilities that are regional/national level;
# For facilities of level 3, find which region they correspond to:
districts_with_regional_level_fac = pd.unique(sf_merge.loc[sf_merge.fac_type_tlo == '3'].district_std)
district_to_region_mapping = dict()
for _district in districts_with_regional_level_fac:
    _region = mfl.loc[mfl.District == _district].Region.values[0]
    _fac_id = mfl.loc[(mfl.Facility_Level == '3') & (mfl.Region == _region)].Facility_ID.values[0]
    sf_merge.loc[(sf_merge.fac_type_tlo == '3') & (sf_merge.district_std == _district), 'Facility_ID'] = _fac_id

# National Level
fac_id_of_fac_level4 = mfl.loc[mfl.Facility_Level == '4'].Facility_ID.values[0]
sf_merge.loc[sf_merge.fac_type_tlo == '4', 'Facility_ID'] = fac_id_of_fac_level4

# Now, take averages because more than one set of records is forming the estimates for the level 3 facilities
sf_final = sf_merge.groupby(by=['Facility_ID', 'month', 'item_code'])['available_prop'].mean().reset_index()
sf_final.Facility_ID = sf_final.Facility_ID.astype(int)

# %%
# Construct dataset that conforms to the principles expected by the simulation: i.e. that there is an entry for every
# facility_id and for every month for every item_code.

# Generate the dataframe that has the desired size and shape
fac_ids = set(mfl.loc[mfl.Facility_Level != '5'].Facility_ID)
item_codes = set(sf.item_code.unique())
months = range(1, 13)

full_set = pd.Series(
    index=pd.MultiIndex.from_product([fac_ids, months, item_codes], names=['Facility_ID', 'month', 'item_code']),
    data=np.nan,
    name='available_prop')

# Insert the data, where it is available.
full_set = full_set.combine_first(sf_final.set_index(['Facility_ID', 'month', 'item_code'])['available_prop'])

# Fill in the blanks with rules for interpolation.

facilities_by_level = defaultdict(set)
for ix, row in mfl.iterrows():
    facilities_by_level[row['Facility_Level']].add(row['Facility_ID'])


def get_other_facilities_of_same_level(_fac_id):
    """Return a set of facility_id for other facilities that are of the same level as that provided."""
    for v in facilities_by_level.values():
        if _fac_id in v:
            return v - {_fac_id}


def interpolate_missing_with_mean(_ser):
    """Return a series in which any values that are null are replaced with the mean of the non-missing."""
    if pd.isnull(_ser).all():
        raise ValueError
    return _ser.fillna(_ser.mean())


# Create new dataset that include the interpolations (The operation is not done "in place", because the logic is based
# on what results are missing before the interpolations in other facilities).
full_set_interpolated = full_set * np.nan

for fac in fac_ids:
    for item in item_codes:

        print(f"Now doing: fac={fac}, item={item}")

        # Get records of the availability of this item in this facility.
        _monthly_records = full_set.loc[(fac, slice(None), item)].copy()

        if pd.notnull(_monthly_records).any():
            # If there is at least one record of this item at this facility, then interpolate the missing months from
            # the months for there are data on this item in this facility. (If none are missing, this has no effect).
            _monthly_records = interpolate_missing_with_mean(_monthly_records)

        else:
            # If there is no record of this item at this facility, check to see if it's available at other facilities
            # of the same level
            recorded_at_other_facilities_of_same_level = pd.notnull(
                full_set.loc[(get_other_facilities_of_same_level(fac), slice(None), item)]
            ).any()

            if recorded_at_other_facilities_of_same_level:
                # If it recorded at other facilities of same level, find the average availability of the item at other
                # facilities of the same level.
                _monthly_records = interpolate_missing_with_mean(
                    full_set.loc[(get_other_facilities_of_same_level(fac), slice(None), item)].groupby(level=1).mean()
                )

            else:
                # If it is not recorded at other facilities of same level, then assume it is never available at the
                # facility.
                _monthly_records = _monthly_records.fillna(0.0)

        # Insert values (including corrections) into the resulting dataset.
        full_set_interpolated.loc[(fac, slice(None), item)] = _monthly_records.values

# Check that there are not missing values
assert not pd.isnull(full_set_interpolated).any().any()

# --- Check that the exported file has the properties required of it by the model code. --- #
check_format_of_consumables_file(df=full_set_interpolated.reset_index(), fac_ids=fac_ids)


# %%
# Save
full_set_interpolated.reset_index().to_csv(
    path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv",
    index=False
)

# %%
# 8. CALIBRATION TO HHFA DATA, 2018/19 ##
#########################################################################################
# --- 8.1 Prepare calibration dataframe --- ##
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

# --- 8.2 Compare OpenLMIS estimates with HHFA estimates (CALIBRATION) --- ##
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
# plt.savefig(outputfilepath / 'consumableavailability_calibration_level1a.png')

cond = calibration_df['fac_type_tlo'] == 'Facility_level_1b'
ax = calibration_df[cond].plot.line(x='labels', y=['available_prop', 'available_prop_hhfa'])
ax.set_xticks(np.arange(len(calibration_df[cond]['labels'])))
ax.set_xticklabels(calibration_df[cond]['labels'], rotation=90, fontsize=7)
plt.title('Level 1b', fontsize=size, weight="bold")
# plt.savefig(outputfilepath / 'consumableavailability_calibration_level1b.png')

cond = calibration_df['fac_type_tlo'] == 'Facility_level_2'
ax = calibration_df[cond].plot.line(x='labels', y=['available_prop', 'available_prop_hhfa'])
ax.set_xticks(np.arange(len(calibration_df[cond]['labels'])))
ax.set_xticklabels(calibration_df[cond]['labels'], rotation=90, fontsize=7)
plt.title('Level 2', fontsize=size, weight="bold")
plt.show()
# plt.savefig(outputfilepath / 'consumableavailability_calibration_level2.png')

cond = calibration_df['fac_type_tlo'] == 'Facility_level_3'
ax = calibration_df[cond].plot.line(x='labels', y=['available_prop', 'available_prop_hhfa'])
ax.set_xticks(np.arange(len(calibration_df[cond]['labels'])))
ax.set_xticklabels(calibration_df[cond]['labels'], rotation=90, fontsize=7)
plt.title('Level 3', fontsize=size, weight="bold")
# plt.savefig(outputfilepath / 'consumableavailability_calibration_level3.png')
