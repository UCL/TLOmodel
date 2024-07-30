"""
This script generates estimates of availability of consumables in 2021 and 2023
"""
import calendar
import copy
import datetime
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from tlo.methods.consumables import check_format_of_consumables_file

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    '/Users/sm2511/Dropbox/Thanzi la Onse'
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
figurespath = Path(outputfilepath / 'openlmis_data')
figurespath.mkdir(parents=True, exist_ok=True) # create directory if it doesn't exist

# Define relevant dictionaries for analysis
months_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
               9: 'September', 10: 'October', 11: 'November', 12: 'December'}
years_dict = {1: 2018, 2: 2021, 3: 2022, 4: 2023}
districts_dict = {1: 'Balaka', 2: 'Blantyre', 3: 'Chikwawa', 4: 'Chiradzulu', 5: 'Chitipa', 6: 'Dedza',
                  7: 'Dowa', 8: 'Karonga', 9: 'Kasungu', 10: 'Lilongwe', 11: 'Machinga', 12: 'Mangochi',
                  13: 'Mchinji', 14: 'Mulanje', 15: 'Mwanza', 16: 'Mzimba North', 17: 'Mzimba South',
                  18: 'Neno', 19: 'Nkhata bay', 20: 'Nkhota Kota', 21: 'Nsanje', 22: 'Ntcheu', 23: 'Ntchisi',
                  24: 'Phalombe', 25: 'Rumphi', 26: 'Salima', 27: 'Thyolo', 28: 'Zomba'}

# Define function to provide consistent column names across excel files
def rename_lmis_columns(_df):
    columns_to_rename = {
        'District Name': 'district',
        'Code': 'fac_code',
        'Managed By': 'fac_owner',
        'Dispensed': 'dispensed',
        'AMC': 'average_monthly_consumption',
        'MOS': 'months_of_stock',
        'Availability': 'available_days',
        'Facility Name': 'fac_name',
        # 2018 dataset variables
        'Name (Geographic Zones)': 'district',
        'Name (Facility Operators)': 'fac_owner',
        'Name': 'fac_name',
        'Name (Facility Types)': 'fac_type',
    }

    # Iterating through existing columns to find matches and rename
    for col in _df.columns:
        if col.lower().replace(" ", "") in ['quantityreceived', 'receivedquantity', 'received']:
            columns_to_rename[col] = 'qty_received'
        elif col.lower().replace(" ", "") in ['quantityissued', 'issuedquantity', 'dispensed']:
            columns_to_rename[col] = 'dispensed'
        elif 'program' in col.lower():
            columns_to_rename[col] = 'program'
        elif 'year' in col.lower():
            columns_to_rename[col] = 'year'
        elif col.lower().replace(" ", "") in ['rmonth', 'month']:
            columns_to_rename[col] = 'month'
        elif col.lower().replace(" ", "") in ['closingbal', 'closingbalance']:
            columns_to_rename[col] = 'closing_bal'
        elif 'productname' in col.lower().replace(" ", ""):
            columns_to_rename[col] = 'item'
        elif 'stockoutdays' in col.lower().replace(" ", ""):
            columns_to_rename[col] = 'stkout_days'

    _df.rename(columns=columns_to_rename, inplace=True)

# Function to remove commas and convert to numeric
def clean_convert(x):
    if isinstance(x, str):  # Checks if the element is a string
        return pd.to_numeric(x.replace(',', ''))
    return x  # Returns the element unchanged if it's not a string

# Define function to assign facilities to TLO model levels based on facility names
# See link: https://docs.google.com/spreadsheets/d/1fcp2-smCwbo0xQDh7bRUnMunCKguBzOIZjPFZKlHh5Y/edit#gid=0
def assign_facilty_level_based_on_facility_names_or_types(_df):
    cond_level_0 = (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('healthpost'))
    cond_level_1a = (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('clinic')) | \
                    (_df['fac_name'].str.replace(' ', '').str.replace('center','centre').str.lower().str.contains('healthcentre')) | \
                    (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('maternity')) | \
                    (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('dispensary')) | \
                    (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('opd')) | \
                    (_df['fac_type'] == 'Clinic') | (_df['fac_type'] == 'Health Centre')
    cond_level_1b = (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('communityhospital')) | \
                   (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('ruralhospital')) | \
                   (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('stpetershospital')) | \
                   (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('policecollegehospital')) | \
                   (_df['fac_owner'] == 'CHAM') | (_df['fac_type'] == 'CHAM') | \
                   (_df['fac_type'] == 'Rural/Community Hospital')
    cond_level_2 =  (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('districthealthoffice')) | \
                   (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('districthospital')) | \
                   (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('dho')) | \
                    (_df['fac_type'] == 'District Health Office')
    cond_level_3 =  (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('centralhospital')) | \
                    (_df['fac_type'] == 'Central Hospital')
    cond_level_4 = (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('zombamentalhospital'))
    _df['fac_level'] = ''
    _df.loc[cond_level_0, 'fac_level'] = '0'
    _df.loc[cond_level_1a, 'fac_level'] = '1a'
    _df.loc[cond_level_1b, 'fac_level'] = '1b'
    _df.loc[cond_level_2, 'fac_level'] = '2'
    _df.loc[cond_level_3, 'fac_level'] = '3'
    _df.loc[cond_level_4, 'fac_level'] = '4'
    print("The following facilities have been assumed to be level 1a \n", _df[_df.fac_level == '']['fac_name'].unique())
    _df.loc[_df.fac_level == '', 'fac_level'] = '1a'

def generate_summary_heatmap(_df,
                             x_var,
                             y_var,
                             value_var,
                             value_label,
                             summary_func='mean'):
    # Get the appropriate function from the pandas Series object
    if hasattr(pd.Series, summary_func):
        agg_func = getattr(pd.Series, summary_func)
    else:
        raise ValueError(f"Unsupported summary function: {summary_func}")

    heatmap_df = _df.groupby([x_var, y_var])[value_var].apply(agg_func).reset_index()
    heatmap_df = heatmap_df.pivot(x_var, y_var, value_var)

    # Calculate the aggregate row and column
    #aggregate_col = heatmap_df.apply(agg_func, axis=0)
    #aggregate_row = heatmap_df.apply(agg_func, axis=1)
    #overall_aggregate = agg_func(heatmap_df.values.flatten())

    # Add aggregate row and column
    #heatmap_df['Average'] = aggregate_row
    #aggregate_col['Average'] = overall_aggregate
    #heatmap_df.loc['Average'] = aggregate_col

    # Generate the heatmap
    sns.set(font_scale=0.9)
    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_df, annot=True, cmap='RdYlGn', fmt='.1f',
                cbar_kws={'label': value_label})

    # Customize the plot
    plt.title('')
    plt.xlabel(y_var)
    plt.ylabel(x_var)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.savefig(figurespath / f'{value_var}_{x_var}_{y_var}.png', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()

# %%
# 1. DATA IMPORT AND CLEANING ##
#########################################################################################
number_of_months = 3 # 13
for y in range(2,len(years_dict)+1): # the format of the 2018 dataset received was different from the other years so we start at 2 here
    print("processing year ", years_dict[y])
    for m in range(1, number_of_months):
        print("processing month ", months_dict[m])
        if ((m == 1) & (y == 2)):
            lmis = pd.read_excel(path_to_files_in_the_tlo_dropbox / f'OpenLMIS/{years_dict[y]}/{months_dict[m]}.xlsx')
            lmis['month'] = months_dict[m]
            lmis['year'] = years_dict[y]
            rename_lmis_columns(lmis)
        else:
            monthly_lmis = pd.read_excel(path_to_files_in_the_tlo_dropbox / f'OpenLMIS/{years_dict[y]}/{months_dict[m]}.xlsx')
            monthly_lmis['month'] = months_dict[m]
            monthly_lmis['year'] = years_dict[y]
            rename_lmis_columns(monthly_lmis)
            lmis = pd.concat([lmis, monthly_lmis], axis=0, ignore_index=True)
lmis_raw = copy.deepcopy(lmis)
rename_lmis_columns(lmis)

# append 2018 data
col_list = ['year', 'month', 'district', 'fac_owner', 'fac_name', 'program', 'item', 'closing_bal', 'dispensed', 'stkout_days',
            'average_monthly_consumption', 'qty_received', 'fac_type']
lmis_2018 = pd.read_csv(path_to_files_in_the_tlo_dropbox / 'OpenLMIS/ResourceFile_LMIS_2018.csv', low_memory=False)
rename_lmis_columns(lmis_2018)

lmis['fac_type'] = np.nan # create empty column to match with the col_list in the 2018 dataframe
lmis = pd.concat([lmis[col_list], lmis_2018[col_list]], axis=0, ignore_index=True)

# Drop empty rows
lmis = lmis[lmis.fac_name.notna()]

# Remove Private Health facilities from the data
cond_pvt = (lmis['fac_owner'] == 'Private') | (lmis['fac_type'].str.contains("Private"))
lmis = lmis[~cond_pvt]

# Clean ownership information
lmis.loc[lmis.fac_type.isna(), 'fac_type'] = "Unknown"
cond_cham = lmis['fac_type'].str.contains("CHAM")
lmis.loc[cond_cham, 'fac_owner'] = "CHAM"
cond_other = lmis.fac_owner.isna() # because private facilities are dropped and CHAM facilities are correctly categorised
lmis.loc[cond_other, 'fac_owner'] = "Government of Malawi"

# Clean facility types to match with types in the TLO model
assign_facilty_level_based_on_facility_names_or_types(lmis)
lmis = lmis.drop('fac_type', axis = 1) # redundant column

# Update month coding to numeric
months_reversed_dict = {v: k for k, v in months_dict.items()}
lmis['month'] = lmis['month'].map(months_reversed_dict)

# Check the number of facilities at higher levels matches actual for all years
full_district_set = set(districts_dict.values())
for y in range(1,len(years_dict)+1):
    for m in range(1, number_of_months):
        #print("Checking consistency in data for ", months_dict[m], ", ",  years_dict[y])
        monthly_lmis = lmis[(lmis.month == months_dict[m]) & (lmis.year == years_dict[y])]
        #assert(len(monthly_lmis[monthly_lmis.fac_level == '2']['fac_name'].unique()) == 28) # number of District hospitals
        #assert (len(monthly_lmis[monthly_lmis.fac_level == '3']['fac_name'].unique()) == 4)  # number of Central hospitals
        #assert (len(monthly_lmis[monthly_lmis.fac_level == '4']['fac_name'].unique()) == 1)  # Zomba Mental Hospital
        #assert (len(monthly_lmis[monthly_lmis.stkout_days.notna()]) != 0)  # non-empty data
        districts_in_data = set(monthly_lmis[monthly_lmis.fac_level == '2']['district'].unique())
        if (districts_in_data != full_district_set):
            print(years_dict[y], months_dict[m], ": Districts not in data", full_district_set.difference(districts_in_data))
        central_hospitals = monthly_lmis[monthly_lmis.fac_level == '3']['fac_name'].unique()
        if len(central_hospitals) != 4:
            print(years_dict[y], months_dict[m], ": only the following central hospitals included - ", central_hospitals)
        mental_hospitals = monthly_lmis[monthly_lmis.fac_level == '4']['fac_name'].unique()
        if len(mental_hospitals) != 1:
            print(years_dict[y], months_dict[m], ": mental hospitals included - ", mental_hospitals)

        # There is data from levels 0, 1a, and 1b from all districts
        for level in ['0', '1a', '1b']:
            if (len(monthly_lmis[monthly_lmis.fac_level == level]) == 0):
                print(years_dict[y], months_dict[m], ": No facilities reporting at level ", level)
            #assert(len(monthly_lmis[monthly_lmis.fac_level == level]) != 0)

print('Data import complete and ready for analysis; verified that data is complete')
# TODO extract the above missing data as a summary table

# Clean inconsistent names of consumables across years
#--------------------------------------------------------
# Extract a list of consumables along with the year_month during which they were reported
#lmis['program_item'] = lmis['program'].astype(str) + "---" +  lmis['item'].astype(str)
#consumable_reporting_rate = lmis.groupby(['program_item', 'year_month'])['fac_name'].nunique().reset_index()
#consumable_reporting_rate = consumable_reporting_rate.pivot( 'program_item', 'year_month', 'fac_name')
#consumable_reporting_rate.to_csv(figurespath / 'consumable_reporting_rate.csv')


## TEMPORARY CODE TO FACILITATE CLEANING
# Select a subset of items and facilities for initial cleaning
unique_items = lmis['item'].unique()
unique_items_df = pd.DataFrame(unique_items, columns=['item'])
sampled_items = unique_items_df.sample(frac=0.1, random_state=1)

unique_facilities = lmis['fac_name'].unique()
unique_facilities_df = pd.DataFrame(unique_facilities, columns=['fac_name'])
sampled_facilities = unique_facilities_df.sample(frac=0.1, random_state=1)
# Filter the original DataFrame to only include rows with sampled items
lmis = lmis[lmis['item'].isin(sampled_items['item'])]
lmis = lmis[lmis['fac_name'].isin(sampled_facilities['fac_name'])]

# Import manually cleaned list of consumable names
clean_consumables_names = pd.read_csv(path_to_files_in_the_tlo_dropbox / 'OpenLMIS/cleaned_consumable_names.csv', low_memory = False)[['Program', 'Consumable','Alternate consumable name',	'Substitute group']]
# Create a dictionary of cleaned consumable name categories
cons_alternate_name_dict = clean_consumables_names[clean_consumables_names['Consumable'] != clean_consumables_names['Alternate consumable name']].set_index('Consumable')['Alternate consumable name'].to_dict()
cond_substitute_available =  clean_consumables_names['Substitute group'].notna()
cond_substitute_different_from_original =  clean_consumables_names['Consumable'] != clean_consumables_names['Substitute group']
cons_substitutes_dict = clean_consumables_names[cond_substitute_available & cond_substitute_different_from_original].set_index('Consumable')['Substitute group'].to_dict()
def rename_items_to_address_inconsistentencies(_df, item_dict):
    """Return a dataframe with rows for the same item with inconsistent names collapsed into one"""
    old_unique_item_count = _df.item.nunique()

    # replace item names with alternate names - these are mostly spelling inconsistencies
    _df['item'].replace(cons_alternate_name_dict, inplace = True)


    # Make a list of column names to be collapsed using different methods
    columns_to_sum = ['closing_bal', 'dispensed', 'stkout_days', 'average_monthly_consumption', 'qty_received']
    columns_to_preserve = ['program']

    # Remove commas and convert to numeric
    _df[columns_to_sum] = _df[columns_to_sum].applymap(clean_convert)
    _df[columns_to_preserve] = _df[columns_to_preserve].astype(str)

    def custom_agg(x):
        if x.name in columns_to_sum:
            return x.sum(skipna=True) if np.any(
                x.notnull() & (x >= 0)) else np.nan  # this ensures that the NaNs are retained
        # , i.e. not changed to 0, when the corresponding data for both item name variations are NaN, and when there
        # is a 0 or positive value for one or both item name variation, the sum is taken.
        elif x.name in columns_to_preserve:
            return x.iloc[0]  # this function extracts the first value

    # Collapse dataframe
    _collapsed_df = _df.groupby(['item', 'district', 'fac_level', 'fac_owner', 'fac_name', 'year', 'month']).agg(
        {col: custom_agg for col in columns_to_preserve + columns_to_sum}
    ).reset_index()

    # Test that all items in the dictionary have been found in the dataframe
    new_unique_item_count = _collapsed_df.item.nunique()
    #assert len(item_dict) == old_unique_item_count - new_unique_item_count
    return _collapsed_df

# Hold out the dataframe with no naming inconsistencies
list_of_items_with_inconsistent_names_zipped = set(zip(cons_alternate_name_dict.keys(), cons_alternate_name_dict.values()))
list_of_items_with_inconsistent_names = [item for sublist in list_of_items_with_inconsistent_names_zipped for item in sublist]
df_with_consistent_item_names =  lmis[~lmis['item'].isin(list_of_items_with_inconsistent_names)]
df_without_consistent_item_names = lmis[lmis['item'].isin(list_of_items_with_inconsistent_names)]

# Make inconsistently named drugs uniform across the dataframe
df_without_consistent_item_names_corrected = rename_items_to_address_inconsistentencies(
    df_without_consistent_item_names, cons_alternate_name_dict)
# Append holdout and corrected dataframes
lmis = pd.concat([df_without_consistent_item_names_corrected, df_with_consistent_item_names],
                              ignore_index=True)

#lmis_with_updated_names.to_csv(figurespath / 'lmis_with_updated_names.csv')
#lmis_with_updated_names = pd.read_csv(figurespath / 'lmis_with_updated_names.csv', low_memory = False)[1:300000]
#lmis_with_updated_names = lmis_with_updated_names.drop('Unnamed: 0', axis = 1)

# Drop months which have inconsistent data
#-------------------------------------------
# July 2021 records negative stockout days and infeasibly high stockout days

# Feb and Mar 2023 record too many consumables

# Browse data missingness

# Interpolation to address missingness
#-------------------------------------------
# Ensure that the columns that should be numerical are numeric
numeric_cols = ['average_monthly_consumption', 'stkout_days', 'qty_received', 'closing_bal', 'month']
lmis[numeric_cols] = lmis[numeric_cols].applymap(clean_convert)
lmis['data_source'] = 'open_lmis_data'

# Prepare dataset for interpolation
def create_full_dataset(_df):
    _df.reset_index(inplace=True, drop = True)

    # Preserve columns whose data needs to be duplicated into the new rows added
    facility_features_to_preserve = ['district', 'fac_level', 'fac_owner', 'fac_name']
    consumable_features_to_preserve = ['program', 'item']
    facility_features = _df[facility_features_to_preserve]
    facility_features = facility_features[~facility_features.duplicated(keep='first')]
    consumable_features = _df[consumable_features_to_preserve]
    consumable_features = consumable_features[~consumable_features.item.duplicated(keep='first')]
    _df = _df.drop(['district', 'fac_level', 'fac_owner', 'program'], axis = 1) # drop columns which will be preserved

    # create a concatenated version of year and month
    #_df['year_month'] = _df['year'].astype(str) + "_" +  _df['month'].astype(str) # concatenate month and year for plots

    # Make sure that there is a row for every item, facility, year and month
    unique_facilities = _df['fac_name'].unique()
    unique_items = _df['item'].unique()
    all_months = list(months_dict.keys())
    all_years = list(years_dict.values())
    # Create MultiIndex from product of unique facilities and all months
    multi_index = pd.MultiIndex.from_product([unique_facilities, unique_items, all_years, all_months], names=['fac_name', 'item', 'year', 'month'])

    # Reindex dataframe
    _df.set_index(['fac_name', 'item', 'year', 'month'], inplace=True)
    _df = _df[~_df.index.duplicated(keep='first')] # this has occured when an item was reported under 2 programs - A quick glance at the data suggested that all the data for these rows was duplicated
    _df = _df.reindex(multi_index)
    _df.reset_index(inplace=True)

    # Add preserved columns
    _df = _df.merge(facility_features, on='fac_name', how='left', validate = "m:1")
    _df = _df.merge(consumable_features, on='item', how='left', validate = "m:1")

    return _df

# TODO some columns can be dropped fac_name	item	year	month_num	index	district	fac_level	fac_owner	month	program	closing_bal	dispensed	stkout_days	average_monthly_consumption	qty_received	fac_type	year_month	program_item	opening_bal

def prepare_data_for_interpolation(_df):
    # Reset index as columns and sort values for interpolation
    _df = _df.sort_values(by = ['fac_name', 'item', 'year', 'month'])

    # Create columns for opening_balance and the number of days in the month
    _df['opening_bal'] = _df['closing_bal'].shift(1) # closing balance of the previous month
    # Define lists of months with the same number of days
    months_dict31 = [1, 3, 5, 7, 8, 10, 12]
    months_dict30 = [4, 6, 9, 11]
    _df['month_length'] = 31
    cond_30 = _df['month'].isin(months_dict30)
    cond_31 = _df['month'].isin(months_dict31)
    _df.loc[cond_30, 'month_length'] = 30
    _df.loc[~(cond_30 | cond_31), 'month_length'] = 28

    # Before interpolation drop all the consumables which were never reported
    consumable_reporting_rate = _df.groupby('item')['closing_bal'].count().reset_index()
    list_of_consumables_never_reported = consumable_reporting_rate[consumable_reporting_rate.closing_bal == 0].item.unique()
    cond_items_never_reported = _df.item.isin(list_of_consumables_never_reported)
    _df = _df[~cond_items_never_reported]
    print(f"{len(list_of_consumables_never_reported)} items never reported dropped from dataframe")

    return _df

lmis = create_full_dataset(lmis)
lmis = prepare_data_for_interpolation(lmis)

# # RULE 1 --- If i) stockout is missing (or negative?), ii) closing_bal, amc and received are not missing , and iii) amc !=0 and,
# #          then stkout_days[m] = (amc[m] - closing_bal[m-1] - received)/amc * number of days in the month ---
# # (Note that the number of entries for closing balance, dispensed and received is always the same)

# If any stockout_days < 0 after the above interpolation, update to stockout_days = 0; if stkout_days > month_length, update to month_length ---
# If closing balance[previous month] - dispensed[this month] + received[this month] > 0, stockout == 0
def interpolation1_using_other_cols(_df):
    cond_stkout_missing = _df['stkout_days'].isna()
    cond_otherdata_available = _df['closing_bal'].notna() & _df['average_monthly_consumption'].notna() & _df['qty_received'].notna()
    cond_interpolation_1 = cond_stkout_missing & cond_otherdata_available
    _df.loc[cond_interpolation_1, 'stkout_days'] = (_df['average_monthly_consumption'] - _df['opening_bal']- _df['qty_received'])/_df['average_monthly_consumption'] * _df['month_length']

    count_stkout_entries_interpolation1 = _df['stkout_days'].notna().sum()
    print(f"{count_stkout_entries_interpolation1} ({round(count_stkout_entries_interpolation1 / len(_df) * 100, 2)}%) stockout entries after the first interpolation")
    _df['data_source'] = 'lmis_interpolation_rule_1'

    return _df

def correct_infeasible_stockout_days(_df):
    _df.loc[_df['stkout_days'] <0, 'stkout_days'] = 0
    _df.loc[(_df['stkout_days'] > _df['month_length']), 'stkout_days'] = _df['month_length']
    return _df

count_stkout_entries = lmis['stkout_days'].notna().sum()
print(f"{count_stkout_entries} ({round(count_stkout_entries/len(lmis) * 100, 2) }%) stockout entries in original data") # TODO Add percentage of values missing

lmis = interpolation1_using_other_cols(lmis)
lmis = correct_infeasible_stockout_days(lmis)

# RULE 2 --- If the consumable was previously reported and during a given month, if any consumable was reported, assume
# 100% days of stckout ---
# If the balance on a consumable is ever reported and if any consumables are reported during the month, stkout_
# days = number of days of the month

def interpolation2_using_reporting_omission(_df):
    # A. We need one column providing the number of times a consumable has been reported by a facility during a given year
    annual_reporting_frequency_of_cons_by_fac = _df.groupby(['item', 'fac_name', 'year']).apply(lambda x: (x['closing_bal'] >= 0).sum())
    annual_reporting_frequency_of_cons_by_fac = annual_reporting_frequency_of_cons_by_fac.reset_index().rename(columns ={0: 'annual_reporting_frequency_of_cons_by_fac'})
    _df = pd.merge(_df, annual_reporting_frequency_of_cons_by_fac, on = ['item', 'fac_name', 'year'], validate = "m:1", how = "left")
    # TODO make a decision on whether closing_bal recorded as 0 should be counted -
    # See _df[_df.item.str.contains("Rifampicin 75mg+") & (_df.fac_name == "African Bible College Clinic")][['closing_bal', 'year', 'annual_reporting_frequency_of_cons_by_fac']]

    # B. We need one column providing the number of items which have been reported by the facility as available during a given month
    monthly_records_submitted_by_fac = _df.groupby(['fac_name', 'year', 'month']).apply(lambda x: (x['closing_bal'] >= 0).sum())
    monthly_records_submitted_by_fac = monthly_records_submitted_by_fac.reset_index().rename(columns ={0: 'monthly_records_submitted_by_fac'})
    # TODO make a decision on whether closing_bal recorded as 0 should be counted
    _df = pd.merge(_df, monthly_records_submitted_by_fac, on = ['fac_name', 'year', 'month'], validate = "m:1", how = "left")

    # If A > 0, and B > 0, then the consumable must be missing and hence has not been included in the OpenLMIS report
    cond_a = _df['annual_reporting_frequency_of_cons_by_fac'] > 0
    cond_b = _df['monthly_records_submitted_by_fac'] > 0
    cond_stkout_missing = _df['stkout_days'].isna()
    _df.loc[cond_a & cond_b & cond_stkout_missing, 'stkout_days'] = _df['month_length']

    count_stkout_entries_interpolation2 = _df['stkout_days'].notna().sum()
    print(f"{count_stkout_entries_interpolation2} ({round(count_stkout_entries_interpolation2/len(_df) * 100, 2)}%) stockout entries after the second interpolation")

    _df.drop(columns=['monthly_records_submitted_by_fac', 'annual_reporting_frequency_of_cons_by_fac'], inplace=True)
    _df['data_source'] = 'lmis_interpolation_rule_2'
    return _df

lmis = interpolation2_using_reporting_omission(lmis)

# RULE 3 --- If a facility did not report data for a given month, assume same as the average of the three previous months
# Calculate the average stockout days for the previous three months
def interpolation3_using_previous_months_data(_df):
    _df['stkout_days_t-1'] = _df['stkout_days'].shift(1) / _df['month_length'].shift(1)
    _df['stkout_days_t-2'] = _df['stkout_days'].shift(2) / _df['month_length'].shift(2)
    _df['stkout_days_t-3'] = _df['stkout_days'].shift(3) / _df['month_length'].shift(3)
    _df['stkout_days_3mth_moving_average'] = _df[['stkout_days_t-1', 'stkout_days_t-2', 'stkout_days_t-3']].dropna(axis=0, how='any').mean(axis=1) * _df['month_length']

    cond_stkout_missing = _df['stkout_days'].isna()
    cond_stkout_moving_average_available = _df['stkout_days_3mth_moving_average'].notna()
    _df.loc[cond_stkout_missing  & cond_stkout_moving_average_available, 'stkout_days'] = _df['stkout_days_3mth_moving_average']

    count_stkout_entries_interpolation3 = _df['stkout_days'].notna().sum()
    print(f"{count_stkout_entries_interpolation3} ({round(count_stkout_entries_interpolation3 / len(_df) * 100, 2)}%) stockout entries after the third interpolation")

    _df.drop(columns = ['stkout_days_t-1', 'stkout_days_t-2', 'stkout_days_t-3','stkout_days_3mth_moving_average' ], inplace = True)
    _df['data_source'] = 'lmis_interpolation_rule_3'
    return _df

lmis = interpolation3_using_previous_months_data(lmis)

# 4. CALCULATE STOCK OUT RATES BY MONTH and FACILITY ##
#########################################################################################
lmis['stkout_prob'] = lmis['stkout_days']/lmis['month_length']
#lmis_with_updated_names.to_csv(figurespath / 'lmis_with_prob.csv')
#lmis_with_updated_names = pd.read_csv(figurespath / 'lmis_with_prob.csv', low_memory = False)[1:300000]
#lmis_with_updated_names = lmis_with_updated_names.drop('Unnamed: 0', axis = 1)


# Update probability of stockout when there are substitutes
def update_availability_for_substitutable_consumables(_df, groupby_list):
    """Return a dataframe with 'avaiilable_prob' updated for substitutable consumables"""
    # Define column lists based on the aggregation function to be applied
    columns_to_multiply = ['stkout_prob'] # a consumable is stocked out if all its substitutes are also stocked out
    columns_to_sum = ['closing_bal', 'average_monthly_consumption', 'dispensed', 'qty_received']
    columns_to_preserve = []

    # Define aggregation function to be applied to collapse data by item
    def custom_agg_stkout(x):
        if x.name in columns_to_multiply:
            return x.prod(skipna=True) if np.any(
                x.notnull() & (x >= 0)) else np.nan  # this ensures that the NaNs are retained
        elif x.name in columns_to_sum:
            return x.sum(skipna=True) if np.any(
                x.notnull() & (x >= 0)) else np.nan  # this ensures that the NaNs are retained
        # , i.e. not changed to 1, when the corresponding data for both item name variations are NaN, and when there
        # is a 0 or positive value for one or both item name variation, the sum is taken.
        elif x.name in columns_to_preserve:
            return x.iloc[0]  # this function extracts the first value

    # Collapse dataframe
    _collapsed_df = _df.groupby(groupby_list).agg(
        {col: custom_agg_stkout for col in columns_to_multiply + columns_to_sum + columns_to_preserve}
    ).reset_index()

    _collapsed_df['data_source'] = 'lmis_adjusted_for_substitutes'

    return _collapsed_df

# Hold out the dataframe with no substitutes
list_of_items_with_substitutes_zipped = set(zip(cons_substitutes_dict.keys(), cons_substitutes_dict.values()))
list_of_items_with_substitutes = [item for sublist in list_of_items_with_substitutes_zipped for item in sublist]
df_with_no_substitutes =  lmis[~lmis['item'].isin(list_of_items_with_substitutes)]
df_with_substitutes = lmis[lmis['item'].isin(list_of_items_with_substitutes)]

# Update the names of drugs with substitutes to a common name
df_with_substitutes['substitute'] = df_with_substitutes['item'].replace(cons_substitutes_dict)
groupby_list = ['district',	'fac_level', 'fac_owner', 'fac_name', 'substitute', 'year']
df_with_substitutes_adjusted = update_availability_for_substitutable_consumables(df_with_substitutes, groupby_list)
df_with_substitutes_adjusted = pd.merge(df_with_substitutes.drop(['stkout_prob', 'data_source'], axis = 1), df_with_substitutes_adjusted[groupby_list + ['stkout_prob', 'data_source']], on = groupby_list, validate = "m:1", how = 'left')

# Append holdout and corrected dataframes
lmis = pd.concat([df_with_substitutes_adjusted, df_with_no_substitutes],
                              ignore_index=True)

# Calculate the probability of consumables being available (converse of stockout)
lmis['available_prob'] = 1 - lmis['stkout_prob']

# 5. LOAD CLEANED MATCHED CONSUMABLE LIST FROM TLO MODEL AND MERGE WITH LMIS DATA ##
####################################################################################
# Load and clean data
# Import matched list of consumanbles
tlo_lmis_mapping = pd.read_csv(path_for_new_resourcefiles / 'ResourceFile_consumables_matched.csv', low_memory=False,
                             encoding="ISO-8859-1")
cond_remove = tlo_lmis_mapping['matching_status'] == 'Remove'
tlo_lmis_mapping = tlo_lmis_mapping[~cond]  # Remove items which were removed due to updates or the existence of duplicates

# Keep only the correctly matched consumables for stockout analysis based on OpenLMIS
cond1 = tlo_lmis_mapping['matching_status'] == 'Matched'
cond2 = tlo_lmis_mapping['verified_by_DM_lead'] != 'Incorrect'
tlo_lmis_mapping = tlo_lmis_mapping[cond1 & cond2]

# Rename columns
tlo_lmis_mapping.rename(columns = {'consumable_name_lmis': 'item'}, inplace = True)

'''
# Update matched consumable name where the name in the OpenLMIS data was updated in September
def replace_old_item_names_in_lmis_data(_df, item_dict):
    """Return a dataframe with old LMIS consumable names replaced with the new name"""
    for item in item_dict:
        cond_oldname = _df.item == item_dict[item]
        _df.loc[cond_oldname, 'item'] = item
    return _df


matched_consumables = replace_old_item_names_in_lmis_data(matched_consumables, inconsistent_item_names_mapping)
'''

# Merge data with LMIS data
tlo_cons_availability = pd.merge(lmis, tlo_lmis_mapping, how='inner', on='item')
#tlo_cons_availability = tlo_cons_availability.sort_values('data_source')

# Aggregate substitutes and complements
def collapse_stockout_data(_df, groupby_list, var):
    """Return a dataframe with rows for the same TLO model item code collapsed into 1"""
    # Define column lists based on the aggregation function to be applied
    columns_to_multiply = [var]
    columns_to_sum = ['closing_bal', 'average_monthly_consumption', 'dispensed', 'qty_received']
    columns_to_preserve = ['data_source']

    # Define aggregation function to be applied to collapse data by item
    def custom_agg_stkout(x):
        if x.name in columns_to_multiply:
            return x.prod(skipna=True) if np.any(
                x.notnull() & (x >= 0)) else np.nan  # this ensures that the NaNs are retained
        elif x.name in columns_to_sum:
            return x.sum(skipna=True) if np.any(
                x.notnull() & (x >= 0)) else np.nan  # this ensures that the NaNs are retained
        # , i.e. not changed to 1, when the corresponding data for both item name variations are NaN, and when there
        # is a 0 or positive value for one or both item name variation, the sum is taken.
        elif x.name in columns_to_preserve:
            return x.iloc[0]  # this function extracts the first value

    # Collapse dataframe
    _collapsed_df = _df.groupby(groupby_list).agg(
        {col: custom_agg_stkout for col in columns_to_multiply + columns_to_sum + columns_to_preserve}
    ).reset_index()

    return _collapsed_df

# 2.i. For substitable drugs (within drug category), collapse by taking the product of stkout_prop (OR condition)
# This represents Pr(all substitutes with the item code are stocked out)
groupby_list1 = ['module_name', 'district', 'fac_level', 'fac_name', 'year', 'month', 'item_code', 'consumable_name_tlo',
                 'match_level1',
                 'match_level2']
tlo_cons_availability = collapse_stockout_data(tlo_cons_availability, groupby_list1, 'stkout_prob')

# 2.ii. For complementary drugs, collapse by taking the product of (1-stkout_prob)
# This represents Pr(All drugs within item code (in different match_group's) are available)
tlo_cons_availability['available_prob'] = 1 - tlo_cons_availability['stkout_prob']
groupby_list2 = ['module_name', 'district', 'fac_level', 'fac_name', 'year', 'month', 'item_code', 'consumable_name_tlo',
                 'match_level2']
tlo_cons_availability = collapse_stockout_data(tlo_cons_availability, groupby_list2, 'available_prob')

# 2.iii. For substitutable drugs (within consumable_name_tlo), collapse by taking the product of stkout_prop (OR
# condition).
# This represents Pr(all substitutes with the item code are stocked out)
tlo_cons_availability['stkout_prob'] = 1 - tlo_cons_availability['available_prob']
groupby_list3 = ['module_name', 'district', 'fac_level', 'fac_name', 'year', 'month', 'item_code', 'consumable_name_tlo']
tlo_cons_availability = collapse_stockout_data(tlo_cons_availability, groupby_list3, 'stkout_prob')

# Update impossible stockout values (This happens due to some stockout days figures being higher than the number of
#  days in the month)
tlo_cons_availability.loc[tlo_cons_availability['stkout_prob'] < 0, 'stkout_prob'] = 0
tlo_cons_availability.loc[tlo_cons_availability['stkout_prob'] > 1, 'stkout_prob'] = 1

# Eliminate duplicates
collapse_dict = {
    'stkout_prob': 'mean', 'closing_bal': 'mean', 'average_monthly_consumption': 'mean', 'dispensed': 'mean', 'qty_received': 'mean',
    'module_name': 'first', 'consumable_name_tlo': 'first', 'data_source': 'first'
}
tlo_cons_availability = tlo_cons_availability.groupby(['fac_level', 'fac_name', 'district', 'year', 'month', 'item_code'], as_index=False).agg(
    collapse_dict).reset_index()

tlo_cons_availability['available_prob'] = 1 - tlo_cons_availability['stkout_prob']

'''
# Some missing values change to 100% stockouts during the aggregation above. Fix this manually
for var in ['stkout_prob', 'available_prob', 'closing_bal', 'average_monthly_consumption', 'dispensed', 'qty_received']:
    cond = tlo_cons_availability['data_source'].isna()
    tlo_cons_availability.loc[cond, var] = np.nan
'''

tlo_cons_availability = tlo_cons_availability.reset_index()
tlo_cons_availability = tlo_cons_availability[
    ['module_name', 'district', 'fac_level', 'fac_name', 'year', 'month', 'item_code', 'consumable_name_tlo',
     'available_prob', 'closing_bal', 'average_monthly_consumption', 'dispensed', 'qty_received',
     'data_source']]
tlo_cons_availability.groupby(['year', 'module_name'])['available_prob'].mean()

# 6. FILL GAPS USING HHFA SURVEY DATA OR ASSUMPTIONS ##
#######################################################

'''
# this section is added added to load data prepared upto this point because of the time taken by the above code to run
lmis_wide = pd.read_csv(figurespath / 'lmis_wide.csv', low_memory = False, header=[0, 1], skipinitialspace=True)
lmis_wide = lmis_wide.drop([('Unnamed: 0_level_0',         'year_month')], axis = 1)
unnamed_level1_columns = [(level0, '' if 'Unnamed' in level1 else level1) for level0, level1 in lmis_wide.columns]
lmis_wide.columns = pd.MultiIndex.from_tuples(unnamed_level1_columns)
'''

# Bar chart of number of months during which each consumable was reported before and after cleaning names

# List of new consumables which are reported from 2020 onwards

# Calculate probability of availability based on the number of days of stock out

# Address substitutes

# Comparative plot of probability of consumable availability (consumable on X-axis, prob on y-axis)

# Monthly availability plots for each year of data (heat map - program on the x-axis, month on the y-axis - one for each year)

# Average heatmaps by program and level (how has availability change across years)



# Descriptive analysis
# Number of facilities reporting by level
generate_summary_heatmap(_df = lmis,
                         x_var = 'year_month',
                         y_var = 'fac_level',
                         value_var = 'fac_name',
                         value_label = 'Number of facilities reporting',
                         summary_func='nunique')

# Number of facilities reporting by program
generate_summary_heatmap(_df = lmis,
                         x_var = 'year_month',
                         y_var = 'program',
                         value_var = 'fac_name',
                         value_label = 'Number of facilities reporting',
                         summary_func='nunique')


# Number of consumables reported by level
generate_summary_heatmap(_df = lmis,
                         x_var = 'year_month',
                         y_var = 'fac_level',
                         value_var = 'item',
                         value_label = 'Number of consumables reported',
                         summary_func='nunique')

# Number of consumables reported by program
generate_summary_heatmap(_df = lmis,
                         x_var = 'year_month',
                         y_var = 'program',
                         value_var = 'item',
                         value_label = 'Number of consumables reported',
                         summary_func='nunique')

# Number of stkout_days by program
generate_summary_heatmap(_df = lmis,
                         x_var = 'year_month',
                         y_var = 'fac_level',
                         value_var = 'stkout_days',
                         value_label = 'Average number of stockout days',
                         summary_func='mean')

# Number of stkout_days by level
generate_summary_heatmap(_df = lmis,
                         x_var = 'year_month',
                         y_var = 'program',
                         value_var = 'stkout_days',
                         value_label = 'Average number of stockout days',
                         summary_func='mean')

# Number of stkout_days by level and program
generate_summary_heatmap(_df = lmis,
                         x_var = 'program',
                         y_var = 'fac_level',
                         value_var = 'stkout_days',
                         value_label = 'Average number of stockout days',
                         summary_func='mean')

# Number of facilities reporting by level and program
generate_summary_heatmap(_df = lmis,
                         x_var = 'program',
                         y_var = 'fac_level',
                         value_var = 'fac_name',
                         value_label = 'Number of facilities reporting',
                         summary_func='nunique')

# Number of consumables reported by level and program
generate_summary_heatmap(_df = lmis,
                         x_var = 'program',
                         y_var = 'fac_level',
                         value_var = 'item',
                         value_label = 'Number of consumables reported',
                         summary_func='nunique')

