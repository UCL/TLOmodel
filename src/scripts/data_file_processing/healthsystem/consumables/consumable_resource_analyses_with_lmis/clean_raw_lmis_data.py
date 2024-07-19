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
years_dict = {1: 2021, 2: 2022, 3: 2023}
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
        'Facility Name': 'fac_name',
        'Managed By': 'fac_owner',
        'Product Name': 'item',
        'Program Name': 'program',
        'Dispensed': 'dispensed',
        'AMC': 'average_monthly_consumption',
        'Closing Balance': 'closing_bal',
        'MOS': 'months_of_stock',
        'Stockout Days': 'stkout_days',
        'Availability': 'available_days'
    }

    # Iterating through existing columns to find matches and rename
    for col in _df.columns:
        if col.lower().replace(" ", "") in ['quantityreceived', 'receivedquantity', 'received']:
            columns_to_rename[col] = 'qty_received'
        elif col.lower().replace(" ", "") in ['quantityissued', 'issuedquantity', 'dispensed']:
            columns_to_rename[col] = 'dispensed'

    _df.rename(columns=columns_to_rename, inplace=True)

# Define function to assign facilities to TLO model levels based on facility names
# See link: https://docs.google.com/spreadsheets/d/1fcp2-smCwbo0xQDh7bRUnMunCKguBzOIZjPFZKlHh5Y/edit#gid=0
def assign_facilty_level_based_on_facility_names(_df):
    cond_level_0 = (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('healthpost'))
    cond_level_1a = (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('clinic')) | \
                    (_df['fac_name'].str.replace(' ', '').str.replace('center','centre').str.lower().str.contains('healthcentre')) | \
                    (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('maternity')) | \
                    (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('dispensary')) | \
                    (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('opd'))
    cond_level_1b = (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('communityhospital')) | \
                   (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('ruralhospital')) | \
                   (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('stpetershospital')) | \
                   (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('policecollegehospital')) | \
                   (_df['fac_owner'] == 'CHAM')
    cond_level_2 =  (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('districthealthoffice')) | \
                   (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('districthospital')) | \
                   (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('dho'))
    cond_level_3 =  (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('centralhospital'))
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
for y in range(1,len(years_dict)+1):
    print("processing year ", years_dict[y])
    for m in range(1, 13):
        print("processing month ", months_dict[m])
        if ((m == 1) & (y == 1)):
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

# Drop empty rows
lmis = lmis[lmis.fac_name.notna()]

# Remove Private Health facilities from the data
cond_pvt = lmis['fac_owner'] == 'Private'
lmis = lmis[~cond_pvt]
months_reversed_dict = {v: k for k, v in months_dict.items()}
lmis['month_num'] = lmis['month'].map(months_reversed_dict)
lmis['year_month'] = lmis['year'].astype(str) + "_" +  lmis['month_num'].astype(str) # concatenate month and year for plots

# Clean facility types to match with types in the TLO model
assign_facilty_level_based_on_facility_names(lmis)

# Check the number of facilities at higher levels matches actual for all years
full_district_set = set(districts_dict.values())
for y in range(1,len(years_dict)+1):
    for m in range(1, 13):
        print("Checking consistency in data for ", months_dict[m], ", ",  years_dict[y])
        monthly_lmis = lmis[(lmis.month == months_dict[m]) & (lmis.year == years_dict[y])]
        #assert(len(monthly_lmis[monthly_lmis.fac_level == '2']['fac_name'].unique()) == 28) # number of District hospitals
        #assert (len(monthly_lmis[monthly_lmis.fac_level == '3']['fac_name'].unique()) == 4)  # number of Central hospitals
        #assert (len(monthly_lmis[monthly_lmis.fac_level == '4']['fac_name'].unique()) == 1)  # Zomba Mental Hospital
        #assert (len(monthly_lmis[monthly_lmis.stkout_days.notna()]) != 0)  # non-empty data
        districts_in_data = set(monthly_lmis[monthly_lmis.fac_level == '2']['district'].unique())
        if  districts_in_data!= full_district_set:
            print(districts_in_data.difference(full_district_set))
        central_hospitals = monthly_lmis[monthly_lmis.fac_level == '3']['fac_name'].unique()
        if len(central_hospitals) != 4:
            print("central hospitals included ", central_hospitals)
        mental_hospitals = monthly_lmis[monthly_lmis.fac_level == '4']['fac_name'].unique()
        if len(mental_hospitals) != 1:
            print("mental hospitals included ", mental_hospitals)

        # There is data from levels 0, 1a, and 1b from all districts
        for level in ['0', '1a', '1b']:
            assert(len(monthly_lmis[monthly_lmis.fac_level == level]) != 0)

print('Data import complete and ready for analysis; verified that data is complete')

# Extract a list of consumables along with the year_month during which they were reported
lmis['program_item'] = lmis['program'].astype(str) + "---" +  lmis['item'].astype(str)
consumable_reporting_rate = lmis.groupby(['program_item', 'year_month'])['fac_name'].nunique().reset_index()
consumable_reporting_rate = consumable_reporting_rate.pivot( 'program_item', 'year_month', 'fac_name')
consumable_reporting_rate.to_csv(figurespath / 'consumable_reporting_rate.csv')

# Interpolation rules
# If a facility did no report data for a given month, assume same as the average of the three previous months


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

