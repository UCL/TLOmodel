"""
This script extracts the number of health facilities by level and district from the HHFA 2018-19

Inputs:
1. Raw HHFA data - Q1.dta (~Dropbox/Thanzi la Onse/07 - Data/HHFA_2018-19/0 raw/2_Final data/)
2. Cleaned variable names for HHFA data - variable_list.csv (~Dropbox/Thanzi la Onse/07 - Data/HHFA_2018-19/1 processing)

Outputs:
1. updated master facilities list resource file - ResourceFile_Master_Facilities_List.csv
"""

import calendar
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import copy

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
     '/Users/sm2511/Dropbox/Thanzi la Onse'
)

path_to_files_in_the_tlo_dropbox = path_to_dropbox / "07 - Data/HHFA_2018-19/" # <-- point to HHFA data folder in dropbox
resourcefilepath = Path("./resources")

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# %%
## 1. DATA IMPORT ##
raw_hhfa = pd.read_csv(path_to_files_in_the_tlo_dropbox / '0 raw/2_Final data/Q1.csv',
                       low_memory=False)  # import 2018 data
varnames = pd.read_csv(path_to_files_in_the_tlo_dropbox / '1 processing/variable_list.csv',
                       encoding="ISO-8859-1")  # import file with cleaned variable names

# Rename HHFA columns using variable name mapping in loaded .csv
old_var_name = varnames['var']
new_var_name = varnames['new_var_name']

hhfa = copy.deepcopy(raw_hhfa)
for i in range(len(new_var_name)):
    if new_var_name[i] != np.nan:
        hhfa.rename(columns={old_var_name[i]: new_var_name[i]},
                    inplace=True)
    else:
        pass

# Rename columns with missing data to "a" and then drop these columns since these will not be used in the analysis
hhfa.rename({np.nan: "a"}, axis="columns", inplace=True)
hhfa.drop(["a"], axis=1, inplace=True)

# Preserve only relevant columns
facility_identification_columns = ['fac_code', 'fac_name', 'region', 'zone','district', 'fac_type', 'fac_location', 'fac_owner']
hhfa = hhfa[facility_identification_columns]

# %%
## 2. FEATURE CLEANING ##
# Clean district names #
hhfa.loc[hhfa['district'] == 'Blanytyre', 'district'] = 'Blantyre'
hhfa.loc[hhfa['district'] == 'Nkhatabay', 'district'] = 'Nkhata Bay'

# Pvt for profit hospital incorrectly labelled District Hospital
cond = hhfa.fac_code == 5067
hhfa.loc[cond, 'fac_type'] = 'Other Hospital'

# Clean fac_owner
cond = hhfa.fac_owner == 'Private non profit'
hhfa.loc[cond, 'fac_owner'] = 'NGO'

# convert fac_location to binary (Yes/No)
hhfa = hhfa.rename(columns={'fac_location': 'fac_urban'})
cond1 = hhfa.fac_urban.str.lower() == "rural"
hhfa.loc[cond1, 'fac_urban'] = 0
cond2 = hhfa.fac_urban.str.lower() == "urban"
hhfa.loc[cond2, 'fac_urban'] = 1

# Clean facility type
hhfa['fac_type'] = hhfa['fac_type'].str.replace(' ', '').str.lower()
hhfa['Facility_Level'] = ""

def assign_facilty_level_based_on_hhfa_facility_names(_df):
    cond_mch = (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('mzuzucent'))
    _df.loc[cond_mch, 'fac_name'] = 'Mzuzu Central Hospital'
    cond_level0 = (_df['fac_name'].str.replace(' ', '').str.lower().str.contains('healthpost')) | \
                    (_df['fac_type'].str.contains('healthpost'))
    cond_level1a = (_df['fac_type'] == 'clinic') | (_df['fac_type'] == 'healthcentre') | \
            (_df['fac_type'].str.replace(' ', '').str.lower().str.contains('dispensary')) | \
            (_df['fac_type'].str.replace(' ', '').str.lower().str.contains('maternity'))
    cond_level1b =  (_df['fac_type'].str.contains('communityhospital')) | \
                    (_df['fac_type'] == 'otherhospital')
    cond_level2 = (_df['fac_type'] == 'districthospital')
    cond_level3 = _df.fac_name.str.replace(' ', '').str.lower().str.contains("centralhospit")
    cond_level4 = _df.fac_name.str.replace(' ', '').str.lower().str.contains("mentalhospit")

    _df.loc[cond_level0,'Facility_Level'] = '0'
    _df.loc[cond_level1a,'Facility_Level'] = '1a'
    _df.loc[cond_level1b,'Facility_Level'] = '1b'
    _df.loc[cond_level2,'Facility_Level'] = '2'
    _df.loc[cond_level3,'Facility_Level'] = '3'
    _df.loc[cond_level4,'Facility_Level'] = '4'

assign_facilty_level_based_on_hhfa_facility_names(hhfa)
hhfa = hhfa.drop_duplicates('fac_name')

# Count facilities by category
# Count number of private facilities by district
cond_private = hhfa.fac_owner.str.contains("Private")
cond_level0 = hhfa.Facility_Level == '0'
private_facility_count = hhfa[cond_private & ~cond_level0].groupby('district')['fac_name'].count()

# Count number of NGO facilities by district
cond_ngo = hhfa.fac_owner.str.contains("NGO")
ngo_facility_count = hhfa[cond_ngo & ~cond_level0].groupby('district')['fac_name'].count()

# For the TLO model, we are only concerned with government and CHAM facilities
tlo_model_facilities = hhfa[~(cond_ngo|cond_private)]
facility_count_govt_and_cham = tlo_model_facilities.groupby(['district', 'Facility_Level'])['fac_name'].count().reset_index()
# Collapse data for Mzimba  North and South into 'Mzimba'
cond_north = facility_count_govt_and_cham['district'] == 'Mzimba North'
cond_south = facility_count_govt_and_cham['district'] == 'Mzimba South'
facility_count_govt_and_cham.loc[(cond_north|cond_south), 'district'] = 'Mzimba'
facility_count_govt_and_cham = facility_count_govt_and_cham.groupby(['district', 'Facility_Level']).sum()

tlo_model_facilities['govt'] = 0
tlo_model_facilities.loc[tlo_model_facilities.fac_owner == "Government", 'govt'] = 1
proportion_of_facilities_run_by_govt = tlo_model_facilities.groupby(['district', 'Facility_Level'])['govt'].mean()

proportion_of_facilities_in_urban_location = tlo_model_facilities.groupby(['district', 'Facility_Level'])['fac_urban'].mean()

facility_count_data = pd.merge(facility_count_govt_and_cham, proportion_of_facilities_run_by_govt, right_index=True, left_index=True, how = 'left', validate = "1:1")
facility_count_data = pd.merge(facility_count_data, proportion_of_facilities_in_urban_location, right_index=True, left_index=True, how = 'left', validate = "1:1")
facility_count_data = facility_count_data.reset_index().rename(columns = {'district' : 'District',
                                                                          'fac_name' : 'Facility_Count',
                                                                          'govt': 'Proportion_owned_by_government',
                                                                          'fac_urban': 'Proportion_located_in_urban_area'})
facility_count_data = facility_count_data[~(facility_count_data.Facility_Level.isin(['3', '4', '5']))]

#%%
# Add this data to the Master Health Facilities Resource File
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")[['District', 'Facility_Level', 'Region', 'Facility_ID','Facility_Name']]
mfl = mfl.merge(facility_count_data, on = ['District', 'Facility_Level'], how = 'left')
mfl.loc[mfl.Facility_Level.isin(['3', '4', '5']), 'Facility_Count'] = 1
mfl.loc[mfl.Facility_Level.isin(['3', '4', '5']), 'Proportion_owned_by_government'] = 1
mfl.loc[mfl.Facility_Count.isna(), 'Facility_Count'] = 0

# Export Master Health Facilities Resource File with facility count data
mfl.to_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv", index = False)
