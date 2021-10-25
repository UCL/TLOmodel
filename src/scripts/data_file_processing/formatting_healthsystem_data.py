"""
This file set ups the health system resources for each district, each region, and also national level.

It defines 7 levels for facility types, i.e., Facility_Levels = [0,1a,1b,2,3,4,5].

It creates one facility of each level for each district.

It allocates health care workers ('officers') to one of the seven Facility Levels.

"""

import numpy as np
import pandas as pd

# Loading CHAI dataset and preparing workingfile and resourcefilepath

# THE CHAI DATA (Stored in Dropbox/Thanzi la Onse/05 - Resources/Module-healthsystem/chai ehp resource use data/\
# ORIGINAL_Optimization model import_Malawi_20180315 v10.xlsx)
# Path on local desktop
workingfile = '/Users/jdbb1/Desktop/PyCharm/Describing Malawi Healthcare system and human resources/\
ORIGINAL_Optimization model import_Malawi_20180315 v10.xlsx'

# Auxiliary CHAI Data (Stored in Dropbox/Thanzi la Onse/05 - Resources/Module-healthsystem/chai ehp resource use data/\
# Auxiliary CHAI Data from CHAI HR Team 12 Sep 2021)
# Path on local desktop
auxiliaryfile = '/Users/jdbb1/Desktop/PyCharm/Describing Malawi Healthcare system and human resources/\
Auxiliary CHAI Data from CHAI HR Team 12 Sep 2021/'

# OUTPUT RESOURCE_FILES TO:
resourcefilepath = '/Users/jdbb1/Desktop/TLOmodel/resources/healthsystem/'

# ---------------------------------------------------------------------------------------------------------------------
# *** creat and save population_by_district data
population = pd.read_csv(
    '/Users/jdbb1/Desktop/TLOmodel/resources/demography/ResourceFile_PopulationSize_2018Census.csv'
)

pop_by_district = pd.DataFrame(population.groupby('District')['Count'].sum())
# pop_by_region = pd.DataFrame(population.groupby('Region')['Count'].sum())

# Add the column of Region
for d in pop_by_district.index:
    pop_by_district.loc[d, 'Region'] = population.loc[population['District'] == d, 'Region'].values[0]

# Save
pop_by_district.to_csv(resourcefilepath + 'ResourceFile_District_Population_Data.csv')

# ---------------------------------------------------------------------------------------------------------------------
# *** Below we generate staffing tables: fund_staffing_table for funded/established staff, and\
# curr_staffing_table for current staff
# Before generating the tables, we need to prepare wb_import, officer_types_table, and\
# make assumptions of curr_staff_return distribution and fund_staff_return distribution using Auxiliary CHAI Data

# --- wb_import for staff information

# Import all of the 'CurrentStaff' sheet, including both data of current and funded staff
wb_import = pd.read_excel(workingfile, sheet_name='CurrentStaff', header=None)

# --- officer_types_table
# Make dataframe summarising the officer types and the officer codes:
officer_types_table = wb_import.loc[2:3, 64:84].transpose().reset_index(drop=True).copy()
officer_types_table.columns = ['Officer_Type', 'Officer_Type_Code']

# Add the categories of officers
officer_types_table.loc[0:2, 'Officer_Category'] = 'Clinical'
officer_types_table.loc[3:4, 'Officer_Category'] = 'Nursing_and_Midwifery'
officer_types_table.loc[5:7, 'Officer_Category'] = 'Pharmacy'
officer_types_table.loc[8:10, 'Officer_Category'] = 'Laboratory'
officer_types_table.loc[11, 'Officer_Category'] = 'DCSA'
officer_types_table.loc[12:14, 'Officer_Category'] = 'Dental'
officer_types_table.loc[15, 'Officer_Category'] = 'Mental'
officer_types_table.loc[16, 'Officer_Category'] = 'Nutrition'
officer_types_table.loc[17:20, 'Officer_Category'] = 'Radiography'

# Save
officer_types_table.to_csv(resourcefilepath + 'ResourceFile_Officer_Types_Table.csv')

# --- Generate assumptions of current staff distribution at facility levels 0&1a&1b&2
# Read compiled staff return data from CHAI auxiliary datasets
compiled_staff_return = pd.read_excel(auxiliaryfile + 'Compiled Staff Returns.xlsx',
                                      sheet_name='Compiled Staff Returns', skiprows=range(5))

# Get relevant columns
curr_staff_return = compiled_staff_return[['District / Central Hospital', 'MOH/ CHAM', 'Name of Incumbent', 'Cadre',
                                           'Health Facility', 'Health Facility Type']].copy()

# Drop rows with missing elements
curr_staff_return.dropna(inplace=True)

# Drop rows that associate to '_NOT INCLUDED' and '_MISSING'
curr_staff_return.drop(curr_staff_return[curr_staff_return['Cadre'] == '_NOT INCLUDED'].index, inplace=True)
curr_staff_return.drop(curr_staff_return[curr_staff_return['Cadre'] == '_MISSING'].index, inplace=True)

# Drop rows that associate to 'Home Craft Worker' and 'Educ/Environ Health Officer',
# as these cadres are not included in 'Time_Base' and 'PFT'.
curr_staff_return.drop(curr_staff_return[curr_staff_return['Cadre'] == 'Home Craft Worker'].index, inplace=True)
curr_staff_return.drop(curr_staff_return[curr_staff_return['Cadre'] == 'Educ/Environ Health Officer'].index,
                       inplace=True)

# Replace 'HSA' by 'DCSA', 'Nutrition Officer' by 'Nutrition Staff',
# 'Pharmacy Technician' by 'Pharm Technician', 'Pharmacy Assistant' by 'Pharm Assistant',
# to be consistent with officer_types_table
idx_hsa = curr_staff_return[curr_staff_return['Cadre'] == 'HSA'].index
curr_staff_return.loc[idx_hsa, 'Cadre'] = 'DCSA'

idx_nutri = curr_staff_return[curr_staff_return['Cadre'] == 'Nutrition Officer'].index
curr_staff_return.loc[idx_nutri, 'Cadre'] = 'Nutrition Staff'

idx_pt = curr_staff_return[curr_staff_return['Cadre'] == 'Pharmacy Technician'].index
curr_staff_return.loc[idx_pt, 'Cadre'] = 'Pharm Technician'

idx_pa = curr_staff_return[curr_staff_return['Cadre'] == 'Pharmacy Assistant'].index
curr_staff_return.loc[idx_pa, 'Cadre'] = 'Pharm Assistant'

# Replace health facility type "Karonga Hospital" to "District Hospital"
idx_Karonga = curr_staff_return[curr_staff_return['Health Facility Type'] == 'Karonga Hospital'].index
curr_staff_return.loc[idx_Karonga, 'Health Facility Type'] = 'District Hospital'

# Reassign the facility type of Zomba Mental Hospital as 'Zomba Mental Hospital', instead of 'Central Hospital',
# to differentiate it with other central hospitals
idx_ZMH = curr_staff_return[curr_staff_return['Health Facility'] == 'Zomba Mental Hospital'].index
curr_staff_return.loc[idx_ZMH, 'Health Facility Type'] = 'Zomba Mental Hospital'

# Add a column 'Staff_Count' to denote the no. of staff
curr_staff_return['Staff_Count'] = 1

# Reset index
curr_staff_return.reset_index(drop=True, inplace=True)

# Important definition: Facility_Levels = [0, 1a, 1b, 2, 3, 4, 5]
# 0: Community/Local level - HP, Village Health Committee, Community initiatives
# 1a: Primary level - Dispensary, HC, Clinic, Maternity facility
# 1b: Primary level - Community/Rural Hospital, CHAM (Community) Hospitals
# 2: Second level - District hospital, DHO
# 3: Tertiary/Referral level - KCH, MCH, ZCH + QECH as referral hospitals
# 4: Zomba Mental Hospital, which has very limited data in CHAI dataset
# 5: Headquarter, which has staff data (but no Time_Base or Incidence_Curr data)

# Get the Health Facility Type list and Cadre list
# Note three cadres of 'R04 Radiotherapy Technician', 'R03 Sonographer', 'D03 Dental Assistant' have no data
# in CHAI current and funded staff sheet and complied staff return dataset.
fac_types_list = pd.unique(curr_staff_return['Health Facility Type'])  # Level_0 Facs and Headquarter not included
cadre_list = pd.unique(curr_staff_return['Cadre'])  # Radiotherapy Technician/Sonographer/Dental Assistant not included

# Add column 'Facility_Level'; HQ not listed in compiled staff return table
idx_urbhc = curr_staff_return[curr_staff_return['Health Facility Type'] == 'Urban Health Center'].index
curr_staff_return.loc[idx_urbhc, 'Facility_Level'] = 'Facility_Level_1a'  # Including CHAM HCs

idx_rurhc = curr_staff_return[curr_staff_return['Health Facility Type'] == 'Rural Health Center'].index
curr_staff_return.loc[idx_rurhc, 'Facility_Level'] = 'Facility_Level_1a'  # Including CHAM HCs

idx_comhos = curr_staff_return[curr_staff_return['Health Facility Type'] == 'Community Hospital'].index
curr_staff_return.loc[idx_comhos, 'Facility_Level'] = 'Facility_Level_1b'  # Including CHAM community hospitals

idx_dishos = curr_staff_return[curr_staff_return['Health Facility Type'] == 'District Hospital'].index
curr_staff_return.loc[idx_dishos, 'Facility_Level'] = 'Facility_Level_2'

idx_cenhos = curr_staff_return[curr_staff_return['Health Facility Type'] == 'Central Hospital'].index
curr_staff_return.loc[idx_cenhos, 'Facility_Level'] = 'Facility_Level_3'

idx_zmhfac = curr_staff_return[curr_staff_return['Health Facility Type'] == 'Zomba Mental Hospital'].index
curr_staff_return.loc[idx_zmhfac, 'Facility_Level'] = 'Facility_Level_4'

# Add column 'Cadre_Code'
for c in cadre_list:
    curr_staff_return.loc[curr_staff_return['Cadre'] == c, 'Cadre_Code'] = officer_types_table.loc[
        officer_types_table['Officer_Type'] == c, 'Officer_Type_Code'].copy().values[0]

# Check no blanks in this table
assert not pd.isnull(curr_staff_return).any().any()

# curr_staff_return ready!

# Get curr_staff_return distribution among levels 0, 1a, 1b and 2, i.e., staff distribution within a district
# Specifically, only and all DCSAs/HSAs are to be allocated at level 0;
# Other cadres are to be allocated at level 1a and above.

curr_staff_district = curr_staff_return[['Facility_Level', 'Cadre_Code', 'Staff_Count']].copy()

# Group staff by facility level
curr_staff_distribution = pd.DataFrame(
    curr_staff_district.groupby(by=['Cadre_Code', 'Facility_Level'], sort=False).sum())
curr_staff_distribution.sort_index(level=[0, 1], inplace=True)
curr_staff_distribution.reset_index(drop=False, inplace=True)

# Make the curr_staff_distribution includes all cadres and facility levels (0,1a,1b,2,3,4) as index and columns
cadre_faclevel = pd.DataFrame(columns=['Cadre_Code', 'Facility_Level_0', 'Facility_Level_1a',
                                       'Facility_Level_1b', 'Facility_Level_2', 'Facility_Level_3',
                                       'Facility_Level_4'])
cadre_faclevel['Cadre_Code'] = officer_types_table['Officer_Type_Code']
cadre_faclevel = pd.melt(cadre_faclevel, id_vars='Cadre_Code', value_vars=cadre_faclevel.columns[1:],
                         var_name='Facility_Level')
# Merge
curr_staff_distribution = curr_staff_distribution.merge(cadre_faclevel, how='right')
# Fill null with 0
curr_staff_distribution.fillna(0, inplace=True)
# Sort
curr_staff_distribution.set_index(['Cadre_Code', 'Facility_Level'], inplace=True)
curr_staff_distribution.sort_index(level=[0, 1], inplace=True)
curr_staff_distribution.reset_index(drop=False, inplace=True)
curr_staff_distribution.drop(['value'], axis=1, inplace=True)

# Save the the complete current staff distribution table
# curr_staff_distribution_complete = curr_staff_distribution.copy()

# Keep and focus on rows of levels 0, 1a, 1b, and 2
idx_keep = curr_staff_distribution[(curr_staff_distribution['Facility_Level'] == 'Facility_Level_0') |
                                   (curr_staff_distribution['Facility_Level'] == 'Facility_Level_1a') |
                                   (curr_staff_distribution['Facility_Level'] == 'Facility_Level_1b') |
                                   (curr_staff_distribution['Facility_Level'] == 'Facility_Level_2')].index
curr_staff_distribution = curr_staff_distribution.loc[idx_keep, :].copy()
curr_staff_distribution.reset_index(drop=True, inplace=True)

# Add column 'Proportion', denoting the percents of staff per cadre between level 0, level_1a, level_1b, and level_2
for i in range(21):
    # Proportion; Cadres except DCSA are allocated at level 1a and above
    if curr_staff_distribution.loc[4 * i + 1:4 * i + 3, 'Staff_Count'].sum() > 0:  # sum of 4i+1,4i+2,4i+3

        curr_staff_distribution.loc[4 * i + 1, 'Proportion'] = (
            curr_staff_distribution.loc[4 * i + 1, 'Staff_Count'] /
            curr_staff_distribution.loc[4 * i + 1:4 * i + 3, 'Staff_Count'].sum()
        )

        curr_staff_distribution.loc[4 * i + 2, 'Proportion'] = (
            curr_staff_distribution.loc[4 * i + 2, 'Staff_Count'] /
            curr_staff_distribution.loc[4 * i + 1:4 * i + 3, 'Staff_Count'].sum()
        )

        curr_staff_distribution.loc[4 * i + 3, 'Proportion'] = (
            curr_staff_distribution.loc[4 * i + 3, 'Staff_Count'] /
            curr_staff_distribution.loc[4 * i + 1:4 * i + 3, 'Staff_Count'].sum()
        )

# fillna
curr_staff_distribution.fillna(0, inplace=True)

# For DCSA individually, reassign their proportions since we assume all DCSAs are located at level 0
idx_dcsa = curr_staff_distribution[curr_staff_distribution['Cadre_Code'] == 'E01'].index
curr_staff_distribution.loc[idx_dcsa[0], 'Proportion'] = 1.00
curr_staff_distribution.loc[idx_dcsa[1:4], 'Proportion'] = 0.00
# Alternatively, DCSAs 50% at level 0 and 50% at level 1a?

# curr_staff_distribution ready!

# Save
curr_staff_distribution.to_csv(resourcefilepath + 'ResourceFile_Current_Staff_Distribution_Assumption.csv')

# --- Generate assumptions of established/funded staff distribution at facility levels 0&1a&1b&2
# Read 2018-03-09 Facility-level establishment MOH & CHAM from CHAI auxiliary datasets
fund_staff_2018_raw = pd.read_excel(auxiliaryfile + '2018-03-09 Facility-level establishment MOH & CHAM.xlsx',
                                    sheet_name='Establishment listing')

# Get relevant columns
fund_staff_2018 = fund_staff_2018_raw[['Number of positions', 'Facility', 'Facility Type', 'WFOM Cadre']].copy()

# Drop rows with missing/blank elements
fund_staff_2018.dropna(inplace=True)
# Drop rows that associate to '_NOT INCLUDED'
fund_staff_2018.drop(fund_staff_2018[fund_staff_2018['WFOM Cadre'] == '_NOT INCLUDED'].index, inplace=True)
# Drop rows for 'Training Institution'
fund_staff_2018.drop(fund_staff_2018[fund_staff_2018['Facility Type'] == 'Training Institution'].index, inplace=True)
# Reset index after drop
fund_staff_2018.reset_index(drop=True, inplace=True)

# Reform column 'WFOM Cadre'
# Note 'Cadre_Extra' records 'Clinical ' or 'Nursing ' for C01 and C02.
# We combine C01 and C02 into C01 denoting mental health staff cadre to be consistent with 'curr_staff_return'.
fund_staff_2018[['Cadre_No.', 'Cadre_Code', 'Cadre', 'Cadre_Extra']] = \
    fund_staff_2018['WFOM Cadre'].str.split(pat='-| - ', expand=True).copy()
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Cadre_Code'] == 'C02'].index, 'Cadre_Code'] = 'C01'
# Drop columns ['WFOM Cadre','Cadre_No.','Cadre_Extra']
fund_staff_2018.drop(columns=['WFOM Cadre', 'Cadre_No.', 'Cadre_Extra'], inplace=True)

# Drop rows that associate to 'Home Craft Worker', 'Educ/Environ Health Officer', and 'Community Midwife Assistant'
# as these cadres are not included in 'Time_Base' and 'PFT'.
fund_staff_2018.drop(fund_staff_2018[fund_staff_2018['Cadre'] == 'Home Craft Worker'].index, inplace=True)
fund_staff_2018.drop(fund_staff_2018[fund_staff_2018['Cadre'] == 'Educ/Environ Health Officer'].index, inplace=True)
fund_staff_2018.drop(fund_staff_2018[fund_staff_2018['Cadre'] == 'Community Midwife Assistant'].index, inplace=True)
# Reset index
fund_staff_2018.reset_index(drop=True, inplace=True)

# Replace {
# 'HSA' by 'DCSA' (and 'E02' by 'E01') , 'Medical Assistant' by 'Med. Assistant', 'Laboratory Officer' by 'Lab Officer',
# 'Laboratory Technician' by 'Lab Technician', 'Laboratory Assistant' by 'Lab Assistant'
# 'Nursing Officer/Registered Nurse' by 'Nurse Officer', 'Dentist' by 'Dental Officer',
# 'Nutrition Officer' by 'Nutrition Staff', 'Pharmacy Technician' by 'Pharm Technician',
# 'Pharmacy Assistant' by 'Pharm Assistant', 'Pharmacy Officer' by 'Pharmacist' }
# to be consistent with officer_types_table
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Cadre'] == 'HSA'].index, 'Cadre'] = 'DCSA'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Cadre_Code'] == 'E02'].index, 'Cadre_Code'] = 'E01'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Cadre'] == 'Medical Assistant'].index, 'Cadre'] = 'Med. Assistant'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Cadre'] == 'Laboratory Officer'].index, 'Cadre'] = 'Lab Officer'
fund_staff_2018.loc[
    fund_staff_2018[fund_staff_2018['Cadre'] == 'Laboratory Technician'].index, 'Cadre'] = 'Lab Technician'
fund_staff_2018.loc[
    fund_staff_2018[fund_staff_2018['Cadre'] == 'Laboratory Assistant'].index, 'Cadre'] = 'Lab Assistant'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Cadre'] == 'Nursing Officer/Registered Nurse'].index,
                    'Cadre'] = 'Nurse Officer'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Cadre'] == 'Dentist'].index, 'Cadre'] = 'Dental Officer'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Cadre'] == 'Nutrition Officer'].index, 'Cadre'] = 'Nutrition Staff'
fund_staff_2018.loc[
    fund_staff_2018[fund_staff_2018['Cadre'] == 'Pharmacy Technician'].index, 'Cadre'] = 'Pharm Technician'
fund_staff_2018.loc[
    fund_staff_2018[fund_staff_2018['Cadre'] == 'Pharmacy Assistant'].index, 'Cadre'] = 'Pharm Assistant'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Cadre'] == 'Pharmacy Officer'].index, 'Cadre'] = 'Pharmacist'

# Note that {D03 'Dental Assistant', R03 'Radiotherapy Technician', R04 'Sonographer'} are not included in this dataset.
# This is OK because CHAI current and funded staff sheet has no data regarding the three cadres.

# Reassign the facility type of Zomba Mental Hospital as 'Zomba Mental Hospital'.
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility'] == 'Zomba Mental Hospital'].index,
                    'Facility Type'] = 'Zomba Mental Hospital'

# Important definition: Facility_Levels = [0, 1a, 1b, 2, 3, 4, 5]
# 0: Community/Local level - HP, Village Health Committee, Community initiatives
# 1a: Primary level - Dispensary, HC, Clinic, Maternity facility
# 1b: Primary level - Community/Rural Hospital, CHAM (Community) Hospitals
# 2: Second level - District hospital, DHO
# 3: Tertiary/Referral level - KCH, MCH, ZCH + QECH as referral hospitals
# 4: Zomba Mental Hospital, which has very limited data in CHAI dataset
# 5: Headquarter, which has staff data (but no Time_Base or Incidence_Curr data)

# Get the Health Facility Type list
# fac_types_list = pd.unique(fund_staff_2018['Facility Type']) # Level_0 Facs not included

# Add column 'Facility_Level'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility Type'] == 'Urban Health Center'].index,
                    'Facility_Level'] = 'Facility_Level_1a'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility Type'] == 'Rural Health Center'].index,
                    'Facility_Level'] = 'Facility_Level_1a'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility Type'] == 'Health Center (with maternity)'].index,
                    'Facility_Level'] = 'Facility_Level_1a'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility Type'] == 'Health Center (without maternity)'].index,
                    'Facility_Level'] = 'Facility_Level_1a'

fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility Type'] == 'Rural/Community Hospital'].index,
                    'Facility_Level'] = 'Facility_Level_1b'

fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility Type'] == 'District Hospital'].index,
                    'Facility_Level'] = 'Facility_Level_2'
fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility Type'] == 'DHO'].index,
                    'Facility_Level'] = 'Facility_Level_2'

fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility Type'] == 'Central Hospital'].index,
                    'Facility_Level'] = 'Facility_Level_3'

fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility Type'] == 'Zomba Mental Hospital'].index,
                    'Facility_Level'] = 'Facility_Level_4'

fund_staff_2018.loc[fund_staff_2018[fund_staff_2018['Facility Type'] == 'Headquarters'].index,
                    'Facility_Level'] = 'Facility_Level_5'

# Check no blanks in this table
assert not pd.isnull(fund_staff_2018).any().any()

# fund_staff_2018 ready!

# Get fund_staff_return distribution among levels 0, 1a, 1b and 2, i.e., staff distribution within a district
# Specifically, only and all DCSAs/HSAs are to be allocated at level 0;
# Other cadres are to be allocated at level 1a and above.

fund_staff_district = fund_staff_2018[['Facility_Level', 'Cadre_Code', 'Number of positions']].copy()

# Group staff by facility level
fund_staff_distribution = pd.DataFrame(
    fund_staff_district.groupby(by=['Cadre_Code', 'Facility_Level'], sort=False).sum())
fund_staff_distribution.sort_index(level=[0, 1], inplace=True)
fund_staff_distribution.reset_index(drop=False, inplace=True)

# Make the fund_staff_distribution includes all cadres and facility levels (0,1a,1b,2,3,4,5) as index and columns
fund_cadre_faclevel = pd.DataFrame(columns=['Cadre_Code', 'Facility_Level_0', 'Facility_Level_1a',
                                            'Facility_Level_1b', 'Facility_Level_2', 'Facility_Level_3',
                                            'Facility_Level_4', 'Facility_Level_5'])
fund_cadre_faclevel['Cadre_Code'] = officer_types_table['Officer_Type_Code']
fund_cadre_faclevel = pd.melt(fund_cadre_faclevel, id_vars='Cadre_Code', value_vars=fund_cadre_faclevel.columns[1:],
                              var_name='Facility_Level')
# Merge
fund_staff_distribution = fund_staff_distribution.merge(fund_cadre_faclevel, how='right')
# Fill null with 0
fund_staff_distribution.fillna(0, inplace=True)
# Sort
fund_staff_distribution.set_index(['Cadre_Code', 'Facility_Level'], inplace=True)
fund_staff_distribution.sort_index(level=[0, 1], inplace=True)
fund_staff_distribution.reset_index(drop=False, inplace=True)
fund_staff_distribution.drop(['value'], axis=1, inplace=True)

# Save the the complete funded staff distribution table
# fund_staff_distribution_complete = fund_staff_distribution.copy()

# Keep and focus on rows of levels 0, 1a, 1b, and 2
fund_idx_keep = fund_staff_distribution[(fund_staff_distribution['Facility_Level'] == 'Facility_Level_0') |
                                        (fund_staff_distribution['Facility_Level'] == 'Facility_Level_1a') |
                                        (fund_staff_distribution['Facility_Level'] == 'Facility_Level_1b') |
                                        (fund_staff_distribution['Facility_Level'] == 'Facility_Level_2')].index
fund_staff_distribution = fund_staff_distribution.loc[fund_idx_keep, :].copy()
fund_staff_distribution.reset_index(drop=True, inplace=True)

# Add column 'Proportion', denoting the percents of staff per cadre between level 0, level_1a, level_1b, and level_2
for i in range(21):
    # Proportion; Cadres except DCSA are allocated at level 1a and above
    if fund_staff_distribution.loc[4 * i + 1:4 * i + 3, 'Number of positions'].sum() > 0:  # sum of 4i+1,4i+2,4i+3

        fund_staff_distribution.loc[4 * i + 1, 'Proportion_Fund'] = (
            fund_staff_distribution.loc[4 * i + 1, 'Number of positions'] /
            fund_staff_distribution.loc[4 * i + 1:4 * i + 3, 'Number of positions'].sum()
        )

        fund_staff_distribution.loc[4 * i + 2, 'Proportion_Fund'] = (
            fund_staff_distribution.loc[4 * i + 2, 'Number of positions'] /
            fund_staff_distribution.loc[4 * i + 1:4 * i + 3, 'Number of positions'].sum()
        )

        fund_staff_distribution.loc[4 * i + 3, 'Proportion_Fund'] = (
            fund_staff_distribution.loc[4 * i + 3, 'Number of positions'] /
            fund_staff_distribution.loc[4 * i + 1:4 * i + 3, 'Number of positions'].sum()
        )

# fillna
fund_staff_distribution.fillna(0, inplace=True)

# For DCSA individually, reassign their proportions since we assume all DCSAs are located at level 0
fund_idx_dcsa = fund_staff_distribution[fund_staff_distribution['Cadre_Code'] == 'E01'].index
fund_staff_distribution.loc[fund_idx_dcsa[0], 'Proportion_Fund'] = 1.00
fund_staff_distribution.loc[fund_idx_dcsa[1:4], 'Proportion_Fund'] = 0.00
# Alternatively, DCSAs 50% at level 0 and 50% at level 1a?

# fund_staff_distribution ready!

# Save
fund_staff_distribution.to_csv(resourcefilepath + 'ResourceFile_Funded_Staff_Distribution_Assumption.csv')

# We read info from CHAI estimates of optimal and immediately needed workforce for comparison wherever possible
# --- CHAI WFOM optimal workforce and immediately needed staff distribution

# Preparing optimal workforce from CHAI auxiliary datasets
opt_workforce = pd.read_excel(auxiliaryfile + 'MalawiOptimization_OUTPUT2022 SH 2019-10-19.xlsx',
                              sheet_name='Sums by facility type')
# Drop redundant row
opt_workforce.drop(0, inplace=True)
opt_workforce.reset_index(drop=True, inplace=True)

# Add column 'Facility_level'
opt_workforce.insert(2, 'Facility_Level', ['Facility_Level_3',
                                           'Facility_Level_1b',
                                           'Facility_Level_2',
                                           'Facility_Level_1a',
                                           'Facility_Level_1a'])

# Get staff distribution between level_1a, level_1b and level_2 per cadre
cols_matter = opt_workforce.columns[2:24]
opt_workforce_distribution = opt_workforce.loc[1:4, cols_matter].copy()  # drop row Facility_Level_3
opt_workforce_distribution = pd.DataFrame(opt_workforce_distribution.groupby(by=['Facility_Level'], sort=False).sum())
opt_workforce_distribution.sort_index(inplace=True)
# Reset index
opt_workforce_distribution.reset_index(drop=False, inplace=True)

# Transform to long format
opt_workforce_distribution = pd.melt(opt_workforce_distribution, id_vars='Facility_Level', value_vars=cols_matter[1:],
                                     var_name='Cadre_Opt', value_name='Staff_Count_Opt')

# Add column 'Cadre_Code'
for i in range(63):
    opt_workforce_distribution.loc[i, 'Cadre_Code'] = str(opt_workforce_distribution.loc[i, 'Cadre_Opt'])[7:10]

# Sort to be consistent with curr_staff_distribution
# Drop unnecessary column
opt_workforce_distribution.set_index(['Cadre_Code', 'Facility_Level'], inplace=True)
opt_workforce_distribution.sort_index(level=[0, 1], inplace=True)
opt_workforce_distribution.reset_index(drop=False, inplace=True)
opt_workforce_distribution.drop(columns=['Cadre_Opt'], inplace=True)

# Add column 'Proportion', denoting the percents of staff per cadre between level_1a, level_1b and level_2
for i in range(21):
    if opt_workforce_distribution.loc[3 * i:3 * i + 2, 'Staff_Count_Opt'].sum() > 0:  # sum of 3i,3i+1,3i+2
        opt_workforce_distribution.loc[3 * i, 'Proportion_Opt'] = (
            opt_workforce_distribution.loc[3 * i, 'Staff_Count_Opt'] /
            opt_workforce_distribution.loc[3 * i:3 * i + 2, 'Staff_Count_Opt'].sum()
        )

        opt_workforce_distribution.loc[3 * i + 1, 'Proportion_Opt'] = (
            opt_workforce_distribution.loc[3 * i + 1, 'Staff_Count_Opt'] /
            opt_workforce_distribution.loc[3 * i:3 * i + 2, 'Staff_Count_Opt'].sum()
        )

        opt_workforce_distribution.loc[3 * i + 2, 'Proportion_Opt'] = (
            opt_workforce_distribution.loc[3 * i + 2, 'Staff_Count_Opt'] /
            opt_workforce_distribution.loc[3 * i:3 * i + 2, 'Staff_Count_Opt'].sum()
        )

# fillna
opt_workforce_distribution.fillna(0, inplace=True)

# opt_workforce_distribution ready!

# Preparing immediately needed estimates from CHAI auxiliary datasets
immed_need = pd.read_excel(auxiliaryfile + 'MalawiOptimization_OUTPUT_ALLYEARS_Curr.xlsx',
                           sheet_name='CurrBase Output')

# Select relevant data
idx_year = immed_need[immed_need['OutputYear'] == 2016].index
immed_need_distribution = immed_need.loc[idx_year, immed_need.columns[np.r_[1, 3, 49:70]]]
immed_need_distribution.dropna(inplace=True)

# Add column 'Facility_Level'
immed_need_distribution.loc[immed_need_distribution[immed_need_distribution['FacilityType'] ==
                                                    'UrbHC'].index, 'Facility_Level'] = 'Facility_Level_1a'

immed_need_distribution.loc[immed_need_distribution[immed_need_distribution['FacilityType'] ==
                                                    'RurHC'].index, 'Facility_Level'] = 'Facility_Level_1a'

immed_need_distribution.loc[immed_need_distribution[immed_need_distribution['FacilityType'] ==
                                                    'ComHos'].index, 'Facility_Level'] = 'Facility_Level_1b'

immed_need_distribution.loc[immed_need_distribution[immed_need_distribution['FacilityType'] ==
                                                    'DisHos'].index, 'Facility_Level'] = 'Facility_Level_2'

immed_need_distribution.loc[immed_need_distribution[immed_need_distribution['FacilityType'] ==
                                                    'CenHos'].index, 'Facility_Level'] = 'Facility_Level_3'

# Group staff by levels
immed_need_distribution = pd.DataFrame(immed_need_distribution.groupby(by=['Facility_Level'], sort=False).sum())
# Drop level 3
immed_need_distribution.drop(index='Facility_Level_3', inplace=True)
# Reset index
immed_need_distribution.reset_index(inplace=True)

# Transform to long format
assert set(immed_need_distribution.columns[1:]) == set(cols_matter[1:])
immed_need_distribution = pd.melt(immed_need_distribution, id_vars='Facility_Level', value_vars=cols_matter[1:],
                                  var_name='Cadre_ImmedNeed', value_name='Staff_Count_ImmedNeed')

# Add column 'Cadre_Code'
for i in range(63):
    immed_need_distribution.loc[i, 'Cadre_Code'] = str(immed_need_distribution.loc[i, 'Cadre_ImmedNeed'])[7:10]

# Sort to be consistent with curr_staff_distribution
# Drop unnecessary column
immed_need_distribution.set_index(['Cadre_Code', 'Facility_Level'], inplace=True)
immed_need_distribution.sort_index(level=[0, 1], inplace=True)
immed_need_distribution.reset_index(drop=False, inplace=True)
immed_need_distribution.drop(columns=['Cadre_ImmedNeed'], inplace=True)

# Add column 'Proportion', denoting the percents of staff per cadre among level_1a, level_1b, and level_2
for i in range(21):
    if immed_need_distribution.loc[3 * i:3 * i + 2, 'Staff_Count_ImmedNeed'].sum() > 0:  # sum of 3i,3i+1,3i+2
        immed_need_distribution.loc[3 * i, 'Proportion_ImmedNeed'] = (
            immed_need_distribution.loc[3 * i, 'Staff_Count_ImmedNeed'] /
            immed_need_distribution.loc[3 * i:3 * i + 2, 'Staff_Count_ImmedNeed'].sum()
        )

        immed_need_distribution.loc[3 * i + 1, 'Proportion_ImmedNeed'] = (
            immed_need_distribution.loc[3 * i + 1, 'Staff_Count_ImmedNeed'] /
            immed_need_distribution.loc[3 * i:3 * i + 2, 'Staff_Count_ImmedNeed'].sum()
        )

        immed_need_distribution.loc[3 * i + 2, 'Proportion_ImmedNeed'] = (
            immed_need_distribution.loc[3 * i + 2, 'Staff_Count_ImmedNeed'] /
            immed_need_distribution.loc[3 * i:3 * i + 2, 'Staff_Count_ImmedNeed'].sum()
        )

# fillna
immed_need_distribution.fillna(0, inplace=True)

# immed_need_distribution ready!

# --- Combine curr_staff_distribution, fund_staff_distribution, opt_workforce_distribution, and immed_need_distribution
# Compare if possible

# Merge curr and opt data
# First, drop rows of level_0 of curr_staff_distribution, for compare_staff_distribution
idx_level0 = curr_staff_distribution[curr_staff_distribution['Facility_Level'] == 'Facility_Level_0'].index
compare_staff_distribution = curr_staff_distribution.drop(idx_level0, axis=0, inplace=False).copy()
# Merge
compare_staff_distribution = curr_staff_distribution.merge(opt_workforce_distribution, how='right')

# Check before adding ImmedNeed data
assert (compare_staff_distribution['Cadre_Code'] == immed_need_distribution['Cadre_Code']).all()
assert (compare_staff_distribution['Facility_Level'] == immed_need_distribution['Facility_Level']).all()
# Add Staff_Count_ImmedNeed and Proportion_ImmedNeed to the merged table
compare_staff_distribution['Staff_Count_ImmedNeed'] = immed_need_distribution['Staff_Count_ImmedNeed'].copy()
compare_staff_distribution['Proportion_ImmedNeed'] = immed_need_distribution['Proportion_ImmedNeed'].copy()

# Add fund data
# First, drop rows of level_0 of fund_staff_distribution
fund_idx_level0 = fund_staff_distribution[fund_staff_distribution['Facility_Level'] == 'Facility_Level_0'].index
fund_staff_distribution_nolevel0 = fund_staff_distribution.drop(fund_idx_level0, axis=0, inplace=False).copy()
fund_staff_distribution_nolevel0.reset_index(drop=True, inplace=True)
# Check before combination
assert (compare_staff_distribution['Cadre_Code'] == fund_staff_distribution_nolevel0['Cadre_Code']).all()
assert (compare_staff_distribution['Facility_Level'] == fund_staff_distribution_nolevel0['Facility_Level']).all()
# Add Number of positions and Proportion_Fund to the merged table
compare_staff_distribution.insert(4, 'Staff_Count_Fund', fund_staff_distribution_nolevel0['Number of positions'].values)
compare_staff_distribution.insert(5, 'Proportion_Fund', fund_staff_distribution_nolevel0['Proportion_Fund'].values)

# Calculate the difference
for i in range(63):
    # Current data compared with Fund, Opt, and ImmedNeed
    if compare_staff_distribution.loc[i, 'Proportion_Fund'] > 0:
        compare_staff_distribution.loc[i, 'Curr_vs_Fund'] = (
            (compare_staff_distribution.loc[i, 'Proportion'] - compare_staff_distribution.loc[i, 'Proportion_Fund']) /
            compare_staff_distribution.loc[i, 'Proportion_Fund']
        )

    if compare_staff_distribution.loc[i, 'Proportion_Opt'] > 0:
        compare_staff_distribution.loc[i, 'Curr_vs_Opt'] = (
            (compare_staff_distribution.loc[i, 'Proportion'] - compare_staff_distribution.loc[i, 'Proportion_Opt']) /
            compare_staff_distribution.loc[i, 'Proportion_Opt']
        )

    if compare_staff_distribution.loc[i, 'Proportion_ImmedNeed'] > 0:
        compare_staff_distribution.loc[i, 'Curr_vs_ImmedNeed'] = (
            (compare_staff_distribution.loc[i, 'Proportion'] -
             compare_staff_distribution.loc[i, 'Proportion_ImmedNeed']) /
            compare_staff_distribution.loc[i, 'Proportion_ImmedNeed']
        )
    # Funded data compared with Opt and ImmedNeed
    if compare_staff_distribution.loc[i, 'Proportion_Opt'] > 0:
        compare_staff_distribution.loc[i, 'Fund_vs_Opt'] = (
            (compare_staff_distribution.loc[i, 'Proportion_Fund'] -
             compare_staff_distribution.loc[i, 'Proportion_Opt']) /
            compare_staff_distribution.loc[i, 'Proportion_Opt']
        )

    if compare_staff_distribution.loc[i, 'Proportion_ImmedNeed'] > 0:
        compare_staff_distribution.loc[i, 'Fund_vs_ImmedNeed'] = (
            (compare_staff_distribution.loc[i, 'Proportion_Fund'] -
             compare_staff_distribution.loc[i, 'Proportion_ImmedNeed']) /
            compare_staff_distribution.loc[i, 'Proportion_ImmedNeed']
        )

# Save
compare_staff_distribution.to_csv(resourcefilepath + 'ResourceFile_Staff_Distribution_Compare.csv')

# ***
# --- fund_staffing_table for funded/established staff
# Extract just the section about "Funded TOTAl Staff'
wb_extract = wb_import.loc[3:37, 64:84]
wb_extract = wb_extract.drop([4, 5])
wb_extract.columns = wb_extract.iloc[0]
wb_extract = wb_extract.drop([3])
wb_extract = wb_extract.reset_index(drop=True)
wb_extract.fillna(0, inplace=True)  # replace all null values with zero values

# Add in the column to the dataframe for the labels that distinguishes whether
# these officers are allocated to the district-or-lower levels or one of the key hospitals.
labels = wb_import.loc[6:37, 0].reset_index(drop=True)
is_distlevel = labels.copy()
is_distlevel[0:27] = True  # for district-or-lower levels
is_distlevel[27:] = False  # for CenHos-or-above levels

wb_extract.loc[:, 'District_Or_Hospital'] = labels
wb_extract.loc[:, 'Is_DistrictLevel'] = is_distlevel

# Finished import from the CHAI excel:
fund_staffing_table = wb_extract.copy()

# There are a large number of officer_types EO1 (DCSA/Comm Health Workers) at HQ level, which is non-sensible
# Therefore, re-distribute these evenly to the districts.
extra_CHW = fund_staffing_table.loc[fund_staffing_table['District_Or_Hospital'] == 'HQ or missing',
                                    fund_staffing_table.columns[fund_staffing_table.columns == 'E01']].values[0][0]
fund_staffing_table.loc[fund_staffing_table['District_Or_Hospital'] == 'HQ or missing',
                        fund_staffing_table.columns[fund_staffing_table.columns == 'E01']] = 0
extra_CHW_per_district = int(np.floor(extra_CHW / fund_staffing_table['Is_DistrictLevel'].sum()))
fund_staffing_table.loc[fund_staffing_table['Is_DistrictLevel'], 'E01'] = \
    fund_staffing_table.loc[fund_staffing_table['Is_DistrictLevel'], 'E01'] + \
    extra_CHW_per_district

# The imported staffing table suggest that there is 1 Dental officer (D01) in each district,
# but the TimeBase data (below) suggest that no appointment occuring at a district-level Facility can incurr
# the time such an officer. Therefor reallocate the D01 officers to the Referral Hospitals
extra_D01 = fund_staffing_table.loc[
    ~fund_staffing_table['District_Or_Hospital'].isin(['KCH', 'MCH', 'QECH', 'ZCH']),
    fund_staffing_table.columns[fund_staffing_table.columns == 'D01']].sum().values[0]
fund_staffing_table.loc[~fund_staffing_table['District_Or_Hospital'].isin(['KCH', 'MCH', 'QECH', 'ZCH']),
                        fund_staffing_table.columns[fund_staffing_table.columns == 'D01']] = 0
extra_D01_per_referralhosp = extra_D01 / 4  # divided by 4 CenHos
fund_staffing_table.loc[fund_staffing_table['District_Or_Hospital'].isin(['KCH', 'MCH', 'QECH', 'ZCH']), 'D01'] = \
    fund_staffing_table.loc[fund_staffing_table['District_Or_Hospital'].isin(['KCH', 'MCH', 'QECH', 'ZCH']), 'D01'] + \
    extra_D01_per_referralhosp

# Sort out which are district allocations and which are central hospitals and above

# We assign HQ to HQ; KCH as RefHos in Central region; MCH as RefHos in Northern region;
# QECH and ZCH as RefHos in Southern region (QECH is in Southwest and ZCH is in Southeast).
fund_staffing_table.loc[
    fund_staffing_table['District_Or_Hospital'] == 'HQ or missing', 'District_Or_Hospital'] = 'Headquarter'
fund_staffing_table.loc[
    fund_staffing_table['District_Or_Hospital'] == 'KCH', 'District_Or_Hospital'] = 'Referral Hospital_Central'
fund_staffing_table.loc[
    fund_staffing_table['District_Or_Hospital'] == 'MCH', 'District_Or_Hospital'] = 'Referral Hospital_Northern'
fund_staffing_table.loc[
    fund_staffing_table['District_Or_Hospital'] == 'QECH', 'District_Or_Hospital'] = 'Referral Hospital_Southern'
# fund_staffing_table.loc[
# fund_staffing_table['District_Or_Hospital'] == 'QECH', 'District_Or_Hospital'] = 'Referral Hospital_Southwest'
fund_staffing_table.loc[
    fund_staffing_table['District_Or_Hospital'] == 'ZCH', 'District_Or_Hospital'] = 'Referral Hospital_Southern'
# fund_staffing_table.loc[
# fund_staffing_table['District_Or_Hospital'] == 'ZCH', 'District_Or_Hospital'] = 'Referral Hospital_Southeast'

# Group the referral hospitals QECH and ZCH as Referral Hospital_Southern
Is_DistrictLevel = fund_staffing_table['Is_DistrictLevel'].values  # Save the column 'Is_DistrictLevel' first
fund_staffing_table = pd.DataFrame(
    fund_staffing_table.groupby(by=['District_Or_Hospital'], sort=False).sum()).reset_index()
fund_staffing_table.insert(1, 'Is_DistrictLevel', Is_DistrictLevel[:-1])  # Add the column 'Is_DistrictLevel'

# Add a row for Zomba Mental Hospital with 3 C01 mental health staff
# (according to data in 2018-03-09 Facility-level establishment MOH & CHAM)
# (This is much less than the current 12 C01.)
fund_ZMH = pd.DataFrame(columns=fund_staffing_table.columns.copy())
fund_ZMH.loc[0, 'District_Or_Hospital'] = 'Zomba Mental Hospital'
fund_ZMH.loc[0, 'Is_DistrictLevel'] = False
fund_ZMH.loc[0, 'C01'] = 3
# Alternatively, if consider all potential cadres from compiled staff return
# fund_cadres_ZMH = pd.DataFrame(index = [0], columns = ['M01','M02','M03','N01','N02','C01','P02','L02'],
#                          data = np.array([[2,13,14,8,30,3,1,1]]))
# for col in fund_cadres_ZMH.columns:
#    fund_ZMH.loc[0,col] = fund_cadres_ZMH.loc[0,col].copy()

# Concat
fund_staffing_table = pd.concat([fund_staffing_table, fund_ZMH])
fund_staffing_table.reset_index(drop=True, inplace=True)
fund_staffing_table.fillna(0, inplace=True)

# File 2018-03-09 Facility-level establishment MOH & CHAM indicates that ZMH is assigned to Zomba District,
# We therefore subtract 3 C01 are from Zomba District.
fund_idx_ZombaDist = fund_staffing_table[fund_staffing_table['District_Or_Hospital'] == 'Zomba'].index
fund_staffing_table.loc[fund_idx_ZombaDist, 'C01'] = \
    fund_staffing_table.loc[fund_idx_ZombaDist, 'C01'] - fund_ZMH.loc[0, 'C01']
# Alternatively, if consider all potential cadres from compiled staff return
# fund_staffing_table.loc[fund_idx_ZombaDist, :] =\
# fund_staffing_table.loc[fund_idx_ZombaDist, :] - fund_ZMH.loc[0,:]

# Check that fund_staffing_table.loc[fund_idx_ZombaDist, :] >=0
assert (fund_staffing_table.loc[fund_idx_ZombaDist, 'M01':'R04'].values >= 0).all()

# The following districts are not in the CHAI data because they are included within other districts.
# For now, we will say that the division of staff between these cities and the wide district (where they are included)
# is consistent with the population recorded for them.
# i.e., to use population-based weights to reallocate staff

# Add in Likoma (part Nkhata Bay)
# Add in Lilongwe City (part of Lilongwe)
# Add in Mzuzu City (part of Mziba) ASSUMED
# Add in Zomba City (part of Zomba)
# Add in Blantyre City (part of Blantyre)

# create mapping: the new districts : super_district
split_districts = (
    ('Likoma', 'Nkhata Bay'),
    ('Lilongwe City', 'Lilongwe'),
    ('Mzuzu City', 'Mzimba'),
    ('Zomba City', 'Zomba'),
    ('Blantyre City', 'Blantyre')
)

# reallocating staff to the new districts
for i in np.arange(0, len(split_districts)):
    new_district = split_districts[i][0]
    super_district = split_districts[i][1]

    record = fund_staffing_table.iloc[0].copy()  # get a row of the staffing table

    # make a the record for the new district
    record['District_Or_Hospital'] = new_district
    record['Is_DistrictLevel'] = True

    # get total staff level from the super districts
    cols = set(fund_staffing_table.columns).intersection(set(officer_types_table.Officer_Type_Code))

    total_staff = fund_staffing_table.loc[
        fund_staffing_table['District_Or_Hospital'] == super_district, cols].values.squeeze()

    # get the weight; The original weights w0 for the 5 new districts in order are 0.05,0.60,0.24,0.14,1.77(> 1)
    w0 = pop_by_district.loc[new_district, 'Count'] / pop_by_district.loc[super_district, 'Count']
    if w0 < 1:
        w = w0
    else:
        w = 0.5

    # assign w * 100% staff to the new district
    record.loc[cols] = w * total_staff
    fund_staffing_table = fund_staffing_table.append(record).reset_index(drop=True)

    # take staff away from the super district
    fund_staffing_table.loc[fund_staffing_table['District_Or_Hospital'] == super_district, cols] = \
        fund_staffing_table.loc[
            fund_staffing_table[
                'District_Or_Hospital'] == super_district, cols] - record.loc[cols]

# Confirm the merging will be perfect:
pop = pop_by_district.reset_index(drop=False, inplace=False)
assert set(pop['District'].values) == set(
    fund_staffing_table.loc[fund_staffing_table['Is_DistrictLevel'], 'District_Or_Hospital'])
assert len(pop['District'].values) == len(
    fund_staffing_table.loc[fund_staffing_table['Is_DistrictLevel'], 'District_Or_Hospital'])

# ... double check by doing the merge explicitly
pop_districts = pd.DataFrame({'District': pd.unique(pop['District'])})  # data frame
chai_districts = pd.DataFrame(
    {'District': fund_staffing_table.loc[fund_staffing_table['Is_DistrictLevel'], 'District_Or_Hospital']})

merge_result = pop_districts.merge(chai_districts, how='inner', indicator=True)
assert all(merge_result['_merge'] == 'both')
assert len(merge_result) == len(pop_districts)

# Split staff within each district to level 0 (All DCSAs at HP), level 1a (Disp, HC, etc.),
# level 1b (ComHos, CHAM ComHos), and level 2 (DisHos, etc.), according to fund_staff_distribution.

# First, generate a df with all districts and facility levels 0 - 2 per district
district_faclevel = pd.DataFrame(columns=['District_Or_Hospital', 'Facility_Level_0', 'Facility_Level_1a',
                                          'Facility_Level_1b', 'Facility_Level_2'])
district_faclevel['District_Or_Hospital'] = pop['District'].values.copy()
district_faclevel = pd.melt(district_faclevel, id_vars='District_Or_Hospital', value_vars=district_faclevel.columns[1:],
                            var_name='Facility_Level')
district_faclevel.set_index(['District_Or_Hospital', 'Facility_Level'], inplace=True)
district_faclevel.sort_index(level=[0, 1], inplace=True)
district_faclevel.reset_index(drop=False, inplace=True)
district_faclevel.drop(columns=['value'], axis=1, inplace=True)
# Merge
fund_staffing_table = district_faclevel.merge(fund_staffing_table, how='outer')

# Split staff among levels
for district in pop['District']:
    for cadre in set(fund_staffing_table.columns[3:]):
        # The proportions
        weight = fund_staff_distribution.loc[fund_staff_distribution['Cadre_Code'] == cadre,
                                             ['Facility_Level', 'Proportion_Fund']].copy()
        # The staff count before splitting
        old_count = fund_staffing_table.loc[fund_staffing_table['District_Or_Hospital'] == district,
                                            ['Facility_Level', cadre]].copy()

        # Check that Facility levels of weight and old_count are consistent
        assert (weight['Facility_Level'].values == old_count['Facility_Level'].values).all()

        # Check that if old_count is not 0, then weight is not 0, guaranteeing that staff are split
        if (old_count[cadre] > 0).any():
            assert (weight['Proportion_Fund'] > 0).any()

        # Split
        fund_staffing_table.loc[fund_staffing_table['District_Or_Hospital'] == district, cadre] = (
            old_count[cadre].values * weight['Proportion_Fund'].values)

# Add facility levels for HQ, CenHos and ZMH
fund_staffing_table.loc[128:132, 'Facility_Level'] = ['Facility_Level_5', 'Facility_Level_3',
                                                      'Facility_Level_3', 'Facility_Level_3',
                                                      'Facility_Level_4']
# Make values integers (after rounding)
fund_staffing_table.loc[:, fund_staffing_table.columns[3:]] = \
    fund_staffing_table.loc[:, fund_staffing_table.columns[3:]].astype(float).round(0).astype(int)

# fund_staffing_table ready!

# Save the table without column 'Is_DistrictLevel'
fund_staffing_table_to_save = fund_staffing_table.drop(columns='Is_DistrictLevel', inplace=False)
fund_staffing_table_to_save.to_csv(resourcefilepath + 'ResourceFile_Funded_Staff_Table.csv')

# Flip from wide to long format, where one row represents on staff
fund_staff_list = pd.melt(fund_staffing_table, id_vars=['District_Or_Hospital', 'Facility_Level', 'Is_DistrictLevel'],
                          var_name='Officer_Type_Code', value_name='Number')

# Repeat rows so that it is one row per staff member
# fund_staff_list['Number'] = fund_staff_list['Number'].astype(int)
fund_staff_list = fund_staff_list.loc[fund_staff_list.index.repeat(fund_staff_list['Number'])]
fund_staff_list = fund_staff_list.reset_index(drop=True)
fund_staff_list = fund_staff_list.drop(['Number'], axis=1)

# check that the number of rows in this fund_staff_list is equal to the total number of staff from the input data
assert len(fund_staff_list) == fund_staffing_table.iloc[:, 3:].sum().sum()

fund_staff_list['Is_DistrictLevel'] = fund_staff_list['Is_DistrictLevel'].astype(bool)

# assign an arbitrary staff_id
fund_staff_list['Staff_ID'] = fund_staff_list.index

# Save the table without column 'Is_DistrictLevel'
fund_staff_list_to_save = fund_staff_list.drop(columns='Is_DistrictLevel', inplace=False)
fund_staff_list_to_save.to_csv(resourcefilepath + 'ResourceFile_Funded_Staff_List.csv')

# ***
# --- Creating curr_staffing_table and curr_staff_list for current staff
# Extract the section about "Current TOTAl Staff'
hcw_curr_extract = wb_import.loc[3:37, 1:21]
hcw_curr_extract = hcw_curr_extract.drop([4, 5])
hcw_curr_extract.columns = hcw_curr_extract.iloc[0]
hcw_curr_extract = hcw_curr_extract.drop([3])
hcw_curr_extract = hcw_curr_extract.reset_index(drop=True)
hcw_curr_extract.fillna(0, inplace=True)

# Add in the columns to the dataframe for the labels that distinguishes whether
# these officers are allocated to the district-or-lower levels or one of the key hospitals.
hcw_curr_extract.loc[:, 'District_Or_Hospital'] = labels
hcw_curr_extract.loc[:, 'Is_DistrictLevel'] = is_distlevel

# Finished import from the CHAI excel
curr_staffing_table = hcw_curr_extract.copy()

# Check the cadre columns of curr_staffing_table is identical to fund_staffing_table
assert set(curr_staffing_table.columns[0:21]) == set(fund_staffing_table.columns[-21:])

# For curr_staffing_table, reallocating D01 from districts to referral hospitals
# Treat KCH, MCH, QECH, ZCH as referral hospitals
# The operation of reallocating E01 in HQ to districts is not needed for curr_staffing_table,
# as no. of E01 in curr_staffing_table at HQ is zero.

curr_extra_D01 = curr_staffing_table.loc[
    ~curr_staffing_table['District_Or_Hospital'].isin(['KCH', 'MCH', 'QECH', 'ZCH']), curr_staffing_table.columns[
        curr_staffing_table.columns == 'D01']].sum().values[0]
curr_staffing_table.loc[
    ~curr_staffing_table['District_Or_Hospital'].isin(['KCH', 'MCH', 'QECH', 'ZCH']), curr_staffing_table.columns[
        curr_staffing_table.columns == 'D01']] = 0
curr_extra_D01_per_referralhosp = curr_extra_D01 / 4
curr_staffing_table.loc[curr_staffing_table['District_Or_Hospital'].isin(['KCH', 'MCH', 'QECH', 'ZCH']), 'D01'] = \
    curr_staffing_table.loc[curr_staffing_table['District_Or_Hospital'].isin(['KCH', 'MCH', 'QECH', 'ZCH']), 'D01'] + \
    curr_extra_D01_per_referralhosp

# For curr_staffing_table, sort out the districts and central hospitals
curr_staffing_table.loc[
    curr_staffing_table['District_Or_Hospital'] == 'HQ or missing', 'District_Or_Hospital'] = 'Headquarter'
curr_staffing_table.loc[
    curr_staffing_table['District_Or_Hospital'] == 'KCH', 'District_Or_Hospital'] = 'Referral Hospital_Central'
curr_staffing_table.loc[
    curr_staffing_table['District_Or_Hospital'] == 'MCH', 'District_Or_Hospital'] = 'Referral Hospital_Northern'
curr_staffing_table.loc[
    curr_staffing_table['District_Or_Hospital'] == 'QECH', 'District_Or_Hospital'] = 'Referral Hospital_Southern'
curr_staffing_table.loc[
    curr_staffing_table['District_Or_Hospital'] == 'ZCH', 'District_Or_Hospital'] = 'Referral Hospital_Southern'

# Group the referral hospitals QECH and ZCH as Referral Hospital_Southern
Is_DistrictLevel = curr_staffing_table['Is_DistrictLevel'].values  # Save the column 'Is_DistrictLevel' first
curr_staffing_table = pd.DataFrame(
    curr_staffing_table.groupby(by=['District_Or_Hospital'], sort=False).sum()).reset_index()
curr_staffing_table.insert(1, 'Is_DistrictLevel', Is_DistrictLevel[:-1])  # Add the column 'Is_DistrictLevel'

# Add a row for Zomba Mental Hospital, which has 12 mental health staff according to compiled staff return
curr_ZMH = pd.DataFrame(columns=curr_staffing_table.columns.copy())
curr_ZMH.loc[0, 'District_Or_Hospital'] = 'Zomba Mental Hospital'
curr_ZMH.loc[0, 'Is_DistrictLevel'] = False
curr_ZMH.loc[0, 'C01'] = 12
# Alternatively, if consider all potential cadres from compiled staff return
# curr_cadres_ZMH = pd.DataFrame(index = [0], columns = ['M01','M02','N01','N02','C01','P02','P03'],
#                          data = np.array([[2,5,19,27,12,1,1]]))
# for col in curr_cadres_ZMH.columns:
#    curr_ZMH.loc[0,col] = curr_cadres_ZMH.loc[0,col].copy()

curr_staffing_table = pd.concat([curr_staffing_table, curr_ZMH])
curr_staffing_table.reset_index(drop=True, inplace=True)
curr_staffing_table.fillna(0, inplace=True)

# For Zomba district, there are 12 mental health staff C01;
# However, compiled staff return does not record any C01 in Zomba district;
# We therefore assume that its 12 C01 are from Zomba Mental Hospital.
curr_idx_ZombaDist = curr_staffing_table[curr_staffing_table['District_Or_Hospital'] == 'Zomba'].index
curr_staffing_table.loc[curr_idx_ZombaDist, 'C01'] = \
    curr_staffing_table.loc[curr_idx_ZombaDist, 'C01'] - curr_ZMH.loc[0, 'C01']
# Alternatively, if consider all potential cadres from compiled staff return
# curr_staffing_table.loc[curr_idx_ZombaDist, :] = curr_staffing_table.loc[curr_idx_ZombaDist, :] - curr_ZMH.loc[0,:]

# Check that curr_staffing_table.loc[curr_idx_ZombaDist, :] >=0
assert (curr_staffing_table.loc[curr_idx_ZombaDist, 'M01':'R04'].values >= 0).all()

# Similarly split staff to 5 special districts as done for funded staff
# split_districts = (
#    ('Likoma', 'Nkhata Bay'),
#    ('Lilongwe City', 'Lilongwe'),
#    ('Mzuzu City', 'Mzimba'),
#    ('Zomba City', 'Zomba'),
#    ('Blantyre City', 'Blantyre')
# )

for i in np.arange(0, len(split_districts)):
    new_district = split_districts[i][0]
    super_district = split_districts[i][1]

    record = curr_staffing_table.iloc[0].copy()  # get a row of the staffing table

    # make a the record for the new district
    record['District_Or_Hospital'] = new_district
    record['Is_DistrictLevel'] = True

    # get total staff level from the super districts
    cols = set(curr_staffing_table.columns).intersection(set(officer_types_table.Officer_Type_Code))

    total_staff = curr_staffing_table.loc[
        curr_staffing_table['District_Or_Hospital'] == super_district, cols].values.squeeze()

    # get the weight
    w0 = pop_by_district.loc[new_district, 'Count'] / pop_by_district.loc[
        super_district, 'Count']  # The values in order are 0.05,0.60,0.24,0.14,1.77
    if w0 < 1:
        w = w0
    else:
        w = 0.5

    # assign w * 100% staff to the new district
    record.loc[cols] = w * total_staff
    curr_staffing_table = curr_staffing_table.append(record).reset_index(drop=True)

    # take staff away from the super district
    curr_staffing_table.loc[curr_staffing_table['District_Or_Hospital'] == super_district, cols] = \
        curr_staffing_table.loc[
            curr_staffing_table[
                'District_Or_Hospital'] == super_district, cols] - record.loc[cols]

# Confirm the merging will be perfect:
# pop = pop_by_district.reset_index(drop = False, inplace = False)
assert set(pop['District'].values) == set(
    curr_staffing_table.loc[curr_staffing_table['Is_DistrictLevel'], 'District_Or_Hospital'])
assert len(pop['District'].values) == len(
    curr_staffing_table.loc[curr_staffing_table['Is_DistrictLevel'], 'District_Or_Hospital'])

# ... double check by doing the merge explicitly
# pop_districts = pd.DataFrame({'District': pd.unique(pop['District'])})
chai_districts = pd.DataFrame(
    {'District': curr_staffing_table.loc[curr_staffing_table['Is_DistrictLevel'], 'District_Or_Hospital']})

merge_result = pop_districts.merge(chai_districts, how='inner', indicator=True)
assert all(merge_result['_merge'] == 'both')
assert len(merge_result) == len(pop_districts)

# Split staff within each district to level 0 (All DCSAs at HP), level 1a (Disp, HC, etc.),
# level 1b (ComHos, CHAM ComHos), and level 2 (DisHos, etc.), according to curr_staff_distribution.

# First, make the table including all districts and facility levels 0 - 2 per district,\
# by merging with district_faclevel defined previously.
curr_staffing_table = district_faclevel.merge(curr_staffing_table, how='outer')

# Split staff among levels
for district in pop['District']:
    for cadre in set(curr_staffing_table.columns[3:]):
        # The proportions
        weight = curr_staff_distribution.loc[curr_staff_distribution['Cadre_Code'] == cadre,
                                             ['Facility_Level', 'Proportion']].copy()
        # The staff count before splitting
        old_count = curr_staffing_table.loc[curr_staffing_table['District_Or_Hospital'] == district,
                                            ['Facility_Level', cadre]].copy()

        # Check that Facility levels of weight and old_count are consistent
        assert (weight['Facility_Level'].values == old_count['Facility_Level'].values).all()

        # Check that if old_count is not 0, then weight is not 0, guaranteeing that staff are split
        if (old_count[cadre] > 0).any():
            assert (weight['Proportion'] > 0).any()

        # Split
        curr_staffing_table.loc[curr_staffing_table['District_Or_Hospital'] == district, cadre] = (
            old_count[cadre].values * weight['Proportion'].values)

# Add facility levels for HQ, CenHos and ZMH
curr_staffing_table.loc[128:133, 'Facility_Level'] = ['Facility_Level_5', 'Facility_Level_3',
                                                      'Facility_Level_3', 'Facility_Level_3',
                                                      'Facility_Level_4']  # 128:132 also OK
# Make values integers (after rounding)
curr_staffing_table.loc[:, curr_staffing_table.columns[3:]] = (
    curr_staffing_table.loc[:, curr_staffing_table.columns[3:]].astype(float).round(0).astype(int))

# Save the table without column 'Is_DistrictLevel'
curr_staffing_table_to_save = curr_staffing_table.drop(columns='Is_DistrictLevel', inplace=False)
curr_staffing_table_to_save.to_csv(resourcefilepath + 'ResourceFile_Current_Staff_Table.csv')

# For curr_staffing_table, flip from wide to long format and create curr_staff_list
# The long format
curr_staff_list = pd.melt(curr_staffing_table, id_vars=['District_Or_Hospital', 'Facility_Level', 'Is_DistrictLevel'],
                          var_name='Officer_Type_Code', value_name='Number')

# Repeat rows so that it is one row per staff member
curr_staff_list = curr_staff_list.loc[curr_staff_list.index.repeat(curr_staff_list['Number'])]
curr_staff_list = curr_staff_list.reset_index(drop=True)
curr_staff_list = curr_staff_list.drop(['Number'], axis=1)

# Check that the number of rows in this curr_staff_list is equal to the total number of staff from the input data
assert len(curr_staff_list) == curr_staffing_table.iloc[:, 3:].sum().sum()

curr_staff_list['Is_DistrictLevel'] = curr_staff_list['Is_DistrictLevel'].astype(bool)

# Assign an arbitrary staff_id
curr_staff_list['Staff_ID'] = curr_staff_list.index

# Check that tables for current and funded staff have the same columns names
assert (curr_staff_list.columns == fund_staff_list.columns).all()
assert (curr_staffing_table.columns == fund_staffing_table.columns).all()

# Save the table without column 'Is_DistrictLevel'
curr_staff_list_to_save = curr_staff_list.drop(columns='Is_DistrictLevel', inplace=False)
curr_staff_list_to_save.to_csv(resourcefilepath + 'ResourceFile_Current_Staff_List.csv')

# ---------------------------------------------------------------------------------------------------------------------
# *** Create the Master Facilities List
# This will be a listing of each facility and the district(s) to which they attach
# The different Facility Types are notional at this stage
# The Facility Level is the important variable for the staffing: staff are assumed to be allocated
# to a particular level within a district，or a referral hospital, or others
# They do not associate with a particular type of Facility

Facility_Levels = [0, '1a', '1b', 2, 3, 4, 5]
# 0: Community/Local level - HP, Village Health Committee, Community initiatives
# 1a: Primary level - Dispensary, HC, Clinic, Maternity facility
# 1b: Primary level - Community/Rural Hospital, CHAM (Community) Hospitals
# 2: Second level - District hospital, DHO
# 3: Tertiary/Referral level - KCH, MCH, ZCH + QECH as referral hospitals
# 4: Zomba Mental Hospital, which has very limited data in CHAI dataset
# 5: Headquarter, which has staff data (but no Time_Base or Incidence_Curr data)

# declare the Facility_Type variable
# Facility_Types = ['Health Post', 'Dispensary', 'Health Centre', 'Community or Rural Hospital', 'CHAM Hospital',
#                   'District Hospital', 'DHO', 'Referral Hospital', 'Zomba Mental Hospital']
# Facility_Types_Levels = dict(zip(Facility_Types, Facility_Levels))


# Create empty dataframe that will be the Master Facilities List (mfl)
mfl = pd.DataFrame(columns=['Facility_Level', 'District', 'Region'])

pop_districts = pop['District'].values  # array; the 'pop_districts' used in previous lines is a DataFrame
pop_regions = pd.unique(pop['Region'])

# Each district is assigned with a set of community level facs, a set of primary level facs,
# and a set of second level facs.
# Therefore, the total sets of facs is 4 * no. of districts + 3 (RefHos per Region) + 1 (HQ) + 1 (ZMH) \
# = 4 * 32 + 5 = 133
for d in pop_districts:
    df = pd.DataFrame({'Facility_Level': Facility_Levels[0:4], 'District': d,
                       'Region': pop.loc[pop['District'] == d, 'Region'].values[0]})
    mfl = mfl.append(df, ignore_index=True, sort=True)

# Add in the Referral Hospitals, one for each region
for r in pop_regions:
    mfl = mfl.append(pd.DataFrame({
        'Facility_Level': Facility_Levels[4], 'District': None, 'Region': r
    }, index=[0]), ignore_index=True, sort=True)

# Add the ZMH
mfl = mfl.append(pd.DataFrame({
    'Facility_Level': Facility_Levels[5], 'District': None, 'Region': None
}, index=[0]), ignore_index=True, sort=True)

# Add the HQ
mfl = mfl.append(pd.DataFrame({
    'Facility_Level': Facility_Levels[6], 'District': None, 'Region': None
}, index=[0]), ignore_index=True, sort=True)

# Create the Facility_ID
mfl.loc[:, 'Facility_ID'] = mfl.index

# Create a unique name for each Facility
name = 'Facility_Level_' + mfl['Facility_Level'].astype(str) + '_' + mfl['District']
name.loc[mfl['Facility_Level'] == 3] = 'Referral Hospital' + '_' + mfl.loc[
    mfl['Facility_Level'] == 3, 'Region']
name.loc[mfl['Facility_Level'] == 4] = 'Zomba Mental Hospital'
name.loc[mfl['Facility_Level'] == 5] = 'Headquarter'

mfl.loc[:, 'Facility_Name'] = name

# Save
mfl.to_csv(resourcefilepath + 'ResourceFile_Master_Facilities_List.csv')

# ---------------------------------------------------------------------------------------------------------------------
# *** Create a simple mapping of all the facilities that persons in a district can access
facilities_by_district = pd.DataFrame(columns=mfl.columns)

# Each district in pop_districts has access to five facility levels.
for d in pop_districts:
    the_region = pop.loc[pop['District'] == d, 'Region'].copy().values[0]

    district_facs = mfl.loc[mfl['District'] == d]  # Include facs from level 0 to level 2

    region_fac = mfl.loc[pd.isnull(mfl['District']) & (mfl['Region'] == the_region)].copy().reset_index(drop=True)
    region_fac.loc[0, 'District'] = d  # Level 3, referral hospital

    zmh_fac = mfl.loc[pd.isnull(mfl['District']) & pd.isnull(mfl['Region']) &
                      (mfl['Facility_Name'] == 'Zomba Mental Hospital')].copy().reset_index(drop=True)
    zmh_fac.loc[0, 'District'] = d  # Level 4, Zomba Mental Hospital

    headquarter_fac = mfl.loc[pd.isnull(mfl['District']) & pd.isnull(mfl['Region']) &
                              (mfl['Facility_Name'] == 'Headquarter')].copy().reset_index(drop=True)
    headquarter_fac.loc[0, 'District'] = d  # Level 5, Headquarter

    facilities_by_district = pd.concat([facilities_by_district, district_facs, region_fac, zmh_fac, headquarter_fac],
                                       ignore_index=True)

# check that the no. of facs is no. of districts times no. of fac levels = 32 * 7 = 224
assert len(facilities_by_district) == len(pop_districts) * len(Facility_Levels)

# Save
facilities_by_district.to_csv(resourcefilepath + 'ResourceFile_Facilities_For_Each_District.csv')

# ---------------------------------------------------------------------------------------------------------------------
# *** Now look at the types of appointments
sheet = pd.read_excel(workingfile, sheet_name='Time_Base', header=None)

# get rid of the junk rows
trimmed = sheet.loc[[7, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27]]
data_import = pd.DataFrame(data=trimmed.iloc[1:, 2:].values, columns=trimmed.iloc[0, 2:], index=trimmed.iloc[1:, 1])

data_import = data_import.dropna(axis='columns', how='all')  # get rid of the 'spacer' columns
data_import = data_import.fillna(0)

# get rid of records for which there is no call on time of any type of officer
data_import = data_import.drop(columns=data_import.columns[data_import.sum() == 0])

# We note that the DCSA (CHW) never has a time requirement and that no appointments can be serviced at the HealthPost.
# We remedy this by inserting a new type of appointment, which only the DCSA can service, \
# and the time taken is 10 minutes.
new_appt_for_CHW = pd.Series(index=data_import.index,
                             name='E01_ConWithDCSA',
                             # New appointment type is a consultation with the DCSA (community health worker)
                             data=[
                                 0,  # Central Hosp - Time
                                 0,  # Central Hosp - Percent
                                 0,  # District Hosp - Time
                                 0,  # District Hosp - Percent
                                 0,  # Comm Hosp - Time
                                 0,  # Comm Hosp - Percent
                                 0,  # Urban Health Centre - Time     #10 mins
                                 0,  # Urban Health Centre - Percent  #100%
                                 0,  # Rural Health Centre - Time     #10 mins
                                 0,  # Rural Health Centre - Percent  #100%
                                 10.0,  # Health Post - Time
                                 1.0,  # Health Post - Percent
                                 0,  # Dispensary - Time              #10 mins
                                 0,  # Dispensary - Percent           #100%
                             ])

data_import = pd.concat([data_import, new_appt_for_CHW], axis=1)

# Add service times for DHOs, which has quite a few data in 'Incidence_Curr', by copying the data of DisHos
new_rows_for_DHO = pd.DataFrame(index=['DHO', 'DHO_Per'], columns=data_import.columns.copy(),
                                data=data_import.loc[['DisHos', 'DisHos_Per'], :].copy().values)

# Add service times (Mental OPD and Mental Clinic Visit) for Zomba Mental Hospital, by copying data of CenHos
new_rows_for_ZMH = pd.DataFrame(index=['ZMH', 'ZMH_Per'], columns=data_import.columns.copy(),
                                data=0)
new_rows_for_ZMH.loc[:, ['C01_MentOPD', 'C01_MentClinic']] = data_import.loc[
    ['CenHos', 'CenHos_Per'], ['C01_MentOPD', 'C01_MentClinic']].copy().values
# If consider all potential cadres from compiled staff return and all associated services
# new_rows_for_ZMH = pd.DataFrame(index=['ZMH','ZMH_Per'],columns=data_import.columns.copy(),
#                               data=data_import.loc[['CenHos','CenHos_Per'],:].copy().values)

data_import = pd.concat([data_import, new_rows_for_DHO, new_rows_for_ZMH])

# data_import ready!

# Break apart composite to give the appt_type and the officer_type
# This is used to know which column to read below...
chai_composite_code = pd.Series(data_import.columns)
chai_code = chai_composite_code.str.split(pat='_', expand=True).reset_index(drop=True)
chai_code = chai_code.rename(columns={0: 'Officer_Type_Code', 1: 'Appt_Type_Code'})

# check that officer codes line up with the officer codes already imported
assert set(chai_code['Officer_Type_Code']).issubset(set(officer_types_table['Officer_Type_Code']))

# Make dataframe summarising the types of appointments

retained_appt_type_code = pd.unique(chai_code['Appt_Type_Code'])

appt_types_table_import = sheet.loc[(1, 2, 6), 2:].transpose().reset_index(drop=True).copy()
appt_types_table_import = appt_types_table_import.rename(columns={1: 'Appt_Cat', 2: 'Appt_Type', 6: 'Appt_Type_Code'})
appt_types_table_import['Appt_Cat'] = pd.Series(appt_types_table_import['Appt_Cat']).fillna(method='ffill')
appt_types_table_import['Appt_Type'] = pd.Series(appt_types_table_import['Appt_Type']).fillna(method='ffill')
appt_types_table_import['Appt_Type_Code'] = pd.Series(appt_types_table_import['Appt_Type_Code']).fillna(method='ffill')
appt_types_table_import = appt_types_table_import.drop_duplicates().reset_index(drop=True)

# starting with the retained appt codes, merge in these descriptions
appt_types_table = pd.DataFrame(data={'Appt_Type_Code': retained_appt_type_code}).merge(appt_types_table_import,
                                                                                        on='Appt_Type_Code', how='left',
                                                                                        indicator=True)

# Fill in the missing information about the appointment type that was added above
appt_types_table.loc[appt_types_table['Appt_Type_Code'] == new_appt_for_CHW.name.split('_')[1], 'Appt_Cat'] = \
    new_appt_for_CHW.name.split('_')[1]
appt_types_table.loc[appt_types_table['Appt_Type_Code'] == new_appt_for_CHW.name.split('_')[1], 'Appt_Type'] = \
    new_appt_for_CHW.name.split('_')[1]

# drop the merge check column
appt_types_table.drop(columns='_merge', inplace=True)

# Replace space with underscore in the Appt_Cat
appt_types_table['Appt_Cat'].replace(to_replace='  ', value='_', regex=True, inplace=True)
appt_types_table['Appt_Cat'].replace(to_replace=' ', value='_', regex=True, inplace=True)

# Check no holes
assert not pd.isnull(appt_types_table).any().any()

# Save
appt_types_table.to_csv(resourcefilepath + 'ResourceFile_Appt_Types_Table.csv')

# ---------------------------------------------------------------------------------------------------------------------
# *** Now, make the ApptTimeTable
# (Table that gives for each appointment, when occurring in each appt_type at each facility type, the time of each \
# type of officer required

# The sheet gives the % of appointments that require a particular type of officer and the time taken if it does
# So, turn that into an Expectation of the time taken for each type of officer (multiplying together)

# This sheet distinguished between different types of facility in terms of the time taken by appointments occurring \
# at each.
# But the CHAI data do not distinguish how many officers work at each different level of facility
# (Available staff counts for only districts (level = 0,1a,1b,2), CenHos (level = 3), and HQ (level = 5))
# Therefore, we will map these to the facility level that have been defined.
# NB. In doing this, we:
# - assume that the time taken for all appointments at each level is modelled by that for the average of \
#     facility types at that level

# CHAI: Headquarter ---> our "headquarter" (level = 5)
# CHAI: Zomba Mental Hospital ---> our 'Zomba Mental Hospital' / 'ZMH' (level = 4)
# CHAI: Central_Hospital ---> our "Referral Hospital" (level = 3)
# CHAI: District_Hospital ---> averaged into our "second level" facilities (level = 2)
# CHAI: DHO ---> averaged into our "second level" facilities (level = 2)
# CHAI: Community_Hospital ---> averaged into our "primary level" facilities (level = 1b)
# CHAI: Urban_HealthCentre ---> averaged into our "primary level" facilities (level = 1a)
# CHAI: Rural_HealthCentre ---> averaged into our "primary level" facilities (level = 1a)
# CHAI: Dispensary ---> averaged into our "primary level" facilities (level = 1a)
# CHAI: HealthPost ---> averaged into our "community level" facilities (level = 0)

# level 4
ZMH_ExpectTime = data_import.loc['ZMH'] * data_import.loc['ZMH_Per']

# Level 3
Central_Hospital_ExpecTime = data_import.loc['CenHos'] * data_import.loc['CenHos_Per']

# level 5; No data available for Headquarter; we assign NAN to it
HQ_ExpecTime = Central_Hospital_ExpecTime.copy()
HQ_ExpecTime.loc[:] = np.nan

# level 2
District_Hospital_ExpecTime = data_import.loc['DisHos'] * data_import.loc['DisHos_Per']
DHO_ExpecTime = data_import.loc['DHO'] * data_import.loc['DHO_Per']

# level 1b
Community_Hospital_ExpecTime = data_import.loc['ComHos'] * data_import.loc['ComHos_Per']

# level 1a
Urban_HealthCentre_ExpecTime = data_import.loc['UrbHC'] * data_import.loc['UrbHC_Per']
Rural_HealthCentre_ExpecTime = data_import.loc['RurHC'] * data_import.loc['RurHC_Per']
Disp_ExpecTime = data_import.loc['Disp'] * data_import.loc['Disp_Per']

# level 0
HealthPost_ExpecTime = data_import.loc['HP'] * data_import.loc['HP_Per']

# Average time for levels 2 and 1a, which have data for more than 1 facility types
Avg_Level2_ExpectTime = (District_Hospital_ExpecTime + DHO_ExpecTime) / 2  # Identical to DisHos Expected Time
Avg_Level1a_ExpectTime = (Disp_ExpecTime + Urban_HealthCentre_ExpecTime + Rural_HealthCentre_ExpecTime) / 3

# Assemble
X = pd.DataFrame({
    5: HQ_ExpecTime,  # (Headquarter)
    4: ZMH_ExpectTime,  # (Zomba Mental Hospital)
    3: Central_Hospital_ExpecTime,  # (our "Referral Hospital" at region level)
    2: Avg_Level2_ExpectTime,  # (DHO and DisHos at second level )
    '1b': Community_Hospital_ExpecTime,  # (ComHos at primary level)
    '1a': Avg_Level1a_ExpectTime,  # (UrbHC,RurHC and Disp at primary level)
    0: HealthPost_ExpecTime  # (HP at community level)
})

assert set(X.columns) == set(Facility_Levels)

# Split out the index into appointment type and officer type
labels = pd.Series(X.index, index=X.index).str.split(pat='_', expand=True)
labels = labels.rename(columns={0: 'Officer_Type_Code', 1: 'Appt_Type_Code'})
Y = pd.concat([X, labels], axis=1)
ApptTimeTable = pd.melt(Y, id_vars=['Officer_Type_Code', 'Appt_Type_Code'],
                        var_name='Facility_Level', value_name='Time_Taken_Mins')

# Confirm that Facility_Level is an int ---> No longer needed, as level 1a and 1b are not integers
# ApptTimeTable['Facility_Level'] = ApptTimeTable['Facility_Level'].astype(int)

# Merge in Officer_Type
ApptTimeTable = ApptTimeTable.merge(officer_types_table, on='Officer_Type_Code')

# confirm that we have the same number of entries as we were expecting
assert len(ApptTimeTable) == len(Facility_Levels) * len(data_import.columns)

# drop the rows that contain no call on resources, including NAN values
ApptTimeTable = ApptTimeTable.drop(ApptTimeTable[ApptTimeTable['Time_Taken_Mins'] == 0].index)
ApptTimeTable = ApptTimeTable.drop(ApptTimeTable[pd.isnull(ApptTimeTable['Time_Taken_Mins'])].index)
# reset index
ApptTimeTable.reset_index(drop=True, inplace=True)

# Save
ApptTimeTable.to_csv(resourcefilepath + 'ResourceFile_Appt_Time_Table.csv')

# ---------------------------------------------------------------------------------------------------------------------
# *** Create a table that determines what kind of appointment can be serviced in each Facility Level
ApptType_By_FacLevel = pd.DataFrame(index=appt_types_table['Appt_Type_Code'],
                                    columns=Facility_Levels,
                                    data=False,
                                    dtype=bool)

for appt_type in ApptType_By_FacLevel.index:
    for fac_level in ApptType_By_FacLevel.columns:
        # Can this appt_type happen at this facility_level?
        # Check to see if ApptTimeTable has any time requirement

        ApptType_By_FacLevel.at[appt_type, fac_level] = \
            ((ApptTimeTable['Facility_Level'] == fac_level) & (ApptTimeTable['Appt_Type_Code'] == appt_type)).any()

ApptType_By_FacLevel = ApptType_By_FacLevel.add_prefix('Facility_Level_')

# Save
ApptType_By_FacLevel.to_csv(resourcefilepath + 'ResourceFile_ApptType_By_FacLevel.csv')

# --- check
# Look to see where different types of staff member need to be located:
# This is just a reverse reading of where there are non-zero requests for time of particular officer-types

Officers_Need_For_Appt = pd.DataFrame(columns=['Facility_Level', 'Appt_Type_Code', 'Officer_Type_Codes'])

for a in appt_types_table['Appt_Type_Code'].values:
    for f in Facility_Levels:

        # get the staff types required for this appt

        block = ApptTimeTable.loc[(ApptTimeTable['Appt_Type_Code'] == a) & (ApptTimeTable['Facility_Level'] == f)]

        if len(block) == 0:
            # no requirement expressed => The appt is not possible at this location
            Officers_Need_For_Appt = Officers_Need_For_Appt.append(
                {'Facility_Level': f,
                 'Appt_Type_Code': a,
                 'Officer_Type_Codes': False
                 }, ignore_index=True)

        else:
            need_officer_types = list(block['Officer_Type_Code'])
            Officers_Need_For_Appt = Officers_Need_For_Appt.append(
                {'Facility_Level': f,
                 'Appt_Type_Code': a,
                 'Officer_Type_Codes': need_officer_types
                 }, ignore_index=True)

# Turn this into the the set of staff that are required for each type of appointment
FacLevel_By_Officer = pd.DataFrame(columns=Facility_Levels,
                                   index=officer_types_table['Officer_Type_Code'].values)
FacLevel_By_Officer = FacLevel_By_Officer.fillna(False)

for o in officer_types_table['Officer_Type_Code'].values:

    for i in Officers_Need_For_Appt.index:

        fac_level = Officers_Need_For_Appt.loc[i].Facility_Level
        officer_types = Officers_Need_For_Appt.loc[i].Officer_Type_Codes

        if officer_types is not False:  # (i.e. such an appointment at such a a facility is possible)

            if o in officer_types:
                FacLevel_By_Officer.loc[(FacLevel_By_Officer.index == o), fac_level] = True

# We note that three officer_types ("T01: Nutrition Staff", "R03: Sonographer" and "RO4: Radiotherapy technician") are\
#  apparently not called by any appointment type

# Assign that the Nutrition Staff will go to the Referral Hospitals (level = 3)
FacLevel_By_Officer.loc['T01', 3] = True

# Assign that the Sonographer will go to the Referral Hospitals (level = 3)
FacLevel_By_Officer.loc['R03', 3] = True

# Assign that the Radiotherapist will go to the Referral Hospitals (level = 3)
FacLevel_By_Officer.loc['R04', 3] = True

# As an option, we could assign staff at HQ to level 5 according to the info of staff
# Get the sets of officers of funded and current staff
fund_staff_HQ = fund_staffing_table[fund_staffing_table['District_Or_Hospital'] == 'Headquarter'].copy()
curr_staff_HQ = curr_staffing_table[curr_staffing_table['District_Or_Hospital'] == 'Headquarter'].copy()
fund_staff_HQ.drop(columns=['District_Or_Hospital', 'Facility_Level', 'Is_DistrictLevel'], inplace=True)
curr_staff_HQ.drop(columns=['District_Or_Hospital', 'Facility_Level', 'Is_DistrictLevel'], inplace=True)
fund_staff_HQ_Positive = fund_staff_HQ.loc[:, (fund_staff_HQ > 0).any(axis=0)]
curr_staff_HQ_Positive = curr_staff_HQ.loc[:, (curr_staff_HQ > 0).any(axis=0)]
# The union of the two sets
staff_call_at_HQ = fund_staff_HQ_Positive.columns.union(curr_staff_HQ_Positive.columns)
# Assign true value to staff_call_at_HQ
for s in staff_call_at_HQ:
    FacLevel_By_Officer.loc[s, 5] = True

# Check that all types of officer are allocated to at least one type of facility excl. HQ/Level_5
assert (FacLevel_By_Officer.iloc[:, 0:6].sum(axis=1) > 0).all()

# Change columns names: 0 -> Facility_Level_0
FacLevel_By_Officer = FacLevel_By_Officer.add_prefix('Facility_Level_')

# ---------------------------------------------------------------------------------------------------------------------
# *** Determine how current staff are allocated to the facilities, \
# based on curr_staffing_table and curr_staff_list

# Create pd.Series for holding the assignments between staff member (in curr_staff_list) and facility_level (in mfl)

curr_facility_assignment = pd.DataFrame(index=curr_staff_list.index, columns=['Staff_ID', 'Facility_ID'])
curr_facility_assignment['Staff_ID'] = curr_staff_list['Staff_ID']

# Loop through each staff member and allocate them to facilities
for staffmember in curr_staff_list.index:
    # Staff within each district
    if curr_staff_list.Is_DistrictLevel[staffmember]:

        chosen_facility_name = (curr_staff_list.loc[staffmember, 'Facility_Level'] + '_' +
                                curr_staff_list.loc[staffmember, 'District_Or_Hospital'])

        chosen_facility_ID = mfl.loc[mfl['Facility_Name'] == chosen_facility_name, 'Facility_ID'].values[0]
    # Staff at CenHos, ZMH and Headquarter
    else:
        chosen_facility_name = curr_staff_list.loc[staffmember, 'District_Or_Hospital']

        chosen_facility_ID = mfl.loc[mfl['Facility_Name'] == chosen_facility_name, 'Facility_ID'].values[0]

    curr_facility_assignment.loc[
        curr_facility_assignment['Staff_ID'] == staffmember, 'Facility_ID'] = chosen_facility_ID

# Do some checks
assert pd.notnull(curr_facility_assignment).all().all()

# More checks; This check takes about 1 minute.
# Check that every appointment that can be raised by someone in any district is going to be possible to be met by at\
# least one of their facilities (if staff numbers (if >0) were not a limiting factor: ie. just checking the \
# distribution of the officer types between the facilities)

for d in pop_districts:

    for a in appt_types_table['Appt_Type_Code'].values:

        # we require that every appt type is possible in at least one facility

        facilities_in_this_district = list(
            facilities_by_district.loc[facilities_by_district['District'] == d, 'Facility_ID'])

        set_of_facility_levels_in_this_district = set(
            facilities_by_district.loc[facilities_by_district['District'] == d, 'Facility_Level'])

        assert len(facilities_in_this_district) == len(Facility_Levels)
        assert set(set_of_facility_levels_in_this_district) == set(Facility_Levels)

        # generate an array of False;
        appt_ok_per_fac = dict(zip(facilities_in_this_district,
                                   np.zeros(len(facilities_in_this_district)) > 0))

        for f in facilities_in_this_district:
            f_level = mfl.loc[mfl['Facility_ID'] == f, 'Facility_Level'].values[0]

            req_officer = set(ApptTimeTable.loc[
                                  (ApptTimeTable['Appt_Type_Code'] == a) &
                                  (ApptTimeTable['Facility_Level'] == f_level),
                                  'Officer_Type_Code'])

            # is there at least one of every type of officer in req_officer at this facility?
            staff_ids = curr_facility_assignment.loc[curr_facility_assignment['Facility_ID'] == f, 'Staff_ID'].values
            staff_types = curr_staff_list.loc[curr_staff_list['Staff_ID'].isin(staff_ids), 'Officer_Type_Code']
            unique_staff_types = set(pd.unique(staff_types))

            appt_ok_in_this_fac = req_officer.issubset(unique_staff_types)
            appt_ok_per_fac[f] = appt_ok_in_this_fac

        assert np.asarray(list(appt_ok_per_fac.values())).any()

# Save
curr_facility_assignment.to_csv(resourcefilepath + 'ResourceFile_Current_Staff_Facility_Assignment.csv')

# ---------------------------------------------------------------------------------------------------------------------
# *** Determine how funded staff are allocated to the facilities \
# based on fund_staffing_table and fund_staff_list

# Create pd.Series for holding the assignments between staff member (in fund_staff_list) and facility_level (in mfl)
fund_facility_assignment = pd.DataFrame(index=fund_staff_list.index, columns=['Staff_ID', 'Facility_ID'])
fund_facility_assignment['Staff_ID'] = fund_staff_list['Staff_ID']

# Loop through each staff member and allocate them to facilities
for staffmember in fund_staff_list.index:
    # Staff within each district
    if fund_staff_list.Is_DistrictLevel[staffmember]:

        chosen_facility_name = (fund_staff_list.loc[staffmember, 'Facility_Level'] + '_' +
                                fund_staff_list.loc[staffmember, 'District_Or_Hospital'])

        chosen_facility_ID = mfl.loc[mfl['Facility_Name'] == chosen_facility_name, 'Facility_ID'].values[0]
    # Staff at CenHos, ZMH and Headquarter
    else:
        chosen_facility_name = fund_staff_list.loc[staffmember, 'District_Or_Hospital']

        chosen_facility_ID = mfl.loc[mfl['Facility_Name'] == chosen_facility_name, 'Facility_ID'].values[0]

    fund_facility_assignment.loc[
        fund_facility_assignment['Staff_ID'] == staffmember, 'Facility_ID'] = chosen_facility_ID

# Do some checks that there is no nan entry
# Be careful with ~ when do assert
assert pd.notnull(fund_facility_assignment).all().all()
# Alternative: assert not (pd.isnull(fund_facility_assignment).any().any())

# Check that every appointment that can be raised by someone in any district is going to be possible to be met by at\
# least one of their facilities (if staff numbers (if >0) were not a limiting factor: ie. just checking the \
# distribution of the officer types between the facilities)

for d in pop_districts:

    for a in appt_types_table['Appt_Type_Code'].values:

        # we require that every appt type is possible in at least one facility

        facilities_in_this_district = list(
            facilities_by_district.loc[facilities_by_district['District'] == d, 'Facility_ID'])

        set_of_facility_levels_in_this_district = set(
            facilities_by_district.loc[facilities_by_district['District'] == d, 'Facility_Level'])

        assert len(facilities_in_this_district) == len(Facility_Levels)
        assert set(set_of_facility_levels_in_this_district) == set(Facility_Levels)

        # generate an array of False;
        appt_ok_per_fac = dict(zip(facilities_in_this_district,
                                   np.zeros(len(facilities_in_this_district)) > 0))

        for f in facilities_in_this_district:
            f_level = mfl.loc[mfl['Facility_ID'] == f, 'Facility_Level'].values[0]

            req_officer = set(ApptTimeTable.loc[
                                  (ApptTimeTable['Appt_Type_Code'] == a) &
                                  (ApptTimeTable['Facility_Level'] == f_level),
                                  'Officer_Type_Code'])

            # is there at least one of every type of officer in req_officer at this facility?
            staff_ids = fund_facility_assignment.loc[fund_facility_assignment['Facility_ID'] == f, 'Staff_ID'].values
            staff_types = fund_staff_list.loc[fund_staff_list['Staff_ID'].isin(staff_ids), 'Officer_Type_Code']
            unique_staff_types = set(pd.unique(staff_types))

            appt_ok_in_this_fac = req_officer.issubset(unique_staff_types)
            appt_ok_per_fac[f] = appt_ok_in_this_fac

        assert np.asarray(list(appt_ok_per_fac.values())).any()

# Save
fund_facility_assignment.to_csv(resourcefilepath + 'ResourceFile_Funded_Staff_Facility_Assignment.csv')

# Check that tables for current staff and funded staff have the same columns
assert (fund_facility_assignment.columns == curr_facility_assignment.columns).all()

# ---------------------------------------------------------------------------------------------------------------------
# *** Get Hours and Minutes Worked Per Staff Member, i.e., the daily capabilities
# Reading-in the number of working hours and days for each type of officer

pft_sheet = pd.read_excel(workingfile, sheet_name='PFT', header=None)
officer_types_import = pft_sheet.iloc[2, np.arange(2, 23)]

assert set(officer_types_import) == set(officer_types_table['Officer_Type_Code'])
assert len(officer_types_import) == len(officer_types_table['Officer_Type_Code'])

# patient facing hours daily at hospitals
hours_hospital = pft_sheet.iloc[38, np.arange(2, 23)]

# patient facing hours daily at health centres
work_mins_hc = pft_sheet.iloc[26, np.arange(2, 23)]
admin_mins_hc = pft_sheet.iloc[34, np.arange(2, 23)]
hours_hc = (work_mins_hc - admin_mins_hc) / 60

# Total working days per year
days_per_year_men = pft_sheet.iloc[15, np.arange(2, 23)]
days_per_year_women = pft_sheet.iloc[16, np.arange(2, 23)]
days_per_year_pregwomen = pft_sheet.iloc[17, np.arange(2, 23)]

# Percents of men, nonpregnant women, and pregnant women
fr_men = pft_sheet.iloc[53, np.arange(2, 23)]
fr_pregwomen = pft_sheet.iloc[55, np.arange(2, 23)] * pft_sheet.iloc[57, np.arange(2, 23)]
fr_nonpregwomen = pft_sheet.iloc[55, np.arange(2, 23)] * (1 - pft_sheet.iloc[57, np.arange(2, 23)])

# Total average working days
workingdays = (fr_men * days_per_year_men) + (fr_nonpregwomen * days_per_year_women) + (
    fr_pregwomen * days_per_year_pregwomen)

# --- patient facing time
# Average mins per year, Average hours per day, Average number of mins per day in Malawi

mins_per_day_hospital = hours_hospital * 60
mins_per_day_hc = hours_hc * 60

mins_per_year_hospital = mins_per_day_hospital * workingdays
mins_per_year_hc = mins_per_day_hc * workingdays

av_mins_per_day_hospital = mins_per_year_hospital / 365.25
av_mins_per_day_hc = mins_per_year_hc / 365.25

# PFT - hospital and health centre individually
HosHC_patient_facing_time = pd.DataFrame(
    {'Officer_Type_Code': officer_types_import, 'Working_Days_Per_Year': workingdays,
     'Hospital_Hours_Per_Day': hours_hospital, 'HC_Hours_Per_Day': hours_hc,
     'Hospital_Av_Mins_Per_Day': av_mins_per_day_hospital,
     'HC_Av_Mins_Per_Day': av_mins_per_day_hc}
).reset_index(drop=True)

# PFT table ready!

# Create final tables of daily time available at each facility by officer type: Facility_ID, Facility_Type,
# Facility_Level, Officer_Type, Officer_Type_Code, Total Average Minutes Per Day, Staff_Count

# --- Daily capability for current staff

# Merge in officer type
X = curr_staff_list.merge(curr_facility_assignment, on='Staff_ID', how='left')
X.drop(columns=['Is_DistrictLevel'], inplace=True)

# Add average mins per day from HosHC_patient_facing_time table
# We use HosHc_patient_facing_time table to allocate HC mins/hours to staff at level 1a and below,\
# and Hospital mins/hours to staff at level 1b and above. (Time Differences are happening only at M01,M02,M03,N01.)
X_HC = X.loc[X['Facility_Level'].isin(['Facility_Level_1a', 'Facility_Level_0'])].copy()
X_Hos = X.loc[~X['Facility_Level'].isin(['Facility_Level_1a', 'Facility_Level_0'])].copy()

X_HC = X_HC.merge(HosHC_patient_facing_time, on='Officer_Type_Code', how='left')
X_HC.drop(columns=['Working_Days_Per_Year', 'Hospital_Hours_Per_Day', 'Hospital_Av_Mins_Per_Day', 'HC_Hours_Per_Day'],
          inplace=True)
X_HC.rename(columns={'HC_Av_Mins_Per_Day': 'Av_Mins_Per_Day'}, inplace=True)

X_Hos = X_Hos.merge(HosHC_patient_facing_time, on='Officer_Type_Code', how='left')
X_Hos.drop(columns=['Working_Days_Per_Year', 'HC_Hours_Per_Day', 'HC_Av_Mins_Per_Day', 'Hospital_Hours_Per_Day'],
           inplace=True)
X_Hos.rename(columns={'Hospital_Av_Mins_Per_Day': 'Av_Mins_Per_Day'}, inplace=True)

# Concat X_HC and X_Hos
assert (X_HC.columns == X_Hos.columns).all()
X = pd.concat([X_HC, X_Hos])

# sort so that the table's index is identical to curr_staff_list
X.set_index(['Staff_ID'], inplace=True)
X.sort_index(inplace=True)
X.reset_index(drop=False, inplace=True)
assert (X.index.values == X['Staff_ID'].values).all()

# Now collapse across the staff_ID in order to give statistics per facility type and officer type:
# total mins per day, staff count
curr_daily_capability = pd.DataFrame(
    X.groupby(['Facility_ID', 'Officer_Type_Code']).agg(
        Total_Mins_Per_Day=('Av_Mins_Per_Day', 'sum'),
        Staff_Count=('Staff_ID', 'count')
    )).reset_index()

# Merge in information about the facility:
curr_daily_capability = curr_daily_capability.merge(mfl, on='Facility_ID', how='left')

# Merge in the officer code name
curr_daily_capability = curr_daily_capability.merge(officer_types_table, on='Officer_Type_Code', how='left')

# Checks: every facility has at least one person in
assert ((curr_daily_capability.groupby('Facility_ID')['Facility_Level'].count()) > 0).all()
assert ((curr_daily_capability.groupby('Facility_ID')['Staff_Count'].sum()) > 0).all()

# --- Daily capability for established staff

# Merge in officer type
Y = fund_staff_list.merge(fund_facility_assignment, on='Staff_ID', how='left')
Y.drop(columns=['Is_DistrictLevel'], inplace=True)

# Add average mins per day from HosHC_patient_facing_time table
# We use HosHc_patient_facing_time table to allocate HC mins/hours to staff at level 1a and below,
# and Hospital mins/hours to staff at level 1b and above. (Time Differences are happening only at M01,M02,M03,N01.)
Y_HC = Y.loc[Y['Facility_Level'].isin(['Facility_Level_1a', 'Facility_Level_0'])].copy()
Y_Hos = Y.loc[~Y['Facility_Level'].isin(['Facility_Level_1a', 'Facility_Level_0'])].copy()

Y_HC = Y_HC.merge(HosHC_patient_facing_time, on='Officer_Type_Code', how='left')
Y_HC.drop(columns=['Working_Days_Per_Year', 'Hospital_Hours_Per_Day', 'Hospital_Av_Mins_Per_Day', 'HC_Hours_Per_Day'],
          inplace=True)
Y_HC.rename(columns={'HC_Av_Mins_Per_Day': 'Av_Mins_Per_Day'}, inplace=True)

Y_Hos = Y_Hos.merge(HosHC_patient_facing_time, on='Officer_Type_Code', how='left')
Y_Hos.drop(columns=['Working_Days_Per_Year', 'HC_Hours_Per_Day', 'HC_Av_Mins_Per_Day', 'Hospital_Hours_Per_Day'],
           inplace=True)
Y_Hos.rename(columns={'Hospital_Av_Mins_Per_Day': 'Av_Mins_Per_Day'}, inplace=True)

# Concat Y_HC and Y_Hos
assert (Y_HC.columns == Y_Hos.columns).all()
Y = pd.concat([Y_HC, Y_Hos])

# sort so that the table's index is identical to fund_staff_list
Y.set_index(['Staff_ID'], inplace=True)
Y.sort_index(inplace=True)
Y.reset_index(drop=False, inplace=True)
assert (Y.index.values == Y['Staff_ID'].values).all()

# Now collapse across the staff_ID in order to give statistics per facility type and officer type:
# total mins per day, staff count
fund_daily_capability = pd.DataFrame(
    Y.groupby(['Facility_ID', 'Officer_Type_Code']).agg(
        Total_Mins_Per_Day=('Av_Mins_Per_Day', 'sum'),
        Staff_Count=('Staff_ID', 'count')
    )).reset_index()

# Merge in information about the facility:
fund_daily_capability = fund_daily_capability.merge(mfl, on='Facility_ID', how='left')

# Merge in the officer code name
fund_daily_capability = fund_daily_capability.merge(officer_types_table, on='Officer_Type_Code', how='left')

# Checks: every facility has at least one person in
assert ((fund_daily_capability.groupby('Facility_ID')['Facility_Level'].count()) > 0).all()
assert ((fund_daily_capability.groupby('Facility_ID')['Staff_Count'].sum()) > 0).all()

# Last check that tables for current and funded staff have same columns
assert (fund_daily_capability.columns == curr_daily_capability.columns).all()

# Save
HosHC_patient_facing_time.to_csv(resourcefilepath + 'ResourceFile_Patient_Facing_Time.csv')
curr_daily_capability.to_csv(resourcefilepath + 'ResourceFile_Current_Staff_Daily_Capabilities.csv')
fund_daily_capability.to_csv(resourcefilepath + 'ResourceFile_Funded_Staff_Daily_Capabilities.csv')
