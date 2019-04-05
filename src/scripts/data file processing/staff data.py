"""
This file outputs the current staffing
"""

import pandas as pd
import numpy as np

workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/Formatting for ResourceFile.xlsx'

outputfile_current_staff='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/ResourceFile_CurrentStaff.csv'


wb =pd.read_excel(workingfile,sheet_name='RF_CurrentStaff')


# replace all null values with zero values

wb.fillna(0, inplace=True)

# check that the district will match up:
chai_districts = pd.unique( wb['District'] )

# get the distrcts from the resource file:
pop = pd.read_csv('/Users/tbh03/PycharmProjects/TLOmodel/resources/ResourceFile_PopBreakdownByVillage.csv')

pop_districts = pd.unique(pop['District'])


# check that every district in the resource file is represented in the CHAI list:
for d in pop_districts:
    print('Resource File district, ', d , ' in chai tables: ', (d in chai_districts))
# Likoma not in the Chai data (which may be classified as Nkhata Bay)


# check that every district in the chai table is represented in the resoure file list:
for d in chai_districts:
    print('Chai file district, ', d , ' in resource file: ', (d in pop_districts))
# HQ or missing', 'KCH', 'MCH', 'QECH' not in the pop data for districts


# Remedy the mismatches:

wb.loc[wb['District']=='KCH','District'] = 'Lilongwe'

wb.loc[wb['District']=='HQ or missing','District'] = 'Lilongwe'

wb.loc[wb['District']=='MCH','District'] = 'Mzimba'

wb.loc[wb['District']=='QECH','District'] = 'Blantyre'

wb.loc[wb['District']=='ZCH','District'] = 'Zomba'

# Add in Likoma (with zero health workers)
record = wb.loc[wb['District']==chai_districts[0]].copy()
record['District']='Likoma'
cols=record.columns
officer_types=cols[2:]
for c in officer_types:
    record.iloc[record.index >= 0, record.columns == c]=0

wb=wb.append(record)

# now check that merging:
pop_districts = pd.unique(pop['District'])
chai_districts = pd.unique( wb['District'] )

# check that every district in the resource file is represented in the CHAI list:
for d in pop_districts:
    print('Resource File district, ', d , ' in chai tables: ', (d in chai_districts))

# check that every district in the chai table is represented in the resoure file list:
for d in chai_districts:
    print('Chai file district, ', d , ' in resource file: ', (d in pop_districts))

# Decide which type of count we want to use:
wb.loc[wb['Count']=='CURRENT TOTAL STAFF (NON-VACANT POSITIONS)','Nurse Officer'].sum()
wb.loc[wb['Count']=='Funded TOTAL STAFF (NON-VACANT POSITIONS)','Nurse Officer'].sum()

# Flip from wide to long format
current_staff=pd.melt(wb, id_vars=['District','Count'], value_vars=officer_types,var_name='Officer', value_name='Number')

current_staff.to_csv(outputfile_current_staff)


#-----------------

# Now to write to the file the number of patient-facing hours per day

outputfile_working_hours='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/ResourceFile_CurrentStaffWorkingHours.csv'

sheet= pd.read_excel(workingfile,sheet_name='PFT',header=None)
officers=sheet.iloc[1,np.arange(2,23)]
hours=sheet.iloc[38,np.arange(2,23)]

patient_facing_hours = pd.DataFrame({'Officer': officers,'Hours':hours})
patient_facing_hours.to_csv(outputfile_working_hours)


# ---- check that the column name and types of officers are aligned

names_hourssheet=patient_facing_hours['Officer'].values
names_staffsheet = pd.unique(current_staff['Officer'])

for n in names_hourssheet:
    print(n in names_staffsheet)

for n in names_staffsheet:
    print(n in names_hourssheet)




#  -------- Examine how these staff members will be allocated to a facility

# Create a mapping of how these different officer types will go to particular facility types
# The level is important (as this determines how many other people are drawing on their time)

array(['District Hospital', 'Community Health Worker', 'Health Centre',
       'Hospital', 'Referral Hospital'])


Facility_By_Officer=pd.DataFrame({'Officer': officer_types, 'Facility_Type':None})

Facility_By_Officer.at[Facility_By_Officer['Officer']=='Medical Officer / Specialist','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Clinical Officer / Technician','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Med. Assistant','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Nurse Officer','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Nurse Midwife Technician','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Pharmacist','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Pharm Technician','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Pharm Assistant','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Lab Officer','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Lab Technician','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Lab Assistant','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='DCSA','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Dental Officer','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Dental Therapist','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Dental Assistant','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Mental Health Staff','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Nutrition Staff','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Radiographer','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Radiography Technician','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Sonographer','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Radiotherapy Technician','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}



ResourceFile_MasterFacilitiesList = '/Users/tbh03/PycharmProjects/TLOmodel/resources/ResourceFile_MasterFacilitiesList.csv'

fac=pd.read_csv(ResourceFile_MasterFacilitiesList)

fac_in_Balaka=fac.loc[fac['District']=='Balaka'].copy()

staff_in_Balaka=current_staff.loc[current_staff['District']=='Balaka'].copy()

MedOff_in_Balaka=staff_in_Balaka.loc[staff_in_Balaka['Officer']==officer_types[0]]


