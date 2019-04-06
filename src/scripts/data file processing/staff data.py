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

# Decide which type of count we want to use: For simplicity, now use the funded positions:
wb.loc[wb['Count']=='CURRENT TOTAL STAFF (NON-VACANT POSITIONS)','Nurse Officer'].sum()
wb.loc[wb['Count']=='Funded TOTAL STAFF (NON-VACANT POSITIONS)','Nurse Officer'].sum()


# Flip from wide to long format
current_staff_long=pd.melt(wb.loc[wb['Count']=='Funded TOTAL STAFF (NON-VACANT POSITIONS)'], id_vars=['District','Count'], value_vars=officer_types,var_name='Officer', value_name='Number')

# Repeat rows so that it is one row per staff member
current_staff_long['Number']=current_staff_long['Number'].astype(int)
stafflist = current_staff_long.loc[current_staff_long.index.repeat(current_staff_long['Number'])]
#reset insex
stafflist = stafflist.reset_index(drop=True)
stafflist=stafflist.drop(['Number','Count'],axis=1)

len(stafflist)
current_staff_long['Number'].sum()

stafflist.loc[:,'Staff_ID']=pd.Series(stafflist.index)
stafflist.to_csv(outputfile_current_staff)

# Could identify the CHAM staff and sort those to particular facilities:


#-----------------

# Now to write to the file the number of patient-facing hours per day

outputfile_working_hours='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/ResourceFile_CurrentStaffWorkingHours.csv'

sheet= pd.read_excel(workingfile,sheet_name='PFT',header=None)
officers=sheet.iloc[1,np.arange(2,23)]
hours=sheet.iloc[38,np.arange(2,23)]

days_per_year_men=sheet.iloc[15,np.arange(2,23)]
days_per_year_women=sheet.iloc[16,np.arange(2,23)]
days_per_year_pregwomen=sheet.iloc[17,np.arange(2,23)]

fr_men=sheet.iloc[53,np.arange(2,23)]
fr_women=sheet.iloc[55,np.arange(2,23)] - sheet.iloc[57,np.arange(2,23)] # taking off pregnant women: this is for non-pregnant women
fr_pregwomen= sheet.iloc[57,np.arange(2,23)]

workingdays= ( fr_men*days_per_year_men ) + ( fr_women*days_per_year_women ) + ( fr_pregwomen*days_per_year_pregwomen )

minutes_per_day = hours * 60

minutes_per_year = minutes_per_day * workingdays

patient_facing_hours = pd.DataFrame({'Officer': officers,'Working_Days_Per_Year':workingdays,'Hours_Per_Day':hours})
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


Facility_By_Officer=pd.DataFrame({'Officer': officer_types, 'Facility_Type':None})

Facility_By_Officer.at[Facility_By_Officer['Officer']=='Medical Officer / Specialist','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Clinical Officer / Technician','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Med. Assistant','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Nurse Officer','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Nurse Midwife Technician','Facility_Type']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Pharmacist','Facility_Type']={'District Hospital', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Pharm Technician','Facility_Type']={'District Hospital', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Pharm Assistant','Facility_Type']={'District Hospital', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Lab Officer','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Lab Technician','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Lab Assistant','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='DCSA','Facility_Type']={'Community Health Worker'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Dental Officer','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Dental Therapist','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Dental Assistant','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Mental Health Staff','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Nutrition Staff','Facility_Type']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Radiographer','Facility_Type']={'District Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Radiography Technician','Facility_Type']={'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Sonographer','Facility_Type']={'District Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Radiotherapy Technician','Facility_Type']={'Referral Hospital'}





# Load the master facilities list:

ResourceFile_MasterFacilitiesList = '/Users/tbh03/PycharmProjects/TLOmodel/resources/ResourceFile_MasterFacilitiesList.csv'

fac=pd.read_csv(ResourceFile_MasterFacilitiesList)

# Prepare dataframe for holding the assignments between Staff_ID and Facility_ID
Facility_Assignment =pd.DataFrame(stafflist['Staff_ID'])
Facility_Assignment.loc[:,'Facility_ID']=None

# Loop through each staff member and allocate them to

for staffmember in stafflist.index:

    officer =stafflist.loc[staffmember].Officer

    district= stafflist.loc[staffmember].District

    fac_types= Facility_By_Officer.loc[Facility_By_Officer['Officer']==officer,'Facility_Type']
    fac_types_set=fac_types.iloc[0]

    # convert the set of relevant facilities to a list
    fac_types_list = []
    for x in fac_types_set:
        fac_types_list.append(x)

    # Get the facilities to which this staff member might be allocated
    suitable_facilities=fac.loc[ ( fac['Facility Type'].isin(fac_types_list) ) & ( fac['District'] == district ) ]

    # what is there are not any suitable facilities????

    # choose one for this staff member:


    if len(suitable_facilities)>0 :

        # if there is a suitable facility for this office to go into
        x=np.random.choice(len(suitable_facilities))

        assigned_facility=suitable_facilities.iloc[x].Facility_ID

        Facility_Assignment.at[staffmember,'Facility_ID']=assigned_facility

    else:

        # if there is no suitable facility for this officer: assign randomly

        fac_in_any_district=fac.loc[fac['Facility Type'].isin(fac_types_list)]
        x = np.random.choice(len(fac_in_any_district))

        Facility_Assignment.at[staffmember, 'Facility_ID'] = fac_in_any_district.iloc[x].Facility_ID


# perform checks!!!

outputfile_facility_assignment='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/ResourceFile_StaffAssignmentToFacility.csv'

Facility_Assignment.to_csv(outputfile_facility_assignment)



