"""
This file outputs the current staffing
"""

import pandas as pd
import numpy as np

ResourceFile_MasterFacilitiesList = '/Users/tbh03/PycharmProjects/TLOmodel/resources/ResourceFile_MasterFacilitiesList.csv'


workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/ORIGINAL_Optimization model import_Malawi_20180315 v10.xlsx'

output_path='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/'


# Import all of the 'CurrentStaff' sheet
wb_import= pd.read_excel(workingfile,sheet_name='CurrentStaff',header=None)

# Make dataframe summarising the officer types and the officer codes:
officer_types_table=wb_import.loc[2:3,64:84].transpose().reset_index(drop=True).copy()
officer_types_table.columns={'Officer_Type','Officer_Type_Code'}
officer_types_table.to_csv(output_path + 'officer_types_table.csv')

# Extract just the section about "Funded TOTAl Staff'
wb_extract = wb_import.loc[3:37,64:84]
wb_extract=wb_extract.drop([4,5])
wb_extract.columns = wb_extract.iloc[0]
wb_extract=wb_extract.drop([3])
wb_extract=wb_extract.reset_index(drop=True)
wb_extract.fillna(0, inplace=True) # replace all null values with zero values

# Add in the colum to the dataframe for the districts:
districts= wb_import.loc[6:37,0].reset_index(drop=True)
wb_extract.loc[:,'District'] = districts

# Finished import from the CHAI excel:
staffing_table = wb_extract

# ----------

# Check that the district name will match up with what is in
chai_districts = pd.unique( staffing_table ['District'] )

# get the districts from the resource file:
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
staffing_table .loc[staffing_table ['District']=='KCH','District'] = 'Lilongwe'
staffing_table .loc[staffing_table ['District']=='HQ or missing','District'] = 'Lilongwe'
staffing_table .loc[staffing_table ['District']=='MCH','District'] = 'Mzimba'
staffing_table .loc[staffing_table ['District']=='QECH','District'] = 'Blantyre'
staffing_table .loc[staffing_table ['District']=='ZCH','District'] = 'Zomba'

# Collapse any duplicated districts together:
staffing_table=pd.DataFrame(staffing_table.groupby(by=['District']).sum()).reset_index()

# Add in Likoma (with zero health workers)
record = staffing_table.loc[staffing_table['District']==chai_districts[0]].copy()
record['District']='Likoma'
record.iloc[0,1:]=0
staffing_table =staffing_table .append(record).reset_index(drop=True)


# Now re-confirm the merging will be perfect:
assert set(pd.unique(pop['District'])) == set(staffing_table['District'])

    # ... double check by doing the merge explicitly
    pop_districts = pd.DataFrame({'District': pd.unique(pop['District'])})
    chai_districts = pd.DataFrame({'District': staffing_table['District']})

    merge_result=pop_districts.merge(chai_districts,how='inner',indicator=True)
    assert all(merge_result['_merge']=='both')
    assert len(merge_result) == len(pop_districts)


# check that every district in the resource file is represented in the CHAI list:
for d in pop_districts.values:
    print('Resource File district, ', d , ' in chai tables: ', (d in chai_districts.values))

# check that every district in the chai table is represented in the resoure file list:
for d in chai_districts.values:
    print('Chai file district, ', d , ' in resource file: ', (d in pop_districts.values))


# --------------

# Flip from wide to long format
staff_list=pd.melt(staffing_table, id_vars=['District'], var_name='Officer_Type', value_name='Number')

# Repeat rows so that it is one row per staff member
staff_list['Number']=staff_list['Number'].astype(int)
staff_list = staff_list.loc[staff_list.index.repeat(staff_list['Number'])]
staff_list = staff_list.reset_index(drop=True)
staff_list=staff_list.drop(['Number'],axis=1)

# check that the number of rows in this staff_list is equal to the total number of staff from the input data
assert len(staff_list) == staffing_table.iloc[:,1:].sum().sum()


#  --------
#  Determine how these staff members will be allocated to a facility

# Create a mapping of how these different officer types will go to particular facility types
# The level is important as this determines how many other people are drawing on their time.

# Load up the Master Facilities List
mfl=pd.read_csv(ResourceFile_MasterFacilitiesList)

# to ensure that the types of facilities used here conforms
facility_types=pd.unique(mfl.Facility_Type)

Facility_By_Officer=pd.DataFrame({'Officer_Type': officer_types_table.Officer_Type, 'Officer_Type_Code':officer_types_table.Officer_Type_Code ,'Facility_Type_Can_Work_In':None})

Facility_By_Officer.at[Facility_By_Officer['Officer']=='Medical Officer / Specialist','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Clinical Officer / Technician','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Med. Assistant','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Nurse Officer','Facility_Type_Can_Work_In']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Nurse Midwife Technician','Facility_Type_Can_Work_In']={'District Hospital', 'Community Health Worker', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Pharmacist','Facility_Type_Can_Work_In']={'District Hospital', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Pharm Technician','Facility_Type_Can_Work_In']={'District Hospital', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Pharm Assistant','Facility_Type_Can_Work_In']={'District Hospital', 'Health Centre','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Lab Officer','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Lab Technician','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Lab Assistant','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='DCSA','Facility_Type_Can_Work_In']={'Community Health Worker'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Dental Officer','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Dental Therapist','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Dental Assistant','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Mental Health Staff','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Nutrition Staff','Facility_Type_Can_Work_In']={'District Hospital','Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Radiographer','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Radiography Technician','Facility_Type_Can_Work_In']={'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Sonographer','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer']=='Radiotherapy Technician','Facility_Type_Can_Work_In']={'Referral Hospital'}




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




#-----------------

# Reading-in the number of working hours and days for each type of officer

sheet= pd.read_excel(workingfile,sheet_name='PFT',header=None)
officer_types_import=sheet.iloc[2,np.arange(2,23)]

assert set(officer_types_import) == set( officer_types_table['Officer_Type_Code'] )


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

# What we will use is the average number of minutes worked per day:
av_minutes_per_day = minutes_per_year / 365.25

patient_facing_hours = pd.DataFrame({'Officer_Type': officer_types_import,'Working_Days_Per_Year':workingdays,'Hours_Per_Day':hours, 'Av_Minutes_Per_Day':av_minutes_per_day })




# --- Create final file: Facility_ID, Officer Type, Total Average Minutes Per Day


# ADD FACILITY TYPE

outputfile_time_per_facility='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/ResourceFile_Time_Per_Facility.csv'

# merge in officer type
X=Facility_Assignment.merge(stafflist, on='Staff_ID')
X=X.drop(columns='District')

# merge in time that each officer type can spent on appointments
X= X.merge(patient_facing_hours,on='Officer')

# Now collapse across the staff_ID in order to give a summary per facility type and officer type
Y = pd.DataFrame(X.groupby(['Facility_ID','Officer'])[['Av_Minutes_Per_Day']].sum()).reset_index()

Y.to_csv(outputfile_time_per_facility)

