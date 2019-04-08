"""

This file set ups the health system resources for each district.

It creates one facility of each type for each district, and allows the 'Referral Hospitals' to connect to all the
districts in a region,  and the 'National' Hospital' to connect to all districts in the country.

"""

# CHAI DATA SET:
workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/ORIGINAL_Optimization model import_Malawi_20180315 v10.xlsx'

# OUTPUT RESOURCE_FILES TO:
resourcefilepath='/Users/tbh03/PycharmProjects/TLOmodel/resources/'


# ----------
# ----------
# ----------

import pandas as pd
import numpy as np

# ----------

# Import all of the 'CurrentStaff' sheet
wb_import= pd.read_excel(workingfile,sheet_name='CurrentStaff',header=None)

# ----------

# Make dataframe summarising the officer types and the officer codes:
officer_types_table=wb_import.loc[2:3,64:84].transpose().reset_index(drop=True).copy()
officer_types_table.columns=['Officer_Type','Officer_Type_Code']
officer_types_table.to_csv(resourcefilepath + 'ResourceFile_officer_types_table.csv')

# ----------

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

# ** Check that the district name will match up with what is in
chai_districts = pd.unique( staffing_table ['District'] )

# get the districts from the resource file:
pop = pd.read_csv(resourcefilepath+'ResourceFile_DistrictPopulationData.csv')
pop_districts = pop['District'].values

# check that every district in the resource file is represented in the CHAI list:
for d in pop_districts:
    print('Resource File district, ', d , ' in chai tables: ', (d in chai_districts))
# The city divisions are missing

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


# Get blank record for inserting the missing districts:
record = staffing_table.loc[staffing_table['District']==chai_districts[0]].copy()
record['District']=None
record.iloc[0,1:]=0

# The following districts are not in the CHAI data because they are included within other districts.
# For now, we will say thay the division beween these cities and the wide district (in which they are included) is equal.

# Add in Likoma (part Nkhata Bay)
record['District']='Likoma'
super_district= 'Nkhata Bay'
record.loc[0,staffing_table.columns[1:]] = (0.5 * (staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]]).values.squeeze())
staffing_table =staffing_table.append(record).reset_index(drop=True)
staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]]= staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]] - record.loc[0,staffing_table.columns[1:]]

# Add in Lilongwe City (part of Lilongwe)
record['District']='Lilongwe City'
super_district= 'Lilongwe'
record.loc[0,staffing_table.columns[1:]] = (0.5 * (staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]]).values.squeeze()).astype(int)
staffing_table =staffing_table.append(record).reset_index(drop=True)
staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]]= staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]] - record.loc[0,staffing_table.columns[1:]]

# Add in Mzuzu City
record['District']='Mzuzu City'
super_district= 'Mzimba'
record.loc[0,staffing_table.columns[1:]] = (0.5 * (staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]]).values.squeeze()).astype(int)
staffing_table =staffing_table.append(record).reset_index(drop=True)
staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]]= staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]] - record.loc[0,staffing_table.columns[1:]]

# Add in Zomba City (part of Zomba)
record['District']='Zomba City'
super_district= 'Zomba'
record.loc[0,staffing_table.columns[1:]] = (0.5 * (staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]]).values.squeeze()).astype(int)
staffing_table =staffing_table.append(record).reset_index(drop=True)
staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]]= staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]] - record.loc[0,staffing_table.columns[1:]]

# Add in Blantyre City (part of Blantyre)
record['District']='Blantyre City'
super_district= 'Blantyre'
record.loc[0,staffing_table.columns[1:]] =(0.5 * (staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]]).values.squeeze()).astype(int)
staffing_table =staffing_table.append(record).reset_index(drop=True)
staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]]= staffing_table.loc[staffing_table['District']==super_district,staffing_table.columns[1:]] - record.loc[0,staffing_table.columns[1:]]


# Now re-confirm the merging will be perfect:
assert set(pop['District'].values) == set(staffing_table['District'])
assert len(pop['District'].values) == len(staffing_table['District'])

# ... double check by doing the merge explicitly
pop_districts = pd.DataFrame({'District': pd.unique(pop['District'])})
chai_districts = pd.DataFrame({'District': staffing_table['District']})

merge_result=pop_districts.merge(chai_districts,how='inner',indicator=True)
assert all(merge_result['_merge']=='both')
assert len(merge_result) == len(pop_districts)


# --------------

# Flip from wide to long format
staffing_table.loc[:,staffing_table.columns[1:]]=staffing_table.loc[:,staffing_table.columns[1:]].round(0).astype(int) # Make values integers (after rounding)

staff_list=pd.melt(staffing_table, id_vars=['District'], var_name='Officer_Type_Code', value_name='Number')

# Repeat rows so that it is one row per staff member
staff_list['Number']=staff_list['Number'].astype(int)
staff_list = staff_list.loc[staff_list.index.repeat(staff_list['Number'])]
staff_list = staff_list.reset_index(drop=True)
staff_list=staff_list.drop(['Number'],axis=1)

# check that the number of rows in this staff_list is equal to the total number of staff from the input data
assert len(staff_list) == staffing_table.iloc[:,1:].sum().sum()


#  --------

#  Determine how these staff members will be allocated to a facility

# Create the Master Facilities List
# This will be a listing of each facility and the district(s) to which they attach
# We assume that the set of facilities of one type operate as one big facility type within the district.

# declare the Facility_Type variable
Facility_Types = ['Community Health Worker', 'Health Centre', 'Non-District Hospital', 'District Hospital', 'Referral Hospital', 'National Hospital']

# Create empty dataframe that will be the Master Facilities List (mfl)
mfl = pd.DataFrame(columns= ['Facility_Type','Facility_Level','District','Region'])

pop_districts = pop['District'].values
pop_regions = pd.unique(pop['Region'])

for d in pop_districts:
    df = pd.DataFrame({'Facility_Type': Facility_Types[0:4], 'Facility_Level' : [0, 1, 2, 3], 'District': d, 'Region': pop.loc[pop['District']==d,'Region'].values[0] })
    mfl= mfl.append(df,ignore_index=True,sort=True)

# Add in the Referral Hospitals, one for each region
for r in pop_regions:
    mfl=mfl.append(pd.DataFrame({
        'Facility_Type': Facility_Types[4], 'Facility_Level' : [4], 'District': None, 'Region': r
    }),ignore_index=True,sort=True)

# Add in the National Hosital, one for whole country
mfl=mfl.append(pd.DataFrame({
    'Facility_Type': Facility_Types[5], 'Facility_Level' : [5], 'District': None, 'Region': None
}),ignore_index=True,sort=True)

# Create the Facility_ID
mfl.loc[:,'Facility_ID']=mfl.index

# Create a unique name for each Facility
name=mfl['Facility_Type'] + '_' + mfl['District']
name.loc[mfl['Facility_Type']=='Referral Hospital'] ='Referral Hospital' + '_' + mfl.loc[mfl['Facility_Type']=='Referral Hospital','Region']
name.loc[mfl['Facility_Type']=='National Hospital'] ='National Hospital'

mfl.loc[:,'Facility_Name']=name

officer_types_table.to_csv(resourcefilepath + 'ResourceFile_MasterFacilitiesList.csv')

#  --------

# Create a simple mapping of all the facilities that persons in a district can access
facilities_by_district=pd.DataFrame(columns=mfl.columns)

for d in pop_districts:
    the_region = pop.loc[pop['District']==d,'Region'].copy().values[0]
    district_facs = mfl.loc[mfl['District']==d]

    region_fac = mfl.loc[ pd.isnull(mfl['District']) & (mfl['Region']==the_region) ].copy().reset_index(drop=True)
    region_fac.loc[0,'District']=d

    national_fac = mfl.loc[ pd.isnull(mfl['District']) & pd.isnull(mfl['Region']) ].copy().reset_index(drop=True)
    national_fac.loc[0,'District'] = d
    national_fac.loc[0, 'Region'] = the_region

    facilities_by_district = pd.concat([facilities_by_district , district_facs,region_fac, national_fac],ignore_index=True)

assert len(facilities_by_district) == len(pop_districts) * len(Facility_Types)


#  --------

# Create a mapping of how these different officer types will go to particular facility types
# The level is important as this determines how many other people are drawing on their time.

Facility_By_Officer=pd.DataFrame({'Officer_Type': officer_types_table.Officer_Type, 'Officer_Type_Code':officer_types_table.Officer_Type_Code ,'Facility_Type_Can_Work_In':None})

Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Medical Officer / Specialist','Facility_Type_Can_Work_In']={'District Hospital','Non-District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Clinical Officer / Technician','Facility_Type_Can_Work_In']={'District Hospital','Non-District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Med. Assistant','Facility_Type_Can_Work_In']={'District Hospital','Non-District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Nurse Officer','Facility_Type_Can_Work_In']={'District Hospital', 'Community Health Worker','Non-District Hospital', 'Health Centre', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Nurse Midwife Technician','Facility_Type_Can_Work_In']={'District Hospital', 'Non-District Hospital','Community Health Worker', 'Health Centre', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Pharmacist','Facility_Type_Can_Work_In']={'District Hospital', 'Health Centre', 'Non-District Hospital','Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Pharm Technician','Facility_Type_Can_Work_In']={'District Hospital', 'Health Centre', 'Non-District Hospital','Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Pharm Assistant','Facility_Type_Can_Work_In']={'District Hospital', 'Health Centre','Non-District Hospital','Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Lab Officer','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','Non-District Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Lab Technician','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','Non-District Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Lab Assistant','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','Non-District Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='DCSA','Facility_Type_Can_Work_In']={'Community Health Worker'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Dental Officer','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Dental Therapist','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Dental Assistant','Facility_Type_Can_Work_In']={'District Hospital','Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Mental Health Staff','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Nutrition Staff','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Radiographer','Facility_Type_Can_Work_In']={'National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Radiography Technician','Facility_Type_Can_Work_In']={'National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Sonographer','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Radiotherapy Technician','Facility_Type_Can_Work_In']={'National Hospital'}


# check that have only ever used correct Facility_Type labels
for i in np.arange(0, len(Facility_By_Officer)):
    x = Facility_By_Officer['Facility_Type_Can_Work_In'][i]
    assert x.issubset(set(Facility_Types))


# check that every Officer_Type has been covered
assert set(Facility_By_Officer['Officer_Type_Code']) == set(officer_types_table.Officer_Type_Code)


#  --------


# Create pd.Series for holding the assignments between staff member (in staff_list) and facility (in mfl)

facility_assignment =pd.Series(index=staff_list.index)

# Loop through each staff member and allocate them to
for staffmember in staff_list.index:

    officer =staff_list.loc[staffmember].Officer_Type_Code
    district= staff_list.loc[staffmember].District
    possible_fac_types= list((Facility_By_Officer.loc[Facility_By_Officer['Officer_Type_Code']==officer,'Facility_Type_Can_Work_In']).iloc[0])

    # Get the facilities to which this staff member might be allocated
    suitable_facilities=mfl.loc[ (mfl['Facility_Type'].isin(possible_fac_types) ) & ( mfl['District'] == district ) ]

    assert len(suitable_facilities)>0

    # if there is a suitable facility for this office to go into
    x=np.random.choice(len(suitable_facilities))
    assigned_facility=suitable_facilities.iloc[x].Facility_ID
    facility_assignment.at[staffmember]=assigned_facility




# ****BUT, HOW WILL STAFF GET ALLOCATED TO THE REFERRAL AND NATIONAL HOSPITALS ?

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

