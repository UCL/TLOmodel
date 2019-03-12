# This is a scratch file by Tim Hallett that will get the UNCIEF Master Facility Level data in the right format for importing


import pandas as pd
import numpy as np

workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Health System Resources/ORIGINAL_Master List Free Service Health Facilities in Malawi_no contact details.xlsx'
outputfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Health System Resources/ResourceFile_MasterFacilitiesList.csv'

wb=pd.read_excel(workingfile,sheet_name='All HF Malawi')

wb=wb.drop(columns='Date')

wb['Facility_ID']=np.arange(len(wb))

# Combine "Dispensaries" and "Health Centres"; Combine 'Health Post', 'Outreach', and 'Village Clinic' as "Community Health Worker" (= the Community Health Worker under a tree in each village)
wb['Facility Type']=wb['Facility Type'].map({
            'Dispensary':'Health Centre',
            'Health Post': 'Community Health Worker',
            'Outreach': 'Community Health Worker',
            'Village Clinic': 'Community Health Worker',
            'Hospital': 'Hospital',
            'Health Centre': 'Health Centre'
        })

# Label the referral Hospital as Refrral Hospitals:
wb.loc[wb['Facility Name']=='QUEEN ELIZABETH','Facility Type'] = 'Referral Hospital'
wb.loc[wb['Facility Name']=='KAMUZU CENTRAL HOSPITAL','Facility Type'] = 'Referral Hospital'

# 3) Label the district hopsitals
wb.loc[wb['Facility Name'].str.contains(' DH'),'Facility Type']='District Hospital'

# look at number of district hospitals per distirct (check it's 1 per district)
wb.loc[wb['Facility Type']=='District Hospital',['District','Facility Type']].groupby(by=['District']).count()

# Save output file for information about facilities
wb.to_csv(outputfile)

#--------

# Make the file that maps the connections between villages and the health facilities
# When used we will .loc onto this to find (CHW (Community Health Worker, Near-Hospital (Nearest Hospital),District Hospital,Referral Hospital)

villages=wb['Village'].unique()
villages=villages[~pd.isnull(villages)] # take out the nans

# **** Get the listing of CHW per village
df_CHW=wb.loc[wb['Facility Type']=='Community Health Worker',['Village','Facility Type','Facility_ID']]
df_CHW.groupby(by='Village').count()

#TODO: Got this to this point
# **** Attach the nearest hospital to each village (can be the distrct hospital but not the referral hospital)
df_NearHospital=pd.DataFrame(columns={'Village','Facility Type','Facility_ID'})

hospitals=wb.loc[ (wb['Facility Type']=='Hospital') & (wb['Facility Type']!='Referral Hospital') ]

for v in villages:
    # take the average location of all facility types in village
    vill_x=np.mean( wb.loc[wb['Village']==v,'Eastings'] )
    vill_y=np.mean( wb.loc[wb['Village']==v,'Northings'] )

    dist_sq= np.power(hospitals['Eastings']-vill_x,2) + np.power(hospitals['Northings']-vill_y,2)

    nearest_hosp_id = hospitals.loc[dist_sq.idxmin(),'Facility_ID']

    df_NearHospital=df_NearHospital.append(  {'Village':v, 'Facility Type':'Near Hospital','Facility_ID': nearest_hosp_id }, ignore_index=True)



# **** Add in linkage to district hospitals and the nearest hospital
df_DistrictHospital=pd.DataFrame(columns={'Village','Facility Type','Facility_ID'})

district_hosps=wb.loc[ (wb['Facility Type']=='District Hospital') ]
districts=wb['District'].unique()

for d in districts:
    the_district_hospital_idx = wb.loc[(wb['District'] == d) & (wb['Facility Type'] == 'District Hospital')].Facility_ID.values[0]
    print(the_district_hospital_idx)



# **** Add in linkage to referral hosital



# **** Concatenate all the sets



#TODO: Alter how the healthsystem read in the two files and uses them


#
#
# # 3) Attach Referral Hospitals to each village:
#
# # make a dataframe that will hold the new records
# villages=wb['Village'].unique()
# df=pd.DataFrame(data={'Village':villages},columns=wb.columns)
#
#
# for v in villages:
#
#     if not pd.isnull(v):
#         #new record for each village attaching to a referral hospital:
#         region = wb.loc[wb['Village'] == v, 'Region']
#         region = region.iloc[0].strip()
#
#         if region=='South':
#             record=wb.loc[wb['Facility Name']=='QUEEN ELIZABETH']
#         else:
#             record = wb.loc[wb['Facility Name'] == 'KAMUZU CENTRAL HOSPITAL']
#
#         record.loc[record.index, 'Village'] = v
#         df.loc[df['Village']==v]=record.values
#
#
#
# wb=wb.append(df)
#
#
#
#
#
# tas=wb.loc[wb['Facility Type']=='Hospital','Facility Name'].unique
#
#
#
# wb.loc[wb['Facility Type']=='Hospital','Facility Name'].unique
#
# # 4) Check to see how breakdown of facilities look by village;
#
# wb.loc[wb['Village']==villages[0]]
#
#
# # Clean out facilities which do not attach to a village
# wb[pd.isnull(wb['Village'])]
#
#
# # Clean out columns that are not useful to useful for the mapping
#
#
#
#
#
#
#
#
# #-------------------------------------------------------------------------
#
# # Add that every village is also attaching to a Referral Hospital based on regio
#
# # Add that every village is also attching to all the hospitals in the district
#
#
# # Add it that there is a central hospital at the top of the hierarchy for all villages
#
# # experiment with the data:
# # reduce it to one villge
#
#
# myvillage='Iyela'
#
# df=wb.loc[wb['Village']==myvillage]
#
#
