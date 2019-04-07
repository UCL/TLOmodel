# This is a scratch file by Tim Hallett that will get the UNCIEF Master Facility Level data in the right format for importing


import pandas as pd
import numpy as np

workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/ORIGINAL_Master List Free Service Health Facilities in Malawi_no contact details.xlsx'
# outputpath='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Health System Resources/'

resourcefilepath='/Users/tbh03/PycharmProjects/TLOmodel/resources/'


wb=pd.read_excel(workingfile,sheet_name='All HF Malawi')

wb=wb.drop(columns='Date')

wb['Facility_ID']=np.arange(len(wb))
wb['Facility_ID']=wb['Facility_ID'].astype(int)

# Combine "Dispensaries" and "Health Centres" as "Health Centre"; Combine 'Health Post', 'Outreach', and 'Village Clinic' as "Community Health Worker" (= the Community Health Worker under a tree in each village)
wb['Facility_Type']=wb['Facility Type'].map({
            'Dispensary':'Health Centre',
            'Health Post': 'Community Health Worker',
            'Outreach': 'Community Health Worker',
            'Village Clinic': 'Community Health Worker',
            'Hospital': 'Non-District Hospital',
            'Health Centre': 'Health Centre'
        })
wb=wb.drop('Facility Type',axis=1)

# Label the referral Hospital as Referral Hospitals:
wb.loc[wb['Facility Name']=='QUEEN ELIZABETH','Facility_Type'] = 'Referral Hospital'
wb.loc[wb['Facility Name']=='KAMUZU CENTRAL HOSPITAL','Facility_Type'] = 'Referral Hospital'
wb.loc[wb['Facility Name']=='MZUZU CH','Facility_Type'] = 'Referral Hospital'


# 3) Label the district hopsitals
wb.loc[wb['Facility Name'].str.contains(' DH'),'Facility_Type']='District Hospital'

# look at number of district hospitals per distirct (check it's 1 per district)
wb.loc[wb['Facility_Type']=='District Hospital',['District','Facility_Type']].groupby(by=['District']).count()

# assign a level for each facility, based on the facility type
wb['Facility_Level'] = wb['Facility_Type'].map({
            'Community Health Worker':0,
            'Health Centre':1,
            'Non-District Hospital':2,
            'District Hospital':3,
            'Referral Hospital':4,
    })

assert not any(pd.isnull(wb['Facility_Level']))

# Clean up the village string
wb['Village']=wb['Village'].str.strip()

# Save output file for information about facilities
mfl=wb
mfl.to_csv(resourcefilepath+'ResourceFile_MasterFacilitiesList.csv')



#--------

# Make the file that maps the connections between villages and the health facilities.
# Each row gives one connection betweeen a village and a facilities that is attached to it.
# There are multiple row per village and per facility: one row per connection.
# When used we will .loc onto this to find (CHW (Community Health Worker, Near-Hospital (Nearest Hospital),District Hospital,Referral Hospital)
# We guarantee that each village has is attaching to at least oen facility of each level.


# 1) Get the complete listing of villages:

pop=pd.read_csv(resourcefilepath+'ResourceFile_PopBreakdownByVillage.csv')

villages_pop = pop.Village

# Our listing of villages
villages=wb['Village'].unique()
villages=villages[~pd.isnull(villages)] # take out the nans

# **** Get the listing of CHW per village
df_CHW=wb.loc[wb['Facility Type']=='Community Health Worker',['Village','Facility Type','Facility_ID']]
df_CHW.groupby(by='Village').count()


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
df_DistrictHospital['Village']=villages
df_DistrictHospital['Facility Type']='District Hospital'
district_hosps=wb.loc[ (wb['Facility Type']=='District Hospital'), ['District','Facility_ID']]

for v in villages:
    the_district=wb.loc[wb['Village']==v,'District'].values[0]
    the_district_hosp= district_hosps.loc[district_hosps['District']==the_district,'Facility_ID']

    if len(the_district_hosp)>0: # some district do not have district hospital (e.g. Likoma)
        df_DistrictHospital.loc[df_DistrictHospital['Village']==v,'Facility_ID']=the_district_hosp.values[0]

# delete rows for places without district hospitals
df_DistrictHospital=df_DistrictHospital.dropna()


# **** Add in linkage to referral hosital
df_ReferralHospital=pd.DataFrame(columns={'Village','Facility Type','Facility_ID'})
df_ReferralHospital['Village']=villages
df_ReferralHospital['Facility Type']='Referral Hospital'
# **** Concatenate all the sets

for v in villages:
        # Look-up region
        region = wb.loc[wb['Village'] == v, 'Region']
        region = region.iloc[0].strip()

        if region=='South':
            ref_hosp_facility_id =wb.loc[wb['Facility Name']=='QUEEN ELIZABETH'].Facility_ID.values[0]
        else:
            ref_hosp_facility_id= wb.loc[wb['Facility Name'] == 'KAMUZU CENTRAL HOSPITAL'].Facility_ID.values[0]

        df_ReferralHospital.loc[df_ReferralHospital['Village']==v,'Facility_ID']=ref_hosp_facility_id


# **** Concatenate the dataframes together

x=df_CHW.append([df_NearHospital,df_DistrictHospital,df_ReferralHospital],ignore_index=True)


# *** Merge back in the names of the faciliites and their other information

fac_details=wb[['Facility_ID','Facility Name','Eastings','Northings']]
fac_details.Facility_ID=fac_details.Facility_ID.astype(np.int64) # coerce the typing ready for the merge
x.Facility_ID=x.Facility_ID.astype(np.int64)

y=x.merge(fac_details,how='left',on='Facility_ID')


# **** Save:
y.to_csv(outputpath + 'ResourceFile_Village_To_Facility_Mapping.csv')




