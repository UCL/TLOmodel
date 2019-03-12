# This is a scratch file by Tim Hallett that will get the UNCIEF Master Facility Level data in the right format for importing


import pandas as pd
import numpy as np

workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Health System Resources/ORIGINAL_Master List Free Service Health Facilities in Malawi_no contact details.xlsx'
outputfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Health System Resources/ResourceFile_MasterFacilitiesList.csv'

wb=pd.read_excel(workingfile,sheet_name='All HF Malawi')

wb=wb.drop(columns='Date')

wb['Facility_ID']=np.arange(len(wb))

# Save output file for information about facilities
wb.to_csv(outputfile)

#--------

# Make the file that maps the connections between villages and the health facilities
# This will have one row per village...
#                               ... and columns of (Community Health Worker,Nearest Hospital,District Hospital,Referral Hospital)






# Reduce the number of types of facilities

# 1) Combine "Dispensaries" and "Health Centres"; Combine 'Health Post', 'Outreach', and 'Village Clinic' as "Community Health Worker" (= the Community Health Worker under a tree in each village)
wb['Facility Type']=wb['Facility Type'].map({
            'Dispensary':'Health Centre',
            'Health Post': 'Community Health Worker',
            'Outreach': 'Community Health Worker',
            'Village Clinic': 'Community Health Worker',
            'Hospital': 'Hospital',
            'Health Centre': 'Health Centre'
        })

# (now there are only three type: hospital, community health workers, health centre)
wb.groupby(by='Facility Type').count()


# 2) Label the referral Hospital as Refrral Hospitals:
wb.loc[wb['Facility Name']=='QUEEN ELIZABETH','Facility Type'] = 'Referral Hospital'
wb.loc[wb['Facility Name']=='KAMUZU CENTRAL HOSPITAL','Facility Type'] = 'Referral Hospital'


# 3) Label the district hopsitals

wb.loc[wb['Facility Name'].str.contains(' DH'),'Facility Type']='District Hospital'

# look at number of district hospitals per distirct (check it's 1 per district)
wb.loc[wb['Facility Type']=='District Hospital',['District','Facility Type']].groupby(by=['District']).count()






# 3) Add in linkage to district hospitals; nearest (one) hospital only

villages=wb['Village'].unique()
hospitals=wb.loc[wb['Facility Type']=='Hospital']

# dataframe that will hold these additional linkages
df=pd.DataFrame(data={'Village':villages},columns=wb.columns)

v=villages[2] # for a particular village.....

# take the average location of all facility types in village
vill_x=np.mean(wb.loc[(wb['Village']==v) & (wb['Facility Type']!='Referral Hospital'),'Eastings'])
vill_y=np.mean(wb.loc[(wb['Village']==v) & (wb['Facility Type']!='Referral Hospital'),'Northings'])

hosp_x=hospitals['Eastings']
hosp_y=hospitals['Northings']

dist_sq= np.power(hosp_x-vill_x,2) + np.power(hosp_y-vill_y,2)

record=hospitals.iloc[dist_sq.idxmin()]
record.loc[record.index, 'Village'] = v

df.loc[df['Village']==v]=record.values




# 3) Attach Referral Hospitals to each village:

# make a dataframe that will hold the new records
villages=wb['Village'].unique()
df=pd.DataFrame(data={'Village':villages},columns=wb.columns)


for v in villages:

    if not pd.isnull(v):
        #new record for each village attaching to a referral hospital:
        region = wb.loc[wb['Village'] == v, 'Region']
        region = region.iloc[0].strip()

        if region=='South':
            record=wb.loc[wb['Facility Name']=='QUEEN ELIZABETH']
        else:
            record = wb.loc[wb['Facility Name'] == 'KAMUZU CENTRAL HOSPITAL']

        record.loc[record.index, 'Village'] = v
        df.loc[df['Village']==v]=record.values



wb=wb.append(df)





tas=wb.loc[wb['Facility Type']=='Hospital','Facility Name'].unique



wb.loc[wb['Facility Type']=='Hospital','Facility Name'].unique

# 4) Check to see how breakdown of facilities look by village;

wb.loc[wb['Village']==villages[0]]


# Clean out facilities which do not attach to a village
wb[pd.isnull(wb['Village'])]


# Clean out columns that are not useful to useful for the mapping








#-------------------------------------------------------------------------

# Add that every village is also attaching to a Referral Hospital based on regio

# Add that every village is also attching to all the hospitals in the district


# Add it that there is a central hospital at the top of the hierarchy for all villages

# experiment with the data:
# reduce it to one villge


myvillage='Iyela'

df=wb.loc[wb['Village']==myvillage]


