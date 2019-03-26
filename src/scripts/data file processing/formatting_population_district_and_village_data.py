"""
This is a scratch file by Tim Hallett
Purpose is to create a nice file for import that gives the population breakdown by region/district/village
Hopefully one day such a file will be provided to us, but for now we make the following assumptions:
* 1) The region and district breakdown is taken from the preliminary report (Dec 2018) of new census
* 2) The complete list of villages (and the district they belong to) is from the UNICEF MasterFacility list file
* 3) The allocation of population to villages assumes that the villages of are equal sizes.

NB. There are some issues with the merge, but this will be resolved new data given population sizes by village
"""

import pandas as pd

workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Demographic data/village-district-breakdown/Census Data and Health System Data.xlsx'
outputfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Demographic data/village-district-breakdown/ResourceFile_PopBreakdownByVillage.csv'

# Load census data on pop sizes in each distrct
district_wb=pd.read_excel(workingfile,sheet_name='PopBreakdownByDistrict_Census')

# need to tidy up what's come in (take of the "xao"-tails from the entries and ensure numbers are numbers!
district_wb.District=district_wb.District.str.strip()
district_wb.Region=district_wb.Region.str.strip()
district_wb.Total=district_wb.Total.str.strip()
district_wb.Men=district_wb.Men.str.strip()
district_wb.Women=district_wb.Women.str.strip()
district_wb.Total=district_wb.Total.str.replace(',','')
district_wb.Men=district_wb.Men.str.replace(',','')
district_wb.Women=district_wb.Women.str.replace(',','')
district_wb.District = district_wb.District.astype(str)
district_wb.Region = district_wb.Region.astype(str)
district_wb.Total = district_wb.Total.astype(float)
district_wb.Men = district_wb.Men.astype(float)
district_wb.Women = district_wb.Women.astype(float)

assert (~pd.isnull(district_wb).any()).all() # check for any null values

# Trim down the sheet to the basic for the merge
district_wb_trimmed=district_wb.drop(['Region','Men','Women'],axis=1)
district_wb_trimmed=district_wb_trimmed.rename(columns={'Total':'District Total'})


# Load listing of villages (Which we get from the UNICEF file
# (Note that the UNICEF file includes duplicates because it it listing all health facilities)
villages_wb=pd.read_excel(workingfile,sheet_name='Listing of villages from MFL')
villages_wb=villages_wb.drop('TA',axis=1)
villages_wb.Village=villages_wb.Village.str.strip()
villages_wb.District=villages_wb.District.str.strip()
villages_wb.Region=villages_wb.Region.str.strip()
villages_wb.Village = villages_wb.Village.astype(str)
villages_wb.District = villages_wb.District.astype(str)
villages_wb.Region = villages_wb.Region.astype(str)

# drop duplicated villages
villages_wb.drop_duplicates(keep='first',inplace=True)

# drop villages with a name of 'nan' (as a string not as a real pandas null value):
villages_wb.drop[villages_wb['Village']=='nan']

villages_wb.dr

# check for no null valuyes
assert not pd.isnull(villages_wb['Village']).any()
assert not pd.isnull(villages_wb['District']).any()
assert not pd.isnull(villages_wb['Region']).any()


# Coerce District list in the villages_wb to match that provided in the censsus data
len(villages_wb.District.unique())
len(district_wb_trimmed.District.unique())

set(villages_wb.District.unique()) - set(district_wb_trimmed.District.unique())
# join the "Mzimba South" and "Mzimba South in the Health System Dataset
villages_wb.loc[villages_wb['District']=='Mzimba North','District']='Mzimba'
villages_wb.loc[villages_wb['District']=='Mzimba South','District']='Mzimba'

set(district_wb_trimmed.District.unique()) - set(villages_wb.District.unique())

# rename Blantyre City --> Blantyre
district_wb_trimmed.loc[district_wb_trimmed['District']=='Blantyre City','District']='Blantyre'

# rename Zomba City --> Zomba
district_wb_trimmed.loc[district_wb_trimmed['District']=='Zomba City','District']='Zomba'

# Join Lilongwe City and Lilongwe in the census dataset
num_ppl_in_Lilongwe_total=district_wb_trimmed.loc[district_wb_trimmed['District']=='Lilongwe','District Total'].values + district_wb_trimmed.loc[district_wb_trimmed['District']=='Lilongwe City','District Total'].values
district_wb_trimmed=district_wb_trimmed.drop(district_wb_trimmed.index[district_wb_trimmed['District']=='Lilongwe City'])

# "Mzuzu" is identified in the census data but NOT the UNICEF health-facility data
# For now, just add those people from Mzuzu into Lilongwe and remove Mzuzu
# TODO: Resolve this better
num_ppl_in_Mzuzu=district_wb_trimmed.loc[district_wb_trimmed['District']=='Mzuzu City','District Total'].values
district_wb_trimmed.loc[district_wb_trimmed['District']=='Lilongwe','District Total']=num_ppl_in_Lilongwe_total+num_ppl_in_Mzuzu
district_wb_trimmed=district_wb_trimmed.drop(district_wb_trimmed.index[district_wb_trimmed['District']=='Mzuzu City'])

len(villages_wb.District.unique())
len(district_wb_trimmed.District.unique())


# merge in the disrict total population size
joined=villages_wb.merge(district_wb_trimmed,how='left',on='District')

# now divide the district total into equal parts for each village and add that into dataframe
num_villages_per_district=joined.groupby(['District'],as_index=False).count()
num_villages_per_district=num_villages_per_district.rename(columns={'Village':'NumVillagesPerDistrict'})
num_villages_per_district=num_villages_per_district.drop(['Region','District Total'],axis=1)

joined=joined.merge(num_villages_per_district,how='left',on='District')

joined['Village Total']=joined['District Total'] / joined['NumVillagesPerDistrict']

PopBreakdownByVillage=joined.drop(['District Total','NumVillagesPerDistrict'],axis=1)
PopBreakdownByVillage=PopBreakdownByVillage.rename(columns={'Village Total':'Population'})
PopBreakdownByVillage.to_csv(outputfile)

# checks
PopBreakdownByVillage['Population'].sum()
district_wb_trimmed['District Total'].sum()


