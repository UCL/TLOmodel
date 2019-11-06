"""
This is a file written by Tim Hallett to process the data from the Malawi 2018 Census that is downloaded into a
form that a useful for TLO Model.
It creates:
* ResourceFile_Population
* ResourceFile_Mortality
* ResourceFile_Births

"""


import pandas as pd






workingfile = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/Census_Main_Report/Series A. Population Tables.xlsx'


#%% Totals by Sex for Each District

# Clean up the data that is imported
a1 = pd.read_excel(workingfile, sheet_name='A1')
a1 = a1.drop([0,1])
a1 = a1.drop(a1.index[0])
a1.index = a1.iloc[:,0]
a1 = a1.drop(a1.columns[[0]], axis=1)
column_titles =['Total_2018','Male_2018','Female_2018','Total_2008','Male_2008','Female_2008']
a1.columns =column_titles
a1= a1.dropna(axis=0)

# organise regional and national totals nicely
region_names = ['Northern','Central','Southern']

region_totals = a1.loc[region_names].copy()
national_total =  a1.loc['Malawi'].copy()

a1= a1.drop(a1.index[0])
a1['Region'] = None
a1.loc[region_names,'Region']=region_names
a1['Region'] = a1['Region'].ffill()
a1 = a1.drop(region_names)

# Check that the everything is ok:
    # Sum of districts = Total for Nation
    assert a1.drop(columns='Region').sum().astype(int).equals(national_total.astype(int))

    # Sum of district by region = Total for regionals reported.
    assert a1.groupby('Region').sum().astype(int).eq(region_totals.astype(int)).all().all()

# Get definitive list of district:
district_names = [name.strip() for name in list(a1.index)]

#%%
# extract the age-breakdown for each district


wb = pd.read_excel(workingfile, sheet_name='A7',header=1,index_col=0,usecols=[0]+list(range(2,11)))








