"""
This is a file written by Tim Hallett to process the data from the Malawi 2018 Census that is downloaded into a
form that a useful for TLO Model.
It creates:
* ResourceFile_PopulationSize
* ResourceFile_Mortality
* ResourceFile_Births

"""
from pathlib import Path

import pandas as pd

workingfile = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/Census_Main_Report/Series A. Population Tables.xlsx'

resourcefilepath = Path("./resources")


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
a1.index = [name.strip() for name in list(a1.index)]

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
district_names = list(a1.index)

#%%
# extract the age-breakdown for each district by %

a7 = pd.read_excel(workingfile, sheet_name='A7',usecols=[0]+list(range(2,10))+list(range(12,21)),header=1,index_col=0)

# There is a typo in the table: correct manually
a7.loc['TA Kameme','10-14'] = None

a7= a7.dropna()

a7 = a7.astype('int')

# do some renaming to get a result for each district in the master list
a7.rename(index={'Blantyre Rural':'Blantyre'},inplace=True)
a7.rename(index={'Lilongwe Rural':'Lilongwe'},inplace=True)
a7.rename(index={'Zomba Rural':'Zomba'},inplace=True)
a7.rename(index={'Nkhatabay':'Nkhata Bay'},inplace=True)

# extract results for districts
extract = a7.loc[a7.index.isin(district_names)].copy()

# checks
assert len(extract) == len(district_names)
assert 0 == len( set(district_names)  - set(extract.index))
assert extract.sum(axis=1).astype(int).eq(a1['Total_2018']).all()

# Compute fraction of population in each age-group
frac_in_each_age_grp = extract.div(extract.sum(axis=1),axis=0)
assert (frac_in_each_age_grp.sum(axis=1).astype('float32')==(1.0)).all()

#%% Get district-specific age/sex breakdowns
# Use the district-specific age breakdown and district-specific sex breakdown to create district/age/sex breakdown
# (Assuming that the age-breakdown is the same for men and women)

males = frac_in_each_age_grp.mul(a1['Male_2018'],axis=0)
assert (males.sum(axis=1).astype('float32') == a1['Male_2018'].astype('float32')).all()
males['district'] = males.index
males_melt = males.melt(id_vars='district',var_name='age_grp',value_name='number')
males_melt['sex'] = 'M'

females = frac_in_each_age_grp.mul(a1['Female_2018'],axis=0)
assert (females.sum(axis=1).astype('float32') == a1['Female_2018'].astype('float32')).all()
females['district'] = females.index
females_melt = females.melt(id_vars='district',var_name='age_grp',value_name='number')
females_melt['sex'] = 'F'

# Melt into long-format and save
table = pd.concat([males_melt,females_melt])

table['number'] = table['number'].astype(float)
table = table[table.columns[[0, 1, 3, 2]]]

table.to_csv(resourcefilepath / 'ResourceFile_PopulationSize.csv')
