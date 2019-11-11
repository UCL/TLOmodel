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

resourcefilepath = Path("./resources")



#%% Totals by Sex for Each District

workingfile_popsizes = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/Census_Main_Report/Series A. Population Tables.xlsx'

# Clean up the data that is imported
a1 = pd.read_excel(workingfile_popsizes, sheet_name='A1')
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

a7 = pd.read_excel(workingfile_popsizes, sheet_name='A7', usecols=[0] + list(range(2, 10)) + list(range(12, 21)), header=1, index_col=0)

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

#%% Compute  district-specific age/sex breakdowns
# Use the district-specific age breakdown and district-specific sex breakdown to create district/age/sex breakdown
# (Assuming that the age-breakdown is the same for men and women)

males = frac_in_each_age_grp.mul(a1['Male_2018'],axis=0)
assert (males.sum(axis=1).astype('float32') == a1['Male_2018'].astype('float32')).all()
males['district'] = males.index
males = males.merge(a1[['Region']],left_index=True,right_index=True,validate='1:1')
males_melt = males.melt(id_vars=['district','Region'],var_name='age_grp',value_name='number')
males_melt['sex'] = 'M'

females = frac_in_each_age_grp.mul(a1['Female_2018'],axis=0)
assert (females.sum(axis=1).astype('float32') == a1['Female_2018'].astype('float32')).all()
females['district'] = females.index
females = females.merge(a1[['Region']],left_index=True,right_index=True,validate='1:1')
females_melt = females.melt(id_vars=['district','Region'],var_name='age_grp',value_name='number')
females_melt['sex'] = 'F'

# Melt into long-format and save
table = pd.concat([males_melt,females_melt])

table['number'] = table['number'].astype(float)
table = table.rename(columns={'Region':'region'})
table = table[table.columns[[0, 1, 4, 2, 3]]]

table.to_csv(resourcefilepath / 'ResourceFile_PopulationSize_2018Census.csv',index=False)


#%% Number of births

workingfile_fertility = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/Census_Main_Report/Series B. Fertility Tables.xlsx'


b1 = pd.read_excel(workingfile_fertility, sheet_name='TABLE B1')
b1 = b1.dropna()
b1 = b1.rename(columns={b1.columns[0]:'Age/Region',b1.columns[1]:'Num_Women_1549',b1.columns[2]:'Live_Births',b1.columns[3]:'Babies_Live_12m',b1.columns[4]:'Babies_Dead_12m'})
b1 = b1.drop(list(range(2,10)),axis=0)
b1['region']=None
region_labels = [r + ' Region' for r in region_names]
b1.loc[b1['Age/Region'].isin(region_labels),'region']=b1['Age/Region']
b1['region']=b1['region'].ffill()
b1 = b1.drop(b1.index[b1['Age/Region'].isin(region_labels)],axis=0)
b1 = b1.rename(columns={'Age/Region':'age_grp'})
b1 = b1[b1.columns[[0, 5, 1, 2, 3, 4]]]

# take the word 'region' out of the values in the 'region' column
b1['region'] = b1['region'].str.replace(' Region','')

# check that number of women 15-49 by region is close to the estimate for population size
from_fert_data = b1.groupby('region')['Num_Women_1549'].sum()
from_pop_data = table.loc[(table['sex']=='F') & (table['age_grp'].isin(['15-19','20-24','25-29','30-34','35-39','40-44','45-49']))].groupby('region')['number'].sum()
diff = 100* (from_fert_data - from_pop_data) / from_fert_data

# save the file:
b1.to_csv(resourcefilepath / 'ResourceFile_Births_2018Census.csv',index=False)

#%% Number of deaths

workingfile_mortality = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/Census_Main_Report/Series K. Mortality Tables.xlsx'

k2 = pd.read_excel(workingfile_mortality,sheet_name = 'K2')

k2 = k2.dropna()
k2.columns = ['Age/Region','Pop_Total','Pop_Males','Pop_Females','Deaths_Total','Deaths_Males','Deaths_Females']

k2 = k2.drop(list(range(2,24)),axis=0)

k2['region']=None
k2.loc[k2['Age/Region'].isin(region_names),'region']=k2['Age/Region']
k2['region']=k2['region'].ffill()
k2 = k2.drop(k2.index[k2['Age/Region'].isin(region_names)],axis=0)
k2 = k2.rename(columns={'Age/Region':'age_grp'})

k2 = k2[k2.columns[[7,0,1,2,3,4,5,6]]]

# save the file:
k2.to_csv(resourcefilepath / 'ResourceFile_Deaths_2018Census.csv',index=False)

