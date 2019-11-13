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

# *** USE OF THE CENSUS DATA ****

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



#%% **** USE OF THE WPP DATA ****
#TODO: relabel the calendar periods to be the inclusive year range (2010-2014 instead of 2010-2015)
# wpp['t_lo'], wpp['t_hi']=wpp['Period'].str.split('-',1).str
# wpp['t_hi'] = wpp['t_hi'].astype(int) - 1
# wpp['period'] = wpp['t_lo'].astype(str) + '-' + wpp['t_hi'].astype(str)

#%% Population size: age groups
wpp_pop_males_file= '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_POP_F07_2_POPULATION_BY_AGE_MALE.xlsx'

wpp_pop_females_file= '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_POP_F07_3_POPULATION_BY_AGE_FEMALE.xlsx'


# Males
dat = pd.concat([
    pd.read_excel(wpp_pop_males_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(wpp_pop_males_file, sheet_name='LOW VARIANT', header=16),
    pd.read_excel(wpp_pop_males_file, sheet_name='MEDIUM VARIANT', header=16),
    pd.read_excel(wpp_pop_males_file, sheet_name='HIGH VARIANT', header=16)
], sort=False)

ests_males = dat.loc[dat[dat.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
ests_males['Sex']= 'M'


# Females
dat = pd.concat([
    pd.read_excel(wpp_pop_females_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(wpp_pop_females_file, sheet_name='LOW VARIANT', header=16),
    pd.read_excel(wpp_pop_females_file, sheet_name='MEDIUM VARIANT', header=16),
    pd.read_excel(wpp_pop_females_file, sheet_name='HIGH VARIANT', header=16)
])

ests_females = dat.loc[dat[dat.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
ests_females['Sex']= 'F'

# Join and tidy up
ests = pd.concat([ests_males,ests_females],sort=False)
ests = ests.drop(ests.columns[[0,2,3,4,5,6]],axis=1)
ests[ests.columns[2:23]] = ests[ests.columns[2:23]]*1000  # given numbers are in 1000's, so multiply by 1000 to give actual

ests.to_csv(resourcefilepath / 'ResourceFile_Pop_WPP.csv',index=False)

#%% Population size: single-year age/time steps
wpp_pop_males_file= '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_INT_F03_2_POPULATION_BY_AGE_ANNUAL_MALE.xlsx'

wpp_pop_females_file= '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_INT_F03_3_POPULATION_BY_AGE_ANNUAL_FEMALE.xlsx'


# Males
dat = pd.concat([
    pd.read_excel(wpp_pop_males_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(wpp_pop_males_file, sheet_name='MEDIUM VARIANT', header=16)
], sort=False)

ests_males = dat.loc[dat[dat.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
ests_males['Sex']= 'M'


# Females
dat = pd.concat([
    pd.read_excel(wpp_pop_females_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(wpp_pop_females_file, sheet_name='MEDIUM VARIANT', header=16)
])

ests_females = dat.loc[dat[dat.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
ests_females['Sex']= 'F'

# Join and tidy up
ests = pd.concat([ests_males,ests_females],sort=False)
ests = ests.drop(ests.columns[[0,2,3,4,5,6]],axis=1)
ests[ests.columns[2:23]] = ests[ests.columns[2:23]]*1000  # given numbers are in 1000's, so multiply by 1000 to give actual
ests = ests.rename(columns={ests.columns[1]:'Year'})
ests.to_csv(resourcefilepath / 'ResourceFile_Pop_Annual_WPP.csv',index=False)

#TODO: year 2020 is duplicated as medium and variant - remove. also remove 'variant' as not neededd.

#%% Fertility and births

tot_births_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_FERT_F01_BIRTHS_BOTH_SEXES.xlsx'

tot_births = pd.concat([
    pd.read_excel(tot_births_file,sheet_name='ESTIMATES',header=16),
    pd.read_excel(tot_births_file,sheet_name='LOW VARIANT',header=16),
    pd.read_excel(tot_births_file,sheet_name='MEDIUM VARIANT',header=16),
    pd.read_excel(tot_births_file,sheet_name='HIGH VARIANT',header=16)
], sort=False)

tot_births = tot_births.loc[tot_births[tot_births.columns[2]]=='Malawi'].copy().reset_index(drop=True)
tot_births = tot_births.drop(tot_births.columns[[0,2,3,4,5,6]],axis='columns')

tot_births = tot_births.melt(id_vars=['Variant'],var_name='Period',value_name='Total_Births').dropna()
tot_births['Total_Births'] = 1000 * tot_births['Total_Births']  # Imported units are 1000's

# Sex Ratio at Birth
sex_ratio_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_FERT_F02_SEX_RATIO_AT_BIRTH.xlsx'

sex_ratio = pd.concat([
    pd.read_excel(sex_ratio_file,sheet_name='ESTIMATES',header=16),
    pd.read_excel(sex_ratio_file,sheet_name='MEDIUM VARIANT',header=16)
], sort=False)

sex_ratio = sex_ratio.loc[sex_ratio[sex_ratio.columns[2]]=='Malawi'].copy().reset_index(drop=True)
sex_ratio  = sex_ratio .drop(sex_ratio .columns[[0,2,3,4,5,6]],axis='columns')
sex_ratio = sex_ratio.melt(id_vars=['Variant'],var_name='Period',value_name='M_to_F_Sex_Ratio').dropna()

# copy the medium variant sex ratio project for the low and high variants (in order to merge with the total births)
copy_high = sex_ratio.loc[sex_ratio['Variant']=='Medium variant',['Period','M_to_F_Sex_Ratio']].copy()
copy_high['Variant']='High variant'
sex_ratio = sex_ratio.append(copy_high, sort=False)

copy_low = sex_ratio.loc[sex_ratio['Variant']=='Medium variant',['Period','M_to_F_Sex_Ratio']].copy()
copy_low['Variant']='Low variant'
sex_ratio = sex_ratio.append(copy_low, sort=False)

# Combine these together
births = tot_births.merge(sex_ratio,on=['Variant','Period'],validate='1:1')

births.to_csv(resourcefilepath / 'ResourceFile_TotalBirths_WPP.csv',index=False)


# Age-specific Fertility Rates
asfr_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_FERT_F07_AGE_SPECIFIC_FERTILITY.xlsx'

asfr = pd.concat([
    pd.read_excel(asfr_file,sheet_name='ESTIMATES',header=16),
    pd.read_excel(asfr_file,sheet_name='LOW VARIANT',header=16),
    pd.read_excel(asfr_file,sheet_name='MEDIUM VARIANT',header=16),
    pd.read_excel(asfr_file,sheet_name='HIGH VARIANT',header=16)
], sort=False)


asfr = asfr.loc[asfr[asfr.columns[2]]=='Malawi'].copy().reset_index(drop=True)
asfr = asfr.drop(asfr.columns[[0,2,3,4,5,6]],axis='columns')
asfr[asfr.columns[2:9]] = asfr[asfr.columns[2:9]]/1000  # given numbers are per 1000, so divide by 1000 to make 'per woman'
asfr.to_csv(resourcefilepath / 'ResourceFile_ASFR_WPP.csv',index=False)


#%% Deaths

deaths_males_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_MORT_F04_2_DEATHS_BY_AGE_MALE.xlsx'

deaths_females_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_MORT_F04_3_DEATHS_BY_AGE_FEMALE.xlsx'

deaths_males = pd.concat([
    pd.read_excel(deaths_males_file,sheet_name='ESTIMATES',header=16),
    pd.read_excel(deaths_males_file,sheet_name='LOW VARIANT',header=16),
    pd.read_excel(deaths_males_file,sheet_name='MEDIUM VARIANT',header=16),
    pd.read_excel(deaths_males_file,sheet_name='HIGH VARIANT',header=16)
], sort=False)

deaths_males  = deaths_males.loc[deaths_males[deaths_males.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
deaths_males ['Sex']= 'M'

deaths_females = pd.concat([
    pd.read_excel(deaths_females_file,sheet_name='ESTIMATES',header=16),
    pd.read_excel(deaths_females_file,sheet_name='LOW VARIANT',header=16),
    pd.read_excel(deaths_females_file,sheet_name='MEDIUM VARIANT',header=16),
    pd.read_excel(deaths_females_file,sheet_name='HIGH VARIANT',header=16)
], sort=False)

deaths_females  = deaths_females.loc[deaths_females[deaths_females.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
deaths_females ['Sex']= 'F'


# Join and tidy up
deaths = pd.concat([deaths_males,deaths_females],sort=False)
deaths  = deaths .drop(deaths.columns[[0,2,3,4,5,6]],axis=1)
deaths[deaths.columns[2:22]] = deaths[deaths.columns[2:22]]*1000  # given numbers are in 1000's, so multiply by 1000 to give actual
deaths.to_csv(resourcefilepath / 'ResourceFile_TotalDeaths_WPP.csv',index=False)


# The ASMR from the LifeTable
lt_males_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_MORT_F17_2_ABRIDGED_LIFE_TABLE_MALE.xlsx'

lt_females_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_MORT_F17_3_ABRIDGED_LIFE_TABLE_FEMALE.xlsx'


lt_males = pd.concat([pd.read_excel(lt_males_file, sheet_name ='ESTIMATES', header=16, usecols ='B,C,H,I,J,K'),
                      pd.read_excel(lt_males_file, sheet_name ='MEDIUM 2020-2050', header=16, usecols ='B,C,H,I,J,K'),
                      pd.read_excel(lt_males_file, sheet_name ='MEDIUM 2050-2100', header=16, usecols ='B,C,H,I,J,K')
                      ])

lt_males = lt_males.loc[lt_males[lt_males.columns[1]] == 'Malawi'].copy().reset_index(drop=True)
lt_males['Sex'] = 'M'


lt_females = pd.concat([pd.read_excel(lt_females_file, sheet_name ='ESTIMATES', header=16, usecols ='B,C,H,I,J,K'),
                      pd.read_excel(lt_females_file, sheet_name ='MEDIUM 2020-2050', header=16, usecols ='B,C,H,I,J,K'),
                      pd.read_excel(lt_females_file, sheet_name ='MEDIUM 2050-2100', header=16, usecols ='B,C,H,I,J,K')
                      ])

lt_females = lt_females.loc[lt_females[lt_females.columns[1]] == 'Malawi'].copy().reset_index(drop=True)
lt_females['Sex'] = 'F'


# Join and tidy up
lt = pd.concat([lt_males,lt_females],sort=False)
lt = lt.drop(lt.columns[[1]],axis=1)
lt.loc[lt['Variant'].str.contains('Medium'),'Variant']='Medium'
lt.to_csv(resourcefilepath / 'ResourceFile_Pop_DeathRates_WPP.csv',index=False)

#%%
# *** USE OF THE GBD DATA ****
#%%

gbd_working_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/GBD/IHME-GBD_2017_DATA-1629962a-1/IHME-GBD_2017_DATA-1629962a-1.csv'

gbd = pd.read_csv(gbd_working_file)

gbd.to_csv(resourcefilepath / 'ResourceFile_Deaths_And_Causes_DeathRates_GBD.csv',index=False)


#%% *** DHS DATA

dhs_working_file =  '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/DHS/STATcompilerExport20191112_211640.xlsx'

dhs_asfr = pd.read_excel(dhs_working_file, sheet_name='ASFR')
dhs_asfr[dhs_asfr.columns[1:]] = dhs_asfr[dhs_asfr.columns[1:]] / 1000  # to make the ASFR per women
dhs_asfr.to_csv(resourcefilepath / 'ResourceFile_ASFR_DHS.csv', index=False)

dhs_u5 = pd.read_excel(dhs_working_file, sheet_name='UNDER_5_MORT',header=1,index=False)
dhs_u5['Year']=dhs_u5.index
dhs_u5 = dhs_u5.reset_index(drop=True)
dhs_u5 = dhs_u5[dhs_u5.columns[[3,0,1,2]]]
dhs_u5[dhs_u5.columns[1:]] = dhs_u5[dhs_u5.columns[1:]] / 1000  # to make it mortality risk per person
dhs_u5.to_csv(resourcefilepath / 'ResourceFile_Under_Five_Mortality_DHS.csv', index=False)
