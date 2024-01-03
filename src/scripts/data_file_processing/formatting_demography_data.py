"""
This is a script to process the data from the Malawi 2018 Census, WPP 2019 and DHS to create the ResourceFiles used for
model running and calibration checks.

It reads in the files that were downloaded externally and saves them as ResourceFiles in the `resources` directory:
    resources/demography/

The following files are created:
* 'ResourceFile_Population_2010.csv': used in model
* 'ResourceFile_Pop_Frac_Births_Male.csv': used in model
* 'ResourceFile_Pop_DeathRates_Expanded_WPP.csv': used in model
* `ResourceFile_ASFR_WPP.csv`: used in model (SimplifiedBirths module)

* `ResourceFile_PopulationSize_2018Census.csv`: used for scaling results to actual size of population in census
* `ResourceFile_Pop_Annual_WPP.csv`: used for calibration checks
* `ResourceFile_TotalBirths_WPP.csv`: used for calibration checks
* `ResourceFile_TotalDeaths_WPP.csv`: used for calibration checks

* 'ResourceFile_Birth_2018Census.csv': Not used currently
* 'ResourceFile_Deaths_2018Census.csv': Not used currently

* `ResourceFile_Pop_DeathRates_WPP.csv`: Not used currently
* `ResourceFile_Pop_WPP.csv`: Not used currently
* `ResourceFile_ASFR_DHS.csv`: Not used currently
* `ResourceFile_Under_Five_Mortality_DHS.csv`: Not used currently

"""

from pathlib import Path

import numpy as np
import pandas as pd

from tlo.analysis.utils import make_calendar_period_lookup
from tlo.util import create_age_range_lookup

path_for_saved_files = Path("./resources/demography")

(__tmp__, calendar_period_lookup) = make_calendar_period_lookup()

# *** USE OF THE CENSUS DATA ****

# %% Totals by Sex for Each District

workingfile_popsizes = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/Census_Main_Report/Series A. Population Tables.xlsx'

# Clean up the data that is imported
a1 = pd.read_excel(workingfile_popsizes, sheet_name='A1')
a1 = a1.drop([0, 1])
a1 = a1.drop(a1.index[0])
a1.index = a1.iloc[:, 0]
a1 = a1.drop(a1.columns[[0]], axis=1)
column_titles = ['Total_2018', 'Male_2018', 'Female_2018', 'Total_2008', 'Male_2008', 'Female_2008']
a1 = a1.dropna(how='all', axis=1)
a1.columns = column_titles
a1 = a1.dropna(axis=0)
a1.index = [name.strip() for name in list(a1.index)]

# organise regional and national totals nicely
region_names = ['Northern', 'Central', 'Southern']

region_totals = a1.loc[region_names].copy()
national_total = a1.loc['Malawi'].copy()

a1 = a1.drop(a1.index[0])
a1['Region'] = None
a1.loc[region_names, 'Region'] = region_names
a1['Region'] = a1['Region'].ffill()
a1 = a1.drop(region_names)

# Check that everything is ok:
# Sum of districts = Total for Nation
assert a1.drop(columns='Region').sum().astype(int).equals(national_total.astype(int))

# Sum of district by region = Total for regionals reported.
assert a1.groupby('Region').sum().astype(int).eq(region_totals.astype(int)).all().all()

# Get definitive list of district.
district_names = list(a1.index)

# Make table of District, District Number and Region
district_nums = pd.DataFrame(data={
    'Region': a1['Region'],
    'District_Num': np.arange(len(a1['Region']))})

# %%

# extract the age-breakdown for each district by %

a7 = pd.read_excel(workingfile_popsizes, sheet_name='A7', usecols=[0] + list(range(2, 10)) + list(range(12, 21)),
                   header=1, index_col=0)

# There is a typo in the table: correct manually
a7.loc['TA Kameme', '10-14'] = None

a7 = a7.dropna()

a7 = a7.astype('int')

# do some renaming to get a result for each district in the master list
a7.rename(index={'Blantyre Rural': 'Blantyre'}, inplace=True)
a7.rename(index={'Lilongwe Rural': 'Lilongwe'}, inplace=True)
a7.rename(index={'Zomba Rural': 'Zomba'}, inplace=True)
a7.rename(index={'Nkhatabay': 'Nkhata Bay'}, inplace=True)

# extract results for districts
extract = a7.loc[a7.index.isin(district_nums.index)].copy()

# checks
assert set(extract.index) == set(district_nums.index)
assert len(extract) == len(district_names)
assert extract.sum(axis=1).astype(int).eq(a1['Total_2018']).all()

# Compute fraction of population in each age-group
frac_in_each_age_grp = extract.div(extract.sum(axis=1), axis=0)
assert (frac_in_each_age_grp.sum(axis=1).astype('float32') == (1.0)).all()

# Compute  district-specific age/sex breakdowns
# Use the district-specific age breakdown and district-specific sex breakdown to create district/age/sex breakdown
# (Assuming that the age-breakdown is the same for men and women)

males = frac_in_each_age_grp.mul(a1['Male_2018'], axis=0)
assert (males.sum(axis=1).astype('float32') == a1['Male_2018'].astype('float32')).all()
males['district'] = males.index
males_melt = males.melt(id_vars=['district'], var_name='age_grp', value_name='number')
males_melt['sex'] = 'M'
males_melt = males_melt.merge(a1[['Region']], left_on='district', right_index=True)

females = frac_in_each_age_grp.mul(a1['Female_2018'], axis=0)
assert (females.sum(axis=1).astype('float32') == a1['Female_2018'].astype('float32')).all()
females['district'] = females.index
females_melt = females.melt(id_vars=['district'], var_name='age_grp', value_name='number')
females_melt['sex'] = 'F'
females_melt = females_melt.merge(a1[['Region']], left_on='district', right_index=True)

# Melt into long-format
table = pd.concat([males_melt, females_melt])
table['number'] = table['number'].astype(float)
table.rename(columns={'district': 'District', 'age_grp': 'Age_Grp_Special', 'sex': 'Sex', 'number': 'Count'},
             inplace=True)
table['Age_Grp_Special'] = table['Age_Grp_Special'].replace({'Less than 1 Year': '0-1'})
table['Variant'] = 'Census_2018'
table['Year'] = 2018
table['Period'] = table['Year'].map(calendar_period_lookup)

# Collapse the 0-1 and 1-4 age-groups into 0-4
table['Age_Grp'] = table['Age_Grp_Special'].replace({'0-1': '0-4', '1-4': '0-4'})
table['Count_By_Age_Grp'] = table.groupby(by=['Age_Grp', 'District', 'Sex'])['Count'].transform('sum')
table = table.drop_duplicates(subset=['Age_Grp', 'District', 'Sex'])
table = table.rename(columns={'Count_By_Age_Grp': 'Count', 'Count': 'Count_By_Age_Grp_Special'})

# Merge in District_Num
table = table.merge(district_nums[['District_Num']], left_on=['District'], right_index=True, how='left')
assert 0 == len(set(district_nums['District_Num']).difference(set(pd.unique(table['District_Num']))))

# Re-order columns
table = table[[
    'Variant',
    'District',
    'District_Num',
    'Region',
    'Year',
    'Period',
    'Age_Grp',
    'Sex',
    'Count'
]]

# Save
table.to_csv(path_for_saved_files / 'ResourceFile_PopulationSize_2018Census.csv', index=False)

# %% Number of births

workingfile_fertility = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/Census_Main_Report/Series B. Fertility Tables.xlsx'

b1 = pd.read_excel(workingfile_fertility, sheet_name='TABLE B1')
b1 = b1.dropna()
b1 = b1.rename(columns={b1.columns[0]: 'Age/Region', b1.columns[1]: 'Num_Women_1549', b1.columns[2]: 'Live_Births',
                        b1.columns[3]: 'Babies_Live_12m', b1.columns[4]: 'Babies_Dead_12m'})
b1 = b1.drop(list(range(2, 10)), axis=0)
b1['region'] = None
region_labels = [r + ' Region' for r in region_names]
b1.loc[b1['Age/Region'].isin(region_labels), 'region'] = b1['Age/Region']
b1['region'] = b1['region'].ffill()
b1 = b1.drop(b1.index[b1['Age/Region'].isin(region_labels)], axis=0)
b1 = b1.rename(columns={'Age/Region': 'Age_Grp'})
b1 = b1[b1.columns[[0, 5, 1, 2, 3, 4]]]

# take the word 'region' out of the values in the 'region' column
b1['Region'] = b1['region'].str.replace(' Region', '')

# check that number of women 15-49 by region is close to the estimate for population size
from_fert_data = b1.groupby('region')['Num_Women_1549'].sum()
from_pop_data = table.loc[(table['Sex'] == 'F') & (
    table['Age_Grp'].isin(['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']))].groupby('Region')[
    'Count'].sum()
diff = 100 * (from_fert_data - from_pop_data) / from_fert_data

b1['Variant'] = 'Census_2018'
b1['Year'] = 2018
b1['Period'] = b1['Year'].map(calendar_period_lookup)
b1['Count'] = b1['Live_Births']

# save the file:
b1[['Variant', 'Year', 'Period', 'Region', 'Count']].to_csv(path_for_saved_files / 'ResourceFile_Births_2018Census.csv',
                                                            index=False)

# %% Number of deaths

workingfile_mortality = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/Census_Main_Report/Series K. Mortality Tables.xlsx'

k2 = pd.read_excel(workingfile_mortality, sheet_name='K2')

k2 = k2.dropna(how='all', axis=1)
k2.columns = ['Age/Region', 'Pop_Total', 'Pop_Males', 'Pop_Females', 'Deaths_Total', 'Deaths_Males', 'Deaths_Females']

k2 = k2.drop(list(range(2, 24)), axis=0)

k2['region'] = None
k2.loc[k2['Age/Region'].isin(region_names), 'region'] = k2['Age/Region']
k2['region'] = k2['region'].ffill()
k2 = k2.drop(k2.index[k2['Age/Region'].isin(region_names)], axis=0)
k2 = k2.rename(columns={'Age/Region': 'age_grp'})

k2 = k2[k2.columns[[7, 0, 1, 2, 3, 4, 5, 6]]]

k2['Variant'] = 'Census_2018'
k2['Year'] = 2018
k2['Period'] = k2['Year'].map(calendar_period_lookup)
k2['Age_Grp'] = k2['age_grp']
k2['Region'] = k2['region']

k2_melt = k2.melt(id_vars=['Variant', 'Year', 'Period', 'Age_Grp', 'Region'],
                  value_vars=['Deaths_Males', 'Deaths_Females'],
                  var_name='Sex',
                  value_name='Count')

k2_melt['Sex'] = k2_melt['Sex'].replace({'Deaths_Males': 'M', 'Deaths_Females': 'F'})

# save the file:
k2_melt.to_csv(path_for_saved_files / 'ResourceFile_Deaths_2018Census.csv', index=False)

# %% **** USE OF THE WPP DATA ****

# Population size: age groups
wpp_pop_males_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_POP_F07_2_POPULATION_BY_AGE_MALE.xlsx'

wpp_pop_females_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_POP_F07_3_POPULATION_BY_AGE_FEMALE.xlsx'

# Males
dat = pd.concat([
    pd.read_excel(wpp_pop_males_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(wpp_pop_males_file, sheet_name='LOW VARIANT', header=16),
    pd.read_excel(wpp_pop_males_file, sheet_name='MEDIUM VARIANT', header=16),
    pd.read_excel(wpp_pop_males_file, sheet_name='HIGH VARIANT', header=16)
], sort=False)

ests_males = dat.loc[dat[dat.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
ests_males['Sex'] = 'M'

# Females
dat = pd.concat([
    pd.read_excel(wpp_pop_females_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(wpp_pop_females_file, sheet_name='LOW VARIANT', header=16),
    pd.read_excel(wpp_pop_females_file, sheet_name='MEDIUM VARIANT', header=16),
    pd.read_excel(wpp_pop_females_file, sheet_name='HIGH VARIANT', header=16)
])

ests_females = dat.loc[dat[dat.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
ests_females['Sex'] = 'F'

# Join and tidy up
ests = pd.concat([ests_males, ests_females], sort=False)
ests = ests.drop(ests.columns[[0, 2, 3, 4, 5, 6]], axis=1)
ests[ests.columns[2:23]] = ests[ests.columns[
                                2:23]] * 1000  # given numbers are in 1000's, so multiply by 1000 to give actual

ests['Variant'] = 'WPP_' + ests['Variant']
ests = ests.rename(columns={ests.columns[1]: 'Year'})
ests_melt = ests.melt(id_vars=['Variant', 'Year', 'Sex'], value_name='Count', var_name='Age_Grp')
ests_melt['Period'] = ests_melt['Year'].map(calendar_period_lookup)

ests_melt.to_csv(path_for_saved_files / 'ResourceFile_Pop_WPP.csv', index=False)

# pop in 2010:
ests_melt.loc[ests_melt['Year'] == 2010, 'Count'].sum()  # ~14M

# %% Population size: single-year age/time steps
wpp_pop_males_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_INT_F03_2_POPULATION_BY_AGE_ANNUAL_MALE.xlsx'

wpp_pop_females_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_INT_F03_3_POPULATION_BY_AGE_ANNUAL_FEMALE.xlsx'

# Males
dat = pd.concat([
    pd.read_excel(wpp_pop_males_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(wpp_pop_males_file, sheet_name='MEDIUM VARIANT', header=16)
], sort=False)

ests_males = dat.loc[dat[dat.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
ests_males['Sex'] = 'M'

# Females
dat = pd.concat([
    pd.read_excel(wpp_pop_females_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(wpp_pop_females_file, sheet_name='MEDIUM VARIANT', header=16)
])

ests_females = dat.loc[dat[dat.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
ests_females['Sex'] = 'F'

# Join and tidy up
ests = pd.concat([ests_males, ests_females], sort=False)
ests = ests.drop(ests.columns[[0, 2, 3, 4, 5, 6]], axis=1)

ests[ests.columns[2:103]] = ests[ests.columns[
                                 2:103]] * 1000  # given numbers are in 1000's, so multiply by 1000 to give actual
ests = ests.rename(columns={ests.columns[1]: 'Year'})

# Remove duplicates (year 2020 is provided in each of the individual datasets)
ests.loc[ests.duplicated(subset=['Year', 'Sex']), ['Year']]
ests.drop_duplicates(subset=['Year', 'Sex'], inplace=True)

ests['Variant'] = 'WPP_' + ests['Variant']
ests_melt = ests.melt(id_vars=['Variant', 'Year', 'Sex'], value_name='Count', var_name='Age')

ests_melt['Period'] = ests_melt['Year'].map(calendar_period_lookup)

(__tmp__, age_grp_lookup) = create_age_range_lookup(min_age=0, max_age=100, range_size=5)
ests_melt['Age_Grp'] = ests_melt['Age'].astype(int).map(age_grp_lookup)
ests_melt.to_csv(path_for_saved_files / 'ResourceFile_Pop_Annual_WPP.csv', index=False)

# Make the initial population size for the model in 2010
# Age/sex breakdown from annual WPP - split by district breakdown from Census 2018
pop_2010 = ests_melt.loc[ests_melt['Year'] == 2010, ['Sex', 'Age', 'Count']].copy().reset_index(drop=True)
pop_2010['Age'] = pop_2010['Age'].astype(int)
pop_2010.sum()  # 14M
district_breakdown = table[['District', 'Count']].groupby(['District']).sum() / table['Count'].sum()

# There will be a neater way to do this, but....
init_pop_list = list()

for Sex in ['M', 'F']:
    for Age in range(101):
        for District in district_nums.index:
            tot_agesex_pop_across_districts = pop_2010.loc[
                (pop_2010['Sex'] == Sex) & (pop_2010['Age'] == Age),
                'Count'].values[0]

            frac_in_district = district_breakdown.loc[district_breakdown.index == District]['Count'].values[0]

            tot_agesex_pop_this_district = tot_agesex_pop_across_districts * frac_in_district

            record = {
                'District': District,
                'Sex': Sex,
                'Age': Age,
                'Count': tot_agesex_pop_this_district
            }
            init_pop_list.append(record)

init_pop = pd.DataFrame(init_pop_list)
init_pop = init_pop.merge(district_nums, left_on='District', right_index=True)
init_pop = init_pop[['District', 'District_Num', 'Region', 'Sex', 'Age', 'Count']].reset_index(drop=True)
assert init_pop['Count'].sum() == pop_2010['Count'].sum()

init_pop.to_csv(path_for_saved_files / 'ResourceFile_Population_2010.csv', index=False)

# %% Fertility and births

tot_births_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_FERT_F01_BIRTHS_BOTH_SEXES.xlsx'

tot_births = pd.concat([
    pd.read_excel(tot_births_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(tot_births_file, sheet_name='LOW VARIANT', header=16),
    pd.read_excel(tot_births_file, sheet_name='MEDIUM VARIANT', header=16),
    pd.read_excel(tot_births_file, sheet_name='HIGH VARIANT', header=16)
], sort=False)

tot_births = tot_births.loc[tot_births[tot_births.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
tot_births = tot_births.drop(tot_births.columns[[0, 2, 3, 4, 5, 6]], axis='columns')

tot_births = tot_births.melt(id_vars=['Variant'], var_name='Period', value_name='Total_Births').dropna()
tot_births['Total_Births'] = 1000 * tot_births['Total_Births']  # Imported units are 1000's

# Sex Ratio at Birth
sex_ratio_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_FERT_F02_SEX_RATIO_AT_BIRTH.xlsx'

sex_ratio = pd.concat([
    pd.read_excel(sex_ratio_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(sex_ratio_file, sheet_name='MEDIUM VARIANT', header=16)
], sort=False)

sex_ratio = sex_ratio.loc[sex_ratio[sex_ratio.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
sex_ratio = sex_ratio.drop(sex_ratio.columns[[0, 2, 3, 4, 5, 6]], axis='columns')
sex_ratio = sex_ratio.melt(id_vars=['Variant'], var_name='Period', value_name='M_to_F_Sex_Ratio').dropna()

# copy the medium variant sex ratio project for the low and high variants (in order to merge with the total births)
copy_high = sex_ratio.loc[sex_ratio['Variant'] == 'Medium variant', ['Period', 'M_to_F_Sex_Ratio']].copy()
copy_high['Variant'] = 'High variant'
sex_ratio = sex_ratio.append(copy_high, sort=False)

copy_low = sex_ratio.loc[sex_ratio['Variant'] == 'Medium variant', ['Period', 'M_to_F_Sex_Ratio']].copy()
copy_low['Variant'] = 'Low variant'
sex_ratio = sex_ratio.append(copy_low, sort=False)

# Combine these together
births = tot_births.merge(sex_ratio, on=['Variant', 'Period'], validate='1:1')


def reformat_date_period_for_wpp(wpp_import):
    # Relabel the calendar periods to be the inclusive year range (2010-2014 instead of 2010-2015)
    wpp_import['t_lo'], wpp_import['t_hi'] = wpp_import['Period'].str.split('-', 1).str
    wpp_import['t_hi'] = wpp_import['t_hi'].astype(int) - 1
    wpp_import['Period'] = wpp_import['t_lo'].astype(str) + '-' + wpp_import['t_hi'].astype(str)
    wpp_import.drop(columns=['t_lo', 't_hi'], inplace=True)


reformat_date_period_for_wpp(births)

births.to_csv(path_for_saved_files / 'ResourceFile_TotalBirths_WPP.csv', index=False)

# Give Fraction of births that are male for each year for easy importing to demography module
frac_birth_male = births.copy()
frac_birth_male['frac_births_male'] = frac_birth_male['M_to_F_Sex_Ratio'] / (1 + frac_birth_male['M_to_F_Sex_Ratio'])
frac_birth_male.drop(frac_birth_male.index[frac_birth_male['Variant'] == 'Low variant'], axis=0, inplace=True)
frac_birth_male.drop(frac_birth_male.index[frac_birth_male['Variant'] == 'High variant'], axis=0, inplace=True)
frac_birth_male.drop(['Variant', 'Total_Births', 'M_to_F_Sex_Ratio'], axis=1, inplace=True)
frac_birth_male['low_year'], frac_birth_male['high_year'] = frac_birth_male['Period'].str.split('-', 1).str
frac_birth_male['low_year'] = frac_birth_male['low_year'].astype(int)
frac_birth_male['high_year'] = frac_birth_male['high_year'].astype(int)

frac_birth_male_list = list()

# expand dataframe to give a frac_births_male for each year
for year in range(1950, 2100):
    frac_this_year = frac_birth_male.loc[
        (int(year) >= frac_birth_male['low_year']) & (int(year) <= frac_birth_male['high_year']),
        'frac_births_male'].values[0]

    record = {
        'Year': year,
        'frac_births_male': frac_this_year
    }

    frac_birth_male_list.append(record)

frac_birth_male_for_export = pd.DataFrame(frac_birth_male_list)
frac_birth_male_for_export.to_csv(path_for_saved_files / 'ResourceFile_Pop_Frac_Births_Male.csv', index=False)

# Age-specific Fertility Rates
asfr_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_FERT_F07_AGE_SPECIFIC_FERTILITY.xlsx'

asfr = pd.concat([
    pd.read_excel(asfr_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(asfr_file, sheet_name='LOW VARIANT', header=16),
    pd.read_excel(asfr_file, sheet_name='MEDIUM VARIANT', header=16),
    pd.read_excel(asfr_file, sheet_name='HIGH VARIANT', header=16)
], sort=False)

asfr = asfr.loc[asfr[asfr.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
asfr = asfr.drop(asfr.columns[[0, 2, 3, 4, 5, 6]], axis='columns')
asfr[asfr.columns[2:9]] = asfr[asfr.columns[
                               2:9]] / 1000  # given numbers are per 1000, so divide by 1000 to make 'per woman'

reformat_date_period_for_wpp(asfr)

# pivot into the usual long-format:
asfr['Variant'] = 'WPP_' + asfr['Variant']
asfr_melt = asfr.melt(id_vars=['Variant', 'Period'], value_name='asfr', var_name='Age_Grp')

asfr_melt.to_csv(path_for_saved_files / 'ResourceFile_ASFR_WPP.csv', index=False)

# %% Deaths

deaths_males_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_MORT_F04_2_DEATHS_BY_AGE_MALE.xlsx'

deaths_females_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_MORT_F04_3_DEATHS_BY_AGE_FEMALE.xlsx'

deaths_males = pd.concat([
    pd.read_excel(deaths_males_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(deaths_males_file, sheet_name='LOW VARIANT', header=16),
    pd.read_excel(deaths_males_file, sheet_name='MEDIUM VARIANT', header=16),
    pd.read_excel(deaths_males_file, sheet_name='HIGH VARIANT', header=16)
], sort=False, ignore_index=True)

deaths_males = deaths_males.loc[deaths_males[deaths_males.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
deaths_males['Sex'] = 'M'

deaths_females = pd.concat([
    pd.read_excel(deaths_females_file, sheet_name='ESTIMATES', header=16),
    pd.read_excel(deaths_females_file, sheet_name='LOW VARIANT', header=16),
    pd.read_excel(deaths_females_file, sheet_name='MEDIUM VARIANT', header=16),
    pd.read_excel(deaths_females_file, sheet_name='HIGH VARIANT', header=16)
], sort=False)

deaths_females = deaths_females.loc[deaths_females[deaths_females.columns[2]] == 'Malawi'].copy().reset_index(drop=True)
deaths_females['Sex'] = 'F'

# Join and tidy up
deaths = pd.concat([deaths_males, deaths_females], sort=False)
deaths = deaths.drop(deaths.columns[[0, 2, 3, 4, 5, 6]], axis=1)
deaths[deaths.columns[2:22]] = deaths[deaths.columns[
                                      2:22]] * 1000  # given numbers are in 1000's, so multiply by 1000 to give actual

reformat_date_period_for_wpp(deaths)

deaths_melt = deaths.melt(id_vars=['Variant', 'Period', 'Sex'], value_name='Count', var_name='Age_Grp')
deaths_melt['Count'].sum()
deaths_melt['Variant'] = 'WPP_' + deaths_melt['Variant']

deaths_melt.to_csv(path_for_saved_files / 'ResourceFile_TotalDeaths_WPP.csv', index=False)

# The ASMR from the LifeTable
lt_males_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_MORT_F17_2_ABRIDGED_LIFE_TABLE_MALE.xlsx'

lt_females_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/WPP_2019/WPP2019_MORT_F17_3_ABRIDGED_LIFE_TABLE_FEMALE.xlsx'

lt_males = pd.concat([pd.read_excel(lt_males_file, sheet_name='ESTIMATES', header=16, usecols='B,C,H,I,J,K'),
                      pd.read_excel(lt_males_file, sheet_name='MEDIUM 2020-2050', header=16, usecols='B,C,H,I,J,K'),
                      pd.read_excel(lt_males_file, sheet_name='MEDIUM 2050-2100', header=16, usecols='B,C,H,I,J,K')
                      ])

lt_males = lt_males.loc[lt_males[lt_males.columns[1]] == 'Malawi'].copy().reset_index(drop=True)
lt_males['Sex'] = 'M'

lt_females = pd.concat([pd.read_excel(lt_females_file, sheet_name='ESTIMATES', header=16, usecols='B,C,H,I,J,K'),
                        pd.read_excel(lt_females_file, sheet_name='MEDIUM 2020-2050', header=16, usecols='B,C,H,I,J,K'),
                        pd.read_excel(lt_females_file, sheet_name='MEDIUM 2050-2100', header=16, usecols='B,C,H,I,J,K')
                        ])

lt_females = lt_females.loc[lt_females[lt_females.columns[1]] == 'Malawi'].copy().reset_index(drop=True)
lt_females['Sex'] = 'F'

# Join and tidy up
lt = pd.concat([lt_males, lt_females], sort=False)
lt = lt.drop(lt.columns[[1]], axis=1)
lt.loc[lt['Variant'].str.contains('Medium'), 'Variant'] = 'Medium'
lt = lt.rename(columns={'Central death rate m(x,n)': 'death_rate'})  # NB. it is indeed an instantaneous rate
lt['Variant'] = 'WPP_' + lt['Variant']
lt.drop(lt.index[(lt['Age (x)'] == 100.0)], axis=0, inplace=True)

lt['Age_Grp'] = lt['Age (x)'].astype(int).astype(str) + '-' + (lt['Age (x)'] + lt['Age interval (n)'] - 1).astype(
    int).astype(str)
reformat_date_period_for_wpp(lt)

lt[['Variant', 'Period', 'Sex', 'Age_Grp', 'death_rate']].to_csv(
    path_for_saved_files / 'ResourceFile_Pop_DeathRates_WPP.csv', index=False)

# Expand the life-table to create a row for each age year, for ease of indexing in the simulation
mort_sched = lt.copy()

mort_sched['low_age'], mort_sched['high_age'] = mort_sched['Age_Grp'].str.split('-', 1).str
mort_sched['low_age'] = mort_sched['low_age'].astype(int)
mort_sched['high_age'] = mort_sched['high_age'].astype(int)

mort_sched_expanded_as_list = list()
for period in pd.unique(mort_sched['Period']):
    for sex in ['M', 'F']:
        for age_years in range(120):  # MAX_AGE

            if age_years > 99:
                age_years_to_look_up = 99
            else:
                age_years_to_look_up = age_years

            mask = (period == mort_sched['Period']) & \
                   (age_years_to_look_up >= mort_sched['low_age']) & \
                   (age_years_to_look_up <= mort_sched['high_age']) & \
                   (sex == mort_sched['Sex'])

            assert mask.sum() == 1

            the_death_rate = float(mort_sched.loc[mask, 'death_rate'].values[0])

            record = {
                'fallbackyear': int(period.split('-')[0]),
                'age_years': age_years,
                'sex': sex,
                'death_rate': the_death_rate
            }
            mort_sched_expanded_as_list.append(record)

mort_sched_expanded = pd.DataFrame(mort_sched_expanded_as_list,
                                   columns=['fallbackyear', 'sex', 'age_years', 'death_rate'])

mort_sched_expanded.to_csv(path_for_saved_files / 'ResourceFile_Pop_DeathRates_Expanded_WPP.csv', index=False)


# %% *** DHS DATA

dhs_working_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/\
Module-demography/DHS/STATcompilerExport20191112_211640.xlsx'

dhs_asfr = pd.read_excel(dhs_working_file, sheet_name='ASFR')
dhs_asfr[dhs_asfr.columns[1:]] = dhs_asfr[dhs_asfr.columns[1:]] / 1000  # to make the ASFR per women
dhs_asfr.to_csv(path_for_saved_files / 'ResourceFile_ASFR_DHS.csv', index=False)


dhs_u5 = pd.read_excel(dhs_working_file, sheet_name='UNDER_5_MORT', header=1, index=False)
dhs_u5['Year'] = dhs_u5.index
dhs_u5 = dhs_u5.reset_index(drop=True)
dhs_u5 = dhs_u5[dhs_u5.columns[[3, 0, 1, 2]]]
dhs_u5[dhs_u5.columns[1:]] = dhs_u5[dhs_u5.columns[1:]] / 1000  # to make it mortality risk per person
dhs_u5.to_csv(path_for_saved_files / 'ResourceFile_Under_Five_Mortality_DHS.csv', index=False)
