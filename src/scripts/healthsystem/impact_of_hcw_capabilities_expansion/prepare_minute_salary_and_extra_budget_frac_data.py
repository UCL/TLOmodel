"""
We calculate the salary cost of current and funded plus HCW.
"""
import itertools
# import pickle
from pathlib import Path

import numpy as np
import pandas as pd

resourcefilepath = Path('./resources')

mfl = pd.read_csv(resourcefilepath / 'healthsystem' / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')

hr_salary = pd.read_csv(resourcefilepath /
                        'costing' / 'ResourceFile_Annual_Salary_Per_Cadre.csv', index_col=False)
hr_salary_per_level = pd.read_excel(resourcefilepath /
                                    'costing' / 'ResourceFile_Costing.xlsx', sheet_name='human_resources')
hr_current = pd.read_csv(resourcefilepath /
                         'healthsystem' / 'human_resources' / 'actual' / 'ResourceFile_Daily_Capabilities.csv')
hr_established = pd.read_csv(resourcefilepath /
                             'healthsystem' / 'human_resources' / 'funded_plus' / 'ResourceFile_Daily_Capabilities.csv')

# to get minute salary per cadre per level
Annual_PFT = hr_current.groupby(['Facility_Level', 'Officer_Category']).agg(
    {'Total_Mins_Per_Day': 'sum', 'Staff_Count': 'sum'}).reset_index()
Annual_PFT['Annual_Mins_Per_Staff'] = 365.25 * Annual_PFT['Total_Mins_Per_Day']/Annual_PFT['Staff_Count']

# the hr salary by minute and facility id
Minute_Salary = Annual_PFT.merge(hr_salary, on=['Officer_Category'], how='outer')
Minute_Salary['Minute_Salary_USD'] = Minute_Salary['Annual_Salary_USD']/Minute_Salary['Annual_Mins_Per_Staff']
# store the minute salary by cadre and level
Minute_Salary_by_Cadre_Level = Minute_Salary[
    ['Facility_Level', 'Officer_Category', 'Minute_Salary_USD']
].copy().fillna(0.0)
Minute_Salary = Minute_Salary[['Facility_Level', 'Officer_Category', 'Minute_Salary_USD']].merge(
    mfl[['Facility_Level', 'Facility_ID']], on=['Facility_Level'], how='outer'
)
Minute_Salary.drop(columns=['Facility_Level'], inplace=True)
Minute_Salary = Minute_Salary.fillna(0.0)
Minute_Salary.rename(columns={'Officer_Category': 'Officer_Type_Code'}, inplace=True)

Minute_Salary.to_csv(resourcefilepath / 'costing' / 'Minute_Salary_HR.csv', index=False)

# calculate the current cost distribution of all cadres
cadre_all = ['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy',
             'Dental', 'Laboratory', 'Mental', 'Nutrition', 'Radiography']
staff_count = hr_current.groupby('Officer_Category')['Staff_Count'].sum().reset_index()
staff_cost = staff_count.merge(hr_salary, on=['Officer_Category'], how='outer')
staff_cost['annual_cost'] = staff_cost['Staff_Count'] * staff_cost['Annual_Salary_USD']
staff_cost['cost_frac'] = (staff_cost['annual_cost'] / staff_cost['annual_cost'].sum())
assert staff_cost.cost_frac.sum() == 1
staff_cost.set_index('Officer_Category', inplace=True)
staff_cost = staff_cost.reindex(index=cadre_all)

# No expansion scenario, or zero-extra-budget-fraction scenario, "s_0"
# Define the current cost fractions among all cadres as extra-budget-fraction scenario "s_1" \
# to be matched with Margherita's 4.2% scenario.
# Add in the scenario that is indicated by hcw cost gap distribution \
# resulted from never ran services in no expansion scenario, "s_2"
# Define all other scenarios so that the extra budget fraction of each cadre, \
# i.e., four main cadres and the "Other" cadre that groups up all other cadres, is the same (fair allocation)

cadre_group = ['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy', 'Other']  # main cadres
other_group = ['Dental', 'Laboratory', 'Mental', 'Nutrition', 'Radiography']

# create scenarios
combination_list = ['s_0', 's_1', 's_2']  # the three special scenarios
for n in range(1, len(cadre_group)+1):
    for subset in itertools.combinations(cadre_group, n):
        combination_list.append(str(subset))  # other equal-fraction scenarios

# cadre groups to expand
cadre_to_expand = pd.DataFrame(index=cadre_group, columns=combination_list).fillna(0.0)
for c in cadre_group:
    for i in cadre_to_expand.columns[3:]:  # for all equal-fraction scenarios
        if c in i:
            cadre_to_expand.loc[c, i] = 1  # value 1 indicate the cadre group will be expanded

# prepare auxiliary dataframe for equal extra budget fractions scenarios
auxiliary = cadre_to_expand.copy()
for i in auxiliary.columns[3:]:  # for all equal-fraction scenarios
    auxiliary.loc[:, i] = auxiliary.loc[:, i] / auxiliary.loc[:, i].sum()
auxiliary.loc[:, 's_2'] = [0.4586, 0.0272, 0.3502, 0.1476, 0.0164]  # without historical scaling
# auxiliary.loc[:, 's_2'] = [0.4322, 0.0201, 0.3701, 0.1408, 0.0368]  # with historical scaling

# define extra budget fracs for each cadre
extra_budget_fracs = pd.DataFrame(index=cadre_all, columns=combination_list)
assert (extra_budget_fracs.columns == auxiliary.columns).all()
assert (extra_budget_fracs.index[0:4] == auxiliary.index[0:4]).all()

extra_budget_fracs.loc[:, 's_0'] = 0
assert (staff_cost.index == extra_budget_fracs.index).all()
extra_budget_fracs.loc[:, 's_1'] = staff_cost.loc[:, 'cost_frac'].values

for i in extra_budget_fracs.columns[2:]:
    for c in extra_budget_fracs.index:
        if c in auxiliary.index:  # the four main cadres
            extra_budget_fracs.loc[c, i] = auxiliary.loc[c, i]
        else:  # the other 5 cadres
            extra_budget_fracs.loc[c, i] = auxiliary.loc['Other', i] * (
                staff_cost.loc[c, 'cost_frac'] / staff_cost.loc[staff_cost.index.isin(other_group), 'cost_frac'].sum()
            )  # current cost distribution among the 5 other cadres
            # extra_budget_fracs.loc[c, i] = auxiliary.loc['Other', i] / 5  # equal fracs among the 5 other cadres

assert (abs(extra_budget_fracs.iloc[:, 1:len(extra_budget_fracs.columns)].sum(axis=0) - 1.0) < 1/1e10).all()

# rename scenarios
# make the scenario of equal fracs for all five cadre groups (i.e., the last column) to be s_3
simple_scenario_name = {extra_budget_fracs.columns[-1]: 's_3'}
for i in range(3, len(extra_budget_fracs.columns)-1):
    simple_scenario_name[extra_budget_fracs.columns[i]] = 's_' + str(i+1)  # name scenario from s_4
extra_budget_fracs.rename(columns=simple_scenario_name, inplace=True)

# reorder columns
col_order = ['s_' + str(i) for i in range(0, len(extra_budget_fracs.columns))]
assert len(col_order) == len(extra_budget_fracs.columns)
extra_budget_fracs = extra_budget_fracs.reindex(columns=col_order)


# calculate hr scale up factor for years 2020-2030 (10 years in total) outside the healthsystem module

def calculate_hr_scale_up_factor(extra_budget_frac, yr, scenario) -> pd.DataFrame:
    """This function calculates the yearly hr scale up factor for cadres for a year yr,
    given a fraction of an extra budget allocated to each cadre and a yearly budget growth rate of 4.2%.
    Parameter extra_budget_frac (list) is a list of 9 floats, representing the fractions.
    Parameter yr (int) is a year between 2019 and 2030.
    Parameter scenario (string) is a column name in the extra budget fractions resource file.
    Output dataframe stores scale up factors and relevant for the year yr.
    """
    # get data of previous year
    prev_year = yr - 1
    prev_data = scale_up_factor_dict[scenario][prev_year].copy()

    # calculate and update scale_up_factor
    prev_data['extra_budget_frac'] = extra_budget_frac
    prev_data['extra_budget'] = 0.042 * prev_data.annual_cost.sum() * prev_data.extra_budget_frac
    prev_data['extra_staff'] = prev_data.extra_budget / prev_data.Annual_Salary_USD
    prev_data['scale_up_factor'] = (prev_data.Staff_Count + prev_data.extra_staff) / prev_data.Staff_Count

    # store the updated data for the year yr
    new_data = prev_data[['Annual_Salary_USD', 'scale_up_factor']].copy()
    new_data['Staff_Count'] = prev_data.Staff_Count + prev_data.extra_staff
    new_data['annual_cost'] = prev_data.annual_cost + prev_data.extra_budget

    return new_data


# calculate scale up factors for all defined scenarios and years
staff_cost['scale_up_factor'] = 1
scale_up_factor_dict = {s: {y: {} for y in range(2018, 2030)} for s in extra_budget_fracs.columns}
for s in extra_budget_fracs.columns:
    # for the initial/current year of 2018
    scale_up_factor_dict[s][2018] = staff_cost.drop(columns='cost_frac').copy()
    # for the years with scaled up hr
    for y in range(2019, 2030):
        scale_up_factor_dict[s][y] = calculate_hr_scale_up_factor(list(extra_budget_fracs[s]), y, s)

# get the total cost and staff count for each year between 2020-2030 and each scenario
total_cost = pd.DataFrame(index=range(2018, 2030), columns=extra_budget_fracs.columns)
total_staff = pd.DataFrame(index=range(2018, 2030), columns=extra_budget_fracs.columns)
for y in total_cost.index:
    for s in extra_budget_fracs.columns:
        total_cost.loc[y, s] = scale_up_factor_dict[s][y].annual_cost.sum()
        total_staff.loc[y, s] = scale_up_factor_dict[s][y].Staff_Count.sum()

# check the total cost after 11 years are increased as expected
assert (
    abs(total_cost.loc[2029, total_cost.columns[1:]] - (1 + 0.042) ** 11 * total_cost.loc[2029, 's_0']) < 1/1e7
).all()

# get the integrated scale up factors by the end of year 2029 and each scenario
integrated_scale_up_factor = pd.DataFrame(index=cadre_all, columns=total_cost.columns).fillna(1.0)
for s in total_cost.columns[1:]:
    for yr in range(2019, 2030):
        integrated_scale_up_factor.loc[:, s] = np.multiply(
            integrated_scale_up_factor.loc[:, s].values,
            scale_up_factor_dict[s][yr].loc[:, 'scale_up_factor'].values
        )

# Checked that for s_2, the integrated scale up factors of C/N/P cadres are comparable with shortage estimates from \
# She et al 2024: https://human-resources-health.biomedcentral.com/articles/10.1186/s12960-024-00949-2
# C: 2.21, N: 1.44, P: 4.14 vs C: 2.83, N: 1.57, P:6.37
# todo: This might provide a short-cut way (no simulation, but mathematical calculation) to calculate \
# an extra budget allocation scenario 's_2+' that is comparable with s_2.

# # save and read pickle file
# pickle_file_path = Path(resourcefilepath / 'healthsystem' / 'human_resources' / 'scaling_capabilities' /
#                         'ResourceFile_HR_expansion_by_officer_type_yearly_scale_up_factors.pickle')
#
# with open(pickle_file_path, 'wb') as f:
#     pickle.dump(scale_up_factor_dict, f)
#
# with open(pickle_file_path, 'rb') as f:
#     x = pickle.load(f)
