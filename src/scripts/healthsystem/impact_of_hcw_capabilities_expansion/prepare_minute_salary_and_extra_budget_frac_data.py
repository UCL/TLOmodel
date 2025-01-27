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
# as of 2019
hr_current = pd.read_csv(resourcefilepath /
                         'healthsystem' / 'human_resources' / 'actual' / 'ResourceFile_Daily_Capabilities.csv')
hr_established = pd.read_csv(resourcefilepath /
                             'healthsystem' / 'human_resources' / 'funded_plus' / 'ResourceFile_Daily_Capabilities.csv')
# for 2020-2024
historical_scaling = pd.read_excel(resourcefilepath /
                                   'healthsystem' / 'human_resources' / 'scaling_capabilities' /
                                   'ResourceFile_dynamic_HR_scaling.xlsx', sheet_name='historical_scaling'
                                   ).set_index('year')
integrated_historical_scaling = (
    historical_scaling.loc[2020, 'dynamic_HR_scaling_factor'] *
    historical_scaling.loc[2021, 'dynamic_HR_scaling_factor'] *
    historical_scaling.loc[2022, 'dynamic_HR_scaling_factor'] *
    historical_scaling.loc[2023, 'dynamic_HR_scaling_factor'] *
    historical_scaling.loc[2024, 'dynamic_HR_scaling_factor']
)

# to get minute salary per cadre per level
Annual_PFT = hr_current.groupby(['Facility_Level', 'Officer_Category']).agg(
    {'Total_Mins_Per_Day': 'sum', 'Staff_Count': 'sum'}).reset_index()
Annual_PFT['Annual_Mins_Per_Staff'] = 365.25 * Annual_PFT['Total_Mins_Per_Day']/Annual_PFT['Staff_Count']

# the hr salary by minute and facility id, as of 2019
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

# implement historical scaling to hr_current
hr_current['Total_Mins_Per_Day'] *= integrated_historical_scaling
hr_current['Staff_Count'] *= integrated_historical_scaling

# calculate the current cost distribution of all cadres, as of 2024
cadre_all = ['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy',
             'Dental', 'Laboratory', 'Mental', 'Nutrition', 'Radiography']
staff_count = hr_current.groupby('Officer_Category')['Staff_Count'].sum().reset_index()
staff_cost = staff_count.merge(hr_salary, on=['Officer_Category'], how='outer')
staff_cost['annual_cost'] = staff_cost['Staff_Count'] * staff_cost['Annual_Salary_USD']
staff_cost['cost_frac'] = (staff_cost['annual_cost'] / staff_cost['annual_cost'].sum())
assert abs(staff_cost.cost_frac.sum() - 1) < 1/1e8
staff_cost.set_index('Officer_Category', inplace=True)
staff_cost = staff_cost.reindex(index=cadre_all)

# No expansion scenario, or zero-extra-budget-fraction scenario, "s_0"
# Define the current cost fractions among all cadres as extra-budget-fraction scenario "s_1" \
# to be matched with Margherita's 4.2% scenario.
# Add in the scenario that is indicated by hcw cost gap distribution \
# resulted from never ran services in no expansion scenario, "s_2"
# Add in the scenario that is indicated by the regression analysis of all other scenarios, "s_*"
# Define all other scenarios so that the extra budget fraction of each cadre, \
# i.e., four main cadres and the "Other" cadre that groups up all other cadres, is the same (fair allocation)

cadre_group = ['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy', 'Other']  # main cadres
other_group = ['Dental', 'Laboratory', 'Mental', 'Nutrition', 'Radiography']

# create scenarios
combination_list = ['s_0', 's_1', 's_2']  # the three special scenarios
for n in range(1, len(cadre_group)+1):
    for subset in itertools.combinations(cadre_group, n):
        combination_list.append(str(subset))  # other equal-fraction scenarios
# add in "s_*" in the end
combination_list.append('s_*')

# cadre groups to expand
cadre_to_expand = pd.DataFrame(index=cadre_group, columns=combination_list).fillna(0.0)
for c in cadre_group:
    for i in cadre_to_expand.columns[3:len(combination_list) - 1]:  # for all equal-fraction scenarios
        if c in i:
            cadre_to_expand.loc[c, i] = 1  # value 1 indicate the cadre group will be expanded

# prepare auxiliary dataframe for equal extra budget fractions scenarios
auxiliary = cadre_to_expand.copy()
for i in auxiliary.columns[3:len(combination_list) - 1]:  # for all equal-fraction scenarios
    auxiliary.loc[:, i] = auxiliary.loc[:, i] / auxiliary.loc[:, i].sum()
# for "gap" allocation strategy
# auxiliary.loc[:, 's_2'] = [0.4586, 0.0272, 0.3502, 0.1476, 0.0164]  # without historical scaling; "default" settings
auxiliary.loc[:, 's_2'] = [0.4314, 0.0214, 0.3701, 0.1406, 0.0365]  # historical scaling + main settings
# auxiliary.loc[:, 's_2'] = [0.4314, 0.0214, 0.3701, 0.1406, 0.0365]  # historical scaling + more_budget; same as above
# auxiliary.loc[:, 's_2'] = [0.4314, 0.0214, 0.3701, 0.1406, 0.0365]  # historical scaling + less_budget; same as above
# auxiliary.loc[:, 's_2'] = [0.4252, 0.0261, 0.3752, 0.1362, 0.0373]  # historical scaling + default_cons
# auxiliary.loc[:, 's_2'] = [0.5133, 0.0085, 0.2501, 0.1551, 0.073]  # historical scaling + max_hs_function
# for "optimal" allocation strategy
auxiliary.loc[:, 's_*'] = [0.6068, 0.0, 0.0830, 0.2496, 0.0606]  # historical scaling + main settings
# auxiliary.loc[:, 's_*'] = [0.5827, 0.0, 0.1083, 0.2409, 0.0681]  # historical scaling + more_budget; same as above
# auxiliary.loc[:, 's_*'] = [0.5981, 0.0, 0.0902, 0.2649, 0.0468]  # historical scaling + less_budget; same as above
# auxiliary.loc[:, 's_*'] = [0.6109, 0.0, 0.1494, 0.2033, 0.0364]  # historical scaling + default_cons
# auxiliary.loc[:, 's_*'] = [0.5430, 0.0, 0.3631, 0.0939, 0.0]  # historical scaling + max_hs_function

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
simple_scenario_name = {extra_budget_fracs.columns[-2]: 's_3'}
for i in range(3, len(extra_budget_fracs.columns)-2):
    simple_scenario_name[extra_budget_fracs.columns[i]] = 's_' + str(i+1)  # name scenario from s_4 to s_33
extra_budget_fracs.rename(columns=simple_scenario_name, inplace=True)

# reorder columns
col_order = ['s_' + str(i) for i in range(0, len(extra_budget_fracs.columns) - 1)]
col_order += ['s_*']
assert len(col_order) == len(extra_budget_fracs.columns)
extra_budget_fracs = extra_budget_fracs.reindex(columns=col_order)

# prepare samples for extra budget fracs that changes values for C, NM and P
# (the main cadres for service delivery and directly impacting health outcomes),
# where DCSA = 2% and Other = 4% -> 3% are fixed according to "gap" strategies
# and that these cadres either have limited impacts as estimated, deliver a very small proportion of services,
# or can deliver relevant services without being constrained by other cadres.
# value_list = list(np.arange(0, 100, 5))
# combinations = []
# for i in itertools.product(value_list, repeat=3):
#     if sum(i) == 95:
#         combinations.append(i)
# extra_budget_fracs_sample = pd.DataFrame(index=extra_budget_fracs.index, columns=range(len(combinations)+1))
# extra_budget_fracs_sample.iloc[:, 0] = 0
# extra_budget_fracs_sample.loc['DCSA', 1:] = 2
# for c in other_group:
#     extra_budget_fracs_sample.loc[c, 1:] = 3 * (
#         staff_cost.loc[c, 'cost_frac'] / staff_cost.loc[staff_cost.index.isin(other_group), 'cost_frac'].sum())
# for i in range(1, len(combinations)+1):
#     extra_budget_fracs_sample.loc[['Clinical', 'Nursing_and_Midwifery', 'Pharmacy'], i] = combinations[i-1]
# extra_budget_fracs_sample /= 100
# assert (abs(extra_budget_fracs_sample.iloc[:, 1:].sum(axis=0) - 1.0) < 1e-9).all()
# extra_budget_fracs_sample.rename(columns={0: 's_0'}, inplace=True)
#
# extra_budget_fracs = extra_budget_fracs_sample.copy()


# calculate hr scale up factor for years 2020-2030 (10 years in total) outside the healthsystem module
def calculate_hr_scale_up_factor(extra_budget_frac, yr, scenario) -> pd.DataFrame:
    """This function calculates the yearly hr scale up factor for cadres for a year yr,
    given a fraction of an extra budget allocated to each cadre and a yearly budget growth rate of 4.2%.
    Parameter extra_budget_frac (list) is a list of 9 floats, representing the fractions.
    Parameter yr (int) is a year between 2025 and 2035 (exclusive).
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
    new_data['increase_rate'] = new_data['scale_up_factor'] - 1.0

    return new_data


# calculate scale up factors for all defined scenarios and years
staff_cost['scale_up_factor'] = 1
staff_cost['increase_rate'] = 0.0
scale_up_factor_dict = {s: {y: {} for y in range(2025, 2035)} for s in extra_budget_fracs.columns}
for s in extra_budget_fracs.columns:
    # for the initial/current year of 2024
    scale_up_factor_dict[s][2024] = staff_cost.drop(columns='cost_frac').copy()
    # for the years with scaled up hr
    for y in range(2025, 2035):
        scale_up_factor_dict[s][y] = calculate_hr_scale_up_factor(list(extra_budget_fracs[s]), y, s)

# get the total cost and staff count for each year between 2024-2034 and each scenario
total_cost = pd.DataFrame(index=range(2024, 2035), columns=extra_budget_fracs.columns)
total_staff = pd.DataFrame(index=range(2024, 2035), columns=extra_budget_fracs.columns)
for y in total_cost.index:
    for s in extra_budget_fracs.columns:
        total_cost.loc[y, s] = scale_up_factor_dict[s][y].annual_cost.sum()
        total_staff.loc[y, s] = scale_up_factor_dict[s][y].Staff_Count.sum()

# check the total cost after 10 years are increased as expected
assert (
    abs(total_cost.loc[2034, total_cost.columns[1:]] - (1 + 0.042) ** 10 * total_cost.loc[2024, 's_0']) < 1/1e6
).all()

# get the integrated scale up factors by the end of year 2034 and each scenario
integrated_scale_up_factor = pd.DataFrame(index=cadre_all, columns=total_cost.columns).fillna(1.0)
for s in total_cost.columns[1:]:
    for yr in range(2025, 2035):
        integrated_scale_up_factor.loc[:, s] = np.multiply(
            integrated_scale_up_factor.loc[:, s].values,
            scale_up_factor_dict[s][yr].loc[:, 'scale_up_factor'].values
        )

# get normal average increase rate over all years
sum_increase_rate = pd.DataFrame(index=cadre_all, columns=total_cost.columns).fillna(0.0)
for s in total_cost.columns[1:]:
    for yr in range(2025, 2035):
        sum_increase_rate.loc[:, s] = (
            sum_increase_rate.loc[:, s].values +
            scale_up_factor_dict[s][yr].loc[:, 'increase_rate'].values
        )
avg_increase_rate = pd.DataFrame(sum_increase_rate / 10)

# get the staff increase rate: 2034 vs 2025
increase_rate_2034 = pd.DataFrame(integrated_scale_up_factor - 1.0)
avg_increase_rate_exp = pd.DataFrame(integrated_scale_up_factor**(1/10) - 1.0)

# get the linear regression prediction
# 1.003	0.4122	1.0178	0.269	0.2002	-0.0686
# const = -0.0686
# coefs = [1.003, 0.4122, 1.0178, 0.269, 0.2002]
# predict_dalys_averted_percent = avg_increase_rate_exp.loc[
#                                 ['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy', 'Dental'],
#                                 :].mul(coefs, axis=0).sum() + const
# extra_budget_fracs_sample = extra_budget_fracs_sample.T
# extra_budget_fracs_sample.loc[:, 'DALYs averted %'] = predict_dalys_averted_percent.values * 100
# extra_budget_fracs_sample.drop(
#     index=extra_budget_fracs_sample[extra_budget_fracs_sample['DALYs averted %'] < 8.0].index, inplace=True)
# extra_budget_fracs_sample['C + P'] = extra_budget_fracs_sample['Clinical'] + extra_budget_fracs_sample['Pharmacy']
# extra_budget_fracs_sample['C + NM'] = (extra_budget_fracs_sample['Clinical']
#                                        + extra_budget_fracs_sample['Nursing_and_Midwifery'])
# extra_budget_fracs_sample['NM + P'] = (extra_budget_fracs_sample['Nursing_and_Midwifery']
#                                        + extra_budget_fracs_sample['Pharmacy'])
# min_row = pd.DataFrame(extra_budget_fracs_sample.min(axis=0)).T.rename(index={0: 'Min'})
# max_row = pd.DataFrame(extra_budget_fracs_sample.max(axis=0)).T.rename(index={0: 'Max'})
# extra_budget_fracs_sample = pd.concat([extra_budget_fracs_sample, min_row, max_row])
# extra_budget_fracs_sample.drop(columns=other_group, inplace=True)


def func_of_avg_increase_rate(cadre, scenario='s_2', r=0.042):
    """
    This return the average growth rate of the staff of a cadre from 2025 to 2034.
    The total HRH cost growth rate is r.
    """
    overall_scale_up = 1 + (staff_cost.annual_cost.sum()
                            * extra_budget_fracs.loc[cadre, scenario]
                            / staff_cost.loc[cadre, 'annual_cost']
                            * ((1+r)**10 - 1)
                            )

    return overall_scale_up ** (1/10) - 1.0


# prepare 2024 cost info for Other cadre and Total
extra_rows = pd.DataFrame(columns=staff_cost.columns, index=['Other', 'Total'])
staff_cost = pd.concat([staff_cost, extra_rows], axis=0)
staff_cost.loc['Other', 'annual_cost'] = staff_cost.loc[staff_cost.index.isin(other_group), 'annual_cost'].sum()
staff_cost.loc['Total', 'annual_cost'] = staff_cost.loc[staff_cost.index.isin(cadre_all), 'annual_cost'].sum()
staff_cost.loc['Other', 'Staff_Count'] = staff_cost.loc[staff_cost.index.isin(other_group), 'Staff_Count'].sum()
staff_cost.loc['Total', 'Staff_Count'] = staff_cost.loc[staff_cost.index.isin(cadre_all), 'Staff_Count'].sum()
staff_cost.loc['Other', 'cost_frac'] = (staff_cost.loc['Other', 'annual_cost']
                                        / staff_cost.loc[staff_cost.index.isin(cadre_all), 'annual_cost'].sum())
staff_cost.loc['Total', 'cost_frac'] = (staff_cost.loc['Total', 'annual_cost']
                                        / staff_cost.loc[staff_cost.index.isin(cadre_all), 'annual_cost'].sum())
staff_cost.annual_cost = staff_cost.annual_cost.astype(str)
staff_cost.cost_frac = staff_cost.cost_frac.astype(str)

# # save and read pickle file
# pickle_file_path = Path(resourcefilepath / 'healthsystem' / 'human_resources' / 'scaling_capabilities' /
#                         'ResourceFile_HR_expansion_by_officer_type_yearly_scale_up_factors.pickle')
#
# with open(pickle_file_path, 'wb') as f:
#     pickle.dump(scale_up_factor_dict, f)
#
# with open(pickle_file_path, 'rb') as f:
#     x = pickle.load(f)
