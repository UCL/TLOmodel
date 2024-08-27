"""
We calculate the salar cost of current and funded plus HCW.
"""
import itertools
import pickle
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
Minute_Salary = Minute_Salary[['Facility_Level', 'Officer_Category', 'Minute_Salary_USD']].merge(
    mfl[['Facility_Level', 'Facility_ID']], on=['Facility_Level'], how='outer'
)
Minute_Salary.drop(columns=['Facility_Level'], inplace=True)
Minute_Salary = Minute_Salary.fillna(0)
Minute_Salary.rename(columns={'Officer_Category': 'Officer_Type_Code'}, inplace=True)

Minute_Salary.to_csv(resourcefilepath / 'costing' / 'Minute_Salary_HR.csv', index=False)

# calculate the current cost distribution of the four cadres
staff_count = hr_current.groupby('Officer_Category')['Staff_Count'].sum().reset_index()
staff_cost = staff_count.merge(hr_salary, on=['Officer_Category'], how='outer')
staff_cost['annual_cost'] = staff_cost['Staff_Count'] * staff_cost['Annual_Salary_USD']
four_cadres_cost = staff_cost.loc[
    staff_cost.Officer_Category.isin(['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy'])].reset_index(drop=True)
four_cadres_cost['cost_frac'] = (four_cadres_cost['annual_cost'] / four_cadres_cost['annual_cost'].sum())
# x = four_cadres_cost.loc[0, 'cost_frac'].as_integer_ratio()
assert four_cadres_cost.cost_frac.sum() == 1

# Calculate the current cost distribution of one/two/three/four cadres and define them as scenarios
# We confirmed/can prove that in such expansion scenarios of two/three/four cadres,
# the annual scale up factors are actually equal for cadres,
# equal to 1 + annual extra cost / total current cost of two/three/four cadres.
# One possible issue is that Pharmacy cost has only small fractions in all multi-cadre scenarios,
# as its current fraction is small; we have estimated that Pharmacy cadre is extremely in shortage,
# thus these scenarios might still face huge shortages.
cadres = ['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy']
combination_list = ['']
for n in range(1, len(cadres)+1):
    for subset in itertools.combinations(cadres, n):
        combination_list.append(str(subset))

cadre_to_expand = pd.DataFrame(index=cadres, columns=combination_list).fillna(0)
cadre_to_expand.loc[:, ''] = 0  # no_expansion scenario
for c in cadres:
    for i in cadre_to_expand.columns:
        if c in i:
            cadre_to_expand.loc[c, i] = staff_cost.loc[staff_cost.Officer_Category == c, 'annual_cost'].values[0]

extra_budget_fracs = pd.DataFrame(index=cadre_to_expand.index, columns=cadre_to_expand.columns).fillna(0)
for i in extra_budget_fracs.columns[1:]:
    extra_budget_fracs.loc[:, i] = cadre_to_expand.loc[:, i] / cadre_to_expand.loc[:, i].sum()

assert (abs(extra_budget_fracs.iloc[:, 1:len(extra_budget_fracs.columns)].sum(axis=0) - 1.0) < 1/1e10).all()

simple_scenario_name = {}
for i in range(len(extra_budget_fracs.columns)):
    simple_scenario_name[extra_budget_fracs.columns[i]] = 's_' + str(i+1)  # name scenario from s_1
extra_budget_fracs.rename(columns=simple_scenario_name, inplace=True)


# calculate hr scale up factor for years 2020-2030 (10 years in total) outside the healthsystem module

def calculate_hr_scale_up_factor(extra_budget_frac, yr, scenario) -> pd.DataFrame:
    """This function calculates the yearly hr scale up factor for Clinical, DCSA, Nursing_and_Midwifery,
    and Pharmacy cadres for a year yr, given a fraction of an extra budget allocated to each cadre and
    a yearly budget growth rate of 4.2%.
    Parameter extra_budget_frac (list) is a list of four floats, representing the fractions.
    Parameter yr (int) is a year between 2020 and 2030.
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
    new_data = prev_data[['Officer_Category', 'Annual_Salary_USD', 'scale_up_factor']].copy()
    new_data['Staff_Count'] = prev_data.Staff_Count + prev_data.extra_staff
    new_data['annual_cost'] = prev_data.annual_cost + prev_data.extra_budget

    return new_data


# calculate scale up factors for all defined scenarios and years
four_cadres_cost['scale_up_factor'] = 1
scale_up_factor_dict = {s: {y: {} for y in range(2019, 2030)} for s in extra_budget_fracs.columns}
for s in extra_budget_fracs.columns:
    # for the initial/current year of 2019
    scale_up_factor_dict[s][2019] = four_cadres_cost.drop(columns='cost_frac').copy()
    # for the years with scaled up hr
    for y in range(2020, 2030):
        scale_up_factor_dict[s][y] = calculate_hr_scale_up_factor(list(extra_budget_fracs[s]), y, s)

# get the total cost and staff count for each year between 2020-2030 and each scenario
total_cost = pd.DataFrame(index=range(2020, 2030), columns=extra_budget_fracs.columns)
total_staff = pd.DataFrame(index=range(2020, 2030), columns=extra_budget_fracs.columns)
for y in total_cost.index:
    for s in extra_budget_fracs.columns:
        total_cost.loc[y, s] = scale_up_factor_dict[s][y].annual_cost.sum()
        total_staff.loc[y, s] = scale_up_factor_dict[s][y].Staff_Count.sum()

# check the total cost after 10 years are increased as expected
assert (
    abs(total_cost.loc[2029, total_cost.columns[1:]] - (1 + 0.042) ** 10 * total_cost.loc[2029, 's_1']) < 1/1e6
).all()

# get the integrated scale up factors for year 2029 and each scenario
integrated_scale_up_factor = pd.DataFrame(index=cadres, columns=total_cost.columns).fillna(1.0)
for s in total_cost.columns[1:]:
    for yr in range(2020, 2030):
        integrated_scale_up_factor.loc[:, s] = np.multiply(
            integrated_scale_up_factor.loc[:, s].values,
            scale_up_factor_dict[s][yr].loc[:, 'scale_up_factor'].values
        )

# # save and read pickle file
# pickle_file_path = Path(resourcefilepath / 'healthsystem' / 'human_resources' / 'scaling_capabilities' /
#                         'ResourceFile_HR_expansion_by_officer_type_yearly_scale_up_factors.pickle')
#
# with open(pickle_file_path, 'wb') as f:
#     pickle.dump(scale_up_factor_dict, f)
#
# with open(pickle_file_path, 'rb') as f:
#     x = pickle.load(f)
