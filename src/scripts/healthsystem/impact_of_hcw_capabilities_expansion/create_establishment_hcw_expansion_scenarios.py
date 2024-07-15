"""
We calculate the salar cost of current and funded plus HCW.
"""

from pathlib import Path

import numpy as np
import pandas as pd

resourcefilepath = Path('./resources')

hr_salary = pd.read_csv(resourcefilepath /
                        'costing' / 'ResourceFile_Annual_Salary_Per_Cadre.csv', index_col=False)
hr_current = pd.read_csv(resourcefilepath /
                         'healthsystem' / 'human_resources' / 'actual' / 'ResourceFile_Daily_Capabilities.csv')
hr_established = pd.read_csv(resourcefilepath /
                             'healthsystem' / 'human_resources' / 'funded_plus' / 'ResourceFile_Daily_Capabilities.csv')

hr_curr_count = hr_current.groupby('Officer_Category').agg({'Staff_Count': 'sum'})
hr_estab_count = (hr_established.groupby('Officer_Category').agg({'Staff_Count': 'sum'}))

hr = hr_curr_count.merge(hr_estab_count, on='Officer_Category', how='outer'
                         ).merge(hr_salary, on='Officer_Category', how='left')

hr['total_curr_salary'] = hr['Staff_Count_x'] * hr['Annual_Salary_USD']
hr['total_estab_salary'] = hr['Staff_Count_y'] * hr['Annual_Salary_USD']

total_curr_salary = hr['total_curr_salary'].sum()  # 107.82 million
total_estab_salary = hr['total_estab_salary'].sum()  # 201.36 million

# now consider expanding establishment HCW
# assuming annual GDP growth rate is 4.2% and
# a fixed proportion of GDP is allocated to human resource expansion, thus assuming
# the annual growth rate of HR salary cost is also 4.2%.

# the annual extra budget and
# if to expand one individual cadre in ['Clinical', 'Nursing_and_Midwifery', 'Pharmacy', 'DCSA']
hr['extra_budget'] = total_estab_salary * 4.2 / 100  # 8.46 million
hr['individual_increase'] = np.floor(hr['extra_budget'] / hr['Annual_Salary_USD'])
# do not increase other cadres
for c in hr.Officer_Category:
    if c not in ['Clinical', 'Nursing_and_Midwifery', 'Pharmacy', 'DCSA']:
        hr.loc[hr.Officer_Category == c, 'individual_increase'] = 0
hr['individual_scale_up_factor'] = (hr['individual_increase'] + hr['Staff_Count_y']) / hr['Staff_Count_y']
hr['individual_increase_%'] = hr['individual_increase'] * 100 / hr['Staff_Count_y']

# if to expand multiple cadres in ['Clinical', 'Nursing_and_Midwifery', 'Pharmacy', 'DCSA']
hr_expand = hr.loc[hr.Officer_Category.isin(['Clinical', 'Nursing_and_Midwifery', 'Pharmacy', 'DCSA']),
                   ['Officer_Category', 'individual_increase']].copy()
hr_expand['individual_increase_0'] = np.floor(hr_expand['individual_increase'] * 0)
hr_expand['individual_increase_20%'] = np.floor(hr_expand['individual_increase'] * 0.2)
hr_expand['individual_increase_40%'] = np.floor(hr_expand['individual_increase'] * 0.4)
hr_expand['individual_increase_60%'] = np.floor(hr_expand['individual_increase'] * 0.6)
hr_expand['individual_increase_80%'] = np.floor(hr_expand['individual_increase'] * 0.8)
hr_expand['individual_increase_1'] = np.floor(hr_expand['individual_increase'] * 1)

hr_expand.drop(columns='individual_increase', inplace=True)
hr_expand.set_index('Officer_Category', inplace=True)

c_array = hr_expand.loc['Clinical'].values
nm_array = hr_expand.loc['Nursing_and_Midwifery'].values
p_array = hr_expand.loc['Pharmacy'].values
d_array = hr_expand.loc['DCSA'].values
hr_meshgrid = np.array(
    np.meshgrid(c_array, nm_array, p_array, d_array)).T.reshape(-1, 4)

hr_expand_scenario = pd.DataFrame({'Clinical': hr_meshgrid[:, 0],
                                   'Nursing_and_Midwifery': hr_meshgrid[:, 1],
                                   'Pharmacy': hr_meshgrid[:, 2],
                                   'DCSA': hr_meshgrid[:, 3]}).T

hr_expand_salary = hr.loc[hr.Officer_Category.isin(['Clinical', 'Nursing_and_Midwifery', 'Pharmacy', 'DCSA']),
                          ['Officer_Category', 'Annual_Salary_USD']].copy()
hr_expand_salary.set_index('Officer_Category', inplace=True)

hr_expand_scenario_cost = hr_expand_salary.merge(hr_expand_scenario, left_index=True, right_index=True)
hr_expand_scenario_cost.loc[:, hr_expand_scenario_cost.columns[1:]] = \
    hr_expand_scenario_cost.loc[:, hr_expand_scenario_cost.columns[1:]].multiply(
    hr_expand_scenario_cost.loc[:, hr_expand_scenario_cost.columns[0]], axis='index')
hr_expand_scenario_cost.loc['Total'] = hr_expand_scenario_cost.sum()
# hr_expand_scenario_cost.drop(columns=['Annual_Salary_USD'], inplace=True)

cond = (hr_expand_scenario_cost.loc['Total'] <= total_estab_salary * 4.2 / 100)
hr_expand_scenario_budget = hr_expand_scenario_cost.loc[:, cond].copy()
hr_expand_scenario_budget.drop(index='Total', inplace=True)
hr_expand_scenario_budget.loc[:, hr_expand_scenario_budget.columns[1:]] = \
    hr_expand_scenario_budget.loc[:, hr_expand_scenario_budget.columns[1:]].div(
    hr_expand_scenario_budget.loc[:, hr_expand_scenario_budget.columns[0]], axis='index')
hr_expand_scenario_budget.loc['Total'] = hr_expand_scenario_budget.sum()

# further reduce scenarios
# to examine marginal impact of each cadre, do keep the individual increase (0%, 25%, 50%, 100%) of the four cadres
# to examine combined impact of multiple cadres, do keep the increase of a cadre that is as large as possible
# to do this selection in Excel

# if the resulted 48 scenarios are too many, try reducing the individual increase of each cadre into 4 cases,
# step by 1/3
# if the resulted 29 scenarios are too many, try reducing the individual increase of each cadre into 3 cases,
# step by 50%.
# if try increasing the individual increase of each cadre into 6 cases, a step by 20%


