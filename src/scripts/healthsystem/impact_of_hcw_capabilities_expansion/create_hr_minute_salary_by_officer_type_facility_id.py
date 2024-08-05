"""
We calculate the salar cost of current and funded plus HCW.
"""

from pathlib import Path

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
four_cadres_cost['cost_frac'] = four_cadres_cost['annual_cost'] / four_cadres_cost['annual_cost'].sum()
assert four_cadres_cost.cost_frac.sum() == 1
