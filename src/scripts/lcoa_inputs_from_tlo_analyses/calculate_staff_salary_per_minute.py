"""
We calculate the salary cost of current and funded plus HCW.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

resourcefilepath = Path('./resources')

mfl = pd.read_csv(resourcefilepath / 'healthsystem' / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')

hr_salary = pd.read_csv(resourcefilepath /
                        'costing' / 'ResourceFile_Annual_Salary_Per_Cadre.csv', index_col=False)
# hr_salary_per_level = pd.read_excel(resourcefilepath /
#                                     'costing' / 'ResourceFile_Costing.xlsx', sheet_name='human_resources')
# as of 2019
hr_current = pd.read_csv(resourcefilepath /
                         'healthsystem' / 'human_resources' / 'actual' / 'ResourceFile_Daily_Capabilities.csv')
hr_established = pd.read_csv(resourcefilepath /
                             'healthsystem' / 'human_resources' / 'funded_plus' / 'ResourceFile_Daily_Capabilities.csv')
# for 2020-2024
historical_scaling = pd.read_csv(resourcefilepath /
                                 'healthsystem' / 'human_resources' / 'scaling_capabilities' /
                                 'ResourceFile_dynamic_HR_scaling' / 'historical_scaling.csv'
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

Minute_Salary = Minute_Salary.fillna(0.0)
Minute_Salary.rename(columns={'Officer_Category': 'Officer_Type_Code'}, inplace=True)

Minute_Salary.to_csv(resourcefilepath / 'costing' / 'ResourceFile_Minute_Salary_HR.csv', index=False)
