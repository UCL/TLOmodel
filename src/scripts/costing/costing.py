import argparse
from pathlib import Path
import calendar
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    summarize,
)

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define a pathway to the data folder (note: currently outside the TLO model directory)
# remember to set working directory to TLOmodel/
outputfilepath = Path('./outputs/sakshi.mohan@york.ac.uk')
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"

# %% Gathering basic information

# Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('example_costing_scenario.py', outputfilepath)[0] # impact_of_cons_regression_scenarios

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


# 1. HR cost
# 1.1 Overall HR Cost
# Annual salary by officer type and facility level
workbook = pd.read_excel((resourcefilepath / "ResourceFile_Costing.xlsx"),
                                    sheet_name = None)
hr_annual_salary = workbook["human_resources"]
hr_annual_salary['OfficerType_FacilityLevel'] = 'Officer_Type=' + hr_annual_salary['Officer_Category'].astype(str) + '|Facility_Level=' + hr_annual_salary['Facility_Level'].astype(str)

# For total HR cost, multiply above with total capabilities X 'Frac_Time_Used_By_OfficerType' by facility level
frac_time_used_by_officer_type = pd.DataFrame(log['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_By_OfficerType'].to_list())
aggregate_frac_time_used_by_officer_type = pd.DataFrame(frac_time_used_by_officer_type.sum(axis=0))
aggregate_frac_time_used_by_officer_type.columns = ['Value']
aggregate_frac_time_used_by_officer_type['OfficerType_FacilityLevel'] = aggregate_frac_time_used_by_officer_type.index

salary_df = pd.merge(hr_annual_salary, aggregate_frac_time_used_by_officer_type, on = ['OfficerType_FacilityLevel'])
salary_df['Total_salary_by_cadre_and_level'] = salary_df['Salary_USD'] * salary_df['Value']
scenario_cost = pd.DataFrame()
scenario_cost['HR'] = salary_df['Total_salary_by_cadre_and_level'].sum()

# Plot salary costs by cadre and facility level
# Group by cadre and level
total_salary_by_cadre = salary_df.groupby('Officer_Category')['Total_salary_by_cadre_and_level'].sum()
total_salary_by_level = salary_df.groupby('Facility_Level')['Total_salary_by_cadre_and_level'].sum()

# If the folder doesn't exist, create it
costing_outputs_folder = Path('./outputs/costing')
if not os.path.exists(costing_outputs_folder):
    os.makedirs(costing_outputs_folder)

# Plot by cadre
total_salary_by_cadre.plot(kind='bar')
plt.xlabel('Officer_category')
plt.ylabel('Total Salary')
plt.title('Total Salary by Cadre')
plt.savefig(costing_outputs_folder /  'total_salary_by_cadre.png')

# Plot by level
total_salary_by_level.plot(kind='bar')
plt.xlabel('Facility_Level')
plt.ylabel('Total Salary')
plt.title('Total Salary by Facility_Level')
plt.savefig(costing_outputs_folder /  'total_salary_by_level.png')

# TODO Disaggregate by district using 'Frac_Time_Used_By_Facility_ID'
# TODO Disaggregate by Treatment_ID - will need this for cost-effectiveness estimates - current log does not provide this



'''
# Scratch pad

log['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_By_Facility_ID'] # for district disaggregation

# Aggregate Daily capabilities to total used by cadre and facility level

# log['tlo.methods.healthsystem.summary']['Capacity']['Frac_Time_Used_By_OfficerType']
# 1.2 HR cost by Treatment_ID
# For HR cost by Treatment_ID, multiply total cost by Officer type by fraction of time used for treatment_ID
log['tlo.methods.healthsystem.summary']['HSI_Event']['TREATMENT_ID'] # what does this represent? why are there 3 rows (2 scenarios)
# But what we need is the HR use by Treatment_ID  - Leave this for later?

# log['tlo.scenario']
log['tlo.methods.healthsystem.summary']['HSI_Event']['Number_By_Appt_Type_Code']


df = pd.DataFrame(log['tlo.methods.healthsystem.summary'])
df.to_csv(outputfilepath / 'temp.csv')

def read_parameters(self, data_folder):
    """
    1. Reads the costing resource file
    2. Declares the costing parameters
    """
    # Read the resourcefile
    # Short cut to parameters dict
    p = self.parameters

    workbook = pd.read_excel((resourcefilepath / "ResourceFile_Costing.xlsx"),
                                    sheet_name = None)

    p["human_resources"] = workbook["human_resources"]

workbook = pd.read_excel((resourcefilepath / "ResourceFile_Costing.xlsx"),
                                    sheet_name = None)
human_resources = workbook["human_resources"]

'''

