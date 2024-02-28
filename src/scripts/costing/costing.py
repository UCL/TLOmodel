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
costing_outputs_folder = Path('./outputs/costing')
if not os.path.exists(costing_outputs_folder):
    os.makedirs(costing_outputs_folder)

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
# 1.1 HR Cost - Financial (Given the staff available)
# Annual salary by officer type and facility level
workbook_cost = pd.read_excel((resourcefilepath / "ResourceFile_Costing.xlsx"),
                                    sheet_name = None)
hr_annual_salary = workbook_cost["human_resources"]
hr_annual_salary['OfficerType_FacilityLevel'] = 'Officer_Type=' + hr_annual_salary['Officer_Category'].astype(str) + '|Facility_Level=' + hr_annual_salary['Facility_Level'].astype(str)

# Load scenario staffing level
hr_scenario = log[ 'tlo.scenario'][ 'override_parameter']['new_value'][log[ 'tlo.scenario'][ 'override_parameter']['name'] == 'use_funded_or_actual_staffing']

if hr_scenario.empty:
    current_staff_count = pd.read_csv(
        resourcefilepath / "healthsystem/human_resources/actual/ResourceFile_Daily_Capabilities.csv")

else:
    current_staff_count = pd.read_csv(
        resourcefilepath / 'healthsystem'/ 'human_resources' / f'{hr_scenario}' / 'ResourceFile_Daily_Capabilities.csv')

    current_staff_count_by_level_and_officer_type = current_staff_count.groupby(['Facility_Level', 'Officer_Category'])[
        'Staff_Count'].sum().reset_index()

salary_actualstaff_df = pd.merge(hr_annual_salary, current_staff_count_by_level_and_officer_type, on = ['Officer_Category', 'Facility_Level'])
salary_actualstaff_df['Total_salary_by_cadre_and_level'] = salary_actualstaff_df['Salary_USD'] * salary_actualstaff_df['Staff_Count']

# Create a dataframe to store financial costs
scenario_cost_financial = pd.DataFrame({'HR': salary_actualstaff_df['Total_salary_by_cadre_and_level'].sum()}, index=[0])

# 1.2 HR Cost - Economic (Staff needed for interventions delivered in the simulation)
# For total HR cost, multiply above with total capabilities X 'Frac_Time_Used_By_OfficerType' by facility leve
# Use log['tlo.methods.population']['scaling_factor']
frac_time_used_by_officer_type = pd.DataFrame(log['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_By_OfficerType'].to_list())
aggregate_frac_time_used_by_officer_type = pd.DataFrame(frac_time_used_by_officer_type.sum(axis=0))
aggregate_frac_time_used_by_officer_type.columns = ['Value']
aggregate_frac_time_used_by_officer_type['OfficerType_FacilityLevel'] = aggregate_frac_time_used_by_officer_type.index

salary_staffneeded_df = pd.merge(hr_annual_salary, aggregate_frac_time_used_by_officer_type, on = ['OfficerType_FacilityLevel'])
salary_staffneeded_df['Total_salary_by_cadre_and_level'] = salary_df['Salary_USD'] * salary_df['Value']

# Create a dataframe to store economic costs
scenario_cost_economic = pd.DataFrame({'HR': salary_staffneeded_df['Total_salary_by_cadre_and_level'].sum()}, index=[0])

# Compare financial costs with actual budget data
####################################################
salary_budget_2018 = 69478749
consuambles_budget_2018 = 228934188
real_budget = [salary_budget_2018, consuambles_budget_2018]
model_cost = [scenario_cost_financial['HR'][0], 0]
labels = ['HR_salary', 'Consumables']

plt.scatter(real_budget, model_cost)
# Plot a line representing a 45-degree angle
min_val = min(min(real_budget), min(model_cost))
max_val = max(max(real_budget), max(model_cost))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='45-degree line')

# Format x and y axis labels to display in millions
formatter = FuncFormatter(lambda x, _: '{:,.0f}M'.format(x / 1e6))
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
# Add labels for each point
hr_label = 'HR_salary ' + f'{round(model_cost[0] / real_budget[0], 2)}'
plotlabels = [hr_label, 'Consumables']
for i, txt in enumerate(plotlabels):
    plt.text(real_budget[i], model_cost[i], txt, ha='right')

plt.xlabel('Real Budget')
plt.ylabel('Model Cost')
plt.title('Real Budget vs Model Cost')
plt.savefig(costing_outputs_folder /  'Cost_validation.png')


# Plot salary costs by cadre and facility level
# Group by cadre and level
total_salary_by_cadre = salary_df.groupby('Officer_Category')['Total_salary_by_cadre_and_level'].sum()
total_salary_by_level = salary_df.groupby('Facility_Level')['Total_salary_by_cadre_and_level'].sum()

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
# TODO Add scaling factor

# Consumables
log['tlo.methods.healthsystem']['Consumables']
# Aggregate Items_Available by Treatment_ID
# Multiply by the cost per item (need to check quantity)

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

