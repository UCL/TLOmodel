import argparse
from pathlib import Path
from tlo import Date
from collections import Counter, defaultdict

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

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2020, 1, 1), Date(2025, 12, 31))


def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

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
# Load annual salary by officer type and facility level
workbook_cost = pd.read_excel((resourcefilepath / "costing/ResourceFile_Costing.xlsx"),
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

# Calculate salary cost for modelled health workforce (Staff count X Annual salary)
salary_for_modelled_staff = pd.merge(hr_annual_salary, current_staff_count_by_level_and_officer_type, on = ['Officer_Category', 'Facility_Level'])
salary_for_modelled_staff['Total_salary_by_cadre_and_level'] = salary_for_modelled_staff['Salary_USD'] * salary_for_modelled_staff['Staff_Count']

# Create a dataframe to store financial costs
scenario_cost_financial = pd.DataFrame({'HR': salary_for_modelled_staff['Total_salary_by_cadre_and_level'].sum()}, index=[0])

# 1.2 HR Cost - Economic (Staff needed for interventions delivered in the simulation)
# For HR required, multiply above with total capabilities X 'Frac_Time_Used_By_OfficerType' by facility level
frac_time_used_by_officer_type = pd.DataFrame(log['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_By_OfficerType'].to_list())
aggregate_frac_time_used_by_officer_type = pd.DataFrame(frac_time_used_by_officer_type.sum(axis=0))/len(frac_time_used_by_officer_type)
aggregate_frac_time_used_by_officer_type.columns = ['Value']
aggregate_frac_time_used_by_officer_type['OfficerType_FacilityLevel'] = aggregate_frac_time_used_by_officer_type.index

salary_for_required_staff = pd.merge(hr_annual_salary, aggregate_frac_time_used_by_officer_type, on = ['OfficerType_FacilityLevel'])
salary_for_required_staff = pd.merge(salary_for_required_staff, current_staff_count_by_level_and_officer_type, on = ['Officer_Category', 'Facility_Level'])

# Calculate salary cost for required  health workforce (Staff count X Fraction of staff time needed X Annual salary)
salary_for_required_staff['Total_salary_by_cadre_and_level'] = salary_for_required_staff['Salary_USD'] * salary_for_required_staff['Value'] * salary_for_required_staff['Staff_Count']

# Create a dataframe to store economic costs
scenario_cost_economic = pd.DataFrame({'HR': salary_for_required_staff['Total_salary_by_cadre_and_level'].sum()}, index=[0])

# 1. Consumables cost
# 2.1 Consumables cost - Financial (What needs to be purchased given what is made available)
_df = log['tlo.methods.healthsystem']['Consumables']

counts_of_available = defaultdict(int)
counts_of_not_available = defaultdict(int)
for _, row in _df.iterrows():
    for item, num in eval(row['Item_Available']).items():
        counts_of_available[item] += num
    for item, num in eval(row['Item_NotAvailable']).items():
        counts_of_not_available[item] += num
consumables_count_df = pd.concat(
        {'Available': pd.Series(counts_of_available), 'Not_Available': pd.Series(counts_of_not_available)},
        axis=1
    ).fillna(0).astype(int).stack()

# Load consumables cost data
unit_price_consumable = workbook_cost["consumables"][['Item_Code', 'Chosen_price_per_unit (USD)', 'Number of units needed per HSI']]
unit_price_consumable = unit_price_consumable.set_index('Item_Code').to_dict(orient='index')

# Multiply number of items needed by cost of consumable
cost_of_consumables_dispensed = dict(zip(unit_price_consumable, (unit_price_consumable[key]['Chosen_price_per_unit (USD)'] *
                                                unit_price_consumable[key]['Number of units needed per HSI'] *
                                                counts_of_available[key] for key in unit_price_consumable)))
total_cost_of_consumables_dispensed = sum(value for value in cost_of_consumables_dispensed.values() if not np.isnan(value))

# Cost of consumables stocked
# Estimate the stock to dispensed ratio from OpenLMIS data
lmis_consumable_usage = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")
# Collapse by item_code
lmis_consumable_usage_by_item = lmis_consumable_usage.groupby(['item_code'])[['closing_bal', 'amc', 'dispensed', 'received']].sum()
lmis_consumable_usage_by_item['stock_to_dispensed_ratio'] = lmis_consumable_usage_by_item['closing_bal']/lmis_consumable_usage_by_item['dispensed']
#lmis_consumable_usage_by_item = lmis_consumable_usage_by_item[['item_code', 'stock_to_dispensed_ratio']]
# Trim top and bottom 5 percentile value for stock_to_dispensed_ratio
percentile_5 = lmis_consumable_usage_by_item['stock_to_dispensed_ratio'].quantile(0.05)
percentile_95 = lmis_consumable_usage_by_item['stock_to_dispensed_ratio'].quantile(0.95)
lmis_consumable_usage_by_item.loc[lmis_consumable_usage_by_item['stock_to_dispensed_ratio'] > percentile_95, 'stock_to_dispensed_ratio'] = percentile_95
lmis_consumable_usage_by_item.loc[lmis_consumable_usage_by_item['stock_to_dispensed_ratio'] < percentile_5, 'stock_to_dispensed_ratio'] = percentile_5
lmis_stock_to_dispensed_ratio_by_item = lmis_consumable_usage_by_item['stock_to_dispensed_ratio']
lmis_stock_to_dispensed_ratio_by_item.to_dict()
average_stock_to_dispensed_ratio = lmis_stock_to_dispensed_ratio_by_item.mean()


# Multiply number of items needed by cost of consumable
cost_of_consumables_stocked = dict(zip(unit_price_consumable, (unit_price_consumable[key]['Chosen_price_per_unit (USD)'] *
                                                unit_price_consumable[key]['Number of units needed per HSI'] *
                                                counts_of_available[key] *
                                                lmis_stock_to_dispensed_ratio_by_item.get(key, average_stock_to_dispensed_ratio)
                                                for key in counts_of_available)))
total_cost_of_consumables_stocked = sum(value for value in cost_of_consumables_stocked.values() if not np.isnan(value))

scenario_cost_financial['Consumables'] = total_cost_of_consumables_stocked

# Explore the ratio of dispensed drugs to drug stock
####################################################
# Collapse monthly data
lmis_consumable_usage_by_district_and_level = lmis_consumable_usage.groupby(['district', 'fac_type_tlo','category', 'item_code'])[['closing_bal', 'amc', 'dispensed', 'received']].sum()
lmis_consumable_usage_by_district_and_level.reset_index()
lmis_consumable_usage_by_district_and_level['stock_to_dispensed_ratio'] = lmis_consumable_usage_by_district_and_level['closing_bal']/lmis_consumable_usage_by_district_and_level['dispensed']

# TODO: Only consider the months for which original OpenLMIS data was available for closing_stock and dispensed
# TODO Ensure that expected units per case are expected units per HSI
def plot_stock_to_dispensed(_df, plot_var, groupby_var, outlier_percentile):
    # Exclude the top x percentile (outliers) from the plot
    percentile_excluded = _df[plot_var].quantile(outlier_percentile)
    _df_without_outliers = _df[_df[plot_var] <= percentile_excluded]

    # Plot the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=_df_without_outliers, x=groupby_var, y=plot_var, ci=None)

    # Add points representing the distribution of individual values
    sns.stripplot(data=_df_without_outliers, x=groupby_var, y=plot_var, color='black', size=5, alpha=0.2)

    # Set labels and title
    plt.xlabel(groupby_var)
    plt.ylabel('Stock to Dispensed Ratio')
    plt.title('Average Stock to Dispensed Ratio by ' + f'{groupby_var}')
    plt.xticks(rotation=45)

    # Show plot
    plt.tight_layout()
    plt.savefig(costing_outputs_folder / 'stock_to_dispensed_ratio_by' f'{groupby_var}' )

plot_stock_to_dispensed(lmis_consumable_usage_by_district_and_level, 'stock_to_dispensed_ratio',
                        'fac_type_tlo', 0.95)
plot_stock_to_dispensed(lmis_consumable_usage_by_district_and_level, 'stock_to_dispensed_ratio',
                        'district', 0.95)
plot_stock_to_dispensed(lmis_consumable_usage_by_district_and_level, 'stock_to_dispensed_ratio',
                        'category', 0.95)
plot_stock_to_dispensed(lmis_consumable_usage_by_district_and_level, 'stock_to_dispensed_ratio',
                        'item_code', 0.95)


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

# Plot fraction staff time used
fraction_stafftime_average = salary_staffneeded_df.groupby('Officer_Category')['Value'].sum()
fraction_stafftime_average. plot(kind = "bar")
plt.xlabel('Cadre')
plt.ylabel('Fraction time needed')
plt.savefig(costing_outputs_folder /  'hr_time_need_economic_cost.png')

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
