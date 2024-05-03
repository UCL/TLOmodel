import argparse
from pathlib import Path
from tlo import Date
from collections import Counter, defaultdict

import calendar
import datetime
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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
    create_pickles_locally,
    parse_log_file
)

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define a pathway to the data folder (note: currently outside the TLO model directory)
# remember to set working directory to TLOmodel/
#outputfilepath = Path('./outputs/sakshi.mohan@york.ac.uk')
outputfilepath = Path('./outputs/tbh03@ic.ac.uk')
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"
costing_outputs_folder = Path('./outputs/costing')
if not os.path.exists(costing_outputs_folder):
    os.makedirs(costing_outputs_folder)

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2000, 1, 1), Date(2050, 12, 31))
def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

# %% Gathering basic information
# Find results_folder associated with a given batch_file and get most recent
#results_folder = get_scenario_outputs('example_costing_scenario.py', outputfilepath)[0] # impact_of_cons_regression_scenarios
results_folder = get_scenario_outputs('long_run_all_diseases.py', outputfilepath)[0] # impact_of_cons_regression_scenarios
#results_folder = get_scenario_outputs('scenario_impact_of_consumables_availability.py', outputfilepath)[0] # impact_of_cons_regression_scenarios

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# Load costing resourcefile
workbook_cost = pd.read_excel((resourcefilepath / "costing/ResourceFile_Costing.xlsx"),
                                    sheet_name = None)

# 1. HR cost
# TODO apply attrition rate to the cost calculation  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9237893/#SP1
# 1.1 HR Cost - Financial (Given the staff available)
# Load annual salary by officer type and facility level
hr_cost_parameters = workbook_cost["human_resources"]
hr_annual_salary = hr_cost_parameters[hr_cost_parameters['Parameter_name'] == 'salary_usd']
hr_annual_salary['OfficerType_FacilityLevel'] = 'Officer_Type=' + hr_annual_salary['Officer_Category'].astype(str) + '|Facility_Level=' + hr_annual_salary['Facility_Level'].astype(str)

# Load scenario staffing level
hr_scenario = log[ 'tlo.scenario']['override_parameter']['new_value'][log[ 'tlo.scenario'][ 'override_parameter']['name'] == 'use_funded_or_actual_staffing']

if hr_scenario.empty:
    current_staff_count = pd.read_csv(
        resourcefilepath / "healthsystem/human_resources/actual/ResourceFile_Daily_Capabilities.csv")
else:
    current_staff_count = pd.read_csv(
        resourcefilepath / 'healthsystem'/ 'human_resources' / f'{hr_scenario[2]}' / 'ResourceFile_Daily_Capabilities.csv')

current_staff_count_by_level_and_officer_type = current_staff_count.groupby(['Facility_Level', 'Officer_Category'])[
    'Staff_Count'].sum().reset_index()

# Check if any cadres were not utilised at particular levels of care in the simulation
def expand_capacity_by_officer_type_and_facility_level(_df: pd.Series) -> pd.Series:
    """Summarise the parsed logged-key results for one draw (as dataframe) into a pd.Series."""
    _df = _df.set_axis(_df['date'].dt.year).drop(columns=['date'])
    _df.index.name = 'year'
    return unflatten_flattened_multi_index_in_logging(_df).stack(level=[0, 1])  # expanded flattened axis

extracted_results = extract_results(
    Path(results_folder),
    module='tlo.methods.healthsystem.summary',
    key='Capacity_By_OfficerType_And_FacilityLevel',
    custom_generate_series=expand_capacity_by_officer_type_and_facility_level,
    do_scaling=False,
)

# Mean of results across the runs
summarized_results = summarize(extracted_results, only_mean=True, collapse_columns=True)

# Take mean across the entire simulation
mean_across_simulation = summarized_results.groupby(['OfficerType', 'FacilityLevel']).mean()

# Unstack to make it look like a nice table
cadres_utilisation_rate = mean_across_simulation.reset_index(drop=False)
cadres_utilisation_rate['OfficerType_FacilityLevel'] = 'Officer_Type=' + cadres_utilisation_rate['OfficerType'].astype(str) + '|Facility_Level=' + cadres_utilisation_rate['FacilityLevel'].astype(str)
list_of_cadre_and_level_combinations_used = cadres_utilisation_rate[cadres_utilisation_rate['mean'] != 0]['OfficerType_FacilityLevel']

# Subset scenario staffing level to only include cadre-level combinations used in the simulation
current_staff_count_by_level_and_officer_type['OfficerType_FacilityLevel'] = 'Officer_Type=' + current_staff_count_by_level_and_officer_type['Officer_Category'].astype(str) + '|Facility_Level=' + current_staff_count_by_level_and_officer_type['Facility_Level'].astype(str)
used_staff_count_by_level_and_officer_type = current_staff_count_by_level_and_officer_type[current_staff_count_by_level_and_officer_type['OfficerType_FacilityLevel'].isin(list_of_cadre_and_level_combinations_used)]

# Calculate salary cost for modelled health workforce (Staff count X Annual salary)
salary_for_all_staff = pd.merge(current_staff_count_by_level_and_officer_type[['OfficerType_FacilityLevel', 'Staff_Count']],
                                     hr_annual_salary[['OfficerType_FacilityLevel', 'Value']], on = ['OfficerType_FacilityLevel'], how = "left")
salary_for_all_staff['Total_salary_by_cadre_and_level'] = salary_for_all_staff['Value'] * salary_for_all_staff['Staff_Count']

# Calculate salary cost for current total staff
salary_for_modelled_staff = pd.merge(used_staff_count_by_level_and_officer_type[['OfficerType_FacilityLevel', 'Staff_Count']],
                                     hr_annual_salary[['OfficerType_FacilityLevel', 'Value']], on = ['OfficerType_FacilityLevel'], how = "left")
salary_for_modelled_staff['Total_salary_by_cadre_and_level'] = salary_for_modelled_staff['Value'] * salary_for_modelled_staff['Staff_Count']

# Other costs to maintain a fixed size of health workforce
# TODO consider annual_preservice_training_cost_percapita_usd, annual_preservice_training_cost_percapita_usd, annual_attrition_rate, absorption_rate_of_students_into_public_workforce,
# TODO consider proportion_of_workforce_recruited_from_abroad, recruitment_cost_per_person_recruited_usd, licensure_exam_passing_rate, graduation_rate

# Create a dataframe to store financial costs
scenario_cost_financial = pd.DataFrame({'HR': salary_for_modelled_staff['Total_salary_by_cadre_and_level'].sum()}, index=[0])

# TODO Consider calculating economic cost of HR by multiplying salary times staff count with cadres_utilisation_rate

# 2. Consumables cost
# 2.1 Consumables cost - Financial (What needs to be purchased given what is dispensed)
_df = log['tlo.methods.healthsystem.summary']['Consumables']

counts_of_available = defaultdict(int)
counts_of_not_available = defaultdict(int)
for _, row in _df.iterrows():
    for item, num in row['Item_Available'].items(): # if using 'tlo.methods.healthsystem' eval(row['Item_Available'])
        counts_of_available[item] += num

counts_of_available = defaultdict(int, {int(key): value for key, value in counts_of_available.items()}) # Convert string keys to integer
# for consistency with other dictionaries

# Load consumables cost data
unit_price_consumable = workbook_cost["consumables"]
unit_price_consumable = unit_price_consumable.rename(columns=unit_price_consumable.iloc[0])
unit_price_consumable = unit_price_consumable[['Item_Code', 'Final_price_per_chosen_unit (USD, 2023)']].reset_index(drop=True).iloc[1:]
unit_price_consumable = unit_price_consumable[unit_price_consumable['Item_Code'].notna()]
unit_price_consumable = unit_price_consumable.set_index('Item_Code').to_dict(orient='index')

# Multiply number of items needed by cost of consumable
cost_of_consumables_dispensed = dict(zip(unit_price_consumable, (unit_price_consumable[key]['Final_price_per_chosen_unit (USD, 2023)'] *
                                                counts_of_available[key] for key in unit_price_consumable)))
total_cost_of_consumables_dispensed = sum(value for value in cost_of_consumables_dispensed.values() if not np.isnan(value))

# Cost of consumables stocked
# Estimate the stock to dispensed ratio from OpenLMIS data
lmis_consumable_usage = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")
# Collapse individual facilities
lmis_consumable_usage_by_item_level_month = lmis_consumable_usage.groupby(['category', 'item_code', 'district', 'fac_type_tlo', 'month'])[['closing_bal', 'dispensed', 'received']].sum()
df = lmis_consumable_usage_by_item_level_month # Drop rows where monthly OpenLMIS data wasn't available
df = df.loc[df.index.get_level_values('month') != "Aggregate"]
opening_bal_january = df.loc[df.index.get_level_values('month') == 'January', 'closing_bal'] + \
                      df.loc[df.index.get_level_values('month') == 'January', 'dispensed'] - \
                      df.loc[df.index.get_level_values('month') == 'January', 'received']
closing_bal_december = df.loc[df.index.get_level_values('month') == 'December', 'closing_bal']
total_consumables_inflow_during_the_year = df.loc[df.index.get_level_values('month') != 'January', 'received'].groupby(level=[0,1,2,3]).sum() +\
                                         opening_bal_january.reset_index(level='month', drop=True) -\
                                         closing_bal_december.reset_index(level='month', drop=True)
total_consumables_outflow_during_the_year  = df['dispensed'].groupby(level=[0,1,2,3]).sum()
inflow_to_outflow_ratio = total_consumables_inflow_during_the_year.div(total_consumables_outflow_during_the_year, fill_value=1)

# Edit outlier ratios
inflow_to_outflow_ratio.loc[inflow_to_outflow_ratio < 1] = 1 # Ratio can't be less than 1
inflow_to_outflow_ratio.loc[inflow_to_outflow_ratio > inflow_to_outflow_ratio.quantile(0.95)] = inflow_to_outflow_ratio.quantile(0.95) # Trim values greater than the 95th percentile
average_inflow_to_outflow_ratio_ratio = inflow_to_outflow_ratio.mean()
#inflow_to_outflow_ratio.loc[inflow_to_outflow_ratio.isna()] = average_inflow_to_outflow_ratio_ratio # replace missing with average

# Multiply number of items needed by cost of consumable
inflow_to_outflow_ratio_by_consumable = inflow_to_outflow_ratio.groupby(level='item_code').mean()
inflow_to_outflow_ratio_by_consumable = inflow_to_outflow_ratio_by_consumable.to_dict()
# TODO Consider whether a more disaggregated version of the ratio dictionary should be applied
cost_of_consumables_stocked = dict(zip(unit_price_consumable, (unit_price_consumable[key]['Final_price_per_chosen_unit (USD, 2023)'] *
                                                counts_of_available[key] *
                                                inflow_to_outflow_ratio_by_consumable.get(key, average_inflow_to_outflow_ratio_ratio)
                                                for key in counts_of_available)))
total_cost_of_consumables_stocked = sum(value for value in cost_of_consumables_stocked.values() if not np.isnan(value))

scenario_cost_financial['Consumables'] = total_cost_of_consumables_stocked

# 3. Equipment cost
# Total cost of equipment required as per SEL (HSSP-III) only at facility IDs where it been used in the simulation
unit_cost_equipment = workbook_cost["equipment"]
unit_cost_equipment =   unit_cost_equipment.rename(columns=unit_cost_equipment.iloc[7]).reset_index(drop=True).iloc[8:]
# Calculate necessary costs based on HSSP-III assumptions
unit_cost_equipment['service_fee_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.8 / 8 if row['unit_purchase_cost'] > 1000 else 0, axis=1) # 80% of the value of the item over 8 years
unit_cost_equipment['spare_parts_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.2 / 8 if row['unit_purchase_cost'] > 1000 else 0, axis=1) # 20% of the value of the item over 8 years
unit_cost_equipment['upfront_repair_cost_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.2 * 0.2 / 8 if row['unit_purchase_cost'] < 250000 else 0, axis=1) # 20% of the value of 20% of the items over 8 years
unit_cost_equipment['replacement_cost_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.1 / 8 if row['unit_purchase_cost'] < 250000 else 0, axis=1) # 10% of the items over 8 years

# TODO From the log, extract the facility IDs which use any equipment item
# TODO Collapse facility IDs by level of care to get the total number of facilities at each level using an item
# TODO Multiply number of facilities by level with the quantity needed of each equipment and collapse to get total number of equipment (nationally)
# TODO Multiply quantity needed with cost per item (this is the repair, replacement, and maintenance cost)
# TODO Which equipment needs to be newly purchased (currently no assumption made for equipment with cost > $250,000)


# 4. Facility running costs
# Average running costs by facility level and district times the number of facilities  in the simulation

# Explore the ratio of consumable inflows to outflows
######################################################
# TODO: Only consider the months for which original OpenLMIS data was available for closing_stock and dispensed
def plot_inflow_to_outflow_ratio(_dict, groupby_var):
    # Convert Dict to dataframe
    flattened_data = [(level1, level2, level3, level4, value) for (level1, level2, level3, level4), value in
                      inflow_to_outflow_ratio.items()] # Flatten dictionary into a list of tuples
    _df = pd.DataFrame(flattened_data, columns=['category', 'item_code', 'district', 'fac_type_tlo', 'inflow_to_outflow_ratio']) # Convert flattened data to DataFrame

    # Plot the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=_df , x=groupby_var, y= 'inflow_to_outflow_ratio', errorbar=None)

    # Add points representing the distribution of individual values
    sns.stripplot(data=_df, x=groupby_var, y='inflow_to_outflow_ratio', color='black', size=5, alpha=0.2)

    # Set labels and title
    plt.xlabel(groupby_var)
    plt.ylabel('Inflow to Outflow Ratio')
    plt.title('Average Inflow to Outflow Ratio by ' + f'{groupby_var}')
    plt.xticks(rotation=45)

    # Show plot
    plt.tight_layout()
    plt.savefig(costing_outputs_folder / 'inflow_to_outflow_ratio_by' f'{groupby_var}' )

plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'fac_type_tlo')
plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'district')
plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'item_code')
plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'category')

# Compare financial costs with actual budget data
####################################################
salary_budget_2018 = 69478749
consuambles_budget_2018 = 228934188
real_budget = [salary_budget_2018, consuambles_budget_2018]
model_cost = [scenario_cost_financial['HR'][0], scenario_cost_financial['Consumables'][0]]
labels = ['HR_salary', 'Consumables']

plt.clf()
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
consumables_label = 'Consumables ' + f'{round(model_cost[1] / real_budget[1], 5)}'
plotlabels = [hr_label, consumables_label]
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
salary_for_all_staff[['Officer_Type', 'Facility_Level']] = salary_for_all_staff['OfficerType_FacilityLevel'].str.split('|', expand=True)
salary_for_all_staff['Officer_Type'] = salary_for_all_staff['Officer_Type'].str.replace('Officer_Type=', '')
salary_for_all_staff['Facility_Level'] = salary_for_all_staff['Facility_Level'].str.replace('Facility_Level=', '')
total_salary_by_cadre = salary_for_all_staff.groupby('Officer_Type')['Total_salary_by_cadre_and_level'].sum()
total_salary_by_level = salary_for_all_staff.groupby('Facility_Level')['Total_salary_by_cadre_and_level'].sum()

# Plot by cadre
plt.clf()
total_salary_by_cadre.plot(kind='bar')
plt.xlabel('Officer_category')
plt.ylabel('Total Salary')
plt.title('Total Salary by Cadre')
plt.savefig(costing_outputs_folder /  'total_salary_by_cadre.png')

# Plot by level
plt.clf()
total_salary_by_level.plot(kind='bar')
plt.xlabel('Facility_Level')
plt.ylabel('Total Salary')
plt.title('Total Salary by Facility_Level')
plt.savefig(costing_outputs_folder /  'total_salary_by_level.png')

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
