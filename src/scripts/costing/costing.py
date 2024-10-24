import argparse
from pathlib import Path
from tlo import Date
from collections import Counter, defaultdict

import calendar
import datetime
import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import ast
import math

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
    parse_log_file,
    unflatten_flattened_multi_index_in_logging
)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Define a pathway to relevant folders
outputfilepath = Path('./outputs/t.mangal@imperial.ac.uk')
resourcefilepath = Path("./resources")
path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"
costing_outputs_folder = Path('./outputs/costing')
if not os.path.exists(costing_outputs_folder):
    os.makedirs(costing_outputs_folder)
figurespath = costing_outputs_folder / "figures"
if not os.path.exists(figurespath):
    os.makedirs(figurespath)

# Useful common functions
def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

def melt_model_output_draws_and_runs(_df, id_vars):
    multi_index = pd.MultiIndex.from_tuples(_df.columns)
    _df.columns = multi_index
    melted_df = pd.melt(_df, id_vars=id_vars).rename(columns = {'variable_0': 'draw', 'variable_1': 'run'})
    return melted_df


# %% Gathering basic information
# Load result files
#-------------------
#results_folder = get_scenario_outputs('example_costing_scenario.py', outputfilepath)[0]
#results_folder = get_scenario_outputs('long_run_all_diseases.py', outputfilepath)[0]
#results_folder = get_scenario_outputs('scenario_impact_of_consumables_availability.py', outputfilepath)[0] # impact_of_cons_regression_scenarios

#results_folder = get_scenario_outputs('htm_with_and_without_hss-2024-09-04T143044Z.py', outputfilepath)[0] # Tara's FCDO/GF scenarios version 1
#results_folder = get_scenario_outputs('hss_elements-2024-09-04T142900Z.py', outputfilepath)[0] # Tara's FCDO/GF scenarios version 1
results_folder = get_scenario_outputs('htm_with_and_without_hss-2024-10-12T111720Z.py', outputfilepath)[0] # Tara's FCDO/GF scenarios version 2
#results_folder = get_scenario_outputs('hss_elements-2024-10-12T111649Z.py', outputfilepath)[0] # Tara's FCDO/GF scenarios version 2

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0) # look at one log (so can decide what to extract)
info = get_scenario_info(results_folder) # get basic information about the results
# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
final_year_of_simulation = max(log['tlo.methods.healthsystem.summary']['hsi_event_counts']['date']).year
first_year_of_simulation = min(log['tlo.methods.healthsystem.summary']['hsi_event_counts']['date']).year
population_scaling_factor = log['tlo.methods.demography']['scaling_factor']['scaling_factor'].iloc[0]


# Load basic simulation parameters
#-------------------------------------
draws = range(0, info['number_of_draws'])
runs = range(0, info['runs_per_draw'])
years = list(range(first_year_of_simulation, final_year_of_simulation + 1))

# Load cost input files
#------------------------
# Load primary costing resourcefile
workbook_cost = pd.read_excel((resourcefilepath / "costing/ResourceFile_Costing.xlsx"),
                                    sheet_name = None)

# Extract districts and facility levels from the Master Facility List
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
district_dict = pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')[['District_Num', 'District']].drop_duplicates()
district_dict = dict(zip(district_dict['District_Num'], district_dict['District']))
facility_id_levels_dict = dict(zip(mfl['Facility_ID'], mfl['Facility_Level']))
fac_levels = set(mfl.Facility_Level)

# Overall cost assumptions
TARGET_PERIOD = (Date(2010, 1, 1), Date(2030, 12, 31)) # Declare period for which the results will be generated (defined inclusively)
discount_rate = 0.03

# Read all cost parameters
#---------------------------------------
# Read parameters for HR costs
hr_cost_parameters = workbook_cost["human_resources"]
hr_cost_parameters['Facility_Level'] =  hr_cost_parameters['Facility_Level'].astype(str) # Store Facility_Level as string

# Read parameters for consumables costs
# Load consumables cost data
unit_price_consumable = workbook_cost["consumables"]
unit_price_consumable = unit_price_consumable.rename(columns=unit_price_consumable.iloc[0])
unit_price_consumable = unit_price_consumable[['Item_Code', 'Final_price_per_chosen_unit (USD, 2023)']].reset_index(drop=True).iloc[1:]
unit_price_consumable = unit_price_consumable[unit_price_consumable['Item_Code'].notna()]

# Load and prepare equipment cost parameters
# Unit costs of equipment
unit_cost_equipment = workbook_cost["equipment"]
unit_cost_equipment = unit_cost_equipment.rename(columns=unit_cost_equipment.iloc[7]).reset_index(drop=True).iloc[8:]
unit_cost_equipment = unit_cost_equipment[unit_cost_equipment['Item_code'].notna()] # drop empty row
# Calculate necessary costs based on HSSP-III assumptions
unit_cost_equipment['replacement_cost_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost']/(1+(1-(1+discount_rate)**(-row['Life span']+1))/discount_rate), axis=1) # Annuitised over the life span of the equipment assuming outlay at the beginning of the year
unit_cost_equipment['service_fee_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.8 / 8 if row['unit_purchase_cost'] > 1000 else 0, axis=1) # 80% of the value of the item over 8 years
unit_cost_equipment['spare_parts_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.2 / 8 if row['unit_purchase_cost'] > 1000 else 0, axis=1) # 20% of the value of the item over 8 years
unit_cost_equipment['major_corrective_maintenance_cost_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.2 * 0.2 / 8 if row['unit_purchase_cost'] < 250000 else 0, axis=1) # 20% of the value of 20% of the items over 8 years
# TODO consider discounting the other components
# Quantity needed for each equipment by facility
unit_cost_equipment = unit_cost_equipment[['Item_code','Equipment_tlo',
                                           'replacement_cost_annual', 'service_fee_annual', 'spare_parts_annual',  'major_corrective_maintenance_cost_annual',
                                           'Health Post_prioritised', 'Community_prioritised', 'Health Center_prioritised', 'District_prioritised', 'Central_prioritised']]
unit_cost_equipment = unit_cost_equipment.rename(columns={col: 'Quantity_' + col.replace('_prioritised', '') for col in unit_cost_equipment.columns if col.endswith('_prioritised')})
unit_cost_equipment = unit_cost_equipment.rename(columns={col: col.replace(' ', '_') for col in unit_cost_equipment.columns})

unit_cost_equipment = pd.wide_to_long(unit_cost_equipment, stubnames=['Quantity_'],
                          i=['Item_code', 'Equipment_tlo', 'replacement_cost_annual', 'service_fee_annual', 'spare_parts_annual', 'major_corrective_maintenance_cost_annual'],
                          j='Facility_Level', suffix='(\d+|\w+)').reset_index()
facility_level_mapping = {'Health_Post': '0', 'Health_Center': '1a', 'Community': '1b', 'District': '2', 'Central': '3'}
unit_cost_equipment['Facility_Level'] = unit_cost_equipment['Facility_Level'].replace(facility_level_mapping)
unit_cost_equipment = unit_cost_equipment.rename(columns = {'Quantity_': 'Quantity'})

# Function to prepare cost dataframe ready to be merged across cross categories
def retain_relevant_column_subset(_df, _category_specific_group):
    columns_to_retain = ['draw', 'run', 'year', 'cost_subcategory', 'Facility_Level', _category_specific_group, 'cost']
    _df = _df[columns_to_retain]
    return _df
def prepare_cost_dataframe(_df, _category_specific_group, _cost_category):
    _df = _df.rename(columns = {_category_specific_group: 'cost_subgroup'})
    _df['cost_category'] = _cost_category
    return retain_relevant_column_subset(_df, 'cost_subgroup')


# CALCULATE FINANCIAL COSTS
#%%
# 1. HR cost
#------------------------
# Define a function to merge unit cost data with model outputs
def merge_cost_and_model_data(cost_df, model_df, varnames):
    merged_df = model_df.copy()
    for varname in varnames:
        new_cost_df = cost_df[cost_df['Parameter_name'] == varname][['OfficerType', 'Facility_Level', 'Value']]
        new_cost_df = new_cost_df.rename(columns={"Value": varname})
        if ((new_cost_df['OfficerType'] == 'All').all()) and ((new_cost_df['Facility_Level'] == 'All').all()):
            merged_df[varname] = new_cost_df[varname].mean()
        elif ((new_cost_df['OfficerType'] == 'All').all()) and ((new_cost_df['Facility_Level'] == 'All').all() == False):
            merged_df = pd.merge(merged_df, new_cost_df[['Facility_Level',varname]], on=['Facility_Level'], how="left")
        elif ((new_cost_df['OfficerType'] == 'All').all() == False) and ((new_cost_df['Facility_Level'] == 'All').all()):
            merged_df = pd.merge(merged_df, new_cost_df[['OfficerType',varname]], on=['OfficerType'], how="left")
        else:
            merged_df = pd.merge(merged_df, new_cost_df, on=['OfficerType', 'Facility_Level'], how="left")
    return merged_df

# Get available staff count for each year and draw
def get_staff_count_by_facid_and_officer_type(_df: pd.Series) -> pd.Series:
    """Summarise the parsed logged-key results for one draw (as dataframe) into a pd.Series."""
    _df = _df.set_axis(_df['date'].dt.year).drop(columns=['date'])
    _df.index.name = 'year'

    def change_to_standard_flattened_index_format(col):
        parts = col.split("_", 3)  # Split by "_" only up to 3 parts
        if len(parts) > 2:
            return parts[0] + "=" + parts[1] + "|" + parts[2] + "=" + parts[3]  # Rejoin with "I" at the second occurrence
        return col  # If there's no second underscore, return the string as it is
    _df.columns = [change_to_standard_flattened_index_format(col) for col in _df.columns]

    return unflatten_flattened_multi_index_in_logging(_df).stack(level=[0, 1])  # expanded flattened axis

# Staff count by Facility ID
available_staff_count_by_facid_and_officertype = extract_results(
    Path(results_folder),
    module='tlo.methods.healthsystem.summary',
    key='number_of_hcw_staff',
    custom_generate_series=get_staff_count_by_facid_and_officer_type,
    do_scaling=True,
)

# Update above series to get staff count by Facility_Level
available_staff_count_by_facid_and_officertype = available_staff_count_by_facid_and_officertype.reset_index().rename(columns= {'FacilityID': 'Facility_ID', 'Officer': 'OfficerType'})
available_staff_count_by_facid_and_officertype['Facility_ID'] = pd.to_numeric(available_staff_count_by_facid_and_officertype['Facility_ID'])
available_staff_count_by_facid_and_officertype['Facility_Level'] = available_staff_count_by_facid_and_officertype['Facility_ID'].map(facility_id_levels_dict)
idx = pd.IndexSlice
available_staff_count_by_level_and_officer_type = available_staff_count_by_facid_and_officertype.drop(columns = [idx['Facility_ID']]).groupby([idx['year'], idx['Facility_Level'], idx['OfficerType']]).sum()
available_staff_count_by_level_and_officer_type = melt_model_output_draws_and_runs(available_staff_count_by_level_and_officer_type.reset_index(), id_vars= ['year', 'Facility_Level', 'OfficerType'])
available_staff_count_by_level_and_officer_type['Facility_Level'] = available_staff_count_by_level_and_officer_type['Facility_Level'].astype(str) # make sure facility level is stored as string
available_staff_count_by_level_and_officer_type = available_staff_count_by_level_and_officer_type.drop(available_staff_count_by_level_and_officer_type[available_staff_count_by_level_and_officer_type['Facility_Level'] == '5'].index) # drop headquarters because we're only concerned with staff engaged in service delivery
available_staff_count_by_level_and_officer_type.rename(columns ={'value': 'staff_count'}, inplace=True)

# Get list of cadres which were utilised in each run to get the count of staff used in the simulation
# Note that we still cost the full staff count for any cadre-Facility_Level combination that was ever used in a run, and
# not the amount of time which was used
def get_capacity_used_by_officer_type_and_facility_level(_df: pd.Series) -> pd.Series:
    """Summarise the parsed logged-key results for one draw (as dataframe) into a pd.Series."""
    _df = _df.set_axis(_df['date'].dt.year).drop(columns=['date'])
    _df.index.name = 'year'
    return unflatten_flattened_multi_index_in_logging(_df).stack(level=[0, 1])  # expanded flattened axis

annual_capacity_used_by_cadre_and_level = extract_results(
    Path(results_folder),
    module='tlo.methods.healthsystem.summary',
    key='Capacity_By_OfficerType_And_FacilityLevel',
    custom_generate_series=get_capacity_used_by_officer_type_and_facility_level,
    do_scaling=False,
)

# Prepare capacity used dataframe to be multiplied by staff count
average_capacity_used_by_cadre_and_level = annual_capacity_used_by_cadre_and_level.groupby(['OfficerType', 'FacilityLevel']).mean().reset_index(drop=False)
# TODO see if cadre-level combinations should be chosen by year
average_capacity_used_by_cadre_and_level.reset_index(drop=True) # Flatten multi=index column
average_capacity_used_by_cadre_and_level = average_capacity_used_by_cadre_and_level.melt(id_vars=['OfficerType', 'FacilityLevel'],
                        var_name=['draw', 'run'],
                        value_name='capacity_used')
list_of_cadre_and_level_combinations_used = average_capacity_used_by_cadre_and_level[average_capacity_used_by_cadre_and_level['capacity_used'] != 0][['OfficerType', 'FacilityLevel', 'draw', 'run']]
print(f"Out of {average_capacity_used_by_cadre_and_level.groupby(['OfficerType', 'FacilityLevel']).size().count()} cadre and level combinations available, {list_of_cadre_and_level_combinations_used.groupby(['OfficerType', 'FacilityLevel']).size().count()} are used across the simulations")
list_of_cadre_and_level_combinations_used = list_of_cadre_and_level_combinations_used.rename(columns = {'FacilityLevel':'Facility_Level'})

# Subset scenario staffing level to only include cadre-level combinations used in the simulation
used_staff_count_by_level_and_officer_type = available_staff_count_by_level_and_officer_type.merge(list_of_cadre_and_level_combinations_used, on = ['draw','run','OfficerType', 'Facility_Level'], how = 'right', validate = 'm:m')
used_staff_count_by_level_and_officer_type.rename(columns ={'value': 'staff_count'}, inplace=True)

# Calculate various components of HR cost
# 1.1 Salary cost for current total staff
#---------------------------------------------------------------------------------------------------------------
salary_for_all_staff = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = available_staff_count_by_level_and_officer_type,
                                                     varnames = ['salary_usd'])
salary_for_all_staff['cost'] = salary_for_all_staff['salary_usd'] * salary_for_all_staff['staff_count']

# 1.2 Salary cost for health workforce cadres used in the simulation (Staff count X Annual salary)
#---------------------------------------------------------------------------------------------------------------
salary_for_staff_used_in_scenario = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = used_staff_count_by_level_and_officer_type,
                                                     varnames = ['salary_usd'])
salary_for_staff_used_in_scenario['cost'] = salary_for_staff_used_in_scenario['salary_usd'] * salary_for_staff_used_in_scenario['staff_count']
# summarize(salary_for_staff_used_in_scenario, only_mean = True, collapse_columns=True)
#.set_index(['draw', 'run', 'year', 'OfficerType', 'Facility_Level']).unstack(level=['draw', 'run'])
#salary_for_staff_used_in_scenario = salary_for_staff_used_in_scenario.apply(lambda x: pd.to_numeric(x, errors='coerce'))

# 1.3 Recruitment cost to fill gap created by attrition
#---------------------------------------------------------------------------------------------------------------
recruitment_cost = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = used_staff_count_by_level_and_officer_type,
                                                     varnames = ['annual_attrition_rate', 'recruitment_cost_per_person_recruited_usd'])
recruitment_cost['cost'] = recruitment_cost['annual_attrition_rate'] * recruitment_cost['staff_count'] * \
                      recruitment_cost['recruitment_cost_per_person_recruited_usd']
recruitment_cost = recruitment_cost[['draw', 'run', 'year', 'Facility_Level', 'OfficerType', 'cost']]

# 1.4 Pre-service training cost to fill gap created by attrition
#---------------------------------------------------------------------------------------------------------------
preservice_training_cost = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = used_staff_count_by_level_and_officer_type,
                                                     varnames = ['annual_attrition_rate',
                                                                 'licensure_exam_passing_rate', 'graduation_rate',
                                                                 'absorption_rate_of_students_into_public_workforce', 'proportion_of_workforce_recruited_from_abroad',
                                                                 'preservice_training_cost_per_staff_recruited_usd'])
preservice_training_cost['Annual_cost_per_staff_recruited'] = preservice_training_cost['preservice_training_cost_per_staff_recruited_usd'] *\
                                                (1/(preservice_training_cost['absorption_rate_of_students_into_public_workforce'] + preservice_training_cost['proportion_of_workforce_recruited_from_abroad'])) *\
                                                (1/preservice_training_cost['graduation_rate']) * (1/preservice_training_cost['licensure_exam_passing_rate']) *\
                                                preservice_training_cost['annual_attrition_rate']
# Cost per student trained * 1/Rate of absorption from the local and foreign graduates * 1/Graduation rate * attrition rate
# the inverse of attrition rate is the average expected tenure; and the preservice training cost needs to be divided by the average tenure
preservice_training_cost['cost'] = preservice_training_cost['Annual_cost_per_staff_recruited'] * preservice_training_cost['staff_count'] # not multiplied with attrition rate again because this is already factored into 'Annual_cost_per_staff_recruited'
preservice_training_cost = preservice_training_cost[['draw', 'run', 'year', 'OfficerType', 'Facility_Level', 'cost']]

# 1.5 In-service training cost to train all staff
#---------------------------------------------------------------------------------------------------------------
inservice_training_cost = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = used_staff_count_by_level_and_officer_type,
                                                     varnames = ['annual_inservice_training_cost_usd'])
inservice_training_cost['cost'] = inservice_training_cost['staff_count'] * inservice_training_cost['annual_inservice_training_cost_usd']
inservice_training_cost = inservice_training_cost[['draw', 'run', 'year', 'OfficerType', 'Facility_Level', 'cost']]
# TODO Consider calculating economic cost of HR by multiplying salary times staff count with cadres_utilisation_rate

# 1.6 Store all HR costs in one standard format dataframe
#---------------------------------------------------------------------------------------------------------------
# Function to melt and label the cost category
def label_rows_of_cost_dataframe(_df, label_var, label):
    _df = _df.reset_index()
    _df[label_var] = label
    return _df

# Initialize HR with the salary data
human_resource_costs = retain_relevant_column_subset(label_rows_of_cost_dataframe(salary_for_staff_used_in_scenario, 'cost_subcategory', 'salary_for_used_cadres'), 'OfficerType')

# Concatenate additional cost categories
additional_costs = [
    (recruitment_cost , 'recruitment_cost_for_attrited_workers'),
    (preservice_training_cost, 'preservice_training_cost_for_attrited_workers'),
    (inservice_training_cost, 'inservice_training_cost_for_all_staff')
]
# Iterate through additional costs, melt and concatenate
for df, label in additional_costs:
    labelled_df = retain_relevant_column_subset(label_rows_of_cost_dataframe(df, 'cost_subcategory', label), 'OfficerType')
    human_resource_costs = pd.concat([human_resource_costs, labelled_df])

human_resource_costs = prepare_cost_dataframe(human_resource_costs, _category_specific_group = 'OfficerType', _cost_category = 'human resources for health')

# %%
# 2. Consumables cost
#------------------------
def get_quantity_of_consumables_dispensed(results_folder):
    def get_counts_of_items_requested(_df):
        _df = drop_outside_period(_df)
        counts_of_available = defaultdict(lambda: defaultdict(int))
        counts_of_not_available = defaultdict(lambda: defaultdict(int))

        for _, row in _df.iterrows():
            date = row['date']
            for item, num in row['Item_Available'].items():
                counts_of_available[date][item] += num
            for item, num in row['Item_NotAvailable'].items():
                counts_of_not_available[date][item] += num
        available_df = pd.DataFrame(counts_of_available).fillna(0).astype(int).stack().rename('Available')
        not_available_df = pd.DataFrame(counts_of_not_available).fillna(0).astype(int).stack().rename('Not_Available')

        # Combine the two dataframes into one series with MultiIndex (date, item, availability_status)
        combined_df = pd.concat([available_df, not_available_df], axis=1).fillna(0).astype(int)

        # Convert to a pd.Series, as expected by the custom_generate_series function
        return combined_df.stack()

    cons_req = extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='Consumables',
            custom_generate_series=get_counts_of_items_requested,
            do_scaling=True)

    cons_dispensed = cons_req.xs("Available", level=2) # only keep actual dispensed amount, i.e. when available
    return cons_dispensed
# TODO Extract year of dispensing drugs

consumables_dispensed = get_quantity_of_consumables_dispensed(results_folder)
consumables_dispensed = consumables_dispensed.reset_index().rename(columns = {'level_0': 'Item_Code', 'level_1': 'year'})
consumables_dispensed[idx['year']] = pd.to_datetime(consumables_dispensed[idx['year']]).dt.year # Extract only year from date
consumables_dispensed[idx['Item_Code']] = pd.to_numeric(consumables_dispensed[idx['Item_Code']])
# Make a list of columns in the DataFrame pertaining to quantity dispensed
quantity_columns = consumables_dispensed.columns.to_list()
quantity_columns = [tup for tup in quantity_columns if tup not in [('Item_Code', ''), ('year', '')]]

# 2.1 Cost of consumables dispensed
#---------------------------------------------------------------------------------------------------------------
# Multiply number of items needed by cost of consumable
#consumables_dispensed.columns = consumables_dispensed.columns.get_level_values(0).str() + "_" + consumables_dispensed.columns.get_level_values(1) # Flatten multi-level columns for pandas merge
unit_price_consumable.columns = pd.MultiIndex.from_arrays([unit_price_consumable.columns, [''] * len(unit_price_consumable.columns)])
cost_of_consumables_dispensed = consumables_dispensed.merge(unit_price_consumable, on = idx['Item_Code'], validate = 'm:1', how = 'left')
price_column = 'Final_price_per_chosen_unit (USD, 2023)'
cost_of_consumables_dispensed[quantity_columns] = cost_of_consumables_dispensed[quantity_columns].multiply(
    cost_of_consumables_dispensed[price_column], axis=0)

# 2.2 Cost of consumables stocked (quantity needed for what is dispensed)
#---------------------------------------------------------------------------------------------------------------
# Stocked amount should be higher than dispensed because of i. excess capacity, ii. theft, iii. expiry
# While there are estimates in the literature of what % these might be, we agreed that it is better to rely upon
# an empirical estimate based on OpenLMIS data
# Estimate the stock to dispensed ratio from OpenLMIS data
lmis_consumable_usage = pd.read_csv(path_for_consumable_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")
# TODO Generate a smaller version of this file
# Collapse individual facilities
lmis_consumable_usage_by_item_level_month = lmis_consumable_usage.groupby(['category', 'item_code', 'district', 'fac_type_tlo', 'month'])[['closing_bal', 'dispensed', 'received']].sum()
df = lmis_consumable_usage_by_item_level_month # Drop rows where monthly OpenLMIS data wasn't available
df = df.loc[df.index.get_level_values('month') != "Aggregate"]
# Opening balance in January is the closing balance for the month minus what was received during the month plus what was dispensed
opening_bal_january = df.loc[df.index.get_level_values('month') == 'January', 'closing_bal'] + \
                      df.loc[df.index.get_level_values('month') == 'January', 'dispensed'] - \
                      df.loc[df.index.get_level_values('month') == 'January', 'received']
closing_bal_december = df.loc[df.index.get_level_values('month') == 'December', 'closing_bal']
# the consumable inflow during the year is the opening balance in January + what was received throughout the year - what was transferred to the next year (i.e. closing bal of December)
total_consumables_inflow_during_the_year = df.loc[df.index.get_level_values('month') != 'January', 'received'].groupby(level=[0,1,2,3]).sum() +\
                                         opening_bal_january.reset_index(level='month', drop=True) -\
                                         closing_bal_december.reset_index(level='month', drop=True)
total_consumables_outflow_during_the_year  = df['dispensed'].groupby(level=[0,1,2,3]).sum()
inflow_to_outflow_ratio = total_consumables_inflow_during_the_year.div(total_consumables_outflow_during_the_year, fill_value=1)

# Edit outlier ratios
inflow_to_outflow_ratio.loc[inflow_to_outflow_ratio < 1] = 1 # Ratio can't be less than 1
inflow_to_outflow_ratio.loc[inflow_to_outflow_ratio > inflow_to_outflow_ratio.quantile(0.95)] = inflow_to_outflow_ratio.quantile(0.95) # Trim values greater than the 95th percentile
average_inflow_to_outflow_ratio_ratio = inflow_to_outflow_ratio.mean() # Use average where item-specific ratio is not available

# Multiply number of items needed by cost of consumable
inflow_to_outflow_ratio_by_consumable = inflow_to_outflow_ratio.groupby(level='item_code').mean()
excess_stock_ratio = inflow_to_outflow_ratio_by_consumable - 1
excess_stock_ratio = excess_stock_ratio.reset_index().rename(columns = {0: 'excess_stock_proportion_of_dispensed'})
# TODO Consider whether a more disaggregated version of the ratio dictionary should be applied
cost_of_excess_consumables_stocked = consumables_dispensed.merge(unit_price_consumable, left_on = 'Item_Code', right_on = 'Item_Code', validate = 'm:1', how = 'left')
excess_stock_ratio.columns = pd.MultiIndex.from_arrays([excess_stock_ratio.columns, [''] * len(excess_stock_ratio.columns)]) # TODO convert this into a funciton
cost_of_excess_consumables_stocked = cost_of_excess_consumables_stocked.merge(excess_stock_ratio, left_on = 'Item_Code', right_on = 'item_code', validate = 'm:1', how = 'left')
cost_of_excess_consumables_stocked.loc[cost_of_excess_consumables_stocked.excess_stock_proportion_of_dispensed.isna(), 'excess_stock_proportion_of_dispensed'] = average_inflow_to_outflow_ratio_ratio - 1# TODO disaggregate the average by program
cost_of_excess_consumables_stocked[quantity_columns] = cost_of_excess_consumables_stocked[quantity_columns].multiply(cost_of_excess_consumables_stocked[idx[price_column]], axis=0)
cost_of_excess_consumables_stocked[quantity_columns] = cost_of_excess_consumables_stocked[quantity_columns].multiply(cost_of_excess_consumables_stocked[idx['excess_stock_proportion_of_dispensed']], axis=0)

# 2.3 Store all HR costs in one standard format dataframe
#---------------------------------------------------------------------------------------------------------------
# Function to melt and label the cost category
consumables_dict = pd.read_csv(path_for_consumable_resourcefiles / 'ResourceFile_consumables_matched.csv', low_memory=False,
                             encoding="ISO-8859-1")[['item_code', 'consumable_name_tlo']]
consumables_dict = consumables_dict.rename(columns = {'item_code': 'Item_Code'})
consumables_dict = dict(zip(consumables_dict['Item_Code'], consumables_dict['consumable_name_tlo']))
def melt_and_label_consumables_cost(_df, label):
    multi_index = pd.MultiIndex.from_tuples(_df.columns)
    _df.columns = multi_index
    # Select 'Item_Code', 'year', and all columns where both levels of the MultiIndex are numeric (these are the (draw,run) columns with cost values)
    selected_columns = [col for col in _df.columns if
                        (col[0] in ['Item_Code', 'year']) or (isinstance(col[0], int) and isinstance(col[1], int))]
    _df = _df[selected_columns]    # Subset the dataframe with the selected columns

    # reshape dataframe and assign 'draw' and 'run' as the correct column headers
    melted_df = pd.melt(_df, id_vars=['year', 'Item_Code']).rename(columns = {'variable_0': 'draw', 'variable_1': 'run'})
    # Replace item_code with consumable_name_tlo
    melted_df['consumable'] = melted_df['Item_Code'].map(consumables_dict)
    melted_df['cost_subcategory'] = label
    melted_df['Facility_Level'] = 'all' #TODO this is temporary until 'tlo.methods.healthsystem.summary' only logs consumable at the aggregate level
    melted_df = melted_df.rename(columns = {'value': 'cost'})
    return melted_df

cost_of_consumables_dispensed = retain_relevant_column_subset(melt_and_label_consumables_cost(cost_of_consumables_dispensed, 'cost_of_consumables_dispensed'), 'consumable')
cost_of_excess_consumables_stocked = retain_relevant_column_subset(melt_and_label_consumables_cost(cost_of_excess_consumables_stocked, 'cost_of_excess_consumables_stocked'), 'consumable')
consumable_costs = pd.concat([cost_of_consumables_dispensed, cost_of_excess_consumables_stocked])
consumable_costs = prepare_cost_dataframe(consumable_costs, _category_specific_group = 'consumable', _cost_category = 'medical consumables')

# %%
# 3. Equipment cost
#--------------------------------------------
# Total cost of equipment required as per SEL (HSSP-III) only at facility IDs where it has been used in the simulation
# Get list of equipment used in the simulation by district and level
def get_equipment_used_by_district_and_facility(_df: pd.Series) -> pd.Series:
    """Summarise the parsed logged-key results for one draw (as dataframe) into a pd.Series."""
    _df = _df.pivot_table(index=['District', 'Facility_Level'],
                    values='EquipmentEverUsed',
                    aggfunc='first')
    _df.index.name = 'year'
    return _df['EquipmentEverUsed']

list_of_equipment_used_by_draw_and_run = extract_results(
    Path(results_folder),
    module='tlo.methods.healthsystem.summary',
    key='EquipmentEverUsed_ByFacilityID',
    custom_generate_series=get_equipment_used_by_district_and_facility,
    do_scaling=False,
)
for col in list_of_equipment_used_by_draw_and_run.columns:
    list_of_equipment_used_by_draw_and_run[col] = list_of_equipment_used_by_draw_and_run[col].apply(ast.literal_eval)

# Initialize an empty DataFrame
equipment_cost_across_sim = pd.DataFrame()

# Extract equipment cost for each draw and run
for d in draws:
    for r in runs:
        print(f"Now processing draw {d} and run {r}")
        # Extract a list of equipment which was used at each facility level within each district
        equipment_used = {district: {level: [] for level in fac_levels} for district in list(district_dict.values())} # create a dictionary with a key for each district and facility level
        list_of_equipment_used_by_current_draw_and_run = list_of_equipment_used_by_draw_and_run[(d, r)].reset_index()
        for dist in list(district_dict.values()):
            for level in fac_levels:
                equipment_used_subset = list_of_equipment_used_by_current_draw_and_run[(list_of_equipment_used_by_current_draw_and_run['District'] == dist) & (list_of_equipment_used_by_current_draw_and_run['Facility_Level'] == level)]
                equipment_used_subset.columns = ['District', 'Facility_Level', 'EquipmentEverUsed']
                equipment_used[dist][level] = set().union(*equipment_used_subset['EquipmentEverUsed'])
        equipment_used = pd.concat({
                k: pd.DataFrame.from_dict(v, 'index') for k, v in equipment_used.items()},
                axis=0)
        full_list_of_equipment_used = set(equipment_used.values.flatten())
        full_list_of_equipment_used = set(filter(pd.notnull, full_list_of_equipment_used))

        equipment_df = pd.DataFrame()
        equipment_df.index = equipment_used.index
        for item in full_list_of_equipment_used:
            equipment_df[str(item)] = 0
            for dist_fac_index in equipment_df.index:
                equipment_df.loc[equipment_df.index == dist_fac_index, str(item)] = equipment_used[equipment_used.index == dist_fac_index].isin([item]).any(axis=1)
        #equipment_df.to_csv('./outputs/equipment_use.csv')

        equipment_df = equipment_df.reset_index().rename(columns = {'level_0' : 'District', 'level_1': 'Facility_Level'})
        equipment_df = pd.melt(equipment_df, id_vars = ['District', 'Facility_Level']).rename(columns = {'variable': 'Item_code', 'value': 'whether_item_was_used'})
        equipment_df['Item_code'] = pd.to_numeric(equipment_df['Item_code'])
        # Merge the count of facilities by district and level
        equipment_df = equipment_df.merge(mfl[['District', 'Facility_Level','Facility_Count']], on = ['District', 'Facility_Level'], how = 'left')
        equipment_df.loc[equipment_df.Facility_Count.isna(), 'Facility_Count'] = 0

        # Because levels 1b and 2 are collapsed together, we assume that the same equipment is used by level 1b as that recorded for level 2
        def update_itemuse_for_level1b_using_level2_data(_df):
            # Create a list of District and Item_code combinations for which use == True
            list_of_equipment_used_at_level2 = _df[(_df.Facility_Level == '2') & (_df['whether_item_was_used'] == True)][['District', 'Item_code']]
            # Now update the 'whether_item_was_used' for 'Facility_Level' == '1b' to match that of level '2'
            _df.loc[
                (_df['Facility_Level'] == '1b') &
                (_df[['District', 'Item_code']].apply(tuple, axis=1).isin(
                    list_of_equipment_used_at_level2.apply(tuple, axis=1))),
                'whether_item_was_used'
            ] = True

            return _df

        equipment_df = update_itemuse_for_level1b_using_level2_data(equipment_df)

        # Merge the two datasets to calculate cost
        equipment_cost = pd.merge(equipment_df, unit_cost_equipment[['Item_code', 'Equipment_tlo', 'Facility_Level', 'Quantity', 'replacement_cost_annual', 'service_fee_annual', 'spare_parts_annual', 'major_corrective_maintenance_cost_annual']],
                                  on = ['Item_code', 'Facility_Level'], how = 'left', validate = "m:1")
        categories_of_equipment_cost = ['replacement_cost', 'service_fee', 'spare_parts', 'major_corrective_maintenance_cost']
        for cost_category in categories_of_equipment_cost:
            # Rename unit cost columns
            unit_cost_column = cost_category + '_annual_unit'
            equipment_cost = equipment_cost.rename(columns = {cost_category + '_annual':unit_cost_column })
            equipment_cost[cost_category + '_annual_total'] = equipment_cost[cost_category + '_annual_unit'] * equipment_cost['whether_item_was_used'] * equipment_cost['Quantity'] * equipment_cost['Facility_Count']
        equipment_cost['year'] = final_year_of_simulation - 1
        if equipment_cost_across_sim.empty:
            equipment_cost_across_sim = equipment_cost.groupby(['year', 'Facility_Level', 'Equipment_tlo'])[[item  + '_annual_total' for item in categories_of_equipment_cost]].sum()
            equipment_cost_across_sim['draw'] = d
            equipment_cost_across_sim['run'] = r
        else:
            equipment_cost_for_current_sim = equipment_cost.groupby(['year', 'Facility_Level', 'Equipment_tlo'])[[item  + '_annual_total' for item in categories_of_equipment_cost]].sum()
            equipment_cost_for_current_sim['draw'] = d
            equipment_cost_for_current_sim['run'] = r
            # Concatenate the results
            equipment_cost_across_sim = pd.concat([equipment_cost_across_sim, equipment_cost_for_current_sim], axis=0)

equipment_costs = pd.melt(equipment_cost_across_sim.reset_index(),
                  id_vars=['draw', 'run', 'Facility_Level', 'Equipment_tlo'],  # Columns to keep
                  value_vars=[col for col in equipment_cost_across_sim.columns if col.endswith('_annual_total')],  # Columns to unpivot
                  var_name='cost_subcategory',  # New column name for the 'sub-category' of cost
                  value_name='cost')  # New column name for the values

# Assume that the annual costs are constant each year of the simulation
equipment_costs = pd.concat([equipment_costs.assign(year=year) for year in years])
# TODO If the logger is updated to include year, we may wish to calculate equipment costs by year - currently we assume the same annuitised equipment cost each year
equipment_costs = equipment_costs.reset_index(drop=True)
equipment_costs = equipment_costs.rename(columns = {'Equipment_tlo': 'Equipment'})
equipment_costs = prepare_cost_dataframe(equipment_costs, _category_specific_group = 'Equipment', _cost_category = 'Medical Equipment')

'''
equipment_costs_summary = pd.concat(
    {
        'mean': equipment_costs.groupby(by=['draw', 'Cost_Sub-category'], sort=False)['value'].mean(),
        'lower': equipment_costs.groupby(by=['draw', 'Cost_Sub-category'], sort=False)['value'].quantile(0.025),
        'upper': equipment_costs.groupby(by=['draw', 'Cost_Sub-category'], sort=False)['value'].quantile(0.975),
    },
    axis=1
)
equipment_costs_summary =  pd.melt(equipment_costs_summary.reset_index(),
                  id_vars=['draw', 'Cost_Sub-category'],  # Columns to keep
                  value_vars=['mean', 'lower', 'upper'],  # Columns to unpivot
                  var_name='stat',  # New column name for the 'sub-category' of cost
                  value_name='value')
'''

# 4. Facility running costs
# Average running costs by facility level and district times the number of facilities  in the simulation

# %%
# Store all costs in single dataframe
#--------------------------------------------
scenario_cost = pd.concat([human_resource_costs, consumable_costs, equipment_costs], ignore_index=True)

# Additional costs pertaining to simulation
# IRS costs
irs_coverage_rate = 0.8
districts_with_irs_scaleup = ['Kasungu', 'Mchinji', 'Lilongwe', 'Lilongwe City', 'Dowa', 'Ntchisi', 'Salima', 'Mangochi',
                              'Mwanza', 'Likoma', 'Nkhotakota']
proportion_of_district_with_irs_coverage = len(districts_with_irs_scaleup)/mfl.District.nunique()
# Get total population (11/total number o districts in mfl)
#TARGET_PERIOD_MALARIA_SCALEUP = (Date(2024, 1, 1), Date(2030, 12, 31))
def get_total_population(_df):
    years_needed = [i.year for i in TARGET_PERIOD]
    _df['year'] = pd.to_datetime(_df['date']).dt.year
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['male', 'female', 'date']).set_index('year').sum(axis = 1)
    )

total_population_by_year = summarize(extract_results(
    results_folder,
    module='tlo.methods.demography',
    key='population',
    custom_generate_series=get_total_population,
    do_scaling=True
))

#years_with_no_malaria_scaleup = set(TARGET_PERIOD).symmetric_difference(set(TARGET_PERIOD_MALARIA_SCALEUP))
#years_with_no_malaria_scaleup = sorted(list(years_with_no_malaria_scaleup))
#years_with_no_malaria_scaleup =  [i.year for i in years_with_no_malaria_scaleup]

irs_cost_per_person = unit_price_consumable[unit_price_consumable.Item_Code == 161]['Final_price_per_chosen_unit (USD, 2023)']
irs_multiplication_factor = irs_cost_per_person * irs_coverage_rate * proportion_of_district_with_irs_coverage
total_irs_cost = irs_multiplication_factor.iloc[0] * total_population_by_year # for districts and scenarios included
# TODO scenarios_with_irs_scaleup
# TODO population_in_district from malria logger

# Bednet costs
bednet_coverage_rate = 0.7
# All districts covered
# We can assume 3-year lifespan of a bednet, each bednet covering 1.8 people.
unit_cost_of_bednet = unit_price_consumable[unit_price_consumable.Item_Code == 160]['Final_price_per_chosen_unit (USD, 2023)']
annual_bednet_cost_per_person = unit_cost_of_bednet / 1.8 / 3
bednet_multiplication_factor = bednet_coverage_rate * annual_bednet_cost_per_person
total_bednet_cost = bednet_multiplication_factor.iloc[0] * total_population_by_year  # for scenarios included

years_with_no_malaria_scaleup = list(range(first_year_of_simulation, 2024))
def set_cost_during_years_before_malaria_scaleup_to_zero(_df):
    for col in _df.columns:
        for y in years_with_no_malaria_scaleup:
            _df.loc[_df.index.get_level_values(0) == y, col] = 0
    return _df

total_bednet_cost = set_cost_during_years_before_malaria_scaleup_to_zero(total_bednet_cost)
total_irs_cost = set_cost_during_years_before_malaria_scaleup_to_zero(total_irs_cost)
# TODO Scale-up programmes are implemented from 01/01/2024

# Malaria scale-up costs - TOTAL
malaria_scaleup_costs = [
    (total_irs_cost.reset_index(), 'cost_of_IRS_scaleup'),
    (total_bednet_cost.reset_index(), 'cost_of_bednet_scaleup'),
]
# Iterate through additional costs, melt and concatenate
for df, label in malaria_scaleup_costs:
    new_df = melt_and_label_consumables_cost(df, label)
    scenario_cost = pd.concat([scenario_cost, new_df], ignore_index=True)
scenario_cost.loc[scenario_cost['Cost_Category'].isna(),'Cost_Category'] = 'IRS and Bednet Scale-up Costs'

# Extract all costs to a .csv
scenario_cost.to_csv(costing_outputs_folder / 'scenario_cost.csv', index = False)

# Calculate total cost
total_scenario_cost = scenario_cost[(scenario_cost.year >= 2020) & (scenario_cost.year <= 2030)].groupby(['draw', 'stat'])['value'].sum().unstack()
total_scenario_cost = total_scenario_cost.unstack().reset_index()
total_scenario_cost_wide = total_scenario_cost.pivot_table(index=None, columns=['draw', 'stat'], values=0)

# Calculate incremental cost
def find_difference_relative_to_comparison(_ser: pd.Series,
                                           comparison: str,
                                           scaled: bool = False,
                                           drop_comparison: bool = True,
                                           ):
    """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
    within the runs (level 1), relative to where draw = `comparison`.
    The comparison is `X - COMPARISON`."""
    return _ser \
        .unstack(level=0) \
        .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
        .drop(columns=([comparison] if drop_comparison else [])) \
        .stack()

# TODO the following calculation should first capture the different by run and then be summarised
incremental_scenario_cost = (pd.DataFrame(
            find_difference_relative_to_comparison(
                total_scenario_cost_wide.loc[0],
                comparison= 0) # sets the comparator to 0 which is the Actual scenario
        ).T.iloc[0].unstack()).T

# %%
# Monetary value of health impact
TARGET_PERIOD_INTERVENTION = (Date(2020, 1, 1), Date(2030, 12, 31))
def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = [i.year for i in TARGET_PERIOD_INTERVENTION]
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=True
    )

num_dalys_summarized = summarize(num_dalys).loc[0].unstack()
#num_dalys_summarized['scenario'] = scenarios.to_list() # add when scenarios have names
#num_dalys_summarized = num_dalys_summarized.set_index('scenario')

# Get absolute DALYs averted
num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison= 0) # sets the comparator to 0 which is the Actual scenario
        ).T
    ).iloc[0].unstack()
#num_dalys_averted['scenario'] = scenarios.to_list()[1:12]
#num_dalys_averted = num_dalys_averted.set_index('scenario')

chosen_cet = 77.4 # based on Ochalek et al (2018) - the paper provided the value $61 in 2016 USD terms, this value is in 2023 USD terms
monetary_value_of_incremental_health = num_dalys_averted * chosen_cet
max_ability_to_pay_for_implementation = monetary_value_of_incremental_health - incremental_scenario_cost # monetary value - change in costs

# Plot costs
####################################################
# 1. Stacked bar plot (Total cost + Cost categories)
#----------------------------------------------------
def do_stacked_bar_plot(_df, cost_category, year, actual_expenditure):
    # Subset and Pivot the data to have 'Cost Sub-category' as columns
    # Make a copy of the dataframe to avoid modifying the original
    _df = _df[_df.stat == 'mean'].copy()
    # Convert 'value' to millions
    _df['value'] = _df['value'] / 1e6
    if year == 'all':
        subset_df = _df
    else:
        subset_df = _df[_df['year'].isin(year)]
    if cost_category == 'all':
        subset_df = subset_df
        pivot_df = subset_df.pivot_table(index='draw', columns='Cost_Category', values='value', aggfunc='sum')
    else:
        subset_df = subset_df[subset_df['Cost_Category'] == cost_category]
        pivot_df = subset_df.pivot_table(index='draw', columns='Cost_Sub-category', values='value', aggfunc='sum')

    # Plot a stacked bar chart
    pivot_df.plot(kind='bar', stacked=True)
    # Add a horizontal red line to represent 2018 Expenditure as per resource mapping
    #plt.axhline(y=actual_expenditure/1e6, color='red', linestyle='--', label='Actual expenditure recorded in 2018')

    # Save plot
    plt.xlabel('Scenario')
    plt.ylabel('Cost (2023 USD), millions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.title(f'Costs by Scenario \n (Cost Category = {cost_category} ; Year = {year})')
    plt.savefig(figurespath / f'stacked_bar_chart_{cost_category}_year_{year}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

do_stacked_bar_plot(_df = scenario_cost, cost_category = 'Medical consumables', year = 2018, actual_expenditure = 206_747_565)
do_stacked_bar_plot(_df = scenario_cost, cost_category = 'Human Resources for Health', year = 2018, actual_expenditure = 128_593_787)
do_stacked_bar_plot(_df = scenario_cost, cost_category = 'Equipment purchase and maintenance', year = 2018, actual_expenditure = 6_048_481)
do_stacked_bar_plot(_df = scenario_cost, cost_category = 'all', year = [2020], actual_expenditure = 624_054_027)
do_stacked_bar_plot(_df = scenario_cost, cost_category = 'all', year = [2024], actual_expenditure = 624_054_027)
do_stacked_bar_plot(_df = scenario_cost, cost_category = 'all', year = list(range(2020,2031)), actual_expenditure = np.nan)

# 2. Line plots of total costs
#----------------------------------------------------
def do_line_plot(_df, cost_category, actual_expenditure, _draw):
    # Filter the dataframe based on the selected draw
    subset_df = _df[_df.draw == _draw]

    if cost_category != 'all':
        subset_df = subset_df[subset_df['Cost_Category'] == cost_category]

    # Reset the index for plotting purposes
    subset_df = subset_df.reset_index()

    # Extract mean, lower, and upper values for the plot
    mean_values = subset_df[subset_df.stat == 'mean'].groupby(['Cost_Category', 'year'])['value'].sum() / 1e6
    lower_values = subset_df[subset_df.stat == 'lower'].groupby(['Cost_Category', 'year'])['value'].sum() / 1e6
    upper_values = subset_df[subset_df.stat == 'upper'].groupby(['Cost_Category', 'year'])['value'].sum() / 1e6
    years = subset_df[subset_df.stat == 'mean']['year']

    # Plot the line for 'mean'
    plt.plot(mean_values.index.get_level_values(1), mean_values, marker='o', linestyle='-', color='b', label='Mean')

    # Add confidence interval using fill_between
    plt.fill_between(mean_values.index.get_level_values(1), lower_values, upper_values, color='b', alpha=0.2, label='95% CI')

    # Add a horizontal red line to represent the actual expenditure
    plt.axhline(y=actual_expenditure / 1e6, color='red', linestyle='--', label='Actual expenditure recorded in 2018')

    # Set plot labels and title
    plt.xlabel('Year')
    plt.ylabel('Cost (2023 USD), millions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.title(f'Costs by Scenario \n (Cost Category = {cost_category} ; Draw = {_draw})')

    # Save the plot
    plt.savefig(figurespath / f'trend_{cost_category}_{first_year_of_simulation}-{final_year_of_simulation}.png',
                dpi=100,
                bbox_inches='tight')
    plt.close()

do_line_plot(_df = scenario_cost, cost_category = 'Medical consumables', _draw = 0, actual_expenditure = 206_747_565)
do_line_plot(_df = scenario_cost, cost_category = 'Human Resources for Health',  _draw = 0, actual_expenditure = 128_593_787)
do_line_plot(_df = scenario_cost, cost_category = 'Equipment purchase and maintenance',  _draw = 0, actual_expenditure = 6_048_481)
do_line_plot(_df = scenario_cost, cost_category = 'all',  _draw = 0, actual_expenditure = 624_054_027)

# 3. Return on Investment Plot
#----------------------------------------------------
# Plot ROI at various levels of cost
roi_outputs_folder = Path(figurespath / 'roi')
if not os.path.exists(roi_outputs_folder):
    os.makedirs(roi_outputs_folder)

# Loop through each row and plot mean, lower, and upper values divided by costs
for index, row in monetary_value_of_incremental_health.iterrows():
    # Step 1: Create an array of implementation costs ranging from 0 to the max value of the max ability to pay
    implementation_costs = np.linspace(0, max_ability_to_pay_for_implementation.loc[index]['mean'], 50)

    plt.figure(figsize=(10, 6))

    # Retrieve the corresponding row from incremental_scenario_cost for the same 'index'
    scenario_cost_row = incremental_scenario_cost.loc[index]
    # Divide rows by the sum of implementation costs and incremental input cost
    mean_values = row['mean'] / (implementation_costs + scenario_cost_row['mean'])
    lower_values = row['lower'] / (implementation_costs + scenario_cost_row['lower'])
    upper_values = row['upper'] / (implementation_costs + scenario_cost_row['upper'])
    # Plot mean line
    plt.plot(implementation_costs/1e6, mean_values, label=f'Draw {index}')
    # Plot the confidence interval as a shaded region
    plt.fill_between(implementation_costs/1e6, lower_values, upper_values, alpha=0.2)

    # Step 4: Set plot labels and title
    plt.xlabel('Implementation cost, millions')
    plt.ylabel('Return on Investment')
    plt.title('Return on Investment of scenarios at different levels of implementation cost')

    plt.text(x=0.95, y=0.8, s=f"Monetary value of incremental health = USD {round(monetary_value_of_incremental_health.loc[index]['mean']/1e6,2)}m (USD {round(monetary_value_of_incremental_health.loc[index]['lower']/1e6,2)}m-{round(monetary_value_of_incremental_health.loc[index]['upper']/1e6,2)}m);\n "
                             f"Incremental input cost of scenario = USD {round(scenario_cost_row['mean']/1e6,2)}m (USD {round(scenario_cost_row['lower']/1e6,2)}m-{round(scenario_cost_row['upper']/1e6,2)}m)",
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=9, weight='bold', color='black')


    # Show legend
    plt.legend()
    # Save
    plt.savefig(figurespath / f'roi/ROI_draw{index}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

# 4. Plot Maximum ability-to-pay
#----------------------------------------------------
def do_bar_plot_with_ci(_df, annotations=None, xticklabels_horizontal_and_wrapped=False):
    """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
    extent of the error bar."""

    yerr = np.array([
        (_df['mean'] - _df['lower']).values,
        (_df['upper'] - _df['mean']).values,
    ])

    xticks = {(i+1): k for i, k in enumerate(_df.index)}

    fig, ax = plt.subplots()
    ax.bar(
        xticks.keys(),
        _df['mean'].values,
        yerr=yerr,
        alpha=1,
        ecolor='black',
        capsize=10,
        label=xticks.values()
    )
    '''
    if annotations:
        for xpos, ypos, text in zip(xticks.keys(), _df['upper'].values, annotations):
            ax.text(xpos, ypos * 1.05, text, horizontalalignment='center', fontsize=11)

    ax.set_xticks(list(xticks.keys()))
    if not xticklabels_horizontal_and_wrapped:
        wrapped_labs = ["\n".join(textwrap.wrap(_lab, 20)) for _lab in xticks.values()]
        ax.set_xticklabels(wrapped_labs, rotation=45, ha='right', fontsize=10)
    else:
        wrapped_labs = ["\n".join(textwrap.wrap(_lab, 20)) for _lab in xticks.values()]
        ax.set_xticklabels(wrapped_labs, fontsize=10)
    '''

    # Set font size for y-tick labels
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=11)

    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig, ax

# Plot Max ability to pay
name_of_plot = f'Maximum ability to pay, 2020-2030' #f'Maximum ability to pay, {first_year_of_simulation} - {final_year_of_simulation}'
fig, ax = do_bar_plot_with_ci(
    (max_ability_to_pay_for_implementation / 1e6).clip(lower=0.0),
    annotations=[
        f"{round(row['mean']/1e6, 1)} \n ({round(row['lower']/1e6, 1)}-{round(row['upper']/1e6, 1)})"
        for _, row in max_ability_to_pay_for_implementation.clip(lower=0.0).iterrows()
    ],
    xticklabels_horizontal_and_wrapped=False,
)
ax.set_title(name_of_plot)
#ax.set_ylim(0, 120)
#ax.set_yticks(np.arange(0, 120, 10))
ax.set_ylabel('Maximum ability to pay \n(Millions)')
fig.tight_layout()
fig.savefig(figurespath / name_of_plot.replace(' ', '_').replace(',', ''))
fig.show()
plt.close(fig)

# 5. Calibration plots
# Steps: 1. Create a mapping of data labels in model_costing and relevant calibration data, 2. Create a dataframe with model_costs and calibration costs;
# Prepare data for calibration
calibration_data = workbook_cost["resource_mapping_r7_summary"]
# Make sure values are numeric
budget_columns = ['BUDGETS (USD) (Jul 2019 - Jun 2020)', 'BUDGETS (USD) (Jul 2020 - Jun 2021)',
       'BUDGETS (USD) (Jul 2021 - Jun 2022)']
expenditure_columns = ['EXPENDITURE (USD) (Jul 2018 - Jun 2019)']
calibration_data[budget_columns + expenditure_columns] = calibration_data[budget_columns + expenditure_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))
# For calibration to budget figures, we take the maximum value across the three years in the RM to provide an
# upper limit to calibrate to (expenditure providing the lower limit)
calibration_data['max_annual_budget_2020-22'] = calibration_data[budget_columns].max(axis=1, skipna = True)
calibration_data = calibration_data.rename(columns = {'EXPENDITURE (USD) (Jul 2018 - Jun 2019)': 'actual_expenditure_2019',
                                                      'Calibration_category': 'calibration_category'})
calibration_data = calibration_data[['calibration_category','actual_expenditure_2019', 'max_annual_budget_2020-22']]
calibration_data = calibration_data.groupby('calibration_category')[['actual_expenditure_2019', 'max_annual_budget_2020-22']].sum().reset_index()
calibration_data1 = calibration_data.copy()
calibration_data1['stat'] = 'lower'
calibration_data2 = calibration_data.copy()
calibration_data2['stat'] = 'mean'
calibration_data3 = calibration_data.copy()
calibration_data3['stat'] = 'upper'
calibration_data = pd.concat([calibration_data1, calibration_data2, calibration_data3], axis = 0)
calibration_data = calibration_data.set_index(['calibration_category', 'stat'])

# Manually create a dataframe of model costs and relevant calibration values
def get_calibration_relevant_subset(_df):
    cond_calibration_subset = (_df.year == 2018) & (_df.draw == 0)
    return _df[cond_calibration_subset]
def get_calibration_relevant_subset_of_consumables_cost(_df, item):
    #_df =_df.rename(columns = {('year', ''):'year'})
    for col in ['Item_Code', 'Final_price_per_chosen_unit (USD, 2023)', 'excess_stock_proportion_of_dispensed','item_code']:
        try:
            _df = _df.drop(columns = col)
        except:
            pass
    _df.columns = pd.MultiIndex.from_tuples(_df.columns)
    _df = _df.melt(id_vars = ['year', 'Item_Code'], var_name=['draw', 'stat'], value_name='value')
    _df = _df[_df['Item_Code'].isin(item)]
    _df = _df.groupby(['year', 'draw', 'stat'])['value'].sum()
    return get_calibration_relevant_subset(_df.reset_index())
def merged_calibration_relevant_consumables_costs(item, category):
    merged_df = pd.merge(get_calibration_relevant_subset_of_consumables_cost(cost_of_consumables_dispensed, item),
                         get_calibration_relevant_subset_of_consumables_cost(cost_of_excess_consumables_stocked, item),
                         on=['year', 'draw', 'stat'], how='outer', suffixes=('_dispensed', '_excess_stock'))
    # Fill any missing values in the value columns with 0 (for cases where only one dataframe has a value)
    # and sum to get total consumable cost
    merged_df['value'] = merged_df['value_dispensed'].fillna(0) + merged_df['value_excess_stock'].fillna(0)
    merged_df['calibration_category'] = category
    return merged_df.set_index(['calibration_category', 'stat'])['value']

def first_positive(series):
    return next((x for x in series if pd.notna(x) and x > 0), np.nan)

def get_calibration_relevant_subset_of_other_costs(_df, _subcategory, _calibration_category):
    new_data = get_calibration_relevant_subset(_df[_df['Cost_Sub-category'].isin(_subcategory)]).groupby('stat')['value'].sum()
    new_data = new_data.reset_index()
    new_data['calibration_category'] = _calibration_category
    new_data = new_data.rename(columns =  {'value':'model_cost'})
    return new_data.set_index(['calibration_category', 'stat'])['model_cost']

# Consumables
calibration_data['model_cost'] = np.nan
# Note that the main ARV  regimen in 2018 was tenofovir/lamivudine/efavirenz as opposed to Tenofovir/Lamivudine/Dolutegravir as used in the RF_Costing. The price of this
# was $80 per year (80/(0.103*365)) times what's estimated by the model so let's update this
art = [2671, 2672, 2673]
tb_treatment = [176, 177, 179, 178, 181, 2678]
antimalarials = [162,164,170]
malaria_rdts = [163]
hiv_screening = [190,191,196]
condoms = [2,25]
tb_tests = [184,187, 175]
other_drugs = set(cost_of_consumables_dispensed['Item_Code'].unique()) - set(art) - set(tb_treatment) - set(antimalarials) - set(malaria_rdts) - set(hiv_screening)\
              - set(condoms) - set(tb_tests)

calibration_data['model_cost'] = calibration_data['model_cost'].fillna(merged_calibration_relevant_consumables_costs(art, 'Antiretrovirals') * 80/(0.103*365))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(merged_calibration_relevant_consumables_costs(tb_treatment, 'TB Treatment'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(merged_calibration_relevant_consumables_costs(antimalarials, 'Antimalarials'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(merged_calibration_relevant_consumables_costs(malaria_rdts, 'Malaria RDTs'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(merged_calibration_relevant_consumables_costs(hiv_screening, 'HIV Screening/Diagnostic Tests'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(merged_calibration_relevant_consumables_costs(condoms, 'Condoms and Lubricants'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(merged_calibration_relevant_consumables_costs(tb_tests, 'TB Tests (including RDTs)'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(merged_calibration_relevant_consumables_costs(other_drugs, 'Other Drugs, medical supplies, and commodities'))

# HR
ratio_of_all_to_used_staff = total_salary_for_all_staff[(0,2018)]/total_salary_for_staff_used_in_scenario[( 0, 'lower')][2018]
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_other_costs(scenario_cost, ['salary_for_used_cadres'], 'Health Worker Salaries') * ratio_of_all_to_used_staff)
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_other_costs(scenario_cost, ['preservice_training_cost_for_attrited_workers'], 'Health Worker Training - Pre-Service') * ratio_of_all_to_used_staff)
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_other_costs(scenario_cost, ['inservice_training_cost_for_all_staff'], 'Health Worker Training - In-Service') * ratio_of_all_to_used_staff)
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_other_costs(scenario_cost, ['recruitment_cost_for_attrited_workers'], 'Other Human Resources for Health expenses') * ratio_of_all_to_used_staff)

# Equipment
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_other_costs(scenario_cost, ['replacement_cost_annual_total'], 'Medical Equipment - Purchase'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_other_costs(scenario_cost, ['upfront_repair_cost_annual_total', 'spare_parts_annual_total',
       'service_fee_annual_total'], 'Medical Equipment - Maintenance'))
#calibration_data[calibration_data['calibration_category'] == 'Vehicles - Purchase and Maintenance'] = get_calibration_relevant_subset()
#calibration_data[calibration_data['calibration_category'] == 'Vehicles - Purchase and Maintenance'] = get_calibration_relevant_subset()

# Facility operation costs
#calibration_data[calibration_data['calibration_category'] == 'Facility utility bills - ICT', 'Infrastructure - New Builds'] = get_calibration_relevant_subset()

# Infrastructure
#calibration_data[calibration_data['calibration_category'] == 'Infrastructure - Rehabilitation'] = get_calibration_relevant_subset()

# 3. Create calibration plot
list_of_consumables_costs_for_calibration_only_hiv = ['Antiretrovirals', 'HIV Screening/Diagnostic Tests']
list_of_consumables_costs_for_calibration_without_hiv =['Antimalarials', 'Condoms and Lubricants','Malaria RDTs', 'TB Tests (including RDTs)', 'TB Treatment', 'Other Drugs, medical supplies, and commodities']
list_of_hr_costs_for_calibration = [ 'Health Worker Training - In-Service',  'Health Worker Salaries', 'Health Worker Training - Pre-Service', 'Other Human Resources for Health expenses']
list_of_equipment_costs_for_calibration = ['Medical Equipment - Purchase', 'Medical Equipment - Maintenance']
# Add folder to store calibration plots

calibration_outputs_folder = Path(figurespath / 'calibration')
if not os.path.exists(calibration_outputs_folder):
    os.makedirs(calibration_outputs_folder)

def do_cost_calibration_plot(_df, _costs_included):
    # Filter the dataframe
    _df = _df[(_df.model_cost.notna()) & (_df.index.get_level_values(0).isin(_costs_included))]

    # For df_mean
    df_mean = _df.loc[_df.index.get_level_values('stat') == 'mean'].reset_index(level='stat', drop=True)/1e6
    total_mean = pd.DataFrame(df_mean.sum()).T  # Calculate the total and convert it to a DataFrame
    total_mean.index = ['Total']  # Name the index of the total row as 'Total'
    df_mean = pd.concat([df_mean, total_mean], axis=0)  # Concatenate the total row

    # For df_lower
    df_lower = _df.loc[_df.index.get_level_values('stat') == 'lower'].reset_index(level='stat', drop=True)/1e6
    total_lower = pd.DataFrame(df_lower.sum()).T  # Calculate the total and convert it to a DataFrame
    total_lower.index = ['Total']  # Name the index of the total row as 'Total'
    df_lower = pd.concat([df_lower, total_lower], axis=0)  # Concatenate the total row

    # For df_upper
    df_upper = _df.loc[_df.index.get_level_values('stat') == 'upper'].reset_index(level='stat', drop=True)/1e6
    total_upper = pd.DataFrame(df_upper.sum()).T  # Calculate the total and convert it to a DataFrame
    total_upper.index = ['Total']  # Name the index of the total row as 'Total'
    df_upper = pd.concat([df_upper, total_upper], axis=0)  # Concatenate the total row

    # Create the dot plot
    plt.figure(figsize=(12, 8))

    # Plot model_cost as dots with confidence interval error bars
    plt.errorbar(df_mean.index, df_mean['model_cost'],
                 yerr=[df_mean['model_cost'] - df_lower['model_cost'], df_upper['model_cost'] - df_mean['model_cost']],
                 fmt='o', label='Model Cost', ecolor='gray', capsize=5, color='saddlebrown')

    # Plot annual_expenditure_2019 and max_annual_budget_2020-22 as dots
    plt.plot(df_mean.index, df_mean['actual_expenditure_2019'], 'bo', label='Actual Expenditure 2019', markersize=8)
    plt.plot(df_mean.index, df_mean['max_annual_budget_2020-22'], 'go', label='Max Annual Budget 2020-22', markersize=8)

    # Draw a blue line between annual_expenditure_2019 and max_annual_budget_2020-22
    plt.vlines(df_mean.index, df_mean['actual_expenditure_2019'], df_mean['max_annual_budget_2020-22'], color='blue',
               label='Budget Range')

    # Add labels to the model_cost dots (yellow color, slightly shifted right)
    for i, (x, y) in enumerate(zip(df_mean.index, df_mean['model_cost'])):
        plt.text(i + 0.05, y, f'{y:.2f}', ha='left', va='bottom', fontsize=9,
                 color='saddlebrown')  # label model_cost values

    # Add labels and title
    cost_subcategory = [name for name in globals() if globals()[name] is _costs_included][0]
    cost_subcategory = cost_subcategory.replace('list_of_', '').replace('_for_calibration', '')
    plt.xlabel('Cost Sub-Category')
    plt.ylabel('Costs (USD), millions')
    plt.title(f'Model Cost vs Annual Expenditure 2019 and Max(Annual Budget 2020-22)\n {cost_subcategory}')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')

    # Adding a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    # Tight layout and save the figure
    plt.tight_layout()
    plt.savefig(calibration_outputs_folder / f'calibration_dot_plot_{cost_subcategory}.png', dpi=100,
                bbox_inches='tight')
    plt.close()


# Call the function for each variable and cost list
all_calibration_costs = list_of_consumables_costs_for_calibration_only_hiv + list_of_consumables_costs_for_calibration_without_hiv + list_of_hr_costs_for_calibration + list_of_equipment_costs_for_calibration
all_consumable_costs = list_of_consumables_costs_for_calibration_without_hiv + list_of_consumables_costs_for_calibration_only_hiv

do_cost_calibration_plot(calibration_data,list_of_consumables_costs_for_calibration_without_hiv)
do_cost_calibration_plot(calibration_data,list_of_consumables_costs_for_calibration_only_hiv)
do_cost_calibration_plot(calibration_data,all_consumable_costs)
do_cost_calibration_plot(calibration_data, list_of_hr_costs_for_calibration)
do_cost_calibration_plot(calibration_data, list_of_equipment_costs_for_calibration)
do_cost_calibration_plot(calibration_data,all_calibration_costs)
calibration_data.to_csv(figurespath / 'calibration/calibration.csv')

# TODO all these HR plots need to be looked at
# 1. HR
# Stacked bar chart of salaries by cadre
def get_level_and_cadre_from_concatenated_value(_df, varname):
    _df['Cadre'] = _df[varname].str.extract(r'=(.*?)\|')
    _df['Facility_Level'] = _df[varname].str.extract(r'^[^=]*=[^|]*\|[^=]*=([^|]*)')
    return _df
def plot_cost_by_cadre_and_level(_df, figname_prefix, figname_suffix, draw):
    if ('Facility_Level' in _df.columns) & ('Cadre' in _df.columns):
        pass
    else:
        _df = get_level_and_cadre_from_concatenated_value(_df, 'OfficerType_FacilityLevel')

    _df = _df[_df.draw == draw]
    pivot_df = _df.pivot_table(index='Cadre', columns='Facility_Level', values='Cost',
                               aggfunc='sum', fill_value=0)
    total_salary = round(_df['Cost'].sum(), 0)
    total_salary = f"{total_salary:,.0f}"
    ax  = pivot_df.plot(kind='bar', stacked=True, title='Stacked Bar Graph by Cadre and Facility Level')
    plt.ylabel(f'US Dollars')
    plt.title(f"Annual {figname_prefix} cost by cadre and facility level")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.text(x=0.3, y=-0.5, s=f"Total {figname_prefix} cost = USD {total_salary}", transform=ax.transAxes,
             horizontalalignment='center', fontsize=12, weight='bold', color='black')
    plt.savefig(figurespath / f'{figname_prefix}_by_cadre_and_level_{figname_suffix}{draw}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

plot_cost_by_cadre_and_level(salary_for_all_staff,figname_prefix = "salary", figname_suffix= f"all_staff_draw", draw = 0)
plot_cost_by_cadre_and_level(salary_for_staff_used_in_scenario.reset_index(),figname_prefix = "salary", figname_suffix= "staff_used_in_scenario_draw", draw = 0)
plot_cost_by_cadre_and_level(recruitment_cost, figname_prefix = "recruitment", figname_suffix= "all_staff")
plot_cost_by_cadre_and_level(preservice_training_cost, figname_prefix = "pre-service training", figname_suffix= "all_staff")
plot_cost_by_cadre_and_level(inservice_training_cost, figname_prefix = "in-service training", figname_suffix= "all_staff")

def plot_components_of_cost_category(_df, cost_category, figname_suffix):
    pivot_df = _df[_df['Cost_Category'] == cost_category].pivot_table(index='Cost_Sub-category', values='Cost',
                               aggfunc='sum', fill_value=0)
    ax = pivot_df.plot(kind='bar', stacked=False, title='Scenario Cost by Category')
    plt.ylabel(f'US Dollars')
    plt.title(f"Annual {cost_category} cost")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Add text labels on the bars
    total_cost = pivot_df['Cost'].sum()
    rects = ax.patches
    for rect, cost in zip(rects, pivot_df['Cost']):
        cost_millions = cost / 1e6
        percentage = (cost / total_cost) * 100
        label_text = f"{cost_millions:.1f}M ({percentage:.1f}%)"
        # Place text at the top of the bar
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()
        ax.text(x, y, label_text, ha='center', va='bottom', fontsize=8, rotation=0)

    total_cost = f"{total_cost:,.0f}"
    plt.text(x=0.3, y=-0.5, s=f"Total {cost_category} cost = USD {total_cost}", transform=ax.transAxes,
             horizontalalignment='center', fontsize=12, weight='bold', color='black')

    plt.savefig(figurespath / f'{cost_category}_by_cadre_and_level_{figname_suffix}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

plot_components_of_cost_category(_df = scenario_cost, cost_category = 'Human Resources for Health', figname_suffix = "all_staff")


# Compare financial costs with actual budget data
####################################################
# Import budget data
budget_data = workbook_cost["budget_validation"]
list_of_costs_for_comparison = ['total_salary_for_all_staff', 'total_cost_of_consumables_dispensed', 'total_cost_of_consumables_stocked']
real_budget = [budget_data[budget_data['Category'] == list_of_costs_for_comparison[0]]['Budget_in_2023USD'].values[0],
               budget_data[budget_data['Category'] == list_of_costs_for_comparison[1]]['Budget_in_2023USD'].values[0],
               budget_data[budget_data['Category'] == list_of_costs_for_comparison[1]]['Budget_in_2023USD'].values[0]]
model_cost = [scenario_cost_financial[scenario_cost_financial['Cost_Sub-category'] == list_of_costs_for_comparison[0]]['Value_2023USD'].values[0],
              scenario_cost_financial[scenario_cost_financial['Cost_Sub-category'] == list_of_costs_for_comparison[1]]['Value_2023USD'].values[0],
              scenario_cost_financial[scenario_cost_financial['Cost_Sub-category'] == list_of_costs_for_comparison[2]]['Value_2023USD'].values[0]]

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
consumables_label1= 'Consumables dispensed ' + f'{round(model_cost[1] / real_budget[1], 2)}'
consumables_label2 = 'Consumables stocked ' + f'{round(model_cost[2] / real_budget[2], 2)}'
plotlabels = [hr_label, consumables_label1, consumables_label2]
for i, txt in enumerate(plotlabels):
    plt.text(real_budget[i], model_cost[i], txt, ha='right')

plt.xlabel('Real Budget')
plt.ylabel('Model Cost')
plt.title('Real Budget vs Model Cost')
plt.savefig(costing_outputs_folder /  'Cost_validation.png')

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

'''
consumables_dispensed_under_perfect_availability = get_quantity_of_consumables_dispensed(consumables_results_folder)[9]
consumables_dispensed_under_perfect_availability = consumables_dispensed_under_perfect_availability['mean'].to_dict() # TODO incorporate uncertainty in estimates
consumables_dispensed_under_perfect_availability = defaultdict(int, {int(key): value for key, value in
                                   consumables_dispensed_under_perfect_availability.items()})  # Convert string keys to integer
consumables_dispensed_under_default_availability = get_quantity_of_consumables_dispensed(consumables_results_folder)[0]
consumables_dispensed_under_default_availability = consumables_dispensed_under_default_availability['mean'].to_dict()
consumables_dispensed_under_default_availability = defaultdict(int, {int(key): value for key, value in
                                   consumables_dispensed_under_default_availability.items()})  # Convert string keys to integer

# Load consumables cost data
unit_price_consumable = workbook_cost["consumables"]
unit_price_consumable = unit_price_consumable.rename(columns=unit_price_consumable.iloc[0])
unit_price_consumable = unit_price_consumable[['Item_Code', 'Final_price_per_chosen_unit (USD, 2023)']].reset_index(drop=True).iloc[1:]
unit_price_consumable = unit_price_consumable[unit_price_consumable['Item_Code'].notna()]
unit_price_consumable = unit_price_consumable.set_index('Item_Code').to_dict(orient='index')

# 2.1 Cost of consumables dispensed
#---------------------------------------------------------------------------------------------------------------
# Multiply number of items needed by cost of consumable
cost_of_consumables_dispensed_under_perfect_availability = {key: unit_price_consumable[key]['Final_price_per_chosen_unit (USD, 2023)'] * consumables_dispensed_under_perfect_availability[key] for
                                                            key in unit_price_consumable if key in consumables_dispensed_under_perfect_availability}
total_cost_of_consumables_dispensed_under_perfect_availability = sum(value for value in cost_of_consumables_dispensed_under_perfect_availability.values() if not np.isnan(value))

cost_of_consumables_dispensed_under_default_availability = {key: unit_price_consumable[key]['Final_price_per_chosen_unit (USD, 2023)'] * consumables_dispensed_under_default_availability[key] for
                                                            key in unit_price_consumable if key in consumables_dispensed_under_default_availability}
total_cost_of_consumables_dispensed_under_default_availability = sum(value for value in cost_of_consumables_dispensed_under_default_availability.values() if not np.isnan(value))
def convert_dict_to_dataframe(_dict):
    data = {key: [value] for key, value in _dict.items()}
    _df = pd.DataFrame(data)
    return _df

cost_perfect_df = convert_dict_to_dataframe(cost_of_consumables_dispensed_under_perfect_availability).T.rename(columns = {0:"cost_dispensed_stock_perfect_availability"}).round(2)
cost_default_df = convert_dict_to_dataframe(cost_of_consumables_dispensed_under_default_availability).T.rename(columns = {0:"cost_dispensed_stock_default_availability"}).round(2)
unit_cost_df = convert_dict_to_dataframe(unit_price_consumable).T.rename(columns = {0:"unit_cost"})
dispensed_default_df = convert_dict_to_dataframe(consumables_dispensed_under_default_availability).T.rename(columns = {0:"dispensed_default_availability"}).round(2)
dispensed_perfect_df = convert_dict_to_dataframe(consumables_dispensed_under_perfect_availability).T.rename(columns = {0:"dispensed_perfect_availability"}).round(2)

full_cons_cost_df = pd.merge(cost_perfect_df, cost_default_df, left_index=True, right_index=True)
full_cons_cost_df = pd.merge(full_cons_cost_df, unit_cost_df, left_index=True, right_index=True)
full_cons_cost_df = pd.merge(full_cons_cost_df, dispensed_default_df, left_index=True, right_index=True)
full_cons_cost_df = pd.merge(full_cons_cost_df, dispensed_perfect_df, left_index=True, right_index=True)

# 2.2 Cost of consumables stocked (quantity needed for what is dispensed)
#---------------------------------------------------------------------------------------------------------------
# Stocked amount should be higher than dispensed because of i. excess capacity, ii. theft, iii. expiry
# While there are estimates in the literature of what % these might be, we agreed that it is better to rely upon
# an empirical estimate based on OpenLMIS data
# Estimate the stock to dispensed ratio from OpenLMIS data
lmis_consumable_usage = pd.read_csv(path_for_consumable_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")
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
excess_stock_ratio = inflow_to_outflow_ratio_by_consumable - 1
excess_stock_ratio = excess_stock_ratio.to_dict()
# TODO Consider whether a more disaggregated version of the ratio dictionary should be applied
cost_of_excess_consumables_stocked_under_perfect_availability = dict(zip(unit_price_consumable, (unit_price_consumable[key]['Final_price_per_chosen_unit (USD, 2023)'] *
                                                consumables_dispensed_under_perfect_availability[key] *
                                                excess_stock_ratio.get(key, average_inflow_to_outflow_ratio_ratio - 1)
                                                for key in consumables_dispensed_under_perfect_availability)))
cost_of_excess_consumables_stocked_under_default_availability = dict(zip(unit_price_consumable, (unit_price_consumable[key]['Final_price_per_chosen_unit (USD, 2023)'] *
                                                consumables_dispensed_under_default_availability[key] *
                                                excess_stock_ratio.get(key, average_inflow_to_outflow_ratio_ratio - 1)
                                                for key in consumables_dispensed_under_default_availability)))
cost_excess_stock_perfect_df = convert_dict_to_dataframe(cost_of_excess_consumables_stocked_under_perfect_availability).T.rename(columns = {0:"cost_excess_stock_perfect_availability"}).round(2)
cost_excess_stock_default_df = convert_dict_to_dataframe(cost_of_excess_consumables_stocked_under_default_availability).T.rename(columns = {0:"cost_excess_stock_default_availability"}).round(2)
full_cons_cost_df = pd.merge(full_cons_cost_df, cost_excess_stock_perfect_df, left_index=True, right_index=True)
full_cons_cost_df = pd.merge(full_cons_cost_df, cost_excess_stock_default_df, left_index=True, right_index=True)

total_cost_of_excess_consumables_stocked_under_perfect_availability = sum(value for value in cost_of_excess_consumables_stocked_under_perfect_availability.values() if not np.isnan(value))
total_cost_of_excess_consumables_stocked_under_default_availability = sum(value for value in cost_of_excess_consumables_stocked_under_default_availability.values() if not np.isnan(value))

full_cons_cost_df = full_cons_cost_df.reset_index().rename(columns = {'index' : 'item_code'})
full_cons_cost_df.to_csv(figurespath / 'consumables_cost_220824.csv')

# Import data for plotting
tlo_lmis_mapping = pd.read_csv(path_for_consumable_resourcefiles / 'ResourceFile_consumables_matched.csv', low_memory=False, encoding="ISO-8859-1")[['item_code', 'module_name', 'consumable_name_tlo']]
tlo_lmis_mapping = tlo_lmis_mapping[~tlo_lmis_mapping['item_code'].duplicated(keep='first')]
full_cons_cost_df = pd.merge(full_cons_cost_df, tlo_lmis_mapping, on = 'item_code', how = 'left', validate = "1:1")
full_cons_cost_df['total_cost_perfect_availability'] = full_cons_cost_df['cost_dispensed_stock_perfect_availability'] + full_cons_cost_df['cost_excess_stock_perfect_availability']
full_cons_cost_df['total_cost_default_availability'] = full_cons_cost_df['cost_dispensed_stock_default_availability'] + full_cons_cost_df['cost_excess_stock_default_availability']

def recategorize_modules_into_consumable_categories(_df):
    _df['category'] = _df['module_name'].str.lower()
    cond_RH = (_df['category'].str.contains('care_of_women_during_pregnancy')) | \
              (_df['category'].str.contains('labour'))
    cond_newborn = (_df['category'].str.contains('newborn'))
    cond_newborn[cond_newborn.isna()] = False
    cond_childhood = (_df['category'] == 'acute lower respiratory infections') | \
                     (_df['category'] == 'measles') | \
                     (_df['category'] == 'diarrhoea')
    cond_rti = _df['category'] == 'road traffic injuries'
    cond_cancer = _df['category'].str.contains('cancer')
    cond_cancer[cond_cancer.isna()] = False
    cond_ncds = (_df['category'] == 'epilepsy') | \
                (_df['category'] == 'depression')
    _df.loc[cond_RH, 'category'] = 'reproductive_health'
    _df.loc[cond_cancer, 'category'] = 'cancer'
    _df.loc[cond_newborn, 'category'] = 'neonatal_health'
    _df.loc[cond_childhood, 'category'] = 'other_childhood_illnesses'
    _df.loc[cond_rti, 'category'] = 'road_traffic_injuries'
    _df.loc[cond_ncds, 'category'] = 'ncds'
    cond_condom = _df['item_code'] == 2
    _df.loc[cond_condom, 'category'] = 'contraception'

    # Create a general consumables category
    general_cons_list = [300, 33, 57, 58, 141, 5, 6, 10, 21, 23, 127, 24, 80, 93, 144, 149, 154, 40, 67, 73, 76,
                         82, 101, 103, 88, 126, 135, 71, 98, 171, 133, 134, 244, 247, 49, 112, 1933, 1960]
    cond_general = _df['item_code'].isin(general_cons_list)
    _df.loc[cond_general, 'category'] = 'general'

    return _df

full_cons_cost_df = recategorize_modules_into_consumable_categories(full_cons_cost_df)
# Fill gaps in categories
dict_for_missing_categories =  {292: 'acute lower respiratory infections',  293: 'acute lower respiratory infections',
                                307: 'reproductive_health', 2019: 'reproductive_health',
                                2678: 'tb', 1171: 'other_childhood_illnesses', 1237: 'cancer', 1239: 'cancer'}
# Use map to create a new series from item_code to fill missing values in category
mapped_categories = full_cons_cost_df['item_code'].map(dict_for_missing_categories)
# Use fillna on the 'category' column to fill missing values using the mapped_categories
full_cons_cost_df['category'] = full_cons_cost_df['category'].fillna(mapped_categories)

# Bar plot of cost of dispensed consumables
def plot_consumable_cost(_df, suffix, groupby_var, top_x_values =  float('nan')):
    pivot_df = _df.groupby(groupby_var)['cost_' + suffix].sum().reset_index()
    pivot_df['cost_' + suffix] = pivot_df['cost_' + suffix]/1e6
    if math.isnan(top_x_values):
        pass
    else:
        pivot_df = pivot_df.sort_values('cost_' + suffix, ascending = False)[1:top_x_values]
    total_cost = round(_df['cost_' + suffix].sum(), 0)
    total_cost = f"{total_cost:,.0f}"
    ax  = pivot_df['cost_' + suffix].plot(kind='bar', stacked=False, title=f'Consumables cost by {groupby_var}')
    # Setting x-ticks explicitly
    #ax.set_xticks(range(len(pivot_df['category'])))
    ax.set_xticklabels(pivot_df[groupby_var], rotation=45)
    plt.ylabel(f'US Dollars (millions)')
    plt.title(f"Annual consumables cost by {groupby_var} (assuming {suffix})")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.text(x=0.5, y=-0.8, s=f"Total consumables cost =\n USD {total_cost}", transform=ax.transAxes,
             horizontalalignment='center', fontsize=12, weight='bold', color='black')
    plt.savefig(figurespath / f'consumables_cost_by_{groupby_var}_{suffix}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

plot_consumable_cost(_df = full_cons_cost_df,suffix =  'dispensed_stock_perfect_availability', groupby_var = 'category')
plot_consumable_cost(_df = full_cons_cost_df, suffix =  'dispensed_stock_default_availability', groupby_var = 'category')

# Plot the 10 consumables with the highest cost
plot_consumable_cost(_df = full_cons_cost_df,suffix =  'dispensed_stock_perfect_availability', groupby_var = 'consumable_name_tlo', top_x_values = 10)
plot_consumable_cost(_df = full_cons_cost_df,suffix =  'dispensed_stock_default_availability', groupby_var = 'consumable_name_tlo', top_x_values = 10)

def plot_cost_by_category(_df, suffix , figname_prefix = 'Consumables'):
    pivot_df = full_cons_cost_df[['category', 'cost_dispensed_stock_' + suffix, 'cost_excess_stock_' + suffix]]
    pivot_df = pivot_df.groupby('category')[['cost_dispensed_stock_' + suffix, 'cost_excess_stock_' + suffix]].sum()
    total_cost = round(_df['total_cost_' + suffix].sum(), 0)
    total_cost = f"{total_cost:,.0f}"
    ax  = pivot_df.plot(kind='bar', stacked=True, title='Stacked Bar Graph by Category')
    plt.ylabel(f'US Dollars')
    plt.title(f"Annual {figname_prefix} cost by category")
    plt.xticks(rotation=90, size = 9)
    plt.yticks(rotation=0)
    plt.text(x=0.3, y=-0.5, s=f"Total {figname_prefix} cost = USD {total_cost}", transform=ax.transAxes,
             horizontalalignment='center', fontsize=12, weight='bold', color='black')
    plt.savefig(figurespath / f'{figname_prefix}_by_category_{suffix}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

plot_cost_by_category(full_cons_cost_df, suffix = 'perfect_availability' , figname_prefix = 'Consumables')
plot_cost_by_category(full_cons_cost_df, suffix = 'default_availability' , figname_prefix = 'Consumables')
'''

'''
# Plot equipment cost
# Plot different categories of cost by level of care
def plot_components_of_cost_category(_df, cost_category, figname_suffix):
    pivot_df = _df[_df['Cost_Category'] == cost_category].pivot_table(index='Cost_Sub-category', values='Cost',
                               aggfunc='sum', fill_value=0)
    ax = pivot_df.plot(kind='bar', stacked=False, title='Scenario Cost by Category')
    plt.ylabel(f'US Dollars')
    plt.title(f"Annual {cost_category} cost")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Add text labels on the bars
    total_cost = pivot_df['Cost'].sum()
    rects = ax.patches
    for rect, cost in zip(rects, pivot_df['Cost']):
        cost_millions = cost / 1e6
        percentage = (cost / total_cost) * 100
        label_text = f"{cost_millions:.1f}M ({percentage:.1f}%)"
        # Place text at the top of the bar
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()
        ax.text(x, y, label_text, ha='center', va='bottom', fontsize=8, rotation=0)

    total_cost = f"{total_cost:,.0f}"
    plt.text(x=0.3, y=-0.5, s=f"Total {cost_category} cost = USD {total_cost}", transform=ax.transAxes,
             horizontalalignment='center', fontsize=12, weight='bold', color='black')

    plt.savefig(figurespath / f'{cost_category}_{figname_suffix}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

plot_components_of_cost_category(_df = scenario_cost, cost_category = 'Equipment', figname_suffix = "")

# Plot top 10 most expensive items
def plot_most_expensive_equipment(_df, top_x_values = 10, figname_prefix = "Equipment"):
    top_x_items = _df.groupby('Item_code')['annual_cost'].sum().sort_values(ascending = False)[0:top_x_values-1].index
    _df_subset = _df[_df.Item_code.isin(top_x_items)]

    pivot_df = _df_subset.pivot_table(index='Equipment_tlo', columns='Facility_Level', values='annual_cost',
                               aggfunc='sum', fill_value=0)
    ax = pivot_df.plot(kind='bar', stacked=True, title='Stacked Bar Graph by Item and Facility Level')
    plt.ylabel(f'US Dollars')
    plt.title(f"Annual {figname_prefix} cost by item and facility level")
    plt.xticks(rotation=90, size = 8)
    plt.yticks(rotation=0)
    plt.savefig(figurespath / f'{figname_prefix}_by_item_and_level.png', dpi=100,
                bbox_inches='tight')
    plt.close()


plot_most_expensive_equipment(equipment_cost)

# TODO PLot which equipment is used by district and facility or a heatmap of the number of facilities at which an equipment is used
# TODO Collapse facility IDs by level of care to get the total number of facilities at each level using an item
# TODO Multiply number of facilities by level with the quantity needed of each equipment and collapse to get total number of equipment (nationally)
# TODO Which equipment needs to be newly purchased (currently no assumption made for equipment with cost > $250,000)

# Calibration scatter plots
def do_cost_calibration_plot(_df, _costs_included, _calibration_var):
    _df = _df[(_df.model_cost.notna()) & (_df.index.get_level_values(0).isin(_costs_included))]
    df_mean = _df.loc[_df.index.get_level_values('stat') == 'mean'].reset_index(level='stat', drop=True)
    df_lower = _df.loc[_df.index.get_level_values('stat') == 'lower'].reset_index(level='stat', drop=True)
    df_upper = _df.loc[_df.index.get_level_values('stat') == 'upper'].reset_index(level='stat', drop=True)

    # Create the scatter plot
    plt.figure(figsize=(10, 6))

    # Plot each point with error bars (for confidence interval)
    plt.errorbar(df_mean[_calibration_var],
                 df_mean['model_cost'],
                 yerr=[df_mean['model_cost'] - df_lower['model_cost'], df_upper['model_cost'] - df_mean['model_cost']],
                 fmt='o',
                 ecolor='gray',
                 capsize=5,
                 label='Calibration Category')

    # Adding the 45-degree line (where y = x)
    min_val = min(df_mean[_calibration_var].min(), df_mean['model_cost'].min())
    max_val = max(df_mean[_calibration_var].max(), df_mean['model_cost'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='45-degree line')  # Red dashed line

    # Add labels for each calibration_category
    for i, label in enumerate(df_mean.index):
        plt.annotate(label, (df_mean[_calibration_var].iloc[i], df_mean['model_cost'].iloc[i]))

    # Add labels and title
    plt.xlabel('Actual Expenditure 2019')
    plt.ylabel('Model Cost (with confidence interval)')
    plt.title(f'Model Cost vs {_calibration_var}')

    # Show the plot
    plt.tight_layout()
    cost_subcategory = [name for name in globals() if globals()[name] is _costs_included][0]
    cost_subcategory = cost_subcategory.replace('list_of_', '').replace('_for_calibration', '')
    plt.savefig(calibration_outputs_folder / f'calibration_{_calibration_var}_{cost_subcategory}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

for var in ['actual_expenditure_2019', 'max_annual_budget_2020-22']:
    do_cost_calibration_plot(calibration_data, list_of_consumables_costs_for_calibration_only_hiv, var)
    do_cost_calibration_plot(calibration_data, list_of_consumables_costs_for_calibration_without_hiv, var)
    do_cost_calibration_plot(calibration_data, list_of_hr_costs_for_calibration, var)
    do_cost_calibration_plot(calibration_data, list_of_equipment_costs_for_calibration, var)


'''
