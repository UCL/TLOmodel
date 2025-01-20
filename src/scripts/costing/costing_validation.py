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
from scripts.costing.cost_estimation import (estimate_input_cost_of_scenarios,
                                             do_stacked_bar_plot_of_cost_by_category)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Establish common paths
resourcefilepath = Path("./resources")

# Steps: 1. Create a mapping of data labels in model_costing and relevant calibration data, 2. Create a dataframe with model_costs and calibration costs;
# Load costing resourcefile
workbook_cost = pd.read_excel((resourcefilepath / "costing/ResourceFile_Costing.xlsx"),
                              sheet_name=None)
# Prepare data for calibration
calibration_data = workbook_cost["resource_mapping_r7_summary"]
# Make sure values are numeric
budget_columns = ['BUDGETS (USD) (Jul 2019 - Jun 2020)', 'BUDGETS (USD) (Jul 2020 - Jun 2021)',
       'BUDGETS (USD) (Jul 2021 - Jun 2022)']
expenditure_columns = ['EXPENDITURE (USD) (Jul 2018 - Jun 2019)']
calibration_data[budget_columns + expenditure_columns] = calibration_data[budget_columns + expenditure_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))
# For calibration to budget figures, we take the maximum value across the three years in the RM to provide an
# the maximum of the budget between 2020 and 2022 provides the upper limit to calibrate to (expenditure providing the lower limit)
calibration_data['max_annual_budget_2020-22'] = calibration_data[budget_columns].max(axis=1, skipna = True)
calibration_data = calibration_data.rename(columns = {'EXPENDITURE (USD) (Jul 2018 - Jun 2019)': 'actual_expenditure_2019',
                                                      'Calibration_category': 'calibration_category'})
calibration_data = calibration_data[['calibration_category','actual_expenditure_2019', 'max_annual_budget_2020-22']]
calibration_data = calibration_data.groupby('calibration_category')[['actual_expenditure_2019', 'max_annual_budget_2020-22']].sum().reset_index()
# Repeat this dataframe three times to map to the lower, upper and mean stats in the cost data
calibration_data1 = calibration_data.copy()
calibration_data1['stat'] = 'lower'
calibration_data2 = calibration_data.copy()
calibration_data2['stat'] = 'mean'
calibration_data3 = calibration_data.copy()
calibration_data3['stat'] = 'upper'
calibration_data = pd.concat([calibration_data1, calibration_data2, calibration_data3], axis = 0)
calibration_data = calibration_data.set_index(['calibration_category', 'stat'])

# %%
# Estimate cost for validation
#-----------------------------
# Load result files
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/t.mangal@imperial.ac.uk')
results_folder = get_scenario_outputs('hss_elements-2024-11-12T172311Z.py', outputfilepath)[0]

# Estimate costs for 2018
input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath, _years = [2018], _draws = [0], summarize = True, cost_only_used_staff=False)
#input_costs = input_costs[input_costs.year == 2018]

# Manually create a dataframe of model costs and relevant calibration values
def assign_item_codes_to_consumables(_df):
    path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"
    # Retain only consumable costs
    _df = _df[_df['cost_category'] == 'medical consumables']

    '''
    consumables_dict = pd.read_csv(path_for_consumable_resourcefiles / 'ResourceFile_consumables_matched.csv', low_memory=False,
                                 encoding="ISO-8859-1")[['item_code', 'consumable_name_tlo']]
    consumables_dict = consumables_dict.rename(columns = {'item_code': 'Item_Code'})
    consumables_dict = dict(zip(consumables_dict['consumable_name_tlo'], consumables_dict['Item_Code']))
    '''

    # Create dictionary mapping item_codes to consumables names
    consumables_df = workbook_cost["consumables"]
    consumables_df = consumables_df.rename(columns=consumables_df.iloc[0])
    consumables_df = consumables_df[['Item_Code', 'Consumable_name_tlo']].reset_index(
        drop=True).iloc[1:]
    consumables_df = consumables_df[consumables_df['Item_Code'].notna()]
    consumables_dict = dict(zip(consumables_df['Consumable_name_tlo'], consumables_df['Item_Code']))

    # Replace consumable_name_tlo with item_code
    _df = _df.copy()
    _df['cost_subgroup'] = _df['cost_subgroup'].map(consumables_dict)

    return _df

def get_calibration_relevant_subset_of_costs(_df, _col, _col_value, _calibration_category):
    if (len(_col_value) == 1):
        _df = _df[_df[_col] == _col_value[0]]
    else:
        _df = _df[_df[_col].isin(_col_value)]
    _df['calibration_category'] = _calibration_category
    return _df.groupby(['calibration_category' ,'stat'])['cost'].sum()

'''
def get_calibration_relevant_subset_of_consumables_cost(_df, item):
    for col in ['Item_Code', 'Final_price_per_chosen_unit (USD, 2023)', 'excess_stock_proportion_of_dispensed','item_code']:
        try:
            _df = _df.drop(columns = col)
        except:
            pass
    _df.columns = pd.MultiIndex.from_tuples(_df.columns)
    _df = _df.melt(id_vars = ['year', 'Item_Code'], var_name=['draw', 'stat'], value_name='value')
    _df = _df[_df['Item_Code'].isin(item)]
    _df = _df.groupby(['year', 'draw', 'stat'])['value'].sum()
    return _df.reset_index()
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
'''

# Consumables
#-----------------------------------------------------------------------------------------------------------------------
calibration_data['model_cost'] = np.nan
consumables_costs_by_item_code = assign_item_codes_to_consumables(input_costs)

irs = [161]
bednets = [160]
undernutrition = [213, 1220, 1221, 1223, 1227]
cervical_cancer = [261, 1239]
other_family_planning = [1, 3,7,12,13]
vaccines = [150, 151, 153, 155, 157, 158, 1197]
art = [2671, 2672, 2673]
tb_treatment = [176, 177, 179, 178, 181, 2678]
antimalarials = [162,164,170]
malaria_rdts = [163]
hiv_screening = [190,191,196]
condoms = [2,25]
tb_tests = [184,187, 175]
circumcision = [197]
other_drugs = set(consumables_costs_by_item_code['cost_subgroup'].unique()) - set(irs) - set(bednets) - set(undernutrition) - set(cervical_cancer) - set(other_family_planning) - set(vaccines) \
              - set(art) - set(tb_treatment) - set(antimalarials) - set(malaria_rdts) - set(hiv_screening)\
              - set(condoms) - set(tb_tests)

# Note that the main ARV  regimen in 2018 was tenofovir/lamivudine/efavirenz as opposed to Tenofovir/Lamivudine/Dolutegravir as used in the RF_Costing. The price of this
# was $80 per year (80/(0.103*365)) times what's estimated by the model so let's update this
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = art, _calibration_category = 'Antiretrovirals')*  80/(0.103*365))
# Other consumables costs do not need to be adjusted
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = irs, _calibration_category = 'Indoor Residual Spray'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = bednets, _calibration_category = 'Bednets'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = undernutrition, _calibration_category = 'Undernutrition commodities'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = cervical_cancer, _calibration_category = 'Cervical Cancer'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = other_family_planning, _calibration_category = 'Other family planning commodities'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = vaccines, _calibration_category = 'Vaccines'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = tb_treatment, _calibration_category = 'TB Treatment'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = antimalarials, _calibration_category = 'Antimalarials'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = malaria_rdts, _calibration_category = 'Malaria RDTs'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = hiv_screening, _calibration_category = 'HIV Screening/Diagnostic Tests'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = condoms, _calibration_category = 'Condoms and Lubricants'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = tb_tests, _calibration_category = 'TB Tests (including RDTs)'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = other_drugs, _calibration_category = 'Other Drugs, medical supplies, and commodities'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = consumables_costs_by_item_code, _col = 'cost_subgroup', _col_value = circumcision, _calibration_category = 'Voluntary Male Medical Circumcision'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = input_costs, _col = 'cost_subcategory', _col_value = ['supply_chain'], _calibration_category = 'Supply Chain'))

# HR
#-----------------------------------------------------------------------------------------------------------------------
hr_costs = input_costs[input_costs['cost_category'] == 'human resources for health']
#ratio_of_all_to_used_staff = total_salary_for_all_staff[(0,2018)]/total_salary_for_staff_used_in_scenario[( 0, 'lower')][2018]
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = hr_costs, _col = 'cost_subcategory', _col_value = ['salary_for_all_staff'], _calibration_category = 'Health Worker Salaries'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = hr_costs, _col = 'cost_subcategory', _col_value = ['preservice_training_and_recruitment_cost_for_attrited_workers'], _calibration_category = 'Health Worker Training - Pre-Service')) # TODO remove recruitment costs?
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = hr_costs, _col = 'cost_subcategory', _col_value = ['inservice_training_cost_for_all_staff'], _calibration_category = 'Health Worker Training - In-Service'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = hr_costs, _col = 'cost_subcategory', _col_value = ['mentorship_and_supportive_cost_for_all_staff'], _calibration_category = 'Mentorships & Supportive Supervision'))

# Equipment
equipment_costs = input_costs[input_costs['cost_category'] == 'medical equipment']
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = equipment_costs, _col = 'cost_subcategory', _col_value = ['replacement_cost_annual_total'], _calibration_category = 'Medical Equipment - Purchase'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = equipment_costs, _col = 'cost_subcategory',
                                                                                                                _col_value = ['service_fee_annual_total', 'spare_parts_annual_total','major_corrective_maintenance_cost_annual_total'],
                                                                                                                _calibration_category = 'Medical Equipment - Maintenance'))
#calibration_data[calibration_data['calibration_category'] == 'Vehicles - Purchase and Maintenance'] = get_calibration_relevant_subset()
#calibration_data[calibration_data['calibration_category'] == 'Vehicles - Purchase and Maintenance'] = get_calibration_relevant_subset()

# Facility operation costs
#-----------------------------------------------------------------------------------------------------------------------
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = input_costs, _col = 'cost_subgroup', _col_value = ['Electricity', 'Water', 'Cleaning', 'Security', 'Food for inpatient cases', 'Facility management'], _calibration_category = 'Facility utility bills'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = input_costs, _col = 'cost_subgroup', _col_value = ['Building maintenance'], _calibration_category = 'Infrastructure - Rehabilitation'))
calibration_data['model_cost'] = calibration_data['model_cost'].fillna(get_calibration_relevant_subset_of_costs(_df = input_costs, _col = 'cost_subgroup', _col_value = ['Vehicle maintenance', 'Ambulance fuel'], _calibration_category = 'Vehicles - Fuel and Maintenance'))

# %%
# 3. Create calibration plot
list_of_consumables_costs_for_calibration_only_hiv = ['Voluntary Male Medical Circumcision', 'HIV Screening/Diagnostic Tests', 'Antiretrovirals']
list_of_consumables_costs_for_calibration_without_hiv =['Indoor Residual Spray', 'Bednets', 'Malaria RDTs', 'Antimalarials', 'TB Tests (including RDTs)', 'TB Treatment', 'Vaccines',
                                                        'Condoms and Lubricants', 'Other family planning commodities',
                                                        'Undernutrition commodities', 'Cervical Cancer', 'Other Drugs, medical supplies, and commodities']
list_of_hr_costs_for_calibration = ['Health Worker Salaries', 'Health Worker Training - In-Service', 'Health Worker Training - Pre-Service', 'Mentorships & Supportive Supervision']
list_of_equipment_costs_for_calibration = ['Medical Equipment - Purchase', 'Medical Equipment - Maintenance']
list_of_operating_costs_for_calibration = ['Facility utility bills', 'Infrastructure - Rehabilitation', 'Vehicles - Maintenance','Vehicles - Fuel and Maintenance']

# Create folders to store results
costing_outputs_folder = Path('./outputs/costing')
if not os.path.exists(costing_outputs_folder):
    os.makedirs(costing_outputs_folder)
figurespath = costing_outputs_folder / "figures_post_cons_fix"
if not os.path.exists(figurespath):
    os.makedirs(figurespath)
calibration_outputs_folder = Path(figurespath / 'calibration')
if not os.path.exists(calibration_outputs_folder):
    os.makedirs(calibration_outputs_folder)

def do_cost_calibration_plot(_df, _costs_included, _xtick_fontsize = 10):
    # Filter the dataframe
    _df = _df[(_df.model_cost.notna()) & (_df.index.get_level_values(0).isin(_costs_included))]

    # Reorder the first level of the index based on _costs_included while keeping the second level intact
    _df.index = pd.MultiIndex.from_arrays([
        pd.CategoricalIndex(_df.index.get_level_values(0), categories=_costs_included, ordered=True),
        _df.index.get_level_values(1)
    ])
    _df = _df.sort_index()  # Apply the custom order by sorting the DataFrame

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
    yerr_lower = (df_mean['model_cost'] - df_lower['model_cost']).clip(lower = 0)
    yerr_upper = (df_upper['model_cost'] - df_mean['model_cost']).clip(lower = 0)
    plt.errorbar(df_mean.index, df_mean['model_cost'],
                 yerr=[yerr_lower, yerr_upper],
                 fmt='o', label='Model Cost', ecolor='gray', capsize=5, color='saddlebrown')

    # Plot annual_expenditure_2019 and max_annual_budget_2020-22 as dots
    plt.plot(df_mean.index, df_mean['actual_expenditure_2019'], 'bo', label='Actual Expenditure 2019', markersize=8)
    plt.plot(df_mean.index, df_mean['max_annual_budget_2020-22'], 'go', label='Max Annual Budget 2020-22', markersize=8)

    # Draw a blue line between annual_expenditure_2019 and max_annual_budget_2020-22
    plt.vlines(df_mean.index, df_mean['actual_expenditure_2019'], df_mean['max_annual_budget_2020-22'], color='blue',
               label='Expenditure-Budget Range')

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

    # Set a white background and black border
    plt.grid(False)
    ax = plt.gca()  # Get current axes
    ax.set_facecolor('white')  # Set the background color to white
    for spine in ax.spines.values():  # Iterate over all borders (spines)
        spine.set_edgecolor('black')  # Set the border color to black
        spine.set_linewidth(1.5)  # Adjust the border width if desired

    # Customize x-axis labels for readability
    max_label_length = 15  # Define a maximum label length for wrapping
    wrapped_labels = [textwrap.fill(str(label), max_label_length) for label in df_mean.index]
    plt.xticks(ticks=range(len(wrapped_labels)), labels=wrapped_labels, rotation=45, ha='right', fontsize=_xtick_fontsize)

    # Adding a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    # Tight layout and save the figure
    plt.tight_layout()
    plt.savefig(calibration_outputs_folder / f'calibration_dot_plot_{cost_subcategory}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

# Call the function for each variable and cost list
all_consumable_costs = list_of_consumables_costs_for_calibration_only_hiv + list_of_consumables_costs_for_calibration_without_hiv + ['Supply Chain']
all_calibration_costs = all_consumable_costs + list_of_hr_costs_for_calibration + list_of_equipment_costs_for_calibration + list_of_operating_costs_for_calibration

do_cost_calibration_plot(calibration_data,list_of_consumables_costs_for_calibration_without_hiv)
do_cost_calibration_plot(calibration_data,list_of_consumables_costs_for_calibration_only_hiv)
do_cost_calibration_plot(calibration_data,all_consumable_costs)
do_cost_calibration_plot(calibration_data, list_of_hr_costs_for_calibration)
do_cost_calibration_plot(calibration_data, list_of_equipment_costs_for_calibration)
do_cost_calibration_plot(calibration_data, list_of_operating_costs_for_calibration)
do_cost_calibration_plot(calibration_data,all_calibration_costs, _xtick_fontsize = 7)
calibration_data.to_csv(figurespath / 'calibration/calibration.csv')

# Extract calibration data table for manuscript appendix
calibration_data_extract = calibration_data[calibration_data.index.get_level_values(1) == 'mean']
calibration_data_extract = calibration_data_extract.droplevel(level=1).reset_index()
# Create a higher level cost category in the calibration data
calibration_categories_dict = {'Other Drugs, medical supplies, and commodities': 'medical consumables',
'Program Management & Administration': 'Not represented in TLO model',
'Non-EHP consumables': 'Not represented in TLO model',
'Voluntary Male Medical Circumcision': 'medical consumables',
'Indoor Residual Spray': 'medical consumables',
'Bednets': 'medical consumables',
'Antimalarials': 'medical consumables',
'Undernutrition commodities': 'medical consumables',
'Cervical Cancer': 'medical consumables',
'Condoms and Lubricants': 'medical consumables',
'Other family planning commodities': 'medical consumables',
'TB Tests (including RDTs)': 'medical consumables',
'TB Treatment': 'medical consumables',
'Vaccines': 'medical consumables',
'Malaria RDTs': 'medical consumables',
'HIV Screening/Diagnostic Tests': 'medical consumables',
'Antiretrovirals': 'medical consumables',
'Health Worker Salaries': 'human resources for health',
'Health Worker Training - In-Service': 'human resources for health',
'Health Worker Training - Pre-Service': 'human resources for health',
'Mentorships & Supportive Supervision': 'human resources for health',
'Facility utility bills': 'facility operating cost',
'Infrastructure - New Builds': 'Not represented in TLO model',
'Infrastructure - Rehabilitation': 'facility operating cost',
'Infrastructure - Upgrades': 'Not represented in TLO model',
'Medical Equipment - Maintenance': 'medical equipment',
'Medical Equipment - Purchase': 'medical equipment',
'Vehicles - Fuel and Maintenance': 'facility operating cost',
'Vehicles - Purchase': 'Not represented in TLO model',
'Vehicles - Fuel and Maintenance (Beyond Government and CHAM)': 'Not represented in TLO model',
'Supply Chain': 'medical consumables',
'Supply Chain - non-EHP consumables': 'Not represented in TLO model',
'Unclassified': 'Not represented in TLO model'}
calibration_data_extract['cost_category'] = calibration_data_extract['calibration_category'].map(calibration_categories_dict)

# Add a column show deviation from actual expenditure
calibration_data_extract['Deviation of estimated cost from actual expenditure (%)'] = (
    (calibration_data_extract['model_cost'] - calibration_data_extract['actual_expenditure_2019'])
    /calibration_data_extract['actual_expenditure_2019'])

# Format the deviation as a percentage with 2 decimal points
calibration_data_extract['Deviation of estimated cost from actual expenditure (%)'] = (
    calibration_data_extract['Deviation of estimated cost from actual expenditure (%)']
    .map(lambda x: f"{x * 100:.2f}%")
)
calibration_data_extract.loc[calibration_data_extract['Deviation of estimated cost from actual expenditure (%)'] == 'nan%', 'Deviation of estimated cost from actual expenditure (%)'] = 'NA'
# Replace if calibration is fine
calibration_condition_met = ((calibration_data_extract['model_cost'] > calibration_data_extract[['actual_expenditure_2019', 'max_annual_budget_2020-22']].min(axis=1)) &
    (calibration_data_extract['model_cost'] < calibration_data_extract[['actual_expenditure_2019', 'max_annual_budget_2020-22']].max(axis=1)))

calibration_data_extract.loc[calibration_condition_met,
    'Deviation of estimated cost from actual expenditure (%)'
] = 'Within target range'

calibration_data_extract.loc[calibration_data_extract['model_cost'].isna(), 'model_cost'] = 'NA'

calibration_data_extract = calibration_data_extract.sort_values(by=['cost_category', 'calibration_category'])
calibration_data_extract = calibration_data_extract[['cost_category', 'calibration_category', 'actual_expenditure_2019', 'max_annual_budget_2020-22', 'model_cost', 'Deviation of estimated cost from actual expenditure (%)']]
calibration_data_extract = calibration_data_extract.rename(columns = {'cost_category': 'Cost Category',
                                                            'calibration_category': 'Relevant RM group',
                                                            'actual_expenditure_2019': 'Recorded Expenditure (FY 2018/19)',
                                                            'max_annual_budget_2020-22': 'Maximum Recorded Annual Budget (FY 2019/20 - 2021/22)',
                                                            'model_cost': 'Estimated cost (TLO Model, 2018)'
    })
def convert_df_to_latex(_df, _longtable = False, numeric_columns = []):
    _df['Relevant RM group'] = _df['Relevant RM group'].str.replace('&', r'\&', regex=False)
    # Format numbers to the XX,XX,XXX.XX format for all numeric columns
    _df[numeric_columns] = _df[numeric_columns].applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

    # Convert to LaTeX format with horizontal lines after every row
    latex_table = _df.to_latex(
        longtable=_longtable,  # Use the longtable environment for large tables
        column_format='|R{3.5cm}|R{3.5cm}|R{2.1cm}|R{2.1cm}|R{2.1cm}|R{2.1cm}|',
        caption=f"Comparison of Model Estimates with Resource Mapping data",
        label=f"tab:calibration_breakdown",
        position="h",
        index=False,
        escape=False,  # Prevent escaping special characters like \n
        header=True
    )

    # Add \hline after the header and after every row for horizontal lines
    latex_table = latex_table.replace("\\\\", "\\\\ \\hline")  # Add \hline after each row
    latex_table = latex_table.replace("%", "\%")  # Add \hline after each row
    latex_table = latex_table.replace("Program Management & Administration", "Program Management \& Administration")  # Add \hline after each row
    latex_table = latex_table.replace("Mentorships & Supportive Supervision", "Mentorships \& Supportive Supervision")  # Add \hline after each row

    # latex_table = latex_table.replace("_", " ")  # Add \hline after each row

    # Specify the file path to save
    latex_file_path = calibration_outputs_folder / f'calibration_breakdown.tex'

    # Write to a file
    with open(latex_file_path, 'w') as latex_file:
        latex_file.write(latex_table)

    # Print latex for reference
    print(latex_table)

convert_df_to_latex(calibration_data_extract, _longtable = True, numeric_columns = ['Recorded Expenditure (FY 2018/19)',
                                                                                     'Maximum Recorded Annual Budget (FY 2019/20 - 2021/22)',
                                                                                     'Estimated cost (TLO Model, 2018)'])

# Stacked bar charts to represent all cost sub-groups
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'medical consumables',
                                        _disaggregate_by_subgroup = True,
                                        _outputfilepath = calibration_outputs_folder)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'human resources for health',
                                        _disaggregate_by_subgroup = True,
                                        _outputfilepath = calibration_outputs_folder)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'medical equipment',
                                        _disaggregate_by_subgroup = True,
                                        _outputfilepath = calibration_outputs_folder)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'other',
                                        _disaggregate_by_subgroup = True,
                                        _outputfilepath = calibration_outputs_folder)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'all',
                                        _disaggregate_by_subgroup = False,
                                        _outputfilepath = calibration_outputs_folder)



'''

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
