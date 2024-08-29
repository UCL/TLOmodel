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
    parse_log_file,
    unflatten_flattened_multi_index_in_logging
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
figurespath = costing_outputs_folder / "figures"
if not os.path.exists(figurespath):
    os.makedirs(figurespath)

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2015, 1, 1), Date(2015, 12, 31)) # TODO allow for multi-year costing
def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

# %% Gathering basic information
# Load result files
#-------------------
#results_folder = get_scenario_outputs('example_costing_scenario.py', outputfilepath)[0] # impact_of_cons_regression_scenarios
results_folder = get_scenario_outputs('long_run_all_diseases.py', outputfilepath)[0] # impact_of_cons_regression_scenarios
#results_folder = get_scenario_outputs('scenario_impact_of_consumables_availability.py', outputfilepath)[0] # impact_of_cons_regression_scenarios
equipment_results_folder = Path('./outputs/sakshi.mohan@york.ac.uk/021_long_run_all_diseases_run')
consumables_results_folder = Path('./outputs/sakshi.mohan@york.ac.uk/impact_of_consumables_scenarios-2024-06-11T204007Z/')
# TODO When the costing module is ready the above results_folder should be the same for the calculation of all costs

# check can read results from draw=0, run=0
log_equipment = load_pickled_dataframes(equipment_results_folder, 0, 0)

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# Load cost input files
#------------------------
# Load primary costing resourcefile
workbook_cost = pd.read_excel((resourcefilepath / "costing/ResourceFile_Costing.xlsx"),
                                    sheet_name = None)

# Extract districts and facility levels from the Master Facility List
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = set(mfl.Facility_Level)

# Extract count of facilities from Actual Facilities List
#afl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Actual_Facilities_List.csv")

#%% Calculate financial costs
# 1. HR cost
# Load annual salary by officer type and facility level
hr_cost_parameters = workbook_cost["human_resources"]
hr_cost_parameters['Facility_Level'] =  hr_cost_parameters['Facility_Level'].astype(str)
hr_annual_salary = hr_cost_parameters[hr_cost_parameters['Parameter_name'] == 'salary_usd']
hr_annual_salary['OfficerType_FacilityLevel'] = 'Officer_Type=' + hr_annual_salary['Officer_Category'].astype(str) + '|Facility_Level=' + hr_annual_salary['Facility_Level'].astype(str) # create column for merging with model log

# Load scenario staffing level
hr_scenario = log[ 'tlo.scenario']['override_parameter']['new_value'][log[ 'tlo.scenario'][ 'override_parameter']['name'] == 'use_funded_or_actual_staffing']

if hr_scenario.empty:
    staff_count = pd.read_csv(
        resourcefilepath / "healthsystem/human_resources/actual/ResourceFile_Daily_Capabilities.csv") # if missing default to reading actual capabilities
else:
    staff_count = pd.read_csv(
        resourcefilepath / 'healthsystem'/ 'human_resources' / f'{hr_scenario[2]}' / 'ResourceFile_Daily_Capabilities.csv')

staff_count_by_level_and_officer_type = staff_count.groupby(['Facility_Level', 'Officer_Category'])[
    'Staff_Count'].sum().reset_index()
staff_count_by_level_and_officer_type['Facility_Level'] = staff_count_by_level_and_officer_type['Facility_Level'].astype(str)

# Check if any cadres were not utilised at particular levels of care in the simulation
def expand_capacity_by_officer_type_and_facility_level(_df: pd.Series) -> pd.Series:
    """Summarise the parsed logged-key results for one draw (as dataframe) into a pd.Series."""
    _df = _df.set_axis(_df['date'].dt.year).drop(columns=['date'])
    _df.index.name = 'year'
    return unflatten_flattened_multi_index_in_logging(_df).stack(level=[0, 1])  # expanded flattened axis

annual_capacity_used_by_cadre_and_level = summarize(extract_results(
    Path(results_folder),
    module='tlo.methods.healthsystem.summary',
    key='Capacity_By_OfficerType_And_FacilityLevel',
    custom_generate_series=expand_capacity_by_officer_type_and_facility_level,
    do_scaling=False,
), only_mean=True, collapse_columns=True)

# Take mean across the entire simulation
average_capacity_used_by_cadre_and_level = annual_capacity_used_by_cadre_and_level.groupby(['OfficerType', 'FacilityLevel']).mean().reset_index(drop=False)
# Unstack to make it look like a nice table
average_capacity_used_by_cadre_and_level['OfficerType_FacilityLevel'] = 'Officer_Type=' + average_capacity_used_by_cadre_and_level['OfficerType'].astype(str) + '|Facility_Level=' + average_capacity_used_by_cadre_and_level['FacilityLevel'].astype(str)
list_of_cadre_and_level_combinations_used = average_capacity_used_by_cadre_and_level[average_capacity_used_by_cadre_and_level['mean'] != 0]['OfficerType_FacilityLevel']
print(f"Out of {len(average_capacity_used_by_cadre_and_level)} cadre and level combinations available, {len(list_of_cadre_and_level_combinations_used)} are used in the simulation")

# Subset scenario staffing level to only include cadre-level combinations used in the simulation
staff_count_by_level_and_officer_type['OfficerType_FacilityLevel'] = 'Officer_Type=' + staff_count_by_level_and_officer_type['Officer_Category'].astype(str) + '|Facility_Level=' + staff_count_by_level_and_officer_type['Facility_Level'].astype(str)
used_staff_count_by_level_and_officer_type = staff_count_by_level_and_officer_type[staff_count_by_level_and_officer_type['OfficerType_FacilityLevel'].isin(list_of_cadre_and_level_combinations_used)]

# Calculate various components of HR cost
# 1.1 Salary cost for current total staff
#---------------------------------------------------------------------------------------------------------------
staff_count_by_level_and_officer_type = staff_count_by_level_and_officer_type.drop(staff_count_by_level_and_officer_type[staff_count_by_level_and_officer_type.Facility_Level == '5'].index) # drop headquarters because we're only concerned with staff engaged in service delivery
salary_for_all_staff = pd.merge(staff_count_by_level_and_officer_type[['OfficerType_FacilityLevel', 'Staff_Count']],
                                     hr_annual_salary[['OfficerType_FacilityLevel', 'Value']], on = ['OfficerType_FacilityLevel'], how = "left")
salary_for_all_staff['Cost'] = salary_for_all_staff['Value'] * salary_for_all_staff['Staff_Count']
total_salary_for_all_staff = salary_for_all_staff['Cost'].sum()

# 1.2 Salary cost for health workforce cadres used in the simulation (Staff count X Annual salary)
#---------------------------------------------------------------------------------------------------------------
salary_for_staff_used_in_scenario = pd.merge(used_staff_count_by_level_and_officer_type[['OfficerType_FacilityLevel', 'Staff_Count']],
                                     hr_annual_salary[['OfficerType_FacilityLevel', 'Value']], on = ['OfficerType_FacilityLevel'], how = "left")
salary_for_staff_used_in_scenario['Cost'] = salary_for_staff_used_in_scenario['Value'] * salary_for_staff_used_in_scenario['Staff_Count']
total_salary_for_staff_used_in_scenario = salary_for_staff_used_in_scenario['Cost'].sum()

# Bar chart of salaries by cadre which goes into the HR folder in outputs (stacked for levels of care and two series for modelled and all)
def get_level_and_cadre_from_concatenated_value(_df, varname):
    _df['Cadre'] = _df[varname].str.extract(r'=(.*?)\|')
    _df['Facility_Level'] = _df[varname].str.extract(r'^[^=]*=[^|]*\|[^=]*=([^|]*)')
    return _df
def plot_cost_by_cadre_and_level(_df, figname_prefix, figname_suffix):
    if ('Facility_Level' in _df.columns) & ('Cadre' in _df.columns):
        pass
    else:
        _df = get_level_and_cadre_from_concatenated_value(_df, 'OfficerType_FacilityLevel')

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
    plt.savefig(figurespath / f'{figname_prefix}_by_cadre_and_level_{figname_suffix}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

plot_cost_by_cadre_and_level(salary_for_all_staff,figname_prefix = "salary", figname_suffix= "all_staff")
plot_cost_by_cadre_and_level(salary_for_staff_used_in_scenario,figname_prefix = "salary", figname_suffix= "staff_used_in_scenario")

# 1.3 Recruitment cost to fill gap created by attrition
#---------------------------------------------------------------------------------------------------------------
def merge_cost_and_model_data(cost_df, model_df, varnames):
    merged_df = model_df.copy()
    for varname in varnames:
        new_cost_df = cost_df[cost_df['Parameter_name'] == varname][['Officer_Category', 'Facility_Level', 'Value']]
        new_cost_df = new_cost_df.rename(columns={"Value": varname})
        if ((new_cost_df['Officer_Category'] == 'All').all()) and ((new_cost_df['Facility_Level'] == 'All').all()):
            merged_df[varname] = new_cost_df[varname].mean()
        elif ((new_cost_df['Officer_Category'] == 'All').all()) and ((new_cost_df['Facility_Level'] == 'All').all() == False):
            merged_df = pd.merge(merged_df, new_cost_df[['Facility_Level',varname]], on=['Facility_Level'], how="left")
        elif ((new_cost_df['Officer_Category'] == 'All').all() == False) and ((new_cost_df['Facility_Level'] == 'All').all()):
            merged_df = pd.merge(merged_df, new_cost_df[['Officer_Category',varname]], on=['Officer_Category'], how="left")
        else:
            merged_df = pd.merge(merged_df, new_cost_df, on=['Officer_Category', 'Facility_Level'], how="left")
    return merged_df

recruitment_cost = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = staff_count_by_level_and_officer_type,
                                                     varnames = ['annual_attrition_rate', 'recruitment_cost_per_person_recruited_usd'])
recruitment_cost['Cost'] = recruitment_cost['annual_attrition_rate'] * recruitment_cost['Staff_Count'] * \
                      recruitment_cost['recruitment_cost_per_person_recruited_usd']
total_recruitment_cost_for_attrited_workers = recruitment_cost['Cost'].sum()

plot_cost_by_cadre_and_level(recruitment_cost, figname_prefix = "recruitment", figname_suffix= "all_staff")

# 1.4 Pre-service training cost to fill gap created by attrition
#---------------------------------------------------------------------------------------------------------------
preservice_training_cost = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = staff_count_by_level_and_officer_type,
                                                     varnames = ['annual_attrition_rate',
                                                                 'licensure_exam_passing_rate', 'graduation_rate',
                                                                 'absorption_rate_of_students_into_public_workforce', 'proportion_of_workforce_recruited_from_abroad',
                                                                 'annual_preservice_training_cost_percapita_usd'])
preservice_training_cost['Cost'] = preservice_training_cost['annual_attrition_rate'] * preservice_training_cost['Staff_Count'] * \
                                                (1/(preservice_training_cost['absorption_rate_of_students_into_public_workforce'] + preservice_training_cost['proportion_of_workforce_recruited_from_abroad'])) * \
                                                (1/preservice_training_cost['graduation_rate']) * (1/preservice_training_cost['licensure_exam_passing_rate']) * \
                                                preservice_training_cost['annual_preservice_training_cost_percapita_usd']
preservice_training_cost_for_attrited_workers = preservice_training_cost['Cost'].sum()

plot_cost_by_cadre_and_level(preservice_training_cost, figname_prefix = "pre-service training", figname_suffix= "all_staff")

# 1.5 In-service training cost to train all staff
#---------------------------------------------------------------------------------------------------------------
inservice_training_cost = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = staff_count_by_level_and_officer_type,
                                                     varnames = ['annual_inservice_training_cost_usd'])
inservice_training_cost['Cost'] = inservice_training_cost['Staff_Count'] * inservice_training_cost['annual_inservice_training_cost_usd']
inservice_training_cost_for_all_staff = inservice_training_cost['Cost'].sum()

plot_cost_by_cadre_and_level(inservice_training_cost, figname_prefix = "in-service training", figname_suffix= "all_staff")

# TODO check why annual_inservice_training_cost for DCSA is NaN in the merged_df

# Create a dataframe to store financial costs
hr_cost_subcategories = ['salary_for_all_staff', 'recruitment_cost',
                         'preservice_training_cost', 'inservice_training_cost']
scenario_cost = pd.DataFrame({
    'Cost_Category': ['Human Resources for Health'] * len(hr_cost_subcategories),
    'Cost_Sub-category': hr_cost_subcategories,
    'Cost': [salary_for_all_staff['Cost'].sum(), recruitment_cost['Cost'].sum(),
                      preservice_training_cost['Cost'].sum(), preservice_training_cost['Cost'].sum()]
})
# TODO 'Value_2023USD' - use hr_cost_subcategories rather than the hardcoded list
# TODO Consider calculating economic cost of HR by multiplying salary times staff count with cadres_utilisation_rate

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

# %%
# 2. Consumables cost
def get_quantity_of_consumables_dispensed(results_folder):
    def get_counts_of_items_requested(_df):
        _df = drop_outside_period(_df)
        counts_of_available = defaultdict(int)
        counts_of_not_available = defaultdict(int)
        for _, row in _df.iterrows():
            for item, num in row['Item_Available'].items():
                counts_of_available[item] += num
            for item, num in row['Item_NotAvailable'].items():
                counts_of_not_available[item] += num
        return pd.concat(
            {'Available': pd.Series(counts_of_available), 'Not_Available': pd.Series(counts_of_not_available)},
            axis=1
        ).fillna(0).astype(int).stack()

    cons_req = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='Consumables',
            custom_generate_series=get_counts_of_items_requested,
            do_scaling=True)
    )

    cons_dispensed = cons_req.xs("Available", level=1) # only keep actual dispensed amount, i.e. when available
    return cons_dispensed

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

# Extract cost to .csv
def convert_dict_to_dataframe(_dict):
    data = {key: [value] for key, value in _dict.items()}
    _df = pd.DataFrame(data)
    return _df

cost_perfect_df = convert_dict_to_dataframe(cost_of_consumables_dispensed_under_perfect_availability).T.rename(columns = {0:"cost_perfect_availability"}).round(2)
cost_default_df = convert_dict_to_dataframe(cost_of_consumables_dispensed_under_default_availability).T.rename(columns = {0:"cost_default_availability"}).round(2)
unit_cost_df = convert_dict_to_dataframe(unit_price_consumable).T.rename(columns = {0:"unit_cost"})
dispensed_default_df = convert_dict_to_dataframe(consumables_dispensed_under_default_availability).T.rename(columns = {0:"dispensed_default_availability"}).round(2)
dispensed_perfect_df = convert_dict_to_dataframe(consumables_dispensed_under_perfect_availability).T.rename(columns = {0:"dispensed_perfect_availability"}).round(2)

full_cons_cost_df = pd.merge(cost_perfect_df, cost_default_df, left_index=True, right_index=True)
full_cons_cost_df = pd.merge(full_cons_cost_df, unit_cost_df, left_index=True, right_index=True)
full_cons_cost_df = pd.merge(full_cons_cost_df, dispensed_default_df, left_index=True, right_index=True)
full_cons_cost_df = pd.merge(full_cons_cost_df, dispensed_perfect_df, left_index=True, right_index=True)
full_cons_cost_df = full_cons_cost_df.reset_index().rename(columns = {'index' : 'item_code'})
full_cons_cost_df.to_csv(figurespath / 'consumables_cost_220824.csv')

# Import data for plotting
tlo_lmis_mapping = pd.read_csv(path_for_new_resourcefiles / 'ResourceFile_consumables_matched.csv', low_memory=False, encoding="ISO-8859-1")[['item_code', 'module_name', 'consumable_name_tlo']]
tlo_lmis_mapping = tlo_lmis_mapping[~tlo_lmis_mapping['item_code'].duplicated(keep='first')]
full_cons_cost_df = pd.merge(full_cons_cost_df, tlo_lmis_mapping, on = 'item_code', how = 'left', validate = "1:1")

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

# Bar plot of cost by category
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

plot_consumable_cost(_df = full_cons_cost_df,suffix =  'perfect_availability', groupby_var = 'category')
plot_consumable_cost(_df = full_cons_cost_df, suffix =  'default_availability', groupby_var = 'category')

# Plot the 10 consumables with the highest cost
plot_consumable_cost(_df = full_cons_cost_df,suffix =  'perfect_availability', groupby_var = 'consumable_name_tlo', top_x_values = 10)
plot_consumable_cost(_df = full_cons_cost_df,suffix =  'default_availability', groupby_var = 'consumable_name_tlo', top_x_values = 10)


# 2.2 Cost of consumables stocked (quantity needed for what is dispensed)
#---------------------------------------------------------------------------------------------------------------
# Stocked amount should be higher than dispensed because of i. excess capacity, ii. theft, iii. expiry
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
                                                cons_dispensed[key] *
                                                inflow_to_outflow_ratio_by_consumable.get(key, average_inflow_to_outflow_ratio_ratio)
                                                for key in cons_dispensed)))
# TODO Make sure that the above code runs
total_cost_of_consumables_stocked = sum(value for value in cost_of_consumables_stocked.values() if not np.isnan(value))

# Add consumable costs to the financial cost dataframe
consumable_cost_subcategories = ['total_cost_of_consumables_dispensed', 'total_cost_of_consumables_stocked']
consumable_costs = pd.DataFrame({
    'Cost_Category': ['Consumables'] * len(consumable_cost_subcategories),
    'Cost_Sub-category': consumable_cost_subcategories,
    'Value_2023USD': [total_cost_of_consumables_dispensed, total_cost_of_consumables_stocked]
})
# Append new_data to scenario_cost_financial
scenario_cost_financial = pd.concat([scenario_cost_financial, consumable_costs], ignore_index=True)

# 3. Equipment cost
# Total cost of equipment required as per SEL (HSSP-III) only at facility IDs where it been used in the simulation
unit_cost_equipment = workbook_cost["equipment"]
unit_cost_equipment =   unit_cost_equipment.rename(columns=unit_cost_equipment.iloc[7]).reset_index(drop=True).iloc[8:]
# Calculate necessary costs based on HSSP-III assumptions
unit_cost_equipment['service_fee_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.8 / 8 if row['unit_purchase_cost'] > 1000 else 0, axis=1) # 80% of the value of the item over 8 years
unit_cost_equipment['spare_parts_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.2 / 8 if row['unit_purchase_cost'] > 1000 else 0, axis=1) # 20% of the value of the item over 8 years
unit_cost_equipment['upfront_repair_cost_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.2 * 0.2 / 8 if row['unit_purchase_cost'] < 250000 else 0, axis=1) # 20% of the value of 20% of the items over 8 years
unit_cost_equipment['replacement_cost_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] * 0.1 / 8 if row['unit_purchase_cost'] < 250000 else 0, axis=1) # 10% of the items over 8 years

unit_cost_equipment = unit_cost_equipment[['Item_code','Equipment_tlo',
                                           'service_fee_annual', 'spare_parts_annual',  'upfront_repair_cost_annual', 'replacement_cost_annual',
                                           'Health Post_prioritised', 'Community_prioritised', 'Health Center_prioritised', 'District_prioritised', 'Central_prioritised']]
unit_cost_equipment = unit_cost_equipment.rename(columns={col: 'Quantity_' + col.replace('_prioritised', '') for col in unit_cost_equipment.columns if col.endswith('_prioritised')})
unit_cost_equipment = unit_cost_equipment.rename(columns={col: col.replace(' ', '_') for col in unit_cost_equipment.columns})
unit_cost_equipment = unit_cost_equipment[unit_cost_equipment.Item_code.notna()]

unit_cost_equipment = pd.wide_to_long(unit_cost_equipment, stubnames=['Quantity_'],
                          i=['Item_code', 'Equipment_tlo', 'service_fee_annual', 'spare_parts_annual', 'upfront_repair_cost_annual', 'replacement_cost_annual'],
                          j='Facility_Level', suffix='(\d+|\w+)').reset_index()
facility_level_mapping = {'Health_Post': '0', 'Health_Center': '1a', 'Community': '1b', 'District': '2', 'Central': '3'}
unit_cost_equipment['Facility_Level'] = unit_cost_equipment['Facility_Level'].replace(facility_level_mapping)
unit_cost_equipment = unit_cost_equipment.rename(columns = {'Quantity_': 'Quantity'})
#unit_cost_equipment_small  = unit_cost_equipment[['Item_code', 'Facility_Level', 'Quantity','service_fee_annual', 'spare_parts_annual', 'upfront_repair_cost_annual', 'replacement_cost_annual']]
#equipment_cost_dict = unit_cost_equipment_small.groupby('Facility_Level').apply(lambda x: x.to_dict(orient='records')).to_dict()

# Get list of equipment used by district and level
equip = pd.DataFrame(
    log_equipment['tlo.methods.healthsystem.summary']['EquipmentEverUsed_ByFacilityID']
)

equip['EquipmentEverUsed'] = equip['EquipmentEverUsed'].apply(ast.literal_eval)
equip.loc[equip.Facility_Level.isin(['3', '4', '5']),'District'] = 'Central' # Assign a district name for Central health facilities
districts.add('Central')

# Extract a list of equipment which was used at each facility level within each district
equipment_used = {district: {level: [] for level in fac_levels} for district in districts} # create a dictionary with a key for each district and facility level

for dist in districts:
    for level in fac_levels:
        equip_subset = equip[(equip['District'] == dist) & (equip['Facility_Level'] == level)]
        equipment_used[dist][level] = set().union(*equip_subset['EquipmentEverUsed'])
equipment_used = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index') for k, v in equipment_used.items()},
        axis=0)
list_of_equipment_used = set().union(*equip['EquipmentEverUsed'])

equipment_df = pd.DataFrame()
equipment_df.index = equipment_used.index
for item in list_of_equipment_used:
    equipment_df[str(item)] = 0
    for dist_fac_index in equipment_df.index:
        equipment_df.loc[equipment_df.index == dist_fac_index, str(item)] = equipment_used[equipment_used.index == dist_fac_index].isin([item]).any(axis=1)
equipment_df.to_csv('./outputs/equipment_use.csv')

equipment_df = equipment_df.reset_index().rename(columns = {'level_0' : 'District', 'level_1': 'Facility_Level'})
equipment_df = pd.melt(equipment_df, id_vars = ['District', 'Facility_Level']).rename(columns = {'variable': 'Item_code', 'value': 'whether_item_was_used'})
equipment_df['Item_code'] = pd.to_numeric(equipment_df['Item_code'])

# Merge the two datasets to calculate cost
equipment_cost = pd.merge(equipment_df, unit_cost_equipment[['Item_code', 'Equipment_tlo', 'Facility_Level', 'Quantity','service_fee_annual', 'spare_parts_annual', 'upfront_repair_cost_annual', 'replacement_cost_annual']],
                          on = ['Item_code', 'Facility_Level'], how = 'left', validate = "m:1")
categories_of_equipment_cost = ['replacement_cost', 'upfront_repair_cost', 'spare_parts', 'service_fee']
for cost_category in categories_of_equipment_cost:
    equipment_cost['total_' + cost_category] = equipment_cost[cost_category + '_annual'] * equipment_cost['whether_item_was_used'] * equipment_cost['Quantity']
equipment_cost['annual_cost'] = equipment_cost[['total_' + item for item in categories_of_equipment_cost]].sum(axis = 1)
#equipment_cost.to_csv('./outputs/equipment_cost.csv')

equipment_costs = pd.DataFrame({
    'Cost_Category': ['Equipment'] * len(categories_of_equipment_cost),
    'Cost_Sub-category': categories_of_equipment_cost,
    'Value_2023USD': equipment_cost[['total_' + item for item in categories_of_equipment_cost]].sum().values.tolist()
})
# Append new_data to scenario_cost_financial
scenario_cost_financial = pd.concat([scenario_cost_financial, equipment_costs], ignore_index=True)

# TODO Use AFL to multiple the number of facilities at each level
# TODO PLot which equipment is used by district and facility or a heatmap of the number of facilities at which an equipment is used
# TODO From the log, extract the facility IDs which use any equipment item
# TODO Collapse facility IDs by level of care to get the total number of facilities at each level using an item
# TODO Multiply number of facilities by level with the quantity needed of each equipment and collapse to get total number of equipment (nationally)
# TODO Multiply quantity needed with cost per item (this is the repair, replacement, and maintenance cost)
# TODO Which equipment needs to be newly purchased (currently no assumption made for equipment with cost > $250,000)

# 4. Facility running costs
# Average running costs by facility level and district times the number of facilities  in the simulation

# Extract all costs to a .csv
scenario_cost_financial.to_csv(costing_outputs_folder / 'scenario_cost.csv')


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
