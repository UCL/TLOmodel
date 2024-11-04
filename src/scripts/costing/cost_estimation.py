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

#%%
def estimate_input_cost_of_scenarios(results_folder: Path, resourcefilepath: Path = None, draws = None, runs = None,
                                     summarize: bool = False, cost_only_used_staff: bool = True):
    # Useful common functions
    def drop_outside_period(_df):
        """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
        return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

    def melt_model_output_draws_and_runs(_df, id_vars):
        multi_index = pd.MultiIndex.from_tuples(_df.columns)
        _df.columns = multi_index
        melted_df = pd.melt(_df, id_vars=id_vars).rename(columns={'variable_0': 'draw', 'variable_1': 'run'})
        return melted_df

    # Define a relative pathway for relavant folders
    path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"

    # %% Gathering basic information
    # Load basic simulation parameters
    #-------------------------------------
    log = load_pickled_dataframes(results_folder, 0, 0)  # read from 1 draw and run
    info = get_scenario_info(results_folder)  # get basic information about the results
    if draws is None:
        draws = range(0, info['number_of_draws'])
    if runs is None:
        runs = range(0, info['runs_per_draw'])
    final_year_of_simulation = max(log['tlo.methods.healthsystem.summary']['hsi_event_counts']['date']).year
    first_year_of_simulation = min(log['tlo.methods.healthsystem.summary']['hsi_event_counts']['date']).year
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
    TARGET_PERIOD = (Date(first_year_of_simulation, 1, 1), Date(final_year_of_simulation, 12, 31)) # Declare period for which the results will be generated (defined inclusively)
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
        if 'cost_category' in _df.columns:
            columns_to_retain.append('cost_category')
        _df = _df[columns_to_retain]
        return _df
    def prepare_cost_dataframe(_df, _category_specific_group, _cost_category):
        _df = _df.rename(columns = {_category_specific_group: 'cost_subgroup'})
        _df['cost_category'] = _cost_category
        return retain_relevant_column_subset(_df, 'cost_subgroup')


    # CALCULATE ECONOMIC COSTS
    #%%
    # 1. HR cost
    #------------------------
    print("Now estimating HR costs...")
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

    if (cost_only_used_staff):
        print("The input for 'cost_only_used_staff' implies that only cadre-level combinations which have been used in the run are costed")
        staff_size_chosen_for_costing = used_staff_count_by_level_and_officer_type
    else:
        print("The input for 'cost_only_used_staff' implies that all staff are costed regardless of the cadre-level combinations which have been used in the run are costed")
        staff_size_chosen_for_costing = available_staff_count_by_level_and_officer_type

    # Calculate various components of HR cost
    # 1.1 Salary cost for health workforce cadres used in the simulation (Staff count X Annual salary)
    #---------------------------------------------------------------------------------------------------------------
    salary_for_staff = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = staff_size_chosen_for_costing,
                                                         varnames = ['salary_usd'])
    salary_for_staff['cost'] = salary_for_staff['salary_usd'] * salary_for_staff['staff_count']

    # 1.2 Pre-service training & recruitment cost to fill gap created by attrition
    #---------------------------------------------------------------------------------------------------------------
    preservice_training_cost = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = staff_size_chosen_for_costing,
                                                         varnames = ['annual_attrition_rate',
                                                                     'licensure_exam_passing_rate', 'graduation_rate',
                                                                     'absorption_rate_of_students_into_public_workforce', 'proportion_of_workforce_recruited_from_abroad',
                                                                     'average_annual_preservice_training_cost_for_cadre', 'preservice_training_duration', 'recruitment_cost_per_person_recruited_usd',
                                                                     'average_length_of_tenure_in_the_public_sector'])

    def calculate_npv_past_training_expenses_by_row(row, r = discount_rate):
        # Initialize the NPV for the row
        npv = 0
        annual_cost = row['average_annual_preservice_training_cost_for_cadre']
        full_years = int(row['preservice_training_duration'])  # Extract integer part of the year
        partial_year = row['preservice_training_duration'] - full_years  # Fractional part of the year

        # Iterate over each year of the training duration to calculate compounded cost to the present
        # Calculate NPV for each full year of training
        for t in range(full_years):
            npv += annual_cost * (1 + r) ** (t+1+1) # 1 added twice because range(4) is [0,1,2,3]

        # Account for the fractional year at the end if it exists
        if partial_year > 0:
            npv += annual_cost * partial_year * (1 + r) ** (1+r)

        # Add recruitment cost assuming this happens during the partial year or the year after graduation if partial year == 0
        npv += row['recruitment_cost_per_person_recruited_usd'] * (1+r)

        return npv

    # Calculate NPV for each row using iterrows and store in a new column
    npv_values = []
    for index, row in preservice_training_cost.iterrows():
        npv = calculate_npv_past_training_expenses_by_row(row, r=discount_rate)
        npv_values.append(npv)

    preservice_training_cost['npv_of_training_and_recruitment_cost'] = npv_values
    preservice_training_cost['npv_of_training_and_recruitment_cost_per_recruit'] = preservice_training_cost['npv_of_training_and_recruitment_cost'] *\
                                                    (1/(preservice_training_cost['absorption_rate_of_students_into_public_workforce'] + preservice_training_cost['proportion_of_workforce_recruited_from_abroad'])) *\
                                                    (1/preservice_training_cost['graduation_rate']) * (1/preservice_training_cost['licensure_exam_passing_rate'])
    preservice_training_cost['annuitisation_rate']  = 1 + (1 - (1 + discount_rate) ** (-preservice_training_cost['average_length_of_tenure_in_the_public_sector'] + 1)) / discount_rate
    preservice_training_cost['annuitised_training_and_recruitment_cost_per_recruit'] = preservice_training_cost['npv_of_training_and_recruitment_cost_per_recruit']/preservice_training_cost['annuitisation_rate']

    # Cost per student trained * 1/Rate of absorption from the local and foreign graduates * 1/Graduation rate * attrition rate
    # the inverse of attrition rate is the average expected tenure; and the preservice training cost needs to be divided by the average tenure
    preservice_training_cost['cost'] = preservice_training_cost['annuitised_training_and_recruitment_cost_per_recruit'] * preservice_training_cost['staff_count'] * preservice_training_cost['annual_attrition_rate'] # not multiplied with attrition rate again because this is already factored into 'Annual_cost_per_staff_recruited'
    preservice_training_cost = preservice_training_cost[['draw', 'run', 'year', 'OfficerType', 'Facility_Level', 'cost']]

    # 1.3 In-service training cost to train all staff
    #---------------------------------------------------------------------------------------------------------------
    inservice_training_cost = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = staff_size_chosen_for_costing,
                                                         varnames = ['annual_inservice_training_cost_usd'])
    inservice_training_cost['cost'] = inservice_training_cost['staff_count'] * inservice_training_cost['annual_inservice_training_cost_usd']
    inservice_training_cost = inservice_training_cost[['draw', 'run', 'year', 'OfficerType', 'Facility_Level', 'cost']]
    # TODO Consider calculating economic cost of HR by multiplying salary times staff count with cadres_utilisation_rate

    # 1.4 Regular mentorship and supportive supervision costs
    #---------------------------------------------------------------------------------------------------------------
    mentorship_and_supportive_cost = merge_cost_and_model_data(cost_df = hr_cost_parameters, model_df = staff_size_chosen_for_costing,
                                                         varnames = ['annual_mentorship_and_supervision_cost'])
    mentorship_and_supportive_cost['cost'] = mentorship_and_supportive_cost['staff_count'] * mentorship_and_supportive_cost['annual_mentorship_and_supervision_cost']
    mentorship_and_supportive_cost = mentorship_and_supportive_cost[['draw', 'run', 'year', 'OfficerType', 'Facility_Level', 'cost']]
    # TODO Consider calculating economic cost of HR by multiplying salary times staff count with cadres_utilisation_rate

    # 1.5 Store all HR costs in one standard format dataframe
    #---------------------------------------------------------------------------------------------------------------
    # Function to melt and label the cost category
    def label_rows_of_cost_dataframe(_df, label_var, label):
        _df = _df.reset_index()
        _df[label_var] = label
        return _df

    # Initialize HR with the salary data
    if (cost_only_used_staff):
        human_resource_costs = retain_relevant_column_subset(label_rows_of_cost_dataframe(salary_for_staff, 'cost_subcategory', 'salary_for_cadres_used'), 'OfficerType')
        # Concatenate additional cost categories
        additional_costs = [
            (preservice_training_cost, 'preservice_training_and_recruitment_cost_for_attrited_workers'),
            (inservice_training_cost, 'inservice_training_cost_for_cadres_used'),
            (mentorship_and_supportive_cost, 'mentorship_and_supportive_cost_for_cadres_used')
        ]
    else:
        human_resource_costs = retain_relevant_column_subset(label_rows_of_cost_dataframe(salary_for_staff, 'cost_subcategory', 'salary_for_all_staff'), 'OfficerType')
        # Concatenate additional cost categories
        additional_costs = [
            (preservice_training_cost, 'preservice_training_and_recruitment_cost_for_attrited_workers'),
            (inservice_training_cost, 'inservice_training_cost_for_all_staff'),
            (mentorship_and_supportive_cost, 'mentorship_and_supportive_cost_for_all_staff')
        ]

    # Iterate through additional costs, melt and concatenate
    for df, label in additional_costs:
        labelled_df = retain_relevant_column_subset(label_rows_of_cost_dataframe(df, 'cost_subcategory', label), 'OfficerType')
        human_resource_costs = pd.concat([human_resource_costs, labelled_df])

    human_resource_costs = prepare_cost_dataframe(human_resource_costs, _category_specific_group = 'OfficerType', _cost_category = 'human resources for health')

    # Only preserve the draws and runs requested
    if draws is not None:
        human_resource_costs = human_resource_costs[human_resource_costs.draw.isin(draws)]
    if runs is not None:
        human_resource_costs = human_resource_costs[human_resource_costs.run.isin(runs)]

    # %%
    # 2. Consumables cost
    #------------------------
    print("Now estimating Consumables costs...")
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

    # 2.3 Store all consumable costs in one standard format dataframe
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

    def disaggregate_separately_managed_medical_supplies_from_consumable_costs(_df,
                                                                   _consumables_dict, # This is a dictionary mapping codes to names
                                                                   list_of_unique_medical_products):
        reversed_consumables_dict = {value: key for key, value in _consumables_dict.items()} # reverse dictionary to map names to codes
        new_df = _df.copy()
        new_df['item_code'] = new_df['consumable'].map(reversed_consumables_dict)
        cost_of_consumables = new_df[~new_df['item_code'].isin(list_of_unique_medical_products)]
        cost_of_separately_managed_medical_supplies = new_df[new_df['item_code'].isin(list_of_unique_medical_products)]
        cost_of_separately_managed_medical_supplies['cost_subcategory'] = cost_of_separately_managed_medical_supplies['cost_subcategory'].replace(
            {'consumables_dispensed': 'separately_managed_medical_supplies_dispensed', 'consumables_stocked': 'separately_managed_medical_supplies_stocked'}, regex=True)
        return cost_of_consumables.drop(columns = 'item_code'), cost_of_separately_managed_medical_supplies.drop(columns = 'item_code')

    separately_managed_medical_supplies = [127, 141, 161] # Oxygen, Blood, IRS
    cost_of_consumables_dispensed, cost_of_separately_managed_medical_supplies_dispensed = disaggregate_separately_managed_medical_supplies_from_consumable_costs(_df = retain_relevant_column_subset(melt_and_label_consumables_cost(cost_of_consumables_dispensed, 'cost_of_consumables_dispensed'), 'consumable'),
                                                                                               _consumables_dict = consumables_dict,
                                                                                               list_of_unique_medical_products = separately_managed_medical_supplies)
    cost_of_excess_consumables_stocked, cost_of_separately_managed_medical_supplies_excess_stock = disaggregate_separately_managed_medical_supplies_from_consumable_costs(_df = retain_relevant_column_subset(melt_and_label_consumables_cost(cost_of_excess_consumables_stocked, 'cost_of_excess_consumables_stocked'), 'consumable'),
                                                                                                    _consumables_dict=consumables_dict,
                                                                                                    list_of_unique_medical_products=separately_managed_medical_supplies)

    consumable_costs = pd.concat([cost_of_consumables_dispensed, cost_of_excess_consumables_stocked])

    # 2.4 Supply chain costs
    #---------------------------------------------------------------------------------------------------------------
    # Assume that the cost of procurement, warehousing and distribution is a fixed proportion of consumable purchase costs
    # The fixed proportion is based on Resource Mapping Expenditure data from 2018
    resource_mapping_data = workbook_cost["resource_mapping_r7_summary"]
    # Make sure values are numeric
    expenditure_column = ['EXPENDITURE (USD) (Jul 2018 - Jun 2019)']
    resource_mapping_data[expenditure_column] = resource_mapping_data[expenditure_column].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    supply_chain_expenditure = resource_mapping_data[resource_mapping_data['Cost Type'] == 'Supply Chain'][expenditure_column].sum()[0]
    consumables_purchase_expenditure = resource_mapping_data[resource_mapping_data['Cost Type'] == 'Drugs and Commodities'][expenditure_column].sum()[0]
    supply_chain_cost_proportion = supply_chain_expenditure / consumables_purchase_expenditure

    # Estimate supply chain costs based on the total consumable purchase cost calculated above
    # Note that  Oxygen, IRS, and Blood costs are already excluded because the unit_cost of these commodities already
    # includes the procurement/production, storage and distribution costs
    supply_chain_costs = (consumable_costs.groupby(['draw', 'run', 'year'])[
                              'cost'].sum() * supply_chain_cost_proportion).reset_index()
    # Assign relevant additional columns to match the format of the rest of consumables costs
    supply_chain_costs['Facility_Level'] = 'all'
    supply_chain_costs['consumable'] = 'NA'
    supply_chain_costs['cost_subcategory'] = 'supply_chain'
    assert set(supply_chain_costs.columns) == set(consumable_costs.columns)

    # Append supply chain costs to the full consumable cost dataframe
    consumable_costs = pd.concat([consumable_costs, supply_chain_costs])
    other_costs = pd.concat([cost_of_separately_managed_medical_supplies_dispensed, cost_of_separately_managed_medical_supplies_excess_stock])

    consumable_costs = prepare_cost_dataframe(consumable_costs, _category_specific_group = 'consumable', _cost_category = 'medical consumables')
    other_costs = prepare_cost_dataframe(other_costs, _category_specific_group = 'consumable', _cost_category = 'other')

    # Only preserve the draws and runs requested
    if draws is not None:
        consumable_costs = consumable_costs[consumable_costs.draw.isin(draws)]
        other_costs = other_costs[other_costs.draw.isin(draws)]
    if runs is not None:
        consumable_costs = consumable_costs[consumable_costs.run.isin(runs)]
        other_costs = other_costs[other_costs.run.isin(runs)]


    # %%
    # 3. Equipment cost
    #--------------------------------------------
    print("Now estimating Medical equipment costs...")
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
            print(f"Processing draw {d} and run {r} of equipment costs")
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
    equipment_costs = prepare_cost_dataframe(equipment_costs, _category_specific_group = 'Equipment', _cost_category = 'medical equipment')

    # 4. Facility running costs
    # Average running costs by facility level and district times the number of facilities  in the simulation

    # %%
    # Store all costs in single dataframe
    #--------------------------------------------
    scenario_cost = pd.concat([human_resource_costs, consumable_costs, equipment_costs, other_costs], ignore_index=True)
    scenario_cost['cost'] = pd.to_numeric(scenario_cost['cost'], errors='coerce')

    # Summarize costs
    if summarize == True:
        groupby_cols = [col for col in scenario_cost.columns if ((col != 'run') & (col != 'cost'))]
        scenario_cost = pd.concat(
            {
                'mean': scenario_cost.groupby(by=groupby_cols, sort=False)['cost'].mean(),
                'lower': scenario_cost.groupby(by=groupby_cols, sort=False)['cost'].quantile(0.025),
                'upper': scenario_cost.groupby(by=groupby_cols, sort=False)['cost'].quantile(0.975),
            },
            axis=1
        )
        scenario_cost =  pd.melt(scenario_cost.reset_index(),
                  id_vars=groupby_cols,  # Columns to keep
                  value_vars=['mean', 'lower', 'upper'],  # Columns to unpivot
                  var_name='stat',  # New column name for the 'sub-category' of cost
                  value_name='cost')

    return scenario_cost

# Plot costs
####################################################
# 1. Stacked bar plot (Total cost + Cost categories)
#----------------------------------------------------
def do_stacked_bar_plot_of_cost_by_category(_df, _cost_category = 'all', _disaggregate_by_subgroup: bool = False,_year = 'all', _draws = None, _outputfilepath: Path = None):
    # Subset and Pivot the data to have 'Cost Sub-category' as columns
    # Make a copy of the dataframe to avoid modifying the original
    _df = _df[_df.stat == 'mean'].copy()
    # Convert 'value' to millions
    _df['cost'] = _df['cost'] / 1e6
    if _draws == None:
        subset_df = _df
    else:
        subset_df = _df[_df.draw.isin(_draws)]

    if _year == 'all':
        subset_df = subset_df
    else:
        subset_df = subset_df[subset_df['year'].isin(_year)]

    if _cost_category == 'all':
        if (_disaggregate_by_subgroup == True):
            raise ValueError(f"Invalid input for _disaggregate_by_subgroup: '{_disaggregate_by_subgroup}'. "
                             f"Value can be True only when plotting a specific _cost_category")
        else:
            pivot_df = subset_df.pivot_table(index='draw', columns='cost_category', values='cost', aggfunc='sum')
            plt_name_suffix = ''
    else:
        subset_df = subset_df[subset_df['cost_category'] == _cost_category]
        if (_disaggregate_by_subgroup == True):
            # If sub-groups are more than 10 in number, then disaggregate the top 10 and group the rest into an 'other' category
            if (len(subset_df['cost_subgroup']) > 10):
                # Calculate total cost per subgroup
                subgroup_totals = subset_df.groupby('cost_subgroup')['cost'].sum()
                # Identify the top 10 subgroups by cost
                top_10_subgroups = subgroup_totals.nlargest(10).index.tolist()
                # Label the remaining subgroups as 'other'
                subset_df['cost_subgroup'] = subset_df['cost_subgroup'].apply(
                    lambda x: x if x in top_10_subgroups else 'other'
                )

                pivot_df = subset_df.pivot_table(index=['draw', 'cost_subcategory'], columns='cost_subgroup',
                                                 values='cost', aggfunc='sum')

            else:
                pivot_df = subset_df.pivot_table(index=['draw', 'cost_subcategory'], columns='cost_subgroup',
                                                 values='cost', aggfunc='sum')

            plt_name_suffix = '_by_subgroup'
        else:
            pivot_df = subset_df.pivot_table(index='draw', columns='cost_subcategory', values='cost', aggfunc='sum')
            plt_name_suffix = ''

    # Sort pivot_df columns in ascending order by total cost
    sorted_columns = pivot_df.sum(axis=0).sort_values().index
    pivot_df = pivot_df[sorted_columns]  # Rearrange columns by sorted order

    # Plot the stacked bar chart
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6))

    # Period included for plot title and name
    if _year == 'all':
        period = (f"{min(_df['year'].unique())} - {max(_df['year'].unique())}")
    elif (len(_year) == 1):
        period = (f"{_year[0]}")
    else:
        period = (f"{min(_year)} - {max(_year)}")

    # Save plot
    plt.xlabel('Scenario')
    plt.ylabel('Cost (2023 USD), millions')

    # Arrange the legend in the same ascending order
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper right')

    plt.title(f'Costs by Scenario \n (Cost Category = {_cost_category} ; Period = {period})')
    plt.savefig(_outputfilepath / f'stacked_bar_chart_{_cost_category}_{period}{plt_name_suffix}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

# 2. Line plots of total costs
#----------------------------------------------------
# TODO: Check why line plot get save without a file name
def do_line_plot_of_cost(_df, _cost_category='all', _year='all', _draws=None, disaggregate_by=None,
                         _outputfilepath: Path = None):
    # Validate disaggregation options
    valid_disaggregations = ['cost_category', 'cost_subcategory', 'cost_subgroup']
    if disaggregate_by not in valid_disaggregations and disaggregate_by is not None:
        raise ValueError(f"Invalid disaggregation option: {disaggregate_by}. Choose from {valid_disaggregations}.")

    #
    if ((_draws is None) or (len(_draws) > 1)) & (disaggregate_by is not None):
        raise ValueError(f"The disaggregate_by option only works if only one draw is plotted, for exmaple _draws = [0]")

    # Filter the dataframe by draws, if specified
    subset_df = _df if _draws is None else _df[_df.draw.isin(_draws)]

    # Filter by year if specified
    if _year != 'all':
        subset_df = subset_df[subset_df['year'].isin(_year)]

    # Handle scenarios based on `_cost_category` and `disaggregate_by` conditions
    if _cost_category == 'all':
        if disaggregate_by == 'cost_subgroup':
            raise ValueError("Cannot disaggregate by 'cost_subgroup' when `_cost_category='all'` due to data size.")
    else:
        # Filter subset_df by specific cost category if specified
        subset_df = subset_df[subset_df['cost_category'] == _cost_category]

    # Set grouping columns based on the disaggregation level
    if disaggregate_by == 'cost_category':
        groupby_columns = ['year', 'cost_category']
    elif disaggregate_by == 'cost_subcategory':
        groupby_columns = ['year', 'cost_subcategory']
    elif disaggregate_by == 'cost_subgroup':
        # If disaggregating by 'cost_subgroup' and there are more than 10 subgroups, limit to the top 10 + "Other"
        if len(subset_df['cost_subgroup'].unique()) > 10:
            # Calculate total cost per subgroup
            subgroup_totals = subset_df[subset_df.stat == 'mean'].groupby('cost_subgroup')['cost'].sum()
            # Identify the top 10 subgroups by cost
            top_10_subgroups = subgroup_totals.nlargest(10).index.tolist()
            # Reassign smaller subgroups to an "Other" category
            subset_df['cost_subgroup'] = subset_df['cost_subgroup'].apply(
                lambda x: x if x in top_10_subgroups else 'Other'
            )
        groupby_columns = ['year', 'cost_subgroup']
    else:
        groupby_columns = ['year']

    # Extract mean, lower, and upper values for the plot
    mean_values = subset_df[subset_df.stat == 'mean'].groupby(groupby_columns)['cost'].sum() / 1e6
    lower_values = subset_df[subset_df.stat == 'lower'].groupby(groupby_columns)['cost'].sum() / 1e6
    upper_values = subset_df[subset_df.stat == 'upper'].groupby(groupby_columns)['cost'].sum() / 1e6

    # Prepare to store lines and labels for the legend
    lines = []
    labels = []

    # Define a list of colors to rotate through
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'gray']  # Add more colors as needed
    color_cycle = iter(colors)  # Create an iterator from the color list

    # Plot each line for the disaggregated values
    if disaggregate_by:
        for disaggregate_value in mean_values.index.get_level_values(disaggregate_by).unique():
            # Get mean, lower, and upper values for each disaggregated group
            value_mean = mean_values.xs(disaggregate_value, level=disaggregate_by)
            value_lower = lower_values.xs(disaggregate_value, level=disaggregate_by)
            value_upper = upper_values.xs(disaggregate_value, level=disaggregate_by)

            # Get the next color from the cycle
            color = next(color_cycle)

            # Plot line for mean and shaded region for 95% CI
            line, = plt.plot(value_mean.index, value_mean, marker='o', linestyle='-', color=color, label=f'{disaggregate_value} - Mean')
            plt.fill_between(value_mean.index, value_lower, value_upper, color=color, alpha=0.2)

            # Append to lines and labels for sorting later
            lines.append(line)
            labels.append(disaggregate_value)
    else:
        line, = plt.plot(mean_values.index, mean_values, marker='o', linestyle='-', color='b', label='Mean')
        plt.fill_between(mean_values.index, lower_values, upper_values, color='b', alpha=0.2)

        # Append to lines and labels for sorting later
        lines.append(line)
        labels.append('Mean')

    # Sort the legend based on total costs
    total_costs = {label: mean_values.xs(label, level=disaggregate_by).sum() for label in labels}
    sorted_labels = sorted(total_costs.keys(), key=lambda x: total_costs[x])

    # Reorder lines based on sorted labels
    handles = [lines[labels.index(label)] for label in sorted_labels]

    # Define period for plot title
    if _year == 'all':
        period = f"{min(subset_df['year'].unique())} - {max(subset_df['year'].unique())}"
    elif len(_year) == 1:
        period = str(_year[0])
    else:
        period = f"{min(_year)} - {max(_year)}"

    # Set labels, legend, and title
    plt.xlabel('Year')
    plt.ylabel('Cost (2023 USD), millions')
    plt.legend(handles[::-1], sorted_labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')
    plot_title = f'Total input cost \n (Category = {_cost_category}, Period = {period})'
    plt.title(plot_title)

    # Save plot with a proper filename
    if disaggregate_by is None:
        filename_suffix = ""
    else:
        filename_suffix = f"_by_{disaggregate_by}"

    filename = f'trend_{_cost_category}_{period}{filename_suffix}.png'
    plt.savefig(_outputfilepath / filename, dpi=100, bbox_inches='tight')
    plt.close()

'''
# Scratch pad
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

'''
