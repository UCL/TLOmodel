
from pathlib import Path

from tlo import Date
from collections import Counter, defaultdict
from typing import Optional, Union, Literal

import datetime
import textwrap

import matplotlib.pyplot as plt
import squarify
import numpy as np
import pandas as pd
import ast
import math
import itertools
from itertools import cycle
import matplotlib.container as mpc

from tlo.analysis.utils import (
    extract_results,
    get_scenario_info,
    load_pickled_dataframes,
    unflatten_flattened_multi_index_in_logging
)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

#%%

# Define a function to discount and summarise costs by cost_category
def apply_discounting_to_cost_data(_df, _discount_rate=0, _initial_year=None, _column_for_discounting = 'cost'):
    if _initial_year is None:
        # Determine the initial year from the dataframe
        _initial_year = min(_df['year'].unique())

    def get_discount_factor(year):
        """Compute the cumulative discount factor for a given year."""
        if isinstance(_discount_rate, dict):
            # Compute the cumulative discount factor as the product of (1 + discount_rate) for all previous years
            discount_factor = 1
            for y in range(_initial_year + 1, year + 1): # only starting from initial year + 1 as the discount factor for initial year should be 1
                discount_factor *= (1 + _discount_rate.get(y, 0))  # Default to 0 if year not in dictionary
            return discount_factor
        else:
            # If a single value is provided, use standard discounting
            return (1 + _discount_rate) ** (year - _initial_year)

    # Apply discounting to each row
    _df.loc[:, _column_for_discounting] = _df[_column_for_discounting] / _df['year'].apply(get_discount_factor)

    return _df

def estimate_input_cost_of_scenarios(results_folder: Path,
                                     resourcefilepath: Path = None,
                                     _draws: list[int] = None,
                                     _runs: list[int] = None,
                                     summarize: bool = False,
                                     _metric: Literal['mean', 'median'] = 'mean',
                                     _years: list[int] = None,
                                     cost_only_used_staff: bool = True,
                                     _discount_rate: Union[float, dict[int, float]] = 0) -> pd.DataFrame:
    """
    Estimate health system input costs for a given simulation.

    Parameters:
    ----------
    results_folder : Path
        Path to the directory containing simulation output files.
    resourcefilepath : Path, optional
        Path to the resource files
    _draws : list, optional
        Specific draws to include in the cost estimation. Defaults to all available draws.
    _runs : list, optional
        Specific runs to include in the cost estimation. Defaults to all runs.
    summarize : bool, default False
        Whether to summarize the costs across draws/runs with central metric (specified below) and confidence intervals.
    _metric : {'mean', 'median'}, default 'mean'
        Summary statistic to use if `summarize=True`.
    _years : list of int, optional
        Years to include in the cost output. If None, all years are included.
    cost_only_used_staff : bool, default True
        If True, only costs for level-cadre combinations ever used in simulation are included.
    _discount_rate : float or dict of {int: float}, default 0
        Discount rate to apply to future costs. Can be a constant or year-specific dictionary.

    Returns:
    -------
    pd.DataFrame
        A dataframe containing discounted costs disaggregated by category, sub-category, category-specific subgroup, year, draw, and run.
        Note that if a discount rate is used, the dataframe will provide cost as the NPV during the first year of the dataframe
    """

    # Useful common functions
    def drop_outside_period(_df):
        """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
        return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

    # todo replaced this after error
    #  KeyError: "The following id_vars or value_vars are not present in the DataFrame: ['year', 'Facility_Level', 'OfficerType']"
    # def melt_model_output_draws_and_runs(_df, id_vars):
    #     multi_index = pd.MultiIndex.from_tuples(_df.columns)
    #     _df.columns = multi_index
    #     melted_df = pd.melt(_df, id_vars=id_vars).rename(columns={'variable_0': 'draw', 'variable_1': 'run'})
    #     return melted_df

    def melt_model_output_draws_and_runs(_df):
        # Step 1: Reset index (gets 'year', 'Facility_Level', 'OfficerType' as columns)
        _df = _df.reset_index()

        new_columns = []
        for col in _df.columns:
            if isinstance(col, tuple):  # for columns that are tuples (draw, run)
                # check if the tuple represents 'year', 'Facility_Level', or 'OfficerType'
                if col[0] == 'year':
                    new_columns.append('year')
                elif col[0] == 'Facility_Level':
                    new_columns.append('Facility_Level')
                elif col[0] == 'OfficerType':
                    new_columns.append('OfficerType')
                else:
                    new_columns.append(f"draw_{col[0]}_run_{col[1]}")
            else:
                new_columns.append(col)

        # Assign the new column names to the DataFrame
        _df.columns = new_columns

        # Melt to long format
        _df_long = _df.melt(
            id_vars=['year', 'Facility_Level', 'OfficerType'],  # keeping 'year', 'Facility_Level', 'OfficerType'
            value_vars=[col for col in _df.columns if isinstance(col, str) and col.startswith("draw")],
            var_name='draw_run',  # this will create the 'draw_run' column
            value_name='value'
        )

        # Step 4: Split the 'draw_run' column into two columns: 'draw' and 'run'
        _df_long[['draw', 'run']] = _df_long['draw_run'].str.extract(r'draw_(\d+)_run_(\d+)')
        _df_long['draw'] = _df_long['draw'].astype(int)
        _df_long['run'] = _df_long['run'].astype(int)

        # Drop the 'draw_run' column
        _df_long = _df_long.drop(columns=['draw_run'])

        return _df_long





    # Define a relative pathway for relevant folders
    path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"

    # %% Gathering basic information
    # Load basic simulation parameters
    #-------------------------------------
    log = load_pickled_dataframes(results_folder, 0, 0)  # read from 1 draw and run
    info = get_scenario_info(results_folder)  # get basic information about the results
    if _draws is None:
        _draws = range(0, info['number_of_draws'])
    if _runs is None:
        _runs = range(0, info['runs_per_draw'])
    final_year_of_simulation = max(log['tlo.methods.healthsystem.summary']['hsi_event_counts']['date']).year
    first_year_of_simulation = min(log['tlo.methods.healthsystem.summary']['hsi_event_counts']['date']).year
    years = list(range(first_year_of_simulation, final_year_of_simulation + 1)) # this is the full period of the simulation but at the end of the function, years not needed for the final cost estimate are dropped

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

    # If variable discount rate is provided, use the average across the relevant years for the purpose of annuitization of HR and equipment costs
    def calculate_annuitization_rate(_discount_rate, _years):
        if isinstance(_discount_rate, (int, float)):
            # Single discount rate, return as is
            return _discount_rate
        elif isinstance(_discount_rate, dict):
            # Extract rates for the specified years (default to 0 if year is missing)
            rates = [_discount_rate.get(year, 0) for year in _years]
            return sum(rates) / len(rates)  # Average discount rate
        else:
            raise ValueError("`_discount_rate` must be either a number (single rate) or a dictionary {year: rate}.")

    annuitization_rate = calculate_annuitization_rate(_discount_rate, _years)

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
    if _discount_rate == 0:
        unit_cost_equipment['replacement_cost_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost'] / row['Life span'], axis=1)  # straight line depreciation is discount rate is 0
    else:
        unit_cost_equipment['replacement_cost_annual'] = unit_cost_equipment.apply(lambda row: row['unit_purchase_cost']/(1+(1-(1+annuitization_rate)**(-row['Life span']+1))/annuitization_rate), axis=1) # Annuitised over the life span of the equipment assuming outlay at the beginning of the year
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

    # Load and prepare facility operation cost parameters
    unit_cost_fac_operations = workbook_cost["facility_operations"]

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
    available_staff_count_by_facid_and_officertype = available_staff_count_by_facid_and_officertype.reset_index().rename(columns={'FacilityID': 'Facility_ID', 'Officer': 'OfficerType'})
    available_staff_count_by_facid_and_officertype['Facility_ID'] = pd.to_numeric(available_staff_count_by_facid_and_officertype['Facility_ID'])
    available_staff_count_by_facid_and_officertype['Facility_Level'] = available_staff_count_by_facid_and_officertype['Facility_ID'].map(facility_id_levels_dict)
    idx = pd.IndexSlice
    available_staff_count_by_level_and_officer_type = available_staff_count_by_facid_and_officertype.drop(
        columns=[idx['Facility_ID']]).groupby([idx['year'], idx['Facility_Level'], idx['OfficerType']]).sum()


    # todo fixed this
    available_staff_count_by_level_and_officer_type = melt_model_output_draws_and_runs(
        available_staff_count_by_level_and_officer_type)

    # back to original script
    available_staff_count_by_level_and_officer_type['Facility_Level'] = available_staff_count_by_level_and_officer_type['Facility_Level'].astype(str)  # make sure facility level is stored as string
    available_staff_count_by_level_and_officer_type = available_staff_count_by_level_and_officer_type.drop(
        available_staff_count_by_level_and_officer_type[available_staff_count_by_level_and_officer_type['Facility_Level'] == '5'].index) # drop headquarters because we're only concerned with staff engaged in service delivery

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
    # average_capacity_used_by_cadre_and_level.reset_index(drop=True)  # Flatten multi=index column
    # todo replacing the line above
    # Step 1: Separate out the id_vars and value_vars
    id_vars = ['OfficerType', 'FacilityLevel']
    value_vars = [col for col in average_capacity_used_by_cadre_and_level.columns
                  if col not in id_vars]

    # Step 2: Melt using MultiIndex columns
    average_capacity_used_by_cadre_and_level = average_capacity_used_by_cadre_and_level.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=['draw', 'run'],
        value_name='capacity_used'
    )

    # average_capacity_used_by_cadre_and_level = average_capacity_used_by_cadre_and_level.melt(id_vars=['OfficerType', 'FacilityLevel'],
    #                         var_name=['draw', 'run'],
    #                         value_name='capacity_used')

    list_of_cadre_and_level_combinations_used = average_capacity_used_by_cadre_and_level[average_capacity_used_by_cadre_and_level['capacity_used'] != 0][['OfficerType', 'FacilityLevel', 'draw', 'run']]
    print(f"Out of {average_capacity_used_by_cadre_and_level.groupby(['OfficerType', 'FacilityLevel']).size().count()} cadre and level combinations available, {list_of_cadre_and_level_combinations_used.groupby(['OfficerType', 'FacilityLevel']).size().count()} are used across the simulations")
    list_of_cadre_and_level_combinations_used = list_of_cadre_and_level_combinations_used.rename(columns = {'FacilityLevel':'Facility_Level'})

    # todo add this to counter errors
    # Ensure consistent types for merge keys
    average_capacity_used_by_cadre_and_level['draw'] = average_capacity_used_by_cadre_and_level['draw'].astype(int)
    average_capacity_used_by_cadre_and_level['run'] = average_capacity_used_by_cadre_and_level['run'].astype(int)

    list_of_cadre_and_level_combinations_used['draw'] = list_of_cadre_and_level_combinations_used['draw'].astype(int)
    list_of_cadre_and_level_combinations_used['run'] = list_of_cadre_and_level_combinations_used['run'].astype(int)

    available_staff_count_by_level_and_officer_type['draw'] = available_staff_count_by_level_and_officer_type[
        'draw'].astype(int)
    available_staff_count_by_level_and_officer_type['run'] = available_staff_count_by_level_and_officer_type[
        'run'].astype(int)

    used_staff_count_by_level_and_officer_type = available_staff_count_by_level_and_officer_type.merge(
        list_of_cadre_and_level_combinations_used,
        on=['draw', 'run', 'OfficerType', 'Facility_Level'],
        how='right',
        validate='m:m'
    )

    # Subset scenario staffing level to only include cadre-level combinations used in the simulation
    # used_staff_count_by_level_and_officer_type = available_staff_count_by_level_and_officer_type.merge(
    #     list_of_cadre_and_level_combinations_used,
    #     on=['draw','run','OfficerType', 'Facility_Level'], how='right', validate='m:m')
    #
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

    def calculate_npv_past_training_expenses_by_row(row, r = _discount_rate):
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
        npv = calculate_npv_past_training_expenses_by_row(row, r=annuitization_rate)
        npv_values.append(npv)

    preservice_training_cost['npv_of_training_and_recruitment_cost'] = npv_values
    preservice_training_cost['npv_of_training_and_recruitment_cost_per_recruit'] = preservice_training_cost['npv_of_training_and_recruitment_cost'] *\
                                                    (1/(preservice_training_cost['absorption_rate_of_students_into_public_workforce'] + preservice_training_cost['proportion_of_workforce_recruited_from_abroad'])) *\
                                                    (1/preservice_training_cost['graduation_rate']) * (1/preservice_training_cost['licensure_exam_passing_rate'])
    if _discount_rate == 0: # if the discount rate is 0, then the pre-service + recruitment cost simply needs to be divided by the number of years in tenure
        preservice_training_cost['annuitisation_rate'] = preservice_training_cost['average_length_of_tenure_in_the_public_sector']
    else:
        preservice_training_cost['annuitisation_rate']  = 1 + (1 - (1 + annuitization_rate) ** (-preservice_training_cost['average_length_of_tenure_in_the_public_sector'] + 1)) / annuitization_rate
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
    if _draws is not None:
        human_resource_costs = human_resource_costs[human_resource_costs.draw.isin(_draws)]
    if _runs is not None:
        human_resource_costs = human_resource_costs[human_resource_costs.run.isin(_runs)]

    # %%
    # 2. Consumables cost
    #------------------------
    print("Now estimating Consumables costs...")
    def get_quantity_of_consumables_dispensed(results_folder):
        def get_counts_of_items_requested(_df):
            _df = drop_outside_period(_df)
            counts_of_used = defaultdict(lambda: defaultdict(int))
            counts_of_not_available = defaultdict(lambda: defaultdict(int))

            for _, row in _df.iterrows():
                date = row['date']
                for item, num in row['Item_Used'].items():
                    counts_of_used[date][item] += num
                for item, num in row['Item_NotAvailable'].items():
                    counts_of_not_available[date][item] += num
            used_df = pd.DataFrame(counts_of_used).fillna(0).astype(int).stack().rename('Used')
            not_available_df = pd.DataFrame(counts_of_not_available).fillna(0).astype(int).stack().rename('Not_Available')

            # Combine the two dataframes into one series with MultiIndex (date, item, availability_status)
            combined_df = pd.concat([used_df, not_available_df], axis=1).fillna(0).astype(int)

            # Convert to a pd.Series, as expected by the custom_generate_series function
            return combined_df.stack()

        cons_req = extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Consumables',
                custom_generate_series=get_counts_of_items_requested,
                do_scaling=True)

        cons_dispensed = cons_req.xs("Used", level=2) # only keep actual dispensed amount, i.e. when available
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
    consumables_dict = pd.read_csv(path_for_consumable_resourcefiles / 'ResourceFile_Consumables_Items_and_Packages.csv', low_memory=False,
                                 encoding="ISO-8859-1")[['Items','Item_Code']]
    consumables_dict = dict(zip(consumables_dict['Item_Code'], consumables_dict['Items']))
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
    # The numerator includes Supply chain expenditure for EHP consumables
    supply_chain_expenditure = \
        resource_mapping_data[resource_mapping_data['Cost Type'] == 'Supply Chain'][expenditure_column].sum()[0]
    # The denominator include all drugs and commodities expenditure, excluding what is recategorised as non-EHP or admin
    drug_expenditure_condition = resource_mapping_data['Cost Type'].str.contains('Drugs and Commodities')
    excluded_drug_expenditure_condition = (resource_mapping_data[
                                               'Calibration_category'] == 'Program Management & Administration') | (
                                                  resource_mapping_data[
                                                      'Calibration_category'] == 'Non-EHP consumables')
    consumables_purchase_expenditure = \
        resource_mapping_data[drug_expenditure_condition][expenditure_column].sum()[0] - \
        resource_mapping_data[drug_expenditure_condition & excluded_drug_expenditure_condition][
            expenditure_column].sum()[0]
    supply_chain_cost_proportion = supply_chain_expenditure / consumables_purchase_expenditure

    # Estimate supply chain costs based on the total consumable purchase cost calculated above
    # Note that  Oxygen, IRS, and Blood costs are already excluded because the unit_cost of these commodities already
    # includes the procurement/production, storage and distribution costs
    supply_chain_costs = (consumable_costs.groupby(['draw', 'run', 'year'])[
                              'cost'].sum() * supply_chain_cost_proportion).reset_index()
    # Assign relevant additional columns to match the format of the rest of consumables costs
    supply_chain_costs['Facility_Level'] = 'all'
    supply_chain_costs['consumable'] = 'supply chain (all consumables)'
    supply_chain_costs['cost_subcategory'] = 'supply_chain'
    assert set(supply_chain_costs.columns) == set(consumable_costs.columns)

    # Append supply chain costs to the full consumable cost dataframe
    consumable_costs = pd.concat([consumable_costs, supply_chain_costs])
    other_costs = pd.concat([cost_of_separately_managed_medical_supplies_dispensed, cost_of_separately_managed_medical_supplies_excess_stock])

    consumable_costs = prepare_cost_dataframe(consumable_costs, _category_specific_group = 'consumable', _cost_category = 'medical consumables')
    other_costs = prepare_cost_dataframe(other_costs, _category_specific_group = 'consumable', _cost_category = 'medical consumables')

    # Only preserve the draws and runs requested
    if _draws is not None:
        consumable_costs = consumable_costs[consumable_costs.draw.isin(_draws)]
        other_costs = other_costs[other_costs.draw.isin(_draws)]
    if _runs is not None:
        consumable_costs = consumable_costs[consumable_costs.run.isin(_runs)]
        other_costs = other_costs[other_costs.run.isin(_runs)]


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
    for d in _draws:
        for r in _runs:
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
    # Convert unit_costs to long format
    unit_cost_fac_operations = pd.melt(
        unit_cost_fac_operations,
        id_vars=["Facility_Level"],  # Columns to keep as identifiers
        var_name="operating_cost_type",  # Name for the new 'cost_category' column
        value_name="unit_cost"  # Name for the new 'cost' column
    )
    unit_cost_fac_operations['Facility_Level'] = unit_cost_fac_operations['Facility_Level'].astype(str)
    fac_count_by_level = mfl[['Facility_Level', 'Facility_Count']].groupby(['Facility_Level']).sum().reset_index()

    facility_operation_cost = pd.merge(unit_cost_fac_operations, fac_count_by_level, on = 'Facility_Level', how = 'left', validate = 'm:m')
    facility_operation_cost['Facility_Count'] = facility_operation_cost['Facility_Count'].fillna(0).astype(int)
    facility_operation_cost['cost'] =  facility_operation_cost['unit_cost'] * facility_operation_cost['Facility_Count']

    # Duplicate the same set of facility operation costs for all draws and runs
    # Create the Cartesian product of `_draws` and `_runs`
    combinations = list(itertools.product(_draws, _runs))
    comb_df = pd.DataFrame(combinations, columns=["draw", "run"])
    facility_operation_cost = facility_operation_cost.merge(comb_df, how="cross")
    facility_operation_cost['cost_category'] = 'Facility operating cost'
    operating_cost_mapping = {'Electricity': 'utilities_and_maintenance', 'Water': 'utilities_and_maintenance', 'Cleaning':'utilities_and_maintenance',
                              'Security':'utilities_and_maintenance', 'Building maintenance': 'building_maintenance',
                             'Facility management': 'utilities_and_maintenance', 'Vehicle maintenance': 'vehicle_maintenance',
                              'Ambulance fuel': 'fuel_for_ambulance', 'Food for inpatient cases': 'food_for_inpatient_care'}
    facility_operation_cost['cost_subcategory'] = facility_operation_cost['operating_cost_type']
    facility_operation_cost['cost_subcategory'] = facility_operation_cost['cost_subcategory'].map(operating_cost_mapping)
    # Assume that the annual costs are constant each year of the simulation
    facility_operation_cost = pd.concat([facility_operation_cost.assign(year=year) for year in years])

    # Assume that the annual costs are constant each year of the simulation
    facility_operation_cost = prepare_cost_dataframe(facility_operation_cost, _category_specific_group = 'operating_cost_type', _cost_category = 'facility operating cost')


    # %%
    # Store all costs in single dataframe
    #--------------------------------------------
    scenario_cost = pd.concat([human_resource_costs, consumable_costs, equipment_costs, other_costs, facility_operation_cost], ignore_index=True)
    scenario_cost['cost'] = pd.to_numeric(scenario_cost['cost'], errors='coerce')

    # Summarize costs
    if summarize:
        groupby_cols = [col for col in scenario_cost.columns if col not in ['run', 'cost']]
        # Use the summary metric specific in the inputs
        if _metric not in ['mean', 'median']:
            raise ValueError(f"Invalid input for _metric: '{_metric}'. "
                             f"Values need to be one of 'mean' or 'median'")
        else:
            # Define aggregation function based on _metric input (mean or median)
            agg_func = np.mean if _metric == 'mean' else np.median

            scenario_cost = pd.concat(
                {
                    _metric: scenario_cost.groupby(by=groupby_cols, sort=False)['cost'].agg(agg_func),
                    'lower': scenario_cost.groupby(by=groupby_cols, sort=False)['cost'].quantile(0.025),
                    'upper': scenario_cost.groupby(by=groupby_cols, sort=False)['cost'].quantile(0.975),
                },
                axis=1
            )

            scenario_cost = pd.melt(
                scenario_cost.reset_index(),
                id_vars=groupby_cols,  # Columns to keep
                value_vars=[_metric, 'lower', 'upper'],  # Columns to unpivot
                var_name='stat',  # New column name for the 'sub-category' of cost
                value_name='cost'
            )

    if _years is None:
        return apply_discounting_to_cost_data(_df = scenario_cost,
                                              _discount_rate = _discount_rate, _column_for_discounting = 'cost')
    else:
        return apply_discounting_to_cost_data(_df = scenario_cost[scenario_cost.year.isin(_years)],
                                              _discount_rate = _discount_rate,
                                              _column_for_discounting = 'cost')

# Define a function to summarize cost data from
# Note that the dataframe needs to have draw as index and run as columns. if the dataframe is long with draw and run as index, then
# first unstack the dataframe and subsequently apply the summarize function
def summarize_cost_data(_df,
                        _metric: Literal['mean', 'median'] = 'mean') -> pd.DataFrame:
    """
    Summarize cost data across runs by computing central tendency and 95% confidence intervals.

    Parameters:
    ----------
    _df : pd.DataFrame
        A DataFrame with draw as index and run as columns, where each cell contains a cost value.
            - Rows = draw IDs (e.g., 0, 1, 2)
            - Columns = run IDs (e.g., 0, 1, 2)
            - Values = cost estimates

    _metric : {'mean', 'median'}, default 'mean'
        The central summary statistic to compute across runs.

    Returns:
    -------
    pd.DataFrame
        A pivoted DataFrame with draws as index and a MultiIndex of columns:
        (run ID, ['mean' or 'median', 'lower', 'upper']), where:
        - 'lower' = 2.5th percentile
        - 'upper' = 97.5th percentile
    """

    if _metric not in ['mean', 'median']:
        raise ValueError(f"Invalid input for _metric: '{_metric}'. "
                         f"Values need to be one of 'mean' or 'median'")

    _df = _df.stack()
    collapsed_df = _df.groupby(level='draw').agg([
            _metric,
            ('lower', lambda x: x.quantile(0.025)),
            ('upper', lambda x: x.quantile(0.975))
        ])

    collapsed_df = collapsed_df.unstack()
    collapsed_df.index = collapsed_df.index.set_names('stat', level=0)
    collapsed_df = collapsed_df.unstack(level='stat')
    return collapsed_df

# Estimate projected health spending
####################################################
def estimate_projected_health_spending(resourcefilepath: Path = None,
                                      results_folder: Path =  None,
                                     _draws: list[int] = None,
                                      _runs: list[int] = None,
                                     _years: list[int] = None,
                                     _discount_rate: float = 0,
                                     _summarize: bool = False,
                                    _metric: Literal['mean', 'median'] = 'mean') -> pd.DataFrame:
    """
    Estimate total projected health spending for a simulation period.

    Combines health spending per capita projections (Dieleman et al, 2019) with simulated population estimates to calculate
    total health expenditure, optionally applying a discount rate and summarizing across runs.

    Parameters:
    ----------
    resourcefilepath : Path
        Path to the folder containing the costing resource Excel files.
    results_folder : Path
        Path to the simulation results folder.
    _draws : list or range, optional
        Draws to include. If None, all available draws are used.
    _runs : list or range, optional
        Runs to include. If None, all available runs are used.
    _years : list of int, optional
        Years to include. If None, includes the full simulation period.
    _discount_rate : float, default 0
        Discount rate applied to future costs.
    _summarize : bool, default False
        Whether to summarize output across runs using mean/median and 95% confidence intervals.
    _metric : {'mean', 'median'}, default 'mean'
        Central tendency metric used if summarizing.

    Returns:
    -------
    pd.DataFrame
        If `_summarize=True`, returns a DataFrame with:
            - Index = draw
            - Columns = 'mean'/'median', 'lower', 'upper' ROI values

        If `_summarize=False`, returns a DataFrame with:
            - Index = draw
            - Columns = run
        - Values = discounted total health spending for the selected years
    """

    # %% Gathering basic information
    # Load basic simulation parameters
    #-------------------------------------
    log = load_pickled_dataframes(results_folder, 0, 0)  # read from 1 draw and run
    info = get_scenario_info(results_folder)  # get basic information about the results
    if _draws is None:
        _draws = range(0, info['number_of_draws'])
    if _runs is None:
        _runs = range(0, info['runs_per_draw'])
    final_year_of_simulation = max(log['tlo.methods.healthsystem.summary']['hsi_event_counts']['date']).year
    first_year_of_simulation = min(log['tlo.methods.healthsystem.summary']['hsi_event_counts']['date']).year
    if _years == None:
        _years = list(range(first_year_of_simulation, final_year_of_simulation + 1))

    # Load health spending per capita projections
    #----------------------------------------
    # Load health spending projections
    workbook_cost = pd.read_excel((resourcefilepath / "costing/ResourceFile_Costing.xlsx"),
                                  sheet_name=None)
    health_spending_per_capita = workbook_cost["health_spending_projections"]
    # Assign the fourth row as column names
    health_spending_per_capita.columns = health_spending_per_capita.iloc[1]
    health_spending_per_capita = health_spending_per_capita.iloc[2:].reset_index(drop=True)
    health_spending_per_capita = health_spending_per_capita[
        health_spending_per_capita.year.isin(list(range(2015, 2041)))]
    total_health_spending_per_capita_mean = health_spending_per_capita[['year', 'total_mean']].set_index('year')
    total_health_spending_per_capita_mean.columns = pd.MultiIndex.from_tuples([('total_mean', '')])

    # Load population projections
    # ----------------------------------------
    def get_total_population(_df):
        years_needed = [min(_years), max(_years)]  # we only consider the population for the malaria scale-up period
        # because those are the years relevant for malaria scale-up costing
        _df['year'] = pd.to_datetime(_df['date']).dt.year
        _df = _df[['year', 'total']]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        return pd.Series(_df.loc[_df.year.between(*years_needed)].set_index('year')['total'])

    total_population_by_year = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='population',
        custom_generate_series=get_total_population,
        do_scaling=True
    )
    population_columns = total_population_by_year.columns

    # Estimate total health spending
    projected_health_spending = pd.merge(total_health_spending_per_capita_mean,
                    total_population_by_year,
                    left_index=True, right_index=True,how='inner')
    projected_health_spending = projected_health_spending.apply(pd.to_numeric, errors='coerce')
    projected_health_spending[population_columns] = projected_health_spending[population_columns].multiply(
        projected_health_spending['total_mean'], axis=0)
    projected_health_spending = projected_health_spending[population_columns]

    # Apply discount rate
    # Reformat dataframe to apply discounting function
    projected_health_spending.columns.names = ['draw', 'run']
    projected_health_spending = projected_health_spending.stack(level=['draw', 'run']).reset_index()
    projected_health_spending.columns = ['year', 'draw', 'run', 'total_spending']

    # Initial year and discount rate
    initial_year = min(projected_health_spending['year'].unique())
    projected_health_spending_discounted = apply_discounting_to_cost_data(
        projected_health_spending, _discount_rate= _discount_rate,
        _column_for_discounting='total_spending', _initial_year = initial_year)
    projected_health_spending_discounted = projected_health_spending_discounted.groupby(['draw', 'run'])['total_spending'].sum()

    if _summarize == True:
        if _metric == 'mean':
            # Calculate the mean and 95% confidence intervals for each group
            projected_health_spending_discounted = projected_health_spending_discounted.groupby(level="draw").agg(
                mean=np.mean,
                lower=lambda x: np.percentile(x, 2.5),
                upper=lambda x: np.percentile(x, 97.5)
            )

        elif _metric == 'median':
            # Calculate the mean and 95% confidence intervals for each group
            projected_health_spending_discounted = projected_health_spending_discounted.groupby(level="draw").agg(
                median=np.median,
                lower=lambda x: np.percentile(x, 2.5),
                upper=lambda x: np.percentile(x, 97.5)
            )

        else:
            raise ValueError(f"Invalid input for _metric: '{_metric}'. "
                             f"Values need to be one of 'mean' or 'median'")
        # Flatten the resulting DataFrame into a single-level MultiIndex Series
        projected_health_spending_discounted = projected_health_spending_discounted.stack().rename_axis(["draw", "stat"]).rename("value")

    return projected_health_spending_discounted.unstack()

# Plot costs
####################################################
# 1. Stacked bar plot (Total cost + Cost categories)
#----------------------------------------------------
def do_stacked_bar_plot_of_cost_by_category(_df: pd.DataFrame,
                                            _cost_category: Literal['all', 'human resources for health', 'medical consumables',
                                            'medical equipment', 'facility operating cost'] = 'all',
                                            _disaggregate_by_subgroup: bool = False,
                                            _year: list[int] = 'all',
                                            _draws: list[int] = None,
                                            _scenario_dict: dict[int,str] = None,
                                            show_title: bool = True,
                                            _outputfilepath: Path = None,
                                            _add_figname_suffix: str = ''):
    """
        Create and save a stacked bar chart of costs by category, subcategory or subgroup.

        Parameters:
        ----------
        _df : pd.DataFrame
            DataFrame with cost results, including columns:
            ['draw', 'year', 'cost_category', 'cost_subcategory', 'cost_subgroup',
             'cost', 'stat'] — typically produced by `estimate_input_cost_of_scenarios`.

        _cost_category : str, default 'all'
            If 'all', compares high-level categories (e.g., HR, consumables, equipment, facilty operations).
            Otherwise, filters to a specific category and optionally disaggregates.

        _disaggregate_by_subgroup : bool, default False
            If True and a single `_cost_category` is selected, breaks down costs by `cost_subgroup`.

        _year : str or list of int, default 'all'
            Year or years to include. Can be:
                - 'all' to include all available years
                - a single year or multiple years as a list: [2025]

        _draws : list of int, optional
            If specified, only includes the specified draws.

        _scenario_dict : dict, optional
            Dictionary mapping draw numbers to scenario names, used for x-axis labels.

        show_title : bool, default True
            Whether to display the chart title.

        _outputfilepath : Path, optional
            Folder to save the plot. File will be saved as a PNG using `_cost_category`
            and `_add_figname_suffix` in the filename.

        _add_figname_suffix : str, default ''
            Optional string to append to the saved figure's filename

        Returns:
        -------
        None
            The chart is saved to disk as a PNG.
        """
    # Subset and Pivot the data to have 'Cost Sub-category' as columns
    # Check what's the correct central metric to use (either 'mean' or 'median')
    central_metric = [stat for stat in _df.stat.unique() if stat not in ['lower', 'upper']][0]

    # Make a copy of the dataframe to avoid modifying the original
    _df_central = _df[_df.stat == central_metric].copy()
    _df_lower = _df[_df.stat == 'lower'].copy()
    _df_upper = _df[_df.stat == 'upper'].copy()

    # Subset the dataframes to keep the s=relevant categories for the plot
    dfs = {"_df_central": _df_central, "_df_lower": _df_lower, "_df_upper": _df_upper} # create a dict of dataframes
    for name, df in dfs.items():
        dfs[name] = df.copy()  # Choose the dataframe to modify
        # Convert 'cost' to millions
        dfs[name]['cost'] = dfs[name]['cost'] / 1e6
        # Subset data
        if _draws is not None:
            dfs[name] = dfs[name][dfs[name].draw.isin(_draws)]
        if _year != 'all':
            dfs[name] = dfs[name][dfs[name]['year'].isin(_year)]
        if _cost_category != 'all':
            dfs[name] = dfs[name][dfs[name]['cost_category'] == _cost_category]

    # Extract the updated DataFrames back from the dictionary
    _df_central, _df_lower, _df_upper = dfs["_df_central"], dfs["_df_lower"], dfs["_df_upper"]

    if _cost_category == 'all':
        if (_disaggregate_by_subgroup == True):
            raise ValueError(f"Invalid input for _disaggregate_by_subgroup: '{_disaggregate_by_subgroup}'. "
                             f"Value can be True only when plotting a specific _cost_category")
        else:
            pivot_central = _df_central.pivot_table(index='draw', columns='cost_category', values='cost', aggfunc='sum')
            pivot_lower = _df_lower.pivot_table(index='draw', columns='cost_category', values='cost', aggfunc='sum')
            pivot_upper = _df_upper.pivot_table(index='draw', columns='cost_category', values='cost', aggfunc='sum')
    else:
        if (_disaggregate_by_subgroup == True):
            for name, df in dfs.items():
                dfs[name] = df.copy()  # Choose the dataframe to modify
                # If sub-groups are more than 10 in number, then disaggregate the top 10 and group the rest into an 'other' category
                if (len(dfs[name]['cost_subgroup'].unique()) > 10):
                    # Calculate total cost per subgroup
                    subgroup_totals = dfs[name].groupby('cost_subgroup')['cost'].sum()
                    # Identify the top 10 subgroups by cost
                    top_10_subgroups = subgroup_totals.nlargest(10).index.tolist()
                    # Label the remaining subgroups as 'other'
                    dfs[name]['cost_subgroup'] = dfs[name]['cost_subgroup'].apply(
                        lambda x: x if x in top_10_subgroups else 'All other items'
                    )

            # Extract the updated DataFrames back from the dictionary
            _df_central, _df_lower, _df_upper = dfs["_df_central"], dfs["_df_lower"], dfs["_df_upper"]

            pivot_central = _df_central.pivot_table(index='draw', columns='cost_subgroup',
                                             values='cost', aggfunc='sum')
            pivot_lower = _df_lower.pivot_table(index='draw', columns='cost_subgroup',
                                        values='cost', aggfunc='sum')
            pivot_upper = _df_upper.pivot_table(index='draw', columns='cost_subgroup',
                                        values='cost', aggfunc='sum')

            plt_name_suffix = '_by_subgroup'
        else:
            pivot_central = _df_central.pivot_table(index='draw', columns='cost_subcategory', values='cost', aggfunc='sum')
            pivot_lower = _df_lower.pivot_table(index='draw', columns='cost_subcategory', values='cost', aggfunc='sum')
            pivot_upper = _df_upper.pivot_table(index='draw', columns='cost_subcategory', values='cost', aggfunc='sum')
            plt_name_suffix = ''

    # Sort pivot_df columns in ascending order by total cost
    sorted_columns = pivot_central.sum(axis=0).sort_values().index
    pivot_central = pivot_central[sorted_columns]
    pivot_lower = pivot_lower[sorted_columns]
    pivot_upper = pivot_upper[sorted_columns]

    # Error bars
    lower_bounds = pivot_central.sum(axis=1) - pivot_lower.sum(axis=1)
    lower_bounds[lower_bounds<0] = 0
    upper_bounds = pivot_upper.sum(axis=1) - pivot_central.sum(axis=1)

    if _cost_category == 'all':
        # Predefined color mapping for cost categories
        color_mapping = {
            'human resources for health': '#1f77b4',  # Muted blue
            'medical consumables': '#ff7f0e',  # Muted orange
            'medical equipment': '#2ca02c',  # Muted green
            'other': '#d62728',  # Muted red
            'facility operating cost': '#9467bd',  # Muted purple
        }
        # Default color for unexpected categories
        default_color = 'gray'
        plt_name_suffix = ''

    # Define custom colors for the bars
    if _cost_category == 'all':
        column_colors = [color_mapping.get(col, default_color) for col in sorted_columns]
        # Plot the stacked bar chart with set colours
        ax = pivot_central.plot(kind='bar', stacked=True, figsize=(10, 6), color=column_colors)

        # Add error bars
        x_pos = np.arange(len(pivot_central.index))
        total_central = pivot_central.sum(axis=1)
        error_bars = [lower_bounds, upper_bounds]
        ax.errorbar(x_pos, total_central, yerr=error_bars, fmt='o', color='black', capsize=5)

    else:
        # Plot the stacked bar chart without set colours
        ax = pivot_central.plot(kind='bar', stacked=True, figsize=(10, 6))

        # Add error bars
        x_pos = np.arange(len(pivot_central.index))
        total_central = pivot_central.sum(axis=1)
        error_bars = [lower_bounds, upper_bounds]
        ax.errorbar(x_pos, total_central, yerr=error_bars, fmt='o', color='black', capsize=5)

    # Add data labels such that the stacked block has a superimposed white label is the value is >=2% of the Y-axis limit
    # and a black label adjusted to the right of the bar (for visibility) if the value is <2%
    # Get max y-limit for threshold
    max_y = ax.get_ylim()[1]
    threshold = max_y * 0.02  # 2% of ylim

    for container in ax.containers:
        if isinstance(container, mpc.BarContainer):  # Ensure we're working with bars, not error bars
            for rect in container:
                height = rect.get_height()
                if height > 0:  # Avoid labeling zero-height bars
                    x = rect.get_x() + rect.get_width() / 2  # Center of bar
                    y = rect.get_y() + height / 2  # Middle of segment

                    if height < threshold:  # Small segment -> place label outside
                        ax.annotate(
                            f'{round(height, 1)}',
                            xy=(x, rect.get_y() + height),  # Arrow start
                            xytext=(x + 0.3, rect.get_y() + height + threshold),  # Offset text
                            arrowprops=dict(arrowstyle="->", color='black', lw=0.8),
                            fontsize='small', ha='left', va='center', color='black'
                        )
                    else:  # Large segment -> label inside
                        ax.text(x, y, f'{round(height, 1)}', ha='center', va='center', fontsize='small', color='white')

    # Set custom x-tick labels if _scenario_dict is provided
    if _scenario_dict:
        labels = [_scenario_dict.get(label, label) for label in pivot_central.index]
    else:
        labels = pivot_central.index.astype(str)

    # Wrap x-tick labels for readability
    wrapped_labels = [textwrap.fill(str(label), 20) for label in labels]
    ax.set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize='small')

    # Period included for plot title and name
    if _year == 'all':
        period = (f"{min(_df_central['year'].unique())} - {max(_df_central['year'].unique())}")
    elif (len(_year) == 1):
        period = (f"{_year[0]}")
    else:
        period = (f"{min(_year)} - {max(_year)}")

    # Save plot
    plt.xlabel('Scenario')
    plt.ylabel('Cost (2023 USD), millions')

    # Arrange the legend in the same ascending order
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 0.7), loc='center left', fontsize='small')

    # Extend the y-axis by 25%
    max_y = ax.get_ylim()[1]
    ax.set_ylim(0, max_y*1.25)

    # Save the plot with tight layout
    plt.tight_layout(pad=2.0)  # Ensure there is enough space for the legend
    plt.subplots_adjust(right=0.8) # Adjust to ensure legend doesn't overlap

    # Add gridlines and border
    plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray')
    #plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.edgecolor'] = 'gray'
    plt.rcParams['figure.frameon'] = True

    if show_title != False:
        plt.title(f'Costs by Scenario \n (Cost Category = {_cost_category} ; Period = {period})')
    plt.savefig(_outputfilepath / f'stacked_bar_chart_{_cost_category}_{period}{plt_name_suffix}{_add_figname_suffix}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

# 2. Line plots of total costs
#----------------------------------------------------
# TODO: Check why line plot get save without a file name
def do_line_plot_of_cost(_df: pd.DataFrame,
                         _cost_category: Literal['all', 'human resources for health', 'medical consumables',
                                            'medical equipment', 'facility operating cost'] = 'all',
                         _year: list[int] ='all',
                         _draws: list[int]=None,
                         disaggregate_by: Literal['cost_category', 'cost_subcategory', 'cost_subgroup']=None,
                         _y_lim: float = None,
                         show_title: bool = True,
                         _outputfilepath: Path = None)-> None:
    """
        Plot and save a line chart of cost trends over time by category or subcategory.

        Parameters:
        ----------
        _df : pd.DataFrame
            A cost summary DataFrame (usually from `estimate_input_cost_of_scenarios`)
            containing columns like ['year', 'draw', 'cost', 'stat', 'cost_category', etc.].

        _cost_category : str, default 'all'
            If 'all', plots total cost across all categories. Otherwise, filters to a specific category.

        _year : str or list of int, default 'all'
            Year(s) to include. Can be:
                - 'all' to include all
                - a single year or multiple years as a list: [2025]

        _draws : list of int, optional
            If specified, filters to those draws. Required if `disaggregate_by` is set.

        disaggregate_by : {'cost_category', 'cost_subcategory', 'cost_subgroup'}, optional
            Controls disaggregation on the plot
            Note: If disaggregating, `_draws` must contain **only one draw**.

        _y_lim : float, optional
            Custom upper limit for the y-axis. If None, uses automatic scaling.

        show_title : bool, default True
            Whether to show the plot title.

        _outputfilepath : Path, optional
            Directory where the plot image will be saved. Filename is auto-generated based on inputs.

        Returns:
        -------
        None
            Saves a PNG chart to `_outputfilepath`.
        """

    # Check what's the correct central metric to use (either 'mean' or 'median')
    central_metric = [stat for stat in _df.stat.unique() if stat not in ['lower', 'upper']][0]

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
            subgroup_totals = subset_df[subset_df.stat == central_metric].groupby('cost_subgroup')['cost'].sum()
            # Identify the top 10 subgroups by cost
            top_10_subgroups = subgroup_totals.nlargest(10).index.tolist()
            # Reassign smaller subgroups to an "Other" category
            subset_df['cost_subgroup'] = subset_df['cost_subgroup'].apply(
                lambda x: x if x in top_10_subgroups else 'Other'
            )
        groupby_columns = ['year', 'cost_subgroup']
    else:
        groupby_columns = ['year']

    # Extract central, lower, and upper values for the plot
    central_values = subset_df[subset_df.stat == central_metric].groupby(groupby_columns)['cost'].sum() / 1e6
    lower_values = subset_df[subset_df.stat == 'lower'].groupby(groupby_columns)['cost'].sum() / 1e6
    upper_values = subset_df[subset_df.stat == 'upper'].groupby(groupby_columns)['cost'].sum() / 1e6

    # Prepare to store lines and labels for the legend
    lines = []
    labels = []

    # Define a list of colors
    if disaggregate_by == 'cost_category':
        color_mapping = {
            'human resources for health': '#1f77b4',  # Muted blue
            'medical consumables': '#ff7f0e',  # Muted orange
            'medical equipment': '#2ca02c',  # Muted green
            'other': '#d62728',  # Muted red
            'facility operating cost': '#9467bd',  # Muted purple
        }
        # Default color for unexpected categories
        default_color = 'gray'
    else:
        # Define a list of colors to rotate through
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'gray']  # Add more colors as needed
        color_cycle = iter(colors)  # Create an iterator from the color list

    # Plot each line for the disaggregated values
    if disaggregate_by:
        for disaggregate_value in central_values.index.get_level_values(disaggregate_by).unique():
            # Get central, lower, and upper values for each disaggregated group
            value_central = central_values.xs(disaggregate_value, level=disaggregate_by)
            value_lower = lower_values.xs(disaggregate_value, level=disaggregate_by)
            value_upper = upper_values.xs(disaggregate_value, level=disaggregate_by)

            if disaggregate_by == 'cost_category':
                color = color_mapping.get(disaggregate_value, default_color)
            else:
                # Get the next color from the cycle
                color = next(color_cycle)

            # Plot line for central and shaded region for 95% CI
            line, = plt.plot(value_central.index, value_central, marker='o', linestyle='-', color=color, label=f'{disaggregate_value} - {central_metric}')
            plt.fill_between(value_central.index, value_lower, value_upper, color=color, alpha=0.2)

            # Append to lines and labels for sorting later
            lines.append(line)
            labels.append(disaggregate_value)
    else:
        line, = plt.plot(central_values.index, central_values, marker='o', linestyle='-', color='b', label=central_metric)
        plt.fill_between(central_values.index, lower_values, upper_values, color='b', alpha=0.2)

        # Append to lines and labels for sorting later
        lines.append(line)
        labels.append(central_metric)

    # Sort the legend based on total costs
    total_costs = {label: central_values.xs(label, level=disaggregate_by).sum() for label in labels}
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
    # Set y-axis limit if provided
    if _y_lim is not None:
        plt.ylim(0, _y_lim)

    # Add gridlines and border
    plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray')
    plt.xlabel('Year')
    plt.ylabel('Cost (2023 USD), millions')
    plt.legend(handles[::-1], sorted_labels[::-1], loc='upper right', bbox_to_anchor=(0.98, 0.98), framealpha=0.6)
    if (show_title != False):
        plot_title = f'Total input cost \n (Category = {_cost_category}, Period = {period})'
        plt.title(plot_title)

    # Save plot with a proper filename
    if disaggregate_by is None:
        filename_suffix = ""
    else:
        filename_suffix = f"_by_{disaggregate_by}"

    draw_suffix = 'all' if _draws is None else str(_draws)
    filename = f'trend_{_cost_category}_{period}{filename_suffix}_draw-{draw_suffix}.png'
    plt.savefig(_outputfilepath / filename, dpi=100, bbox_inches='tight')
    plt.close()

# Treemap by category subgroup
#-----------------------------------------------------------------------------------------------
def create_summary_treemap_by_cost_subgroup(_df: pd.DataFrame,
                                            _cost_category: Literal['all', 'human resources for health', 'medical consumables',
                                            'medical equipment', 'facility operating cost'] = None,
                                            _draw: list[int] = None,
                                            _year: list[int] = 'all',
                                            _color_map: dict[str, str] = None,
                                            _label_fontsize: int = 10,
                                            show_title: bool = True,
                                            _outputfilepath: Path = None) -> None:
    """
        Generate and save a treemap visualizing cost composition by subgroup within a cost category.

        Parameters:
        ----------
        _df : pd.DataFrame
            DataFrame of costs with columns: ['cost_category', 'cost_subgroup', 'draw', 'year', 'cost'].
            Typically output from `estimate_input_cost_of_scenarios`.

        _cost_category : str, required
            The high-level cost category to visualize (e.g., 'human resources for health',
            'medical consumables', 'medical equipment', 'facility operating cost').

        _draw : int, optional
            Specific draw to visualize. If None, uses the full dataset.

        _year : str or list of int, default 'all'
            Year or list of years to include in the treemap. If 'all', includes all available years.

        _color_map : dict, optional
            Dictionary mapping cost subgroups to specific colors. If None, a default colormap is used.
            eg. _color_map = {'First-line ART regimen: adult':'#1f77b4',
                             'Test, HIV EIA Elisa': '#ff7f0e',
                             'VL Test': '#2ca02c'}

        _label_fontsize : int, default 10
            Font size used for labels inside treemap tiles.

        show_title : bool, default True
            Whether to display a plot title.

        _outputfilepath : Path, optional
            Directory where the treemap image should be saved.

        Returns:
        -------
        None
            Saves the treemap as a PNG file named `treemap_{category}_{draw}_{period}.png`.
        """
    # Function to wrap text to fit within treemap rectangles
    def wrap_text(text, width=15):
        return "\n".join(textwrap.wrap(text, width))

    valid_cost_categories = ['human resources for health', 'medical consumables',
       'medical equipment', 'facility operating cost']
    if _cost_category == None:
        raise ValueError(f"Specify one of the following as _cost_category - {valid_cost_categories})")
    elif _cost_category not in valid_cost_categories:
        raise ValueError(f"Invalid input for _cost_category: '{_cost_category}'. "
                     f"Specify one of the following - {valid_cost_categories})")
    else:
        _df = _df[_df['cost_category'] == _cost_category]

    if _draw != None:
        _df = _df[_df.draw == _draw]

    # Remove non-specific subgroup for consumables
    if _cost_category == 'medical consumables':
        _df = _df[~(_df.cost_subgroup == 'supply chain (all consumables)')]

    # Create summary dataframe for treemap
    _df = _df.groupby('cost_subgroup')['cost'].sum().reset_index()
    _df = _df.sort_values(by="cost", ascending=False)
    top_10 = _df.iloc[:10]

    if (len(_df['cost_subgroup'].unique()) > 10):
        # Step 2: Group all other consumables into "Other"
        other_cost = _df.iloc[10:]["cost"].sum()
        top_10 = pd.concat([top_10, pd.DataFrame([{"cost_subgroup": "Other", "cost": other_cost}])], ignore_index=True)

    # Prepare data for the treemap
    total_cost = top_10["cost"].sum()
    top_10["proportion"] = top_10["cost"]/total_cost
    sizes = top_10["cost"]

    # Handle color map
    if _color_map is None:
        # Generate automatic colors if no color map is provided
        auto_colors = plt.cm.Paired.colors
        color_cycle = cycle(auto_colors)  # Cycle through the automatic colors
        color_map = {subgroup: next(color_cycle) for subgroup in top_10["cost_subgroup"]}
    else:
        # Use the provided color map, fallback to a default color for missing subgroups
        fallback_color = '#cccccc'
        color_map = {subgroup: _color_map.get(subgroup, fallback_color) for subgroup in top_10["cost_subgroup"]}

    # Get colors for each subgroup
    colors = [color_map[subgroup] for subgroup in top_10["cost_subgroup"]]

    # Exclude labels for small proportions
    labels = [
        f"{wrap_text(name)}\n${round(cost, 1)}m\n({round(prop * 100, 1)}%)"
        if prop >= 0.01 else ""
        for name, cost, prop in zip(top_10["cost_subgroup"], top_10["cost"] / 1e6, top_10["proportion"])
    ]
    # Period included for plot title and name
    if _year == 'all':
        period = (f"{min(_df['year'].unique())} - {max(_df['year'].unique())}")
    elif (len(_year) == 1):
        period = (f"{_year[0]}")
    else:
        period = (f"{min(_year)} - {max(_year)}")

    # Plot the treemap
    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, alpha=0.8, color=colors, text_kwargs={'fontsize': _label_fontsize})
    plt.axis("off")
    if (show_title != False):
        plt.title(f'{_cost_category} ; Period = {period}')

    plt.savefig(_outputfilepath / f'treemap_{_cost_category}_[{_draw}]_{period}.png',
                dpi=100,
                bbox_inches='tight')
    plt.close()

# Plot ROI
# TODO update this function to include an input for the monetary value of DALY
def generate_roi_plots(_monetary_value_of_incremental_health: pd.DataFrame,
                       _incremental_input_cost: pd.DataFrame,
                       _scenario_dict: dict,
                       _outputfilepath: Path,
                       _value_of_life_suffix = ''):
    # Calculate maximum ability to pay for implementation
    max_ability_to_pay_for_implementation = (_monetary_value_of_incremental_health - _incremental_input_cost).clip(
        lower=0.0)  # monetary value - change in costs

    # Iterate over each draw in monetary_value_of_incremental_health
    for draw_index, row in _monetary_value_of_incremental_health.iterrows():
        print("Plotting ROI for draw ", draw_index)
        # Initialize an empty DataFrame to store values for each 'run'
        all_run_values = pd.DataFrame()

        # Create an array of implementation costs ranging from 0 to the max value of max ability to pay for the current draw
        implementation_costs = np.linspace(0, max_ability_to_pay_for_implementation.loc[draw_index].max(), 50)

        # Retrieve the corresponding row from incremental_scenario_cost for the same draw
        incremental_scenario_cost_row = _incremental_input_cost.loc[draw_index]

        # Calculate the values for each individual run
        for run in incremental_scenario_cost_row.index:  # Assuming 'run' columns are labeled by numbers
            # Calculate the total costs for the current run
            total_costs = implementation_costs + incremental_scenario_cost_row[run]

            # Initialize run_values as an empty series with the same index as total_costs
            run_values = pd.Series(index=total_costs, dtype=float)

            # For negative total_costs, set corresponding run_values to infinity
            run_values[total_costs < 0] = np.inf

            # For non-negative total_costs, calculate the metric and clip at 0
            non_negative_mask = total_costs >= 0
            run_values[non_negative_mask] = np.clip(
                (row[run] - total_costs[non_negative_mask]) / total_costs[non_negative_mask],
                0,
                None
            )

            # Create a DataFrame with index as (draw_index, run) and columns as implementation costs
            run_values = run_values.values # remove index and convert to array
            run_df = pd.DataFrame([run_values], index=pd.MultiIndex.from_tuples([(draw_index, run)], names=['draw', 'run']),
                                  columns=implementation_costs)

            # Append the run DataFrame to all_run_values
            all_run_values = pd.concat([all_run_values, run_df])

        # Replace inf with NaN temporarily to handle quantile calculation correctly
        temp_data = all_run_values.replace([np.inf, -np.inf], np.nan)

        collapsed_data = temp_data.groupby(level='draw').agg([
            'mean',
            ('lower', lambda x: x.quantile(0.025)),
            ('upper', lambda x: x.quantile(0.975))
        ])

        # Revert the NaNs back to inf
        collapsed_data = collapsed_data.replace([np.nan], np.inf)

        collapsed_data = collapsed_data.unstack()
        collapsed_data.index = collapsed_data.index.set_names('implementation_cost', level=0)
        collapsed_data.index = collapsed_data.index.set_names('stat', level=1)
        collapsed_data = collapsed_data.reset_index().rename(columns = {0: 'roi'})
        #collapsed_data = collapsed_data.reorder_levels(['draw', 'stat', 'implementation_cost'])

        # Divide rows by the sum of implementation costs and incremental input cost
        mean_values = collapsed_data[collapsed_data['stat'] == 'mean'][['implementation_cost', 'roi']]
        lower_values = collapsed_data[collapsed_data['stat'] == 'lower'][['implementation_cost', 'roi']]
        upper_values = collapsed_data[collapsed_data['stat']  == 'upper'][['implementation_cost', 'roi']]

        fig, ax = plt.subplots()  # Create a figure and axis

        # Plot mean line
        plt.plot(implementation_costs / 1e6, mean_values['roi'], label=f'{_scenario_dict[draw_index]}')
        # Plot the confidence interval as a shaded region
        plt.fill_between(implementation_costs / 1e6, lower_values['roi'], upper_values['roi'], alpha=0.2)

        # Set y-axis limit to upper max + 500
        ax.set_ylim(0, mean_values[~np.isinf(mean_values.roi)]['roi'].max()*(1+0.05))

        plt.xlabel('Implementation cost, millions')
        plt.ylabel('Return on Investment')
        plt.title('Return on Investment of scenario at different levels of implementation cost')

        monetary_value_of_incremental_health_summarized = summarize_cost_data(_monetary_value_of_incremental_health)
        incremental_scenario_cost_row_summarized =  incremental_scenario_cost_row.agg(
                                                        mean='mean',
                                                        lower=lambda x: x.quantile(0.025),
                                                        upper=lambda x: x.quantile(0.975))

        plt.text(x=0.95, y=0.8,
                 s=f"Monetary value of incremental health = \n USD {round(monetary_value_of_incremental_health_summarized.loc[draw_index]['mean'] / 1e6, 2)}m (USD {round(monetary_value_of_incremental_health_summarized.loc[draw_index]['lower'] / 1e6, 2)}m-{round(monetary_value_of_incremental_health_summarized.loc[draw_index]['upper'] / 1e6, 2)}m);\n "
                   f"Incremental input cost of scenario = \n USD {round(incremental_scenario_cost_row_summarized['mean'] / 1e6, 2)}m (USD {round(incremental_scenario_cost_row_summarized['lower'] / 1e6, 2)}m-{round(incremental_scenario_cost_row_summarized['upper'] / 1e6, 2)}m)",
                 horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=9,
                 weight='bold', color='black')

        # Show legend
        plt.legend()
        # Save
        plt.savefig(_outputfilepath / f'draw{draw_index}_{_scenario_dict[draw_index]}_ROI_at_{_value_of_life_suffix}.png', dpi=100,
                    bbox_inches='tight')
        plt.close()

def generate_multiple_scenarios_roi_plot(_monetary_value_of_incremental_health: pd.DataFrame,
                       _incremental_input_cost: pd.DataFrame,
                       _draws:None,
                       _scenario_dict: dict,
                       _outputfilepath: Path,
                       _value_of_life_suffix = '',
                       _metric: str = 'mean',
                       _y_axis_lim = None,
                      _plot_vertical_lines_at: list = None,
                      _year_suffix = '',
                      _projected_health_spending = None,
                      _draw_colors = None,
                      show_title_and_legend = None):
    if _metric not in ['mean', 'median']:
        raise ValueError(f"Invalid input for _metric: '{_metric}'. "
                         f"Values need to be one of 'mean' or 'median'")

    # Default color mapping if not provided
    if _draw_colors is None:
        _draw_colors = {draw: color for draw, color in zip(_draws, plt.cm.tab10.colors[:len(_draws)])}

    # Calculate maximum ability to pay for implementation
    _monetary_value_of_incremental_health = _monetary_value_of_incremental_health[_monetary_value_of_incremental_health.index.get_level_values('draw').isin(_draws)]
    _incremental_input_cost =  _incremental_input_cost[_incremental_input_cost.index.get_level_values('draw').isin(_draws)]
    max_ability_to_pay_for_implementation = (_monetary_value_of_incremental_health - _incremental_input_cost).clip(lower=0.0)  # monetary value - change in costs

    # Create a figure and axis to plot all draws together
    fig, ax = plt.subplots(figsize=(10, 6))

    # Store ROI values for specific costs
    max_roi = []
    roi_at_costs = {cost: [] for cost in (_plot_vertical_lines_at or [])}

    # Iterate over each draw in monetary_value_of_incremental_health
    for draw_index, row in _monetary_value_of_incremental_health.iterrows():
        print("Plotting ROI for draw ", draw_index)
        # Initialize an empty DataFrame to store values for each 'run'
        all_run_values = pd.DataFrame()

        # Create an array of implementation costs ranging from 0 to the max value of max ability to pay for the current draw
        implementation_costs = np.linspace(0, max_ability_to_pay_for_implementation.loc[draw_index].max(), 50)
        # Add fixed values for ROI ratio calculation
        additional_costs = np.array([1_000_000_000, 3_000_000_000])
        implementation_costs = np.sort(np.unique(np.concatenate([implementation_costs, additional_costs])))

        # Retrieve the corresponding row from incremental_scenario_cost for the same draw
        incremental_scenario_cost_row = _incremental_input_cost.loc[draw_index]

        # Calculate the values for each individual run
        for run in incremental_scenario_cost_row.index:  # Assuming 'run' columns are labeled by numbers
            # Calculate the total costs for the current run
            total_costs = implementation_costs + incremental_scenario_cost_row[run]

            # Initialize run_values as an empty series with the same index as total_costs
            run_values = pd.Series(index=total_costs, dtype=float)

            # For negative total_costs, set corresponding run_values to infinity
            run_values[total_costs < 0] = np.inf

            # For non-negative total_costs, calculate the metric and clip at 0
            non_negative_mask = total_costs >= 0
            run_values[non_negative_mask] = np.clip(
                (row[run] - total_costs[non_negative_mask]) / total_costs[non_negative_mask],
                0,
                None
            )

            # Create a DataFrame with index as (draw_index, run) and columns as implementation costs
            run_values = run_values.values # remove index and convert to array
            run_df = pd.DataFrame([run_values], index=pd.MultiIndex.from_tuples([(draw_index, run)], names=['draw', 'run']),
                                  columns=implementation_costs)

            # Append the run DataFrame to all_run_values
            all_run_values = pd.concat([all_run_values, run_df])

        # Replace inf with NaN temporarily to handle quantile calculation correctly
        temp_data = all_run_values.replace([np.inf, -np.inf], np.nan)

        collapsed_data = temp_data.groupby(level='draw').agg([
            _metric,
            ('lower', lambda x: x.quantile(0.025)),
            ('upper', lambda x: x.quantile(0.975))
        ])

        # Revert the NaNs back to inf
        collapsed_data = collapsed_data.replace([np.nan], np.inf)

        collapsed_data = collapsed_data.unstack()
        collapsed_data.index = collapsed_data.index.set_names('implementation_cost', level=0)
        collapsed_data.index = collapsed_data.index.set_names('stat', level=1)
        collapsed_data = collapsed_data.reset_index().rename(columns = {0: 'roi'})

        # Divide rows by the sum of implementation costs and incremental input cost
        central_values = collapsed_data[collapsed_data['stat'] == _metric][['implementation_cost', 'roi']]
        lower_values = collapsed_data[collapsed_data['stat'] == 'lower'][['implementation_cost', 'roi']]
        upper_values = collapsed_data[collapsed_data['stat']  == 'upper'][['implementation_cost', 'roi']]

        # Plot central line and confidence interval
        ax.plot(
            implementation_costs / 1e6,
            central_values['roi'],
            label=f'{_scenario_dict[draw_index]}',
            color=_draw_colors.get(draw_index, 'black'),
        )
        ax.fill_between(
            implementation_costs / 1e6,
            lower_values['roi'],
            upper_values['roi'],
            alpha=0.2,
            color=_draw_colors.get(draw_index, 'black'),
        )

        max_val = central_values[~np.isinf(central_values['roi'])]['roi'].max()
        max_roi.append(max_val)

        # Capture ROI at specific costs
        if _plot_vertical_lines_at:
            for cost in _plot_vertical_lines_at:
                roi_value = collapsed_data[
                    (collapsed_data.implementation_cost == cost) &
                    (collapsed_data.stat == _metric)
                    ]['roi']
                if not roi_value.empty:
                    roi_at_costs[cost].append(roi_value.iloc[0])

    # Calculate and annotate ROI ratios
    if _plot_vertical_lines_at:
        for cost in _plot_vertical_lines_at:
            if cost in roi_at_costs:
                ratio = max(roi_at_costs[cost]) / min(roi_at_costs[cost])
                ax.axvline(x=cost / 1e6, color='black', linestyle='--', linewidth=1)
                ax.text(cost / 1e6 + ax.get_xlim()[1] * 0.011, ax.get_ylim()[1] * 0.75,
                        f'At ${cost / 1e6:.0f}M, ratio of ROI curves = {round(ratio, 2)}',
                        color='black', fontsize=10, rotation=90, verticalalignment='top')

    # Define fixed x-tick positions with a gap of 2000
    step_size = (ax.get_xlim()[1] - 0)/5
    xticks = np.arange(0, ax.get_xlim()[1] + 1, int(round(step_size, -3)))  # From 0 to max x-limit with 5 steps
    # Get labels
    xtick_labels = [f'{tick:,.0f}' for tick in xticks]  # Default labels for all ticks

    # Replace specific x-ticks with % of health spending values
    if _projected_health_spending:
        xtick_labels[1] = f'{xticks[1]:,.0f}\n({xticks[1] / (_projected_health_spending / 1e6) :.2%} of \n projected total \n health spend)'
        for i, tick in enumerate(xticks):
            if (i != 0) & (i != 1):  # Replace for 4000
                xtick_labels[i] = f'{tick:,.0f}\n({tick / (_projected_health_spending/1e6) :.2%})'

        # Update the x-ticks and labels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=10)

    # Set y-axis limit
    if _y_axis_lim == None:
        ax.set_ylim(0, max(max_roi) * 1.25)
    else:
        ax.set_ylim(0, _y_axis_lim)
    ax.set_xlim(left = 0)

    plt.xlabel('Implementation cost, USD millions')
    plt.ylabel('Return on Investment')

    # Show legend and title
    if (show_title_and_legend != False):
        plt.title(f'Return on Investment at different levels of implementation cost{_year_suffix}')
        plt.legend()

    # Add gridlines and border
    plt.grid(False)
    fig.patch.set_facecolor("white")  # White background for the entire figure

    # Save
    plt.savefig(_outputfilepath / f'draws_{_draws}_ROI_at_{_value_of_life_suffix}_{_year_suffix}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

def tabulated_roi_estimates(_monetary_value_of_incremental_health: pd.DataFrame,
                       _incremental_input_cost: pd.DataFrame,
                       _draws:None,
                       _scenario_dict: dict,
                       _metric = 'mean'):
    # Calculate maximum ability to pay for implementation
    _monetary_value_of_incremental_health = _monetary_value_of_incremental_health[_monetary_value_of_incremental_health.index.get_level_values('draw').isin(_draws)]
    _incremental_input_cost =  _incremental_input_cost[_incremental_input_cost.index.get_level_values('draw').isin(_draws)]
    max_ability_to_pay_for_implementation = (_monetary_value_of_incremental_health - _incremental_input_cost).clip(lower=0.0)  # monetary value - change in costs

    roi_df = pd.DataFrame()

    # Create an array of implementation costs ranging from 0 to the max value of max ability to pay for the current draw
    max_ability_to_pay_for_implementation_rounded_value = math.ceil(max_ability_to_pay_for_implementation.max().max() / 1_000_000_000) * 1_000_000_000
    implementation_costs = np.linspace(0, max_ability_to_pay_for_implementation_rounded_value, 20)
    implementation_costs = np.ceil(implementation_costs / 1_000_000_000) * 1_000_000_000  # Round each to nearest billion

    # Iterate over each draw in monetary_value_of_incremental_health
    for draw_index, row in _monetary_value_of_incremental_health.iterrows():
        print("Tablulating ROI for draw ", draw_index)
        # Initialize an empty DataFrame to store values for each 'run'
        all_run_values = pd.DataFrame()

        # Retrieve the corresponding row from incremental_scenario_cost for the same draw
        incremental_scenario_cost_row = _incremental_input_cost.loc[draw_index]

        # Calculate the values for each individual run
        for run in incremental_scenario_cost_row.index:  # Assuming 'run' columns are labeled by numbers
            # Calculate the total costs for the current run
            total_costs = implementation_costs + incremental_scenario_cost_row[run]

            # Initialize run_values as an empty series with the same index as total_costs
            run_values = pd.Series(index=total_costs, dtype=float)

            # For negative total_costs, set corresponding run_values to infinity
            run_values[total_costs < 0] = np.inf

            # For non-negative total_costs, calculate the metric and clip at 0
            non_negative_mask = total_costs >= 0
            run_values[non_negative_mask] = (row[run] - total_costs[non_negative_mask]) / total_costs[non_negative_mask]

            # Create a DataFrame with index as (draw_index, run) and columns as implementation costs
            run_values = run_values.values # remove index and convert to array
            run_df = pd.DataFrame([run_values], index=pd.MultiIndex.from_tuples([(draw_index, run)], names=['draw', 'run']),
                                  columns=implementation_costs)

            # Append the run DataFrame to all_run_values
            all_run_values = pd.concat([all_run_values, run_df])

        # Replace inf with NaN temporarily to handle quantile calculation correctly
        temp_data = all_run_values.replace([np.inf, -np.inf], np.nan)

        collapsed_data = temp_data.groupby(level='draw').agg([
            _metric,
            ('lower', lambda x: x.quantile(0.025)),
            ('upper', lambda x: x.quantile(0.975))
        ])

        # Revert the NaNs back to inf
        collapsed_data = collapsed_data.replace([np.nan], np.inf)

        collapsed_data = collapsed_data.unstack()
        collapsed_data.index = collapsed_data.index.set_names('implementation_cost', level=0)
        collapsed_data.index = collapsed_data.index.set_names('stat', level=1)
        collapsed_data = collapsed_data.reset_index().rename(columns = {0: 'roi'})

        if roi_df.empty:
            roi_df = collapsed_data
        else:
            roi_df =  pd.concat([roi_df, collapsed_data], ignore_index=True)
    return roi_df
