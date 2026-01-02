"""
This script generates estimates of availability of consumables used by disease modules:

OUTPUTS:
* ResourceFile_Consumables_availability_small.csv (estimate of consumable available - file for use in the
 simulation).
* ResourceFile_Consumables_Inflow_Outflow_Ratio.csv (a file that gives the ratio of inflow of consumables to outflow to
* capture the extent of wastage as a proportion of use for each consumable by month, district and level.

INPUTS:
* `ResourceFile_Consumables_matched.csv` - matches consumable names in OpenLMIS to those in TLO model
*  `ResourceFile_LMIS_2018.csv` - consumable availability in OpenLMIS 2018. Data from OpenLMIS includes closing balance,
quantity received, quantity dispensed, and average monthly consumption for each month by facility.
* `ResourceFile_hhfa_consumables.xlsx` - provides consumable availability from other sources, mainly Harmonised Health
 Facility Assessment 2018-19 (to fill gaps in Open LMIS data
* `ResourceFile_Consumables_Item_Designations.csv` to categorise consumables into disease/public health programs
* `ResourceFile_Master_Facilities_List.csv` - to obtain the Facility_Level associated with each Facility_ID
* `ResourceFile_Population_2010.csv` - to get the list of districts

It creates one row for each consumable for availability at a specific facility and month when the data is extracted from
the OpenLMIS dataset and one row for each consumable for availability aggregated across all facilities when the data is
extracted from the Harmonised Health Facility Assessment 2018/19.

Consumable availability is measured as probability of stockout at any point in time.

Steps:
1. Prepare OpenLMIS data (A. Import, B. Reshape, C. Interpolate, D. Summarise by month and facility)
2. Match with TLO Model consumable names
3. Add data from other sources where OpenLMIS data is missing
4. Interpolate missing data
5. Add alternative availability scenarios
6. Check format and save as resourcefile
7. Produce validation plots
8. Plot summary of availability across scenarios
"""

import calendar
import datetime
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List

from tlo.methods.consumables import check_format_of_consumables_file
from scripts.data_file_processing.healthsystem.consumables.generating_consumable_scenarios import generate_alternative_availability_scenarios, generate_descriptive_consumable_availability_plots

# Set local shared folder source
path_to_share = Path(  # <-- point to the shared folder
    '/Users/sm2511/CloudStorage/OneDrive-SharedLibraries-ImperialCollegeLondon/TLOModel - WP - Documents/'
)

path_to_files_in_the_tlo_shared_drive = path_to_share / "07 - Data/Consumables data/"

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define a pathway to the data folder (note: currently outside the TLO model directory)
# remember to set working directory to TLOmodel/
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"

# Define necessary functions
# Functions to clean LMIS data
def change_colnames(df, NameChangeList):  # Change column names
    ColNames = df.columns
    ColNames2 = ColNames
    for (a, b) in NameChangeList:
        print(a, '-->', b)
        ColNames2 = [col.replace(a, b) for col in ColNames2]
    df.columns = ColNames2
    return df

def rename_items_to_address_inconsistentencies(_df, item_dict):
    """Return a dataframe with rows for the same item with inconsistent names collapsed into one"""
    # Recode item names appearing from Jan to Aug to the new names adopted from September onwards
    old_unique_item_count = _df.item.nunique()
    for item in item_dict:
        print(len(_df[_df.item == item_dict[item]]), ''' instances of "''', item_dict[item], '''"'''
                                                                                             ''' changed to "''', item,
              '''"''')
        # row_newname = _df.item == item
        row_oldname = _df.item == item_dict[item]
        _df.loc[row_oldname, 'item'] = item

    # Make a list of column names to be collapsed using different methods
    columns_to_sum = [col for col in _df.columns if
                      col[0].startswith(('amc', 'closing_bal', 'dispensed', 'received', 'stkout_days'))]
    columns_to_preserve = [col for col in _df.columns if
                           col[0].startswith(('data_source'))]

    # Define aggregation function to be applied to collapse data by item
    def custom_agg(x):
        if x.name in columns_to_sum:
            return x.sum(skipna=True) if np.any(
                x.notnull() & (x >= 0)) else np.nan  # this ensures that the NaNs are retained
        # , i.e. not changed to 0, when the corresponding data for both item name variations are NaN, and when there
        # is a 0 or positive value for one or both item name variation, the sum is taken.
        elif x.name in columns_to_preserve:
            return x.str.cat(
                sep='')  # for the data_source column, this function concatenates the string values

    # Collapse dataframe
    _collapsed_df = _df.groupby(['program', 'item', 'district', 'fac_type_tlo', 'fac_name']).agg(
        {col: custom_agg for col in columns_to_preserve + columns_to_sum}
    ).reset_index()

    # Test that all items in the dictionary have been found in the dataframe
    new_unique_item_count = _collapsed_df.item.nunique()
    assert len(item_dict) == old_unique_item_count - new_unique_item_count
    return _collapsed_df

def replace_old_item_names_in_lmis_data(_df, item_dict):
    """Return a dataframe with old LMIS consumable names replaced with the new name"""
    for item in item_dict:
        cond_oldname = _df.item == item_dict[item]
        _df.loc[cond_oldname, 'item'] = item
    return _df

def recategorize_modules_into_consumable_categories(_df):
    _df['item_category'] = _df['module_name'].str.lower()
    cond_RH = (_df['item_category'].str.contains('care_of_women_during_pregnancy')) | \
              (_df['item_category'].str.contains('labour'))
    cond_newborn = (_df['item_category'].str.contains('newborn'))
    cond_newborn[cond_newborn.isna()] = False
    cond_childhood = (_df['item_category'] == 'acute lower respiratory infections') | \
                     (_df['item_category'] == 'measles') | \
                     (_df['item_category'] == 'diarrhoea')
    cond_rti = _df['item_category'] == 'road traffic injuries'
    cond_cancer = _df['item_category'].str.contains('cancer')
    cond_cancer[cond_cancer.isna()] = False
    cond_ncds = (_df['item_category'] == 'epilepsy') | \
                (_df['item_category'] == 'depression')
    _df.loc[cond_RH, 'item_category'] = 'reproductive_health'
    _df.loc[cond_cancer, 'item_category'] = 'cancer'
    _df.loc[cond_newborn, 'item_category'] = 'neonatal_health'
    _df.loc[cond_childhood, 'item_category'] = 'other_childhood_illnesses'
    _df.loc[cond_rti, 'item_category'] = 'road_traffic_injuries'
    _df.loc[cond_ncds, 'item_category'] = 'ncds'
    cond_condom = _df['item_code'] == 2
    _df.loc[cond_condom, 'item_category'] = 'contraception'

    # Create a general consumables category
    general_cons_list = [300, 33, 57, 58, 141, 5, 6, 10, 21, 23, 127, 24, 80, 93, 144, 149, 154, 40, 67, 73, 76,
                         82, 101, 103, 88, 126, 135, 71, 98, 171, 133, 134, 244, 247, 49, 112, 1933, 1960]
    cond_general = _df['item_code'].isin(general_cons_list)
    _df.loc[cond_general, 'item_category'] = 'general'

    # Fill gaps in categories
    dict_for_missing_categories = {292: 'acute lower respiratory infections', 293: 'acute lower respiratory infections',
                                   307: 'reproductive_health', 2019: 'reproductive_health',
                                   2678: 'tb', 1171: 'other_childhood_illnesses', 1237: 'cancer', 1239: 'cancer'}
    # Use map to create a new series from item_code to fill missing values in category
    mapped_categories = _df['item_code'].map(dict_for_missing_categories)
    # Use fillna on the 'item_category' column to fill missing values using the mapped_categories
    _df['item_category'] = _df['item_category'].fillna(mapped_categories)

    return _df

# Function to extract inflow to outflow ratio for costing
def get_inflow_to_outflow_ratio_by_item_and_facilitylevel(_df):
    df_by_item_level_month = \
    _df.groupby(['item_category', 'item_code', 'district', 'fac_type_tlo', 'month'])[
        ['closing_bal', 'dispensed', 'received']].sum()
    df_by_item_level_month = df_by_item_level_month.loc[df_by_item_level_month.index.get_level_values('month') != "Aggregate"]
    # Opening balance in January is the closing balance for the month minus what was received during the month plus what was dispensed
    opening_bal_january = df_by_item_level_month.loc[df_by_item_level_month.index.get_level_values('month') == 'January', 'closing_bal'] + \
                          df_by_item_level_month.loc[df_by_item_level_month.index.get_level_values('month') == 'January', 'dispensed'] - \
                          df_by_item_level_month.loc[df_by_item_level_month.index.get_level_values('month') == 'January', 'received']
    closing_bal_december = df_by_item_level_month.loc[df_by_item_level_month.index.get_level_values('month') == 'December', 'closing_bal']
    # the consumable inflow during the year is the opening balance in January + what was received throughout the year - what was transferred to the next year (i.e. closing bal of December)
    total_consumables_inflow_during_the_year = df_by_item_level_month['received'].groupby(level=['item_category', 'item_code', 'district', 'fac_type_tlo']).sum() +\
                                             opening_bal_january.reset_index(level='month', drop=True) -\
                                             closing_bal_december.reset_index(level='month', drop=True)
    total_consumables_outflow_during_the_year  = df_by_item_level_month['dispensed'].groupby(level=['item_category', 'item_code', 'district', 'fac_type_tlo']).sum()
    inflow_to_outflow_ratio = total_consumables_inflow_during_the_year.div(total_consumables_outflow_during_the_year, fill_value=1)
    inflow_to_outflow_ratio.loc[inflow_to_outflow_ratio < 1] = 1  # Ratio can't be less than 1

    return inflow_to_outflow_ratio

def update_level1b_availability(
    availability_df: pd.DataFrame,
    facilities_by_level: dict,
    resourcefilepath: Path,
    district_to_city_dict: dict,
    weighting: str = "district_1b_to_2_ratio"
) -> pd.DataFrame:
    """
    Updates the availability of Level 1b facilities to be the weighted average
    of availability at Level 1b and 2 facilities, since these levels are merged
    together in simulations.

    weighting : {'level2', 'national_1b_to_2_ratio', 'district_1b_to_2_ratio'}, default 'district_1b_to_2_ratio'
        Weighting strategy:
            - 'level2': Replace 1b availability entirely with level 2 values.
            - 'national_1b_to_2_ratio': Apply a single national 1b:2 ratio to all districts.
            - 'district_1b_to_2_ratio': (default) Use district-specific 1b:2 ratios.
    """
    # Load and prepare base weights (facility counts)
    # ---------------------------------------------------------------------
    weight = (
        pd.read_csv(resourcefilepath / 'healthsystem' / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')
        [["District", "Facility_Level", "Facility_ID", "Facility_Count"]]
    )

    # Keep only Level 1b and 2 facilities
    lvl1b2_weights = weight[weight["Facility_Level"].isin(["1b", "2"])].copy()

    # Compute weights depending on strategy
    # ---------------------------------------------------------------------
    if weighting == "level2":
        # Force all weight on level 2
        lvl1b2_weights = lvl1b2_weights[~lvl1b2_weights.District.str.contains("City")]
        lvl1b2_weights["weight"] = (lvl1b2_weights["Facility_Level"] == "2").astype(float)
        lvl1b2_weights = lvl1b2_weights.drop(columns = 'Facility_ID')

    elif weighting == "national_1b_to_2_ratio":
        lvl1b2_weights = lvl1b2_weights[~lvl1b2_weights.District.str.contains("City")]
        # National total counts
        national_counts = (
            lvl1b2_weights.groupby("Facility_Level")["Facility_Count"].sum().to_dict()
        )
        total_fac = national_counts.get("1b", 0) + national_counts.get("2", 0)
        if total_fac == 0:
            raise ValueError("No facilities found at levels 1b or 2.")
        lvl1b2_weights["weight"] = lvl1b2_weights["Facility_Level"].map(
            {lvl: cnt / total_fac for lvl, cnt in national_counts.items()}
        )
        lvl1b2_weights = lvl1b2_weights.drop(columns='Facility_ID')

    elif weighting == "district_1b_to_2_ratio":
        # Replace city names with their parent districts (temporarily for grouping)
        city_to_district_dict = {v: k for k, v in district_to_city_dict.items()}
        lvl1b2_weights["District"] = lvl1b2_weights["District"].replace(city_to_district_dict)

        # District-level weighting (default)
        lvl1b2_weights = (
            lvl1b2_weights
            .groupby(["District", "Facility_Level"], as_index=False)["Facility_Count"]
            .sum()
        )

        lvl1b2_weights["total_facilities"] = lvl1b2_weights.groupby("District")["Facility_Count"].transform("sum")
        lvl1b2_weights["weight"] = lvl1b2_weights["Facility_Count"] / lvl1b2_weights["total_facilities"]

    else:
        raise ValueError(
            f"Invalid weighting '{weighting}'. Choose from "
            "'level2', 'national_1b_to_2_ratio', or 'district_1b_to_2_ratio'."
        )

    # Add back city districts (reverse mapping)
    for source, destination in copy_source_to_destination.items():
        new_rows = lvl1b2_weights.loc[lvl1b2_weights.District == source].copy()
        new_rows.District = destination
        lvl1b2_weights = pd.concat([lvl1b2_weights, new_rows], axis=0, ignore_index=True)

    # Merge Facility_ID back
    lvl1b2_weights = lvl1b2_weights.merge(
        weight.loc[weight["Facility_Level"].isin(["1b", "2"]), ["District", "Facility_Level", "Facility_ID"]],
        on=["District", "Facility_Level"],
        how="left",
        validate="1:1"
    )

    # Subset Level 1b and 2 facilities and apply weights
    # ---------------------------------------------------------------------
    lvl1b2_ids = list(facilities_by_level.get("1b", [])) + list(facilities_by_level.get("2", []))
    availability_levels1b2 = availability_df[
        availability_df["Facility_ID"].isin(lvl1b2_ids)
    ].copy()

    availability_levels1b2 = availability_levels1b2.merge(
        lvl1b2_weights[["District", "Facility_Level", "Facility_ID", "weight"]],
        on="Facility_ID",
        how="left",
        validate="m:1"
    )

    # Apply weighting
    available_cols = [c for c in availability_levels1b2.columns if c.startswith("available_prop")]
    availability_levels1b2[available_cols] *= availability_levels1b2["weight"].values[:, None]

    # Aggregate to district-month-item level
    availability_levels1b2 = (
        availability_levels1b2
        .groupby(["District", "month", "item_code"], as_index=False)[available_cols]
        .sum()
    )

    # Add facility level
    availability_levels1b2["Facility_Level"] = "1b"

    # Reattach Facility_IDs and weights for level 1b
    full_set_interpolated_levels1b2 = availability_levels1b2.merge(
        lvl1b2_weights.query("Facility_Level == '1b'")[["District", "Facility_Level", "Facility_ID", "weight"]],
        on=["District", "Facility_Level"],
        how="left",
        validate="m:1"
    )

    # Replace old level 1b facilities and recompute weighted availability
    # ---------------------------------------------------------------------
    # Drop old Level 1b facilities
    availability_df = availability_df[
        ~availability_df["Facility_ID"].isin(facilities_by_level.get("1b", []))
    ]

    # Append new 1b facility data
    availability_df = pd.concat(
        [
            availability_df,
            full_set_interpolated_levels1b2[["Facility_ID", "month", "item_code", *available_cols]]
        ],
        axis=0,
        ignore_index=True
    )

    return availability_df

# Function to compute average availability by facility level
def compute_avg_availability_by_var(df: pd.DataFrame = None, # TLO availability dataframe with each row representing one Facility_ID, item, month,
                             mfl: Optional[pd.DataFrame] = None, # Master Facility list mapping Facility_Level to Faciility_ID
                             program_item_mapping: Optional[pd.DataFrame] = None,
                             groupby_var:str = 'month',
                             available_cols: List[str] = ['available_prop'], # List of availability columns to summarise
                             label:str = "Average"):
    if groupby_var == 'Facility_Level':
        # Merge Facility_Level
        df = (df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],on=['Facility_ID'], how='left'))
    if groupby_var == 'item_category':
        # Merge Program
        program_item_mapping = program_item_mapping.rename(columns ={'Item_Code': 'item_code'})[program_item_mapping.item_category.notna()]
        df = df.merge(program_item_mapping, on = ['item_code'], how='left')

    out = (
        df
        .groupby(groupby_var)[available_cols]
        .mean()
        .reset_index()
        .melt(
            id_vars=groupby_var,
            value_vars=available_cols,
            var_name="Scenario",
            value_name="Average_Availability"
        )
    )
    out["Dataset"] = label
    return out

def plot_availability_before_and_after_level1b_fix(old_df: pd.DataFrame = None,
                                                   new_df: pd.DataFrame = None,
                                                   mfl: pd.DataFrame = None,
                                                   available_cols: List[str] = ['available_prop'], # List of availability columns to summarise
                                                   save_figure_as:Path = None):
    avg_old = compute_avg_availability_by_var(df=old_df,
                                              mfl=mfl,
                                              groupby_var='Facility_Level',
                                              available_cols=available_cols,
                                              label="Original")

    avg_new = compute_avg_availability_by_var(df=new_df,
                                              mfl=mfl,
                                              groupby_var='Facility_Level',
                                              available_cols=available_cols,
                                              label="Updated")

    plot_df = pd.concat([avg_old, avg_new], ignore_index=True)
    facility_levels = plot_df["Facility_Level"].unique()
    scenarios = plot_df["Scenario"].unique()

    x = np.arange(len(scenarios))
    width = 0.35

    fig, axes = plt.subplots(
        nrows=len(facility_levels),
        figsize=(14, 5 * len(facility_levels)),
        sharey=True
    )

    if len(facility_levels) == 1:
        axes = [axes]

    for ax, fl in zip(axes, facility_levels):
        sub = plot_df[plot_df["Facility_Level"] == fl]

        orig = sub[sub["Dataset"] == "Original"].set_index("Scenario").loc[scenarios]
        new = sub[sub["Dataset"] == "Updated"].set_index("Scenario").loc[scenarios]

        ax.bar(x - width / 2, orig["Average_Availability"], width, label="Original")
        ax.bar(x + width / 2, new["Average_Availability"], width, label="Updated")

        ax.set_title(f"Average Availability by Scenario – Facility Level {fl}")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.set_ylabel("Average Availability")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_figure_as)

def collapse_stockout_data(_df, groupby_list, var):
    """Return a dataframe with rows for the same TLO model item code collapsed into 1"""
    # Define column lists based on the aggregation function to be applied
    columns_to_multiply = [var]
    columns_to_sum = ['closing_bal', 'amc', 'dispensed', 'received']
    columns_to_preserve = ['data_source', 'consumable_reporting_freq', 'consumables_reported_in_mth']

    # Define aggregation function to be applied to collapse data by item
    def custom_agg_stkout(x):
        if x.name in columns_to_multiply:
            return x.prod(skipna=True) if np.any(
                x.notnull() & (x >= 0)) else np.nan  # this ensures that the NaNs are retained
        elif x.name in columns_to_sum:
            return x.sum(skipna=True) if np.any(
                x.notnull() & (x >= 0)) else np.nan  # this ensures that the NaNs are retained
        # , i.e. not changed to 1, when the corresponding data for both item name variations are NaN, and when there
        # is a 0 or positive value for one or both item name variation, the sum is taken.
        elif x.name in columns_to_preserve:
            return x.iloc[0]  # this function extracts the first value

    # Collapse dataframe
    _collapsed_df = _df.groupby(groupby_list).agg(
        {col: custom_agg_stkout for col in columns_to_multiply + columns_to_sum + columns_to_preserve}
    ).reset_index()

    return _collapsed_df

# Functions for interpolation
def get_other_facilities_of_same_level(_fac_id):
    """Return a set of facility_id for other facilities that are of the same level as that provided."""
    for v in facilities_by_level.values():
        if _fac_id in v:
            return v - {_fac_id}

def interpolate_missing_with_mean(_ser):
    """Return a series in which any values that are null are replaced with the mean of the non-missing."""
    if pd.isnull(_ser).all():
        raise ValueError
    return _ser.fillna(_ser.mean())

# Function to draw calibration plots at different levels of disaggregation (comparing final TLO data to HHFA)
def comparison_plot(level_of_disaggregation, group_by_var, colour):
    comparison_df_agg = comparison_df.groupby([group_by_var],
                                              as_index=False).agg({'available_prop': 'mean',
                                                                   'available_prop_hhfa': 'mean',
                                                                   'Facility_Level': 'first',
                                                                   'consumable_labels': 'first'})
    comparison_df_agg['labels'] = comparison_df_agg[level_of_disaggregation]

    ax = comparison_df_agg.plot.scatter('available_prop', 'available_prop_hhfa', c=colour)
    ax.axline([0, 0], [1, 1])
    for i, label in enumerate(comparison_df_agg['labels']):
        plt.annotate(label,
                     (comparison_df_agg['available_prop'][i] + 0.005,
                      comparison_df_agg['available_prop_hhfa'][i] + 0.005),
                     fontsize=6, rotation=38)
    if level_of_disaggregation != 'aggregate':
        plt.title('Disaggregated by ' + level_of_disaggregation, fontsize=size, weight="bold")
    else:
        plt.title('Aggregate', fontsize=size, weight="bold")
    plt.xlabel('Pr(drug available) as per TLO model')
    plt.ylabel('Pr(drug available) as per HHFA')
    save_name = 'comparison_plots/calibration_to_hhfa_' + level_of_disaggregation + '.png'
    plt.savefig(outputfilepath / save_name)

def comparison_plot_by_level(fac_type):
    cond_fac_type = comparison_df['Facility_Level'] == fac_type
    comparison_df_by_level = comparison_df[cond_fac_type].reset_index()
    plt.scatter(comparison_df_by_level['available_prop'],
                comparison_df_by_level['available_prop_hhfa'])
    plt.axline([0, 0], [1, 1])
    for i, label in enumerate(comparison_df_by_level['consumable_labels']):
        plt.annotate(label, (comparison_df_by_level['available_prop'][i] + 0.005,
                             comparison_df_by_level['available_prop_hhfa'][i] + 0.005),
                     fontsize=6, rotation=27)
    plt.title(fac_type, fontsize=size, weight="bold")
    plt.xlabel('Pr(drug available) as per TLO model')
    plt.ylabel('Pr(drug available) as per HHFA')

# %%
# 1. PREPARE OPENLMIS DATA
########################################################################################################################
# 1A. Import 2018 data
lmis_df = pd.read_csv(path_to_files_in_the_tlo_shared_drive / 'OpenLMIS/2018/ResourceFile_LMIS_2018.csv', low_memory=False)

# 1. BASIC CLEANING ##
# Rename columns
NameChangeList = [('RYear', 'year'),
                  ('RMonth', 'month'),
                  ('Name (Geographic Zones)', 'district'),
                  ('Name (Facility Operators)', 'fac_owner'),
                  ('Name', 'fac_name'),
                  ('fac_name (Programs)', 'program'),
                  ('fac_name (Facility Types)', 'fac_type'),
                  ('Fullproductname', 'item'),
                  ('Closing Bal', 'closing_bal'),
                  ('Dispensed', 'dispensed'),
                  ('AMC', 'amc'),
                  ('Received', 'received'),
                  ('Stockout days', 'stkout_days')]

change_colnames(lmis_df, NameChangeList)

# Remove Private Health facilities from the data
cond_pvt1 = lmis_df['fac_owner'] == 'Private'
cond_pvt2 = lmis_df['fac_type'] == 'Private Hospital'
lmis_df = lmis_df[~cond_pvt1 & ~cond_pvt2]

# Clean facility types to match with types in the TLO model
# See link: https://docs.google.com/spreadsheets/d/1fcp2-smCwbo0xQDh7bRUnMunCKguBzOIZjPFZKlHh5Y/edit#gid=0
cond_level0 = (lmis_df['fac_name'].str.contains('Health Post'))
cond_level1a = (lmis_df['fac_type'] == 'Clinic') | (lmis_df['fac_type'] == 'Health Centre') | \
               (lmis_df['fac_name'].str.contains('Clinic')) | (lmis_df['fac_name'].str.contains('Health Centre')) | \
               (lmis_df['fac_name'].str.contains('Maternity')) | (lmis_df['fac_name'] == 'Chilaweni') | \
               (lmis_df['fac_name'] == 'Chimwawa') | (lmis_df['fac_name'].str.contains('Dispensary'))
cond_level1b = (lmis_df['fac_name'].str.contains('Community Hospital')) | \
               (lmis_df['fac_type'] == 'Rural/Community Hospital') | \
               (lmis_df['fac_name'] == 'St Peters Hospital') | \
               (lmis_df['fac_name'] == 'Police College Hospital') | \
               (lmis_df['fac_type'] == 'CHAM') | (lmis_df['fac_owner'] == 'CHAM')
cond_level2 = (lmis_df['fac_type'] == 'District Health Office') | (lmis_df['fac_name'].str.contains('DHO'))
cond_level3 = (lmis_df['fac_type'] == 'Central Hospital') | \
              (lmis_df['fac_name'].str.contains('Central Hospital'))
cond_level4 = (lmis_df['fac_name'] == 'Zomba Mental Hospital')
lmis_df.loc[cond_level0, 'fac_type_tlo'] = 'Facility_level_0'
lmis_df.loc[cond_level1a, 'fac_type_tlo'] = 'Facility_level_1a'
lmis_df.loc[cond_level1b, 'fac_type_tlo'] = 'Facility_level_1b'
lmis_df.loc[cond_level2, 'fac_type_tlo'] = 'Facility_level_2'
lmis_df.loc[cond_level3, 'fac_type_tlo'] = 'Facility_level_3'
lmis_df.loc[cond_level4, 'fac_type_tlo'] = 'Facility_level_4'

print('Data import complete and ready for analysis')

# Convert values to numeric format
num_cols = ['year',
            'closing_bal',
            'dispensed',
            'stkout_days',
            'amc',
            'received']
lmis_df[num_cols] = lmis_df[num_cols].replace(to_replace=',', value='', regex=True)
lmis_df[num_cols] = lmis_df[num_cols].astype(float)

# Define relevant dictionaries for analysis
months_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
               9: 'September', 10: 'October', 11: 'November', 12: 'December'}
fac_types_dict = {1: 'Facility_level_0', 2: 'Facility_level_1a', 3: 'Facility_level_1b', 4: 'Facility_level_2',
                  5: 'Facility_level_3'}
districts_dict = {1: 'Balaka', 2: 'Blantyre', 3: 'Chikwawa', 4: 'Chiradzulu', 5: 'Chitipa', 6: 'Dedza',
                  7: 'Dowa', 8: 'Karonga', 9: 'Kasungu', 10: 'Lilongwe', 11: 'Machinga', 12: 'Mangochi',
                  13: 'Mchinji', 14: 'Mulanje', 15: 'Mwanza', 16: 'Mzimba North', 17: 'Mzimba South',
                  18: 'Neno', 19: 'Nkhata bay', 20: 'Nkhota Kota', 21: 'Nsanje', 22: 'Ntcheu', 23: 'Ntchisi',
                  24: 'Phalombe', 25: 'Rumphi', 26: 'Salima', 27: 'Thyolo', 28: 'Zomba'}
programs_lmis_dict = {1: 'Essential Meds', 2: 'HIV', 3: 'Malaria', 4: 'Nutrition', 5: 'RH',
                      6: 'TB'}  # programs listed in the OpenLMIS data
months_withdata = ['January', 'February', 'April', 'October', 'November']
months_interpolated = ['March', 'May', 'June', 'July', 'August', 'September', 'December']

# 1B. RESHAPE AND REORDER
# Reshape dataframe so that each row represent a unique consumable and facility
lmis_df_wide = lmis_df.pivot_table(index=['district', 'fac_type_tlo', 'fac_name', 'program', 'item'], columns='month',
                                   values=['closing_bal', 'dispensed', 'received', 'stkout_days', 'amc'],
                                   fill_value=-99)

# Reorder columns in chronological order
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
          'December']
lmis_df_wide = lmis_df_wide.reindex(months, axis=1, level=1)

# Replace all the -99s created in the pivoting process above to NaNs
num = lmis_df_wide._get_numeric_data()
lmis_df_wide[num < 0] = np.nan

# 1C. INTERPOLATE MISSING VALUES ##
# When stkout_days is empty but closing balance, dispensed and received entries are available
lmis_df_wide_flat = lmis_df_wide.reset_index()
count_stkout_entries = lmis_df_wide_flat['stkout_days'].count(axis=1).sum()
print(count_stkout_entries, "stockout entries before first interpolation")

# Define lists of months with the same number of days
months_dict31 = ['January', 'March', 'May', 'July', 'August', 'October', 'December']
months_dict30 = ['April', 'June', 'September', 'November']

for m in range(1, 13):
    # Identify datapoints which come from the original data source (not interpolation)
    lmis_df_wide_flat[('data_source', months_dict[m])] = np.nan  # empty column
    cond1 = lmis_df_wide_flat['stkout_days', months_dict[m]].notna()
    lmis_df_wide_flat.loc[cond1, [('data_source', months_dict[m])]] = 'original_lmis_data'

    # First update received to zero if other columns are avaialble
    cond1 = lmis_df_wide_flat['closing_bal', months_dict[m]].notna() & lmis_df_wide_flat['amc', months_dict[m]].notna()
    cond2 = lmis_df_wide_flat['received', months_dict[m]].notna()
    lmis_df_wide_flat.loc[cond1 & ~cond2, [('received', months_dict[m])]] = 0

# Before interpolation, make corrections for items whose name changed mid-year
inconsistent_item_names_mapping = {
    'Zidovudine/Lamivudine (AZT/3TC), 300+150mg':
        '''Zidovudine (AZT) + Lamivudine (3TC), 300mg+150mg, 60''s (4A, 8A)''',
    'Zidovudine/Lamivudine (AZT/3TC), 60+30mg':
        '''Zidovudine (AZT) + Lamivudine (3TC), 60mg+30mg, 60''s (4P)''',
    'Zidovudine/Lamivudine/Nevirapine (AZT/3TC/NVP), 300+150+200mg':
        '''Zidovudine (AZT) + Lamivudine (3TC) + Nvevirapine (NVP), 300mg + 150mg + 200mg, 60''s (2A)''',
    'Zidovudine/Lamivudine/Nevirapine (AZT/3TC/NVP), 60+30+50mg':
        '''Zidovudine(AZT) + Lamivudine (3TC) + Nevirapine(NVP), 60mg + 30mg + 50mg, 60''s (2P)''',
    'Tenofovir Disoproxil Fumarate/Lamivudine(TDF/3TC ), 300+300mg':
        '''Tenofovir (TDF) + Lamivudine (3TC), 300mg+300mg, 30''s (7A, 6A)''',
    'Tenofovir Disoproxil Fumarate/Lamivudine/Efavirenz(TDF/3TC /EFV), 300+300+600mg':
        '''Tenofovir (TDF) + Lamivudine (3TC) + Efavirenz (EFV), 300+300+600, 30''s (5A)''',
    'Nevirapine (NVP), 50mg': '''Nevirapine 50mg, 60''s''',
    'Nevirapine (NVP), 200mg': '''Nevirapine (NVP), 200mg, 60''s (6A)''',
    'Lopinavir/Ritonavir (LPV/r ), 200+50mg': '''Lopinavir + Ritonavir (LPV/r), 200mg + 50mg, 120''s (7A)''',
    'Lopinavir/Ritonavir(LPV/r ), 100+25mg': '''Lopinavir (LPV/r), 100mg + 25mg, 60''s (9P)''',
    'Gentamicin, 80mg/2ml': '''Gentamicin 40mg/ml, 2ml''',
    '''Nevirapine (NVP syrup with syringe), 10mg/ml''': '''Nevirapine (NVP) Syrup, 10mg/ml''',
    'Efavirenz (EFV), 600mg': '''Efavirenz (EFV), 600mg, 30''s (3A)''',
    'Efavirenz (EFV), 200mg': '''Efavirenz (EFV), 200mg, 90''s (3P)''',
    'Unigold HIV test kits, Kit of 20 Tests': '''Unigold HIV Test Kits''',
    'Determine HIV test Kits, Kit of 100 Tests': '''Determine HIV Test Kits''',
    'Abacavir/Lamivudine (ABC/3TC), 60+30mg': '''Abacavir (ABC) + Lamivudine(3TC), 60mg+30mg, 60''S (9P)''',
    'Atazanavir /Ritonavir (ATV/r), 300+100mg': '''Atazanavir +  Ritonavir, 300mg + 100mg, 30''S (7A)''',
    'SD Bioline, Syphilis test kits, Kit of 30 Tests': 'Determine Syphillis Test Kits',
    'Isoniazid tablets, 100mg': '''Isoniazid 100mg''',
    'Isoniazid tablets, 300mg': '''Isoniazid 300mg''',
    'Morphine slow rel, 30mg': 'Morphine sulphate 10mg (slow release)',
    'Male condoms, Each': 'Male Condoms',
    'Benzathine penicillin, 2.4M': 'Benzathine benzylpenicillin 1.44g (2.4MU), PFR',
    'Doxycycline, 100mg': 'Doxycycline 100mg',
    'Ciprofloxicin, 500mg': 'Ciprofloxacin 500mg',
    'Metronidazole, 200mg': 'Metronidazole 200mg',
    'Cotrimoxazole (dispersible tabs), 100+20mg': 'Cotrimoxazole 120mg Tablets',
    'Cotrimoxazole, 400+ 80mg': 'Cotrimoxazole 480mg tablets',
    'Cotrimoxazole, 960 mg': 'Cotrimoxazole 960mg Tabs',
    'Erythromycin, 250mg': 'Erythromycin 250mg',
    'Clotrimazole 500mg vaginal tablet (blister 10 x 10, with app)': 'Clotrimazole 500mg vaginal (Tablets/Pessaries)',
}

items_introduced_in_september = {
    'Tenofovir Disoproxil Fumarate/Lamivudine/Dolutegravir (TDF/3TC /DTG), 300+300+50mg': '',
    'Abacavir/Lamivudine (ABC/3TC), 600+300mg': '',
    'DBS Bundles, 70 microlitre, Pack of 50 Tests': '',
    'Dolutegravir (DTG), 50mg': '',
    'Lopinavir/Ritonavir(LPV/r ), 40+10mg': '',
    'OraQuick HIV Self Test, Pouch': '',
}


# TODO check whether there is any issue with the above items_introduced_in_september which only show up from September
#  onwards

# Hold out the dataframe with no naming inconsistencies
list_of_items_with_inconsistent_names_zipped = set(zip(inconsistent_item_names_mapping.keys(), inconsistent_item_names_mapping.values()))
list_of_items_with_inconsistent_names = [item for sublist in list_of_items_with_inconsistent_names_zipped for item in sublist]
df_with_consistent_item_names =  lmis_df_wide_flat[~lmis_df_wide_flat[('item',)].isin(list_of_items_with_inconsistent_names)]
df_without_consistent_item_names = lmis_df_wide_flat[lmis_df_wide_flat[('item',)].isin(list_of_items_with_inconsistent_names)]
# Make inconsistently named drugs uniform across the dataframe
df_without_consistent_item_names_corrected = rename_items_to_address_inconsistentencies(
    df_without_consistent_item_names, inconsistent_item_names_mapping)
# Append holdout and corrected dataframes
lmis_df_wide_flat = pd.concat([df_without_consistent_item_names_corrected, df_with_consistent_item_names],
                              ignore_index=True)

# 1. --- RULE: 1.If i) stockout is missing, ii) closing_bal, amc and received are not missing , and iii) amc !=0 and,
#          then stkout_days[m] = (amc[m] - closing_bal[m-1] - received)/amc * number of days in the month ---
# (Note that the number of entries for closing balance, dispensed and received is always the same)
for m in range(2, 13):
    # Now update stkout_days if other columns are available
    cond1 = lmis_df_wide_flat['closing_bal', months_dict[m - 1]].notna() & lmis_df_wide_flat[
        'amc', months_dict[m]].notna() & lmis_df_wide_flat['received', months_dict[m]].notna()
    cond2 = lmis_df_wide_flat['stkout_days', months_dict[m]].notna()
    cond3 = lmis_df_wide_flat['amc', months_dict[m]] != 0
    lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('data_source', months_dict[m])]] = 'lmis_interpolation_rule1'

    if months_dict[m] in months_dict31:
        lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('stkout_days', months_dict[m])]] = \
            (
                lmis_df_wide_flat[('amc', months_dict[m])]
                - lmis_df_wide_flat[('closing_bal', months_dict[m - 1])]
                - lmis_df_wide_flat[('received', months_dict[m])]
            ) / lmis_df_wide_flat[('amc', months_dict[m])] * 31
    elif months_dict[m] in months_dict30:
        lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('stkout_days', months_dict[m])]] = \
            (
                lmis_df_wide_flat[('amc', months_dict[m])]
                - lmis_df_wide_flat[('closing_bal', months_dict[m - 1])]
                - lmis_df_wide_flat[('received', months_dict[m])]
            ) / lmis_df_wide_flat[('amc', months_dict[m])] * 30
    else:
        lmis_df_wide_flat.loc[cond1 & ~cond2 & cond3, [('stkout_days', months_dict[m])]] = \
            (
                lmis_df_wide_flat[('amc', months_dict[m])]
                - lmis_df_wide_flat[('closing_bal', months_dict[m - 1])]
                - lmis_df_wide_flat[('received', months_dict[m])]
            ) / lmis_df_wide_flat[('amc', months_dict[m])] * 28

count_stkout_entries = lmis_df_wide_flat['stkout_days'].count(axis=1).sum()
print(count_stkout_entries, "stockout entries after first interpolation")

# 2. --- If any stockout_days < 0 after the above interpolation, update to stockout_days = 0 ---
# RULE: If closing balance[previous month] - dispensed[this month] + received[this month] > 0, stockout == 0
for m in range(1, 13):
    cond1 = lmis_df_wide_flat['stkout_days', months_dict[m]] < 0
    # print("Negative stockout days ", lmis_df_wide_flat.loc[cond1,[('stkout_days', months_dict[m])]].count(axis = 1
    # ).sum())
    lmis_df_wide_flat.loc[cond1, [('data_source', months_dict[m])]] = 'lmis_interpolation_rule2'
    lmis_df_wide_flat.loc[cond1, [('stkout_days', months_dict[m])]] = 0

count_stkout_entries = lmis_df_wide_flat['stkout_days'].count(axis=1).sum()
print(count_stkout_entries, "stockout entries after second interpolation")

lmis_df_wide_flat['consumable_reporting_freq'] = lmis_df_wide_flat['closing_bal'].count(axis=1)  # generate a column
# which reports the number of entries of closing balance for a specific consumable by a facility during the year

# Flatten multilevel columns
lmis_df_wide_flat.columns = [' '.join(col).strip() for col in lmis_df_wide_flat.columns.values]

# 3. --- If the consumable was previously reported and during a given month, if any consumable was reported, assume
# 100% days of stckout ---
# RULE: If the balance on a consumable is ever reported and if any consumables are reported during the month, stkout_
# days = number of days of the month
for m in range(1, 13):
    month = 'closing_bal ' + months_dict[m]
    var_name = 'consumables_reported_in_mth ' + months_dict[m]
    lmis_df_wide_flat[var_name] = lmis_df_wide_flat.groupby("fac_name")[month].transform('count')  # generate a column
# which reports the number of consumables for which closing balance was recorded by a facility in a given month

for m in range(1, 13):
    cond1 = lmis_df_wide_flat['consumable_reporting_freq'] > 0
    cond2 = lmis_df_wide_flat['consumables_reported_in_mth ' + months_dict[m]] > 0
    cond3 = lmis_df_wide_flat['stkout_days ' + months_dict[m]].notna()
    lmis_df_wide_flat.loc[cond1 & cond2 & ~cond3, [('data_source ' + months_dict[m])]] = 'lmis_interpolation_rule3'

    if months_dict[m] in months_dict31:
        lmis_df_wide_flat.loc[cond1 & cond2 & ~cond3, [('stkout_days ' + months_dict[m])]] = 31
    elif months_dict[m] in months_dict30:
        lmis_df_wide_flat.loc[cond1 & cond2 & ~cond3, [('stkout_days ' + months_dict[m])]] = 30
    else:
        lmis_df_wide_flat.loc[cond1 & cond2 & ~cond3, [('stkout_days ' + months_dict[m])]] = 28

count_stkout_entries = 0
for m in range(1, 13):
    count_stkout_entries = count_stkout_entries + lmis_df_wide_flat['stkout_days ' + months_dict[m]].count().sum()
print(count_stkout_entries, "stockout entries after third interpolation")

# 1D. CALCULATE STOCK OUT RATES BY MONTH and FACILITY
lmis = lmis_df_wide_flat  # choose dataframe

# Generate variables denoting the stockout proportion in each month
for m in range(1, 13):
    if months_dict[m] in months_dict31:
        lmis['stkout_prop ' + months_dict[m]] = lmis['stkout_days ' + months_dict[m]] / 31
    elif months_dict[m] in months_dict30:
        lmis['stkout_prop ' + months_dict[m]] = lmis['stkout_days ' + months_dict[m]] / 30
    else:
        lmis['stkout_prop ' + months_dict[m]] = lmis['stkout_days ' + months_dict[m]] / 28

# Reshape data
lmis = pd.wide_to_long(lmis, stubnames=['closing_bal', 'received', 'amc', 'dispensed', 'stkout_days', 'stkout_prop',
                                        'data_source', 'consumables_reported_in_mth'],
                       i=['district', 'fac_type_tlo', 'fac_name', 'program', 'item'], j='month',
                       sep=' ', suffix=r'\w+')
lmis = lmis.reset_index()

# 2. LOAD CLEANED MATCHED CONSUMABLE LIST FROM TLO MODEL AND MERGE WITH LMIS DATA
########################################################################################################################
# 1. --- Load and clean data ---
# Import matched list of consumanbles
consumables_df = pd.read_csv(path_for_new_resourcefiles / 'ResourceFile_consumables_matched.csv', low_memory=False,
                             encoding="ISO-8859-1")
cond = consumables_df['matching_status'] == 'Remove'
consumables_df = consumables_df[~cond]  # Remove items which were removed due to updates or the existence of duplicates

# Keep only the correctly matched consumables for stockout analysis based on OpenLMIS
cond1 = consumables_df['matching_status'] == 'Matched'
cond2 = consumables_df['verified_by_DM_lead'] != 'Incorrect'
matched_consumables = consumables_df[cond1 & cond2]

# Rename columns
NameChangeList = [('consumable_name_lmis', 'item'), ]
change_colnames(consumables_df, NameChangeList)
change_colnames(matched_consumables, NameChangeList)


# Update matched consumable name where the name in the OpenLMIS data was updated in September
matched_consumables = replace_old_item_names_in_lmis_data(matched_consumables, inconsistent_item_names_mapping)

# 2. --- Merge data with LMIS data ---
lmis_matched_df = pd.merge(lmis, matched_consumables, how='inner', on=['item'])
lmis_matched_df = lmis_matched_df.sort_values('data_source')

# 2.i. For substitable drugs (within drug category), collapse by taking the product of stkout_prop (OR condition)
# This represents Pr(all substitutes with the item code are stocked out)
groupby_list1 = ['module_name', 'district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'consumable_name_tlo',
                 'match_level1',
                 'match_level2']
stkout_df = collapse_stockout_data(lmis_matched_df, groupby_list1, 'stkout_prop')

# 2.ii. For complementary drugs, collapse by taking the product of (1-stkout_prob)
# This represents Pr(All drugs within item code (in different match_group's) are available)
stkout_df['available_prop'] = 1 - stkout_df['stkout_prop']
groupby_list2 = ['module_name', 'district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'consumable_name_tlo',
                 'match_level2']
stkout_df = collapse_stockout_data(stkout_df, groupby_list2, 'available_prop')

# 2.iii. For substitutable drugs (within consumable_name_tlo), collapse by taking the product of stkout_prop (OR
# condition).
# This represents Pr(all substitutes with the item code are stocked out)
stkout_df['stkout_prop'] = 1 - stkout_df['available_prop']
groupby_list3 = ['module_name', 'district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'consumable_name_tlo']
stkout_df = collapse_stockout_data(stkout_df, groupby_list3, 'stkout_prop')

# Update impossible stockout values (This happens due to some stockout days figures being higher than the number of
#  days in the month)
stkout_df.loc[stkout_df['stkout_prop'] < 0, 'stkout_prop'] = 0
stkout_df.loc[stkout_df['stkout_prop'] > 1, 'stkout_prop'] = 1
# Eliminate duplicates
collapse_dict = {
    'stkout_prop': 'mean', 'closing_bal': 'mean', 'amc': 'mean', 'dispensed': 'mean', 'received': 'mean',
    'consumable_reporting_freq': 'mean', 'consumables_reported_in_mth': 'mean',
    'module_name': 'first', 'consumable_name_tlo': 'first', 'data_source': 'first'
}
stkout_df = stkout_df.groupby(['fac_type_tlo', 'fac_name', 'district', 'month', 'item_code'], as_index=False).agg(
    collapse_dict).reset_index()

stkout_df['available_prop'] = 1 - stkout_df['stkout_prop']

# Some missing values change to 100% stockouts during the aggregation above. Fix this manually
for var in ['stkout_prop', 'available_prop', 'closing_bal', 'amc', 'dispensed', 'received']:
    cond = stkout_df['data_source'].isna()
    stkout_df.loc[cond, var] = np.nan

stkout_df = stkout_df.reset_index()
stkout_df = stkout_df[
    ['module_name', 'district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'consumable_name_tlo',
     'available_prop', 'closing_bal', 'amc', 'dispensed', 'received',
     'data_source', 'consumable_reporting_freq', 'consumables_reported_in_mth']]

# 3. ADD STOCKOUT DATA FROM OTHER SOURCES TO COMPLETE STOCKOUT DATAFRAME
########################################################################################################################
# --- 1. Generate a dataframe of stock availability for consumables which were not found in the OpenLMIS data but
# available in the HHFA 2018/19 --- #
# Save the list of items for which a match was not found in the OpenLMIS data
unmatched_consumables = consumables_df.drop_duplicates(['item_code'])
unmatched_consumables = pd.merge(unmatched_consumables, matched_consumables[['item', 'item_code']], how='left',
                                 on='item_code')
unmatched_consumables = unmatched_consumables[unmatched_consumables['item_y'].isna()]

# ** Extract stock availability data from HHFA and clean data **
hhfa_df = pd.read_excel(path_to_files_in_the_tlo_shared_drive / 'ResourceFile_hhfa_consumables.xlsx', sheet_name='hhfa_data')

# Use the ratio of availability rates between levels 1b on one hand and levels 2 and 3 on the other to extrapolate
# availability rates for levels 2 and 3 from the HHFA data
cond1b = stkout_df['fac_type_tlo'] == 'Facility_level_1b'
cond2 = stkout_df['fac_type_tlo'] == 'Facility_level_2'
cond3 = stkout_df['fac_type_tlo'] == 'Facility_level_3'
availratio_2to1b = stkout_df[cond2]['available_prop'].mean() / stkout_df[cond1b]['available_prop'].mean()
availratio_3to1b = stkout_df[cond3]['available_prop'].mean() / stkout_df[cond1b]['available_prop'].mean()

# To disaggregate the avalability rates for levels 1b, 2, and 3 from the HHFA, assume that the ratio of availability
# across the three
# levels is the same as that based on OpenLMIS
# There are a total of 101 "hospitals" surveyed for commodities in the HHFA (we assume that of these 28 are district
# hospitals and 4 are central hospitals -> 69 1b health facilities). The availability rates for the three levels can be
# dereived as follows
scaleparam = 101 / (69 + 28 * availratio_2to1b + 4 * availratio_3to1b)
hhfa_df['available_prop_hhfa_Facility_level_1b'] = scaleparam * hhfa_df['available_prop_hhfa_Facility_level_1b']
hhfa_df['available_prop_hhfa_Facility_level_2'] = availratio_2to1b * hhfa_df['available_prop_hhfa_Facility_level_1b']
hhfa_df['available_prop_hhfa_Facility_level_3'] = availratio_3to1b * hhfa_df['available_prop_hhfa_Facility_level_1b']

for var in ['available_prop_hhfa_Facility_level_2', 'available_prop_hhfa_Facility_level_3']:
    cond = hhfa_df[var] > 1
    hhfa_df.loc[cond, var] = 1

# Add further assumptions on consumable availability from other sources
assumptions_df = pd.read_excel(open(path_to_files_in_the_tlo_shared_drive / 'ResourceFile_hhfa_consumables.xlsx', 'rb'),
                               sheet_name='availability_assumptions')
assumptions_df = assumptions_df[['item_code', 'available_prop_Facility_level_0',
                                 'available_prop_Facility_level_1a', 'available_prop_Facility_level_1b',
                                 'available_prop_Facility_level_2', 'available_prop_Facility_level_3']]

# Merge HHFA data with the list of unmatched consumables from the TLO model
unmatched_consumables_df = pd.merge(unmatched_consumables, hhfa_df, how='left', on='item_code')
unmatched_consumables_df = pd.merge(unmatched_consumables_df, assumptions_df, how='left', on='item_code')
# when not missing, replace with assumption

for level in ['0', '1a', '1b', '2', '3']:
    cond = unmatched_consumables_df['available_prop_hhfa_Facility_level_' + level].notna()
    unmatched_consumables_df.loc[cond, 'data_source'] = 'hhfa_2018-19'

    cond = unmatched_consumables_df['available_prop_Facility_level_' + level].notna()
    unmatched_consumables_df.loc[cond, 'data_source'] = 'other'
    unmatched_consumables_df.loc[cond, 'available_prop_hhfa_Facility_level_' + level] = unmatched_consumables_df[
        'available_prop_Facility_level_' + level]

unmatched_consumables_df = unmatched_consumables_df[
    ['module_name', 'item_code', 'consumable_name_tlo_x', 'available_prop_hhfa_Facility_level_0',
     'available_prop_hhfa_Facility_level_1a', 'available_prop_hhfa_Facility_level_1b',
     'available_prop_hhfa_Facility_level_2', 'available_prop_hhfa_Facility_level_3',
     'fac_count_Facility_level_0', 'fac_count_Facility_level_1a', 'fac_count_Facility_level_1b',
     'data_source']]

# Reshape dataframe of consumable availability taken from the HHFA in the same format as the stockout dataframe based
# on OpenLMIS
unmatched_consumables_df = pd.wide_to_long(unmatched_consumables_df, stubnames=['available_prop_hhfa', 'fac_count'],
                                           i=['item_code', 'consumable_name_tlo_x'], j='fac_type_tlo',
                                           sep='_', suffix=r'\w+')

unmatched_consumables_df = unmatched_consumables_df.reset_index()
n = len(unmatched_consumables_df)

# Final cleaning
NameChangeList = [('consumable_name_tlo_x', 'consumable_name_tlo'),
                  ('available_prop_hhfa', 'available_prop')]
change_colnames(unmatched_consumables_df, NameChangeList)

# --- 2. Append OpenLMIS stockout dataframe with HHFA stockout dataframe and Extract in .csv format --- #
# Append common consumables stockout dataframe with the main dataframe
cond = unmatched_consumables_df['available_prop'].notna()
unmatched_consumables_df.loc[~cond, 'data_source'] = 'Not available'
stkout_df = pd.concat([stkout_df, unmatched_consumables_df], axis=0, ignore_index=True)

# --- 3. Append stockout rate for facility level 0 from HHFA --- #
cond = hhfa_df['item_code'].notna()
hhfa_fac0 = hhfa_df[cond][
    ['item_code', 'consumable_name_tlo', 'fac_count_Facility_level_0', 'available_prop_hhfa_Facility_level_0']]
NameChangeList = [('fac_count_Facility_level_0', 'fac_count'),
                  ('available_prop_hhfa_Facility_level_0', 'available_prop')]
change_colnames(hhfa_fac0, NameChangeList)
hhfa_fac0['fac_type_tlo'] = 'Facility_level_0'
hhfa_fac0['data_source'] = 'hhfa_2018-19'

hhfa_fac0 = pd.merge(hhfa_fac0, consumables_df[['item_code', 'module_name']], on='item_code', how='inner')
hhfa_fac0 = hhfa_fac0.drop_duplicates()

cond = stkout_df['fac_type_tlo'] == 'Facility_level_0'
stkout_df = stkout_df[~cond]
stkout_df = pd.concat([stkout_df, hhfa_fac0], axis=0, ignore_index=True)

# --- 4. Generate new category variable for analysis --- #
stkout_df = recategorize_modules_into_consumable_categories(stkout_df)
item_code_category_mapping = stkout_df[['item_category', 'item_code']].drop_duplicates()

# Add item_category to ResourceFile_Consumables_Item_Designations
item_designations = pd.read_csv(path_for_new_resourcefiles  / 'ResourceFile_Consumables_Item_Designations.csv')
item_designations = item_designations.drop(columns = 'item_category')
item_designations = item_designations.merge(item_code_category_mapping, left_on = 'Item_Code', right_on = 'item_code', how = 'left', validate = '1:1')
item_designations.drop(columns = 'item_code').to_csv(path_for_new_resourcefiles  / 'ResourceFile_Consumables_Item_Designations.csv', index = False)

# --- 5. Replace district/fac_name/month entries where missing --- #
for var in ['district', 'fac_name', 'month']:
    cond = stkout_df[var].isna()
    stkout_df.loc[cond, var] = 'Aggregate'

# --- 6. Export final stockout dataframe --- #
# stkout_df.to_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")
# <-- this line commented out as the file is very large.

# Create a smaller file with the ratio of inflow to outflow of consumables for the purpose of costing, i.e. to cost
# the stock lost to waste/theft/expiryetc.
# Estimate the stock to dispensed ratio from OpenLMIS data
lmis_consumable_usage = stkout_df.copy()
# TODO Generate a smaller version of this file
# Collapse individual facilities
inflow_to_outflow_ratio = get_inflow_to_outflow_ratio_by_item_and_facilitylevel(lmis_consumable_usage)
# Clean values for analysis
inflow_to_outflow_ratio.loc[inflow_to_outflow_ratio < 1] = 1 # Ratio can't be less than 1
inflow_to_outflow_ratio.loc[inflow_to_outflow_ratio > inflow_to_outflow_ratio.quantile(0.95)] = inflow_to_outflow_ratio.quantile(0.95) # Trim values greater than the 95th percentile

inflow_to_outflow_ratio = inflow_to_outflow_ratio.reset_index().rename(columns={0:'inflow_to_outflow_ratio'})
inflow_to_outflow_ratio.to_csv(resourcefilepath / 'costing/ResourceFile_Consumables_Inflow_Outflow_Ratio.csv', index = False)

# Final checks
stkout_df = stkout_df.drop(index=stkout_df.index[pd.isnull(stkout_df.available_prop)])
assert (stkout_df.available_prop >= 0.0).all(), "No Negative values"
assert (stkout_df.available_prop <= 1.0).all(), "No Values greater than 1.0 "
print(stkout_df.loc[(~(stkout_df.available_prop >= 0.0)) | (~(stkout_df.available_prop <= 1.0))].available_prop)
assert not stkout_df.duplicated(['fac_type_tlo', 'fac_name', 'district', 'month', 'item_code']).any(), "No duplicates"

# --- 6.7 Generate file for use in model run --- #
# 1) Smaller file size
# 2) Indexed by the 'Facility_ID' used in the model (which is an amalgmation of district and facility_level, defined in
#  the Master Facilities List.

# unify the set within each facility_id
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}

sf = stkout_df[['item_code', 'month', 'district', 'fac_type_tlo', 'available_prop']].dropna()
sf.loc[sf.month == 'Aggregate', 'month'] = 'January'  # Assign arbitrary month to data only available at aggregate level
sf.loc[sf.district == 'Aggregate', 'district'] = 'Balaka' \
    # Assign arbitrary district to data only available at # aggregate level
sf = sf.drop(index=sf.index[(sf.month == 'NA') | (sf.district == 'NA')])
sf.month = sf.month.map(dict(zip(calendar.month_name[1:13], range(1, 13))))
sf.item_code = sf.item_code.astype(int)
sf['fac_type_tlo'] = sf['fac_type_tlo'].str.replace("Facility_level_", "")

# Do some mapping to make the Districts line-up with the definition of Districts in the model
rename_and_collapse_to_model_districts = {
    'Nkhota Kota': 'Nkhotakota',
    'Mzimba South': 'Mzimba',
    'Mzimba North': 'Mzimba',
    'Nkhata bay': 'Nkhata Bay',
}

sf['district_std'] = sf['district'].replace(rename_and_collapse_to_model_districts)
# Take averages (now that 'Mzimba' is mapped-to by both 'Mzimba South' and 'Mzimba North'.)
sf = sf.groupby(by=['district_std', 'fac_type_tlo', 'month', 'item_code'])['available_prop'].mean().reset_index()

# 4. INTERPOLATE MISSING DATA TO ENSURE DATA IS AVAILABLE FOR ALL ITEMS, MONTHS, LEVELS, DISTRICTS
########################################################################################################################
# 1) Cities to get same results as their respective regions
copy_source_to_destination = {
    'Mzimba': 'Mzuzu City',
    'Lilongwe': 'Lilongwe City',
    'Zomba': 'Zomba City',
    'Blantyre': 'Blantyre City'
}

for source, destination in copy_source_to_destination.items():
    new_rows = sf.loc[sf.district_std == source].copy()
    new_rows.district_std = destination
    sf = pd.concat([sf, new_rows], axis=0, ignore_index=True)

# 2) Fill in Likoma (for which no data) with the means
means = sf.loc[sf.fac_type_tlo.isin(['1a', '1b', '2'])].groupby(by=['fac_type_tlo', 'month', 'item_code'])[
    'available_prop'].mean().reset_index()
new_rows = means.copy()
new_rows['district_std'] = 'Likoma'
sf = pd.concat([sf, new_rows], axis=0, ignore_index=True)

assert sorted(set(districts)) == sorted(set(pd.unique(sf.district_std)))

# 3) copy the results for 'Mwanza/1b' to be equal to 'Mwanza/1a'.
mwanza_1a = sf.loc[(sf.district_std == 'Mwanza') & (sf.fac_type_tlo == '1a')]
mwanza_1b = sf.loc[(sf.district_std == 'Mwanza') & (sf.fac_type_tlo == '1a')].copy().assign(fac_type_tlo='1b')
sf = pd.concat([sf, mwanza_1b], axis=0, ignore_index=True)

# 4) Update the availability Xpert (item_code = 187)
# First add rows for Xpert at level 1b by cloning rows for level 2 -> only if not already present
xpert_item = sf['item_code'].eq(187)
level_2    = sf['fac_type_tlo'].eq('2')
level_1b   = sf['fac_type_tlo'].eq('1b')

# Clone rows from level 2
base   = sf.loc[level_2 & xpert_item].copy()
new_rows  = base.copy()
new_rows['fac_type_tlo'] = '1b'

# Add rows to main availability dataframe and drop duplicates, if any
sf = pd.concat([sf, new_rows], ignore_index=True)
id_cols = [c for c in sf.columns if c != 'available_prop']
dupe_mask = sf.duplicated(subset=id_cols, keep=False)
dupes = sf.loc[dupe_mask].sort_values(id_cols)
sf = sf.drop_duplicates(subset=id_cols, keep='first').reset_index(drop=True)

# Compute the average availability Sep–Dec (months >= 9) for level 2, item 187
sep_to_dec = sf['month'].ge(9)
new_xpert_availability = sf.loc[level_2 & xpert_item & sep_to_dec, 'available_prop'].mean()
# Assign new availability to relevant facility levels
levels_1b_2_or_3 = sf['fac_type_tlo'].isin(['1b', '2', '3'])
xpert_item = sf['item_code'].eq(187)
sf.loc[levels_1b_2_or_3 & xpert_item, 'available_prop'] = new_xpert_availability

# 5) Copy all the results to create a level 0 with an availability equal to half that in the respective 1a
all_1a = sf.loc[sf.fac_type_tlo == '1a']
all_0 = sf.loc[sf.fac_type_tlo == '1a'].copy().assign(fac_type_tlo='0')
all_0.available_prop *= 0.5
sf = pd.concat([sf, all_0], axis=0, ignore_index=True)

# Now, merge-in facility_id
sf_merge = sf.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['district_std', 'fac_type_tlo'],
                    right_on=['District', 'Facility_Level'], how='left', indicator=True)

# 5) Assign the Facility_IDs for those facilities that are regional/national level;
# For facilities of level 3, find which region they correspond to:
districts_with_regional_level_fac = pd.unique(sf_merge.loc[sf_merge.fac_type_tlo == '3'].district_std)
district_to_region_mapping = dict()
for _district in districts_with_regional_level_fac:
    _region = mfl.loc[mfl.District == _district].Region.values[0]
    _fac_id = mfl.loc[(mfl.Facility_Level == '3') & (mfl.Region == _region)].Facility_ID.values[0]
    sf_merge.loc[(sf_merge.fac_type_tlo == '3') & (sf_merge.district_std == _district), 'Facility_ID'] = _fac_id

# National Level
fac_id_of_fac_level4 = mfl.loc[mfl.Facility_Level == '4'].Facility_ID.values[0]
sf_merge.loc[sf_merge.fac_type_tlo == '4', 'Facility_ID'] = fac_id_of_fac_level4

# Now, take averages because more than one set of records is forming the estimates for the level 3 facilities
sf_final = sf_merge.groupby(by=['Facility_ID', 'month', 'item_code'])['available_prop'].mean().reset_index()
sf_final.Facility_ID = sf_final.Facility_ID.astype(int)

# %%
# Construct dataset that conforms to the principles expected by the simulation: i.e. that there is an entry for every
# facility_id and for every month for every item_code.

# Generate the dataframe that has the desired size and shape
fac_ids = set(mfl.loc[mfl.Facility_Level != '5'].Facility_ID)
item_codes = set(sf.item_code.unique())
months = range(1, 13)

full_set = pd.Series(
    index=pd.MultiIndex.from_product([fac_ids, months, item_codes], names=['Facility_ID', 'month', 'item_code']),
    data=np.nan,
    name='available_prop')

# Insert the data, where it is available.
full_set = full_set.combine_first(sf_final.set_index(['Facility_ID', 'month', 'item_code'])['available_prop'])

# Fill in the blanks with rules for interpolation.
facilities_by_level = defaultdict(set)
for ix, row in mfl.iterrows():
    facilities_by_level[row['Facility_Level']].add(row['Facility_ID'])

# Create new dataset that include the interpolations (The operation is not done "in place", because the logic is based
# on what results are missing before the interpolations in other facilities).
full_set_interpolated = full_set * np.nan

for fac in fac_ids:
    for item in item_codes:

        print(f"Now doing: fac={fac}, item={item}")

        # Get records of the availability of this item in this facility.
        _monthly_records = full_set.loc[(fac, slice(None), item)].copy()

        if pd.notnull(_monthly_records).any():
            # If there is at least one record of this item at this facility, then interpolate the missing months from
            # the months for there are data on this item in this facility. (If none are missing, this has no effect).
            _monthly_records = interpolate_missing_with_mean(_monthly_records)

        else:
            # If there is no record of this item at this facility, check to see if it's available at other facilities
            # of the same level
            facilities = list(get_other_facilities_of_same_level(fac))
            recorded_at_other_facilities_of_same_level = pd.notnull(
                full_set.loc[(facilities, slice(None), item)]
            ).any()

            if recorded_at_other_facilities_of_same_level:
                # If it recorded at other facilities of same level, find the average availability of the item at other
                # facilities of the same level.
                facilities = list(get_other_facilities_of_same_level(fac))
                _monthly_records = interpolate_missing_with_mean(
                    full_set.loc[(facilities, slice(None), item)].groupby(level=1).mean()
                )

            else:
                # If it is not recorded at other facilities of same level, then assume it is never available at the
                # facility.
                _monthly_records = _monthly_records.fillna(0.0)

        # Insert values (including corrections) into the resulting dataset.
        full_set_interpolated.loc[(fac, slice(None), item)] = _monthly_records.values

# Check that there are not missing values
assert not pd.isnull(full_set_interpolated).any().any()

full_set_interpolated = full_set_interpolated.reset_index()
#full_set_interpolated = full_set_interpolated.reset_index().merge(item_code_category_mapping, on = 'item_code', how = 'left', validate = 'm:1')

# 5. ADD ALTERNATIVE AVAILABILITY SCENARIOS
########################################################################################################################
# Add alternative availability scenarios to represent improved or reduce consumable availability
full_set_interpolated_with_scenarios = generate_alternative_availability_scenarios(full_set_interpolated)

full_set_interpolated_with_scenarios_level1b_fixed = update_level1b_availability(
    availability_df=full_set_interpolated_with_scenarios,
    facilities_by_level=facilities_by_level,
    resourcefilepath=resourcefilepath,
    district_to_city_dict=copy_source_to_destination,
    weighting = 'district_1b_to_2_ratio',
)

# Compare availability averages by Facility_Level before and after the 1b fix
available_cols = [c for c in full_set_interpolated_with_scenarios.columns if c.startswith("available_prop")]
level1b_fix_plots_path = outputfilepath / 'comparison_plots'
figurespath_scenarios = outputfilepath / 'consumable_scenarios'
if not os.path.exists(level1b_fix_plots_path):
    os.makedirs(level1b_fix_plots_path)
plot_availability_before_and_after_level1b_fix(old_df = full_set_interpolated_with_scenarios,
                                               new_df = full_set_interpolated_with_scenarios_level1b_fixed,
                                               mfl = mfl,
                                               available_cols = available_cols, # List of availability columns to summarise
                                               save_figure_as = level1b_fix_plots_path / 'availability_before_and_after_level1b_fix.png')

# 6. CHECK FORMAT AND SAVE AS RESOURCEFILE
########################################################################################################################
# --- Check that the exported file has the properties required of it by the model code. --- #
check_format_of_consumables_file(df=full_set_interpolated_with_scenarios_level1b_fixed, fac_ids=fac_ids)

# %%
# Save
full_set_interpolated_with_scenarios_level1b_fixed.to_csv(
    path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv",
    index=False
)

# %%
# 7. COMPARISON WITH HHFA DATA, 2018/19
########################################################################################################################
# --- 7.1 Prepare comparison dataframe --- ##
# Note that this only plot consumables for which data is available in the HHFA
# i. Prepare data from HHFA
hhfa_comparison_df = hhfa_df[['item_code', 'consumable_name_tlo', 'item_hhfa', 'available_prop_hhfa_Facility_level_0',
                              'available_prop_hhfa_Facility_level_1a', 'available_prop_hhfa_Facility_level_1b',
                              'available_prop_hhfa_Facility_level_2', 'available_prop_hhfa_Facility_level_3']]
hhfa_comparison_df = pd.wide_to_long(hhfa_comparison_df.dropna(), stubnames='available_prop_hhfa',
                                     i=['consumable_name_tlo', 'item_code', 'item_hhfa'], j='fac_type_tlo',
                                     sep='_', suffix=r'\w+')
hhfa_comparison_df = hhfa_comparison_df.reset_index()
hhfa_comparison_df['fac_type_tlo'] = hhfa_comparison_df['fac_type_tlo'].str.replace("Facility_level_", "")
hhfa_comparison_df = hhfa_comparison_df.rename({'fac_type_tlo': 'Facility_Level'}, axis=1)

# ii. Collapse final model availability data by facility level
final_availability_df = full_set_interpolated_with_scenarios_level1b_fixed
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
final_availability_df = pd.merge(final_availability_df, mfl[['District', 'Facility_Level', 'Facility_ID']], how="left",
                                 on=['Facility_ID'],
                                 indicator=False)
final_availability_df = final_availability_df.groupby(['Facility_Level', 'item_code']).agg(
    {'available_prop': "mean"}).reset_index()

# iii. Merge HHFA with stkout_df
hhfa_comparison_df['item_code'] = hhfa_comparison_df['item_code'].astype(int)
final_availability_df['item_code'] = final_availability_df['item_code'].astype(int)
comparison_df = pd.merge(final_availability_df, hhfa_comparison_df, how='inner', on=['item_code', 'Facility_Level'])
comparison_df['difference'] = (comparison_df['available_prop_hhfa'] - comparison_df['available_prop'])

# --- 7.2 Compare OpenLMIS estimates with HHFA estimates (CALIBRATION) --- ##
# Summary results by level of care
comparison_df.groupby(['Facility_Level'])[['available_prop', 'available_prop_hhfa', 'difference']].mean()

# Plots
size = 10
comparison_df['consumable_labels'] = comparison_df['consumable_name_tlo'].str[:10]

# 7.2.1 Aggregate plot
comparison_df['aggregate'] = 'aggregate'
level_of_disaggregation = 'aggregate'
colour = 'red'
group_by_var = 'aggregate'
comparison_plot(level_of_disaggregation, group_by_var, colour)

# 7.2.2 Plot by facility level
level_of_disaggregation = 'Facility_Level'
group_by_var = 'Facility_Level'
colour = 'orange'
comparison_plot(level_of_disaggregation, group_by_var, colour)

# 7.2.3 Plot by item
level_of_disaggregation = 'consumable_labels'
group_by_var = 'consumable_name_tlo'
colour = 'yellow'
comparison_plot(level_of_disaggregation, group_by_var, colour)

# 7.2.4 Plot by item and facility level
fig = plt.figure(figsize=(22, 22))
plt.subplot(421)
comparison_plot_by_level(comparison_df['Facility_Level'].unique()[1])
plt.subplot(422)
comparison_plot_by_level(comparison_df['Facility_Level'].unique()[2])
plt.subplot(423)
comparison_plot_by_level(comparison_df['Facility_Level'].unique()[3])
plt.subplot(424)
comparison_plot_by_level(comparison_df['Facility_Level'].unique()[4])
plt.savefig(outputfilepath / 'comparison_plots/calibration_to_hhfa_fac_type_and_consumable.png')

# %%
# 8. PLOT SCENARIO SUMMARIES
########################################################################################################################
# Create the directory if it doesn't exist
figurespath_scenarios = outputfilepath / 'consumable_scenarios'
if not os.path.exists(figurespath_scenarios):
    os.makedirs(figurespath_scenarios)

chosen_availability_columns = [c for c in full_set_interpolated_with_scenarios_level1b_fixed.columns if c.startswith("available_prop")]
scenario_names_dict = {'available_prop': 'Actual', 'available_prop_scenario1': 'Non-therapeutic \n consumables', 'available_prop_scenario2': 'Vital medicines',
                'available_prop_scenario3': 'Pharmacist-\n managed', 'available_prop_scenario4': 'Level 1b', 'available_prop_scenario5': 'CHAM',
                'available_prop_scenario6': '75th percentile\n  facility', 'available_prop_scenario7': '90th percentile \n facility', 'available_prop_scenario8': 'Best \n facility',
                'available_prop_scenario9': 'Best facility \n (including DHO)','available_prop_scenario10': 'HIV supply \n chain', 'available_prop_scenario11': 'EPI supply \n chain',
                'available_prop_scenario12': 'HIV moved to \n Govt supply chain \n (Avg by Level)', 'available_prop_scenario13': 'HIV moved to \n Govt supply chain  \n (Avg by Facility_ID)',
                'available_prop_scenario14': 'HIV moved to \n Govt supply chain  \n (Avg by Facility_ID times 1.25)',
                'available_prop_scenario15': 'HIV moved to \n Govt supply chain  \n (Avg by Facility_ID times 0.75)'}

# Generate descriptive plots of consumable availability
program_item_mapping = pd.read_csv(path_for_new_resourcefiles  / 'ResourceFile_Consumables_Item_Designations.csv')[['Item_Code', 'item_category']]
program_item_mapping = program_item_mapping.rename(columns ={'Item_Code': 'item_code'})[program_item_mapping.item_category.notna()]
generate_descriptive_consumable_availability_plots(tlo_availability_df = full_set_interpolated_with_scenarios_level1b_fixed,
                                                       figurespath = figurespath_scenarios,
                                                       mfl = mfl,
                                                       program_item_mapping = program_item_mapping,
                                                       chosen_availability_columns  = None,
                                                       scenario_names_dict = scenario_names_dict,)



