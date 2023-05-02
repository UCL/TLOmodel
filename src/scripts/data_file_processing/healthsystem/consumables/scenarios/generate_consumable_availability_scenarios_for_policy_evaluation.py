"""
This script adds estimates of availability of consumables under different scenarios to the base Resource File:

Outputs:
* Updated version of ResourceFile_Consumables_availability_small.csv (estimate of consumable available - smaller file for use in the
 simulation) which includes new consumable availability estimates for policy evaluation scenarios

Inputs:
* outputs/regression_analysis/predictions/predicted_consumable_availability_computers_scenario.csv - This file is hosted
locally after running consumable_resource_analyses_with_hhfa/regression_analysis/8_predict.R
* ResourceFile_Consumables_availability_small.csv` - This file contains the original consumable availability estimates
from OpenLMIS 2018 data
* `ResourceFile_Consumables_matched.csv` - This file contains the crosswalk of HHFA consumables to consumables in the
TLO model

It creates one row for each consumable for availability at a specific facility and month when the data is extracted from
the OpenLMIS dataset and one row for each consumable for availability aggregated across all facilities when the data is
extracted from the Harmonised Health Facility Assessment 2018/19.

Consumable availability is measured as probability of stockout at any point in time.
"""

import calendar
import datetime
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.methods.consumables import check_format_of_consumables_file

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    'C:/Users/sm2511/Dropbox/Thanzi la Onse'
)

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define a pathway to the data folder (note: currently outside the TLO model directory)
# remember to set working directory to TLOmodel/
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"

# 1. Import data files
# Import TLO model availability data
tlo_availability_df = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv")

# Import scenario data
scenario_availability_df = pd.read_csv(outputfilepath / "regression_analysis/predictions/predicted_consumable_availability_computers_scenario.csv")
scenario_availability_df = scenario_availability_df.rename({'item': 'item_hhfa'}, axis=1)

# Get Scenario data ready to be merged based on TLO model features
scenario_availability_df['fac_type'] = scenario_availability_df['fac_type'].str.replace("Facility_level_", "")

# Do some mapping to make the Districts line-up with the definition of Districts in the model
rename_and_collapse_to_model_districts = {
    'Mzimba South': 'Mzimba',
    'Mzimba North': 'Mzimba',
}
scenario_availability_df['district_std'] = scenario_availability_df['district'].replace(rename_and_collapse_to_model_districts)

# Cities to get same results as their respective regions
copy_source_to_destination = {
    'Mzimba': 'Mzuzu City',
    'Lilongwe': 'Lilongwe City',
    'Zomba': 'Zomba City',
    'Blantyre': 'Blantyre City'
}
for source, destination in copy_source_to_destination.items():
    new_rows = scenario_availability_df.loc[scenario_availability_df.district_std == source].copy()
    new_rows.district_std = destination
    scenario_availability_df = scenario_availability_df.append(new_rows)

# Get TLO Facility_ID for each district and facility level
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}

# Now, merge-in facility_id
scenario_availability_facid_merge = scenario_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['district_std', 'fac_type'],
                    right_on=['District', 'Facility_Level'], how='left', indicator=True)
scenario_availability_facid_merge = scenario_availability_facid_merge.rename({'_merge': 'merge_facid'}, axis=1)

# Extract list of District X Facility Level combinations for which there is no HHFA data
scenario_availability_df_test = scenario_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['district_std', 'fac_type'],
                    right_on=['District', 'Facility_Level'], how='right', indicator=True)
cond = (scenario_availability_df_test['Facility_Level'].isin(['1a', '1b'])) & (scenario_availability_df_test['_merge'] == 'right_only')
scenario_availability_df_test[cond][['District', 'Facility_Level']]

# According to HHFA data, Balaka, Machinga, Mwanza, Ntchisi and Salima do not have level 1b facilities
# Likoma was not covered by the HHFA

# Load TLO - HHFA consumable name crosswalk
consumable_crosswalk_df = pd.read_csv(path_for_new_resourcefiles / 'ResourceFile_consumables_matched.csv')

# Now merge in TLO consumable IDs
scenario_availability_facid_itemcode_merge = scenario_availability_facid_merge.merge(consumable_crosswalk_df[['item_code', 'item_hhfa', 'regression_application']],
                    on = ['item_hhfa'], how='right', indicator=True)
scenario_availability_facid_itemcode_merge = scenario_availability_facid_itemcode_merge.rename({'_merge': 'merge_itemcode'}, axis=1)

items_not_matched = scenario_availability_facid_itemcode_merge['merge_itemcode'] == 'right_only'
scenario_availability_facid_itemcode_merge[items_not_matched]['regression_application'].unique()
len(scenario_availability_facid_itemcode_merge[items_not_matched]['regression_application'].isna())
# 'assume average', 'not relevant to logistic regression analysis', nan - will need to find a way to handle these

# Merge TLO model availability data with scenario data using crosswalk
new_availability_df = tlo_availability_df.merge(scenario_availability_facid_itemcode_merge[['item_code','Facility_ID','availability_change_prop', 'regression_application']],
                               how='left', on=['Facility_ID', 'item_code'])
new_availability_df.to_csv(outputfilepath / 'current_status_of_scenario_merge.csv')
# TODO Check why only 'proxy' regression application has merged in

# Create new consumable availability estimates for TLO model consumables using
# estimates of proportional change from the regression analysis based on HHFA data
new_availability_df['available_prop_scenario1'] = new_availability_df['available_prop'] * new_availability_df['availability_change_prop']

# TODO: What about cases which do not exist in HHFA data?

# Run test to make sure that all consumables are still in the dataset

# Save updated file
