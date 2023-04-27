"""
This script prepares HHFA data on consumable availability for regression analysis on R.

Inputs:
1. Raw HHFA data - Q1.dta (~Dropbox/Thanzi la Onse/07 - Data/HHFA_2018-19/0 raw/2_Final data/)
2. Cleaned variable names for HHFA data - variable_list.csv (~Dropbox/Thanzi la Onse/07 - Data/HHFA_2018-19/1 processing)
3. Relevant distance calculations for regression analysis - facility_distances_hhfa.csv (~Dropbox/Thanzi la Onse/07 - Data/HHFA_2018-19/2 clean/)
4. Consumable categorisations - items_hhfa.xlsx (~Dropbox/Thanzi la Onse/07 - Data/HHFA_2018-19/1 processing/)

Outputs:
1. Cleaned consumable availability data ready for regression analysis - cleaned_hhfa_2019.csv (~Dropbox/Thanzi la Onse/07 - Data/HHFA_2018-19/2 clean/)

Consumable availability is measured as probability of stockout at any point in time.
"""

import calendar
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import copy

from collections import defaultdict

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
     'C:/Users/sm2511/Dropbox/Thanzi la Onse'
)

path_to_files_in_the_tlo_dropbox = path_to_dropbox / "07 - Data/HHFA_2018-19/" # <-- point to HHFA data folder in dropbox

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

## 1. DATA IMPORT ##
#########################################################################################
raw_hhfa = pd.read_csv(path_to_files_in_the_tlo_dropbox / '0 raw/2_Final data/Q1.csv',
                       low_memory=False)  # import 2018 data
varnames = pd.read_csv(path_to_files_in_the_tlo_dropbox / '1 processing/variable_list.csv',
                       encoding="ISO-8859-1")  # import file with cleaned variable names

# Rename HHFA columns using variable name mapping in loaded .csv
old_var_name = varnames['var']
new_var_name = varnames['new_var_name']

hhfa = copy.deepcopy(raw_hhfa)
for i in range(len(new_var_name)):
    if new_var_name[i] != np.nan:
        hhfa.rename(columns={old_var_name[i]: new_var_name[i]},
                    inplace=True)
    else:
        pass

# Rename columns with missing data to "a" and then drop these columns since these will not be used in the analysis
hhfa.rename({np.nan: "a"}, axis="columns", inplace=True)
hhfa.drop(["a"], axis=1, inplace=True)

## 2. FEATURE CLEANING/MANIPULATION ##
#########################################################################################
# Clean districts #
cond = hhfa['district'] == 'Blanytyre'
hhfa.loc[cond, 'district'] = 'Blantyre'

cond = hhfa['district'] == 'Nkhatabay'
hhfa.loc[cond, 'district'] = 'Nkhata Bay'

# Pvt for profit hospital incorrectly labelled District Hospital
cond = hhfa.fac_code == 5067
hhfa.loc[cond, 'fac_type'] = 'Other Hospital'

# Clean fac_owner
cond = hhfa.fac_owner == 'Private non profit'
hhfa.loc[cond, 'fac_owner'] = 'NGO'

# Bed count = 0 if outpatient only
for var in ['bed_count', 'inpatient_visit_count', 'inpatient_days_count']:
    cond = hhfa['outpatient_only'] == "Yes"
    hhfa.loc[cond, var] = 0

# Number of functional computers = 0 if no functional computers
cond = hhfa.functional_computer == "No"
hhfa.loc[cond, 'functional_computer_no'] = 0

for var in ['functional_landline', 'fuctional_mobile', 'functional_radio', 'functional_computer']:
    cond = hhfa[var].str.contains("Yes")
    hhfa.loc[cond, var] = "Yes"

# If facility has a function emergency vehicle, then it has an accessible emergency vehicle
cond = hhfa.functional_emergency_vehicle == "Yes"
hhfa.loc[cond, 'accessible_emergency_vehicle'] = "Yes"

cond = hhfa.accessible_emergency_vehicle == "No"
hhfa.loc[cond, 'fuel_available_today'] = "No"

cond = hhfa.accessible_emergency_vehicle == "No"
hhfa.loc[cond, 'purpose_last_vehicle_trip'] = "Not applicable"

# Correct the "Don't knows" #
for var in ['fuel_available_today']:
    cond = hhfa[var] == "98"
    hhfa.loc[cond, var] = "Don't know"

for var in ['functional_ambulance', 'functional_car', 'functional_bicycle', 'functional_motor_cycle',
            'functional_bike_ambulance']:
    cond = hhfa[var] == "No"
    var_no = var + '_no'
    hhfa.loc[cond, var_no] = 0

# Daily opening hours can't be more than 24 hours
cond = hhfa.fac_daily_opening_hours > 24
hhfa.loc[cond, 'fac_daily_opening_hours'] = 24

# Water source within 500m if piped into facility or facility grounds
hhfa.water_source_main = hhfa.water_source_main.str.lower()
cond1 = hhfa['water_source_main'] == "piped into facility"
cond2 = hhfa['water_source_main'] == "piped onto facility grounds"
hhfa.loc[cond1 | cond2, 'water_source_main_within_500m'] = "Yes"

# Edit water_source variable to reduce the number of categories
cond_other_watersource_1 = hhfa['water_source_main'] == 'protected spring'
cond_other_watersource_2 = hhfa['water_source_main'] == 'unprotected dug well'
cond_other_watersource_3 = hhfa['water_source_main'] == 'unprotected spring'
cond_other_watersource_4 = hhfa['water_source_main'] == 'tanker truck'
cond_other_watersource_5 = hhfa['water_source_main'] == 'cart w/small tank/drum ………'
cond_other_watersource_6 = hhfa['water_source_main'] == 'rainwater collection'
hhfa.loc[cond_other_watersource_1 | cond_other_watersource_2 | cond_other_watersource_3 | \
    cond_other_watersource_4 | cond_other_watersource_5 | cond_other_watersource_6, \
    'water_source_main'] = 'other'

# Convert water disruption duration
cond_hours = hhfa['water_disruption_duration_units'] == "Hours"
cond_weeks = hhfa['water_disruption_duration_units'] == "Weeks"
cond_months = hhfa['water_disruption_duration_units'] == "Months"
hhfa.loc[cond_hours, 'water_disruption_duration'] = hhfa['water_disruption_duration'] / 24
hhfa.loc[cond_weeks, 'water_disruption_duration'] = hhfa['water_disruption_duration'] * 7
hhfa.loc[cond_months, 'water_disruption_duration'] = hhfa['water_disruption_duration'] * 365 / 12

# If no functional toilet, then noe handwashing facility
cond = hhfa.functional_toilet == "No"
hhfa.loc[cond, 'functional_handwashing_facility'] = "No"

# No CVD services at lower level facilities
cond = hhfa.fac_type.str.contains("Hospital")
hhfa.loc[~cond, 'service_cvd'] = "No"

# Clean "other" category for how drug orders are placed
hhfa.drug_resupply_calculation_system_other_spec = hhfa.drug_resupply_calculation_system_other_spec.str.lower()

cond_notna = hhfa['drug_resupply_calculation_system_other_spec'].notna()
cond_both = hhfa['drug_resupply_calculation_system_other_spec'].str.contains("both") | \
            hhfa['drug_resupply_calculation_system_other_spec'].str.contains("for essentials") | \
            hhfa['drug_resupply_calculation_system_other_spec'].str.contains("personal of emergency orders") | \
            hhfa['drug_resupply_calculation_system_other_spec'].str.contains("pull & push")
hhfa.loc[cond_both & cond_notna, 'drug_resupply_calculation_system'] = "Both push and pull"

cond_pull = hhfa['drug_resupply_calculation_system_other_spec'].str.contains("buy") | \
            hhfa['drug_resupply_calculation_system_other_spec'].str.contains("cstock") | \
            hhfa['drug_resupply_calculation_system_other_spec'].str.contains("press") | \
            hhfa['drug_resupply_calculation_system_other_spec'].str.contains("counting") | \
            hhfa['drug_resupply_calculation_system_other_spec'].str.contains("the facility purchases") | \
            hhfa['drug_resupply_calculation_system_other_spec'].str.contains("bought") | \
            hhfa['drug_resupply_calculation_system_other_spec'].str.contains("purchased") | \
            hhfa['drug_resupply_calculation_system_other_spec'].str.contains("private pharmacy")

hhfa.loc[cond_pull & cond_notna, 'drug_resupply_calculation_system'] = "Facility itself (pull distribution system)"

cond_push = hhfa['drug_resupply_calculation_system_other_spec'].str.contains("from the dho")
hhfa.loc[
    cond_push & cond_notna, 'drug_resupply_calculation_system'] = "A higher level facility (push distribution system)"

# Clean "other" catergory for how drugs are transported
# TODO: drug_transport_other_spec

# If a facility does not report whether a particular source of drugs is used but indicates
# that another is used, mark the first as No
source_drugs_varlist = ['source_drugs_cmst', 'source_drugs_local_warehouse', 'source_drugs_ngo',
             'source_drugs_donor', 'source_drugs_pvt']
for source1 in source_drugs_varlist:
    cond_source_empty = hhfa[source1].isna()
    for source2 in source_drugs_varlist:
        cond_source_other = hhfa[source2] == "Yes"
        hhfa.loc[cond_source_empty & cond_source_other,source1] = "No"

# If a facility does not report whether a particular drug transport system is used but indicates
# that another is used, mark the first as No
drug_transport_varlist = ['drug_transport_local_supplier', 'drug_transport_higher_level_supplier',
'drug_transport_self', 'drug_transport_other']
for transport1 in drug_transport_varlist:
    cond_transport_empty = hhfa[transport1].isna()
    for transport2 in drug_transport_varlist:
        cond_transport_other = hhfa[transport2] == "Yes"
        hhfa.loc[cond_transport_empty & cond_transport_other,transport1] = "No"

# Drop outliers
cond = hhfa.travel_time_to_district_hq > 1000
hhfa.loc[cond, 'travel_time_to_district_hq'] = np.nan

cond = hhfa.referrals_to_other_facs == "3"
hhfa.loc[cond, 'referrals_to_other_facs'] = np.nan

# Reduce the number of categories - whether functional refrigerator is available
cond_yes1 = hhfa.functional_refrigerator_epi.str.contains("Yes")
cond_yes2 = hhfa.vaccine_storage.str.contains("Yes")
cond_notna = hhfa.functional_refrigerator_epi.notna()
hhfa.loc[(cond_yes1 | cond_yes2) & cond_notna, 'functional_refrigerator_epi'] = "Yes"
cond_no = hhfa.functional_refrigerator_epi.str.contains("No")
hhfa.loc[cond_no & cond_notna, 'functional_refrigerator_epi'] = "No"

cond_na = hhfa.service_epi == "No"
hhfa.loc[cond_na, 'vaccine_storage'] = "No"

cond_yes = hhfa.functional_refrigerator.str.replace(" ","") == 'AVAILABLEANDFUNCTIONAL'
hhfa.loc[cond_yes, 'functional_refrigerator'] = "Yes"
cond_no1 = hhfa.functional_refrigerator.str.replace(" ","") == 'NOTAVAILABLE'
cond_no2 = hhfa.functional_refrigerator.str.replace(" ","")== 'AVAILABLENOTFUNCTIONAL'
hhfa.loc[cond_no1 | cond_no2, 'functional_refrigerator'] = "No"

hhfa.functional_refrigerator_diagnostics = hhfa.functional_refrigerator_diagnostics.str.lower()
cond_yes = hhfa.functional_refrigerator_diagnostics == "available and functional"
cond_na = hhfa.functional_refrigerator_epi.isna()
cond_notna = hhfa.functional_refrigerator_diagnostics.notna()
hhfa.loc[cond_yes, 'functional_refrigerator_diagnostics'] = "Yes"

# if refrigerator is (not) available as per the EPI section, then it should be same as per the Diagnostics section
hhfa.loc[cond_yes & cond_na, 'functional_refrigerator_epi'] = "Yes"
cond_no = hhfa.functional_refrigerator_diagnostics == "available not functional" | \
          hhfa.functional_refrigerator_diagnostics.str.contains("not available")
hhfa.loc[cond_no, 'functional_refrigerator_diagnostics'] = "No"
hhfa.loc[cond_no & cond_na, 'functional_refrigerator_epi'] = "No"

# Aggregate refrigerator availability
cond_fridge_epi = hhfa.functional_refrigerator_epi == "Yes"
cond_fridge_diag = hhfa.functional_refrigerator_diagnostics == "Yes"
cond_vaccine_storage =  hhfa.vaccine_storage == "Yes"
hhfa.loc[cond_fridge_epi | cond_fridge_diag | cond_vaccine_storage, 'functional_refrigerator'] = "Yes"
cond_no_fridge_epi = hhfa.functional_refrigerator_epi == "No"
cond_no_fridge_diag = hhfa.functional_refrigerator_diagnostics == "No"
hhfa.loc[cond_no_fridge_epi & cond_no_fridge_diag, 'functional_refrigerator'] = "Yes"
cond_no_vaccine_storage = hhfa.vaccine_storage == "No"
cond_fridge_na = hhfa.functional_refrigerator.isna()
hhfa.loc[cond_no_vaccine_storage & cond_fridge_na, 'functional_refrigerator'] = "No"

# convert fac_location to binary (Yes/No)
hhfa = hhfa.rename(columns={'fac_location': 'fac_urban'})
cond1 = hhfa.fac_urban.str.lower() == "rural"
hhfa.loc[cond1, 'fac_urban'] = 'No'
cond2 = hhfa.fac_urban.str.lower() == "urban"
hhfa.loc[cond2, 'fac_urban'] = 'Yes'

# Clean water disruption variable
cond = hhfa['water_disruption_last_3mts'] == "No"
hhfa.loc[cond, 'water_disruption_duration'] = 0

# Change incharge_drugs to lower case
hhfa.incharge_drug_orders = hhfa.incharge_drug_orders.str.lower()

# Clean other category for incharge_drug_orders
incharge_drug_orders_other_mapping = pd.read_csv(path_to_files_in_the_tlo_dropbox / '1 processing/incharge_drug_orders_other_mapping.csv')
hhfa = pd.merge(hhfa, incharge_drug_orders_other_mapping[['incharge_drug_orders_other_spec',
                                                          'cleaned_incharge_drug_orders']],
                on = 'incharge_drug_orders_other_spec',
                how = 'left')
cond = hhfa.incharge_drug_orders == 'other'
hhfa.loc[cond, 'incharge_drug_orders'] = hhfa['cleaned_incharge_drug_orders']

## 3. CREATE CONSUMABLE AVAILABILITY DATAFRAME ##
#########################################################################################
# --- 3.1 Extract dataframe containing consumable availability from the raw HHFA dataframe --- #
# Rename columns variable name mapping in loaded .csv
consumables = copy.deepcopy(raw_hhfa)
for i in range(len(varnames['var'])):
    # if HHFA variable has been mapped to a consumable and is not an equipment
    if pd.notna(varnames['item'][i]) and pd.isna(varnames['equipment'][i]):
        consumables.rename(columns={varnames['var'][i]: varnames['availability_metric'][i] + '_' + varnames['item'][i]},
                           inplace=True)
    elif varnames['var'][i] == 'Fcode':  # keep facility code variable
        consumables.rename(columns={varnames['var'][i]: varnames['new_var_name'][i]},
                           inplace=True)

    # Mark all other unmapped variables to be dropped
    else:
        consumables.rename(columns={varnames['var'][i]: 'Drop'},
                           inplace=True)

consumables = consumables.drop(columns='Drop')

# Check that there are no duplicate consumable + metric entries
assert len(consumables.columns[consumables.columns.duplicated(keep='last')]) == 0

# --- 3.2 Convert availability categories to binary entries --- #
# - 3.2.1 List of columns asking about consumable availability on the day of survey - #
today_cols = [col for col in consumables.columns if 'today' in col]
# Get list of avaibility entries
i = 0
for col in today_cols:
    consumables[col] = consumables[col].str.replace(" ", "")
    if i == 0:
        today_categories = consumables[col].unique()
    else:
        today_categories = np.append(today_categories, consumables[col].unique())
    i = 1

today_categories = list(dict.fromkeys(today_categories))
today_categories = [x for x in today_categories if pd.isnull(x) == False]  # drop nan from list

# Create dictionary to map availability options to a number
today_categories_dict = {'NEVERAVAILABLE': -1}
for i in ['NOTAVAILABLE', 'No', 'AVAILABLENOTFUNCTIONAL', 'NOTAVAILABLETODAY', 'AVAILABLENONVALID']:
    today_categories_dict[i] = 0
for i in ['AVAILABLE,OBSERVED', 'AVAILABLEANDFUNCTIONAL', 'ATLEASTONEVALID']:
    today_categories_dict[i] = 1
for i in ['AVAILABLE,NOTOBSERVED', 'Yes', "AVAILABLEDON'TKNOWIFFUNCTIONAL", 'REPORTEDAVAILABLEBUTNOTSEEN']:
    today_categories_dict[i] = 2

# Assert if any entries did not get featured in the dictionary
today_mapping = {k: today_categories_dict[k] for k in today_categories_dict.keys() & set(today_categories)}
assert len(today_mapping) == len(today_categories)

# - 3.2.2 List of columns asking about consumable availability during the 3 months before survey - #
last_mth_or_3mts_cols = [col for col in consumables.columns if 'last' in col]
# Get list of avaibility entries
i = 0
for col in last_mth_or_3mts_cols:
    consumables[col] = consumables[col].str.replace(" ", "")
    if i == 0:
        last_mth_or_3mts_cols_categories = consumables[col].unique()
    else:
        last_mth_or_3mts_cols_categories = np.append(last_mth_or_3mts_cols_categories, consumables[col].unique())
    i = 1

last_mth_or_3mts_cols_categories = list(dict.fromkeys(last_mth_or_3mts_cols_categories))
last_mth_or_3mts_cols_categories = [x for x in last_mth_or_3mts_cols_categories if
                                    pd.isnull(x) == False]  # drop nan from list

# Create dictionary to map availability options to a number
last_mth_or_3mts_cols_categories_dict = {'PRODUCTNOTOFFERED': -1}
for i in ['NOTINDICATED', 'FACILITYRECORDNOTAVAILABLE']:
    last_mth_or_3mts_cols_categories_dict[i] = -1
for i in ['STOCK-OUTINTHEPAST3MONTHS', 'Yes']:
    last_mth_or_3mts_cols_categories_dict[i] = 0
for i in ['NOSTOCK-OUTINPAST3MONTHS', 'No']:
    last_mth_or_3mts_cols_categories_dict[i] = 1

# Assert if any entries did not get featured in the dictionary
last_mth_or_3mts_mapping = {k: last_mth_or_3mts_cols_categories_dict[k] for k in
                            last_mth_or_3mts_cols_categories_dict.keys() & set(last_mth_or_3mts_cols_categories)}
assert len(last_mth_or_3mts_mapping) == len(last_mth_or_3mts_cols_categories)

# - 3.2.3 Recode all availbility variables - #
consumables_num = copy.deepcopy(consumables)
for col in today_cols:
    consumables_num[col] = consumables[col].map(today_categories_dict).fillna(consumables[col])
for col in last_mth_or_3mts_cols:
    cond1 = (consumables_num[col] == 'NOTINDICATED')
    cond2 = (consumables_num[col] == 'FACILITYRECORDNOTAVAILABLE')
    consumables_num.loc[cond1 | cond2, col] = np.nan
    consumables_num[col] = consumables[col].map(last_mth_or_3mts_cols_categories_dict).fillna(consumables[col])

# - 3.2.4 Convert numeric availability variable to binary - #
consumables_bin = copy.deepcopy(consumables_num)
for col in today_cols:
    cond0 = consumables_bin[col] == -1
    cond1 = consumables_bin[col] == 2
    consumables_bin.loc[cond0, col] = 0
    consumables_bin.loc[cond1, col] = 1
for col in last_mth_or_3mts_cols:
    cond0 = consumables_bin[col] == -1
    consumables_bin.loc[cond0, col] = 0

## 4. RESHAPE DATA AND DROP DUPLICATE CONSUMABLE ENTRIES ##
#########################################################################################
consumables_long = pd.melt(consumables_bin, id_vars= 'fac_code', value_vars= today_cols,
                             var_name= 'item', value_name='value')

consumables_long = consumables_long.rename(columns = {'value': 'available'})

# Split consumable variable name into consumable name (item) and avaibility metric used
consumables_long['temp'] = consumables_long.item.str.rfind('_', start = 0, end = 100)
consumables_long['metric']=consumables_long.apply(lambda x: x['item'][:x['temp']],axis = 1)
consumables_long['item']=consumables_long.apply(lambda x: x['item'][x['temp']+1:],axis = 1)
consumables_long = consumables_long.drop(columns = 'temp')

# For items which are duplicated, keep the metric under which more facilities have reported
# Get list of items which are duplicated
duplicated_items = consumables_long[consumables_long.duplicated(['item', 'fac_code'], keep=False)].item.unique()
cond1 = consumables_long.item.isin(duplicated_items)
cond2 = consumables_long.available.notna()

# Generate variable denoting number of reporting facilities
aggregated = consumables_long[cond2].groupby(['item', 'metric']).nunique()['fac_code']
aggregated.name = 'report_count'
consumables_long = consumables_long.join(aggregated,on=['item', 'metric'])

# Sort in descending order of reporting facilities
consumables_long.sort_values(['item', 'report_count', 'fac_code'], axis=0, ascending=True)


# Drop duplicates based on number of reporting facilities
consumables_long_unique = copy.deepcopy(consumables_long)
consumables_long_unique.drop_duplicates(subset=['item', 'fac_code'], keep='first', inplace=True)

# Assert that all duplicates have been addressed
assert len(consumables_long_unique[consumables_long_unique.duplicated(['item', 'fac_code'], keep=False)].item.unique()) == 0
print(len(consumables_long[consumables_long.duplicated(['item', 'fac_code'], keep=False)].item.unique()),
     "duplicate items were reduced to",
     len(consumables_long_unique[consumables_long_unique.duplicated(['item', 'fac_code'], keep=False)].item.unique()))

## 5. ACCOUNT FOR SUBSTITUTABLE CONSUMABLES ##
#########################################################################################
# --- 5.1 ART components ---
# Component 1: AZT or TDF or d4T
cond1 = consumables_long_unique['item'].isin(['Zidovudine (ZDV, AZT)',
                            'Zidovudine (ZDV, AZT) syrup (ARVs)',
                            'Tenofovir Disoproxil Fumarate (TDF) (ARVs)',
                            'Zidovudine + Lamivudine (AZT + 3TC) (ARVs)',
                            'Zidovudine + Lamivudine + Abacavir (AZT + 3TC + ABC) (ARVs)',
                            'Zidovudine + Lamivudine + Nevirapine (AZT + 3TC + NVP) (ARVs)',
                            'Tenofovir + Emtricitabine (TDF + FTC) (ARVs)',
                            'Tenofovir + Lamivudine (TDF + 3TC) (ARVs)',
                            'Tenofovir + Lamivudine + Efavirenz (TDF + 3TC + EFV) (ARVs)',
                            'Tenofovir + Emtricitabine + Efavirenz (TDF + FTC + EFV)',
                            'Stavudine 30 or 40 (D4T) (ARVs)',
                            'Stavudine syrup (ARVs)',
                            'Stavudine + Lamivudine (D4T + 3TC) (ARVs)',
                            'Stavudine + Lamivudine + Nevirapine (D4T + 3TC + NVP) (ARVs)'])
## Component 2: 3TC or FTC
cond2 = consumables_long_unique['item'].isin(['Lamivudine (3TC) (ARVs)',
                            'Emtricitabine (FTC) (ARVs)',
                            'Lamivudine + Abacavir (3TC + ABC)',
                            'Zidovudine + Lamivudine (AZT + 3TC) (ARVs)',
                            'Zidovudine + Lamivudine + Abacavir (AZT + 3TC + ABC) (ARVs)',
                            'Zidovudine + Lamivudine + Nevirapine (AZT + 3TC + NVP) (ARVs)',
                            'Tenofovir + Emtricitabine (TDF + FTC) (ARVs)',
                            'Tenofovir + Lamivudine (TDF + 3TC) (ARVs)',
                            'Tenofovir + Lamivudine + Efavirenz (TDF + 3TC + EFV) (ARVs)',
                            'Tenofovir + Emtricitabine + Efavirenz (TDF + FTC + EFV)',
                            'Lamivudine (3TC) syrup (ARVs)',
                            'Stavudine + Lamivudine (D4T + 3TC) (ARVs)',
                            'Stavudine + Lamivudine + Nevirapine (D4T + 3TC + NVP) (ARVs)'])
## Component 3: Protease inhibitor
cond3 = consumables_long_unique['item'].isin(['Abacavir (ABC) (ARVs)',
                            'Nevirapine (NVP) (ARVs)',
                            'Nevirapine (NVP) syrup (ARVs)',
                            'Efavirenz (EFV) (ARVs)',
                            'Lamivudine + Abacavir (3TC + ABC)',
                            'Zidovudine + Lamivudine + Abacavir (AZT + 3TC + ABC) (ARVs)',
                            'Zidovudine + Lamivudine + Nevirapine (AZT + 3TC + NVP) (ARVs)',
                            'Tenofovir + Lamivudine + Efavirenz (TDF + 3TC + EFV) (ARVs)',
                            'Tenofovir + Emtricitabine + Efavirenz (TDF + FTC + EFV)',
                            'Efavirenz (EFV) syrup (ARVs)',
                            'Stavudine + Lamivudine + Nevirapine (D4T + 3TC + NVP) (ARVs)',
                            'Lopinavir (LPV) (protease inhibitors)',
                            'Indinavir (IDV) (protease inhibitors)',
                            'Nelfinavir (NFV) (protease inhibitors)',
                            'Saquinavir (SQV) (protease inhibitors)',
                            'Ritonavir (RTV) (protease inhibitors)',
                            'Atazanavir (ATV) (protease inhibitors)',
                            'Fosamprenavir (FPV) (protease inhibitors)',
                            'Tipranavir (TPV) (protease inhibitors)',
                            'Darunavir (DPV) (protease inhibitors)'])

# Component 1
art_component_1 = consumables_long_unique[cond1]
art_component_1['item'] = 'art_component_1'
art_component_1 = art_component_1.groupby(
    ['fac_code', 'item'],
    as_index=False).agg({'available': np.nanmax,
                         'metric': 'first'})

# Component 2
art_component_2 = consumables_long_unique[cond2]
art_component_2['item'] = 'art_component_2'
art_component_2 = art_component_2.groupby(
    ['fac_code', 'item'],
    as_index=False).agg({'available': np.nanmax,
                         'metric': 'first'})

# Component 3
art_component_3 = consumables_long_unique[cond3]
art_component_3['item'] = 'art_component_3'
art_component_3 = art_component_3.groupby(
    ['fac_code', 'item'],
    as_index=False).agg({'available': np.nanmax,
                         'metric': 'first'})

# Append all datasets
art = art_component_1.append(art_component_2)
art = art.append(art_component_3)

consumables_postart = art.append(consumables_long_unique[~cond1 & ~cond2 & ~cond3])

print(consumables_postart.item.nunique(),
      "left out of",
      consumables_long_unique.item.nunique(),
     "items left after accounting for ART substitutes")

# --- 5.2 Tuberculosis treatment substitutes --- #
# Combinations used as per https://stoptb.org/assets/documents/gdf/whatis/faq-brochure.pdf = RHZE, RHZ, RH, EH, TH
# Component 1: Isoniazid
cond1 = consumables_postart['item'].isin(['Isoniazid',
                                          'Isoniazid + Rifampicin (2FDC)',
                                          'Isoniazid + Ethambutol (EH) (2FDC)',
                                          'Isoniazid + Rifampicin + Pyrazinamide (RHZ) (3FDC)',
                                          'Isoniazid + Rifampicin + Ethambutol (RHE) (3FDC)',
                                          'Isoniazid + Rifampicin + Pyrazinamide + Ethambutol (4FDC)'])

# Component 2: Rifampicin
cond2 = consumables_postart['item'].isin(['Rifampicin',
                                          'Isoniazid + Rifampicin (2FDC)',
                                          'Isoniazid + Rifampicin + Pyrazinamide (RHZ) (3FDC)',
                                          'Isoniazid + Rifampicin + Ethambutol (RHE) (3FDC)',
                                          'Isoniazid + Rifampicin + Pyrazinamide + Ethambutol (4FDC)'])

# Component 3: Pyrazinamide
cond3 = consumables_postart['item'].isin(['Pyrazinamide',
                                          'Isoniazid + Rifampicin + Pyrazinamide (RHZ) (3FDC)',
                                          'Isoniazid + Rifampicin + Pyrazinamide + Ethambutol (4FDC)'])

# Component 4: Ethambutol
cond4 = consumables_postart['item'].isin(['Ethambutol',
                                          'Isoniazid + Ethambutol (EH) (2FDC)',
                                          'Isoniazid + Rifampicin + Ethambutol (RHE) (3FDC)',
                                          'Isoniazid + Rifampicin + Pyrazinamide + Ethambutol (4FDC)'])

# Component 1
tb_component_1 = consumables_postart[cond1]
tb_component_1['item'] = 'Isoniazid'
tb_component_1 = tb_component_1.groupby(
    ['fac_code', 'item'],
    as_index=False).agg({'available': np.nanmax,
                         'metric': 'first'})
# Component 2
tb_component_2 = consumables_postart[cond2]
tb_component_2['item'] = 'Rifampicin'
tb_component_2 = tb_component_2.groupby(
    ['fac_code', 'item'],
    as_index=False).agg({'available': np.nanmax,
                         'metric': 'first'})
# Component 3
tb_component_3 = consumables_postart[cond3]
tb_component_3['item'] = 'Pyrazinamide'
tb_component_3 = tb_component_3.groupby(
    ['fac_code', 'item'],
    as_index=False).agg({'available': np.nanmax,
                         'metric': 'first'})
# Component 4
tb_component_4 = consumables_postart[cond4]
tb_component_4['item'] = 'Ethambutol'
tb_component_4 = tb_component_4.groupby(
    ['fac_code', 'item'],
    as_index=False).agg({'available': np.nanmax,
                         'metric': 'first'})

# Append all datasets
tb = tb_component_1.append(tb_component_2)
tb = tb.append(tb_component_3)
tb = tb.append(tb_component_4)

consumables_posttb = tb.append(consumables_postart[~cond1 & ~cond2 & ~cond3 & ~cond4])

print(consumables_posttb.item.nunique(),
      "left out of",
      consumables_long_unique.item.nunique(),
      "items left after accounting for ART and TB substitutes")

# --- 5.3 Iron and folic acid --- #
# Component 1: Iron
cond1 = consumables_posttb['item'].isin(['Iron tablet',
                            'Iron and folic combined tablets'])

# Component 2: Folic Acid
cond2 = consumables_posttb['item'].isin(['Folic acid tablet',
                            'Iron and folic combined tablets'])

# Component 1
fefo_component_1 = consumables_posttb[cond1]
fefo_component_1['item'] = 'Iron tablet'
fefo_component_1 = fefo_component_1.groupby(
    ['fac_code', 'item'],
    as_index=False).agg({'available': np.nanmax,
                         'metric': 'first'})
# Component 2
fefo_component_2 = consumables_posttb[cond2]
fefo_component_2['item'] = 'Folic acid tablet'
fefo_component_2 = fefo_component_2.groupby(
    ['fac_code', 'item'],
    as_index=False).agg({'available': np.nanmax,
                         'metric': 'first'})

# Append all datasets
fefo = fefo_component_1.append(fefo_component_2)

consumables_postfefo = fefo.append(consumables_posttb[~cond1 & ~cond2])

print(consumables_postfefo.item.nunique(),
      "left out of",
      consumables_long_unique.item.nunique(),
     "items left after accounting for ART, TB, and FeFo substitutes")

# --- 5.4 Other substitutes --- #
# Merge in substitute data
item_categories = pd.read_excel(path_to_files_in_the_tlo_dropbox / '1 processing/items_hhfa.xlsx',
                                index_col=0, sheet_name = "items_hhfa")

cols = ['item', 'substitute_group']
consumables_postfefo = pd.merge(consumables_postfefo,item_categories[cols], how = 'left', on = 'item')
cond = consumables_postfefo.substitute_group == 'manual fix'
consumables_postfefo.loc[cond, 'substitute_group'] = np.nan

j = 0
for i in consumables_postfefo.substitute_group.unique():
    cond = consumables_postfefo.substitute_group == i
    sub_group = consumables_postfefo[cond]
    sub_group = sub_group.groupby(['fac_code', 'substitute_group'], as_index=False).agg({'available': np.nanmax,
                                                                            'metric': 'first',
                                                                            'item': 'first'})
    if j == 0:
        sub_groups_all = sub_group
        j = 1
    else:
        sub_groups_all = sub_groups_all.append(sub_group)

# Append with the remaining data
consumables_final = consumables_postfefo[consumables_postfefo.substitute_group.isna()]
consumables_final = sub_groups_all.append(consumables_final)

print(consumables_final.item.nunique(),
      "left out of",
      consumables_long_unique.item.nunique(),
     "items left after accounting for all substitutes")

## 6. PREPARE DATA FOR REGRESSION ##
#########################################################################################
# Keep columns to be analysed
feature_cols = ['fac_code', 'fac_name',
            # Main characteristics
            'district', 'fac_type','fac_owner', 'fac_urban',

            # types of services offered
            'outpatient_only','bed_count', 'inpatient_visit_count','inpatient_days_count',
            'service_fp','service_anc','service_pmtct','service_delivery','service_pnc','service_epi','service_imci',
            'service_hiv','service_tb','service_othersti','service_malaria','service_blood_transfusion','service_diagnostic',
            'service_cvd','service_chronic_respiratory_mgt', 'service_consumable_stock',

            # operation frequency
            'fac_weekly_opening_days','fac_daily_opening_hours',

            # utilities/facilities available
            'functional_landline','fuctional_mobile','functional_radio','functional_computer','functional_computer_no',
            'internet_access_today',
            'electricity', 'water_source_main','water_source_main_within_500m', # this is in minutes
            'functional_toilet','functional_handwashing_facility',
            'water_disruption_last_3mts','water_disruption_duration', # converted to days

            # vehicles available
            'functional_emergency_vehicle','accessible_emergency_vehicle','fuel_available_today','purpose_last_vehicle_trip',

            'functional_ambulance', 'functional_ambulance_no', # Keep one of the two variables
            'functional_car', 'functional_car_no',
            'functional_motor_cycle', 'functional_motor_cycle_no',
            'functional_bike_ambulance', 'functional_bike_ambulance_no',
            'functional_bicycle', 'functional_bicycle_no',

            'vaccine_storage','functional_refrigerator',

            # Drug ordering process
            'incharge_drug_orders','drug_resupply_calculation_system','drug_resupply_calculation_method','source_drugs_cmst',
            'source_drugs_local_warehouse','source_drugs_ngo','source_drugs_donor','source_drugs_pvt',

            'drug_transport_local_supplier','drug_transport_higher_level_supplier','drug_transport_self','drug_transport_other',

            'drug_order_fulfilment_delay','drug_order_fulfilment_freq_last_3mts',
            'transport_to_district_hq','travel_time_to_district_hq',

            # referral system
            'referral_system_from_community','referrals_to_other_facs',

            # Drug management
            'mathealth_label_and_expdate_visible', 'mathealth_expdate_fefo',
            'childhealth_label_and_expdate_visible', 'childhealth_expdate_fefo']

# Merge consumable availability data
merged_df1 = pd.merge(consumables_final,hhfa[feature_cols], how = 'left', on = 'fac_code')

# Merge distances data
fac_gis = pd.read_csv(path_to_files_in_the_tlo_dropbox / "2 clean/facility_distances_hhfa.csv")
merged_df2 = pd.merge(merged_df1, fac_gis[['fac_code', 'lat', 'long', 'lat_dh', 'long_dh', 'lat_rms', 'long_rms','rms',
                                  'dist_todh', 'dist_torms', 'drivetime_todh', 'drivetime_torms']], how = 'left', on = 'fac_code')

# Merge item categories
item_categories = pd.read_excel(path_to_files_in_the_tlo_dropbox / '1 processing/items_hhfa.xlsx',
                                index_col=0, sheet_name = "items_hhfa")

cols = ['drug_class_rx_list', 'item', 'mode_administration', 'program', 'item_type']
df_for_regression = pd.merge(merged_df2,item_categories[cols], how = 'left', on = 'item')

cond = df_for_regression.item.str.contains('art_component')
df_for_regression.loc[cond, 'program'] = 'hiv'
df_for_regression.loc[cond, 'mode_administration'] = 'oral'
df_for_regression.loc[cond, 'drug_class_rx_list'] = 'antiretroviral'
df_for_regression.loc[cond, 'item_type'] = 'drug'

# Clean program to reduce the number of categories
cond_resp = df_for_regression.program == 'respiratory illness'
df_for_regression.loc[cond_resp,'program'] = 'acute lower respiratory infections'

cond_road = df_for_regression.program == 'road traffic injuries'
df_for_regression.loc[cond_road,'program'] = 'surgical'

cond_fungal = df_for_regression.program == 'fungal infection'
df_for_regression.loc[cond_fungal,'program'] = 'other'

cond_iv = df_for_regression.program == 'IV and injectables'
df_for_regression.loc[cond_iv,'program'] = 'general'

cond_ncd1 = df_for_regression.program == 'hypertension'
cond_ncd2 = df_for_regression.program == 'diabetes'
df_for_regression.loc[cond_ncd1|cond_ncd2,'program'] = 'ncds'

cond_vit = df_for_regression.item == 'Vitamin A (retinol) capsule'
df_for_regression.loc[cond_vit, 'program'] = 'obstetric and newborn care'

cond_mvit = df_for_regression.item == 'Multivitamins'
df_for_regression.loc[cond_mvit, 'program'] = 'general'

cond_rutf = df_for_regression.item == 'Ready to use therapeutic food(RUTF)'
df_for_regression.loc[cond_rutf, 'program'] = 'child health'

# Reduce the number of categories in mode_administration
cond = df_for_regression['mode_administration'] == "implant"
df_for_regression.loc[cond, 'mode_administration'] = "other"

# Clean facility type
df_for_regression = df_for_regression.rename(columns = {'fac_type': 'fac_type_original'})
df_for_regression['fac_type'] = ""

cond_mch = (df_for_regression['fac_name'].str.contains('Mzuzu Cental Hospital'))
df_for_regression.loc[cond_mch, 'fac_name'] = 'Mzuzu Central Hospital'

cond_level0 = (df_for_regression['fac_name'].str.contains('Health Post')) | \
                (df_for_regression['fac_type_original'].str.contains('Health Post'))
cond_level1a = (df_for_regression['fac_type_original'] == 'Clinic') | (df_for_regression['fac_type_original'] == 'Health Centre') | \
        (df_for_regression['fac_type_original'].str.contains('Dispensary')) | \
        (df_for_regression['fac_type_original'].str.contains('Maternity'))
cond_level1b =  (df_for_regression['fac_type_original'].str.contains('Community Hospital')) | \
                (df_for_regression['fac_type_original'] == 'Other Hospital')
cond_level2 = (df_for_regression['fac_type_original'] == 'District Hospital')
cond_level3 = df_for_regression.fac_name.str.contains("Central Hospit")
cond_level4 = df_for_regression.fac_name.str.contains("Mental Hospit")

df_for_regression.loc[cond_level0,'fac_type'] = 'Facility_level_0'
df_for_regression.loc[cond_level1a,'fac_type'] = 'Facility_level_1a'
df_for_regression.loc[cond_level1b,'fac_type'] = 'Facility_level_1b'
df_for_regression.loc[cond_level2,'fac_type'] = 'Facility_level_2'
df_for_regression.loc[cond_level3,'fac_type'] = 'Facility_level_3'
df_for_regression.loc[cond_level4,'fac_type'] = 'Facility_level_4'

# Sort by facility type
df_for_regression['fac_type_original'] = pd.Categorical(df_for_regression['fac_type_original'], ['Health Post',
                'Dispensary',
                'Maternity',
                'Clinic',
                'Health Centre',
                'Rural/Community Hospital',
                'Other Hospital',
                'District Hospital',
                'Central Hospital'])
df_for_regression.sort_values(['fac_name','fac_type_original'], inplace = True)

# Convert
# ry variables to numeric
fac_vars_binary = ['outpatient_only', 'fac_urban',
                'service_fp','service_anc','service_pmtct','service_delivery','service_pnc','service_epi','service_imci',
                'service_hiv','service_tb','service_othersti','service_malaria','service_blood_transfusion','service_diagnostic',
                'service_cvd', 'service_consumable_stock',
                'functional_landline','fuctional_mobile','functional_radio','functional_computer','internet_access_today',
                'electricity', 'functional_toilet','functional_handwashing_facility',
                'water_disruption_last_3mts', 'water_source_main_within_500m',
                'functional_emergency_vehicle','accessible_emergency_vehicle',
                'functional_ambulance',
                'functional_car',
                'functional_motor_cycle',
                'functional_bike_ambulance',
                'functional_bicycle',
                'vaccine_storage','functional_refrigerator',
                'source_drugs_cmst','source_drugs_local_warehouse','source_drugs_ngo','source_drugs_donor','source_drugs_pvt',
                'drug_transport_local_supplier','drug_transport_higher_level_supplier','drug_transport_self','drug_transport_other',
                'referral_system_from_community','referrals_to_other_facs',
                'mathealth_label_and_expdate_visible', 'mathealth_expdate_fefo',
                'childhealth_label_and_expdate_visible']

binary_dict = {'yes': 1, 'no': 0}

df_for_regression_binvars_cleaned = copy.deepcopy(df_for_regression)
for col in fac_vars_binary:
    df_for_regression_binvars_cleaned[col] = df_for_regression_binvars_cleaned[col].str.lower()

    # replace don't know as missing
    cond = df_for_regression_binvars_cleaned[col] == "don't know"
    df_for_regression_binvars_cleaned.loc[cond, col] = np.nan

    assert len([x for x in df_for_regression_binvars_cleaned[col].unique() if
                pd.isnull(x) == False]) == 2  # verify that the variable is binary

    df_for_regression_binvars_cleaned[col] = df_for_regression_binvars_cleaned[col].map(binary_dict).fillna(df_for_regression_binvars_cleaned[col])

# Save dataset ready for regression analysis as a .csv file
df_for_regression_binvars_cleaned.to_csv(path_to_files_in_the_tlo_dropbox / '2 clean/cleaned_hhfa_2019.csv')
