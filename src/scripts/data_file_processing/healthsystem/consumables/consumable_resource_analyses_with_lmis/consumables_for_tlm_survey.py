"""
This script chooses a subset of consumables on which data will be collected through the TLM Data collection protocol.
A representative sample of consumables is selected using the stratified sampling approach. Consumables are divided into
8 groups based on the following combination of characteristics - 1. Top 20 consumables demanded within the TLO model (Oversampled),
2. General vs Disease-specific consumables, 3. Consumables split into 4 groups based on quartiles of average availability
for each level of care. Strata with only 1 consumable were dropped from the list before stratified sampling.
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
    '/Users/sm2511/Dropbox/Thanzi la Onse'
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

# 1. Import and clean data files
#**********************************
# 1.1 Import TLO model availability data
#------------------------------------------------------
tlo_availability_df = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")
# Drop any scenario data previously included in the resourcefile
tlo_availability_df = tlo_availability_df[['fac_type_tlo', 'module_name', 'category','item_code', 'consumable_name_tlo',
                                           'available_prop', 'data_source', 'dispensed']]

# Collapse data by consumable_name_tlo and fac_type_tlo
#------------------------------------------------------
def collapse_stockout_data(_df, groupby_list, columns_to_preserve):
    """Return a dataframe with rows for the same TLO model item code and facility level collapsed into 1"""
    # Define column lists based on the aggregation function to be applied
    columns_to_average = ['available_prop']
    columns_to_sum = ['dispensed']
    columns_to_preserve = columns_to_preserve

    # Define aggregation function to be applied to collapse data by item
    def custom_agg_stkout(x):
        if x.name in columns_to_average:
            return x.mean(skipna=True) if np.any(
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
        {col: custom_agg_stkout for col in columns_to_average + columns_to_sum + columns_to_preserve}
    ).reset_index()

    return _collapsed_df


# Collapse by facility level and item_code
groupby_list = ['fac_type_tlo','item_code', 'consumable_name_tlo']
columns_to_preserve = ['module_name', 'category', 'data_source']
df_by_item_fac = collapse_stockout_data(tlo_availability_df, groupby_list, columns_to_preserve)

# Assign top 20 most requested to a group
item_code_top20_most_requested = [141, # Blood, one unit
                                  130, # Gentamicin 40mg/ml, 2ml_each_CMST
                                  2606, # Benzylpenicillin 1g (1MU), PFR_Each_CMST
                                  202, # Sulfamethoxazole + trimethropin, oral suspension
                                  2673, # First line ART regimen: young child
                                  178, # 2-FDC tablets (E400/H150)
                                  178, # 4-FDC tablets (R150/H75/Z400/E275)
                                  17, # Lidocaine, spray, 10%, 500 ml bottle
                                  29, # Metronidazole, injection, 500 mg in 100 ml vial
                                  170, # Injectable artesunate
                                  98, # Water for injection, 10ml_Each_CMST
                                  133, # Dextrose (glucose) 5%, 1000ml_each_CMST
                                  134, # Tube, feeding CH 8_each_CMST
                                  225, # Frusemide 40mg_1000_CMST (furosemide)
                                  125, # Amoxycillin 250mg_1000_CMST
                                  179, # Streptomycin sulphate powder for injection, 1g
                                  248, # Flucloxacillin 250mg_100_CMST
                                  1191, # Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200mg
                                  ]
# Dianeal + Dextrose is not in the dataset, For the four TB drugs in the top 20, 178 ane 179 have been added
cond_top20 = df_by_item_fac.item_code.isin(item_code_top20_most_requested)
df_by_item_fac['top20_requested'] = df_by_item_fac['item_code'].isin(item_code_top20_most_requested).astype(int)

# Assign groups for general versus disease specific
df_by_item_fac['general_vs_disease-specific'] = df_by_item_fac['category'].isin(['general']).astype(int)

# Assign groups for stock-out rate - Quartiles by facility level
i = 0
for level in ['Facility_level_1a', 'Facility_level_1b',
       'Facility_level_2', 'Facility_level_3', 'Facility_level_4']:
    print("Running ", level)
    df_temp = df_by_item_fac.loc[df_by_item_fac.fac_type_tlo == level]
    df_temp['stockout_group'] = 0
    cond1 = (df_temp['available_prop'] <= df_temp['available_prop'].quantile(0.25))
    cond2 = (df_temp['available_prop'] <= df_temp['available_prop'].quantile(0.5)) & (df_temp['available_prop'] > df_temp['available_prop'].quantile(0.25))
    cond3 = (df_temp['available_prop'] <= df_temp['available_prop'].quantile(0.75)) & (df_temp['available_prop'] > df_temp['available_prop'].quantile(0.5))
    cond4 = (df_temp['available_prop'] > df_temp['available_prop'].quantile(0.75))
    df_temp.loc[cond1,'stockout_group'] = "bottom 25 percentile"
    df_temp.loc[cond2, 'stockout_group'] = "25-50 percentile"
    df_temp.loc[cond3, 'stockout_group'] = "50-75 percentile"
    df_temp.loc[cond4, 'stockout_group'] = "top 75 percentile"

    print(df_temp.stockout_group.unique())
    if i == 0:
        print("running 0")
        df_for_merge = df_temp
    else:
        print("running other")
        df_for_merge = pd.concat([df_for_merge, df_temp], ignore_index=True)
    i = 1

df_by_item_fac = pd.merge(df_by_item_fac, df_for_merge[['fac_type_tlo', 'item_code', 'stockout_group']],
                          on = ['fac_type_tlo', 'item_code'], how = 'left')


df_by_item_fac['sampling_group'] = df_by_item_fac.groupby(['stockout_group','general_vs_disease-specific',
                                           'top20_requested']).ngroup()
df_by_item_fac = df_by_item_fac[df_by_item_fac['sampling_group'].notna()]
df_by_item_fac = df_by_item_fac[df_by_item_fac['available_prop'].notna()]

# Collapse by item_code
groupby_list2 = ['consumable_name_tlo']
columns_to_preserve2 = ['item_code', 'module_name', 'category', 'data_source', 'sampling_group', 'top20_requested', 'general_vs_disease-specific', 'stockout_group']
df_by_item = collapse_stockout_data(df_by_item_fac, groupby_list2, columns_to_preserve2)

df_by_item = df_by_item[df_by_item['data_source'] != 'other']

group_size = df_by_item['sampling_group'].value_counts()
min_group_size_index = group_size[group_size >1].index
df_by_item = df_by_item[df_by_item.sampling_group.isin(min_group_size_index)]

train, test = train_test_split(df_by_item, test_size=21/len(df_by_item), stratify=df_by_item['sampling_group'], random_state=876)
test.to_csv(outputfilepath / "ResourceFile_tlm_survey_sample.csv")

"""
from sklearn.model_selection import StratifiedShuffleSplit
# Define the number of samples you want per stratum
samples_per_stratum = 20

# Create an instance of StratifiedShuffleSplit
stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=samples_per_stratum, random_state=42)

# Split the population into training and test sets
for train_index, test_index in stratified_splitter.split(df_by_item, df_by_item['sampling_group']):
    stratified_sample = df_by_item.iloc[test_index]
"""

# Assuming you have a DataFrame df_by_item and want to oversample from a specific stratum
stratum_to_oversample = [3,7,11,15]  # Replace with the actual stratum name

# Create an instance of StratifiedShuffleSplit
stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=16, random_state=123)

# Get the indices for training and testing sets
for train_index, test_index in stratified_splitter.split(df_by_item, df_by_item['sampling_group']):
    train_set = df_by_item.iloc[train_index]
    test_set = df_by_item.iloc[test_index]

# Oversample from the specified stratum in the training set
oversampled_stratum = train_set[train_set['sampling_group'].isin(stratum_to_oversample)].sample(n=4, replace=True, random_state=123)

# Combine oversampled stratum with the rest of the training set
test_set_oversampled = pd.concat([test_set, oversampled_stratum])
test_set_oversampled.to_csv(outputfilepath / "ResourceFile_tlm_survey_sample.csv")
