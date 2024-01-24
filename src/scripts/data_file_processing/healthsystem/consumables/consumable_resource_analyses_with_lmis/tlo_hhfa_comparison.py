
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

# 1. Import and clean data files
#**********************************
# 1.1 Import TLO model availability data
#------------------------------------------------------
tlo_availability_df = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv")
# Drop any scenario data previously included in the resourcefile
tlo_availability_df = tlo_availability_df[['Facility_ID', 'month', 'item_code', 'available_prop']]

# 1.1.1 Attach district, facility level, program to this dataset
#----------------------------------------------------------------
# Get TLO Facility_ID for each district and facility level
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}
tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    on = ['Facility_ID'], how='left')

# 1.1.2 Attach programs
item_names = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")[['category', 'item_code', 'module_name', 'consumable_name_tlo']]
item_names = item_names.drop_duplicates('item_code')
tlo_availability_df = tlo_availability_df.merge(item_names, on = ['item_code'], how = 'left')

# Extract excel with HIV and TB consumables collapsed by facility level
cond1 = tlo_availability_df.module_name == "HIV"
cond2 = tlo_availability_df.module_name == "TB"
hivtb_df = tlo_availability_df[cond1| cond2]

# Collapse data by consumable_name_tlo and fac_type
#------------------------------------------------------
def collapse_stockout_data(_df, groupby_list, columns_to_preserve):
    """Return a dataframe with rows for the same TLO model item code and facility level collapsed into 1"""
    # Define column lists based on the aggregation function to be applied
    columns_to_average = ['available_prop']
    columns_to_preserve = columns_to_preserve

    # Define aggregation function to be applied to collapse data by item
    def custom_agg_stkout(x):
        if x.name in columns_to_average:
            return x.mean(skipna=True) if np.any(
                x.notnull() & (x >= 0)) else np.nan  # this ensures that the NaNs are retained
        elif x.name in columns_to_preserve:
            return x.iloc[0]  # this function extracts the first value

    # Collapse dataframe
    _collapsed_df = _df.groupby(groupby_list).agg(
        {col: custom_agg_stkout for col in columns_to_average + columns_to_preserve}
    ).reset_index()

    return _collapsed_df


# Collapse by facility level and item_code
groupby_list = ['Facility_Level', 'consumable_name_tlo', 'item_code']
columns_to_preserve = ['module_name', 'category']
hivtb_df_small = collapse_stockout_data(hivtb_df, groupby_list, columns_to_preserve)

hivtb_df_small.to_csv(outputfilepath / "tara_consumable_comparison/tlo_hivtb_data.csv")
