import calendar
import datetime
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    '/Users/sm2511/Dropbox/Thanzi la Onse'
)

path_to_files_in_the_tlo_dropbox = path_to_dropbox / "05 - Resources/Module-healthsystem/equipment/"

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define a pathway to the data folder (note: currently outside the TLO model directory)
# remember to set working directory to TLOmodel/
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/infrastructure_and_equipment/"

# Import Raw data
hhfa_equipment_wide = pd.read_csv(path_to_dropbox / '07 - Data/HHFA_2018-19/2 clean/equipment_and_other_non_consumable_avaibility.csv')
hhfa_equipment_wide = hhfa_equipment_wide.drop(hhfa_equipment_wide.columns[0], axis=1)
# Reshape data
hhfa_equipment = pd.melt(hhfa_equipment_wide, id_vars=['fac_code'], var_name='equipment_availability_var', value_name='response')
hhfa_equipment['equipment'] =  hhfa_equipment['equipment_availability_var'].str.split('_').str[0]
hhfa_equipment['availability_var'] = hhfa_equipment['equipment_availability_var'].str.split('_').str[1]

# Preserve only relevant datapoints
relevant_varlist = ['functional','today', 'today-functional', 'calibrated', 'date-last-calibrated', 'prepared','previous-prep']
hhfa_equipment_df_for_model = hhfa_equipment[hhfa_equipment['availability_var'].isin(relevant_varlist)]

# Create a dataframe with 6 columns of HHFA data on availability - 'available', 'functional', 'calibrated', 'date_last_calibrated' , 'prepared', 'date_last_prepared'
# Reshape data
unique_equipment_df_for_model = hhfa_equipment_df_for_model.pivot(index=['fac_code', 'equipment'], columns='availability_var', values='response')
unique_equipment_df_for_model = unique_equipment_df_for_model.reset_index()

# Rename data columns
new_column_names = {'date-last-calibrated': 'date_last_calibrated', 'previous-prep': 'date_last_prepared',
                    'today': 'available', 'today-functional': 'functional_today'}
unique_equipment_df_for_model = unique_equipment_df_for_model.rename(columns=new_column_names)

# TODO Mark equipment as avaiable if there is info only on functional, calibrated, prepared etc., similary functional if clibrated or preapred

# TODO Assign values to Yes, No, Functional, Don't know etc.
unique_combinations = hhfa_equipment[['equipment', 'availability_var']].drop_duplicates()
print(unique_combinations)
