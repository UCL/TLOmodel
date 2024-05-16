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
hhfa_equipment_wide = pd.read_excel((path_to_dropbox / '07 - Data/HHFA_2018-19/2 clean/equipment_and_other_non_consumable_avaibility_hhfa.xlsx'),
                                    sheet_name = "hhfa_data")
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

# Extrapolate values from other variables
available_missing_before_extrapolation = len(unique_equipment_df_for_model[unique_equipment_df_for_model.available.isna()])
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['functional'].isin(["Yes", "No"]),'available'] = "Yes" # if there is a valid response to "functional" the equipment must be available
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['functional_today'].isin(['AVAILABLE  AND  FUNCTIONAL', 'AVAILABLE  NOT  FUNCTIONAL', "AVAILABLE DON'T KNOW IF  FUNCTIONAL", 'Yes']),'available'] = "Yes" # if equipment is functional then it is available
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['functional_today'].isin(['NOT AVAILABLE', 'No']),'available'] = "No" # if equipment is functional then it is available
available_missing_after_extrapolation = len(unique_equipment_df_for_model[unique_equipment_df_for_model.available.isna()])
print((available_missing_before_extrapolation - available_missing_after_extrapolation)/len(unique_equipment_df_for_model) * 100, "% values for 'available' extrapolated from other survey questions")

functional_missing_before_extrapolation = len(unique_equipment_df_for_model[unique_equipment_df_for_model.functional.isna()])
unique_equipment_df_for_model.loc[(unique_equipment_df_for_model['functional_today'].isin(['AVAILABLE  AND  FUNCTIONAL', 'Yes']) & unique_equipment_df_for_model['functional'].isna()),'functional'] = "Yes" # seperate question asking whether equipment was available today
unique_equipment_df_for_model.loc[(unique_equipment_df_for_model['functional_today'].isin(['AVAILABLE  NOT  FUNCTIONAL', 'No', 'NOT AVAILABLE']) & unique_equipment_df_for_model['functional'].isna()),'functional'] = "No" # seperate question asking whether equipment was available today
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['available'].isin(['AT LEAST ONE VALID', 'Available and functional']),'functional'] = "Yes" # information under the available question captures whether equipment is functional
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['available'].isin(['AVAILABLE NON VALID', 'Available not functional']),'functional'] = "No"  # information under the available question captures whether equipment is functional
functional_missing_after_extrapolation = len(unique_equipment_df_for_model[unique_equipment_df_for_model.functional.isna()])
print((functional_missing_before_extrapolation - functional_missing_after_extrapolation)/len(unique_equipment_df_for_model) * 100, "% values for 'functional' extrapolated from other survey questions")

# Sense check values - if equipment is calibrated or prepared, then it must be functional
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['prepared'] == "Yes", 'functional'] = "Yes"
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['calibrated'] == "Yes", 'functional'] = "Yes"

# Convert to binary values
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['available'].isin(['AT LEAST ONE VALID', 'Available and functional', 'AVAILABLE NON VALID', 'Available not functional', "REPORTED  AVAILABLE BUT NOT SEEN", "Yes"]),'available'] = 1
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['available'].isin(['NOT AVAILABLE TODAY', 'Not available', 'NEVER AVAILABLE', "Not observed", "No"]),'available'] = 0
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['functional'] == 3,'functional'] = np.nan # 3 implies "Don't know"
unique_equipment_df_for_model.loc[unique_equipment_df_for_model['functional'] == 'Yes','functional'] = 1
unique_equipment_df_for_model.loc[(unique_equipment_df_for_model['functional'] == 'No')|(unique_equipment_df_for_model['available'] == 0),'functional'] = 0

# Merge with equipment names and item codes from TLO model
