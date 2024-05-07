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

# TODO Assign values to Yes, No, Functional, Don't know etc.

