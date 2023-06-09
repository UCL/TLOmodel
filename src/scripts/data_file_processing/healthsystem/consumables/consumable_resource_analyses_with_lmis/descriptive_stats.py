"""
This script generates the consumables availability dataset for regression analysis using the outputs of -
consumables_availability_estimation.py and clean_fac_locations.py -
and generates descriptive figures and tables.
"""
import datetime
from pathlib import Path

import pandas as pd

# import numpy as np
# import calendar
# import copy
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from matplotlib import pyplot # for figures
# import seaborn as sns
# import math

# Path to TLO directory
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    'C:/Users/sm2511/Dropbox/Thanzi la Onse'
)

path_to_files_in_the_tlo_dropbox = path_to_dropbox / "05 - Resources/Module-healthsystem/consumables raw files/"

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# 1. DATA IMPORT AND CLEANING #
#########################################################################################
# --- 1.1 Import consumables availability data --- #
stkout_df = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv",
                        low_memory=False)

# Drop rows which can't be used in regression analysis
regsubset_cond1 = stkout_df['data_source'] == 'original_lmis_data'
regsubset_cond2 = stkout_df[
                      'fac_type_tlo'] == 'Facility_level_0'  # since only one facility from Mchinji reported in OpenLMIS
stkout_df_reg = stkout_df[regsubset_cond1 & ~regsubset_cond2]

# Clean some district names to match with master health facility registry
rename_districts = {
    'Nkhota Kota': 'Nkhotakota',
    'Nkhata bay': 'Nkhata Bay'
}
stkout_df['district'] = stkout_df['district'].replace(rename_districts)

# --- 1.2 Import GIS data --- #
fac_gis = pd.read_csv(path_to_files_in_the_tlo_dropbox / "gis_data/facility_distances.csv")

# --- 1.3 Merge cleaned LMIS data with GIS data --- #
consumables_df = pd.merge(stkout_df.drop(columns=['district', 'Unnamed: 0']), fac_gis.drop(columns=['Unnamed: 0']),
                          how='left', on='fac_name')
consumables_df.to_csv(path_to_files_in_the_tlo_dropbox / 'consumables_df.csv')
