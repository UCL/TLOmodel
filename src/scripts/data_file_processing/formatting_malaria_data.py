"""
This file processes the data from the malaria modelling outputs
It modifies ResourceFile_malaria_ClinInc.csv, ResourceFile_malaria_InfInc,csv and ResourceFile_malaria_SevInc.csv
 district list to match the master district list
"""

from pathlib import Path

import numpy as np
import pandas as pd

resourcefilepath = Path("./resources")

mal_clin = pd.read_csv(Path(resourcefilepath) / "malaria/ResourceFile_malaria_ClinInc.csv")
mal_inf = pd.read_csv(Path(resourcefilepath) / "malaria/ResourceFile_malaria_InfInc.csv")
mal_sev = pd.read_csv(Path(resourcefilepath) / "malaria/ResourceFile_malaria_SevInc.csv")

master_district_list = pd.read_csv(Path(resourcefilepath) / "ResourceFile_District_Population_Data.csv")

districts = master_district_list.District

# check how many unique districts are in file
len(mal_clin['admin'].unique())  # 27
len(mal_inf['admin'].unique())  # 27
len(mal_sev['admin'].unique())  # 27

len(master_district_list['District'].unique())  # 32

# find which districts are missing between the two sets
print("Additional values in first list:", (set(mal_clin['admin']).difference(master_district_list['District'])))
print("Additional values in first list:", (set(master_district_list['District']).difference(mal_clin['admin'])))
#  {'Nkhata Bay', 'Mzuzu City', 'Zomba City', 'Lilongwe City', 'Blantyre City'}

print(mal_clin['admin'].unique())
print(master_district_list['District'].unique())

######################################################################
# find which districts are missing in the malaria resource files
map_districts = (
    ('Lilongwe', 'Lilongwe City'),
    ('Blantyre', 'Blantyre City'),
    ('Zomba', 'Zomba City'),
    ('Mzimba', 'Mzuzu City'),
    ('Mzimba', 'Nkhata Bay')
)
######################################################################
# format ResourceFile_malaria_ClinInc.csv
# for each entry in malaria resource file mapping list:
# select all rows with that district
# copy those rows and replace malaria resource file district with missing district (all same coverage values)
mal_clin_formatted = mal_clin

for i in np.arange(0, len(map_districts)):
    mal_district = map_districts[i][0]

    # select all entries for Lilongwe
    d1 = mal_clin.loc[mal_clin.admin == mal_district].copy()
    # change the name to the mapped district name
    d1["admin"] = map_districts[i][1]
    # generate entries for each year for the additional districts in the malaria resource file
    # add to malaria resource file
    mal_clin_formatted = mal_clin_formatted.append(d1)

# check now have same number unique districts as master file
len(mal_clin_formatted['admin'].unique())
print(mal_clin_formatted['admin'].unique())

# output edited resource file to resources folder
mal_clin_formatted.to_csv(Path(resourcefilepath) / 'ResourceFile_malaria_ClinInc_expanded.csv')

######################################################################
# format ResourceFile_malaria_InfInc.csv
mal_inf_formatted = mal_inf

for i in np.arange(0, len(map_districts)):
    mal_district = map_districts[i][0]

    # select all entries for Lilongwe
    d1 = mal_inf.loc[mal_inf.admin == mal_district].copy()
    # change the name to the mapped district name
    d1["admin"] = map_districts[i][1]
    # generate entries for each year for the additional districts in the malaria resource file
    # add to malaria resource file
    mal_inf_formatted = mal_inf_formatted.append(d1)

len(mal_inf_formatted['admin'].unique())
print(mal_inf_formatted['admin'].unique())

# output edited resource file to resources folder
mal_inf_formatted.to_csv(Path(resourcefilepath) / 'ResourceFile_malaria_InfInc_expanded.csv')

######################################################################
# format ResourceFile_malaria_SevInc.csv
mal_sev_formatted = mal_sev

for i in np.arange(0, len(map_districts)):
    mal_district = map_districts[i][0]

    # select all entries for Lilongwe
    d1 = mal_sev.loc[mal_sev.admin == mal_district].copy()
    # change the name to the mapped district name
    d1["admin"] = map_districts[i][1]
    # generate entries for each year for the additional districts in the malaria resource file
    # add to malaria resource file
    mal_sev_formatted = mal_sev_formatted.append(d1)

len(mal_sev_formatted['admin'].unique())
print(mal_sev_formatted['admin'].unique())

# output edited resource file to resources folder
mal_sev_formatted.to_csv(Path(resourcefilepath) / 'ResourceFile_malaria_SevInc_expanded.csv')
