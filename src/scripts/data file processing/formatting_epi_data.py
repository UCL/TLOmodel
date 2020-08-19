"""
This file processes the data from the Malawi EPI JRF
It modifies ResourceFile_EPI_summary district list to match the master district list:

pre-processing from the JRF Reports:
in excel remove the multiline entries in headers
find/replace: find alt 0010, replace ""
doesn't remove all so have to do some manual checks
remove any spaces or special characters in headers
change Nkhatabay / NkhataBay to Nkhata Bay

2010 and 2011 data are by region
2012 is compiled from the pdf report by district
2012 missing Hib estimates - assume same coverage as 2011

"""
from pathlib import Path

import numpy as np
import pandas as pd

resourcefilepath = Path("./resources")

epi = pd.read_csv(Path(resourcefilepath) / "ResourceFile_EPI_summary.csv")
master_district_list = pd.read_csv(Path(resourcefilepath) / "ResourceFile_District_Population_Data.csv")

districts = master_district_list.District

len(epi['District'].unique())  # 28
len(master_district_list['District'].unique())  # 32

# list(set(epi['District']).difference(master_district_list['District']))

print("Additional values in first list:", (set(epi['District']).difference(master_district_list['District'])))
print("Additional values in first list:", (set(master_district_list['District']).difference(epi['District'])))
# {'Lilongwe City', 'Mzuzu City', 'Blantyre City', 'Zomba City'}

print(epi['District'].unique())
print(master_district_list['District'].unique())

######################################################################
# these are the ones that need to match
# master = epi report
# Lilongwe City = Lilongwe
# "Blantyre City" = "Blantyre"
# "Zomba City"= "Zomba"
# "Mzuzu City" = "Mzimba"

# EPI reports are missing these districts. Assume same coverage throughout "new" district as in mapped district

######################################################################
# find which districts are missing in the EPI report
map_districts = (
    ('Lilongwe', 'Lilongwe City'),
    ('Blantyre', 'Blantyre City'),
    ('Zomba', 'Zomba City'),
    ('Mzimba', 'Mzuzu City')
)

# for each missing district in EPI report generate an identical row using the mapped district
# find lilongwe entry
# then create new entry called lilongwe city with same values as lilongwe

# for each entry in EPI mapping list:
# select all rows with that district
# copy those rows and replace EPI district with missing district (all same coverage values)
epi_formatted = epi

for i in np.arange(0, len(map_districts)):
    epi_district = map_districts[i][0]

    # select all entries for Lilongwe
    d1 = epi.loc[epi.District == epi_district].copy()
    # change the name to the mapped district name
    d1["District"] = map_districts[i][1]
    # generate entries for each year for the additional districts in the EPI report
    # add to EPI data
    epi_formatted = epi_formatted.append(d1)

len(epi_formatted['District'].unique())
print(epi_formatted['District'].unique())

epi_formatted.to_csv(Path(resourcefilepath) / 'ResourceFile_EPI_vaccine_coverage.csv')
