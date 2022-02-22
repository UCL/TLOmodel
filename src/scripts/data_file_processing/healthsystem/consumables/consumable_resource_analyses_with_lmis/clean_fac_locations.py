"""
This script generates a file containing the locations of facilities in the LMIS data:
* ResourceFile_Facility_locations.csv

Inputs:
Dropbox location - ~05 - Resources/Module-healthsystem/consumables raw files/gis_data/LMISFacilityLocations_raw.xlsx

"""
import datetime
# Import Statements and initial declarations
from pathlib import Path

import numpy as np
import pandas as pd
# import googlemaps as gmaps
import requests

# import json

path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    'C:/Users/sm2511/Dropbox/Thanzi la Onse')
path_to_files_in_the_tlo_dropbox = path_to_dropbox / "05 - Resources/Module-healthsystem/consumables raw files/"

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define a pathway to the data folder (note: currently outside the TLO model directory)
# remember to set working directory to TLOmodel/
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"

# Load raw GIS dataset and clean column names
fac_gis = pd.read_excel(open(path_to_files_in_the_tlo_dropbox / 'gis_data/LMISFacilityLocations_raw.xlsx',
                             'rb'), sheet_name='final_gis_data')
fac_gis = fac_gis.rename(
    columns={'LMIS Facility List': 'fac_name', 'OWNERSHIP': 'fac_owner', 'TYPE': 'fac_type', 'STATUS': 'fac_status',
             'ZONE': 'zone', 'DISTRICT': 'district', 'DATE OPENED': 'open_date', 'LATITUDE': 'lat',
             'LONGITUDE': 'long'})
# Create a new column providing source of GIS data
fac_gis['gis_source'] = ""

# Store unique district names
districts = fac_gis['district'].unique()

# Preserve rows with missing or incorrect location data in order to derive GIS data using googlemaps API
cond1 = fac_gis['lat'] > -8.5
cond2 = fac_gis['lat'] < -17.5
cond3 = fac_gis['long'] > 36.5
cond4 = fac_gis['long'] < 32.5
conda = cond1 | cond2 | cond3 | cond4  # outside Malawi's boundaries
fac_gis_noloc = fac_gis[fac_gis.lat.isna() | conda]
fac_gis_noloc = fac_gis_noloc.reset_index()
fac_gis_noloc = fac_gis_noloc.drop(columns='index')

# Edit data source
cond_originalmhfr = fac_gis.lat.notna() & ~conda
fac_gis.loc[cond_originalmhfr, 'gis_source'] = 'Master Health Facility Registry'
cond_manual = fac_gis['manual_entry'].notna()
fac_gis.loc[cond_manual, 'gis_source'] = 'Manual google search'

fac_gis_clean = fac_gis[~conda & fac_gis.lat.notna()]  # save clean portion of raw data to be appended later

# Use googlemaps package to obtain GIS coordinates using facility names
GCODE_URL = 'https://maps.googleapis.com/maps/api/geocode/json?'
GCODE_KEY = ""  # PLaceholder to enter googlemaps API


def reverse_gcode(location):
    location = str(location).replace(' ', '+')
    nav_req = 'address={}&key={}'.format(location, GCODE_KEY)
    request = GCODE_URL + nav_req
    result = requests.get(request)
    data = result.json()
    status = data['status']

    geo_location = {}
    if str(status) == "OK":
        sizeofjson = len(data['results'][0]['address_components'])
        for i in range(sizeofjson):
            sizeoftype = len(data['results'][0]['address_components'][i]['types'])
            if sizeoftype == 3:
                geo_location[data['results'][0]['address_components'][i]['types'][2]] = \
                    data['results'][0]['address_components'][i]['long_name']

            else:
                if data['results'][0]['address_components'][i]['types'][0] == 'administrative_area_level_1':
                    geo_location['state'] = data['results'][0]['address_components'][i]['long_name']

                elif data['results'][0]['address_components'][i]['types'][0] == 'administrative_area_level_2':
                    geo_location['city'] = data['results'][0]['address_components'][i]['long_name']
                    geo_location['town'] = geo_location['city']

                else:
                    geo_location[data['results'][0]['address_components'][i]['types'][0]] = \
                        data['results'][0]['address_components'][i]['long_name']

        formatted_address = data['results'][0]['formatted_address']
        geo_location['lat'] = data['results'][0]['geometry']['location']['lat']
        geo_location['lang'] = data['results'][0]['geometry']['location']['lng']
        geo_location['formatted_address'] = formatted_address

        return geo_location


for i in range(len(fac_gis_noloc)):
    try:
        print("Processing facility", fac_gis_noloc['fac_name'][i])
        geo_info = reverse_gcode(fac_gis_noloc['fac_name'][i] + 'Malawi')
        fac_gis_noloc['lat'][i] = geo_info['lat']
        fac_gis_noloc['long'][i] = geo_info['lang']
        fac_gis_noloc['district'][i] = geo_info['city']
        fac_gis_noloc['gis_source'][i] = 'Google maps geolocation'
    except ValueError:
        pass

# Drop incorrect GIS coordinates from the above generated dataset
conda = fac_gis_noloc.district.isin(districts)  # districts not from Malawi
cond1 = fac_gis_noloc['lat'] > -8.5
cond2 = fac_gis_noloc['lat'] < -17.5
cond3 = fac_gis_noloc['long'] > 36.5
cond4 = fac_gis_noloc['long'] < 32.5
condb = cond1 | cond2 | cond3 | cond4  # outside Malawi's boundaries

fac_gis_noloc.loc[~conda | condb, 'lat'] = np.nan
fac_gis_noloc.loc[~conda | condb, 'long'] = np.nan
fac_gis_noloc.loc[~conda | condb, 'district'] = np.nan

# Append newly generated GIS information to the raw data
fac_gis = fac_gis_noloc.append(fac_gis_clean)

# Drop incorrect GIS coordinates based on later comparison with district data from LMIS
list_of_incorrect_locations = ['Bilal Clinic', 'Biliwiri Health Centre', 'Chilonga Health care Health Centre',
                               'Diamphwi Health Centre', 'Matope Health Centre (CHAM)', 'Nambazo Health Centre',
                               'Nkhwayi Health Centre', 'Nsambe Health Centre (CHAM)', 'Padley Pio Health Centre',
                               'Phanga Health Centre', 'Somba Clinic', "St. Martin's Molere Health Centre CHAM",
                               'Ngapani Clinic', 'Mulungu Alinafe Clinic', 'Mdeza Health Centre',
                               'Matandani Health Centre (CHAM)',
                               'Sunrise Clinic', 'Sucoma Clinic']
cond = fac_gis.fac_name.isin(list_of_incorrect_locations)
fac_gis.loc[cond, 'lat'] = np.nan
fac_gis.loc[cond, 'long'] = np.nan
fac_gis.loc[cond, 'gis_source'] = np.nan
fac_gis.loc[cond, 'district'] = np.nan

# Extract into .csv
fac_gis.to_csv(path_for_new_resourcefiles / "ResourceFile_Facility_locations.csv")
