"""This script processes data that we have on the number of beds available for use in the Healthcare System and creates
the Resource File that is used by the BedDays class in the HealthSystem module:

* ResourceFile_Bed_Capacity.csv

"""

from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

resourcefilepath = Path('./resources')

path_to_dropbox = Path(
    '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/')  # <-- point to the TLO dropbox locally

# LOCATION OF INPUT FILE:
workingfile = (path_to_dropbox /
               '05 - Resources' /
               'Module-healthsystem' /
               'beds' /
               'extracted_data_on_beds.xlsx')

# TARGET OUTPUT FILE:
outputfile = resourcefilepath / "healthsystem" / "infrastructure_and_equipment" / "ResourceFile_Bed_Capacity.csv"


# %% Define the function that does the estimation
def estimate_beds_by_facility_id(beds_by_district: pd.Series, beds_by_level: pd.Series) -> pd.Series:
    """Use information on the numbers of bed by district and beds by level to impute the number at each Facility_ID."""

    original_number_of_beds_from_district = beds_by_district.sum()

    rename_districts = {
        'Mzimba North': 'Mzimba',
        'Mzimba South': 'Mzimba',
        'Chikhwawa': 'Chikwawa',
        'Chiradzulo': 'Chiradzulu'
    }
    beds_by_district = beds_by_district.rename(index=rename_districts)

    # Sum across level in case of duplicates
    beds_by_district = beds_by_district.groupby(level=0).sum()

    # Places where only an aggregate is given in the beds data, but a distinction with the city is needed in the model.
    # We solve this by attributing half of the bed capacity to the city and half to the remaining part of the district.
    cities = {
        'Lilongwe': 'Lilongwe City',
        'Blantyre': 'Blantyre City',
        'Zomba': 'Zomba City',
        'Mzimba': 'Mzuzu City'
    }

    total_for_district = beds_by_district.loc[cities.keys()].to_dict()
    beds_by_district.loc[cities.keys()] = pd.Series(total_for_district) * 0.5

    beds_by_district = beds_by_district.append(pd.Series(
        {_city: 0.5 * total_for_district[_district]
         for _district, _city in cities.items()
         }
    ))

    assert set(beds_by_district.index) == districts
    assert not beds_by_district.index.duplicated().any()

    pc_of_beds_by_district = beds_by_district/beds_by_district.sum()

    pc_of_beds_by_region = {
        _region: (
            beds_by_district[beds_by_district.index.isin(districts_in_region[_region])].sum() / beds_by_district.sum()
        )
        for _region in districts_in_region
    }

    assert np.isclose(1.0, pc_of_beds_by_district.sum())
    assert np.isclose(1.0, pd.Series(pc_of_beds_by_region).sum())
    assert original_number_of_beds_from_district == beds_by_district.sum()

    # Beds by Level
    original_number_of_beds_from_level = beds_by_level.sum()

    # Map the types of facilities to the Facility_Level
    beds_by_level.index = beds_by_level.index.map(map_to_level)

    # Sum across level in case of duplicates
    beds_by_level = beds_by_level.groupby(level=0).sum()
    assert beds_by_level.sum() == original_number_of_beds_from_level

    # Attribute to specific levels and districts in these proportions
    beds_by_fac_id = dict()
    for _fac_id, _info in fac_id.iterrows():

        if _info.Facility_Level == "0":
            _num_beds = 0

        elif _info.Facility_Level in district_level_facility_levels:
            _num_beds = pc_of_beds_by_district[_info.District] * beds_by_level[_info.Facility_Level]

        elif _info.Facility_Level == "3":
            _num_beds = pc_of_beds_by_region[_info.Region] * beds_by_level[_info.Facility_Level]

        else:
            # National level
            _num_beds = 0.0

        beds_by_fac_id[_fac_id] = _num_beds

    x = pd.Series(data=beds_by_fac_id, name='num_beds')
    x.index.name = 'Facility_ID'

    # Do the logical checks on non-integer numbers so that they don't fail due to rounding errors
    assert set(x.index) == set(mfl['Facility_ID'])
    assert not x.index.duplicated().any()
    assert not pd.isnull(x).any()
    assert x.sum() == beds_by_level.sum()

    # Return a integers
    return x.round().astype(int).to_dict()



# %% Definitional things

# The Facility_Level to which each type of facility name attaches.
map_to_level = {
    'central_hospital': "3",
    'district_hospital': "2",
    'rural/_community_hosp': "1b",
    'other_hospital': "1b",
    'health_centre': "1a"
}

# Districts and Regions:
popdata = pd.read_csv(resourcefilepath / "demography" / "ResourceFile_Population_2010.csv")
districts = set(popdata['District'])

districts_in_region = defaultdict(set)
for _district in popdata[['District', 'Region']].drop_duplicates().itertuples():
    districts_in_region[_district.Region].add(_district.District)

# Master Facilities List
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
fac_id = mfl.set_index('Facility_ID')[['Region', 'District', 'Facility_Level']]
district_level_facility_levels = ("1a", "1b", "2")

# Bed types
bed_types = {
    'delivery_bed': {'kangaroo_beds', 'delivery_beds'},
    'maternity_bed': {'maternity_beds'},
    'general_bed': None,  # <-- This will be "all beds" minus all other types of defined bed.
}


# %% Load Working files
tab501 = pd.read_excel(workingfile, sheet_name='Table_5-01')

# basic cleaning up of names
clean_up_index = lambda _cols: _cols.str.strip(" ").str.replace(" ", "_").str.lower()
tab501.columns = clean_up_index(tab501.columns)

# Get number of beds, split by District
beds_by_district = tab501.iloc[19:][['', 'all_beds', 'maternity_beds', 'kangaroo_beds', 'delivery_beds']]
beds_by_district = beds_by_district.set_index(beds_by_district.columns[0])
beds_by_district.index = beds_by_district.index.str.strip()

# Get numbers of beds, split by Facility Level / Type
beds_by_level = tab501.iloc[3:8][['', 'all_beds', 'maternity_beds', 'kangaroo_beds', 'delivery_beds']]
beds_by_level = beds_by_level.set_index(beds_by_level.columns[0])
beds_by_level.index = clean_up_index(beds_by_level.index)

# Combine types of beds to get beds in the types needed for the model
def compute_bed_number_custom_type(_df: pd.DataFrame, _bed_types: dict) -> pd.DataFrame:
    """Compute the Custom Bed Types"""
    df = pd.DataFrame(index=_df.index)
    for _custom_type, _types_to_combine in _bed_types.items():
        if _types_to_combine is not None:
            df[_custom_type] = _df[list(_types_to_combine)].sum(axis=1)
        else:
            # None means to let this be "all_beds" minus any other type that has been assigned to a custom type.
            df[_custom_type] = (
                _df['all_beds'] - _df[{x for v in _bed_types.values() if v is not None for x in v}].sum(axis=1)
            )

    assert (df.sum(axis=1) == _df['all_beds']).all()

    df = df.clip(lower=0.0)  # One place has more "maternity beds" than "all beds"!
    assert (df >= 0).all().all()
    return df.round().astype(int)

beds_by_district = compute_bed_number_custom_type(beds_by_district, bed_types)
beds_by_level = compute_bed_number_custom_type(beds_by_level, bed_types)

# These two data tables (for district and for level) _should_ indicate the same total number of bed of each type.
# They do not! But the difference is small. We rely on the number by level as the "truth".
for _bed_type in bed_types:
    print(f"{_bed_type}: difference={beds_by_level[_bed_type].sum() - beds_by_district[_bed_type].sum()}")
    # assert 0 == beds_by_level[_bed_type].sum() - beds_by_district[_bed_type].sum()

# Compute the number of beds in each custom type:
num_beds = {
    _bed_type: estimate_beds_by_facility_id(
        beds_by_district=beds_by_district[_bed_type],
        beds_by_level=beds_by_level[_bed_type]
    )
    for _bed_type in bed_types
}

# Check that we have the right total number of beds (after allowing for some inconsistency in the input data.)
assert abs(21_407 - pd.DataFrame(num_beds).sum().sum()) < 20

# Save:
pd.DataFrame(num_beds).to_csv(outputfile, index_label='Facility_ID')


# %% Cross-check with Table 38, which gives numbers of beds per 10k population

# todo
# tab38 = pd.read_excel(workingfile, sheet_name='Table_38')
# tab38.columns = clean_up_index(tab38.columns)

# todo - plot number of 10k in these estimates and table 38

