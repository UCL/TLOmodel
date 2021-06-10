"""
This is a script to process the data from the GBD to create the ResourceFiles used for model running and calibration checks.

It reads in the files that were downloaded externally and saves them as ResourceFiles in the `resources` directory:
    resources/gbd/

The following files are created:

* 'ResourceFile_Deaths_And_DALYS_GBD2019.csv': the GBD 2019 data, with some light fomatting done to aid use.

* ResourceFile_TotalDeaths_GBD: all-cause deaths by age/sex/year (in standard groups): used in calibrations

todo:
* ResourceFile_CausesOfDeath_GBD: used during simulation by 'Demography' module to construct the death rates used in the
 `OtherDeathPoll`
"""

from pathlib import Path

import numpy as np
import pandas as pd


from tlo.analysis.utils import (
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
)

path_for_saved_files = Path("./resources/gbd")

# %%
# *** USE OF THE GBD DATA ****

# GBD working file: downoaded 22/11/20. This is the version of the data from "GBD 2019"

permalink = 'http://ghdx.healthdata.org/gbd-results-tool?params=gbd-api-2019-permalink/ac9e2a238375d6f6ddec288f174646b7'
gbd_working_file = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/HealthBurden Module/Daly and deaths by cause estimates/IHME-GBD_2019_DATA-1db25232-1/IHME-GBD_2019_DATA-1db25232-1.csv'

# %%
(__tmp__, calendar_period_lookup) = make_calendar_period_lookup()
make_calendar_type = make_calendar_period_type()
(__tmp__, calendar_period_lookup) = make_age_grp_lookup()
age_grp_type= make_age_grp_types()

# %% Load and do light formatting on the GBD dataset overall

gbd = pd.read_csv(gbd_working_file)

# 0) Do some basic processing of the file:

# Reformat sex variable
gbd['Sex'] = gbd['sex_id'].replace({1: 'M', 2: 'F'})

# Rename Year variable
gbd.rename(columns={'year': 'Year'}, inplace=True)

# Reformat the age-groups: GBD-style age-groups (seperating out <1year and 1-4 year-olds)
gbd['age_name'] = gbd['age_name']\
    .str.replace('to', '-')\
    .str.replace('95 plus', '95+')\
    .str.replace(' ', '')
gbd = gbd.drop(gbd.index[gbd['age_name'] == 'AllAges'])
gbd = gbd.rename(columns={'age_name': 'Age_Grp_GBD'})

# standard age-group style (0-4, 5-9, etc)
gbd['Age_Grp'] = gbd['Age_Grp_GBD']\
        .str.replace('1-4', '0-4')\
        .str.replace('<1year', '0-4')

# Add Period information:
gbd['Period'] = gbd['Year'].map(calendar_period_lookup)

# Rename the 'variants'
gbd = gbd.rename(columns={
    'val': 'GBD_Est',
    'upper': 'GBD_Upper',
    'lower': 'GBD_Lower'
})

# drop ununsed columns
gbd = gbd[['measure_name',
           'Age_Grp',
           'Age_Grp_GBD',
           'Sex',
           'Year',
           'Period',
           'cause_name',
           'cause_id',
           'GBD_Est',
           'GBD_Upper',
           'GBD_Lower'
           ]]

# checks
assert not pd.isnull(gbd).any().any()

# %% Save all outputs as `ResourceFile_Deaths_And_DALYS_GBD2019.csv`
gbd.to_csv(path_for_saved_files / 'ResourceFile_Deaths_And_DALYS_GBD2019.csv', index=False)


# %% Make: ResourceFile_TotalDeaths_GBD
# Output Deaths (all-cause) using standard Age-Grps
gbd_deaths = gbd.loc[gbd['measure_name'] == 'Deaths'].copy().reset_index(drop=True)
gbd_deaths = gbd_deaths.groupby(by=['Year', 'Sex', 'Age_Grp'], as_index=False)[['GBD_Est', 'GBD_Lower', 'GBD_Upper']].sum()
gbd_deaths = gbd_deaths.melt(id_vars=['Year', 'Sex', 'Age_Grp'], var_name='Variant', value_name='Count')
gbd_deaths.to_csv(path_for_saved_files / 'ResourceFile_TotalDeaths_GBD.csv', index=False)

# %% Make: ResourceFile_CausesOfDeath_GBD
#todo
