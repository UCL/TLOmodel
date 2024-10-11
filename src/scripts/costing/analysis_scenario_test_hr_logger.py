import argparse
from pathlib import Path
from tlo import Date
from collections import Counter, defaultdict

import calendar
import datetime
import os

import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    summarize,
    create_pickles_locally,
    parse_log_file
)

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

outputfilepath = Path('./outputs/')
resourcefilepath = Path("./resources")
results_folder = get_scenario_outputs('scenario_test_hr_logger.py', outputfilepath)[0]
log = load_pickled_dataframes(results_folder)
final_year_of_simulation = max(log['tlo.simulation']['info']['date']).year
first_year_of_simulation = min(log['tlo.simulation']['info']['date']).year

def expand_capacity_by_officer_type_and_facility_level(_df: pd.Series) -> pd.Series:
    """Summarise the parsed logged-key results for one draw (as dataframe) into a pd.Series."""
    _df = _df.set_axis(_df['date'].dt.year).drop(columns=['date'])
    _df.index.name = 'year'
    return pd.Series(
        data=_df.sum()
    )

staff_count_by_cadre_and_level = extract_results(
    Path(results_folder),
    module='tlo.methods.healthsystem.summary',
    key='number_of_hcw_staff',
    custom_generate_series=expand_capacity_by_officer_type_and_facility_level,
    do_scaling=True,
)

# Staff count as per the logger
staff_count_by_cadre_and_level= staff_count_by_cadre_and_level.reset_index()
staff_count_by_cadre_and_level.columns = ['_'.join([str(i) for i in col]).strip() for col in staff_count_by_cadre_and_level.columns]
staff_count_by_cadre_and_level['Facility_Level'] = staff_count_by_cadre_and_level['index_'].str.split('_').str[1]
staff_count_by_cadre_and_level['Cadre'] = staff_count_by_cadre_and_level['index_'].str.split('_').str[3]
total_staff_count_by_cadre = staff_count_by_cadre_and_level.groupby('Cadre')['0_0'].sum()/(final_year_of_simulation-first_year_of_simulation)

# Staff count as per the resourcefile
actual_staff_count = pd.read_csv(resourcefilepath / 'healthsystem' / 'human_resources' / 'funded_plus' /
                'ResourceFile_Daily_Capabilities.csv')
total_staff_count_by_cadre_actual = actual_staff_count.groupby('Officer_Category')['Staff_Count'].sum()
assert total_staff_count_by_cadre_actual['Clinical'] == total_staff_count_by_cadre['Clinical']
