"""This file uses the results of the results of running `impact_of_consumable_availability_intervention.py`
tob extract summary results for the manuscript - "Rethinking economic evaluation of
system level interventions.

I plan to run the simulation for a short period of 5 years (2020 - 2025) because
holding the consumable availability constant in the short run would be more justifiable
than holding it constant for a long period.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    summarize,
    write_log_to_excel,
    parse_log_file,
)

outputspath = Path('./outputs/')

# %% Gathering basic information

# Find results_folder associated with a given batch_file and get most recent
#results_folder = get_scenario_outputs('impact_of_consumable_availability_intervention.py', outputspath)[-1]
results_folder = Path(outputspath/ 'impact_of_consumables_availability_intervention-2023-05-09T210307Z/')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


# %% Extracting results from run

# 1. DALYs averted
#-----------------------------------------
# 1.1 Difference in total DALYs accrued
def extract_total_dalys(results_folder):

    def extract_dalys_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": df.drop(['date', 'sex', 'age_range', 'year'], axis = 1).sum().sum()})

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=extract_dalys_total,
        do_scaling=True
    )

total_dalys_accrued = extract_total_dalys(results_folder)

# 1.2 (Optional) Difference in total DALYs accrued by disease
def _extract_dalys_by_disease(_df: pd.DataFrame) -> pd.Series:
    """Construct a series with index disease and value of the total of DALYS (stacked) from the
    `dalys_stacked` key logged in `tlo.methods.healthburden`.
    N.B. This limits the time period of interest to 2010-2019"""
    _, calperiodlookup = make_calendar_period_lookup()

    return _df.loc[(_df['year'] >=2009) & (_df['year'] < 2012)]\
             .drop(columns=['date', 'sex', 'age_range', 'year'])\
             .sum(axis=0)

dalys_extracted_by_disease = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked",
    custom_generate_series=_extract_dalys_by_disease,
    do_scaling=True
)
#? some NCDs see a decline in DALYs accrued

dalys_by_disease_summarized = summarize(dalys_extracted_by_disease)
print(dalys_by_disease_summarized[[(0,  'mean'),(1,  'mean')]])
# dalys_by_disease_summarized.to_csv(outputspath / 'dalys_by_disease.csv')

# 2. Services delivered
#-----------------------------------------
# 2.1 Total number of HSIs "completed"
hsi_alternatescenario = load_pickled_dataframes(results_folder, draw=0, run=0)['tlo.methods.healthsystem.summary']['HSI_Event']
hsi_default = load_pickled_dataframes(results_folder, draw=1, run=1)['tlo.methods.healthsystem.summary']['HSI_Event']

# ? Is there data to indicate whether the HSI was successfully delivered?

# 2.2 Number of HSIs completed by disease
# use 'TREATMENT_ID' on above data

# 3. Resource use / Mechanisms of impact
#-----------------------------------------
# 3.1 Proportion of HSIs for which consumable was recorded as not available
# Load pickle files
consumable_alternatescenario = load_pickled_dataframes(results_folder, draw=0, run=0)['tlo.methods.healthsystem']['Consumables']
consumable_default = load_pickled_dataframes(results_folder, draw=1, run=1)['tlo.methods.healthsystem']['Consumables']

# ? How to load from multiple runs?
# ? What does tlo.methods.healthysystem.summary provide?

# ? Use the following columns for estimates
consumable_alternatescenario['Item_NotAvailable']

# ? What is the best way to analyse the data in the dictionary?
# Create a dataframe with individual colunms for each lement in the dictionary
# list_of_item_codes = list(healthsystem_usage_alternatescenario['Consumables']['Item_Available'].keys())

# 3.2 Proportion of HSIs for which consumable was recorded as not available by disease

# 3.3 Proportion of staff time demanded (This could also be measured as number of minutes of
# staff time required under the two scenarios)
# Load pickle files
staffusage_alternatescenario = load_pickled_dataframes(results_folder, draw=0, run=0)['tlo.methods.healthsystem']['Capacity']
staffusage_default = load_pickled_dataframes(results_folder, draw=1, run=1)['tlo.methods.healthsystem']['Capacity']

staffusage_alternatescenario['Frac_Time_Used_Overall']
staffusage_default['Frac_Time_Used_Overall']

# ? How should staff time be summarised?

# 3.4 Proportion of staff time demanded by disease and *facility level*
staffusage_alternatescenario['Frac_Time_Used_By_Facility_ID']
staffusage_default['Frac_Time_Used_By_Facility_ID']


# Extract excel file
#parse_log_file(results_folder, level: int = logging.INFO)
log_dataframes ={
            'healthsystem': {'Consumables': pd.DataFrame(),
                             'Capacity': pd.DataFrame()
                             },
            'healthburden': {'dalys_stacked': pd.DataFrame(),
                             'dalys': pd.DataFrame()
                             }
        }
# ? How should _metadata be specified above?

write_log_to_excel("file.xlsx", log_dataframes)


'''
# Scratch code
#-----------------------
# DALYs by age group and time
def _extract_dalys_by_age_group_and_time_period(_df: pd.DataFrame) -> pd.Series:
    """Construct a series with index age-rage/time-period and value of the total of DALYS (stacked) from the
    `dalys_stacked` key logged in `tlo.methods.healthburden`."""
    _, calperiodlookup = make_calendar_period_lookup()

    return _df.assign(
                Period=lambda x: x['year'].map(calperiodlookup).astype(make_calendar_period_type()),
            ).set_index('Period')\
             .drop(columns=['date', 'sex', 'age_range', 'year'])\
             .groupby(axis=0, level=0)\
             .sum()\
             .sum(axis=1)


#-----------------------
dalys_extracted = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked",
    custom_generate_series=_extract_dalys_by_age_group_and_time_period,
    do_scaling=True
)

dalys_summarized = summarize(dalys_extracted)
dalys_summarized = dalys_summarized.loc[dalys_summarized.index.isin(('2010-2014'))]

# Load the pickle file
file = results_folder / '0/0/tlo.methods.healthsystem.pickle'
with open(file, 'rb') as f:
    healthsystem_usage_alternatescenario = pickle.load(f)

file = results_folder / '1/0/tlo.methods.healthsystem.pickle'
with open(file, 'rb') as f:
    healthsystem_usage_default = pickle.load(f)

#-----------------------
# Extract excel file
# Created a new function because I wasn't sure how metadata needs to be specified
def write_log_to_excel_new(filename, log_dataframes):
    """Takes the output of parse_log_file() and creates an Excel file from dataframes"""
    sheets = list()
    sheet_count = 0
    for module, key_df in log_dataframes.items():
        for key, df in key_df.items():
            sheet_count += 1
            sheets.append([module, key, sheet_count])

    writer = pd.ExcelWriter(filename)
    index = pd.DataFrame(data=sheets, columns=['module', 'key', 'sheet'])
    index.to_excel(writer, sheet_name='Index')

    sheet_count = 0
    for module, key_df in log_dataframes.items():
        for key, df in key_df.items():
            sheet_count += 1
            df.to_excel(writer, sheet_name=f'Sheet {sheet_count}')
    writer.close() # AttributeError: 'OpenpyxlWriter' object has no attribute 'save'

# Write log to excel
#parse_log_file(results_folder, level: int = logging.INFO)
log_dataframes ={
            'healthsystem': {'Consumables': pd.DataFrame(),
                             'Capacity': pd.DataFrame()
                             },
            'healthburden': {'dalys_stacked': pd.DataFrame(),
                             'dalys': pd.DataFrame()
                             }
        }
#write_log_to_excel_new(results_folder, log_dataframes)

'''
