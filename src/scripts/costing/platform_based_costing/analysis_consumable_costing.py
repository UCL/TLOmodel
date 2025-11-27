"""Produce outputs for PLatform-based costing of consumables.
This analysis was performed for MOH's 2025 Platform-based Costing Exercise
"""

import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
from ast import literal_eval

from scripts.costing.costing_validation import consumables_dispensed_summary
from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    compute_summary_statistics
)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/')
figurespath = Path('./outputs/platform_based_costing/')
path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"
if not os.path.exists(figurespath):
    os.makedirs(figurespath)

from tlo.analysis.utils import create_pickles_locally

# Load result files
# ------------------------------------------------------------------------------------------------------------------
results_folder = get_scenario_outputs('consumables_costing-2025-11-25T213613Z.py', outputfilepath)[0]

#log = pd.read_csv(results_folder1 / "0" / "0" / 'impact_of_consumables_availability__2025-11-25T120830.log', sep="\t")
#create_pickles_locally(scenario_output_dir = "outputs/consumables_costing-2025-11-25T213613Z")

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0)  # look at one log (so can decide what to extract)
params = extract_params(results_folder)
info = get_scenario_info(results_folder)

# Declare default parameters for cost analysis
# ------------------------------------------------------------------------------------------------------------------
# Period relevant for costing
TARGET_PERIOD = (Date(2010, 1, 1), Date(2030, 12, 31))  # This is the period that is costed
relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
list_of_years_for_plot = list(range(2010, 2031))
number_of_years_costed = relevant_period_for_costing[1] - 2023 + 1

# Scenarios
cost_scenarios = {0: "Actual", 1: "Perfect consumable availability"}

# Costing parameters
discount_rate = 0
#log['tlo.methods.healthsystem.summary'].keys()
#log['tlo.methods.healthsystem']['Consumables'].to_csv(figurespath / 'draft_cons_log_0.csv')
#log['tlo.methods.healthsystem.summary']['Consumables'].columns

# Extract consumables dispensed data
def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

def get_quantity_of_consumables_dispensed(results_folder):
    def convert_str_to_dict(row_entry):
        if isinstance(row_entry, str):
            row_entry_dict = literal_eval(row_entry)
        else:
            row_entry_dict = row_entry
        return row_entry_dict

    def get_counts_of_items_requested(_df):
        _df = drop_outside_period(_df)
        # Dict with date and Treatment_ID
        counts_used = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        counts_notavail = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for _, row in _df.iterrows():
            date = row['date']
            treatment = row['TREATMENT_ID']

            # Used Items
            for item, num in convert_str_to_dict(row['Item_Used']).items():
                counts_used[date][treatment][item] += num

            # Not Available Items
            for item, num in convert_str_to_dict(row['Item_NotAvailable']).items():
                counts_notavail[date][treatment][item] += num

        # Convert nested dicts → MultiIndex Series

        # Used
        used_records = []
        for date, tdict in counts_used.items():
            for treatment, idict in tdict.items():
                for item, num in idict.items():
                    used_records.append((date, treatment, item, 'Used', num))

        used_df = pd.DataFrame(used_records,
                               columns=['date', 'TREATMENT_ID', 'item', 'status', 'value'])

        # Not Available
        na_records = []
        for date, tdict in counts_notavail.items():
            for treatment, idict in tdict.items():
                for item, num in idict.items():
                    na_records.append((date, treatment, item, 'NotAvailable', num))

        notavail_df = pd.DataFrame(na_records,
                                   columns=['date', 'TREATMENT_ID', 'item', 'status', 'value'])

        # Combine
        combined = pd.concat([used_df, notavail_df], axis=0)

        # Convert to series with MultiIndex
        combined_series = combined.set_index(['date', 'TREATMENT_ID', 'item', 'status'])['value']

        return combined_series

    # Extract results using your existing pipeline
    cons_req = extract_results(
        results_folder,
        module='tlo.methods.healthsystem',
        key='Consumables',
        custom_generate_series=get_counts_of_items_requested,
        do_scaling=True
    )

    # Only keep 'Used'
    cons_dispensed = cons_req.xs("Used", level='status')

    return cons_dispensed

idx = pd.IndexSlice

#info, years, TARGET_PERIOD = load_simulation_metadata(results_folder)

consumables_dispensed = get_quantity_of_consumables_dispensed(results_folder)

consumables_dispensed_summary = compute_summary_statistics(consumables_dispensed, central_measure = 'median')

consumables_dispensed_summary = consumables_dispensed_summary.reset_index()
consumables_dispensed_summary[idx['year']] = pd.to_datetime(
    consumables_dispensed_summary[idx['date']]).dt.year

#consumables_dispensed_summary = consumables_dispensed_summary.rename(columns={
#    ('date',''): 'date',
#    ('TREATMENT_ID',''): 'TREATMENT_ID',
#    ('item',''): 'item'
#})
# Add consumable name
consumables_dict = \
    pd.read_csv(path_for_consumable_resourcefiles / 'ResourceFile_Consumables_Items_and_Packages.csv', low_memory=False,
                encoding="ISO-8859-1")[['Items', 'Item_Code']]
    consumables_dict = dict(zip(consumables_dict['Item_Code'], consumables_dict['Items']))
consumables_dispensed_summary[idx['item_name']] = consumables_dispensed_summary[idx['item']].map(consumables_dict)

consumables_dispensed_summary.to_csv(figurespath / 'sample_output.csv')

"""
# Fix column names
df.columns = df.columns.set_names(['draw', 'run'])

# Make date/TREATMENT_ID/item ordinary columns
df = df.rename(columns={
    ('date',''): 'date',
    ('TREATMENT_ID',''): 'TREATMENT_ID',
    ('item',''): 'item'
})
df = df.set_index(['date', 'TREATMENT_ID', 'item'])
"""
summary_list = []

for idx, row in df.iterrows():
    # row is a Series with MultiIndex columns (draw, run)
    row_df = row.to_frame().T  # convert to 1-row DataFrame
    stats = compute_summary_statistics(row_df)

    # stats will have MultiIndex columns (draw, stat)
    # we want to attach back the index
    stats.insert(0, 'date', idx[0])
    stats.insert(1, 'TREATMENT_ID', idx[1])
    stats.insert(2, 'item', idx[2])

    summary_list.append(stats)

summary_df = pd.concat(summary_list, ignore_index=True)

consumables_dispensed = consumables_dispensed.reset_index().rename(
    columns=['date', 'TREATMENT_ID', 'item_code'])
consumables_dispensed[idx['Item_Code']] = pd.to_numeric(consumables_dispensed[idx['Item_Code']])

consumables_dispensed_summary = compute_summary_statistics(consumables_dispensed, central_measure = 'median')
consumables_dispensed_summary = consumables_dispensed_summary.reset_index()
consumables_dispensed_summary[idx['year']] = pd.to_datetime(
    consumables_dispensed_summary[idx['date']]).dt.year

consumables_dict = \
    pd.read_csv(path_for_consumable_resourcefiles / 'ResourceFile_Consumables_Items_and_Packages.csv', low_memory=False,
                encoding="ISO-8859-1")[['Items', 'Item_Code']]
consumables_dict = dict(zip(consumables_dict['Item_Code'], consumables_dict['Items']))

consumables_dispensed_summary[idx['item_name']] = consumables_dispensed_summary[idx['item']].map(consumables_dict)

consumables_dispensed_summary.to_csv(figurespath / 'sample_output.csv')
