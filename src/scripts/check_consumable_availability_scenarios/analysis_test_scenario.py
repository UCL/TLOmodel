'''
This script analyses the outputs of the following scenarios -
src/scripts/check_consumable_availability_scenarios/test_scenario.py
'''

import argparse
from pathlib import Path
import textwrap
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import Counter, defaultdict
import seaborn as sns
import squarify

from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    extract_results,
    get_color_cause_of_death_or_daly_label,
    make_age_grp_lookup,
    order_of_cause_of_death_or_daly_label,
    summarize,
)
import pickle

from tlo import Date
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
    COARSE_APPT_TYPE_TO_COLOR_MAP,
    SHORT_TREATMENT_ID_TO_COLOR_MAP,
    _standardize_short_treatment_id,
    bin_hsi_event_details,
    compute_mean_across_runs,
    get_coarse_appt_type,
    get_color_short_treatment_id,
    order_of_short_treatment_ids,
    plot_stacked_bar_chart,
    squarify_neat,
    unflatten_flattened_multi_index_in_logging,
)

outputspath = Path('./outputs')
figurespath.mkdir(parents=True, exist_ok=True) # create directory if it doesn't exist
resourcefilepath = Path("./resources")

# Declare period for which the results will be generated (defined inclusively)

TARGET_PERIOD = (Date(2010, 1, 1), Date(2012,12,31))

make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

_, age_grp_lookup = make_age_grp_lookup()

def target_period() -> str:
    """Returns the target period as a string of the form YYYY-YYYY"""
    return "-".join(str(t.year) for t in TARGET_PERIOD)

def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = [i.year for i in TARGET_PERIOD]
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

# Find results_folder associated with a given batch_file and get most recent
#results_folder = get_scenario_outputs('impact_of_consumable_scenarios.py', outputspath)
results_folder = Path(outputspath / 'test_scenario-2024-10-21T085345Z')
#results_folder = Path(outputspath / 'impact_of_consumables_scenarios-2024-09-12T155640Z/')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder,1,0)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# Get summary of consumable availability/non-availability in the scenarios
def get_counts_of_items_requested(_df):
    _df = drop_outside_period(_df)

    counts_of_available = defaultdict(int)
    counts_of_not_available = defaultdict(int)

    for _, row in _df.iterrows():
        for item, num in row['Item_Available'].items():
            counts_of_available[item] += num
        for item, num in row['Item_NotAvailable'].items(): # eval(row['Item_NotAvailable'])
            counts_of_not_available[item] += num

    return pd.concat(
        {'Available': pd.Series(counts_of_available), 'Not_Available': pd.Series(counts_of_not_available)},
        axis=1
    ).fillna(0).astype(int).stack()

cons_req = summarize(
    extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Consumables',
        custom_generate_series=get_counts_of_items_requested,
        do_scaling=True
    ),
    only_mean=True,
    collapse_columns=True
)

cons = cons_req.unstack()
cons_names = pd.read_csv(
    resourcefilepath / 'healthsystem' / 'consumables' / 'ResourceFile_Consumables_Items_and_Packages.csv'
)[['Item_Code', 'Items']].set_index('Item_Code').drop_duplicates()
cons_names.index = cons_names.index.astype(str)
cons = cons.merge(cons_names, left_index=True, right_index=True, how='left').set_index('Items') #.astype(int)
cons = cons.assign(total=cons.sum(1)).sort_values('total').drop(columns='total')

cons.columns = pd.MultiIndex.from_tuples(cons.columns, names=['draw', 'stat', 'var'])
cons_not_available = cons.loc[:, cons.columns.get_level_values(2) == 'Not_Available']
cons_not_available.mean = cons_not_available.loc[:, cons_not_available.columns.get_level_values(1) == 'mean']
cons_available = cons.loc[:, cons.columns.get_level_values(2) == 'Available']

cons_not_available = cons_not_available.unstack().reset_index()
cons_not_available = cons_not_available.rename(columns={0: 'qty_not_available'})
cons_available = cons_available.unstack().reset_index()
cons_available = cons_available.rename(columns={0: 'qty_available'})

# Add instances of consumables being available together
cons_available_summary = cons_available.groupby(['draw', 'stat'])['qty_available'].sum()
cons_available_summary = cons_available_summary[cons_available_summary.index.get_level_values(1) == 'mean']
# Check of the alternative scenarios imply greater consumable availability
assert (cons_available_summary.loc[(0, 'mean')] <
        cons_available_summary.loc[(1, 'mean')])
assert (cons_available_summary.loc[(0, 'mean')] <
        cons_available_summary.loc[(2, 'mean')])

