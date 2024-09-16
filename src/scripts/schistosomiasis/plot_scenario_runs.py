""" use the outputs from scenario_runs.py and produce plots
and summary statistics for paper
"""

from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import pandas as pd
# import lacroix
import matplotlib.colors as colors
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from collections import defaultdict
import textwrap

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
    unflatten_flattened_multi_index_in_logging,
)


resourcefilepath = Path("./resources")
# outputpath = Path("./outputs")

output_folder = Path("./outputs/t.mangal@imperial.ac.uk")

# results_folder = get_scenario_outputs("schisto_calibration.py", outputpath)[-1]
results_folder = get_scenario_outputs("schisto_scenarios.py", output_folder)[-1]

# Declare path for output graphs from this script
def make_graph_file_name(name):
    return results_folder / f"Schisto_{name}.png"


# Name of species that being considered:
species = ('mansoni', 'haematobium')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


# %% FUNCTIONS ##################################################################
TARGET_PERIOD = (Date(2025, 1, 1), Date(2035, 12, 31))
param_names = []  # todo can use params to label scenarios??


def target_period() -> str:
    """Returns the target period as a string of the form YYYY-YYYY"""
    return "-".join(str(t.year) for t in TARGET_PERIOD)


def get_total_num_dalys(_df):
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


total_num_dalys = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_total_num_dalys(),
    do_scaling=True
)


def get_total_num_dalys_by_label(_df):
    """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
    y = _df \
        .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
        .drop(columns=['date', 'year', 'li_wealth']) \
        .sum(axis=0)

    # define course cause mapper for HIV, TB, MALARIA and OTHER
    causes = {
        'Schistosomiasis': 'Schisto',
        'AIDS': 'HIV/AIDS',
        'Cancer (Bladder)': 'Bladder cancer',
        '': 'Other',  # defined in order to use this dict to determine ordering of the causes in output
    }
    causes_relabels = y.index.map(causes).fillna('Other')

    return y.groupby(by=causes_relabels).sum()[list(causes.values())]


total_num_dalys_by_label_results = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_by_wealth_stacked_by_age_and_time",
    custom_generate_series=get_total_num_dalys_by_label,
    do_scaling=True,
)



# todo person-years infected with low/moderate/high intensity infections
# stacked bar plot for each scenario
# separate for mansoni and haematobium


# todo table, rows=districts,
# column level0 with and without WASH:
# classify districts into low/moderate/high burden
# columns=[HML burden, person-years of low/moderate/high infection, # PZQ tablets]




# todo elimination
# years to reach elimination as PH problem
# -- Elimination as a PH problem is defined as reducing the prevalence of heavy infections to less than 1% of population
# years to reach elimination of transmission
# years to reach morbidity control (heavy infections below threshold)
# -- morbidity control is reducing the prevalence of heavy infections to below 5% of the population
# -- (e.g. ≥400 EPG for mansoni or ≥50 eggs/10 mL urine for haematobium).

















