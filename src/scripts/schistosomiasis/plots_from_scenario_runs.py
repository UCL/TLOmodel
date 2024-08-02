""" use the outputs from a local run and plot model outputs
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

outputpath = Path("./outputs/t.mangal@imperial.ac.uk")

results_folder = get_scenario_outputs("schisto_calibration.py", outputpath)[-1]

# Declare path for output graphs from this script
def make_graph_file_name(name):
    return outputpath / f"Schisto_{name}.png"


# Name of species that being considered:
species = ('mansoni', 'haematobium')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# %% Extract and process the `pd.DataFrame`s needed


def construct_dfs(schisto_log) -> dict:
    """Create dict of pd.DataFrames containing counts of infection_status by date, district and age-group."""
    return {
        k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
        for k, v in schisto_log.items() if k in [f'infection_status_{s}' for s in species]
    }


dfs = construct_dfs(log['tlo.methods.schisto'])
