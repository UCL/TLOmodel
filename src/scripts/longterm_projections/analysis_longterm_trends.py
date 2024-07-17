"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
)


resource_filepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
output_path = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/longterm_trends/longterm_trends_all_diseases-2024-07-17T112619Z")

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder =  output_path #= get_scenario_outputs("test_azure_longterm_trends.py", output_path)
print(results_folder)
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)
# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract population over time
pop_model = summarize(extract_results(results_folder,
                                      module="tlo.methods.demography",
                                      key="population",
                                      column="total",
                                      index="date",
                                      do_scaling=True
                                      ),
                      collapse_columns=True
                      )
pop_model.index = pop_model.index.year

print(pop_model)
