"""
Extracts DALYs and mortality from the TB module
 """

import argparse
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
    make_age_grp_lookup,
    make_age_grp_types,
    summarize,
)

from tlo import Date

#results_folder = Path("./outputs")
outputspath = Path("./outputs/nic503@york.ac.uk")


# collecting basic information associated with scenario

# Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('scenario_impact_noXpert_diagnosis.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

num_deaths =extract_results(
        outputspath,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths,
        do_scaling=True
    )
num_dalys = extract_results(
    outputspath,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_num_dalys(),
    do_scaling=True
)

num_deaths_summarized = summarize(num_deaths).loc[0].unstack()
num_dalys_summarized = summarize(num_dalys).loc[0].unstack()

#     # # TB deaths will exclude TB/HIV
#     # # keep if cause = TB
#     # keep = (deaths.cause == "TB")
#     # deaths_TB = deaths.loc[keep].copy()
#     # deaths_TB["year"] = deaths_TB.index.year  # count by year
#     # tot_tb_non_hiv_deaths = deaths_TB.groupby(by=["year"]).size()
#     # tot_tb_non_hiv_deaths.index = pd.to_datetime(tot_tb_non_hiv_deaths.index, format="%Y")

# Plot for total number of DALYs from the scenario
name_of_plot = f'Total DALYS, {target_period()}'
fig, ax = make_plot(num_dalys_summarized / 1e6)
ax.set_title(name_of_plot)
ax.set_ylabel('DALYS (Millions)')
fig.tight_layout()
fig.savefig("DALY_graph.png")
plt.show()

# plot of total number of deaths from the scenario
name_of_plot= f'Total Deaths, {target_period()}'
fig, ax = make_plot(num_deaths_summarized / 1e6)
ax.set_title(name_of_plot)
ax.set_ylabel('Deaths (Millions)')
fig.tight_layout()
fig.savefig("Mortality_graph.png")
plt.show()

