"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_grid,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

from tlo import Simulation
outputspath = Path('./outputs')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs('hiv_prep_baseline_scenario.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# ---------------------- EXTRACT HIV PREVALENCE ---------------------- #
# ------------------------------ ADULTS------------------------------- #
# 2) Extract a specific logged output for all runs, e.g. prevalence:
extracted = extract_results(results_folder,
                            module="tlo.methods.hiv",
                            key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                            column="hiv_prev_adult_1549",
                            index="date")

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extracted, only_mean=True)
res.name = 'z'

# 4) Create a timeseries:
grid = get_grid(params, res)
fig, ax = plt.subplots()
ax.plot(res.index, res.values)

ax.set_title('HIV Prevalence in Adults (15-49 years)')
plt.xlabel('Time (year)')
plt.ylabel('HIV Prevalence (%)')
plt.show()

# ------------------------------ WOMEN (15-49 years) ------------------------------- #
# 2) Extract a specific logged output for all runs, e.g. prevalence:
extracted = extract_results(results_folder,
                            module="tlo.methods.hiv",
                            key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                            column="hiv_prev_women_reproductive_age",
                            index="date")

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extracted, only_mean=True)
res.name = 'z'

# 4) Create a timeseries:
grid = get_grid(params, res)
fig, ax = plt.subplots()
ax.plot(res.index, res.values)

ax.set_title('HIV Prevalence in Women (15-49 years)')
plt.xlabel('Time (years)')
plt.ylabel('HIV Prevalence (%)')
plt.show()

# ------------------------------ CHILDREN ------------------------------- #
# 2) Extract a specific logged output for all runs, e.g. prevalence:
extracted = extract_results(results_folder,
                            module="tlo.methods.hiv",
                            key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                            column="hiv_prev_child",
                            index="date")

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extracted, only_mean=True)
res.name = 'z'

# 4) Create a timeseries:
grid = get_grid(params, res)
fig, ax = plt.subplots()
ax.plot(res.index, res.values)

ax.set_title('HIV Prevalence in Children (0-14 years)')
plt.xlabel('Time (years)')
plt.ylabel('HIV Prevalence (%)')
plt.show()

# ---------------------- EXTRACT HIV INCIDENCE ---------------------- #
# ------------------------------ ADULTS------------------------------- #
# 2) Extract a specific logged output for all runs, e.g. prevalence:
extracted = extract_results(results_folder,
                            module="tlo.methods.hiv",
                            key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                            column="hiv_adult_inc_1549",
                            index="date")

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extracted, only_mean=True)
res.name = 'z'

# 4) Create a timeseries:
grid = get_grid(params, res)
fig, ax = plt.subplots()
ax.plot(res.index, res.values)

ax.set_title('HIV Incidence in Adults (15-49 years)')
plt.xlabel('Time (year)')
plt.ylabel('Incidence (per 100pyar')
plt.show()

# ------------------------------ WOMEN (15-49 years) ------------------------------- #
# 2) Extract a specific logged output for all runs, e.g. prevalence:
extracted = extract_results(results_folder,
                            module="tlo.methods.hiv",
                            key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                            column="hiv_women_reproductive_age_inc",
                            index="date")

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extracted, only_mean=True)
res.name = 'z'

# 4) Create a timeseries:
grid = get_grid(params, res)
fig, ax = plt.subplots()
ax.plot(res.index, res.values)

ax.set_title('HIV Incidence in Women (15-49 years)')
plt.xlabel('Time (years)')
plt.ylabel('Incidence (per 100pyar')
plt.show()

# ------------------------------ CHILDREN ------------------------------- #
# 2) Extract a specific logged output for all runs, e.g. prevalence:
extracted = extract_results(results_folder,
                            module="tlo.methods.hiv",
                            key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                            column="hiv_child_inc",
                            index="date")

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extracted, only_mean=True)
res.name = 'z'

# 4) Create a timeseries:
grid = get_grid(params, res)
fig, ax = plt.subplots()
ax.plot(res.index, res.values)

ax.set_title('HIV Incidence in Children (0-14 years)')
plt.xlabel('Time (years)')
plt.ylabel('Incidence (per 100pyar')
plt.show()

# ---------------------- EXTRACT MTCT RATE ---------------------- #
# ------------------------ MTCT RATE ---------------------------- #
# 2) Extract a specific logged output for all runs, e.g. prevalence:
extracted = extract_results(results_folder,
                            module="tlo.methods.hiv",
                            key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                            column="mtct",
                            index="date")

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extracted, only_mean=True)
res.name = 'z'

# 4) Create a timeseries:
grid = get_grid(params, res)
fig, ax = plt.subplots()
ax.plot(res.index, res.values)

ax.set_title('Mother to Child Transmission Rate')
plt.xlabel('Time (year)')
plt.ylabel('Mother to Child Transmission Rate')
plt.show()

# ---------------------- EXTRACT PROGRAM COVERAGE ---------------------- #
# ------------------------------ PREP ------------------------------- #
# 2) Extract a specific logged output for all runs, e.g. prevalence:
extracted = extract_results(results_folder,
                            module="tlo.methods.hiv",
                            key="hiv_program_coverage",
                            column="prop_preg_and_bf_on_prep",
                            index="date")

# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
res = summarize(extracted, only_mean=True)
res.name = 'z'

# 4) Create a timeseries:
grid = get_grid(params, res)
fig, ax = plt.subplots()
ax.plot(res.index, res.values)

ax.set_title('Proportion of Pregnant and Breastfeeding Women on PrEP')
plt.xlabel('Time (year)')
plt.ylabel('Proportion of Pregnant and Breastfeeding Women on PrEP (%)')
plt.show()
