"""Run simulations to demonstrate the impact of HIV interventions in combination.
This picks up results that are created using 'run_scenarios.py'
"""

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Where will outputs be found
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / 'combination_intervention_results.pickle'


#%% Load the results
with open(results_filename, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    X = pickle.load(f)

ScenarioSet = X['ScenarioSet']
outputs = X['outputs']


#%% Make summary plots:

# Extract summary metrics from each run
cov_circ = pd.DataFrame()
cov_prep = pd.DataFrame()
cov_diagnosed = pd.DataFrame()
cov_art = pd.DataFrame()
cov_behavchg = pd.DataFrame()
epi_inc = pd.DataFrame()
epi_prev = pd.DataFrame()


for scenario_name in ScenarioSet:
    output = outputs[scenario_name]

    # Load the coverage results frame:
    cov_over_time = output['tlo.methods.hiv']['hiv_program_coverage']
    cov_over_time = cov_over_time.set_index('date')

    # Store results is summary dataframe
    cov_behavchg[scenario_name] = cov_over_time['prop_adults_exposed_to_behav_intv']
    cov_circ[scenario_name] = cov_over_time['prop_men_circ']
    cov_prep[scenario_name] = cov_over_time['prop_fsw_on_prep']
    cov_diagnosed[scenario_name] = cov_over_time['dx_adult']
    cov_art[scenario_name] = cov_over_time['art_coverage_adult']

    # Load the epi results frame:
    epi = output['tlo.methods.hiv']['summary_inc_and_prev_for_adults_and_children_and_fsw']
    epi = epi.set_index('date')

    # Store results in summary dataframe
    epi_inc[scenario_name] = epi['hiv_adult_inc']
    epi_prev[scenario_name] = epi['hiv_prev_adult']


epi_prev.plot()
plt.title("Prevalence of HIV Among Adults (15+)")
plt.show()

epi_inc.plot()
plt.title("Incidence of HIV Among Adults (15+)")
plt.show()

cov_art.plot()
plt.title("Proportion of PLHIV Adults (15+) on Treatment")
plt.show()

cov_diagnosed.plot()
plt.title("Proportion of PLHIV Adults (15+) Diagnosed")
plt.show()

# todo - error
cov_circ.plot()
plt.title("Proportion of Adults (15+) Men Circumcised")
plt.show()

cov_prep.plot()
plt.title("Proportion of Female Sex Workers On PrEP")
plt.show()

cov_behavchg.plot()
plt.title("Proportion of Adults with Reduced HIV Risk Behaviours")
plt.show()
