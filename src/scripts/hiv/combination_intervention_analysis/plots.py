"""Run simulations to demonstrate the impact of HIV interventions in combination.
This picks up results that are created using 'run_scenarios.py'
"""

#%% Load the results
# with open('data.pickle', 'rb') as f:
#     # The protocol version used is detected automatically, so we do not
#     # have to specify it.
#     output_loaded_from_pickle = pickle.load(f)


#%% Make summary plots:
import matplotlib.pyplot as plt

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
plt.show()

epi_inc.plot()
plt.show()

cov_art.plot()
plt.show()

cov_diagnosed.plot()
plt.show()

cov_circ.plot()
plt.show()

cov_prep.plot()
plt.show()

cov_behavchg.plot()
plt.show()
