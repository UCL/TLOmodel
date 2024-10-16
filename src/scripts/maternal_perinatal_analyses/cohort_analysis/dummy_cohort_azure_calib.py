import os

import pandas as pd

from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize

outputspath = './outputs/sejjj49@ucl.ac.uk/'
scenario_filename = 'cohort_test-2024-10-09T130546Z'
# scenario_filename2 = 'cohort_test-2024-10-15T122825Z'
scenario_filename2 = 'cohort_test-2024-10-16T071357Z'

results_folder_old = get_scenario_outputs(scenario_filename, outputspath)[-1]
results_folder_new = get_scenario_outputs(scenario_filename2, outputspath)[-1]

def get_data_frames(key, results_folder):
    def sort_df(_df):
        _x = _df.drop(columns=['date'], inplace=False)
        return _x.iloc[0]

    results_df = summarize (extract_results(
                results_folder,
                module="tlo.methods.pregnancy_supervisor",
                key=key,
                custom_generate_series=sort_df,
                do_scaling=False
            ))

    return results_df

results_old = {k:get_data_frames(k, results_folder_old) for k in
               ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
                'yearly_mnh_counter_dict']}

results_new = {k:get_data_frames(k, results_folder_new) for k in
               ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths', 'service_coverage',
                'yearly_mnh_counter_dict']}

import matplotlib.pyplot as plt
import numpy as np
# Sample data
mmr_data = {
    'int_1': [(235, 250, 265), (335, 350, 365)],
    'int_2': [(170, 195, 200), (290, 305, 320)],
    'int_3': [(280, 295, 310), (295 ,310, 325)],
    'int_4': [(165, 180, 195), (385, 400, 415)]
}
# Plotting
fig, ax = plt.subplots()
for key, intervals in mmr_data.items():
    for idx, (lower, mean, upper) in enumerate(intervals):
        x = np.arange(len(mmr_data)) * len(intervals) + idx
        ax.plot(x, mean, 'o', label=f'{key}' if idx == 0 else "")
        ax.fill_between([x, x], [lower, lower], [upper, upper], alpha=0.2)
ax.set_xticks(np.arange(len(mmr_data)) * len(intervals) + 0.5)
ax.set_xticklabels(mmr_data.keys())
plt.legend()
plt.show()
