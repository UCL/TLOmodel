import os

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize

outputspath = './outputs/sejjj49@ucl.ac.uk/'

scenario = 'block_intervention_test-2024-10-17T161423Z'

results_folder= get_scenario_outputs(scenario, outputspath)[-1]

def get_data_frames(key, results_folder):
    def sort_df(_df):
        _x = _df.drop(columns=['date'], inplace=False)
        return _x.iloc[0]

    results_df = extract_results(
                results_folder,
                module="tlo.methods.pregnancy_supervisor",
                key=key,
                custom_generate_series=sort_df,
                do_scaling=False
            )

    results_df_summ = summarize(extract_results(
        results_folder,
        module="tlo.methods.pregnancy_supervisor",
        key=key,
        custom_generate_series=sort_df,
        do_scaling=False
    ))

    return [results_df, results_df_summ]

results_sum = {k:get_data_frames(k, results_folder)[1] for k in
               ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
                'yearly_mnh_counter_dict']}

results = {k:get_data_frames(k, results_folder)[0] for k in
               ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
                'yearly_mnh_counter_dict']}

baseline = results['deaths_and_stillbirths'].loc['direct_mmr', (0, slice(0, 19))].droplevel(0)

def get_mmr_diffs(df, draws):
    diff_results = {}
    baseline = results['deaths_and_stillbirths'][0]

    for draw in draws:
        # diff_df = ((results['deaths_and_stillbirths'][draw] - baseline)/baseline) * 100
        diff_df = results['deaths_and_stillbirths'][draw] - baseline
        diff_df.columns = pd.MultiIndex.from_tuples([(0, v) for v in range(len(diff_df.columns))],
                                                    names=['draw', 'run'])
        results_diff = summarize(diff_df)
        results_diff.fillna(0)
        diff_results.update({draw: results_diff})

    return diff_results

diff_results = get_mmr_diffs(results, range(1,4))

def get_data(df, draw):
    return (df.loc['direct_mmr', (draw, 'lower')],
            df.loc['direct_mmr', (draw, 'mean')],
            df.loc['direct_mmr', (draw, 'upper')])

results = {'baseline':get_data(results['deaths_and_stillbirths'], 0),
           'blood_transfusion':get_data(results['deaths_and_stillbirths'], 1),
           'pph_treatment_uterotonics':get_data(results['deaths_and_stillbirths'], 2),
           'sepsis_treatment':get_data(results['deaths_and_stillbirths'], 3)}

results_diff = {'blood_transfusion':get_data(diff_results[1], 1),
           'pph_treatment_uterotonics':get_data(diff_results[2], 2),
           'sepsis_treatment':get_data(diff_results[3], 3)}

# todo: compare deaths with demography logging...

results = data

# Extract means and errors
labels = data.keys()
means = [vals[1] for vals in data.values()]
lower_errors = [vals[1] - vals[0] for vals in data.values()]
upper_errors = [vals[2] - vals[1] for vals in data.values()]
errors = [lower_errors, upper_errors]

# Create bar chart with error bars
fig, ax = plt.subplots()
ax.bar(labels, means, yerr=errors, capsize=5, alpha=0.7, ecolor='black')
ax.set_ylabel('Values')
ax.set_title('Bar Chart with Error Bars')

# Adjust label size
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.show()

# scenario_filename = 'cohort_test-2024-10-09T130546Z'
# # scenario_filename2 = 'cohort_test-2024-10-15T122825Z'
# scenario_filename2 = 'cohort_test-2024-10-16T071357Z'
#
# results_folder_old = get_scenario_outputs(scenario_filename, outputspath)[-1]
# results_folder_new = get_scenario_outputs(scenario_filename2, outputspath)[-1]
#
# def get_data_frames(key, results_folder):
#     def sort_df(_df):
#         _x = _df.drop(columns=['date'], inplace=False)
#         return _x.iloc[0]
#
#     results_df = summarize (extract_results(
#                 results_folder,
#                 module="tlo.methods.pregnancy_supervisor",
#                 key=key,
#                 custom_generate_series=sort_df,
#                 do_scaling=False
#             ))
#
#     return results_df
#
# results_old = {k:get_data_frames(k, results_folder_old) for k in
#                ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
#                 'yearly_mnh_counter_dict']}
#
# results_new = {k:get_data_frames(k, results_folder_new) for k in
#                ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths', 'service_coverage',
#                 'yearly_mnh_counter_dict']}
#
# import matplotlib.pyplot as plt
# import numpy as np
# # Sample data
# mmr_data = {
#     'int_1': [(235, 250, 265), (335, 350, 365)],
#     'int_2': [(170, 195, 200), (290, 305, 320)],
#     'int_3': [(280, 295, 310), (295 ,310, 325)],
#     'int_4': [(165, 180, 195), (385, 400, 415)]
# }
# # Plotting
# fig, ax = plt.subplots()
# for key, intervals in mmr_data.items():
#     for idx, (lower, mean, upper) in enumerate(intervals):
#         x = np.arange(len(mmr_data)) * len(intervals) + idx
#         ax.plot(x, mean, 'o', label=f'{key}' if idx == 0 else "")
#         ax.fill_between([x, x], [lower, lower], [upper, upper], alpha=0.2)
# ax.set_xticks(np.arange(len(mmr_data)) * len(intervals) + 0.5)
# ax.set_xticklabels(mmr_data.keys())
# plt.legend()
# plt.show()
