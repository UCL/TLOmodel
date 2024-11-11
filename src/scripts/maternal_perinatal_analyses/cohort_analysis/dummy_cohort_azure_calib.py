import os

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize

outputspath = './outputs/sejjj49@ucl.ac.uk/'

scenario = 'block_intervention_test-2024-11-06T145016Z'
ordered_interventions = ['oral_antihypertensives', 'iv_antihypertensives',  'mgso4', 'post_abortion_care_core']

intervention_groups = []
draws = []

results_folder= get_scenario_outputs(scenario, outputspath)[-1]

def get_ps_data_frames(key, results_folder):
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
    results_df_summ = summarize(results_df)

    return [results_df, results_df_summ]

all_dalys_dfs = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=(
            lambda df: df.drop(
                columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
        do_scaling=True)

mat_disorders_all = all_dalys_dfs.loc[(slice(None), 'Maternal Disorders'), :]

mat_dalys_df = mat_disorders_all.loc[2024]
mat_dalys_df_sum = summarize(mat_dalys_df)

results = {k:get_ps_data_frames(k, results_folder)[0] for k in
           ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
            'yearly_mnh_counter_dict']}

results_sum = {k:get_ps_data_frames(k, results_folder)[1] for k in
               ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
                'yearly_mnh_counter_dict']}


def get_data(df, key, draw):
    return (df.loc[key, (draw, 'lower')],
            df.loc[key, (draw, 'mean')],
            df.loc[key, (draw, 'upper')])

mmrs_min = {f'{k}_min':get_data(results_sum['deaths_and_stillbirths'], d) for k, d in zip (ordered_interventions, draws) }
mmrs_max = { }

mmrs = {'baseline':get_data(results_sum['deaths_and_stillbirths'], 0),
           # 'oral_antihypertensives_min':get_data(results_sum['deaths_and_stillbirths'], 1),
           # 'oral_antihypertensives_max': get_data(results_sum['deaths_and_stillbirths'], 2),
           # 'iv_antihypertensives_min':get_data(results_sum['deaths_and_stillbirths'], 3),
           # 'iv_antihypertensives_max': get_data(results_sum['deaths_and_stillbirths'], 4),
           # 'amtsl_min':get_data(results_sum['deaths_and_stillbirths'], 5),
           # 'amtsl_max': get_data(results_sum['deaths_and_stillbirths'], 6),
           'mgso4_min':get_data(results_sum['deaths_and_stillbirths'], 7),
           'mgso4_max': get_data(results_sum['deaths_and_stillbirths'], 8),
           # 'post_abortion_care_core_min':get_data(results_sum['deaths_and_stillbirths'], 9),
           # 'post_abortion_care_core_max': get_data(results_sum['deaths_and_stillbirths'], 10),
           # 'caesarean_section_min':get_data(results_sum['deaths_and_stillbirths'], 11),
           # 'caesarean_section_max': get_data(results_sum['deaths_and_stillbirths'], 12),
           # 'ectopic_pregnancy_treatment_min':get_data(results_sum['deaths_and_stillbirths'], 13),
           # 'ectopic_pregnancy_treatment_max': get_data(results_sum['deaths_and_stillbirths'], 14),
           }


def get_mmr_diffs(df, draws):
    diff_results = {}
    baseline = results['deaths_and_stillbirths'][0]

    for draw in draws:
        # diff_df = ((results['deaths_and_stillbirths'][draw] - baseline)/baseline) * 100
        diff_df = results['deaths_and_stillbirths'][draw] - baseline
        diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
                                                    names=['draw', 'run'])
        results_diff = summarize(diff_df)
        results_diff.fillna(0)
        diff_results.update({draw: results_diff})

    return diff_results

# MMR



# Maternal deaths
# DALYs



mmrs = {'baseline':get_data(results_sum['deaths_and_stillbirths'], 0),
           'oral_antihypertensives_min':get_data(results_sum['deaths_and_stillbirths'], 1),
           'oral_antihypertensives_max': get_data(results_sum['deaths_and_stillbirths'], 2),
           'iv_antihypertensives_min':get_data(results_sum['deaths_and_stillbirths'], 3),
           'iv_antihypertensives_max': get_data(results_sum['deaths_and_stillbirths'], 4),
           'amtsl_min':get_data(results_sum['deaths_and_stillbirths'], 5),
           'amtsl_max': get_data(results_sum['deaths_and_stillbirths'], 6),
           'mgso4_min':get_data(results_sum['deaths_and_stillbirths'], 7),
           'mgso4_max': get_data(results_sum['deaths_and_stillbirths'], 8),
           'post_abortion_care_core_min':get_data(results_sum['deaths_and_stillbirths'], 9),
           'post_abortion_care_core_max': get_data(results_sum['deaths_and_stillbirths'], 10),
           'caesarean_section_min':get_data(results_sum['deaths_and_stillbirths'], 11),
           'caesarean_section_max': get_data(results_sum['deaths_and_stillbirths'], 12),
           'ectopic_pregnancy_treatment_min':get_data(results_sum['deaths_and_stillbirths'], 13),
           'ectopic_pregnancy_treatment_max': get_data(results_sum['deaths_and_stillbirths'], 14),
           }


diff_results = get_mmr_diffs(results, [7,8])


results_diff = {#'oral_antihypertensives_min':get_data(diff_results[1], 1),
#                 'oral_antihypertensives_max':get_data(diff_results[2], 2),
#                 'iv_antihypertensives_min':get_data(diff_results[3], 3),
#                 'iv_antihypertensives_max': get_data(diff_results[4], 4),
#                 'amtsl_min':get_data(diff_results[5], 5),
#                 'amtsl_max': get_data(diff_results[6], 6),
                'mgso4_min':get_data(diff_results[7], 7),
                'mgso4_max':get_data(diff_results[8], 8),
                # 'post_abortion_care_core_min':get_data(diff_results[9], 9),
                # 'post_abortion_care_core_max': get_data(diff_results[10], 10),
                # 'caesarean_section_min':get_data(diff_results[11], 11),
                # 'caesarean_section_max': get_data(diff_results[12], 12),
                # 'ectopic_pregnancy_treatment_min':get_data(diff_results[13], 13),
                # 'ectopic_pregnancy_treatment_max': get_data(diff_results[14], 14)
                }

# todo: compare deaths with demography logging...

data = mmrs

# Extract means and errors
labels = data.keys()
means = [vals[1] for vals in data.values()]
lower_errors = [vals[1] - vals[0] for vals in data.values()]
upper_errors = [vals[2] - vals[1] for vals in data.values()]
errors = [lower_errors, upper_errors]

# Create bar chart with error bars
fig, ax = plt.subplots()
ax.bar(labels, means, yerr=errors, capsize=5, alpha=0.7, ecolor='black')
ax.set_ylabel('MMR')
ax.set_title('Average MMR under each scenario')

# Adjust label size
plt.xticks(fontsize=8, rotation=90)
plt.tight_layout()
plt.show()

#
# # Example data with uncertainties
# parameters = ['Blood Transfusion', 'Uterotonics', 'Sepsis treatment']
# base_value = results_sum['deaths_and_stillbirths'].at['direct_mmr', (0, 'mean')]  # base case value for the output variable
# high_values = [results_sum['deaths_and_stillbirths'].at['direct_mmr', (1, 'mean')],
#               results_sum['deaths_and_stillbirths'].at['direct_mmr', (3, 'mean')],
#               results_sum['deaths_and_stillbirths'].at['direct_mmr', (5, 'mean')]]  # lower-bound values for each parameter
# low_values = [results_sum['deaths_and_stillbirths'].at['direct_mmr', (2, 'mean')],
#               results_sum['deaths_and_stillbirths'].at['direct_mmr', (4, 'mean')],
#               results_sum['deaths_and_stillbirths'].at['direct_mmr', (6, 'mean')]]  # upper-bound values for each parameter
#
# # Calculate deltas from base value
# low_deltas = [base_value - lv for lv in low_values]
# high_deltas = [hv - base_value for hv in high_values]
#
# # Sort parameters by absolute impact
# abs_impacts = [abs(low) + abs(high) for low, high in zip(low_deltas, high_deltas)]
# sorted_indices = np.argsort(abs_impacts)[::-1]
# parameters = [parameters[i] for i in sorted_indices]
# low_deltas = [low_deltas[i] for i in sorted_indices]
# high_deltas = [high_deltas[i] for i in sorted_indices]
#
# # Calculate changes from the base case
# low_deltas = [base_value - lv for lv in low_values]
# high_deltas = [hv - base_value for hv in high_values]
#
# # Sort parameters by absolute impact (for a tornado effect)
# abs_impacts = [abs(low) + abs(high) for low, high in zip(low_deltas, high_deltas)]
# sorted_indices = np.argsort(abs_impacts)[::-1]
# parameters = [parameters[i] for i in sorted_indices]
# low_deltas = [low_deltas[i] for i in sorted_indices]
# high_deltas = [high_deltas[i] for i in sorted_indices]
#
# # Plotting
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # Plot each bar for the low and high values
# for i, (param, low, high) in enumerate(zip(parameters, low_deltas, high_deltas)):
#     ax.barh(param, high, left=base_value, color='skyblue')
#     ax.barh(param, low, left=base_value + low, color='salmon')
#
# # Reference line for base value
# ax.axvline(base_value, color='black', linestyle='--', label="Base Value")
#
# # Labels and title
# ax.set_xlabel('Output Variable')
# ax.set_title('Tornado Plot')
# plt.legend(['Base Value'])
# plt.show()

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
