import os

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize

outputspath = './outputs/sejjj49@ucl.ac.uk/'

scenario = 'block_intervention_test-2024-10-18T113429Z'

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

results = {k:get_data_frames(k, results_folder)[0] for k in
           ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
            'yearly_mnh_counter_dict']}

results_sum = {k:get_data_frames(k, results_folder)[1] for k in
               ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
                'yearly_mnh_counter_dict']}

def get_data(df, draw):
    return (df.loc['direct_mmr', (draw, 'lower')],
            df.loc['direct_mmr', (draw, 'mean')],
            df.loc['direct_mmr', (draw, 'upper')])

mmrs = {'baseline':get_data(results_sum['deaths_and_stillbirths'], 0),
           'blood_transfusion_min':get_data(results_sum['deaths_and_stillbirths'], 1),
           'blood_transfusion_max': get_data(results_sum['deaths_and_stillbirths'], 2),
           'pph_treatment_uterotonics_min':get_data(results_sum['deaths_and_stillbirths'], 3),
           'pph_treatment_uterotonics_max': get_data(results_sum['deaths_and_stillbirths'], 4),
           'sepsis_treatment_min':get_data(results_sum['deaths_and_stillbirths'], 5),
           'sepsis_treatment_max': get_data(results_sum['deaths_and_stillbirths'], 6),
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

diff_results = get_mmr_diffs(results, range(1,7))


results_diff = {'blood_transfusion_min':get_data(diff_results[1], 1),
                'blood_transfusion_max':get_data(diff_results[2], 2),
                'pph_treatment_uterotonics_min':get_data(diff_results[3], 3),
                'pph_treatment_uterotonics_max': get_data(diff_results[4], 4),
                'sepsis_treatment_mins':get_data(diff_results[5], 5),
                'sepsis_treatment_max': get_data(diff_results[6], 6)}

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
plt.xticks(fontsize=8)
plt.tight_layout()
plt.show()


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
