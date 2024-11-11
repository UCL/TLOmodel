import os

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize

outputspath = './outputs/sejjj49@ucl.ac.uk/'

scenario = 'block_intervention_test-2024-11-08T094716Z'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]

interventions = ['oral_antihypertensives', 'iv_antihypertensives',  'mgso4', 'post_abortion_care_core']

int_analysis = ['baseline']

for i in interventions:
    int_analysis.append(f'{i}_min')
    int_analysis.append(f'{i}_max')

draws = [x for x in range(len(int_analysis))]

# Access dataframes generated from pregnancy supervisor
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

    return {'crude':results_df, 'summarised':results_df_summ}

results = {k:get_ps_data_frames(k, results_folder) for k in
           ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
            'yearly_mnh_counter_dict']}

all_dalys_dfs = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=(
            lambda df: df.drop(
                columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
        do_scaling=False)

mat_disorders_all = all_dalys_dfs.loc[(slice(None), 'Maternal Disorders'), :]

mat_dalys_df = mat_disorders_all.loc[2024]
mat_dalys_df_sum = summarize(mat_dalys_df)

results.update({'dalys':{'crude': mat_dalys_df, 'summarised': mat_dalys_df_sum}})

# Summarised results
def get_data(df, key, draw):
    return (df.loc[key, (draw, 'lower')],
            df.loc[key, (draw, 'mean')],
            df.loc[key, (draw, 'upper')])

dalys_by_scenario = {k: get_data(results['dalys']['summarised'], 'Maternal Disorders', d) for k, d in zip (
    int_analysis, draws)}

mmr_by_scnario = {k: get_data(results['deaths_and_stillbirths']['summarised'], 'direct_mmr', d) for k, d in zip (
    int_analysis, draws)}

def barcharts(data, y_label, title):

    # Extract means and errors
    labels = data.keys()
    means = [vals[1] for vals in data.values()]
    lower_errors = [vals[1] - vals[0] for vals in data.values()]
    upper_errors = [vals[2] - vals[1] for vals in data.values()]
    errors = [lower_errors, upper_errors]

    # Create bar chart with error bars
    fig, ax = plt.subplots()
    ax.bar(labels, means, yerr=errors, capsize=5, alpha=0.7, ecolor='black')
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Adjust label size
    plt.xticks(fontsize=8, rotation=90)
    plt.tight_layout()
    plt.show()

barcharts(dalys_by_scenario, 'DALYs', 'Total Maternal Disorders DALYs by scenario')
barcharts(mmr_by_scnario, 'MMR', 'Total Direct MMR by scenario')

# Difference results
def get_diffs(df_key, result_key, ints, draws):
    diff_results = {}
    baseline = results[df_key]['crude'][0]

    for draw, int in zip(draws, ints):
        diff_df = results[df_key]['crude'][draw] - baseline
        diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
                                                    names=['draw', 'run'])
        results_diff = summarize(diff_df)
        results_diff.fillna(0)
        diff_results.update({int: results_diff.loc[result_key].values})

    return diff_results

mmr_diffs = get_diffs('deaths_and_stillbirths', 'direct_mmr', int_analysis, draws)
dalys_diffs = get_diffs('dalys', 'Maternal Disorders', int_analysis, draws)


def get_diff_plots(data, outcome):
    categories = list(data.keys())
    mins = [arr[0] for arr in data.values()]
    means = [arr[1] for arr in data.values()]
    maxs = [arr[2] for arr in data.values()]

    # Error bars (top and bottom of the uncertainty interval)
    errors = [(mean - min_val, max_val - mean) for mean, min_val, max_val in zip(means, mins, maxs)]
    errors = np.array(errors).T

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.errorbar(categories, means, yerr=errors, fmt='o', capsize=5)
    plt.axhline(0, color='gray', linestyle='--')  # Adding a horizontal line at y=0 for reference
    plt.xticks(rotation=90)
    plt.xlabel('Scenarios')
    plt.ylabel('Crude Difference from Baseline Scenario')
    plt.title(f'Difference of {outcome} from Baseline Scenario')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

get_diff_plots(mmr_diffs, 'MMR')
get_diff_plots(dalys_diffs, 'Maternal DALYs')



