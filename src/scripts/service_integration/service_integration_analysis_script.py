from pathlib import Path

import os
import scipy.stats as st
from scipy.stats import t, norm, shapiro

import pandas as pd
import tableone
from tableone import TableOne

import matplotlib.pyplot as plt
import numpy as np

from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize, create_pickles_locally, get_scenario_info

outputspath = './outputs/sejjj49@ucl.ac.uk/'


# create_pickles_locally(results_folder, compressed_file_name_prefix='block_intervention_big_run')

def summarize_confidence_intervals(results: pd.DataFrame) -> pd.DataFrame:
    """Utility function to compute summary statistics
    Finds mean value and 95% interval across the runs for each draw.
    """

    # Calculate summary statistics
    grouped = results.groupby(axis=1, by='draw', sort=False)
    mean = grouped.mean()
    sem = grouped.sem()  # Standard error of the mean

    # Calculate the critical value for a 95% confidence level
    n = grouped.size().max()  # Assuming the largest group size determines the degrees of freedom
    critical_value = t.ppf(0.975, df=n - 1)  # Two-tailed critical value

    # Compute the margin of error
    margin_of_error = critical_value * sem

    # Compute confidence intervals
    lower = mean - margin_of_error
    upper = mean + margin_of_error

    # Combine into a single DataFrame
    summary = pd.concat({'mean': mean, 'lower': lower, 'upper': upper}, axis=1)

    # Format the DataFrame as in the original code
    summary.columns = summary.columns.swaplevel(1, 0)
    summary.columns.names = ['draw', 'stat']
    summary = summary.sort_index(axis=1)

    return summary

scenario = ''
results_folder= get_scenario_outputs(scenario, outputspath)[-1]
int_names = ['status_quo',
             'chronic_care_clinic',
             'screening_htn',
             'screening_dm',
             'screening_hiv',
             'screening_tb',
             'screening_fp',
             'screening_mal',
             'mch_clinic_pnc',
             'mch_clinic_fp',
             'mch_clinic_mal',
             'mch_clinic_epi']

# Create a folder to store graphs (if it hasn't already been created when ran previously)
g_path = f'{outputspath}graphs_{scenario}'

if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}graphs_{scenario}')

results = {}

all_dalys_dfs = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=(
            lambda df: df.drop(
                columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
        do_scaling=False)

dalys_df_sum = summarize(all_dalys_dfs)

results.update({'dalys':{'crude': all_dalys_dfs, 'summarised': dalys_df_sum}})

# Summarised results
def get_data(df, key, draw):
    return (df.loc[key, (draw, 'lower')],
            df.loc[key, (draw, 'mean')],
            df.loc[key, (draw, 'upper')])

dalys_by_scenario = {k: get_data(results['dalys']['summarised'], 'Maternal Disorders', d) for k, d in zip (
    int_analysis, draws)}

# def barcharts(data, y_label, title):
#
#     # Extract means and errors
#     labels = data.keys()
#     means = [vals[1] for vals in data.values()]
#     # lower_errors = [vals[0] for vals in data.values()]
#     # upper_errors = [vals[2] for vals in data.values()]
#
#     lower_errors = [vals[1] - vals[0] for vals in data.values()]
#     upper_errors = [vals[2] - vals[1] for vals in data.values()]
#     errors = [lower_errors, upper_errors]
#
#     # Create bar chart with error bars
#     fig, ax = plt.subplots()
#     ax.bar(labels, means, yerr=errors, capsize=5, alpha=0.7, ecolor='black')
#     ax.set_ylabel(y_label)
#     ax.set_title(title)
#
#     # Adjust label size
#     plt.xticks(fontsize=8, rotation=90)
#     plt.tight_layout()
#     plt.savefig(f'{g_path}/{title}.png', bbox_inches='tight')
#     plt.show()
#
# barcharts(dalys_by_scenario, 'DALYs', 'Total Maternal Disorders DALYs by scenario')
# barcharts(mmr_by_scnario, 'MMR', 'Total MMR by scenario')
# barcharts(mmr_by_scnario, 'MMR', 'Total MMR by scenario')
#


