from pathlib import Path

import os
import scipy.stats as st
from scipy.stats import t, norm, shapiro

import pandas as pd
import tableone
from tableone import TableOne

import matplotlib.pyplot as plt
import numpy as np

from tlo import Date
from tlo.analysis.utils import (extract_results, get_scenario_outputs, compute_summary_statistics,
make_age_grp_types, get_scenario_info, make_calendar_period_lookup, make_calendar_period_type)

outputspath = './outputs/sejjj49@ucl.ac.uk/'

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

scenario = 'service_integration_scenario-2025-05-21T150139Z'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]
# create_pickles_locally(results_folder, compressed_file_name_prefix='service_integration_scenario')


int_names = ['status_quo',
             'chronic_care_clinic',
             'screening_htn',
             'screening_dm',
             'screening_hiv',
             'screening_tb',
             'screening_fp',
             'screening_mal',
             'screening_all',
             'mch_clinic_pnc',
             'mch_clinic_fp',
             'mch_clinic_all',
             'all_integration']

# Create a folder to store graphs (if it hasn't already been created when ran previously)
g_path = f'{outputspath}graphs_{scenario}'

info = get_scenario_info(results_folder)
draws = [x for x in range(info['number_of_draws'])]

if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}graphs_{scenario}')


TARGET_PERIOD = (Date(2020, 1, 1), Date(2050, 1, 1))

def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    # TO DO: this isnt outputting all dalys (missing 2013 onwards)
    years_needed = [i.year for i in TARGET_PERIOD]
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=False
    )

idx = pd.IndexSlice
total_dalys_dfs = {k: num_dalys.loc[:, idx[d, :]] for k, d in zip (int_names, draws)}

def get_diff_multi_index(df, int_name, draw):
    diff = df[int_name][draw] - df['status_quo'][0]
    diff.columns=df[int_name].columns
    return diff

total_dalys_diff_dfs = {k: get_diff_multi_index(total_dalys_dfs, k, d) for k, d in zip(int_names, draws)}

total_dalys_summ = {k:compute_summary_statistics(total_dalys_dfs[k]) for k in int_names}
total_dalys_diff_summ = {k:compute_summary_statistics(total_dalys_diff_dfs[k]) for k in int_names}

all_dalys_dfs = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
            do_scaling=False)
all_dalys_dfs.index.names = ['year', 'cause']
years_to_sum = list(range(2020, 2051))

# Filter the DataFrame to include only those years
df_subset = all_dalys_dfs.loc[all_dalys_dfs.index.get_level_values('year').isin(years_to_sum)]

# Group by 'cause' and sum
cause_totals = df_subset.groupby('cause').sum()
total_cause_dfs = {k: cause_totals.loc[:, idx[d, :]] for k, d in zip (int_names, draws)}
total_cause_summ = {k:compute_summary_statistics(total_cause_dfs[k]) for k in int_names}

total_cause_diff_dfs = {k: get_diff_multi_index(total_cause_dfs, k, d) for k, d in zip(int_names, draws)}
total_cause_summ_diff = {k:compute_summary_statistics(total_cause_diff_dfs[k]) for k in int_names}


# GRAPHS AND CSV FILES

for k in total_cause_diff_dfs:
    total_cause_diff_dfs[k].to_csv(f'{g_path}/{k}_diffs.csv')

for k, d in zip(total_cause_diff_dfs, draws):
    labels = total_cause_summ_diff[k].index
    median = total_cause_summ_diff[k][d]['central'].values
    lower_errors = total_cause_summ_diff[k][d]['lower'].values
    upper_errors = total_cause_summ_diff[k][d]['upper'].values

    # lower_errors = [data[k].loc[0, 'lower'] for k in labels]
    # upper_errors = [data[k].loc[0, 'upper'] for k in labels]

    # lower_errors = [data[k][d].loc[0, 'lower'] - data[k][d].loc[0, 'central']for k, d in zip(labels, draws)]
    # upper_errors = [data[k][d].loc[0, 'upper'] - data[k][d].loc[0, 'lower'] for k, d in zip(labels, draws)]
    # errors = [lower_errors, upper_errors]

    # Compute distances from mean to bounds (must be non-negative)
    yerr_lower = [mean - low for mean, low in zip(median, lower_errors)]
    yerr_upper = [up - mean for mean, up in zip(median, upper_errors)]

    # Create bar chart with error bars
    fig, ax = plt.subplots()
    ax.bar(labels, median, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('Difference in DALYs from SQ')
    ax.set_title(f'{k} Vs status_quo: Difference in DALYs by cause')

    # Adjust label size
    plt.xticks(fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(f'{g_path}/{k}_diff_dalys_cause.png', bbox_inches='tight')
    plt.show()


def barcharts(data, y_label, title):

    # Extract means and errors
    labels = data.keys()
    median = [data[k][d].loc[0, 'central'] for k, d in zip(labels, draws)]
    lower_errors = [data[k][d].loc[0, 'lower'] for k, d in zip(labels, draws)]
    upper_errors = [data[k][d].loc[0, 'upper'] for k, d in zip(labels, draws)]

    # lower_errors = [data[k].loc[0, 'lower'] for k in labels]
    # upper_errors = [data[k].loc[0, 'upper'] for k in labels]

    # lower_errors = [data[k][d].loc[0, 'lower'] - data[k][d].loc[0, 'central']for k, d in zip(labels, draws)]
    # upper_errors = [data[k][d].loc[0, 'upper'] - data[k][d].loc[0, 'lower'] for k, d in zip(labels, draws)]
    # errors = [lower_errors, upper_errors]

    # Compute distances from mean to bounds (must be non-negative)
    yerr_lower = [mean - low for mean, low in zip(median, lower_errors)]
    yerr_upper = [up - mean for mean, up in zip(median, upper_errors)]

    # Create bar chart with error bars
    fig, ax = plt.subplots()
    ax.bar(labels, median, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black')
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Adjust label size
    plt.xticks(fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(f'{g_path}/{title}.png', bbox_inches='tight')
    plt.show()

barcharts(total_dalys_diff_summ, 'Difference in DALYs', 'Total Difference in Total DALYs from Status Quo by '
                                                   'Scenario')
barcharts(total_dalys_summ, 'DALYs', ' Total DALYs from Status Quo by Scenario')


keys = list(total_cause_summ.keys())
baseline_key = keys[0]
baseline_df = total_cause_summ[baseline_key]

categories = baseline_df.index
x = np.arange(len(categories))
width = 0.35  # width of each bar

for key, draw in zip(keys[1:], draws[1:]):
    comp_df = total_cause_summ[key]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data and compute asymmetric error bars
    # Baseline
    baseline_central = baseline_df[0]['central']
    baseline_err_lower = baseline_central - baseline_df[0]['lower']
    baseline_err_upper = baseline_df[0]['upper'] - baseline_central

    # Comparison
    comp_central = comp_df[draw]['central']
    comp_err_lower = comp_central - comp_df[draw]['lower']
    comp_err_upper = comp_df[draw]['upper'] - comp_central

    # Plot bars with asymmetric error bars
    ax.bar(x - width/2, baseline_central, width,
           yerr=[baseline_err_lower, baseline_err_upper],
           capsize=5, label=baseline_key, alpha=0.8)

    ax.bar(x + width/2, comp_central, width,
           yerr=[comp_err_lower, comp_err_upper],
           capsize=5, label=key, alpha=0.8)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    ax.set_title(f"Comparison: {key} vs {baseline_key}")
    ax.set_ylabel("DALYs")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{g_path}/{key}_dalys_cause.png', bbox_inches='tight')
    plt.show()




def get_dalys_by_period_sex_agegrp_label(df):
    """Sum the dalys by period, sex, age-group and label"""
    df['age_grp'] = df['age_range'].astype(make_age_grp_types())
    df = df.drop(columns=['date', 'age_range', 'sex'])
    df = df.groupby(by=["year", "age_grp"]).sum().stack()
    df.index = df.index.set_names('label', level=2)
    return df

dalys = extract_results(
                results_folder,
                module="tlo.methods.healthburden",
                key="dalys_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
                custom_generate_series=get_dalys_by_period_sex_agegrp_label,
                do_scaling=False
            )
dalys.index = dalys.index.set_names('age_group', level=1)

def get_pop_by_agegrp_label(df):
    """Sum the dalys by period, sex, age-group and label"""
    df['year'] = df['date'].dt.year
    df_melted = df.melt(id_vars=['year'], value_vars=[col for col in df.columns if col not in ['date', 'year']],
                        var_name='age_group', value_name='count')
    series_multi = df_melted.set_index(['year', 'age_group'])['count'].sort_index()

    return series_multi


pop_f = extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="age_range_f",  # <-- for DALYS stacked by age and time
                custom_generate_series=get_pop_by_agegrp_label,
                do_scaling=False
            )

pop_m = extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="age_range_m",  # <-- for DALYS stacked by age and time
                custom_generate_series=get_pop_by_agegrp_label,
                do_scaling=False
            )

pop = pop_f + pop_m






pop_summ = compute_summary_statistics(pop)
dalys_summ = compute_summary_statistics(dalys)


# TODO OTHER OUTPUTS
# CONSUMABLES
# HCW TIME
# NUMBER OF APPOINTMENTS



# def get_mean_pop_by_age_for_sex_and_year(sex):
#     years_needed = [i.year for i in TARGET_PERIOD]
#
#     if sex == 'F':
#         key = "age_range_f"
#     else:
#         key = "age_range_m"
#
#     num_by_age = compute_summary_statistics(
#         extract_results(results_folder,
#                         module="tlo.methods.demography",
#                         key=key,
#                         custom_generate_series=(
#
#                             lambda df_: df_.drop(
#                                 columns=['date']
#                             ).melt(
#                                 var_name='age_grp'
#                             ).set_index('age_grp')['value']
#                         ),
#                         do_scaling=False
#                         ),
#         collapse_columns=True,
#     )
#     print(num_by_age.index[num_by_age.index.duplicated()])
#     # num_by_age = num_by_age.reindex(make_age_grp_types().categories)
#     return num_by_age
#
# model_m = get_mean_pop_by_age_for_sex_and_year('M')
# model_f = get_mean_pop_by_age_for_sex_and_year('F')
