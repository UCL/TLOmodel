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

scenario = 'block_intervention_big_run-2024-12-04T115735Z'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]
# create_pickles_locally(results_folder, compressed_file_name_prefix='block_intervention_big_run')

interventions =['sepsis_treatment', 'amtsl', 'blood_transfusion']

int_analysis = ['baseline']

for i in interventions:
    int_analysis.append(f'{i}_min')
    int_analysis.append(f'{i}_max')

draws = [x for x in range(len(int_analysis))]

columns = ['age_years', 'la_parity', 'region_of_residence', 'li_wealth', 'li_bmi', 'li_mar_stat', 'li_ed_lev',
            'li_urban', 'ps_prev_spont_abortion', 'ps_prev_stillbirth', 'ps_prev_pre_eclamp', 'ps_prev_gest_diab']
categorical = ['region_of_residence', 'li_wealth', 'li_bmi' ,'li_mar_stat', 'li_ed_lev', 'li_urban',
                'ps_prev_spont_abortion', 'ps_prev_stillbirth', 'ps_prev_pre_eclamp', 'ps_prev_gest_diab']
continuous = ['age_years', 'la_parity']

rename = {'age_years': 'Age (years)',
           'la_parity': 'Parity',
           'region_of_residence': 'Region',
           'li_wealth': 'Wealth Quintile',
           'li_bmi': 'BMI level',
           'li_mar_stat': 'Marital Status',
           'li_ed_lev': 'Education Level',
           'li_urban': 'Urban/Rural',
           'ps_prev_spont_abortion': 'Previous Miscarriage',
           'ps_prev_stillbirth': 'Previous Stillbirth',
           'ps_prev_pre_eclamp': 'Previous Pre-eclampsia',
           'ps_prev_gest_diab': 'Previous Gestational Diabetes',
          }

all_preg_df = pd.read_excel(Path("./resources/maternal cohort") /
                                    'ResourceFile_All2024PregnanciesCohortModel.xlsx')
population = 40_000

# Only select rows equal to the desired population size
if population <= len(all_preg_df):
    preg_pop = all_preg_df.loc[0:population-1]
else:
    # Calculate the number of rows needed to reach the desired length
    additional_rows = population - len(all_preg_df)

    # Initialize an empty DataFrame for additional rows
    rows_to_add = pd.DataFrame(columns=all_preg_df.columns)

    # Loop to fill the required additional rows
    while additional_rows > 0:
        if additional_rows >= len(all_preg_df):
            rows_to_add = pd.concat([rows_to_add, all_preg_df], ignore_index=True)
            additional_rows -= len(all_preg_df)
        else:
            rows_to_add = pd.concat([rows_to_add, all_preg_df.iloc[:additional_rows]], ignore_index=True)
            additional_rows = 0

    # Concatenate the original DataFrame with the additional rows
    preg_pop = pd.concat([all_preg_df, rows_to_add], ignore_index=True)

mytable = TableOne(preg_pop[columns], categorical=categorical,
                   continuous=continuous, rename=rename, pval=False)
print(mytable.tabulate(tablefmt = "fancy_grid"))
mytable.to_excel(Path(f"{outputspath}/{scenario}/0/table_one.xlsx") )

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
    results_df_summ = summarize_confidence_intervals(results_df)

    return {'crude':results_df, 'summarised':results_df_summ}

results = {k:get_ps_data_frames(k, results_folder) for k in
           ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
            'yearly_mnh_counter_dict']}

direct_deaths = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.loc[(df['label'] == 'Maternal Disorders')].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=False)

br = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="on_birth",
            custom_generate_series=(
                lambda df: df.assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=False
        )

dd_sum = summarize_confidence_intervals(direct_deaths)
dd_mmr = (direct_deaths/br) * 100_000
dd_mr_sum = summarize_confidence_intervals(dd_mmr)

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
mat_dalys_df_sum = summarize_confidence_intervals(mat_dalys_df)

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

mmr_by_scenario_oth_log = {k: get_data(dd_mr_sum, 2024, d) for k, d in zip (
    int_analysis, draws)}

def barcharts(data, y_label, title):

    # Extract means and errors
    labels = data.keys()
    means = [vals[1] for vals in data.values()]
    # lower_errors = [vals[0] for vals in data.values()]
    # upper_errors = [vals[2] for vals in data.values()]

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
barcharts(mmr_by_scnario, 'MMR', 'Total MMR by scenario')
barcharts(mmr_by_scnario, 'MMR', 'Total MMR by scenario')

# Difference results
def get_diffs(df_key, result_key, ints, draws):
    diff_results = {}
    baseline = results[df_key]['crude'][0]

    for draw, int in zip(draws, ints):
        diff_df = results[df_key]['crude'][draw] - baseline
        diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
                                                    names=['draw', 'run'])
        results_diff = summarize_confidence_intervals(diff_df)
        results_diff.fillna(0)
        diff_results.update({int: results_diff.loc[result_key].values})

    return [diff_results, diff_df]

diff_results = {}
baseline = dd_mmr[0]

for draw, int in zip(draws, int_analysis):
    diff_df = dd_mmr[draw] - baseline
    diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
                                                    names=['draw', 'run'])
    results_diff = summarize_confidence_intervals(diff_df)
    results_diff.fillna(0)
    diff_results.update({int: results_diff.loc[2024].values})


mat_deaths = get_diffs('deaths_and_stillbirths', 'direct_maternal_deaths', int_analysis, draws)[0]
mmr_diffs = get_diffs('deaths_and_stillbirths', 'direct_mmr', int_analysis, draws)[0]
dalys_diffs = get_diffs('dalys', 'Maternal Disorders', int_analysis, draws)[0]
mat_deaths_2 = diff_results

def get_diff_plots(data, outcome):
    categories = list(data.keys())
    mins = [arr[0] for arr in data.values()]
    means = [arr[1] for arr in data.values()]
    maxs = [arr[2] for arr in data.values()]

    # Error bars (top and bottom of the uncertainty interval)
    errors = [(mean - min_val, max_val - mean) for mean, min_val, max_val in zip(means, mins, maxs)]
    errors = np.array(errors).T

    # todo: the error bars are slightly off...

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
get_diff_plots(mat_deaths, 'Maternal Deaths (crude)')
get_diff_plots(mat_deaths_2, 'MMR (demog log)')
get_diff_plots(dalys_diffs, 'Maternal DALYs')

def get_tornado_plot(data, outcome):
    grouped_data = {}
    data.pop('baseline', None)

    for key in mat_deaths_2.keys():
        base_key = key.split('_')[0]  # Extract the base name
        if base_key not in grouped_data:
            grouped_data[base_key] = {'min': None, 'max': None}
        if 'min' in key:
            grouped_data[base_key]['min'] = data[key]
        elif 'max' in key:
            grouped_data[base_key]['max'] = data[key]

    # Prepare data for plotting
    categories = list(grouped_data.keys())
    min_values = [np.mean(grouped_data[cat]['min']) for cat in categories]
    max_values = [np.mean(grouped_data[cat]['max']) for cat in categories]

    # categoriesmin_errors_lower = [np.mean(grouped_data[cat]['min']) - min(grouped_data[cat]['min']) for cat in categories]
    # min_errors_upper = [max(grouped_data[cat]['min']) - min(grouped_data[cat]['min']) for cat in categories]
    #
    # max_errors_lower = [np.mean(grouped_data[cat]['max']) - min(grouped_data[cat]['max']) for cat in categories]
    # max_errors_upper = [max(grouped_data[cat]['max']) - min(grouped_data[cat]['max']) for cat in categories]
    #
    # # lower_errors = [vals[1] - vals[0] for vals in data.values()]
    # # upper_errors = [vals[2] - vals[1] for vals in data.values()]
    #
    # # Prepare error bars
    # min_error_bars = [[error, 0] for error in min_errors]  # Negative error only
    # max_error_bars = [[0, error] for error in max_errors]  # Positive error only

    # Plotting
    plt.figure(figsize=(10, 6))
    y_positions = np.arange(len(categories))
    # Plot bars for min and max values
    # plt.barh(y_positions, min_values, xerr=np.array(min_error_bars).T, color='lightcoral', edgecolor='black', alpha=0.7,
    #          label='Min Effect')
    # plt.barh(y_positions, max_values, xerr=np.array(max_error_bars).T, color='skyblue', edgecolor='black', alpha=0.7,
    #          label='Max Effect')

    plt.barh(y_positions, min_values, color='lightcoral', edgecolor='black', alpha=0.7,
            label='Min Effect')
    plt.barh(y_positions, max_values, color='skyblue', edgecolor='black', alpha=0.7,
             label='Max Effect')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')  # Central zero line
    # Add labels
    plt.yticks(y_positions, categories)
    plt.xlabel(f'Difference in {outcome} from Status Quo')
    plt.title(f'Tornado Plot showing current and potenial impact of interventions on {outcome}')
    plt.legend()
    plt.show()

get_tornado_plot(mat_deaths_2, 'MMR')
get_tornado_plot(dalys_diffs, 'DALYs')





# # NORMALITY OF MMR ESTIMATES ACROSS RUNS (NOT DIFFERENCES)
# for draw in draws:
#     data = results['deaths_and_stillbirths']['crude'].loc['direct_mmr', draw].values
#
#     # Importing Shapiro-Wilk test for normality
#     # Conducting Shapiro-Wilk test
#     stat, p_value = shapiro(data)
#     # Plotting histogram
#     plt.hist(data, bins=15, density=True, alpha=0.6, color='skyblue', edgecolor='black')
#
#     # Overlay normal distribution (optional)
#     mean, std = np.mean(data), np.std(data)
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = norm.pdf(x, mean, std)
#     plt.axvline(mean, color='green', linestyle='-', linewidth=2, label='Data Mean')
#     plt.plot(x, p, 'r--', linewidth=2, label='Normal Curve')
#
#     # Adding labels and legend
#     plt.title(f'MMR data Histogram with Normality Test (p-value = {p_value:.4f}) (Draw {draw})')
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.legend()
#     # Show plot
#     plt.show()
#     # Printing Shapiro-Wilk test results
#     print(f"Shapiro-Wilk Test Statistic: {stat:.4f}, p-value: {p_value:.4f}")
#     if p_value > 0.05:
#         print("Result: Data likely follows a normal distribution (p > 0.05).")
#     else:
#         print("Result: Data likely does not follow a normal distribution (p ≤ 0.05).")

