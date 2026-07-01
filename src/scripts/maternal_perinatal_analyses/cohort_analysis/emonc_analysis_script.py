from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import os
from scipy.stats import t

import pandas as pd
from tableone import TableOne

from tlo import Date
from tlo.analysis.utils import extract_results, get_scenario_outputs, get_scenario_info, parse_log_file
from src.scripts.costing.cost_estimation import (do_stacked_bar_plot_of_cost_by_category,
    estimate_input_cost_of_scenarios, summarize_cost_data
)
outputspath = './outputs/sejjj49@ucl.ac.uk/'
resourcefilepath = Path("./resources")

# create_pickles_locally(results_folder, compressed_file_name_prefix='block_intervention_big_run')

#  ======================================= DEFINE SCENARIO INFORMATION  ===============================================
scenario = 'testing_scenario_682612'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]

# Create a folder to store graphs (if it hasn't already been created when ran previously)
g_path = f'{outputspath}graphs_{scenario}'

if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}graphs_{scenario}')

int_analysis = ['baseline',
                'abortion',
                'mat_sepsis_cm',
                'pph_cm',
                'ol_cm',
                'spe_ec_cm',
                'cs_surg',
                'neo_sep_cm',
                'preterm_cm',
                'neo_resus']

info = get_scenario_info(results_folder)
draws = [x for x in range(info['number_of_draws'])]

#  ======================================= DEFINE HELPER FUNCTIONS  =================================================

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

def get_deaths_dalys_demog(group, multiplier):
    direct_deaths = extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="death",
                custom_generate_series=(
                    lambda df: df.loc[(df['label'] == f'{group} Disorders')].assign(
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
    dd_mr = (direct_deaths/br) * multiplier
    dd_mr_sum = summarize_confidence_intervals(dd_mr)

    all_dalys_dfs = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
            do_scaling=False)

    disorders_all = all_dalys_dfs.loc[(slice(None), f'{group} Disorders'), :]

    dalys_df = disorders_all.loc[2025]
    dalys_df_sum = summarize_confidence_intervals(dalys_df)

    return [direct_deaths, dd_sum, dd_mr, dd_mr_sum, dalys_df, dalys_df_sum]

#  ========================================== EXTRACT CORE DATA  =====================================================
results = {k:get_ps_data_frames(k, results_folder) for k in
           ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage',
            'yearly_mnh_counter_dict', 'intervention_coverage']}

# ======================================== FIGURE 1 - MET NEED =======================================================


mat_deaths_dalys = get_deaths_dalys_demog('Maternal', 100_000)
neo_deaths_dalys = get_deaths_dalys_demog('Neonatal', 1000)

# get combined dalys
total_dalys = pd.DataFrame(
    mat_deaths_dalys[4].to_numpy() + neo_deaths_dalys[4].to_numpy(),
    index=['total_dalys'],
    columns=mat_deaths_dalys[4].columns)
total_dalys_sum = summarize_confidence_intervals(total_dalys)

# Get still births
#  TODO: use IP stillbirths for DALY calculations only
stillbirths = results['deaths_and_stillbirths']['crude'].loc[['total_stillbirths']]
stillbirths_summ = summarize_confidence_intervals(stillbirths)
sbr = results['deaths_and_stillbirths']['crude'].loc[['sbr']]
sbr_summ = summarize_confidence_intervals(sbr)

# add IP stillbirth to DALYs
# TODO replace with IB stillbirth only
# TODO adjust YLL for IP stillbirths

yll_from_sb = stillbirths * 90.0
adjusted_dalys = pd.DataFrame(
    total_dalys.to_numpy() + yll_from_sb.to_numpy(),
    index=['adj_total_dalys'],
    columns=total_dalys.columns)
adj_dalys_summ = summarize_confidence_intervals(adjusted_dalys)

results.update({
                'mat_deaths': {'crude': mat_deaths_dalys[0], 'summarised': mat_deaths_dalys[1]},
                'neo_deaths': {'crude': neo_deaths_dalys[0], 'summarised': neo_deaths_dalys[1]},
                'mmr': {'crude': mat_deaths_dalys[2], 'summarised': mat_deaths_dalys[3]},
                'nmr': {'crude': neo_deaths_dalys[2], 'summarised': neo_deaths_dalys[3]},
                'mat_dalys': {'crude': mat_deaths_dalys[4], 'summarised': mat_deaths_dalys[5]},
                'neo_dalys': {'crude': neo_deaths_dalys[4], 'summarised': neo_deaths_dalys[5]},
                'total_dalys': {'crude': total_dalys, 'summarised': total_dalys_sum},
                'total_stillbirths': {'crude': stillbirths, 'summarised': stillbirths_summ},
                'sbr': {'crude': sbr, 'summarised': sbr_summ},
                'adj_total_dalys': {'crude': adjusted_dalys, 'summarised': adj_dalys_summ},
                })

# Summarised results
def get_data(df, key, draw):
    return (df.loc[key, (draw, 'lower')],
            df.loc[key, (draw, 'mean')],
            df.loc[key, (draw, 'upper')])

mat_dalys_by_scenario = {k: get_data(results['mat_dalys']['summarised'], 'Maternal Disorders', d) for k, d in zip (
    int_analysis, draws)}
neo_dalys_by_scenario = {k: get_data(results['neo_dalys']['summarised'], 'Neonatal Disorders', d) for k, d in zip (
    int_analysis, draws)}

mmr_by_scnario = {k: get_data(results['deaths_and_stillbirths']['summarised'], 'direct_mmr', d) for k, d in zip (
    int_analysis, draws)}
nmr_by_scnario = {k: get_data(results['deaths_and_stillbirths']['summarised'], 'nmr', d) for k, d in zip (
    int_analysis, draws)}

mmr_by_scenario_oth_log = {k: get_data(results['mat_deaths']['summarised'], 2025, d) for k, d in zip (
    int_analysis, draws)}
nmr_by_scenario_oth_log = {k: get_data(results['neo_deaths']['summarised'], 2025, d) for k, d in zip (
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
    plt.savefig(f'{g_path}/{title}.png', bbox_inches='tight')
    plt.show()

barcharts(mat_dalys_by_scenario, 'DALYs', 'Total Maternal Disorders DALYs by scenario')
barcharts(neo_dalys_by_scenario, 'DALYs', 'Total Neonatal Disorders DALYs by scenario')

barcharts(mmr_by_scnario, 'MMR', 'Total MMR by scenario')
barcharts(nmr_by_scnario, 'MMR', 'Total NMR by scenario')

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

def get_diffs_demog_log(mr_df):
    diff_results = {}
    baseline = mr_df['crude'][0]

    for draw, int in zip(draws, int_analysis):
        diff_df = mr_df['crude'][draw] - baseline
        diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
                                                        names=['draw', 'run'])
        results_diff = summarize_confidence_intervals(diff_df)
        results_diff.fillna(0)
        diff_results.update({int: results_diff.loc[2025].values})

    return diff_results


mat_deaths = get_diffs('deaths_and_stillbirths', 'direct_maternal_deaths', int_analysis, draws)[0]
neo_deaths = get_diffs('deaths_and_stillbirths', 'neonatal_deaths', int_analysis, draws)[0]

mmr_diffs = get_diffs('deaths_and_stillbirths', 'direct_mmr', int_analysis, draws)[0]
nmr_diffs = get_diffs('deaths_and_stillbirths', 'nmr', int_analysis, draws)[0]

mat_dalys_diffs = get_diffs('mat_dalys', 'Maternal Disorders', int_analysis, draws)[0]
neo_dalys_diffs = get_diffs('neo_dalys', 'Neonatal Disorders', int_analysis, draws)[0]

mat_deaths_2 = get_diffs_demog_log(results['mmr'])
neo_deaths_2 = get_diffs_demog_log(results['nmr'])

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
    plt.savefig(f'{g_path}/{outcome}.png', bbox_inches='tight')

    plt.show()

get_diff_plots(mmr_diffs, 'MMR')
get_diff_plots(nmr_diffs, 'NMR')

get_diff_plots(mat_deaths, 'Maternal Deaths (crude)')
get_diff_plots(neo_deaths, 'Neonatal Deaths (crude)')

get_diff_plots(mat_deaths_2, 'MMR (demog log)')
get_diff_plots(neo_deaths_2, 'NMR (demog log)')

get_diff_plots(mat_dalys_diffs, 'Maternal DALYs')
get_diff_plots(neo_dalys_diffs, 'neonatal DALYs')

# COST CALCULATIONS

# TEST COSTS DIFFER FROM BASELINE
TARGET_PERIOD = (Date(2025, 1, 1), Date(2025, 12, 31))
list_of_relevant_years_for_costing = list(range(TARGET_PERIOD[0].year, TARGET_PERIOD[-1].year + 1))


# input_costs_df = estimate_input_cost_of_scenarios(results_folder=results_folder,
#                                      resourcefilepath=resourcefilepath,
#                                      suspended_results_folder=results_folder,
#                                      _draws=draws,
#                                      _years=list_of_relevant_years_for_costing,
#                                      cost_only_used_staff= True,
#                                      _discount_rate=0.03)
#
# input_costs_df.to_csv(f'{g_path}/input_costs.csv')

# Read in costs (takes a long time to generate)
# TODO: are these results scaled?
input_costs = pd.read_csv(f'{g_path}/input_costs.csv')
input_costs = input_costs.set_index('Unnamed: 0')

cost_by_draw_and_year = input_costs.groupby(['draw', 'run', 'year'])['cost'].sum()
cost_by_draw_and_year_df = cost_by_draw_and_year.reset_index().pivot(index='year', columns=['draw','run'], values='cost')

cost_diff = {}
baseline = cost_by_draw_and_year_df[0]

for draw, intervention in zip(draws, int_analysis):
    diff_df = cost_by_draw_and_year_df[draw] - baseline
    diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
                                                names=['draw', 'run'])
    results_diff = summarize_confidence_intervals(diff_df)
    cost_diff.update({intervention: results_diff.values.flatten().tolist()})

get_diff_plots(cost_diff, 'Cost')

medical_consumables_df = (
    input_costs[input_costs["cost_category"].eq("medical consumables")]
      .pivot_table(
          index="year",
          columns=["draw", "run"],
          values="cost",
          aggfunc="sum"
      )
      .sort_index(axis=1))

# TODO make more robust so we dont have to define year
medical_consumables_summ = summarize_confidence_intervals(medical_consumables_df)
medical_cons_by_scenario = {k: [medical_consumables_summ.loc[2025, (d, 'lower')],
                                medical_consumables_summ.loc[2025, (d, 'mean')],
                                medical_consumables_summ.loc[2025, (d, 'upper')]] for k, d in zip (int_analysis, draws)}
barcharts(medical_cons_by_scenario, 'Cost (USD)', 'Total Consumables Costs')

# todo adjust consumables coming from MRP, investiage why cons reduction for newborn resus (could we look at cons breakdown) ANSWER FOR MRP (BLOOD AND BENPEN ARE DRIVING THE COST DIFFERENCE)



# TODO: THIS CODE WILL EXTRACT DIFFERENCE BETWEEN BASELINE AND USED CONS IN CONS UNIT

from collections import defaultdict
def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

def get_counts_of_items_requested(_df):
    _df = drop_outside_period(_df)

    counts_of_available = defaultdict(int)
    counts_of_not_available = defaultdict(int)

    for _, row in _df.iterrows():
        for item, num in row['Item_Used'].items():
            counts_of_available[item] += num
        for item, num in row['Item_NotAvailable'].items():  # eval(row['Item_NotAvailable'])
            counts_of_not_available[item] += num

    return pd.concat(
        {'Used': pd.Series(counts_of_available), 'Not_Available': pd.Series(counts_of_not_available)},
        axis=1
    ).fillna(0).astype(int).stack()


cons_req = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Consumables',
        custom_generate_series=get_counts_of_items_requested,
        do_scaling=True) # todo change to False

cons_req.fillna(0, inplace=True)

cons_costs = pd.read_csv(Path("./resources/costing") /'ResourceFile_Costing_Consumables.csv')

# costed_scenario_cons = cons_req.copy()
price_lookup = (
    cons_costs
    .assign(Item_Code=cons_costs["Item_Code"].astype(str).str.strip())
    .set_index("Item_Code")["Price_per_unit"])
item_codes = (
    cons_req.index.get_level_values(0)
    .astype(str)
    .str.strip())
prices = item_codes.map(price_lookup)
missing_price_mask = prices.isna()
multipliers = prices.fillna(1)
costed_scenario_cons = cons_req.mul(multipliers.to_numpy(), axis=0)

diff_results = {}

def get_cost_used_cons(level):
    return costed_scenario_cons.loc[costed_scenario_cons.index.get_level_values(1) == "Used"].xs(level, level=0, axis=1)

baseline = get_cost_used_cons(0)
for draw, int in zip(draws, int_analysis):
     diff_df = get_cost_used_cons(draw) - baseline
     diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
                                                 names=['draw', 'run'])
     diff_df.index =  diff_df.index.droplevel(1)
     results_diff = summarize_confidence_intervals(diff_df)
     results_diff.fillna(0)
     diff_results.update({int: results_diff})

for k in diff_results.keys():

    if k != "baseline":

        # Extract values
        categories = np.array(list(diff_results[k].index))

        mins = np.array([arr[0] for arr in diff_results[k].values])
        means = np.array([arr[1] for arr in diff_results[k].values])
        maxs = np.array([arr[2] for arr in diff_results[k].values])

        # Sort by mean difference
        order = np.argsort(means)

        categories = categories[order]
        mins = mins[order]
        means = means[order]
        maxs = maxs[order]

        y = np.arange(len(categories))

        # Error bars
        errors = np.vstack([
            means - mins,
            maxs - means
        ])

        # Identify top/bottom 10
        bottom10 = np.argsort(means)[:10]
        top10 = np.argsort(means)[-10:]

        # Plot
        fig, ax = plt.subplots(figsize=(11, 34))

        # All consumables
        ax.errorbar(
            means,
            y,
            xerr=errors,
            fmt='o',
            color='lightgrey',
            ecolor='lightgrey',
            markersize=3,
            capsize=2,
            elinewidth=0.8,
            linewidth=0.8,
            label='Other consumables'
        )

        # Largest decreases
        ax.errorbar(
            means[bottom10],
            y[bottom10],
            xerr=errors[:, bottom10],
            fmt='o',
            color='red',
            ecolor='red',
            markersize=5,
            capsize=3,
            elinewidth=1,
            label='10 largest decreases'
        )

        # Largest increases
        ax.errorbar(
            means[top10],
            y[top10],
            xerr=errors[:, top10],
            fmt='o',
            color='green',
            ecolor='green',
            markersize=5,
            capsize=3,
            elinewidth=1,
            label='10 largest increases'
        )

        # Baseline reference line
        ax.axvline(0, color='black', linestyle='--', linewidth=1)

        # Show all consumable IDs
        ax.set_yticks(y)
        ax.set_yticklabels(categories, fontsize=7)

        # Labels/title
        ax.set_xlabel("Crude difference from baseline scenario")
        ax.set_ylabel("Consumable item")
        ax.set_title(
            f"Difference in Cost per Consumable from Baseline Scenario vs {k}"
        )

        # Grid only on x-axis
        ax.grid(axis="x", alpha=0.35)

        # Legend
        ax.legend(loc="best")

        # Save and show
        fig.tight_layout()

        fig.savefig(
            f"{g_path}/cons_cost_diff_{k}_horizontal_highlighted.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

# def get_tornado_plot(data, outcome):
#     grouped_data = {}
#     data.pop('baseline', None)
#
#     for key in data.keys():
#         base_key = key.rsplit('_', 1)[0] if key.endswith('_max') or key.endswith('_min') else key
#
#         if base_key not in grouped_data:
#             grouped_data[base_key] = {'min': None, 'max': None}
#         if 'min' in key:
#             grouped_data[base_key]['min'] = data[key]
#         elif 'max' in key:
#             grouped_data[base_key]['max'] = data[key]
#
#     # Prepare data for plotting
#     categories = list(grouped_data.keys())
#     min_values = [np.mean(grouped_data[cat]['min']) for cat in categories]
#     max_values = [np.mean(grouped_data[cat]['max']) for cat in categories]
#
#     # Extracting uncertainty intervals (first and third values in each array)
#     min_lower = [grouped_data[cat]['min'][0] for cat in categories]
#     min_upper = [grouped_data[cat]['min'][2] for cat in categories]
#     max_lower = [grouped_data[cat]['max'][0] for cat in categories]
#     max_upper = [grouped_data[cat]['max'][2] for cat in categories]
#
#     # Calculate error bars (distance from mean to bounds)
#     min_errors = [np.abs(np.array(min_values) - np.array(min_lower)),
#                   np.abs(np.array(min_upper) - np.array(min_values))]
#     max_errors = [np.abs(np.array(max_values) - np.array(max_lower)),
#                   np.abs(np.array(max_upper) - np.array(max_values))]
#
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     y_positions = np.arange(len(categories))
#
#     bars_min = plt.barh(y_positions, min_values, color='lightcoral', edgecolor='black', alpha=0.7, label='Min Effect')
#     bars_max = plt.barh(y_positions, max_values, color='skyblue', edgecolor='black', alpha=0.7, label='Max Effect')
#
#     # Add error bars for uncertainty intervals
#     plt.errorbar(min_values, y_positions, xerr=min_errors, fmt='none', ecolor='darkred', capsize=5, alpha=0.9,
#                  label='Uncertainty (Min)')
#     plt.errorbar(max_values, y_positions, xerr=max_errors, fmt='none', ecolor='navy', capsize=5, alpha=0.9,
#                  label='Uncertainty (Max)')
#
#     # Central zero line
#     plt.axvline(0, color='black', linewidth=1, linestyle='--')
#
#     # Add labels
#     plt.yticks(y_positions, categories)
#     plt.xlabel(f'Difference in {outcome} from Status Quo')
#     plt.title(f'Tornado Plot showing current and potential impact of interventions on {outcome}')
#     plt.legend()
#
#     plt.savefig(f'{g_path}/{outcome}_tornado.png', bbox_inches='tight')
#     plt.show()
#
#
# get_tornado_plot(mat_deaths_2, 'MMR')
# get_tornado_plot(neo_deaths_2, 'NMR')
#
# get_tornado_plot(mat_dalys_diffs, 'Maternal DALYs')
# get_tornado_plot(neo_dalys_diffs, 'Neonatal DALYs')

# Table 1
# def get_table_one():
#     columns = ['age_years', 'la_parity', 'region_of_residence', 'li_wealth', 'li_bmi', 'li_mar_stat', 'li_ed_lev',
#                 'li_urban', 'ps_prev_spont_abortion', 'ps_prev_stillbirth', 'ps_prev_pre_eclamp', 'ps_prev_gest_diab']
#     categorical = ['region_of_residence', 'li_wealth', 'li_bmi' ,'li_mar_stat', 'li_ed_lev', 'li_urban',
#                     'ps_prev_spont_abortion', 'ps_prev_stillbirth', 'ps_prev_pre_eclamp', 'ps_prev_gest_diab']
#     continuous = ['age_years', 'la_parity']
#
#     rename = {'age_years': 'Age (years)',
#                'la_parity': 'Parity',
#                'region_of_residence': 'Region',
#                'li_wealth': 'Wealth Quintile',
#                'li_bmi': 'BMI level',
#                'li_mar_stat': 'Marital Status',
#                'li_ed_lev': 'Education Level',
#                'li_urban': 'Urban/Rural',
#                'ps_prev_spont_abortion': 'Previous Miscarriage',
#                'ps_prev_stillbirth': 'Previous Stillbirth',
#                'ps_prev_pre_eclamp': 'Previous Pre-eclampsia',
#                'ps_prev_gest_diab': 'Previous Gestational Diabetes',
#               }
#
#     all_preg_df = pd.read_excel(Path("./resources/ResourceFile_MaternalCohort") /
#                                         'ResourceFile_All2025PregnanciesCohortModel.xlsx')
#     population = 40_000
#
#     # Only select rows equal to the desired population size
#     if population <= len(all_preg_df):
#         preg_pop = all_preg_df.loc[0:population-1]
#     else:
#         # Calculate the number of rows needed to reach the desired length
#         additional_rows = population - len(all_preg_df)
#
#         # Initialize an empty DataFrame for additional rows
#         rows_to_add = pd.DataFrame(columns=all_preg_df.columns)
#
#         # Loop to fill the required additional rows
#         while additional_rows > 0:
#             if additional_rows >= len(all_preg_df):
#                 rows_to_add = pd.concat([rows_to_add, all_preg_df], ignore_index=True)
#                 additional_rows -= len(all_preg_df)
#             else:
#                 rows_to_add = pd.concat([rows_to_add, all_preg_df.iloc[:additional_rows]], ignore_index=True)
#                 additional_rows = 0
#
#         # Concatenate the original DataFrame with the additional rows
#         preg_pop = pd.concat([all_preg_df, rows_to_add], ignore_index=True)
#
#     mytable = TableOne(preg_pop[columns], categorical=categorical,
#                        continuous=continuous, rename=rename, pval=False)
#     print(mytable.tabulate(tablefmt = "fancy_grid"))
#     mytable.to_excel(Path(f"{outputspath}/{scenario}/0/table_one.xlsx") )
