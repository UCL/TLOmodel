from pathlib import Path

from collections import Counter, defaultdict

import os
import scipy.stats as st
from jinja2 import pass_environment
from pandas import read_excel
from scipy.stats import t, norm, shapiro

import pandas as pd
import tableone
from tableone import TableOne

import matplotlib.pyplot as plt
import numpy as np

from typing import Callable, Dict, Iterable, List, Literal, Optional, TextIO, Tuple, Union

from tlo import Date
from tlo.analysis.utils import (bin_hsi_event_details, extract_results, extract_params,
                                get_scenario_outputs, compute_summary_statistics,
                                make_age_grp_types, get_scenario_info, make_calendar_period_lookup, make_calendar_period_type, parse_log_file)

from src.scripts.costing.cost_estimation import estimate_input_cost_of_scenarios, do_stacked_bar_plot_of_cost_by_category, summarize_cost_data

plt.style.use('seaborn-darkgrid')

# Get results file
resourcefilepath = Path("./resources")

outputspath = './outputs/sejjj49@ucl.ac.uk/'
scenario = 'integration_scenario_max_test_2462999'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]

# Create a dict of {run: 'scenario'} from the updated parameters
params = extract_params(results_folder)
subset = params[params['module_param'] == ('ServiceIntegration:serv_integration')]
p_dict = subset.drop(columns='module_param').to_dict()
scen_draws = p_dict['value']

# create output folder for graphs
g_path = f'{outputspath}graphs_{scenario}_test'
if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}graphs_{scenario}_test')

dalys_folder = f'{g_path}/DALYs'
if not os.path.isdir(dalys_folder):
    os.makedirs(f'{g_path}/DALYs')

hsi_folder = f'{g_path}/HSIs'
if not os.path.isdir(hsi_folder):
    os.makedirs(f'{g_path}/HSIs')

cons_folder = f'{g_path}/Consumables'
if not os.path.isdir(cons_folder):
    os.makedirs(f'{g_path}/Consumables')

# Define target period
TARGET_PERIOD = (Date(2020, 1, 1), Date(2050, 12, 31))

# =================================================HELPER FUNCTIONS ===================================================
# def get_dalys_by_period_sex_agegrp_label(df):
#     """Sum the dalys by period, sex, age-group and label"""
#     df['age_grp'] = df['age_range'].astype(make_age_grp_types())
#     df = df.drop(columns=['date', 'age_range', 'sex'])
#     df = df.groupby(by=["year", "age_grp"]).sum().stack()
#     df.index = df.index.set_names('label', level=2)
#     return df
#
# def get_pop_by_agegrp_label(df):
#     """Sum the dalys by period, sex, age-group and label"""
#     df['year'] = df['date'].dt.year
#     df_melted = df.melt(id_vars=['year'], value_vars=[col for col in df.columns if col not in ['date', 'year']],
#                         var_name='age_group', value_name='count')
#     series_multi = df_melted.set_index(['year', 'age_group'])['count'].sort_index()
#
#     return series_multi
#
# def get_percentage_diff(df):
#     percent_diff = df.copy()
#     for col in df.columns:
#         if col[0] != 0:
#             # Get corresponding (0, col[1]) for comparison
#             base_col = (0, col[1])
#             percent_diff[col] = (df[col] - df[base_col]) / df[base_col] * 100
#         else:
#             percent_diff[col] = 0  # or np.nan if you prefer
#
#     pdiff_sum = compute_summary_statistics(percent_diff)
#     return pdiff_sum
#
# def compute_service_statistics(counters_by_draw_and_run):
#     grouped_data = defaultdict(lambda: defaultdict(list))
#
#     # Step 1: Group counts by first key and service name
#     for (group_idx, _), counter in counters_by_draw_and_run.items():
#         for service_name, count in counter.items():
#             grouped_data[group_idx][service_name].append(count)
#
#     data_df = pd.DataFrame.from_dict(grouped_data)
#
#     def safe_sum_lists(series):
#         # Filter out non-list values (like float/NaN)
#         valid_lists = [x for x in series if isinstance(x, list)]
#         if not valid_lists:
#             return np.nan  # or return [0]*length if you want default
#         return [sum(items) for items in zip(*valid_lists)]
#
#     def pct_diff_from_col0(df):
#         def pct_diff_row(row):
#             base = row[0]
#             result = {}
#             for col in df.columns:
#                 if col == 0:
#                     result[col] = np.nan  # or keep base if needed
#                 else:
#                     val = row[col]
#                     if not isinstance(base, list) or not isinstance(val, list):
#                         result[col] = np.nan
#                     else:
#                         result[col] = [(v - b) / b * 100 if b != 0 else np.nan for v, b in zip(val, base)]
#             return pd.Series(result)
#
#         return df.apply(pct_diff_row, axis=1)
#
#     # Run the function
#     # Apply to entire DataFrame grouped by level
#     appt_type = data_df.groupby(level=1).agg(safe_sum_lists)
#     pdiff_appt_type = pct_diff_from_col0(appt_type)
#
#     width_of_range = 0.95
#     lower_quantile = (1. - width_of_range) / 2.
#
#     def summarize_list(cell):
#         arr = np.array(cell)
#         return {
#             "median": float(np.median(arr)),
#             "lower": float(np.quantile(arr, lower_quantile)),
#             "upper": float(np.quantile(arr, 1 - lower_quantile))
#         }
#     # Apply to every cell in the DataFrame
#     appt_type_summ = appt_type.applymap(summarize_list)
#     pdiff_appt_type_summ = pdiff_appt_type.applymap(summarize_list)
#
#     return appt_type_summ, pdiff_appt_type_summ
#
# def barcharts(data, y_label, title, by_cause, folder):
#     # Extract means and errors
#
#     if by_cause:
#         labels = data.index.values
#
#         median = data['central'].values
#         yerr_lower = median - data['lower'].values
#         yerr_upper = data['upper'].values - median
#
#     else:
#
#         labels = scen_draws.values()
#
#         median = [float(data[v, 'central'].values) for v in scen_draws.keys()]
#         lower_errors = [float(data[v, 'lower'].values) for v in scen_draws.keys()]
#         upper_errors = [float(data[v, 'upper'].values) for v in scen_draws.keys()]
#
#         # Compute distances from mean to bounds (must be non-negative)
#         yerr_lower = [med - low for med, low in zip(median, lower_errors)]
#         yerr_upper = [up - med for med, up in zip(median, upper_errors)]
#
#     # Create bar chart with error bars
#     fig, ax = plt.subplots()
#     ax.bar(labels, median, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black')
#
#     if by_cause:
#         ax.axhline(0, color='gray', linestyle='--', linewidth=1)
#
#     ax.set_ylabel(y_label)
#     ax.set_title(title)
#
#     # Adjust label size
#     plt.xticks(fontsize=8, rotation=90)
#     plt.tight_layout()
#     if by_cause and y_label.endswith('(Weighted)'):
#         plt.savefig(f'{folder}/wtd_pdfiff_{scen_draws[k]}.png', bbox_inches='tight')
#
#     elif by_cause and not y_label.endswith('(Weighted)'):
#         plt.savefig(f'{folder}/wpdfiff_{scen_draws[k]}.png', bbox_inches='tight')
#
#     else:
#         plt.savefig(f'{folder}/{title}.png', bbox_inches='tight')
#
#     plt.show()
#
#
# def grouped_bar_chart(df, draw, title, ylabel, folder):
#     categories = df.index
#     x = np.arange(len(categories))
#     width = 0.35
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     baseline_central = df[0]['central']
#     baseline_err_lower = baseline_central - df[0]['lower']
#     baseline_err_upper = df[0]['upper'] - baseline_central
#
#     # Comparison
#     comp_central = df[draw]['central']
#     comp_err_lower = comp_central - df[draw]['lower']
#     comp_err_upper = df[draw]['upper'] - comp_central
#
#     # Plot bars with asymmetric error bars
#     ax.bar(x - width / 2, baseline_central, width,
#            yerr=[baseline_err_lower, baseline_err_upper],
#            capsize=5, label='Status Quo', alpha=0.8)
#
#     ax.bar(x + width / 2, comp_central, width,
#            yerr=[comp_err_lower, comp_err_upper],
#            capsize=5, label=scen_draws[draw], alpha=0.8)
#
#     ax.axhline(0, color='gray', linestyle='--', linewidth=1)
#
#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.set_xticks(x)
#     ax.set_xticklabels(categories, rotation=45, ha='right')
#     ax.legend()
#     ax.grid(axis='y', linestyle='--', alpha=0.4)
#     plt.tight_layout()
#     if ylabel.startswith('Weighted'):
#         plt.savefig(f'{folder}/{scen_draws[draw]}_wtd_dalys_cause.png', bbox_inches='tight')
#     else:
#         plt.savefig(f'{folder}/{scen_draws[draw]}_dalys_cause.png', bbox_inches='tight')
#
#     plt.show()


# ==================================================== DALYS ==========================================================
# taking the numbers of DALYS by age-group, weighting them by the proportion of the pop in that age-group and
# summing to get a weighted (and more representative) total number of DALYS. This can be done by condition or overall
# if you like.

# dalys_by_age_date_and_cause = extract_results(
#                 results_folder,
#                 module="tlo.methods.healthburden",
#                 key="dalys_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
#                 custom_generate_series=get_dalys_by_period_sex_agegrp_label,
#                 do_scaling=False
#             )
# dalys_by_age_date_and_cause.index = dalys_by_age_date_and_cause.index.set_names('age_group', level=1)
#
# # Get total DALYs by cause across the intervention period (including % diff from status quo)
# dalys_by_year_cause = dalys_by_age_date_and_cause.groupby(by=["year", "label"]).sum()
# dalys_by_year_cause_int_period = dalys_by_year_cause.loc[TARGET_PERIOD[0].year:TARGET_PERIOD[-1].year]
# total_dalys_by_year_cause =  dalys_by_year_cause_int_period.groupby('label').sum()
# total_dalys_by_year_summ = compute_summary_statistics(total_dalys_by_year_cause)
# pdiff_dalys_by_cause = get_percentage_diff(total_dalys_by_year_cause)
#
# # Get total dalys per scenario (unweighted)
# dalys_by_age_date = dalys_by_age_date_and_cause.groupby(by=["year", "age_group"]).sum()
# dalys_unweighted_year = dalys_by_age_date.groupby(by='year').sum()
# total_dalys_unweighted = dalys_unweighted_year.loc[TARGET_PERIOD[0].year:TARGET_PERIOD[-1].year].sum().to_frame().T
# total_dalys_unweighted_summ = compute_summary_statistics(total_dalys_unweighted)
# pdiff_total_dalys_unweighted = get_percentage_diff(total_dalys_unweighted)
#
# # Get total dalys per scenario (weighted by population size across age groups)
# pop_f = extract_results(
#                 results_folder,
#                 module="tlo.methods.demography",
#                 key="age_range_f",
#                 custom_generate_series=get_pop_by_agegrp_label,
#                 do_scaling=False
#             )
#
# pop_m = extract_results(
#                 results_folder,
#                 module="tlo.methods.demography",
#                 key="age_range_m",
#                 custom_generate_series=get_pop_by_agegrp_label,
#                 do_scaling=False
#             )
#
# pop = pop_f + pop_m
# proportion_df = pop.div(pop.groupby(level='year').transform('sum'))
#
# # get weighted dalys by cause
# prop_df_aligned = proportion_df.reindex(dalys_by_age_date_and_cause.index.droplevel('label'))
# dalys_by_cause_age_weighted = dalys_by_age_date_and_cause * prop_df_aligned.values
# d_by_cause_int_period = dalys_by_cause_age_weighted.loc[TARGET_PERIOD[0].year:TARGET_PERIOD[-1].year]
# dalys_by_cause_weighted = d_by_cause_int_period.groupby(level='label').sum()
# dalys_by_cause_weighted_summ = compute_summary_statistics(dalys_by_cause_weighted)
# pdiff_dalys_by_cause_weighted =get_percentage_diff(dalys_by_cause_weighted)
#
# # get weighted total dalys
# weighted_dalys = proportion_df * dalys_by_age_date
# total_weighted_dalys = weighted_dalys.groupby(level='year').sum()
# twd_int_period = total_weighted_dalys.loc[TARGET_PERIOD[0].year:TARGET_PERIOD[-1].year].sum().to_frame().T
# total_weighted_dalys_summ = compute_summary_statistics(twd_int_period)
# pdiff_weighted_dalys_sum = get_percentage_diff(twd_int_period)
#
# # Output and save plots
# # Non-weighted DALYs
# barcharts(total_dalys_unweighted_summ, 'Total Population DALYs',
#           'Total Population DALYs by Scenario', False, g_path)
#
# barcharts(pdiff_total_dalys_unweighted, 'Percentage Diff. Population DALYs',
#           'Percentage Difference from Status Quo for Total DALYs by Scenario', False,
#           g_path)
#
# # Weighted DALYs
# barcharts(pdiff_weighted_dalys_sum, 'Percentage Diff. Population Weighted DALYs',
#           'Percentage Difference from Status Quo for Population Weighted DALYs by Scenario', False, g_path)
#
# barcharts(total_weighted_dalys_summ, 'Total Population Weighted DALYs',
#           'Total Population Weighted DALYs by Scenario', False, g_path)
#
#
#
# # Weighted and Non-weighted DALYs by cause
# for k in scen_draws:
#     if k == 0:
#         pass
#     else:
#         grouped_bar_chart(total_dalys_by_year_summ, k,
#                           f'DALYs by Cause: Status Quo vs {scen_draws[k]}', 'DALYs', dalys_folder)
#         grouped_bar_chart(dalys_by_cause_weighted_summ, k,
#                           f'Weighted DALYs by Cause: Status Quo vs {scen_draws[k]}',
#                           'Weighted DALYs', dalys_folder)
#
#         barcharts(pdiff_dalys_by_cause_weighted[k],
#                   'Percentage Difference (Weighted)',
#                   f'P.diff Weighted DALYs by cause compared to Status Quo: {scen_draws[k]}',
#                   True, dalys_folder)
#
#         barcharts(pdiff_dalys_by_cause[k],
#                   'Percentage Difference',
#                   f'P.diff DALYs by cause compared to Status Quo: {scen_draws[k]}',
#                   True, dalys_folder)
#
# # ============================================= HCW TIME/APPOINTMENTS =================================================
# # I think presenting numbers of appointments by the appointment type may be neater (U5 OPD, O5 OPD etc), could then
# # behind the scenes breakdown into XX% were for HIV services, CMD services etc.  Again, only if needed.
#
# counts_by_treatment_id_and_appt_type =  bin_hsi_event_details(
#             results_folder,
#             lambda event_details, count: sum(
#                 [
#                     Counter({
#                         (
#                             event_details["treatment_id"],
#                             appt_type
#                         ):
#                         count * appt_number
#                     })
#                     for appt_type, appt_number in event_details["appt_footprint"]
#                 ],
#                 Counter()
#             ),
#             *TARGET_PERIOD,
#             False
#         )
# apt_data = compute_service_statistics(counts_by_treatment_id_and_appt_type)
# apt_type_summ = apt_data[0]
# pdiff_apt_type_sum = apt_data[1]
#
# labels = apt_type_summ.index
# sq_data = [[apt_type_summ.at[appt, 0]['median'] for appt in labels],
#            [apt_type_summ.at[appt, 0]['lower'] for appt in labels],
#            [apt_type_summ.at[appt, 0]['upper'] for appt in labels]
#            ]
# sq_yerr_lower = [med - low for med, low in zip(sq_data[0], sq_data[1])]
# sq_yerr_upper = [up - med for med, up in zip(sq_data[0], sq_data[2])]
#
# for k in scen_draws:
#     median = [apt_type_summ.at[appt, k]['median'] for appt in labels]
#     lower_errors = [apt_type_summ.at[appt, k]['lower'] for appt in labels]
#     upper_errors = [apt_type_summ.at[appt, k]['upper'] for appt in labels]
#
#     yerr_lower = [med - low for med, low in zip(median, lower_errors)]
#     yerr_upper = [up - med for med, up in zip(median, upper_errors)]
#
#     x = np.arange(len(labels))
#     width = 0.35
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # Plot bars with asymmetric error bars
#     ax.bar(x - width / 2, sq_data[0], width,
#            yerr=[sq_yerr_lower, sq_yerr_upper],
#            capsize=5, label='Status Quo', alpha=0.8)
#
#     ax.bar(x + width / 2, median, width,
#            yerr=[yerr_lower, yerr_upper],
#            capsize=5, label=scen_draws[k], alpha=0.8)
#
#     ax.set_title(f'Total Appointment Types: Status Quo vs {scen_draws[k]}')
#     ax.set_ylabel('Number of Appointments')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=45, ha='right')
#     ax.legend()
#     ax.grid(axis='y', linestyle='--', alpha=0.4)
#     plt.tight_layout()
#     plt.savefig(f'{hsi_folder}/{scen_draws[k]}_appt_types.png', bbox_inches='tight')
#     plt.show()
#
#     # do percentaga diff from SQ
#     median = [pdiff_apt_type_sum.at[appt, k]['median'] for appt in labels]
#     lower_errors = [pdiff_apt_type_sum.at[appt, k]['lower'] for appt in labels]
#     upper_errors = [pdiff_apt_type_sum.at[appt, k]['upper'] for appt in labels]
#
#     yerr_lower = [med - low for med, low in zip(median, lower_errors)]
#     yerr_upper = [up - med for med, up in zip(median, upper_errors)]
#
#     # Create bar chart with error bars
#     fig, ax = plt.subplots()
#     ax.bar(labels, median, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black')
#     ax.axhline(0, color='gray', linestyle='--', linewidth=1)
#     ax.set_ylabel('Percentage Difference')
#     ax.set_title(f'P Diff. from Status Quo of Number of Appointments: {scen_draws[k]}')
#
#     # Adjust label size
#     plt.xticks(fontsize=8, rotation=90)
#     plt.tight_layout()
#     plt.savefig(f'{hsi_folder}/pdiff_appt_{scen_draws[k]}.png', bbox_inches='tight')
#     plt.show()


# ================================================== CONSUMABLES ======================================================
# Wonder if instead of looking at consumables use by category, you could just look at the relative consumables costs
# using Sakshi's module? This breaks down costs into cons and HR etc so you could possibly just say scenario X led to
# additional cons costs of $Y million!?

# I think I have a list somewhere of the cons required for HIV, TB and malaria services. These may not be up-to-date
# though. Could we put all cons availability and run the scenarios under perfect cons as a comparator to default cons?
# I know this will impact everything outside of your integration scenarios too - with unplanned downstream effects -
# but could still give a guide to optimal impact, like an upper bound...? I'm just thinking of the crudest approaches
# here to try and reduce your workload, feel free to push back here obviously!

list_of_relevant_years_for_costing = list(range(TARGET_PERIOD[0].year, TARGET_PERIOD[-1].year + 1))

input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
                                               _years=list_of_relevant_years_for_costing,
                                               cost_only_used_staff=True,
                                               _discount_rate = 0.03)


total_input_cost = input_costs.groupby(['draw', 'run'])['cost'].sum()
total_input_cost_summarized = summarize_cost_data(total_input_cost.unstack(level='run'))

def find_difference_relative_to_comparison(_ser: pd.Series,
                                           comparison: str,
                                           scaled: bool = False,
                                           drop_comparison: bool = True,
                                           ):
    """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
    within the runs (level 1), relative to where draw = `comparison`.
    The comparison is `X - COMPARISON`."""
    return _ser \
        .unstack(level=0) \
        .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
        .drop(columns=([comparison] if drop_comparison else [])) \
        .stack()

incremental_scenario_cost = (pd.DataFrame(
    find_difference_relative_to_comparison(
        total_input_cost,
        comparison=0)  # sets the comparator to 0 which is the Actual scenario
).T.iloc[0].unstack()).T


# First summarize all input costs
input_costs_for_plot_summarized = input_costs.groupby(['draw', 'year', 'cost_subcategory', 'Facility_Level', 'cost_subgroup', 'cost_category']).agg(
    mean=('cost', 'mean'),
    lower=('cost', lambda x: x.quantile(0.025)),
    upper=('cost', lambda x: x.quantile(0.975))
).reset_index()
input_costs_for_plot_summarized = input_costs_for_plot_summarized.melt(
    id_vars=['draw', 'year', 'cost_subcategory', 'Facility_Level', 'cost_subgroup', 'cost_category'],
    value_vars=['mean', 'lower', 'upper'],
    var_name='stat',
    value_name='cost'
)

do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'all',
_disaggregate_by_subgroup = False, _outputfilepath = Path(g_path),
_scenario_dict = scen_draws)

do_stacked_bar_plot_of_cost_by_category(_df = input_costs_for_plot_summarized, _cost_category = 'medical consumables',
_disaggregate_by_subgroup = False, _outputfilepath = Path(g_path),
_scenario_dict = scen_draws)

# ===================================================== OTHERS ========================================================



