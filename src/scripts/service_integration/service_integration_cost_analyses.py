from pathlib import Path

from collections import Counter, defaultdict

import os
import scipy.stats as st

import pandas as pd

import matplotlib.pyplot as plt
import math

import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker

import matplotlib.patheffects as pe


import numpy as np
import ast  # for safely parsing strings

import seaborn as sns

from tlo import Date
from tlo.analysis.utils import (bin_hsi_event_details, extract_results, extract_params,
                                get_scenario_outputs, compute_summary_statistics,
                                make_age_grp_types)

from src.scripts.costing.cost_estimation import (estimate_input_cost_of_scenarios,
                                                 do_stacked_bar_plot_of_cost_by_category,
                                                 summarize_cost_data)


plt.style.use('seaborn-v0_8')

# Get results folder
resourcefilepath = Path("./resources")
outputspath = './outputs/sejjj49@ucl.ac.uk/'
scenario = 'service_integration_scenario-2025-07-01T144012Z'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]

# Create a dict of {run: 'scenario'} from the updated parameters
params = extract_params(results_folder)
subset = params[params['module_param'] == ('ServiceIntegration:serv_integration')]
p_dict = subset.drop(columns='module_param').to_dict()
scen_draws = p_dict['value']

# create output folder for graphs
def make_folder(path):
    folder_path = path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

g_path = make_folder(f'{outputspath}graphs_{scenario}_final')

# create a dict with proper labels for each scenario
full_lab = {'htn':'Hypertension screening',
            'htn_max': 'Hypertension screening (max. cons)',
            'dm': 'Diabetes screening',
            'dm_max': 'Diabetes screening (max. cons)',
            'hiv': 'HIV screening',
            'hiv_max': 'HIV screening (max. cons)',
            'tb': 'Tb screening',
            'tb_max':'Tb screening (max. cons)',
            'mal':'Malnutrition screening',
            'mal_max':'Malnutrition screening (max. cons)',
            'fp_scr':'Family planning (WRA)',
            'fp_scr_max':'Family planning (WRA) (max. cons)',
            'anc': 'Antenatal care',
            'anc_max': 'Antenatal care (max.cons)',
            'pnc':'Postnatal care',
            'pnc_max':'Postnatal care (max. cons)',
            'fp_pn': 'Family planning (postnatal)',
            'fp_pn_max':'Family planning (postnatal) (max. cons)',
            'epi': 'EPI',
            'chronic_care': 'Chronic care services',
            'chronic_care_max': 'Chronic care services (max.)',
            'all_screening': 'All screening',
            'all_screening_max':'All screening (max. cons)',
            'all_mch': 'MCH services',
            'all_mch_max': 'MCH services (max. cons)',
            'all_int': 'All services',
            'all_int_max': 'All services (max. cons)'}

def get_ratios():

    # TODO DELETE WHEN MOVED INTO MAIN SCRIPT

    appointment_time_table = pd.read_csv(
        resourcefilepath
        / 'healthsystem'
        / 'human_resources'
        / 'definitions'
        / 'ResourceFile_Appt_Time_Table.csv',
        index_col=["Appt_Type_Code", "Facility_Level", "Officer_Category"]
    )

    appt_type_facility_level_officer_category_to_appt_time = (
        appointment_time_table.Time_Taken_Mins.to_dict()
    )

    officer_categories = appointment_time_table.index.levels[
        appointment_time_table.index.names.index("Officer_Category")
    ].to_list()

    hcw_time_by_treatment_id = bin_hsi_event_details(
        results_folder,
        lambda event_details, count: sum(
            [
                Counter({
                    (
                        officer_category,
                        event_details["treatment_id"]
                    ):
                        count
                        * appt_number
                        * appt_type_facility_level_officer_category_to_appt_time.get(
                            (
                                appt_type,
                                event_details["facility_level"],
                                officer_category
                            ),
                            0
                        )
                    for officer_category in officer_categories
                })
                for appt_type, appt_number in event_details["appt_footprint"]
            ],
            Counter()
        ),
        *TARGET_PERIOD,
        True
    )

    # First we calculate average change in pop size
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
        key="age_range_f",
        custom_generate_series=get_pop_by_agegrp_label,
        do_scaling=True
    )

    pop_m = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="age_range_m",
        custom_generate_series=get_pop_by_agegrp_label,
        do_scaling=True
    )

    pop = pop_f + pop_m
    pop = pop.groupby(by='year').sum()
    relative_increase_df = pop.pct_change()
    avg_rel_increase = relative_increase_df.loc[2025:2054].mean(axis=0).to_frame().T
    avg_rel_increase_summ = compute_summary_statistics(avg_rel_increase, use_standard_error=True)

    # Next we calculate the total HCW time use
    hcw_time_by_treatment_id_df = pd.DataFrame.from_dict(hcw_time_by_treatment_id)
    hcw_time_by_treatment_id_df = hcw_time_by_treatment_id_df.fillna(0)
    hcw_time_by_treatment_id_df.index.names = ['first', 'second']
    hcw_time_by_cadre = hcw_time_by_treatment_id_df.groupby(level='first').sum()

    # Next we calculate HCW time by year
    annual_hcw_time_by_cadre = hcw_time_by_cadre / 30

    # Read in capabilities data and sum across facility levels etc.
    daily_cap = pd.read_csv('./resources/healthsystem/human_resources/actual/ResourceFile_Daily_Capabilities.csv')
    daily_mins = daily_cap.set_index('Officer_Category')[['Total_Mins_Per_Day']]
    daily_mins = daily_mins.drop('Dental')
    daily_mins = daily_mins.drop('Nutrition')
    daily_mins = daily_mins.groupby(daily_mins.index).sum()

    # Next we calculate the average HCW capabilities assuming capabilities increase yearly in line with population growth
    yrly_hcw_time_cap = daily_mins * 365.25
    value = 1 + avg_rel_increase_summ[(0, 'central')].values  # TODO: replace with average annual pop growth (SQ?)
    n_times = 30  # number of times to multiply
    steps = [(yrly_hcw_time_cap * (value ** i)) for i in range(n_times + 1)]  # if you want to include original
    pop_corrected_yearly_hcw_time_cap = sum(steps) / len(steps)

    # Now we calculate the ratio of time use to time available (by cadre) and summarise it
    hcw_time_ratio_by_cadre = annual_hcw_time_by_cadre.div(pop_corrected_yearly_hcw_time_cap.iloc[:, 0], axis=0)
    hcw_time_ratio_by_cadre.columns.names = ['draw', 'run']

    hcw_time_ratio_by_cadre_summ = compute_summary_statistics(hcw_time_ratio_by_cadre, use_standard_error=True)

    return hcw_time_ratio_by_cadre_summ

# ==================================== CONSUMABLE COST BY SCENARIO (AND DIFFS) ========================================
# https://github.com/UCL/TLOmodel/blob/ec8929949c694b3a503d34051575f0dc7e7a32c3/src/scripts/comparison_of_horizontal_and_vertical_programs/economic_analysis_for_manuscript/roi_analysis_horizontal_vs_vertical.py#L45
# 606 - 635, 1329-1348

TARGET_PERIOD = (Date(2025, 1, 1), Date(2054, 12, 31))

list_of_relevant_years_for_costing = list(range(TARGET_PERIOD[0].year, TARGET_PERIOD[-1].year + 1))

input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
                                               _years=list_of_relevant_years_for_costing,
                                               cost_only_used_staff=True,
                                               _discount_rate = 0.03)

# --------------------------- Adjust HCW costs based on average difference in HCW use ---------------------------------
# Get ratio of time use by cadre
hcw_ratios = get_ratios()

# Multiply the HCW cost estimates by ratios
central_df = hcw_ratios.xs('central', axis=1, level=1)
# Function to safely get multiplier
def get_multiplier(row):
    subgroup = row['cost_subgroup']
    draw = row['draw']
    if subgroup in central_df.index and draw in central_df.columns:
        return central_df.loc[subgroup, draw]
    else:
        return 1.0  # or np.nan, or row['cost'] unmodified depending on your logic
input_costs['cost'] = input_costs.apply(lambda row: row['cost'] * get_multiplier(row), axis=1)

#  Sum the total
total_input_cost = input_costs.groupby(['draw', 'run'])['cost'].sum()
total_input_cost_annual = total_input_cost / 30

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

incremental_scenario_cost_annual = (pd.DataFrame(
    find_difference_relative_to_comparison(
        total_input_cost_annual,
        comparison=0)  # sets the comparator to 0 which is the Actual scenario
).T.iloc[0].unstack()).T

# Plot incremental costs
incremental_scenario_cost_annual_summarized = summarize_cost_data(incremental_scenario_cost_annual)


def figure_avg_difference_in_cost_from_status_quo_per_year(cost_data):
    name_of_plot = 'Incremental scenario cost relative to baseline during intervention period'

    # === Error bars ===
    yerr = np.array([
        (cost_data['mean'] - cost_data['lower']).values,
        (cost_data['upper'] - cost_data['mean']).values,
    ])

    spacing = 1.55  # increase this value for more spacing
    xticks = {(i * spacing): k for i, k in enumerate(cost_data.index)}
    fig, ax = plt.subplots(figsize=(10, 5))

    # === Color mapping ===
    scenario_ids = cost_data.index.tolist()
    n_scenarios = len(scenario_ids)

    palette = sns.color_palette("husl", n_colors=n_scenarios)
    step = 10
    spread_indices = [(i * step) % n_scenarios for i in range(n_scenarios)]
    spread_palette = [palette[i] for i in spread_indices]

    color_map = {s: spread_palette[i] for i, s in enumerate(scenario_ids)}
    colors = [color_map[s] for s in scenario_ids]

    # === Bar chart ===
    ax.bar(
        xticks.keys(),
        cost_data['mean'].values,
        yerr=yerr,
        ecolor='black',
        capsize=10,
        label=[str(s) for s in scenario_ids],
        color=colors,
    )

    # === Format for currency annotation ===
    def format_currency(val):
        if abs(val) >= 1e9:
            return f"${val / 1e9:.1f}B"
        else:
            return f"${val / 1e6:.0f}M"

    # === Annotate bars ===
    for xpos, mean, lower, upper in zip(
        xticks.keys(),
        cost_data['mean'].values,
        cost_data['lower'].values,
        cost_data['upper'].values
    ):
        text = format_currency(mean)
        if mean >= 0:
            annotation_y = upper + 0.02 * 1e9
            valign = 'bottom'
        else:
            annotation_y = lower - 0.02 * 1e9
            valign = 'top'

        ax.text(
            xpos,
            annotation_y,
            text,
            ha='center',
            va=valign,
            fontsize='x-small',
            rotation='horizontal'
        )

    # === Axis + Labels ===
    ax.set_xticks(list(xticks.keys()))
    ax.set_xticklabels(full_lab.values())  # Assumes full_lab is defined externally
    plt.xticks(rotation=90, fontsize=7)

    ax.grid(axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Difference in annual cost')
    ax.set_ylim(bottom=-0.25 * 1e9)

    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.15, right=0.85)

    # === Save + Show ===
    fig.savefig(Path(g_path) / name_of_plot.replace(' ', '_').replace(',', ''), bbox_inches='tight')
    plt.show()


figure_avg_difference_in_cost_from_status_quo_per_year(incremental_scenario_cost_annual_summarized)

