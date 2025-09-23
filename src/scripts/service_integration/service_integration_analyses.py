# === Standard Library ===
import os
import datetime
from pathlib import Path
from collections import Counter, defaultdict

# === Third-Party Packages ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from scripts.epi.analysis_epi import fontsize
# === Local / Project-Specific Imports ===
from tlo import Date
from tlo.analysis.utils import (
    bin_hsi_event_details, extract_results, extract_params,
    get_scenario_outputs, compute_summary_statistics, make_age_grp_types
)
from tlo.analysis.life_expectancy import get_life_expectancy_estimates
from src.scripts.costing.cost_estimation import (
    estimate_input_cost_of_scenarios, summarize_cost_data
)

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
            'chronic_care_max': 'Chronic care services (max. cons)',
            'all_screening': 'All screening',
            'all_screening_max':'All screening (max. cons)',
            'all_mch': 'MCH services',
            'all_mch_max': 'MCH services (max. cons)',
            'all_int': 'All services',
            'all_int_max': 'All services (max. cons)'}

# -------------------------------------- HELPER FUNCTIONS ------------------------------------------------------------
def get_dalys_by_period_sex_agegrp_label(df):
    """Sum the dalys by period, sex, age-group and label"""
    df['age_grp'] = df['age_range'].astype(make_age_grp_types())
    df = df.drop(columns=['date', 'age_range', 'sex'])
    df = df.groupby(by=["year", "age_grp"]).sum().stack()
    df.index = df.index.set_names('label', level=2)
    return df

def get_diff(df, pdiff):
    """Returns summary statistics of either crude difference or percentage difference from SQ scenario"""
    diff = df.copy()
    for col in df.columns:
        if col[0] != 0:
            base_col = (0, col[1])
            if not pdiff:
                # Get corresponding (0, col[1]) for comparison
                diff[col] = df[base_col] - df[col]
            else:
                diff[col] = ((df[base_col] - df[col]) / df[base_col]) * 100
        else:
            diff[col] = 0  # or np.nan if you prefer

    diff_sum = compute_summary_statistics(diff, use_standard_error=True)
    return diff_sum

def get_full_pop():
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
    return pop


def age_standardize_dalys(dalys_df):
    """Age-standardizes DALYs across draws using the population size from the Status Quo scenario"""

    pop = get_full_pop()

    # ensure population and dalys dataframe indexes match
    if len(dalys_df.index.levels) == 3:
        pop_df = pop.reindex(dalys_df.index)
    else:
        pop_df = pop.reindex(dalys_df.index)

    # Define the 'reference population' - draw 0 is the status quo
    base_level = 0
    subset_cols = pop_df.columns.get_level_values(0) == base_level
    base_columns = pop_df.columns[subset_cols]

    # Drop rows without data
    pop_df  = pop_df.drop(TARGET_PERIOD[1].year + 1, errors='ignore')
    dalys_df = dalys_df.drop(TARGET_PERIOD[1].year + 1, errors='ignore')

    # check indexes are the same
    assert set(pop_df.index) == set(dalys_df.index)

    # Calculate dalys per person per age group across all draws
    dalys_per_person = dalys_df / pop_df
    dalys_per_person = dalys_per_person.fillna(0)
    dalys_age_standardized = dalys_per_person.copy()

    # Loop over each top-level column index
    for level in sorted(set(dalys_age_standardized.columns.get_level_values(0))):

        # Shift base_columns to this new level
        new_columns = [(level, col[1]) for col in base_columns]

        # Ensure these columns exist in both a and result
        if all(col in dalys_age_standardized.columns for col in new_columns):

            # Multiply corresponding columns
            dalys_age_standardized.loc[:, new_columns] = (dalys_age_standardized.loc[:, new_columns].values
                                                                * pop_df.loc[:, base_columns].values)

    # remove any NaN/inf values
    dalys_age_standardized.fillna(0, inplace=True)
    dalys_age_standardized.replace([np.inf, -np.inf], 0, inplace=True)

    return dalys_age_standardized

def compute_service_statistics(counters_by_draw_and_run):
    """Returns summary statistics for total HSI counts and difference in HSI counts from the SQ scenario"""

    grouped_data = defaultdict(lambda: defaultdict(list))

    # Group counts by first key and service name
    for (group_idx, _), counter in counters_by_draw_and_run.items():
        for service_name, count in counter.items():
            grouped_data[group_idx][service_name].append(count)

    data_df = pd.DataFrame.from_dict(grouped_data)

    def safe_sum_lists(series):
        # Filter out non-list values (like float/NaN)
        valid_lists = [x for x in series if isinstance(x, list)]
        if not valid_lists:
            return np.nan  # or return [0]*length if you want default
        return [sum(items) for items in zip(*valid_lists)]

    def p_diff_from_col0(df):
        # Calculates difference from the status quo
        def diff_row(row):
            base = row[0]
            result = {}
            for col in df.columns:
                if col == 0:
                    result[col] = np.nan  # or keep base if needed
                else:
                    val = row[col]
                    if not isinstance(base, list) or not isinstance(val, list):
                        result[col] = np.nan
                    else:
                        result[col] = [((v - b) / b) * 100 if b != 0 else np.nan for v, b in zip(val, base)]
            return pd.Series(result)

        return df.apply(diff_row, axis=1)

    # Run the functioion - apply to entire DataFrame grouped by level
    appt_type = data_df.groupby(level=1).agg(safe_sum_lists)
    diff_appt_type = p_diff_from_col0(appt_type)

    width_of_range = 0.95

    def summarize_list(cell):
        # Calculate mean/CIs (inline with other estimates)
        arr = np.array(cell)
        n = arr.size
        std_deviation = arr.std()
        std_error = std_deviation / np.sqrt(n)
        z_value = st.norm.ppf(1 - (1. - width_of_range) / 2.)

        mean = float(np.mean(arr))

        return {
            "mean": mean,
            "lower": mean - z_value * std_error,
            "upper": mean + z_value * std_error,
        }

    # Apply to every cell in the DataFrame
    appt_type_summ = appt_type.applymap(summarize_list)
    diff_appt_type_summ = diff_appt_type.applymap(summarize_list)

    return appt_type_summ, diff_appt_type_summ

def reform_df_to_save(df, dp):
    """Reformats a dataframe to be saved as CSV"""
    new_df = pd.DataFrame()

    for scenario in df.columns.levels[0]:
        central = round(df[(scenario, 'central')].values[0], dp)
        lower = round(df[(scenario, 'lower')].values[0], dp)
        upper = round(df[(scenario, 'upper')].values[0], dp)
        if scenario == 0:
            col = 'sq'
        else:
            col = scen_draws[scenario]
        new_df[col] = [(central, lower, upper)]

    return new_df

#  Define variables for plotting

scenario_groups = {
        'Integrated screening': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23],
        'MCH clinic': [13, 14, 15, 16, 17, 18, 19, 24, 25],
        'Chronic care clinic': [20, 21],
        'Combined': [26, 27]
    }

# extract data
scenario_groups_pathways_ = {
    'Integrated screening': [22, 23],
    'MCH clinic': [24, 25],
    'Chronic care clinic': [20, 21],
    'Combined': [26, 27]
}

# === Flatten scenarios in the order you want
ordered_scenario_ids = []
ordered_scenario_ids_pathways = []
for group in ['Integrated screening', 'MCH clinic', 'Chronic care clinic', 'Combined']:
    ordered_scenario_ids.extend(scenario_groups[group])
    ordered_scenario_ids_pathways.extend(scenario_groups_pathways_[group])

groupings = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [22, 23], [13, 14], [15, 16], [17, 18], [19], [24, 25],
              [20, 21], [26, 27]]

# Colour pallet
n_scenarios_total = len(ordered_scenario_ids)
palette = sns.color_palette("husl", n_colors=n_scenarios_total)
step = 10
spread_indices = [(i * step) % n_scenarios_total for i in range(n_scenarios_total)]
spread_palette = [palette[i] for i in spread_indices]
color_map_full = {sc: spread_palette[i] for i, sc in enumerate(ordered_scenario_ids)}
color_map_pathways = {sc: color_map_full[sc] for sc in ordered_scenario_ids_pathways}

# -------------------------------------------------- ANALYSIS ---------------------------------------------------------
# Define target period
TARGET_PERIOD = (Date(2025, 1, 1), Date(2054, 12, 31))

# ========================================= TOTAL DALYS AVERTED BY SCENARIO ===========================================

# get DALY df
dalys_by_age_date_and_cause = extract_results(
                results_folder,
                module="tlo.methods.healthburden",
                key="dalys_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
                custom_generate_series=get_dalys_by_period_sex_agegrp_label,
                do_scaling=True
            )
dalys_by_age_date_and_cause.index = dalys_by_age_date_and_cause.index.set_names('age_group', level=1)

# Get the total dalys by year and then the total dalys in the target period
dalys_by_age_date = dalys_by_age_date_and_cause.groupby(by=["year", "age_group"]).sum()
dalys_non_standardized_year = dalys_by_age_date.groupby(by='year').sum()
total_dalys_non_standardized = dalys_non_standardized_year.loc[
                               TARGET_PERIOD[0].year:TARGET_PERIOD[-1].year].sum().to_frame().T

# Get summary statistics for total DALYs (not age adjusted)
total_dalys_non_standardized_summ = compute_summary_statistics(total_dalys_non_standardized, use_standard_error=True)

# calculate the total dalys averted when compared to the status quo
diff_total_dalys_non_standardized = get_diff(total_dalys_non_standardized, False)

# calculate the total dalys averted when compared to the status quo by year
diff_dalys_by_year = get_diff(dalys_non_standardized_year, False)

# Save the total DALYs by scenario and the difference from SQ
total_dalys_to_save = reform_df_to_save(total_dalys_non_standardized_summ, 0)
total_dalys_to_save.to_csv(f'{g_path}/total_dalys_int_period_summ.csv')
diff_total_dalys_to_save = reform_df_to_save(diff_total_dalys_non_standardized, 0)
diff_total_dalys_to_save.to_csv(f'{g_path}/total_dalys_int_period_summ_diff.csv')

# Now repeat the above process but we age standardize the dalys
total_dalys_age_standardized = age_standardize_dalys(dalys_by_age_date)
dalys_age_standardized_year = total_dalys_age_standardized.groupby(by='year').sum()
total_dalys_age_standardized_yr_int = dalys_age_standardized_year.loc[
                                      TARGET_PERIOD[0].year:TARGET_PERIOD[-1].year].sum().to_frame().T
total_age_standardized_dalys_summ = compute_summary_statistics(total_dalys_age_standardized_yr_int,
                                                               use_standard_error=True)

diff_age_standardized_dalys_sum = get_diff(total_dalys_age_standardized_yr_int, False)
diff_dalys_age_standardized_by_year = get_diff(dalys_age_standardized_year, False)

total_age_standardized_dalys_to_save = reform_df_to_save(total_age_standardized_dalys_summ, 0)
total_age_standardized_dalys_to_save.to_csv(f'{g_path}/total_stnd_dalys_int_period_summ.csv')
diff_age_standardized_dalys_to_save = reform_df_to_save(diff_age_standardized_dalys_sum, 0)
diff_age_standardized_dalys_to_save.to_csv(f'{g_path}/total_stnd_dalys_int_period_summ_diff.csv')

p_diff_age_standardized_dalys_sum = get_diff(total_dalys_age_standardized_yr_int, True)
p_diff_age_standardized_dalys_sum_to_save = reform_df_to_save(p_diff_age_standardized_dalys_sum, 2)
p_diff_age_standardized_dalys_sum_to_save.to_csv(f'{g_path}/total_stnd_dalys_int_period_summ_percent_diff.csv')


def figure_total_dalys_averted_by_scenario_with_uncertainty(data, age_standardized):
    """Outputs an annotated bar graph showing the mean total DALYs averted by scenario along with 95% confidence
    intervals"""

    """Outputs an annotated bar graph showing the mean total DALYs averted by scenario along with 95% confidence intervals"""
    color_map = color_map_full

    # === Extract data in the new order
    labels = [full_lab[scen_draws[sc]] for sc in ordered_scenario_ids]
    mean = [float(data[sc, 'central'].values) for sc in ordered_scenario_ids]
    lower_errors = [float(data[sc, 'lower'].values) for sc in ordered_scenario_ids]
    upper_errors = [float(data[sc, 'upper'].values) for sc in ordered_scenario_ids]

    # === Compute error margins
    yerr_lower = [med - low for med, low in zip(mean, lower_errors)]
    yerr_upper = [up - med for med, up in zip(mean, upper_errors)]

    # === Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # === Draw dotted lines between groups
    index_map = {sc: i for i, sc in enumerate(ordered_scenario_ids)}
    for g in groupings[:-1]:  # skip last group
        last_sid = g[-1]
        ix = index_map[last_sid]
        ax.axvline(ix + 0.5, color='black', linestyle=':', linewidth=0.5)

    # === Plot bars
    bar_colors = [color_map[sc] for sc in ordered_scenario_ids]
    bars = ax.bar(labels, mean, yerr=[yerr_lower, yerr_upper],
                  capsize=5, alpha=0.7, ecolor='black', color=bar_colors)

    # === Add horizontal line at zero
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # === Annotate values on bars
    offset = max(upper_errors) * 0.05  # consistent vertical offset
    for bar, value in zip(bars, mean):
        height = bar.get_height()
        scaled_val = value / 1e6  # display in millions
        if height >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + offset,
                    f'{scaled_val:.1f}M',
                    ha='center', va='bottom', fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height - offset,
                    f'{scaled_val:.1f}M',
                    ha='center', va='top', fontsize=6)

    # === Axis and formatting
    ax.set_ylabel('DALYs Averted', fontsize=9)
    plt.xticks(fontsize=8, rotation=90)
    plt.tight_layout()

    # === Save output
    plt.savefig(f'{g_path}/total_dalys_averted_by_scenario.png' if not age_standardized else
                f'{g_path}/total_age_standardized_dalys_averted_by_scenario.png',
                bbox_inches='tight')
    plt.show()


figure_total_dalys_averted_by_scenario_with_uncertainty(data=diff_age_standardized_dalys_sum,
                                                        age_standardized=True)

def figure_total_dalys_averted_by_scenario_by_time_period(data, age_standardized):
    """Outputs plot showing total DALYs averted during three time periods across all scenarios"""

    # Define time periods
    p1 = [2025, 2034]
    p2 = [2035, 2044]
    p3 = [2045, 2055]

    def get_data_for_time_period(tp):
        data_tp = data.loc[tp[0]:tp[-1]].sum().to_frame().T
        mean_tp = [float(data_tp[v, 'central'].values) for v in scen_draws.keys()]
        lower_tp = [float(data_tp[v, 'lower'].values) for v in scen_draws.keys()]
        upper_tp = [float(data_tp[v, 'upper'].values) for v in scen_draws.keys()]

        return mean_tp, lower_tp, upper_tp

    # Extract data
    data_p1 = get_data_for_time_period(p1)
    data_p2 = get_data_for_time_period(p2)
    data_p3 = get_data_for_time_period(p3)

    # Time period labels and scenario order
    period_labels = ['2025–34', '2035–244', '2045–55']
    scenarios = list(scen_draws.keys())

    # Extract data from your function
    mean_p1, lower_p1, upper_p1 = data_p1
    mean_p2, lower_p2, upper_p2 = data_p2
    mean_p3, lower_p3, upper_p3 = data_p3

    # Prepare data dicts
    data_mean = {
        scenario: [mean_p1[i], mean_p2[i], mean_p3[i]]
        for i, scenario in enumerate(scenarios)
    }
    data_err_lower = {
        scenario: [mean_p1[i] - lower_p1[i], mean_p2[i] - lower_p2[i], mean_p3[i] - lower_p3[i]]
        for i, scenario in enumerate(scenarios)
    }
    data_err_upper = {
        scenario: [upper_p1[i] - mean_p1[i], upper_p2[i] - mean_p2[i], upper_p3[i] - mean_p3[i]]
        for i, scenario in enumerate(scenarios)
    }

    # color map
    color_map = color_map_full

    # === Define scenario pairs to plot together ===
    scenario_groups = groupings # adjust as needed

    groups_per_figure = 8
    group_batches = [scenario_groups[:groups_per_figure], scenario_groups[groups_per_figure:]]

    for fig_idx, group_set in enumerate(group_batches):
        n_cols = 2
        n_rows = -(-len(group_set) // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), sharey=True)
        axes = axes.flatten()

        for i, group in enumerate(group_set):
            ax = axes[i]

            for scenario in group:
                means = data_mean[scenario]
                yerr = [data_err_lower[scenario], data_err_upper[scenario]]

                ax.errorbar(
                    period_labels,
                    means,
                    yerr=yerr,
                    fmt='-o',
                    color=color_map[scenario],
                    capsize=3,
                    linewidth=1.5,
                    markersize=5,
                    label=full_lab[scen_draws[scenario]]
                )

            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            title_labels = " vs ".join([full_lab[scen_draws[sc]] for sc in group])
            ax.set_title(title_labels, fontsize=9)

            row_idx = i // n_cols
            is_bottom_row = row_idx == n_rows - 1
            if is_bottom_row:
                ax.set_xticklabels(period_labels)
            else:
                ax.set_xticklabels([])

            ax.legend(fontsize=8)

        # Remove unused axes
        for j in range(len(group_set), len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout(rect=[0.05, 0, 1, 0.95])
        fig.text(0.04, 0.5, 'DALYs averted (millions)', va='center', rotation='vertical', fontsize=12)

        # Save with figure index
        filename = f"{g_path}/dalys_averted_scenario_groups_fig{fig_idx + 1}.png"
        if age_standardized:
            filename = f"{g_path}/age_standardized_dalys_averted_scenario_groups_fig{fig_idx + 1}.png"

        plt.savefig(filename, bbox_inches='tight')
        plt.show()


# Output plots for non age-standardized and age-standardized DALYs
# figure_total_dalys_averted_by_scenario_by_time_period(data=diff_dalys_by_year, age_standardized=False)
figure_total_dalys_averted_by_scenario_by_time_period(data=diff_dalys_age_standardized_by_year, age_standardized=True)

# ================================== CAUSE-SPECIFIC DALYS AVERTED BY SCENARIO ========================================

# Sum the DALYs by cause across the target period
dalys_by_year_cause = dalys_by_age_date_and_cause.groupby(by=["year", "label"]).sum()
dalys_by_year_cause_int_period = dalys_by_year_cause.loc[TARGET_PERIOD[0].year:TARGET_PERIOD[-1].year]
# get the total DALYs by cause 2025-2055
total_dalys_by_year_cause =  dalys_by_year_cause_int_period.groupby('label').sum()

# Get the summary statistics for cause-specific DALYs and the difference from SQ
total_dalys_by_year_cause_summ = compute_summary_statistics(total_dalys_by_year_cause, use_standard_error=True)
diff_dalys_by_cause = get_diff(total_dalys_by_year_cause, False)

# Save data
total_dalys_by_year_cause_summ.to_csv(f'{g_path}/cause_specific_dalys_int_period_summ.csv')
diff_dalys_by_cause.to_csv(f'{g_path}/cause_specific_dalys_int_period_summ_diff.csv')

# Now get the difference in dalys by cause for each year of the target period
diff_dalys_by_cause_year = get_diff(dalys_by_year_cause_int_period, False)
diff_dalys_by_cause_year.to_csv(f'{g_path}/cause_specific_dalys_by_year_int_period_summ_diff.csv')

# Repeat this process with age-standardized DALYs
dalys_by_cause_age_year_standardize = age_standardize_dalys(dalys_by_age_date_and_cause)
d_by_cause_int_period = dalys_by_cause_age_year_standardize.loc[TARGET_PERIOD[0].year:TARGET_PERIOD[-1].year]
dalys_by_cause_age_standardize = d_by_cause_int_period.groupby(level='label').sum()
total_dalys_by_cause_age_standardize_summ = compute_summary_statistics(dalys_by_cause_age_standardize,
                                                                       use_standard_error=True)
diff_dalys_by_cause_age_standardize = get_diff(dalys_by_cause_age_standardize, False)

total_dalys_by_year_cause_summ.to_csv(f'{g_path}/cause_specific_stnd_dalys_int_period_summ.csv')
diff_dalys_by_cause.to_csv(f'{g_path}/cause_specific_stnd_dalys_int_period_summ_diff.csv')

diff_dalys_by_cause_year_age_standardize = get_diff(d_by_cause_int_period, False)
diff_dalys_by_cause_year_age_standardize.to_csv(f'{g_path}/cause_specific_stnd_dalys_by_year_int_period_summ_diff.csv')

def figure_heatmap_cause_specific_dalys_averted(data):
    """Outputs a heatmap showing the difference in DALYs """
    data = data.drop(columns=0, level='draw')

      # Extract central, lower, upper from MultiIndex or flat columns
    central_df = data.loc[:, data.columns.get_level_values(1) == 'central']
    lower_df = data.loc[:, data.columns.get_level_values(1) == 'lower']
    upper_df = data.loc[:, data.columns.get_level_values(1) == 'upper']

    # Clean up column names
    scenario_labels_in_order = [full_lab[scen_draws[sc]] for sc in ordered_scenario_ids]

    # Rename columns
    central_df.columns = [full_lab[scen_draws[col[0]]] for col in central_df.columns]
    lower_df.columns = [full_lab[scen_draws[col[0]]] for col in lower_df.columns]
    upper_df.columns = [full_lab[scen_draws[col[0]]] for col in upper_df.columns]

    # Reindex to enforce consistent order
    central_df = central_df[scenario_labels_in_order]
    lower_df = lower_df[scenario_labels_in_order]
    upper_df = upper_df[scenario_labels_in_order]

    # STEP 3: Masks
    uncertainty_includes_zero = (lower_df <= 0) & (upper_df >= 0)
    significant = ~uncertainty_includes_zero
    positive = central_df > 0
    significant_positive = significant & positive

    # === Format function
    def format_daly(value):
        if pd.isna(value):
            return ""
        abs_val = abs(value)
        if abs_val < 1_000:
            return f"{int(value)}"
        elif abs_val < 1_000_000:
            return f"{value / 1_000:.0f}K"
        else:
            return f"{value / 1_000_000:.1f}M"

    # === Create annotation matrix
    annot = central_df.copy().astype(str)
    for i in range(central_df.shape[0]):
        for j in range(central_df.shape[1]):
            if significant_positive.iloc[i, j]:
                val = central_df.iloc[i, j]
                annot.iloc[i, j] = format_daly(val)
            else:
                annot.iloc[i, j] = ""

    # === Plotting
    vmin = central_df.min().min()
    vmax = central_df.max().max()
    abs_max = max(abs(vmin), abs(vmax))

    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    plt.figure(figsize=(16, 10))
    sns.heatmap(
        central_df,
        cmap="RdBu",  # inverted so blue = higher values
        norm=norm,
        annot=annot,
        fmt="",
        linewidths=0.5,
        cbar_kws={'label': 'DALY Difference from Status Quo'}
    )

    plt.ylabel("Condition")
    plt.xlabel("Scenario")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{g_path}/cause_specific_dalys_heatmap.png', bbox_inches='tight')
    plt.show()

figure_heatmap_cause_specific_dalys_averted(diff_dalys_by_cause_age_standardize)


# =============================================== LIFE EXPECTANCY ===================================================
# Get life expectancy by scenario in 2054 (last year of sim)
le_estimates = get_life_expectancy_estimates(results_folder=results_folder,
                                             target_period=
                                             (datetime.date(2054, 1, 1),
                                              datetime.date(2054, 12, 31)),
                                             summary=False)
le_estimate_avg = le_estimates.mean(axis=0).to_frame().T

# calculate difference
le_diff = le_estimate_avg.copy()
for col in le_estimate_avg.columns:
    if col[0] != 0:
        base_col = (0, col[1])

        le_diff[col] = le_estimate_avg[col] - le_estimate_avg[base_col]
    else:
        le_diff[col] = 0  # or np.nan if you prefer

le_diff = le_diff.drop(columns=0)
le_estimate_avg_diff_summ = compute_summary_statistics(le_diff, use_standard_error=True)

#  Plot LE
# === Extract central, lower, upper for 2055 ===
def figure_life_expectancy_difference(data):

    # === Extract central, lower, upper values ===
    central = data.xs('central', level=1, axis=1).iloc[0]
    lower = data.xs('lower', level=1, axis=1).iloc[0]
    upper = data.xs('upper', level=1, axis=1).iloc[0]

    # === Reorder using ordered_scenario_ids ===
    central = central.loc[ordered_scenario_ids]
    lower = lower.loc[ordered_scenario_ids]
    upper = upper.loc[ordered_scenario_ids]

    # === Significance mask
    significant_mask = (lower > 0) | (upper < 0)

    # === Replace non-significant values with 0s (for plotting)
    central_plot = central.copy()
    yerr_lower = (central - lower).copy()
    yerr_upper = (upper - central).copy()

    central_plot[~significant_mask] = 0
    yerr_lower[~significant_mask] = 0
    yerr_upper[~significant_mask] = 0
    yerr = [yerr_lower.values, yerr_upper.values]

    # === Color setup
    scenario_ids = central.index.tolist()
    color_map = color_map_full
    colors = [color_map[s] for s in scenario_ids]

    # === Plotting
    x = np.arange(len(scenario_ids))
    fig, ax = plt.subplots(figsize=(12, 5))

    bars = ax.bar(
        x,
        central_plot.values,
        yerr=yerr,
        capsize=5,
        color=colors,
        edgecolor='black'
    )

    # === Annotate only significant bars
    offset = 0.02 * max(abs(upper.max()), 1e-6)
    for xpos, is_sig, mean_val, hi, lo in zip(x, significant_mask, central.values, upper.values, lower.values):
        if is_sig:
            y = hi + offset if mean_val >= 0 else lo - offset
            ax.text(
                xpos,
                y,
                f"{mean_val:.2f}",
                ha='center',
                va='bottom' if mean_val >= 0 else 'top',
                fontsize=8,
                fontweight='bold'
            )

    # Map all scenario positions
    index_map = {s: i for i, s in enumerate(scenario_ids)}

    # Draw line between groups only if both scenarios are present
    for g in groupings[:-1]:
        present = [s for s in g if s in index_map]
        if present:
            ix = index_map[present[-1]]
            ax.axvline(ix + 0.5, color='black', linestyle=':', linewidth=0.5)

    # === Axis & aesthetics
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([full_lab[scen_draws[s]] for s in scenario_ids], rotation=90, fontsize=7)
    ax.set_ylabel('Difference in Life Expectancy (2054, years)')

    plt.tight_layout()
    plt.savefig(f'{g_path}/life_expectancy_diff_2054.png', bbox_inches='tight')
    plt.show()

figure_life_expectancy_difference(le_estimate_avg_diff_summ)

# ===================================== DIFFERENCE IN HCW TIME USE BY SCENARIO ========================================
# Output HCW time use by treatment_id
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

# Calculate average change in population size per year in the SQ scenario
pop = get_full_pop()
pop = pop.groupby(by='year').sum()
relative_increase_df = pop.pct_change()
avg_rel_increase = relative_increase_df.loc[2025:2054].mean(axis=0).to_frame().T
avg_rel_increase_summ = compute_summary_statistics(avg_rel_increase, use_standard_error=True)

# Read in HCW capabilities data and sum across facility levels etc.
daily_cap = pd.read_csv('./resources/healthsystem/human_resources/actual/ResourceFile_Daily_Capabilities.csv')
daily_mins = daily_cap.set_index('Officer_Category')[['Total_Mins_Per_Day']]
daily_mins = daily_mins.drop('Dental')
daily_mins = daily_mins.drop('Nutrition')
daily_mins = daily_mins.groupby(daily_mins.index).sum()

# Next we calculate the average HCW capabilities assuming capabilities increase yearly in line with population growth
yearly_mins = daily_mins * 365.25
yearly_mins.rename(columns={'Total_Mins_Per_Day': 'mins_per_year'}, inplace=True)

value = 1 + avg_rel_increase_summ[(0, 'central')].values
n_times = 30       # number of times to multiply (represents number of years in the intervention period)
steps = [(yearly_mins * (value ** i)) for i in range(n_times + 1)]
avg_annual_hcw_capabilities = sum(steps) / len(steps)
avg_annual_hcw_capabilities_sum = avg_annual_hcw_capabilities.sum(axis=0).to_frame().T

# --------------------------------------------- PLOT TOTAL HCW TIME RATIO ------------------------------------------
# Next we calculate the total HCW time use
hcw_time_by_treatment_id_df = pd.DataFrame.from_dict(hcw_time_by_treatment_id)
hcw_time_by_treatment_id_df = hcw_time_by_treatment_id_df.fillna(0)
hcw_time_by_treatment_id_df.index.names = ['first', 'second']

# Now we calculate the demand of HCW time by cadre relative to yearly HCW capabilities (adjusted for predicted
# population growth)

def get_hcw_time_use_ratios(df):
    """Get HCW time used/time available - compare that to the status QUO"""
    def get_hcw_time_fold_diff(df_for_diff):
        diff = df_for_diff.copy()
        for col in df_for_diff.columns:
            if col[0] != 0:
                base_col = (0, col[1])
                diff[col] = df_for_diff[col] / df_for_diff[base_col]
            else:
                diff[col] = 1
        return diff

    # Calculate total and annual HCW time during the intervention period
    total_hcw_time = df.sum(axis=0).to_frame().T
    yearly_total_hcw_time = total_hcw_time / 30

    # Divide total annual HCW time by total annual HCW time capabilities
    total_hcw_time_ratio = yearly_total_hcw_time.div(avg_annual_hcw_capabilities_sum.iloc[:, 0], axis=0)
    total_hcw_time_ratio.columns.names = ['draw', 'run']

    # Find the fold change from the Status Quo scenario
    fold_change = get_hcw_time_fold_diff(total_hcw_time_ratio)
    total_hcw_time_ratio_fc = compute_summary_statistics(fold_change, use_standard_error=True)

    hcw_time_by_cadre = df.groupby(level='first').sum()
    annual_hcw_time_by_cadre = hcw_time_by_cadre / 30

    # Now we calculate the ratio of time use to time available (by cadre) and summarise it
    hcw_time_ratio_by_cadre = annual_hcw_time_by_cadre.div(avg_annual_hcw_capabilities.iloc[:, 0], axis=0)
    hcw_time_ratio_by_cadre.columns.names = ['draw', 'run']

    # Now we get the diff from the SQ
    fold_change_cadre = get_hcw_time_fold_diff(hcw_time_ratio_by_cadre)
    hcw_time_ratio_by_cadre_fc = compute_summary_statistics(fold_change_cadre, use_standard_error=True)

    # We also return the ratios for all scenarios across cadres (not the difference from the SQ)
    hcw_time_ratio_by_cadre_summ = compute_summary_statistics(hcw_time_ratio_by_cadre, use_standard_error=True)

    return [total_hcw_time_ratio_fc, hcw_time_ratio_by_cadre_fc, hcw_time_ratio_by_cadre_summ]

hcw_ratios_unadjusted = get_hcw_time_use_ratios(hcw_time_by_treatment_id_df)


def figure_annual_hcw_time_use_over_annual_capabilities(data, tol=1e-9):

    # --- Extract in requested order
    scenario_ids = ordered_scenario_ids
    labels_all = [full_lab[scen_draws[sc]] for sc in scenario_ids]
    mean_all   = [float(data.loc[:, (sc, 'central')].iloc[0]) for sc in scenario_ids]
    lo_all     = [float(data.loc[:, (sc, 'lower')].iloc[0])   for sc in scenario_ids]
    up_all     = [float(data.loc[:, (sc, 'upper')].iloc[0])   for sc in scenario_ids]

    # --- Normalise bounds
    ci_low_all = [min(lo, up) for lo, up in zip(lo_all, up_all)]
    ci_up_all  = [max(lo, up) for lo, up in zip(lo_all, up_all)]

    # --- Check which bars have CI crossing 1
    crosses_one = [(l <= 1 + tol) and (u >= 1 - tol) for l, u in zip(ci_low_all, ci_up_all)]

    # --- Fixed x positions for ALL scenarios
    x_all = np.arange(len(scenario_ids))

    # --- Error distances for all bars
    yerr_lower = [max(0.0, m - l) for m, l in zip(mean_all, ci_low_all)]
    yerr_upper = [max(0.0, u - m) for m, u in zip(mean_all, ci_up_all)]

    # --- Color mapping (consistent across full set)
    scenarios_except_0 = [s for s in scenario_ids if s != 0]
    color_map_full = {s: spread_palette[i] for i, s in enumerate(scenarios_except_0)}
    if 0 in scenario_ids:
        color_map_full[0] = palette[-1]
    colors_all = [color_map_full[s] for s in scenario_ids]

    # --- Plot
    fig, ax = plt.subplots()

    bars = ax.bar(x_all, mean_all, yerr=[yerr_lower, yerr_upper],
                  capsize=5, alpha=0.7, ecolor='black', color=colors_all)

    # --- Annotations
    max_up_err = max(yerr_upper) if any(yerr_upper) else (0.05 * max(abs(m) for m in mean_all))
    offset = max_up_err * 0.5 if max_up_err > 0 else 0.05
    for bar, value, cross in zip(bars, mean_all, crosses_one):
        h = bar.get_height()
        # Format: 3 dp for <1, 2 dp for >=1
        txt = f"{value:.3f}" if value < 1 else f"{value:.2f}"
        if cross:
            txt += "*"   # add asterisk if CI includes 1
        if h >= 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + offset, txt,
                    ha='center', va='bottom', fontsize=7)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, h - offset, txt,
                    ha='center', va='top', fontsize=7)

    # --- X ticks
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels_all, fontsize=7, rotation=90)

    # --- Dotted lines between defined groups, using full set positions
    index_map_full = {s: i for i, s in enumerate(scenario_ids)}
    for g in groupings[:-1]:
        present = [s for s in g if s in index_map_full]
        if present:
            ix = index_map_full[present[-1]]
            ax.axvline(ix + 0.5, color='black', linestyle=':', linewidth=0.5)

    # --- Reference line and formatting
    ax.axhline(y=1, color='grey', linestyle='--', linewidth=1, label='Status Quo = 1')
    ax.set_ylabel('Relative change in HCW demand – HCW capabilities compared to SQ', fontsize=9)

    # Legend (baseline line only)
    ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=1, frameon=False)

    plt.tight_layout()
    plt.savefig(f'{g_path}/hcw_time_ratios.png', bbox_inches='tight')
    plt.show()

figure_annual_hcw_time_use_over_annual_capabilities(hcw_ratios_unadjusted[0])

def figure_all_cadre_bar_charts_color_coded(df, tol=1e-9):

    central_df = df.xs('central', axis=1, level=1)
    lower_df   = df.xs('lower', axis=1, level=1)
    upper_df   = df.xs('upper', axis=1, level=1)

    scenarios = ordered_scenario_ids
    color_map = color_map_full
    index_map = {s: i for i, s in enumerate(scenarios)}

    cadres = central_df.index
    fig, axes = plt.subplots(len(cadres), 1, figsize=(14, 3.5 * len(cadres)), sharex=True)
    if len(cadres) == 1:
        axes = [axes]

    baseline_handle = None

    for i, cadre in enumerate(cadres):
        central = central_df.loc[cadre]
        lower   = lower_df.loc[cadre]
        upper   = upper_df.loc[cadre]

        # CI filter
        ci_low = np.minimum(lower, upper)
        ci_up  = np.maximum(lower, upper)
        keep_mask = (ci_up < 1 - tol) | (ci_low > 1 + tol)

        # Keep all ticks, NaN for excluded bars
        plot_vals = pd.Series(np.nan, index=scenarios)
        plot_vals[keep_mask] = central[keep_mask]

        ax = axes[i]

        # Bars only for kept scenarios
        for s in scenarios:
            if not np.isnan(plot_vals[s]):
                ax.bar(index_map[s], plot_vals[s], color=color_map[s])

        # Annotate kept bars
        if keep_mask.any():
            offset = 0.02 * max(abs(np.nanmax(plot_vals)), abs(np.nanmin(plot_vals)), 1e-6)
            for s in scenarios:
                val = plot_vals[s]
                if not np.isnan(val):
                    va = 'bottom' if val >= 0 else 'top'
                    txt = f"{val:.3f}" if abs(val) < 1 else f"{val:.2f}"
                    ax.text(
                        index_map[s],
                        val + offset if val >= 0 else val - offset,
                        txt,
                        ha='center',
                        va=va,
                        fontsize=9,
                        fontweight='bold'
                    )

        # Group dividers
        for g in groupings[:-1]:
            present = [s for s in g if s in index_map]
            if present:
                ix = index_map[present[-1]]
                ax.axvline(ix + 0.5, color='black', linestyle=':', linewidth=0.5)

        # Baseline line
        if i == 0:
            baseline_handle = ax.axhline(
                y=1, color='grey', linestyle='--', linewidth=1, label='Status Quo = 1'
            )
        else:
            ax.axhline(y=1, color='grey', linestyle='--', linewidth=1)

        ax.set_title(f"{cadre}")
        ax.set_ylabel("")

        # X ticks only on bottom subplot
        if i < len(cadres) - 1:
            ax.set_xticks([])
        else:
            xtick_pos = [index_map[s] for s in scenarios]
            xtick_labels = [full_lab[scen_draws[s]] for s in scenarios]
            ax.set_xticks(xtick_pos)
            ax.set_xticklabels(xtick_labels, rotation=90, fontsize=8)

    # Legend anchored to the top subplot
    if baseline_handle is not None:
        axes[0].legend(
            handles=[baseline_handle],
            loc='lower center',
            bbox_to_anchor=(0.5, 1.05),  # just above the title
            frameon=False,
            fontsize=9
        )

    plt.tight_layout(rect=[0.05, 0.02, 1, 0.97])
    fig.text(0.01, 0.5, 'Relative change in HCW demand – HCW capabilities compared to SQ',
             va='center', rotation='vertical', fontsize=12)
    plt.savefig(f'{g_path}/hcw_time_ratios_by_cadre.png', bbox_inches='tight')
    plt.show()

# === Call the function ===
desired_order = [
    'Clinical',
    'Nursing_and_Midwifery',
    'Pharmacy',
    'Mental',
    'Laboratory',
    'Radiography'
]

combined_df_ordered = hcw_ratios_unadjusted[1].loc[desired_order]
figure_all_cadre_bar_charts_color_coded(combined_df_ordered)

# ----------------------------------------- HCW TIME ADJUSTED FOR EFFICIENCY ------------------------------------------
# Now we adjust total HCW time in the intervention period for scenarios representing the integration pathways
# We adjust HCW time for a subset of HSIs represeting 'integration interventions'

# Get DF with HCW time by treatment IDs for relevant scenarios
hcw_time_by_treatment_id_pathways_df = hcw_time_by_treatment_id_df[[0, 20, 21, 22, 23, 24, 25, 26, 27]]
hcw_time_by_treatment_id_pathways_df = hcw_time_by_treatment_id_pathways_df.fillna(0)
hcw_time_by_treatment_id_pathways_df.index.names = ['first', 'second']

def multiply_subset(df, col_level_0_vals, care_types_to_update, multiplier):
    """
    Multiply values in selected columns and rows of a multi-index DataFrame.

    Parameters:
        df: pandas DataFrame with multi-level columns and multi-level index.
        col_level_0_vals: collection of ints. First level column values to match (e.g. [0, 1]).
        care_types_to_update: collection. Values from the second index level to update.
        multiplier: float. Value to multiply the selected cells by.
    """
    # Step 1: Select columns where first level is in the provided list
    columns_to_update = [col for col in df.columns if col[0] in col_level_0_vals]

    # Step 2: Get mask for rows to update
    rows_to_update = df.index.get_level_values(1).isin(care_types_to_update)

    # Step 3: Apply multiplication
    df.loc[rows_to_update, columns_to_update] *= multiplier

# Creat copys of HCW time DFs to adjust
hcw_time_by_treatment_id_adj_25 = hcw_time_by_treatment_id_pathways_df.copy()
hcw_time_by_treatment_id_adj_50 = hcw_time_by_treatment_id_pathways_df.copy()
hcw_time_by_treatment_id_adj_75 = hcw_time_by_treatment_id_pathways_df.copy()

# For the relevant scenarios and treatment IDs, we adjust total time (25% reduction, 50% reduction, 75% reduction)
scaling_groups = [
    ([20, 21], [
        'CardioMetabolicDisorders_Investigation_diabetes',
        'CardioMetabolicDisorders_Investigation_hypertension',
        'CardioMetabolicDisorders_Investigation_hypertension_and_diabetes',
        'CardioMetabolicDisorders_Prevention_WeightLoss_diabetes',
        'CardioMetabolicDisorders_Prevention_WeightLoss_hypertension',
        'CardioMetabolicDisorders_Treatment_hypertension',
        'CardioMetabolicDisorders_Treatment_diabetes',
        'Hiv_Test',
        'Hiv_Treatment',
        'Tb_Test_Screening',
        'Tb_Test_Clinical',
        'Tb_Test_Culture',
        'Tb_Test_Xray',
        'Tb_Treatment',
        'Tb_Test_FollowUp',
        'Depression_TalkingTherapy',
        'Depression_Treatment',
        'Epilepsy_Treatment_Start',
        'Epilepsy_Treatment_Followup'
    ]),
    ([22, 23], [
        'CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
        'CardioMetabolicDisorders_Investigation_hypertension',
        'CardioMetabolicDisorders_Investigation_hypertension_and_diabetes',
        'CardioMetabolicDisorders_Investigation_diabetes',
        'Contraception_Routine',
        'Hiv_Test',
        'Tb_Test_Screening',
        'Tb_Test_Clinical',
        'Tb_Test_Culture',
        'Tb_Test_Xray'
    ]),
    ([24, 25], [
        'AntenatalCare_Outpatient',
        'AntenatalCare_FollowUp',
        'PostnatalCare_Neonatal',
        'PostnatalCare_Maternal',
        'Contraception_Routine_Postnatal',
        'Epi_Childhood_Bcg',
        'Epi_Childhood_Opv',
        'Epi_Childhood_DtpHibHep',
        'Epi_Childhood_Rota',
        'Epi_Childhood_Pneumo',
        'Epi_Childhood_MeaslesRubella',
        'Epi_Pregnancy_Td'
    ]),
    ([26, 27], [
        'CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
        'CardioMetabolicDisorders_Investigation_hypertension',
        'CardioMetabolicDisorders_Investigation_hypertension_and_diabetes',
        'CardioMetabolicDisorders_Prevention_WeightLoss_hypertension',
        'CardioMetabolicDisorders_Treatment_hypertension',
        'CardioMetabolicDisorders_Investigation_diabetes',
        'CardioMetabolicDisorders_Prevention_WeightLoss_diabetes',
        'CardioMetabolicDisorders_Treatment_diabetes',
        'Contraception_Routine',
        'Undernutrition_Feeding',
        'Hiv_Test',
        'Hiv_Treatment',
        'Tb_Test_Screening',
        'Tb_Test_Clinical',
        'Tb_Test_Culture',
        'Tb_Test_Xray',
        'Tb_Treatment',
        'Tb_Test_FollowUp',
        'AntenatalCare_Outpatient',
        'AntenatalCare_FollowUp',
        'PostnatalCare_Neonatal',
        'PostnatalCare_Maternal',
        'Contraception_Routine_Postnatal',
        'Epi_Childhood_Bcg',
        'Epi_Childhood_Opv',
        'Epi_Childhood_DtpHibHep',
        'Epi_Childhood_Rota',
        'Epi_Childhood_Pneumo',
        'Epi_Childhood_MeaslesRubella',
        'Epi_Pregnancy_Td',
        'Depression_TalkingTherapy',
        'Depression_Treatment',
        'Epilepsy_Treatment_Start',
        'Epilepsy_Treatment_Followup'
    ])
]

for cols, care_types in scaling_groups:
    multiply_subset(
        df=hcw_time_by_treatment_id_adj_25,
        col_level_0_vals=cols,
        care_types_to_update=care_types,
        multiplier=0.75
    )

for cols, care_types in scaling_groups:
    multiply_subset(
        df=hcw_time_by_treatment_id_adj_50,
        col_level_0_vals=cols,
        care_types_to_update=care_types,
        multiplier=0.5
    )

for cols, care_types in scaling_groups:
    multiply_subset(
        df=hcw_time_by_treatment_id_adj_75,
        col_level_0_vals=cols,
        care_types_to_update=care_types,
        multiplier=0.25
    )

# Get adjusted estimates
total_hcw_time_ratio_diff_adj_25 =  get_hcw_time_use_ratios(hcw_time_by_treatment_id_adj_25)
total_hcw_time_ratio_diff_adj_50 =  get_hcw_time_use_ratios(hcw_time_by_treatment_id_adj_50)
total_hcw_time_ratio_diff_adj_75 =  get_hcw_time_use_ratios(hcw_time_by_treatment_id_adj_75)


def figure_adjusted_hcw_ratios(un_adj, adj_25, adj_50, adj_75):
    # === Labels for x-axis
    labels = [full_lab[scen_draws[sc]] for sc in ordered_scenario_ids_pathways]

    # === Helper to extract values and error bars
    def get_mean_and_errors(data):
        mean = [float(data.loc[:, (sc, 'central')].iloc[0]) for sc in ordered_scenario_ids_pathways]
        lower = [float(data.loc[:, (sc, 'lower')].iloc[0]) for sc in ordered_scenario_ids_pathways]
        upper = [float(data.loc[:, (sc, 'upper')].iloc[0]) for sc in ordered_scenario_ids_pathways]
        yerr_lower = [m - l for m, l in zip(mean, lower)]
        yerr_upper = [u - m for u, m in zip(upper, mean)]
        return np.array(mean), np.array([yerr_lower, yerr_upper])

    # === Extract data
    datasets = {
        'Unadjusted': get_mean_and_errors(un_adj),
        '25% adjusted': get_mean_and_errors(adj_25),
        '50% adjusted': get_mean_and_errors(adj_50),
        '75% adjusted': get_mean_and_errors(adj_75),
    }

    # === Plot settings
    bar_width = 0.2
    x = np.arange(len(ordered_scenario_ids_pathways))
    fig, ax = plt.subplots(figsize=(12, 6))

    # === Color per scenario
    bar_colors = [color_map_pathways[sc] for sc in ordered_scenario_ids_pathways]

    # === Hatching pattern per dataset
    hatches = ["", "//", "xx", ".."]

    # Store for hatch legend
    hatch_patches = []

    # === Plot bars + annotate
    for i, (dlabel, (means, yerr)) in enumerate(datasets.items()):
        offset = (i - 1.5) * bar_width
        bars = ax.bar(
            x + offset, means, width=bar_width, yerr=yerr,
            label=dlabel, capsize=5, color=bar_colors, alpha=0.9,
            hatch=hatches[i], edgecolor='black', linewidth=0.7
        )
        # Annotate each bar with 2 decimal places
        for bar, val in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f"{val:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        # Create patch for hatch legend
        patch = plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch=hatches[i],
                              edgecolor="black", linewidth=0.7)
        hatch_patches.append((patch, dlabel))

    # === X-axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=8)

    # === Dotted lines between defined groups
    groupings = [
        [22, 23],
        [24, 25],
        [20, 21],
        [26, 27]
    ]
    index_map = {s: i for i, s in enumerate(ordered_scenario_ids_pathways)}
    for g in groupings[:-1]:
        present = [s for s in g if s in index_map]
        if present:
            ix = index_map[present[-1]]
            ax.axvline(ix + 0.5, color='black', linestyle=':', linewidth=0.5)

    ax.set_ylabel('Relative change in HCW demand – HCW capabilities compared to SQ')

    # === Hatch legend only
    hatch_handles, hatch_labels = zip(*hatch_patches)
    hatch_legend = fig.legend(hatch_handles, hatch_labels, title="Hypothetical Assumption on Improved HCW Efficiency",
                              loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4)

    # --- Reference line and formatting
    ax.axhline(y=1, color='grey', linestyle='--', linewidth=1, label='Status Quo = 1')

    # # Legend (baseline line only)
    # ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02),
    #           ncol=1, frameon=False)

    # === Save without cropping legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{g_path}/adjusted_hcw_time_ratios.png',
                bbox_inches='tight',
                bbox_extra_artists=[hatch_legend])
    plt.show()

figure_adjusted_hcw_ratios(hcw_ratios_unadjusted[0],
                           total_hcw_time_ratio_diff_adj_25[0],
                           total_hcw_time_ratio_diff_adj_50[0],
                           total_hcw_time_ratio_diff_adj_75[0])

# ==================================== CONSUMABLE COST BY SCENARIO (AND DIFFS) ========================================
TARGET_PERIOD = (Date(2025, 1, 1), Date(2054, 12, 31))

# list_of_relevant_years_for_costing = list(range(TARGET_PERIOD[0].year, TARGET_PERIOD[-1].year + 1))
#
# input_costs_df = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
#                                                _years=list_of_relevant_years_for_costing,
#                                                cost_only_used_staff=True,
#                                                _discount_rate = 0.03)
# input_costs_df.to_csv(f'{g_path}/input_costs.csv')

# Read in costs (takes a long time to generate)
input_costs = pd.read_csv(f'{g_path}/input_costs.csv')
input_costs = input_costs.set_index('Unnamed: 0')

# --------------------------- Adjust HCW costs based on average difference in HCW use ---------------------------------
# Copy cost data to allow for adjustments based on HCW time ratios
adj_25_input_costs = input_costs.copy()
adj_50_input_costs = input_costs.copy()
adj_75_input_costs = input_costs.copy()

def return_cost_adjusted_for_hcw_growth(cost_data, hcw_ratios):
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

    cost_data['cost'] = cost_data.apply(lambda row: row['cost'] * get_multiplier(row), axis=1)
    total_input_cost = cost_data.groupby(['draw', 'run'])['cost'].sum()
    total_input_cost_annual = total_input_cost / 30

    return total_input_cost, total_input_cost_annual

# Adjust cost data based on HCW time ratios (we multiply cadre salary cost by demand ratios)
# TODO: SHOULD WE BE ADJUSTING SQ for population growth...
costs_und_adj_hcw_ratio = return_cost_adjusted_for_hcw_growth(input_costs, hcw_ratios_unadjusted[1])
costs_adj_25_hcw_ratio = return_cost_adjusted_for_hcw_growth(adj_25_input_costs, total_hcw_time_ratio_diff_adj_25[1])
costs_adj_50_hcw_ratio = return_cost_adjusted_for_hcw_growth(adj_50_input_costs, total_hcw_time_ratio_diff_adj_50[1])
costs_adj_75_hcw_ratio = return_cost_adjusted_for_hcw_growth(adj_75_input_costs, total_hcw_time_ratio_diff_adj_75[1])

def find_cost_diff_from_sq_and_sum(data):

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
            data,
            comparison=0)  # sets the comparator to 0 which is the Actual scenario
    ).T.iloc[0].unstack()).T

    incremental_scenario_cost_summarized = summarize_cost_data(incremental_scenario_cost_annual)

    return incremental_scenario_cost_summarized


incremental_scenario_cost_annual_summarized = find_cost_diff_from_sq_and_sum(costs_und_adj_hcw_ratio[1])
incremental_scenario_cost_annual_adj_25  = find_cost_diff_from_sq_and_sum(costs_adj_25_hcw_ratio[1])
incremental_scenario_cost_annual_adj_50 = find_cost_diff_from_sq_and_sum(costs_adj_50_hcw_ratio[1])
incremental_scenario_cost_annual_adj_75 = find_cost_diff_from_sq_and_sum(costs_adj_75_hcw_ratio[1])


def figure_avg_difference_in_cost_from_status_quo_per_year(cost_data):
    name_of_plot = 'Incremental scenario cost relative to baseline during intervention period'

    # === Reorder cost_data
    cost_data = cost_data.loc[ordered_scenario_ids]

    # === Error bars
    yerr = np.array([
        (cost_data['mean'] - cost_data['lower']).values,
        (cost_data['upper'] - cost_data['mean']).values,
    ])

    spacing = 1.55
    xtick_positions = [(i * spacing) for i in range(len(cost_data))]
    xticks = dict(zip(xtick_positions, cost_data.index))
    index_map = {s: x for x, s in zip(xtick_positions, cost_data.index)}

    fig, ax = plt.subplots(figsize=(10, 5))

    color_map = color_map_full
    colors = [color_map[s] for s in cost_data.index]

    # === Bar chart
    ax.bar(
        xtick_positions,
        cost_data['mean'].values,
        yerr=yerr,
        ecolor='black',
        capsize=10,
        label=[str(s) for s in cost_data.index],
        color=colors,
    )

    # === Format for currency annotation
    def format_currency(val):
        if abs(val) >= 1e9:
            return f"${val / 1e9:.1f}B"
        else:
            return f"${val / 1e6:.0f}M"

    # === Annotate bars
    for xpos, mean, lower, upper in zip(
        xtick_positions,
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

    for g in groupings[:-1]:
        present = [s for s in g if s in index_map]
        if present:
            xpos = index_map[present[-1]]
            ax.axvline(xpos + spacing / 2, color='black', linestyle=':', linewidth=0.5)

    # === Axis and labels
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([full_lab[scen_draws[s]] for s in cost_data.index], rotation=90, fontsize=7)

    ax.grid(axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Difference in annual cost (USD) from SQ scenario', fontsize=7)
    ax.set_ylim(bottom=-0.25 * 1e9)

    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.15, right=0.85)

    # === Save + Show
    fig.savefig(Path(g_path) / name_of_plot.replace(' ', '_').replace(',', ''), bbox_inches='tight')
    plt.show()


figure_avg_difference_in_cost_from_status_quo_per_year(incremental_scenario_cost_annual_summarized)

def figure_adjusted_avg_difference_in_cost_from_status_quo_per_year(un_adj, adj_25, adj_50, adj_75):
    # === Labels for x-axis
    labels = [full_lab[scen_draws[sc]] for sc in ordered_scenario_ids_pathways]

    # === Helper to extract means and asymmetric errors in the right order
    def get_mean_and_errors(df):
        sub = df.loc[ordered_scenario_ids_pathways]  # ensure order
        mean = sub['mean'].values.astype(float)
        lower = (sub['mean'] - sub['lower']).values.astype(float)
        upper = (sub['upper'] - sub['mean']).values.astype(float)
        yerr = np.vstack([lower, upper])  # shape (2, N)
        return mean, yerr

    # === Datasets
    datasets = {
        'Unadjusted': get_mean_and_errors(un_adj),
        '25% adjusted': get_mean_and_errors(adj_25),
        '50% adjusted': get_mean_and_errors(adj_50),
        '75% adjusted': get_mean_and_errors(adj_75),
    }

    # === Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ordered_scenario_ids_pathways))
    n_series = len(datasets)
    width = 0.18  # bar width
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * (width + 0.02)

    # === Color per scenario (consistent with previous figure)
    bar_colors = [color_map_pathways[sc] for sc in ordered_scenario_ids_pathways]

    # === Hatching pattern per dataset (consistent with previous figure)
    hatches = ["", "//", "xx", ".."]
    hatch_patches = []  # for legend

    # === Format for currency annotation (user spec)
    def format_currency(val):
        if abs(val) >= 1e9:
            return f"${val / 1e9:.1f}B"
        else:
            return f"${val / 1e6:.0f}M"

    # === Bars + annotations
    for i, (name, (means, yerr)) in enumerate(datasets.items()):
        xpos = x + offsets[i]
        bars = ax.bar(
            xpos,
            means,
            width=width,
            yerr=yerr,
            capsize=6,
            ecolor='black',
            color=bar_colors,          # scenario-consistent colors
            alpha=0.9,
            hatch=hatches[i],          # dataset hatch
            edgecolor='black',         # outlines
            linewidth=0.7
        )

        # Currency annotations per bar
        # Use the asymmetric errors to place the label just beyond the error bar
        for j, (bar, mean_val) in enumerate(zip(bars, means)):
            upper = mean_val + yerr[1, j]
            lower = mean_val - yerr[0, j]
            text = format_currency(mean_val)

            if mean_val >= 0:
                annotation_y = upper + 0.02 * 1e9  # small absolute offset (20M)
                valign = 'bottom'
            else:
                annotation_y = lower - 0.02 * 1e9
                valign = 'top'

            ax.text(
                xpos[j],
                annotation_y,
                text,
                ha='center',
                va=valign,
                fontsize='x-small',
                rotation='horizontal'
            )

        # Build entries for hatch legend
        patch = plt.Rectangle((0, 0), 1, 1, facecolor="white",
                              hatch=hatches[i], edgecolor="black", linewidth=0.7)
        hatch_patches.append((patch, name))

    # === Axes cosmetics
    ax.axhline(0, linewidth=1, linestyle='--', color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=8)
    ax.set_ylabel('Difference in annual cost (USD) from SQ scenario')
    ax.set_xlabel('Scenario')
    ax.margins(x=0.02)

    # Light separators between scenario pairs
    for idx in range(2, len(ordered_scenario_ids_pathways), 2):
        ax.axvline(idx - 0.5, linestyle=':', linewidth=0.8, color='lightgray', zorder=0)

    # === Legend: hatch only (adjustment levels)
    hatch_handles, hatch_labels = zip(*hatch_patches)
    hatch_legend = fig.legend(hatch_handles, hatch_labels, title="Hypothetical Assumption on Improved HCW Efficiency",
                              loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4)

    # === Save without cropping legend
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{g_path}/adj_total_costs.png',
                bbox_inches='tight',
                bbox_extra_artists=[hatch_legend])
    plt.show()

figure_adjusted_avg_difference_in_cost_from_status_quo_per_year(incremental_scenario_cost_annual_summarized,
                                                                incremental_scenario_cost_annual_adj_25,
                                                                incremental_scenario_cost_annual_adj_50,
                                                                incremental_scenario_cost_annual_adj_75)


#  CALCULATE AND PLOT COST PER DALY AVERTED
def get_cost_per_daly_averted(incremental_cost):
    columns_of_interest = [20, 21, 22, 23, 24, 25, 26, 27]

    all_total_dalys_averted = diff_age_standardized_dalys_sum
    total_dalys_averted = all_total_dalys_averted[columns_of_interest].xs('central', axis=1, level=1)

    # pivot costs
    inc_cost = incremental_cost['mean'].to_frame().T
    final_costs = inc_cost[columns_of_interest]

    final_costs.index = total_dalys_averted.index

    cost_per_daly_averted = final_costs / total_dalys_averted

    return cost_per_daly_averted

total_inc_cost_unadj = find_cost_diff_from_sq_and_sum(costs_und_adj_hcw_ratio[0])
inc_cost_adj_25 = find_cost_diff_from_sq_and_sum(costs_adj_25_hcw_ratio[0])
inc_cost_adj_50 = find_cost_diff_from_sq_and_sum(costs_adj_50_hcw_ratio[0])
inc_cost_adj_75 = find_cost_diff_from_sq_and_sum(costs_adj_75_hcw_ratio[0])


cost_per_daly_averted = []
for cost_data in [total_inc_cost_unadj,
                  inc_cost_adj_25,
                  inc_cost_adj_50,
                  inc_cost_adj_75]:
    cost_per_daly_averted.append(get_cost_per_daly_averted(cost_data))
cost_per_daly_averted_df = pd.concat(cost_per_daly_averted)
cost_per_daly_averted_df.index = ['Unadjusted', '25% adjusted', '50% adjusted', '75% adjusted']

def figure_cost_per_daly_averted_group_bar_chart(cost_per_daly_averted_df):
    labels = [full_lab[scen_draws[sc]] for sc in ordered_scenario_ids_pathways]

    # Reorder: columns are scenarios; rows are assumptions (each row = one series)
    data = cost_per_daly_averted_df[ordered_scenario_ids_pathways]  # use the pathways order
    n_groups = data.shape[1]           # scenarios
    n_series = data.shape[0]           # assumptions (bars per group)

    x = np.arange(n_groups, dtype=float)
    width = min(0.8 / max(n_series, 1), 0.2)
    offset_start = - (n_series - 1) * width / 2.0

    fig, ax = plt.subplots(figsize=(12, 6))

    # Colors per scenario (consistent across assumptions)
    bar_colors = [color_map_pathways[sc] for sc in ordered_scenario_ids_pathways]

    # Hatch per assumption
    base_hatches = ["", "//", "xx", "..", "\\\\", "++", "**", "oo", "--"]
    hatches = [base_hatches[i % len(base_hatches)] for i in range(n_series)]
    hatch_patches = []

    # Annotation format
    def fmt_money(val):
        v = float(val)
        return f"${v:.0f}"

    # Plot series (each row) with hatch + outlines; annotate
    for i, row_label in enumerate(data.index):
        vals = data.iloc[i].values.astype(float)
        xpos = x + offset_start + i * width

        bars = ax.bar(
            xpos, vals, width=width,
            color=bar_colors, alpha=0.9,
            hatch=hatches[i],
            edgecolor='black', linewidth=0.7
        )

        # Annotations: slight offset based on data range
        # compute per-plot offset once we know the current y-lims; use a temporary
        # safe default; we'll refine after plotting all bars by resetting texts positions
        for j, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(
                xpos[j], val,
                fmt_money(val),
                ha='center', va='bottom' if val >= 0 else 'top',
                fontsize='x-small'
            )

        # Legend entry for hatch
        patch = plt.Rectangle((0, 0), 1, 1, facecolor="white",
                              hatch=hatches[i], edgecolor="black", linewidth=0.7)
        hatch_patches.append((patch, str(row_label)))

    # Cosmetics
    ax.axhline(0, linewidth=1, linestyle='--', color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7, ha='right')
    ax.set_ylabel("Cost per DALY averted (USD)")
    ax.margins(x=0.02)

    # Light separators between scenario pairs (22–23, 24–25, 20–21, 26–27 in that order)
    for idx in range(2, len(ordered_scenario_ids_pathways), 2):
        ax.axvline(idx - 0.5, linestyle=':', linewidth=0.8, color='lightgray', zorder=0)

    # Hatch legend only
    hatch_handles, hatch_labels = zip(*hatch_patches)
    hatch_legend = fig.legend(hatch_handles, hatch_labels, title="Hypothetical Assumption on Improved HCW Efficiency",
                              loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=min(n_series, 5))

    # Nudge annotation positions slightly away from bars using a fraction of y-range
    ymin, ymax = ax.get_ylim()
    yspan = max(ymax - ymin, 1.0)
    offset = 0.02 * yspan
    # Update positions: re-loop texts in this Axes
    for txt in ax.texts:
        x_t, y_t = txt.get_position()
        if txt.get_va() == 'bottom':
            txt.set_position((x_t, y_t + offset))
        else:
            txt.set_position((x_t, y_t - offset))

    # Save without cropping legend
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{g_path}/cost_per_daly_averted.png',
                bbox_inches='tight',
                bbox_extra_artists=[hatch_legend])
    plt.show()

figure_cost_per_daly_averted_group_bar_chart(cost_per_daly_averted_df)

