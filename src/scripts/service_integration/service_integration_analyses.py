from pathlib import Path

from collections import Counter, defaultdict

import os
import scipy.stats as st

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker

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

for scenario in scen_draws:
    make_folder(f'{g_path}/{scen_draws[scenario]}')

# create a dict with proper labels for each scenario
full_lab = {'htn':'Hypertension screening',
            'htn_max': 'Hypertension screening (max.)',
            'dm': 'Diabetes screening',
            'dm_max': 'Diabetes screening (max.)',
            'hiv': 'HIV screening',
            'hiv_max': 'HIV screening (max.)',
            'tb': 'Tb screening',
            'tb_max':'Tb screening (max.)',
            'mal':'Malnutrition screening',
            'mal_max':'Malnutrition screening (max.)',
            'fp_scr':'Family planning (WRA)',
            'fp_scr_max':'Family planning (WRA) (max.)',
            'anc': 'Antenatal care',
            'anc_max': 'Antenatal care (max.)',
            'pnc':'Postnatal care',
            'pnc_max':'Postnatal care (max.)',
            'fp_pn': 'Family planning (postnatal)',
            'fp_pn_max':'Family planning (postnatal) (max.)',
            'epi': 'EPI',
            'chronic_care': 'Chronic care clinic',
            'chronic_care_max': 'Chronic care clinic (max.)',
            'all_screening': 'Integrated screening',
            'all_screening_max':'Integrated screening (max.)',
            'all_mch': 'Maternal and child health clinic',
            'all_mch_max': 'Maternal and child health clinic (max.)',
            'all_int': 'All pathways',
            'all_int_max': 'All pathways (max.)'}

# define functions to be used throughout
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

def age_standardize_dalys(dalys_df):
    """Age-standardizes DALYs across draws using the population size from the Status Quo scenario"""

    def get_pop_by_agegrp_label(df):
        """Sum the dalys by period, sex, age-group and label"""
        df['year'] = df['date'].dt.year
        df_melted = df.melt(id_vars=['year'], value_vars=[col for col in df.columns if col not in ['date', 'year']],
                            var_name='age_group', value_name='count')
        series_multi = df_melted.set_index(['year', 'age_group'])['count'].sort_index()

        return series_multi

    # Get the total population across the draws
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

# -------------------------------------------------- ANALYSIS ---------------------------------------------------------
# Define target period
TARGET_PERIOD = (Date(2025, 1, 1), Date(2054, 12, 31))

# =============================== TOTAL DALYS AVERTED BY SCENARIO (TIME DELINEATION) =================================

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


def figure_total_dalys_averted_by_scenario_with_uncertainty(data, age_standardized, drop_non_sig_results):
    """Outputs an annotated bar graph showing the mean total DALYs averted by scenario along with 95% confidence
    intervals"""

    labels = [full_lab[val] for val in scen_draws.values()]

    # extract data
    mean = [float(data[v, 'central'].values) for v in scen_draws.keys()]
    lower_errors = [float(data[v, 'lower'].values) for v in scen_draws.keys()]
    upper_errors = [float(data[v, 'upper'].values) for v in scen_draws.keys()]

    # Optional - drop results from the graph if CI includes 0
    if drop_non_sig_results:
        ci_includes_0 = [i for i, (a, b) in enumerate(zip(lower_errors, upper_errors)) if a < 0 and b > 0]

        labels = [val for i, val in enumerate(labels) if i not in ci_includes_0]

        mean = [val for i, val in enumerate(mean) if i not in ci_includes_0]
        lower_errors = [val for i, val in enumerate(lower_errors) if i not in ci_includes_0]
        upper_errors = [val for i, val in enumerate(upper_errors) if i not in ci_includes_0]

    # Compute distances from mean to bounds (must be non-negative)
    yerr_lower = [med - low for med, low in zip(mean, lower_errors)]
    yerr_upper = [up - med for med, up in zip(mean, upper_errors)]

    # generate colours
    cmap = plt.get_cmap('viridis')  # 'tab10', 'tab20', 'viridis', etc.
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    # Create bar chart with error bars
    fig, ax = plt.subplots()
    bars = ax.bar(labels, mean, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black', color=colors)

    # Add horizontal line at y=0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Annotate values on top of bars
    for bar, value, err_up in zip(bars, mean, upper_errors):
        height = bar.get_height()
        scaled_val = value / 1e6  # convert to millions
        offset = max(upper_errors) * 0.05  # increase vertical offset

        if height >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + offset,
                    f'{scaled_val:.1f}M',
                    ha='center', va='bottom', fontsize=6)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height - offset,
                    f'{scaled_val:.1f}M',
                    ha='center', va='top', fontsize=6)

    # Axis labels and title
    ax.set_ylabel('DALYs Averted (ten millions)', fontsize=7)
    ax.set_title('Total DALYs Averted by Scenario between 2025 and 2055 (mean and 95% CI)' if not age_standardized else
                 'Total Age-Standardized DALYs Averted by Scenario between 2025 and 2055 (mean and 95% CI)',
                 fontsize=8)

    # Adjust label size and layout
    plt.xticks(fontsize=7, rotation=90)
    plt.tight_layout()

    # Save and show
    plt.savefig(f'{g_path}/total_dalys_averted_by_scenario.png' if not age_standardized else
                f'{g_path}/total_age_standardized_dalys_averted_by_scenario.png',
                bbox_inches='tight')
    plt.show()

# Output plots for non age-standardized and age-standardized DALYs
figure_total_dalys_averted_by_scenario_with_uncertainty(data=diff_total_dalys_non_standardized,
                                                        age_standardized=False,
                                                        drop_non_sig_results=False)

figure_total_dalys_averted_by_scenario_with_uncertainty(data=diff_age_standardized_dalys_sum,
                                                        age_standardized=True,
                                                        drop_non_sig_results=False)

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

    # Viridis color map
    n_scenarios = len(scenarios)
    cmap = cm.get_cmap('viridis', n_scenarios)
    colors = [cmap(i) for i in range(n_scenarios)]

    # Plot settings
    n_cols = 5
    n_rows = -(-n_scenarios // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.8), sharey=True)
    axes = axes.flatten()

    for i, scenario in enumerate(scenarios):
        ax = axes[i]

        means = data_mean[scenario]
        yerr = [
            data_err_lower[scenario],
            data_err_upper[scenario]
        ]

        # Plot with error bars
        ax.errorbar(
            period_labels,
            means,
            yerr=yerr,
            fmt='-o',
            color=colors[i],
            capsize=3,
            linewidth=1.5,
            markersize=5
        )

        # Add horizontal zero line
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        # Compute y-range for padding
        y_vals = means
        ymin = min([m - l for m, l in zip(means, data_err_lower[scenario])])
        ymax = max([m + u for m, u in zip(means, data_err_upper[scenario])])
        padding = 0.02 * (ymax - ymin)  # 2% of full range

        # Annotate above upper error bar
        for j, val in enumerate(means):
            upper_err = data_err_upper[scenario][j]
            label = f"{val / 1e6:.1f}M" if abs(val) > 1e6 else f"{val / 1e3:.1f}K"
            ax.text(
                j,  # x-position (time period index)
                val + upper_err + padding,  # y-position above upper error bar
                label,
                fontsize=9,
                ha='center',
                va='bottom'
            )

        ax.set_title(str(full_lab[scen_draws[scenario]]), fontsize=9)

        # X-axis labels only on bottom row
        if i // n_cols == n_rows - 1:
            ax.set_xticklabels(period_labels)
        else:
            ax.set_xticklabels([])

        # Y-axis ticks only on leftmost column
        if i % n_cols == 0:
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Layout and labels
    fig.suptitle("Age-standardized DALYs Averted Compared to Baseline by Scenario Disaggregated by Time Period",
                 fontsize=14)
    fig.text(0.04, 0.5, 'DALYs averted (ten millions)', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout(rect=[0.05, 0, 1, 0.95])  # leave space for y-label and title

    plt.savefig(f'{g_path}/total_dalys_averted_by_scenario_time.png' if not age_standardized else
                f'{g_path}/total_age_standardized_dalys_averted_by_scenario_time.png',
                bbox_inches='tight')
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

    # STEP 2: Extract central, lower, upper from MultiIndex or flat columns
    central_df = data.loc[:, data.columns.get_level_values(1) == 'central']
    lower_df = data.loc[:, data.columns.get_level_values(1) == 'lower']
    upper_df = data.loc[:, data.columns.get_level_values(1) == 'upper']

    # Clean up column names
    central_df.columns = [full_lab[scen_draws[col[0]]]  for col in central_df.columns]
    lower_df.columns = [full_lab[scen_draws[col[0]]]  for col in lower_df.columns]
    upper_df.columns = [full_lab[scen_draws[col[0]]]  for col in upper_df.columns]

    # STEP 3: Create a mask for where 0 is within the interval
    uncertainty_includes_zero = (lower_df <= 0) & (upper_df >= 0)
    significant = ~uncertainty_includes_zero

    # Function to format numbers nicely
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

    # Create annotation matrix with formatted numbers and asterisk for significant
    annot = central_df.copy().astype(str)
    for i in range(central_df.shape[0]):
        for j in range(central_df.shape[1]):
            if significant.iloc[i, j]:
                val = central_df.iloc[i, j]
                annot.iloc[i, j] = format_daly(val) + '*'
            else:
                annot.iloc[i, j] = ""

    # Set color scale bounds
    vmin = central_df.min().min()
    vmax = central_df.max().max()
    abs_max = max(abs(vmin), abs(vmax))

    # Optional: emphasize values closer to zero
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # Plot
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        central_df,
        cmap="RdBu_r",
        norm=norm,
        annot=annot,
        fmt="",
        linewidths=0.5,
        cbar_kws={'label': 'DALY Difference from Status Quo'}
    )

    plt.title("Differences in Cause-Specific Age-Standardized DALYs from Status Quo (Central Estimates, Significant Only)")
    plt.ylabel("Condition")
    plt.xlabel("Scenario")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{g_path}/cause_specific_dalys_heatmap.png', bbox_inches='tight')
    plt.show()

figure_heatmap_cause_specific_dalys_averted(diff_dalys_by_cause_age_standardize)


def figure_total_cause_specific_dalys_averted_by_scenario(data_no_year, data_by_years,
                                                          int_scen, drop_non_sig_results,
                                                          age_standardized):

    # FIGURE WITH CIs
    labels = data_no_year.index.values

    mean = data_no_year[k, 'central'].values
    lower_errors = data_no_year[k, 'lower'].values
    upper_errors = data_no_year[k, 'upper'].values

    if drop_non_sig_results:
        ci_includes_0 = [i for i, (a, b) in enumerate(zip(lower_errors, upper_errors)) if a < 0 and b > 0]

        labels = [val for i, val in enumerate(labels) if i not in ci_includes_0]
        mean = [val for i, val in enumerate(mean) if i not in ci_includes_0]
        lower_errors = [val for i, val in enumerate(lower_errors) if i not in ci_includes_0]
        upper_errors = [val for i, val in enumerate(upper_errors) if i not in ci_includes_0]

    yerr_lower = mean - lower_errors
    yerr_upper = upper_errors - mean

    cmap = plt.get_cmap('magma')  # 'tab10', 'tab20', 'viridis', etc.
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    # Create bar chart with error bars
    fig, ax = plt.subplots()
    bars = ax.bar(labels, mean, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black', color=colors)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Annotate values on top of bars
    for bar, value, err_up in zip(bars, mean, upper_errors):
        height = bar.get_height()
        scaled_val = value / 1e3  # convert to millions
        offset = max(upper_errors) * 0.01  # increase vertical offset

        if height >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + offset,
                    f'{scaled_val:.1f}K',
                    ha='center', va='bottom', fontsize=6)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height - offset,
                    f'{scaled_val:.1f}K',
                    ha='center', va='top', fontsize=6)

    ax.set_ylabel('DALYs Averted')
    ax.set_title(f'Cause-specific DALYs Averted for Scenario (with uncertainty):{scen_draws[int_scen]}'
                 if not age_standardized else f'Age Standardized Cause-specific DALYs Averted for '
                                              f'Scenario (with uncertainty):{scen_draws[int_scen]}')

    # Adjust label size
    plt.xticks(fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(f'{g_path}/{scen_draws[int_scen]}/'
                f'cause_specific_dalys_diff_with_ci_{scen_draws[int_scen]}.png' if not age_standardized
                else f'{g_path}/{scen_draws[int_scen]}/cause_specific_dalys_diff_with_ci_stnd_'
                     f'{scen_draws[int_scen]}.png', bbox_inches='tight')
    plt.show()

# for k in scen_draws:
#     figure_total_cause_specific_dalys_averted_by_scenario(data_no_year=diff_dalys_by_cause,
#                                                           data_by_years=diff_dalys_by_cause_year,
#                                                           int_scen=k,
#                                                           drop_non_sig_results=False,
#                                                           age_standardized=False)
#
#
#
# for k in scen_draws:
#     figure_total_cause_specific_dalys_averted_by_scenario(data_no_year=diff_dalys_by_cause_age_standardize,
#                                                           data_by_years=diff_dalys_by_cause_year_age_standardize,
#                                                           int_scen=k,
#                                                           drop_non_sig_results=False,
#                                                           age_standardized=True)

# ===================================== DIFFERENCE IN APPOINTMENTS BY SCENARIO ========================================
# APPOINTMENT TYPES
counts_by_treatment_id_and_appt_type =  bin_hsi_event_details(
            results_folder,
            lambda event_details, count: sum(
                [
                    Counter({
                        (
                            event_details["treatment_id"],
                            appt_type
                        ):
                        count * appt_number
                    })
                    for appt_type, appt_number in event_details["appt_footprint"]
                ],
                Counter()
            ),
            *TARGET_PERIOD,
            True
        )
apt_data = compute_service_statistics(counts_by_treatment_id_and_appt_type)
apt_type_summ = apt_data[0]
p_diff_apt_type_sum = apt_data[1]

# apt_type_summ_to_save = reform_df_to_save(apt_type_summ, 0)
# diff_apt_typ_to_save = reform_df_to_save(apt_type_summ, 0)

# def figure_diff_in_total_appointment_type_by_scenario(data, k):
#     labels = data.index
#
#     mean = [data.at[appt, k]['mean'] for appt in labels]
#     lower_errors = [data.at[appt, k]['lower'] for appt in labels]
#     upper_errors = [data.at[appt, k]['upper'] for appt in labels]
#
#     yerr_lower = [med - low for med, low in zip(mean, lower_errors)]
#     yerr_upper = [up - med for med, up in zip(mean, upper_errors)]
#
#     perc_95 = np.percentile(upper_errors, 95)
#     extreme_max = max(upper_errors)
#     should_split = extreme_max > 1.3 * perc_95  # Only break axis if top values are ~30% larger than the 95th percentile
#
#     if should_split:
#         # Broken axis approach
#         fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), height_ratios=[1, 3])
#         ax1.bar(labels, mean, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black')
#         ax2.bar(labels, mean, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black')
#
#         threshold = perc_95  # You could also choose mean + 2*std for more strict cutoff
#
#         # Set y-limits
#         ax1.set_ylim(threshold, extreme_max * 1.05)
#         ax2.set_ylim(min(lower_errors) * 1.05, threshold)
#
#         # Hide spines and add break marks
#         ax1.spines['bottom'].set_visible(False)
#         ax2.spines['top'].set_visible(False)
#         ax1.tick_params(labeltop=False)
#         ax2.xaxis.tick_bottom()
#
#         d = .015
#         kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#         ax1.plot((-d, +d), (-d, +d), **kwargs)
#         ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
#         kwargs.update(transform=ax2.transAxes)
#         ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
#         ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
#
#         fig.suptitle(f'Total Additional Appointments by Appointment Type: {scen_draws[k]}')
#         ax2.set_ylabel('Additional Appointments')
#
#     else:
#         # Simple plot with full range
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.bar(labels, mean, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black')
#         ax.axhline(0, color='gray', linestyle='--', linewidth=1)
#         ax.set_ylim(min(lower_errors) * 1.05, extreme_max * 1.05)
#         ax.set_ylabel('Additional Appointments')
#         ax.set_title(f'Total Additional Appointments by Appointment Type: {scen_draws[k]}')
#
#     # Shared formatting
#     plt.xticks(fontsize=8, rotation=90)
#     plt.tight_layout()
#     plt.savefig(f'{g_path}/{scen_draws[k]}/add_appt_by_type_{scen_draws[k]}.png', bbox_inches='tight')
#     plt.show()

#
# for k in scen_draws:
#    figure_diff_in_total_appointment_type_by_scenario(p_diff_apt_type_sum, k)


def figure_dotplot_difference_in_appointments_by_scenario(data):
    data = data.drop(columns=0)

    # Extract mean, lower, upper
    mean_df = data.applymap(lambda x: x['mean'])
    lower_df = data.applymap(lambda x: x['lower'])
    upper_df = data.applymap(lambda x: x['upper'])

    # === CONFIG ===
    scenarios = list(mean_df.columns)
    num_scenarios = len(scenarios)
    panels_per_fig = 6
    n_rows, n_cols = 2, 3  # 6 plots per figure
    fig_count = (num_scenarios + panels_per_fig - 1) // panels_per_fig

    top_n = 20
    top_appointments = (
        mean_df.abs().mean(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    # Subset to top appointment types
    mean_df_plot = mean_df.loc[top_appointments]
    lower_df_plot = lower_df.loc[top_appointments]
    upper_df_plot = upper_df.loc[top_appointments]
    significant = ~((lower_df_plot <= 0) & (upper_df_plot >= 0))

    # === Format helper ===
    # def format_number(val):
    #     abs_val = abs(val)
    #     if abs_val < 1_000:
    #         return f"{int(val)}"
    #     elif abs_val < 1_000_000:
    #         return f"{val / 1_000:.0f}K"
    #     elif abs_val < 10_000_000:
    #         return f"{val / 1_000_000:.1f}M"
    #     else:
    #         return f"{val / 1_000_000:.0f}M"

    # === PLOTTING ===
    for fig_idx in range(fig_count):
        start = fig_idx * panels_per_fig
        end = min(start + panels_per_fig, num_scenarios)
        scenario_subset = scenarios[start:end]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharey=True)
        axes = axes.flatten()

        for i, scen in enumerate(scenario_subset):
            ax = axes[i]

            means = mean_df_plot[scen]
            lowers = lower_df_plot[scen]
            uppers = upper_df_plot[scen]
            sig = significant[scen]

            labels = means.index.tolist()
            y_pos = np.arange(len(labels))

            for j, label in enumerate(labels):
                x = means[label]
                y = y_pos[j]
                low = x - lowers[label]
                up = uppers[label] - x

                # Color based on significance and direction
                if sig[label]:
                    color = 'blue' if x > 0 else 'red'
                else:
                    color = 'gray'

                # Plot dot + error bar
                ax.errorbar(
                    x, y,
                    xerr=[[low], [up]],
                    fmt='o',
                    color=color,
                    ecolor=color,
                    capsize=3
                )

                # Annotate if significant
                if sig[label]:
                    ax.text(
                        x + np.sign(x) * 0.05 * abs(x),
                        y,
                        # format_number(x),
                        round(x),
                        va='center',
                        ha='left' if x >= 0 else 'right',
                        fontsize=8,
                        color=color
                    )

            # Reference line
            ax.axvline(0, color='black', linestyle='--', linewidth=1)

            # Title from external mappings
            ax.set_title(full_lab[scen_draws[scen]])

            # Set y-ticks and y-labels
            ax.set_yticks(y_pos)
            if i % n_cols == 0:
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_ylabel("Appointment Type")
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

            # Format x-axis
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: x))
            ax.grid(True, axis='x', linestyle=':', alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, n_rows * n_cols):
            fig.delaxes(axes[j])

        # Title and layout
        fig.suptitle(f"Percentage Difference in Number of Appointments from Status Quo by Scenario", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

figure_dotplot_difference_in_appointments_by_scenario(p_diff_apt_type_sum)

# ===================================== DIFFERENCE IN HCW TIME BY SCENARIO ========================================
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

hsi = {'htn':['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
             'CardioMetabolicDisorders_Investigation_hypertension',
             'CardioMetabolicDisorders_Investigation_hypertension_and_diabetes',
             'CardioMetabolicDisorders_Prevention_WeightLoss_hypertension',
            'CardioMetabolicDisorders_Treatment_hypertension'],

            'dm': [ 'CardioMetabolicDisorders_Investigation_diabetes',
                    'CardioMetabolicDisorders_Investigation_hypertension_and_diabetes',
                    'CardioMetabolicDisorders_Prevention_WeightLoss_diabetes',
                    'CardioMetabolicDisorders_Treatment_diabetes'],

            'hiv': ['Hiv_Test', 'Hiv_Treatment'],

            'tb': ['Tb_Test_Screening',
                    'Tb_Test_Clinical',
                    'Tb_Test_Culture',
                    'Tb_Test_Xray',
                    'Tb_Treatment',
                     'Tb_Test_FollowUp'
                     ],

            'mal':['Undernutrition_Feeding'],

            'fp_scr':['Contraception_Routine'],

            'anc': ['AntenatalCare_Outpatient', 'AntenatalCare_FollowUp'],

            'pnc': ['PostnatalCare_Neonatal', 'PostnatalCare_Maternal'],

            'fp_pn': ['Contraception_Routine_Postnatal'],

            'epi': ['Epi_Childhood_Bcg',
                    'Epi_Childhood_Opv',
                    'Epi_Childhood_DtpHibHep',
                    'Epi_Childhood_Rota',
                    'Epi_Childhood_Pneumo',
                    'Epi_Childhood_MeaslesRubella',
                    'Epi_Pregnancy_Td'
                    ],

            'chronic_care': ['CardioMetabolicDisorders_Investigation_diabetes',
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
                     'Epilepsy_Treatment_Followup'],

            'all_screening': ['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
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
                                        'Tb_Test_FollowUp'
                                        ],

            'all_mch': ['Undernutrition_Feeding',
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
                                        ],

            'all_int': ['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
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
                                         'Epilepsy_Treatment_Followup']}

hcw_time_by_treatment_id_df = pd.DataFrame.from_dict(hcw_time_by_treatment_id)


def adjust_and_plot_hcw_time_use(scen, data, hsis):
    scenario_name = scen_draws[scen]
    if scenario_name.endswith('_max'):
         scenario_name = scenario_name.removesuffix('_max')

    sq_data = data[0]
    sq_data = sq_data.fillna(0)
    sq_data.index.names = ['first', 'second']
    sq_data_grouped = sq_data.groupby(level='first').sum()

    scen_data = data[scen]
    scen_data = scen_data.fillna(0)
    scen_data.index.names = ['first', 'second']

    adj_scen_data_25 = scen_data.copy()
    adj_scen_data_50 = scen_data.copy()

    hsi_list = hsis[scenario_name]

    def adjust(df, factor):
        mask = df.index.get_level_values('second').isin(hsi_list)
        df.loc[mask] = df.loc[mask] * factor

    adjust(adj_scen_data_25, 0.75)
    adjust(adj_scen_data_50, 0.50)

    def get_plot_data(df):
        grouped = df.groupby(level='first').sum()
        p_diff = (grouped - sq_data_grouped) / sq_data_grouped * 100
        p_diff.columns = data.columns[k * 10: k*10 +10]
        p_diff.columns.names = ['draw', 'run']
        summ = compute_summary_statistics(p_diff, use_standard_error=True)

        mean = list(summ[k, 'central'].values)
        lower = list(summ[k, 'lower'].values)
        upper = list(summ[k, 'upper'].values)

        yerr_lower = [med - low for med, low in zip(mean, lower)]
        yerr_upper = [up - med for med, up in zip(mean, upper)]

        labels = summ.index.values
        return mean, yerr_lower, yerr_upper, labels


    scen_data_summ = get_plot_data(scen_data)
    adj_scen_data_25_summ = get_plot_data(adj_scen_data_25)
    adj_scen_data_50_summ = get_plot_data(adj_scen_data_50)

    means1, yerr_low1, yerr_up1, labels = scen_data_summ
    means2, yerr_low2, yerr_up2, _ = adj_scen_data_25_summ
    means3, yerr_low3, yerr_up3, _ = adj_scen_data_50_summ

    # Plot settings
    bar_width = 0.25
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar group 1
    ax.bar(
        x,
        scen_data_summ[0],
        width=bar_width,
        yerr=[scen_data_summ[1], scen_data_summ[2]],
        capsize=4,
        label='Unadjusted P.Diff'
    )

    # Bar group 2
    ax.bar(
        x + bar_width,
        adj_scen_data_25_summ[0],
        width=bar_width,
        yerr=[adj_scen_data_25_summ[1], adj_scen_data_25_summ[2]],
        capsize=4,
        label='Adjusted P.Diff (25% reduction)'
    )

    # Bar group 3
    ax.bar(
        x + 2 * bar_width,
        adj_scen_data_50_summ[0],
        width=bar_width,
        yerr=[adj_scen_data_50_summ[1], adj_scen_data_50_summ[2]],
        capsize=4,
        label='Adjusted P.Diff (50% reduction)'
    )

    # Format axes
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Percentage Difference from SQ(%)')
    ax.set_title(f'Percentage Change in HCW Time use by Cadre for Scenario {full_lab[scen_draws[k]]}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{g_path}/{scen_draws[k]}/hcw_time_by_cadre_{scen_draws[k]}.png', bbox_inches='tight')
    plt.show()


for k in scen_draws:
    adjust_and_plot_hcw_time_use(k, hcw_time_by_treatment_id_df, hsi)


# ==================================== CONSUMABLE COST BY SCENARIO (AND DIFFS) ========================================
# https://github.com/UCL/TLOmodel/blob/ec8929949c694b3a503d34051575f0dc7e7a32c3/src/scripts/comparison_of_horizontal_and_vertical_programs/economic_analysis_for_manuscript/roi_analysis_horizontal_vs_vertical.py#L45
# 606 - 635, 1329-1348

list_of_relevant_years_for_costing = list(range(TARGET_PERIOD[0].year, TARGET_PERIOD[-1].year + 1))

input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
                                               _years=list_of_relevant_years_for_costing,
                                               cost_only_used_staff=True,
                                               _discount_rate = 0.03)

total_input_cost = input_costs.groupby(['draw', 'run'])['cost'].sum()
# TODO: currently this is mean and quartiles (update)
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
input_costs_for_plot_summarized = input_costs.groupby(['draw', 'year', 'cost_subcategory', 'Facility_Level',
                                                       'cost_subgroup', 'cost_category']).agg(
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

# Plot incremental costs
incremental_scenario_cost_summarized = summarize_cost_data(incremental_scenario_cost)

def figure_7_difference_in_cost_from_status_quo(cost_data):
    name_of_plot = f'Incremental scenario cost relative to baseline during intervention period'

    yerr = np.array([
            (cost_data['mean'] - cost_data['lower']).values,
            (cost_data['upper'] - cost_data['mean']).values,
        ])

    xticks = {(i + 0.5): k for i, k in enumerate(cost_data.index)}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        xticks.keys(),
        cost_data['mean'].values,
        yerr=yerr,
        ecolor='black',
        capsize=10,
        label=scen_draws.values()
    )

    annotations=[
            f"{round(row['mean'] / 1e6, 1)} \n ({round(row['lower'] / 1e6, 1)}- \n {round(row['upper'] / 1e6, 1)})"
            for _, row in incremental_scenario_cost_summarized.iterrows()
        ]

    for xpos, (ypos, text) in zip(xticks.keys(), zip(cost_data['upper'].values.flatten(), annotations)):
        annotation_y = ypos + 1e6

        ax.text(
            xpos,
            annotation_y,
            '\n'.join(text.split(' ', 1)),
            horizontalalignment='center',
            verticalalignment='bottom',  # Aligns text at the bottom of the annotation position
            fontsize='x-small',
            rotation='horizontal'
        )

    ax.set_xticks(list(xticks.keys()))
    ax.set_xticklabels(scen_draws.values())
    plt.xticks(rotation=90)

    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.15, right=0.85)

    ax.set_title(name_of_plot)
    ax.set_ylabel('Cost \n(USD Millions)')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(Path(g_path) / name_of_plot.replace(' ', '_').replace(',', ''))
    plt.close(fig)

figure_7_difference_in_cost_from_status_quo(incremental_scenario_cost_summarized)

# =============================================== OTHER FIGURES ======================================================

# 'htn'
# index = diff_dalys_by_year.index
#
# def get_ncd_data(index):
#     htn_diag = extract_results(
#                 results_folder,
#                 module="tlo.methods.cardio_metabolic_disorders",
#                 key="hypertension_diagnosis_prevalence",
#                 # custom_generate_series=(
#                 #     lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].mean()),
#                 column='hypertension_diagnosis_prevalence',
#                 do_scaling=False
#             )
#     htn_diag.index = index
#     htn_diag_summ = compute_summary_statistics(htn_diag, use_standard_error=True)
#
#     htn_med = extract_results(
#                 results_folder,
#                 module="tlo.methods.cardio_metabolic_disorders",
#                 key="hypertension_medication_prevalence",
#                 column='hypertension_medication_prevalence',
#                 do_scaling=False
#             )
#     htn_med_summ = compute_summary_statistics(htn_med, use_standard_error=True)
#
#     dm_diag = extract_results(
#                 results_folder,
#                 module="tlo.methods.cardio_metabolic_disorders",
#                 key="diabetes_diagnosis_prevalence",
#                 column='diabetes_diagnosis_prevalence',
#                 do_scaling=False
#             )
#     dm_diag_summ = compute_summary_statistics(dm_diag, use_standard_error=True)
#
#     dm_med = extract_results(
#                 results_folder,
#                 module="tlo.methods.cardio_metabolic_disorders",
#                 key="diabetes_medication_prevalence",
#                 column='diabetes_medication_prevalence',
#                 do_scaling=False
#             )
#
#     dm_med_summ = compute_summary_statistics(dm_med, use_standard_error=True)
#
#
#     return htn_diag, dm_diag, dm_med
#
# get_ncd_data(index)
#
# def additional_figs_ncd_diag_treatment_coverage(data):
#     pass
#
#
# #  'fp'
#
#
#
# # 'hiv' / 'tb'
# # 'anc'
# # 'pnc'
# # 'mal'
# # 'epi'
