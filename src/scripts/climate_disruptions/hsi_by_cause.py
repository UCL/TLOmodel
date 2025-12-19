import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    summarize,
    get_color_short_treatment_id
)

# Configuration
MIN_YEAR = 2025
MAX_YEAR = 2041
SPACING_OF_YEARS = 1
PREFIX_ON_FILENAME = '1'

# Analysis type flags
CLIMATE_SENSITIVITY_ANALYSIS = True
PARAMETER_SENSITIVITY_ANALYSIS = False
MAIN_TEXT = True

# Scenario definitions
SCENARIO_NAMES_ALL = [
    "Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean",
    "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",
    "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"
]

SCENARIO_COLOURS = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167'] * 4

# Precipitation file paths
PRECIPITATION_FILES = {
    "Baseline": "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv",
    "SSP 1.26 High": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp126/highest_monthly_prediction_weather_by_facility.csv",
    "SSP 1.26 Low": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp126/lowest_monthly_prediction_weather_by_facility.csv",
    "SSP 1.26 Mean": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp126/mean_monthly_prediction_weather_by_facility.csv",
    "SSP 2.45 High": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp245/highest_monthly_prediction_weather_by_facility.csv",
    "SSP 2.45 Low": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp245/lowest_monthly_prediction_weather_by_facility.csv",
    "SSP 2.45 Mean": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp245/mean_monthly_prediction_weather_by_facility.csv",
    "SSP 5.85 High": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp585/highest_monthly_prediction_weather_by_facility.csv",
    "SSP 5.85 Low": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp585/lowest_monthly_prediction_weather_by_facility.csv",
    "SSP 5.85 Mean": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp585/mean_monthly_prediction_weather_by_facility.csv",
}

# Determine scenario configuration based on analysis type
if CLIMATE_SENSITIVITY_ANALYSIS:
    scenario_names = SCENARIO_NAMES_ALL
    suffix = "climate_SA"
    scenarios_of_interest = range(len(scenario_names))
elif PARAMETER_SENSITIVITY_ANALYSIS:
    scenario_names = list(range(0, 10, 1))
    suffix = "parameter_SA"
    scenarios_of_interest = range(len(scenario_names))
else:
    scenario_names = SCENARIO_NAMES_ALL
    suffix = "default"
    scenarios_of_interest = range(len(scenario_names))

if MAIN_TEXT:
    scenario_names = ["Baseline", "SSP 2.45 Mean"]
    suffix = "main_text"
    scenarios_of_interest = [0, 1]


def paired_draw_significance(
    baseline: pd.Series,
    climate: pd.Series,
    alpha_levels=(0.05, 0.01, 0.001),
    return_details=False
):
    """
    Compute statistical significance using paired simulation draws.

    Parameters
    ----------
    baseline : pd.Series
        Draw-level values for the baseline scenario
    climate : pd.Series
        Draw-level values for the climate scenario
    alpha_levels : tuple
        Significance thresholds (default: 0.05, 0.01, 0.001)
    return_details : bool
        If True, return effect size and CI of the difference

    Returns
    -------
    star : str
        Significance stars ('', '*', '**', '***')
    details : dict (optional)
        Contains p-value, median difference, and 95% CI
    """
    diff = climate - baseline

    if np.allclose(diff, 0):
        return ('', {}) if return_details else ''

    try:
        stat, p_value = stats.wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
    except ValueError:
        return ('', {}) if return_details else ''

    # Determine significance stars
    if p_value < alpha_levels[2]:
        star = '***'
    elif p_value < alpha_levels[1]:
        star = '**'
    elif p_value < alpha_levels[0]:
        star = '*'
    else:
        star = ''

    if not return_details:
        return star

    median_diff = np.median(diff)
    lower_ci, upper_ci = np.percentile(diff, [2.5, 97.5])

    details = {
        'p_value': p_value,
        'median_diff': median_diff,
        'ci_lower': lower_ci,
        'ci_upper': upper_ci,
        'n_draws': len(diff),
    }

    return star, details


def create_data_extraction_functions(target_period):
    """Create data extraction functions with the specified target period."""

    def sum_event_counts(_df, column_name):
        """Generic function to sum event counts from a column of dictionaries."""
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*target_period)]

        total = {}
        for d in _df[column_name]:
            for k, v in d.items():
                total[k] = total.get(k, 0) + v
        return pd.Series(sum(total.values()), name="total")

    def get_num_treatments_total(_df):
        return sum_event_counts(_df, "hsi_event_key_to_counts")

    def get_num_treatments_never_ran(_df):
        return sum_event_counts(_df, "never_ran_hsi_event_key_to_counts")

    def get_num_treatments_total_delayed(_df):
        """Count total number of delayed HSI events."""
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*target_period)]
        return pd.Series(len(_df), name="total")

    def get_num_treatments_total_cancelled(_df):
        """Count total number of cancelled HSI events."""
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*target_period)]
        return pd.Series(len(_df), name="total")

    def get_population_total(_df):
        """Returns the total population across the entire period."""
        _df["date"] = pd.to_datetime(_df["date"])
        filtered_df = _df.loc[_df["date"].between(*target_period)]
        numeric_df = filtered_df.drop(columns=["female", "male"], errors="ignore")
        population_mean = numeric_df.sum(numeric_only=True).mean()
        return pd.Series(population_mean, name="population")

    return {
        'get_num_treatments_total': get_num_treatments_total,
        'get_num_treatments_never_ran': get_num_treatments_never_ran,
        'get_num_treatments_total_delayed': get_num_treatments_total_delayed,
        'get_num_treatments_total_cancelled': get_num_treatments_total_cancelled,
        'get_population_total': get_population_total
    }


def extract_yearly_data(results_folder, target_year, extraction_functions, draw, scenario_name):
    """Extract data for a specific year and scenario."""
    target_period = (Date(target_year, 1, 1), Date(target_year, 12, 31))

    data = {}

    # Total treatments
    data['treatments'] = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='hsi_event_counts',
            custom_generate_series=extraction_functions['get_num_treatments_total'],
            do_scaling=True
        ),
        only_mean=False,
        collapse_columns=True,
    )[draw]

    # Never ran appointments
    data['never_ran'] = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='never_ran_hsi_event_counts',
            custom_generate_series=extraction_functions['get_num_treatments_never_ran'],
            do_scaling=True
        ),
        only_mean=False,
        collapse_columns=True,
    )[draw]

    # Weather-related disruptions
    if scenario_name == 'Baseline':
        data['weather_delayed'] = {
            'mean': pd.Series([0], name='mean'),
            'lower': pd.Series([0], name='lower'),
            'upper': pd.Series([0], name='upper')
        }
        data['weather_cancelled'] = {
            'mean': pd.Series([0], name='mean'),
            'lower': pd.Series([0], name='lower'),
            'upper': pd.Series([0], name='upper')
        }
    else:
        if MAIN_TEXT:
            data['weather_delayed'] = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="Weather_delayed_HSI_Event_full_info",
                    custom_generate_series=extraction_functions['get_num_treatments_total_delayed'],
                    do_scaling=True,
                ),
                only_mean=False,
                collapse_columns=True,
            )
            data['weather_cancelled'] = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="Weather_cancelled_HSI_Event_full_info",
                    custom_generate_series=extraction_functions['get_num_treatments_total_cancelled'],
                    do_scaling=True,
                ),
                only_mean=False,
                collapse_columns=True,
            )
        else:
            data['weather_delayed'] = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='Weather_delayed_HSI_Event_full_info',
                    custom_generate_series=extraction_functions['get_num_treatments_total_delayed'],
                    do_scaling=True
                ),
                only_mean=False,
                collapse_columns=True,
            )[draw]
            data['weather_cancelled'] = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='Weather_cancelled_HSI_Event_full_info',
                    custom_generate_series=extraction_functions['get_num_treatments_total_cancelled'],
                    do_scaling=True
                ),
                only_mean=False,
                collapse_columns=True,
            )[draw]

    # Population data
    data['population'] = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='population',
            custom_generate_series=extraction_functions['get_population_total'],
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True,
    )[draw]

    return data


def create_stacked_bar_plot(df_final, final_data_with_ci, scenario_labels, output_path, treatment_significance,
                            total_star):
    """Create stacked bar chart comparing scenarios."""
    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(scenario_labels))

    for treatment in df_final.index:
        values = df_final.loc[treatment]
        ax.bar(scenario_labels, values, bottom=bottom,
               color=get_color_short_treatment_id(treatment),
               label=treatment)
        bottom += values.values

    ax.set_ylabel("Total Number of HSIs", fontsize=12)
    ax.set_xlabel("Scenario", fontsize=12)

    # Update legend with significance
    handles, labels = ax.get_legend_handles_labels()
    labels = [l.replace("*", "") for l in labels]
    original_index = df_final.index
    clean_index = original_index.str.replace("*", "", regex=False)
    clean_to_original = dict(zip(clean_index, original_index))

    new_labels = []
    for label in labels:
        original = clean_to_original.get(label, label)
        star = treatment_significance.get(original, '')
        if star:
            new_labels.append(f"{label} $\\bf{{{star}}}$")
        else:
            new_labels.append(label)

    ax.legend(
        handles, new_labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        title='Treatment Type\n(* p<0.05, ** p<0.01, *** p<0.001)',
        fontsize=10
    )
    ax.tick_params(axis='both', labelsize=11)

    # Add overall significance
    if total_star:
        ax.text(
            0.5, 1.02,
            f'Overall difference: {total_star}',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='darkred',
            transform=ax.transAxes
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def create_scatter_plot(df_final, treatment_significance, total_star, scenario_labels, output_path):
    """Create scatter plot comparing scenarios."""
    fig, ax = plt.subplots(figsize=(7, 7))

    x = df_final[scenario_labels[0]]
    y = df_final[scenario_labels[1]]

    ax.scatter(x, y, alpha=0.8)

    # 1:1 reference line
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, linestyle='--', linewidth=1, color='gray')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Annotate points with significance
    for treatment in df_final.index:
        star = treatment_significance.get(treatment, '')
        label = f"{treatment} {star}" if star else treatment
        ax.text(
            df_final.loc[treatment, scenario_labels[0]],
            df_final.loc[treatment, scenario_labels[1]],
            label,
            fontsize=9, ha='left', va='bottom'
        )

    # Overall significance
    if total_star:
        ax.text(
            0.5, 1.02,
            f'Overall difference: {total_star}',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='darkred',
            transform=ax.transAxes
        )

    ax.set_xlabel(scenario_labels[0])
    ax.set_ylabel(scenario_labels[1])
    ax.set_title('Treatment comparison (scatter)')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing healthcare system utilization across scenarios."""

    target_period = (Date(MIN_YEAR, 1, 1), Date(MAX_YEAR, 12, 31))
    extraction_functions = create_data_extraction_functions(target_period)

    target_year_sequence = range(MIN_YEAR, MAX_YEAR, SPACING_OF_YEARS)

    # Storage for accumulated data
    all_draws_data = {
        'treatments': {'mean': [], 'lower': [], 'upper': []},
        'never_ran': {'mean': [], 'lower': [], 'upper': []},
        'weather_delayed': {'mean': [], 'lower': [], 'upper': []},
        'weather_cancelled': {'mean': [], 'lower': [], 'upper': []}
    }

    all_years_by_draw = {}

    # Extract data for each draw
    for draw in scenarios_of_interest:
        scenario_name = scenario_names[draw]

        yearly_data = {
            'treatments': {'mean': {}, 'lower': {}, 'upper': {}},
            'never_ran': {'mean': {}, 'lower': {}, 'upper': {}},
            'weather_delayed': {'mean': {}, 'lower': {}, 'upper': {}},
            'weather_cancelled': {'mean': {}, 'lower': {}, 'upper': {}},
            'population': {'mean': {}, 'lower': {}, 'upper': {}}
        }

        for target_year in target_year_sequence:
            year_data = extract_yearly_data(
                results_folder, target_year, extraction_functions, draw, scenario_name
            )

            for metric in ['treatments', 'never_ran', 'weather_delayed', 'weather_cancelled']:
                yearly_data[metric]['mean'][target_year] = year_data[metric]['mean']
                yearly_data[metric]['lower'][target_year] = year_data[metric]['lower']
                yearly_data[metric]['upper'][target_year] = year_data[metric]['upper']

            yearly_data['population']['mean'][target_year] = year_data['population']['mean']

        # Convert to DataFrames
        df_data = {}
        for metric in ['treatments', 'never_ran', 'weather_delayed', 'weather_cancelled']:
            df_data[metric] = {
                'mean': pd.DataFrame(yearly_data[metric]['mean']),
                'lower': pd.DataFrame(yearly_data[metric]['lower']),
                'upper': pd.DataFrame(yearly_data[metric]['upper'])
            }

        # Store draw data
        all_years_by_draw[draw] = {
            metric: {
                'mean': df_data[metric]['mean'].sum(),
                'lower': df_data[metric]['lower'].sum(),
                'upper': df_data[metric]['upper'].sum()
            }
            for metric in ['treatments', 'never_ran', 'weather_delayed', 'weather_cancelled']
        }
        all_years_by_draw[draw]['population'] = pd.DataFrame(yearly_data['population']['mean'])

        # Save CSVs
        for metric in ['treatments', 'never_ran', 'weather_delayed', 'weather_cancelled']:
            df_data[metric]['mean'].to_csv(
                output_folder / f"{metric}_by_type_{draw}.csv"
            )

        # Accumulate across draws
        for metric in ['treatments', 'never_ran', 'weather_delayed', 'weather_cancelled']:
            for stat in ['mean', 'lower', 'upper']:
                all_draws_data[metric][stat].append(
                    pd.Series(df_data[metric][stat].sum(), name=f'Draw {draw}')
                )

    # Combine all draws
    df_combined = {}
    for metric in ['treatments', 'never_ran', 'weather_delayed', 'weather_cancelled']:
        df_combined[metric] = {
            stat: pd.concat(all_draws_data[metric][stat], axis=1)
            for stat in ['mean', 'lower', 'upper']
        }

    # Calculate totals and confidence intervals
    totals = {}
    for metric in ['treatments', 'never_ran', 'weather_delayed', 'weather_cancelled']:
        totals[metric] = {
            'mean': df_combined[metric]['mean'].sum(),
            'lower': df_combined[metric]['lower'].sum(),
            'upper': df_combined[metric]['upper'].sum()
        }

    # Final analysis period and scenarios
    target_year_final = MAX_YEAR
    target_period_final = (Date(2025, 1, 1), Date(target_year_final, 12, 31))
    scenario_labels_final = ["Baseline", "SSP2-4.5"]
    scenario_indices_final = [0, 1]

    # Extract treatment-level data
    def get_counts_of_hsi_by_treatment_id(_df):
        _df = _df.loc[pd.to_datetime(_df['date']).between(*target_period_final)]
        _counts_by_treatment_id = _df['TREATMENT_ID'].apply(pd.Series).sum().astype(int)
        return _counts_by_treatment_id.groupby(level=0).sum()

    def get_counts_of_hsi_by_short_treatment_id(_df):
        _counts_by_treatment_id = get_counts_of_hsi_by_treatment_id(_df)
        _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
        return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()

    final_data = {}
    final_data_with_ci = {}

    for i, draw in enumerate(scenario_indices_final):
        result_data_full = summarize(
            extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_counts_of_hsi_by_short_treatment_id,
                do_scaling=True
            ),
            only_mean=False,
            collapse_columns=True,
        )[draw]

        final_data[scenario_labels_final[i]] = result_data_full['mean']
        final_data_with_ci[scenario_labels_final[i]] = {
            'mean': result_data_full['mean'],
            'lower': result_data_full['lower'],
            'upper': result_data_full['upper']
        }

    df_final = pd.DataFrame(final_data).fillna(0)
    df_final.to_csv(output_folder / f"{PREFIX_ON_FILENAME}_Final_Treatments_{suffix}.csv")

    # Calculate significance for each treatment type
    treatment_significance = {}
    baseline_draw_index = df_combined['treatments']['mean'].columns[0]
    climate_draw_index = df_combined['treatments']['mean'].columns[1]

    for treatment in df_final.index:
        if treatment not in df_combined['treatments']['mean'].index:
            treatment_significance[treatment] = ''
            continue

        baseline_draws = df_combined['treatments']['mean'].loc[treatment, baseline_draw_index]
        climate_draws = df_combined['treatments']['mean'].loc[treatment, climate_draw_index]
        treatment_significance[treatment] = paired_draw_significance(baseline_draws, climate_draws)

    # Calculate overall significance
    baseline_total_draws = df_combined['treatments']['mean'].iloc[:, 0].sum()
    climate_total_draws = df_combined['treatments']['mean'].iloc[:, 1].sum()
    total_star = paired_draw_significance(baseline_total_draws, climate_total_draws)

    # Create plots
    create_stacked_bar_plot(
        df_final, final_data_with_ci, scenario_labels_final,
        output_folder / f"{PREFIX_ON_FILENAME}_Final_Treatments_StackedBar_all_years_{suffix}_with_significance.png",
        treatment_significance, total_star
    )

    fig_scatter, ax_scatter = plt.subplots(figsize=(14, 8))
    x_positions = np.arange(len(scenario_labels_final))
    jitter_strength = 0.15

    treatments = df_final.index

    for treatment in treatments:
        y_means = []
        y_lower = []
        y_upper = []

        for scenario in scenario_labels_final:
            y_means.append(final_data_with_ci[scenario]['mean'].get(treatment, 0))
            y_lower.append(final_data_with_ci[scenario]['lower'].get(treatment, 0))
            y_upper.append(final_data_with_ci[scenario]['upper'].get(treatment, 0))

        y_means = np.array(y_means)
        y_lower = np.array(y_lower)
        y_upper = np.array(y_upper)

        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(x_positions))
        x_jittered = x_positions + jitter

        ax_scatter.scatter(
            x_jittered, y_means,
            s=80, alpha=0.8,
            label=treatment.replace("*", "")
        )

        ax_scatter.errorbar(
            x_jittered, y_means,
            yerr=[y_means - y_lower, y_upper - y_means],
            fmt="none", capsize=4, alpha=0.6, linewidth=1.5
        )

        ax_scatter.plot(x_jittered, y_means, linestyle="-", alpha=0.4, linewidth=1)

    ax_scatter.set_yscale('log')
    ax_scatter.set_title(
        f"HSI Events by Treatment Type ({MIN_YEAR}-{target_year_final})",
        fontsize=14, fontweight='bold'
    )
    ax_scatter.set_xticks(x_positions)
    ax_scatter.set_xticklabels(scenario_labels_final, fontsize=12)
    ax_scatter.set_xlabel("Scenario", fontsize=12)
    ax_scatter.set_ylabel("Number of HSI Events", fontsize=12)
    ax_scatter.tick_params(axis='both', labelsize=11)
    ax_scatter.legend(
        title="Treatment Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10
    )
    ax_scatter.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig_scatter.savefig(
        output_folder / f"{PREFIX_ON_FILENAME}_Treatments_Scatter_{suffix}.png",
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig_scatter)

    # Fine-grained treatment analysis - loop over multiple HSI types
    HSI_events_of_interest = ["HSI_Event",
        "HSI_Event_non_blank_appt_footprint",
        "Weather_delayed_HSI_Event_full_info",
        "Weather_cancelled_HSI_Event_full_info"
    ]

    ci_data_to_save = {}

    for HSI_of_interest in HSI_events_of_interest:

        def get_num_treatments_group(_df):
            """Return the number of treatments by short treatment id."""
            _df = _df.loc[pd.to_datetime(_df.date).between(*target_period_final)]

            if 'TREATMENT_ID' not in _df.columns or _df.empty:
                return pd.Series(dtype=int)

            sample_value = _df['TREATMENT_ID'].iloc[0] if len(_df) > 0 else None
            if sample_value is None:
                return pd.Series(dtype=int)

            if isinstance(sample_value, dict):
                _df = _df['TREATMENT_ID'].apply(pd.Series).sum().astype(int)
            else:
                _df = _df['TREATMENT_ID'].value_counts()

            if len(_df) == 0:
                return pd.Series(dtype=int)

            if isinstance(_df.index[0], str):
                _df.index = _df.index.map(lambda x: "_".join(x.split('_')[:2]) + "*")
                _df = _df.groupby(level=0).sum()

            return _df

        final_data_fine = {}
        final_data_fine_with_ci = {}

        for i, draw in enumerate(scenario_indices_final):
            scenario_name = scenario_names[draw]

            # Skip weather-related events for baseline scenario
            if scenario_name == 'Baseline' and HSI_of_interest in ["Weather_delayed_HSI_Event_full_info",
                                                                   "Weather_cancelled_HSI_Event_full_info"]:
                final_data_fine[scenario_labels_final[i]] = pd.Series(dtype=float)
                final_data_fine_with_ci[scenario_labels_final[i]] = {
                    'mean': pd.Series(dtype=float),
                    'lower': pd.Series(dtype=float),
                    'upper': pd.Series(dtype=float)
                }
                continue

            if MAIN_TEXT and HSI_of_interest in ["Weather_delayed_HSI_Event_full_info",
                                                 "Weather_cancelled_HSI_Event_full_info"]:
                if draw == 1:  # Climate scenario
                    result_data_full = summarize(
                        extract_results(
                            results_folder,
                            module="tlo.methods.healthsystem.summary",
                            key=HSI_of_interest,
                            custom_generate_series=get_num_treatments_group,
                            do_scaling=True,
                        ),
                        only_mean=False,
                        collapse_columns=True,
                    )

                    final_data_fine[scenario_labels_final[i]] = result_data_full['mean']
                    final_data_fine_with_ci[scenario_labels_final[i]] = {
                        'mean': result_data_full['mean'],
                        'lower': result_data_full['lower'],
                        'upper': result_data_full['upper']
                    }
                else:  # Baseline
                    final_data_fine[scenario_labels_final[i]] = pd.Series(dtype=float)
                    final_data_fine_with_ci[scenario_labels_final[i]] = {
                        'mean': pd.Series(dtype=float),
                        'lower': pd.Series(dtype=float),
                        'upper': pd.Series(dtype=float)
                    }
            else:
                result_data_full = summarize(
                    extract_results(
                        results_folder,
                        module="tlo.methods.healthsystem.summary",
                        key=HSI_of_interest,
                        custom_generate_series=get_num_treatments_group,
                        do_scaling=True,
                    ),
                    only_mean=False,
                    collapse_columns=True,
                )[draw]

                final_data_fine[scenario_labels_final[i]] = result_data_full['mean']
                final_data_fine_with_ci[scenario_labels_final[i]] = {
                    'mean': result_data_full['mean'],
                    'lower': result_data_full['lower'],
                    'upper': result_data_full['upper']
                }

        # Store CI data for this HSI type
        ci_data_to_save[HSI_of_interest] = final_data_fine_with_ci

        df_final_fine = pd.DataFrame(final_data_fine).fillna(0)

        # Save mean values (as before)
        if not df_final_fine.empty and df_final_fine.sum().sum() > 0:
            df_final_fine.to_csv(
                output_folder / f"{PREFIX_ON_FILENAME}_Final_Coarse_Treatments_{suffix}_{HSI_of_interest}.csv"
            )

            # NEW: Save confidence intervals
            for scenario in scenario_labels_final:
                if scenario in final_data_fine_with_ci:
                    ci_df = pd.DataFrame({
                        'HSI_name': final_data_fine_with_ci[scenario]['mean'].index,
                        'mean': final_data_fine_with_ci[scenario]['mean'].values,
                        'lower': final_data_fine_with_ci[scenario]['lower'].values,
                        'upper': final_data_fine_with_ci[scenario]['upper'].values
                    })
                    ci_df.to_csv(
                        output_folder / f"{PREFIX_ON_FILENAME}_CI_{HSI_of_interest}_{scenario.replace('-', '')}.csv",
                        index=False
                    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
