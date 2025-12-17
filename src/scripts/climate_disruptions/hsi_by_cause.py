import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib.dates as mdates

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    summarize,
    get_color_short_treatment_id
)

min_year = 2025
max_year = 2041
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'
climate_sensitivity_analysis = True
parameter_sensitivity_analysis = False
main_text = True
scenario_names_all = ["Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low",
                      "SSP 2.45 Mean", "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]

if climate_sensitivity_analysis:
    scenario_names = ["Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low",
                      "SSP 2.45 Mean", "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]
    suffix = "climate_SA"
    scenarios_of_interest = range(len(scenario_names))
if parameter_sensitivity_analysis:
    scenario_names_all = range(0, 10, 1)
    scenario_names = scenario_names_all
    suffix = "parameter_SA"

if main_text:
    scenario_names = ["Baseline", "SSP 2.45 Mean", ]
    suffix = "main_text"
    scenarios_of_interest = [0, 1]

precipitation_files = {
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

scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167'] * 4


def add_significance_stars(ax, x_positions, baseline_data, climate_data, y_offset_factor=0.05):
    """
    Add significance stars to a bar plot where baseline and climate scenarios differ significantly.

    Parameters:
    -----------
    ax : matplotlib axis
    x_positions : array-like
        x-positions of the bars
    baseline_data : dict
        Dictionary with 'mean', 'lower', 'upper' for baseline scenario
    climate_data : dict
        Dictionary with 'mean', 'lower', 'upper' for climate scenario
    y_offset_factor : float
        Factor to offset stars above the bars (relative to y-range)
    """
    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_offset = y_range * y_offset_factor

    for i, x_pos in enumerate(x_positions):
        # Extract values for this year
        baseline_mean = baseline_data['mean'].iloc[i]
        baseline_lower = baseline_data['lower'].iloc[i]
        baseline_upper = baseline_data['upper'].iloc[i]

        climate_mean = climate_data['mean'].iloc[i]
        climate_lower = climate_data['lower'].iloc[i]
        climate_upper = climate_data['upper'].iloc[i]

        # Check if confidence intervals overlap
        # Non-overlapping CIs suggest significant difference
        intervals_overlap = not (baseline_upper < climate_lower or climate_upper < baseline_lower)

        # Calculate approximate z-score (assuming normal distribution)
        # SE ≈ (upper - lower) / (2 * 1.96) for 95% CI
        baseline_se = (baseline_upper - baseline_lower) / (2 * 1.96)
        climate_se = (climate_upper - climate_lower) / (2 * 1.96)

        # Calculate difference and pooled SE
        diff = abs(climate_mean - baseline_mean)
        pooled_se = np.sqrt(baseline_se ** 2 + climate_se ** 2)

        if pooled_se > 0:
            z_score = diff / pooled_se
            p_value = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed test
        else:
            p_value = 1.0

        # Determine significance level and star
        if p_value < 0.001:
            star = '***'
        elif p_value < 0.01:
            star = '**'
        elif p_value < 0.05:
            star = '*'
        else:
            star = ''

        # Add star if significant
        if star:
            y_pos = max(baseline_mean, climate_mean) + y_offset
            ax.text(x_pos, y_pos, star,
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    color='red')


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the healthcare system utilization across scenarios.
    - We estimate the healthcare system impact through total treatments and never-ran appointments.
    - Now includes weather-delayed and weather-cancelled appointments.
    """

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions
    def sum_event_counts(_df, column_name):
        """Generic function to sum event counts from a column of dictionaries"""
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]

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
        """Count total number of delayed HSI events from full info logger"""
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]

        # Each row is one delayed event
        return pd.Series(len(_df), name="total")

    def get_num_treatments_total_cancelled(_df):
        """Count total number of cancelled HSI events from full info logger"""
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]

        # Each row is one cancelled event
        return pd.Series(len(_df), name="total")

    def get_population_total(_df):
        """Returns the total population across the entire period"""
        _df["date"] = pd.to_datetime(_df["date"])
        filtered_df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=["female", "male"], errors="ignore")
        # Get the mean population across years
        population_mean = numeric_df.sum(numeric_only=True).mean()
        return pd.Series(population_mean, name="population")

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    all_draws_treatments_mean = []
    all_draws_treatments_lower = []
    all_draws_treatments_upper = []

    all_draws_never_ran_mean = []
    all_draws_never_ran_lower = []
    all_draws_never_ran_upper = []

    all_draws_weather_delayed_mean = []
    all_draws_weather_delayed_lower = []
    all_draws_weather_delayed_upper = []

    all_draws_weather_cancelled_mean = []
    all_draws_weather_cancelled_lower = []
    all_draws_weather_cancelled_upper = []

    all_draws_treatments_mean_1000 = []
    all_draws_treatments_lower_1000 = []
    all_draws_treatments_upper_1000 = []

    all_draws_never_ran_mean_1000 = []
    all_draws_never_ran_lower_1000 = []
    all_draws_never_ran_upper_1000 = []

    all_draws_weather_delayed_mean_1000 = []
    all_draws_weather_cancelled_mean_1000 = []

    # Store raw data for significance testing
    all_years_by_draw = {}

    for draw in range(len(scenario_names_all)):
        if draw not in scenarios_of_interest:
            continue
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"  # noqa: E731

        all_years_data_treatments_mean = {}
        all_years_data_treatments_upper = {}
        all_years_data_treatments_lower = {}

        all_years_data_never_ran_mean = {}
        all_years_data_never_ran_upper = {}
        all_years_data_never_ran_lower = {}

        all_years_data_weather_delayed_mean = {}
        all_years_data_weather_delayed_upper = {}
        all_years_data_weather_delayed_lower = {}

        all_years_data_weather_cancelled_mean = {}
        all_years_data_weather_cancelled_upper = {}
        all_years_data_weather_cancelled_lower = {}

        all_years_data_population_mean = {}
        all_years_data_population_lower = {}
        all_years_data_population_upper = {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (
                Date(target_year, 1, 1), Date(target_year, 12, 31))

            # Total treatments
            num_treatments_total = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='hsi_event_counts',
                custom_generate_series=get_num_treatments_total,
                do_scaling=True
            ),
                only_mean=False,
                collapse_columns=True,
            )[draw]

            all_years_data_treatments_mean[target_year] = num_treatments_total['mean']
            all_years_data_treatments_lower[target_year] = num_treatments_total['lower']
            all_years_data_treatments_upper[target_year] = num_treatments_total['upper']

            result_data_population = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_total,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            all_years_data_population_mean[target_year] = result_data_population['mean']
            all_years_data_population_lower[target_year] = result_data_population['lower']
            all_years_data_population_upper[target_year] = result_data_population['upper']

            # Never ran appointments

            num_never_ran_appts = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='never_ran_hsi_event_counts',
                    custom_generate_series=get_num_treatments_never_ran,
                    do_scaling=True
                ),
                only_mean=False,
                collapse_columns=True,
            )[draw]
            all_years_data_never_ran_mean[target_year] = num_never_ran_appts['mean']
            all_years_data_never_ran_lower[target_year] = num_never_ran_appts['lower']
            all_years_data_never_ran_upper[target_year] = num_never_ran_appts['upper']

            if scenario_names[draw] == 'Baseline':
                all_years_data_weather_delayed_mean[target_year] = pd.Series([0], name='mean')
                all_years_data_weather_delayed_lower[target_year] = pd.Series([0], name='lower')
                all_years_data_weather_delayed_upper[target_year] = pd.Series([0], name='upper')

                all_years_data_weather_cancelled_mean[target_year] = pd.Series([0], name='mean')
                all_years_data_weather_cancelled_lower[target_year] = pd.Series([0], name='lower')
                all_years_data_weather_cancelled_upper[target_year] = pd.Series([0], name='upper')
            elif main_text:
                num_weather_delayed_appointments = summarize(extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="Weather_delayed_HSI_Event_full_info",
                    custom_generate_series=get_num_treatments_total_delayed,
                    do_scaling=True,
                ),
                    only_mean=False,
                    collapse_columns=True,
                )

                all_years_data_weather_delayed_mean[target_year] = num_weather_delayed_appointments['mean']
                all_years_data_weather_delayed_lower[target_year] = num_weather_delayed_appointments['lower']
                all_years_data_weather_delayed_upper[target_year] = num_weather_delayed_appointments['upper']

                num_weather_cancelled_appointments = summarize(
                    extract_results(
                        results_folder,
                        module="tlo.methods.healthsystem.summary",
                        key="Weather_cancelled_HSI_Event_full_info",
                        custom_generate_series=get_num_treatments_total_cancelled,
                        do_scaling=True,
                    ),
                    only_mean=False,
                    collapse_columns=True,
                )
                all_years_data_weather_cancelled_mean[target_year] = num_weather_cancelled_appointments['mean']
                all_years_data_weather_cancelled_lower[target_year] = num_weather_cancelled_appointments['lower']
                all_years_data_weather_cancelled_upper[target_year] = num_weather_cancelled_appointments['upper']


            else:

                # Weather delayed appointments

                num_weather_delayed_appointments = summarize(
                    extract_results(
                        results_folder,
                        module='tlo.methods.healthsystem.summary',
                        key='Weather_delayed_HSI_Event_full_info',
                        custom_generate_series=get_num_treatments_total_delayed,
                        do_scaling=True
                    ),
                    only_mean=False,
                    collapse_columns=True,
                )[draw]

                all_years_data_weather_delayed_mean[target_year] = num_weather_delayed_appointments['mean']
                all_years_data_weather_delayed_lower[target_year] = num_weather_delayed_appointments['lower']
                all_years_data_weather_delayed_upper[target_year] = num_weather_delayed_appointments['upper']

                # Weather cancelled appointments
                num_weather_cancelled_appointments = summarize(
                    extract_results(
                        results_folder,
                        module='tlo.methods.healthsystem.summary',
                        key='Weather_cancelled_HSI_Event_full_info',
                        custom_generate_series=get_num_treatments_total_cancelled,
                        do_scaling=True
                    ),
                    only_mean=False,
                    collapse_columns=True,
                )[draw]

                all_years_data_weather_cancelled_mean[target_year] = num_weather_cancelled_appointments['mean']
                all_years_data_weather_cancelled_lower[target_year] = num_weather_cancelled_appointments['lower']
                all_years_data_weather_cancelled_upper[target_year] = num_weather_cancelled_appointments['upper']

            # Population data for normalization
            result_data_population = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_total,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            all_years_data_population_mean[target_year] = result_data_population['mean']
            all_years_data_population_lower[target_year] = result_data_population['lower']
            all_years_data_population_upper[target_year] = result_data_population['upper']

        # Convert the accumulated data into DataFrames for plotting
        df_all_years_treatments_mean = pd.DataFrame(all_years_data_treatments_mean)
        print(df_all_years_treatments_mean)
        df_all_years_treatments_lower = pd.DataFrame(all_years_data_treatments_lower)
        df_all_years_treatments_upper = pd.DataFrame(all_years_data_treatments_upper)

        df_all_years_never_ran_mean = pd.DataFrame(all_years_data_never_ran_mean)
        df_all_years_never_ran_lower = pd.DataFrame(all_years_data_never_ran_lower)
        df_all_years_never_ran_upper = pd.DataFrame(all_years_data_never_ran_upper)

        df_all_years_weather_delayed_mean = pd.DataFrame(all_years_data_weather_delayed_mean)
        df_all_years_weather_delayed_lower = pd.DataFrame(all_years_data_weather_delayed_lower)
        df_all_years_weather_delayed_upper = pd.DataFrame(all_years_data_weather_delayed_upper)

        df_all_years_weather_cancelled_mean = pd.DataFrame(all_years_data_weather_cancelled_mean)
        df_all_years_weather_cancelled_lower = pd.DataFrame(all_years_data_weather_cancelled_lower)
        df_all_years_weather_cancelled_upper = pd.DataFrame(all_years_data_weather_cancelled_upper)

        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)

        # Store data for this draw
        all_years_by_draw[draw] = {
            'treatments': {
                'mean': df_all_years_treatments_mean.sum(),
                'lower': df_all_years_treatments_lower.sum(),
                'upper': df_all_years_treatments_upper.sum()
            },
            'never_ran': {
                'mean': df_all_years_never_ran_mean.sum(),
                'lower': df_all_years_never_ran_lower.sum(),
                'upper': df_all_years_never_ran_upper.sum()
            },
            'weather_delayed': {
                'mean': df_all_years_weather_delayed_mean.sum(),
                'lower': df_all_years_weather_delayed_lower.sum(),
                'upper': df_all_years_weather_delayed_upper.sum()
            },
            'weather_cancelled': {
                'mean': df_all_years_weather_cancelled_mean.sum(),
                'lower': df_all_years_weather_cancelled_lower.sum(),
                'upper': df_all_years_weather_cancelled_upper.sum()
            },
            'population': df_all_years_data_population_mean
        }

        # PER 1000 POPULATION
        fig, axes = plt.subplots(2, 2, figsize=(25, 20))

        # Calculate per 1000 rates
        df_treatments_per_1000_mean = df_all_years_treatments_mean / df_all_years_data_population_mean.iloc[0, 0] * 1000
        df_never_ran_per_1000_mean = df_all_years_never_ran_mean / df_all_years_data_population_mean.iloc[0, 0] * 1000
        df_weather_delayed_per_1000_mean = df_all_years_weather_delayed_mean / df_all_years_data_population_mean.iloc[
            0, 0] * 1000
        df_weather_cancelled_per_1000_mean = df_all_years_weather_cancelled_mean / \
                                             df_all_years_data_population_mean.iloc[0, 0] * 1000

        # Save data to CSV
        df_all_years_treatments_mean.to_csv(output_folder / f"treatments_by_type_{draw}.csv")
        df_all_years_never_ran_mean.to_csv(output_folder / f"never_ran_by_type_{draw}.csv")
        df_all_years_weather_delayed_mean.to_csv(output_folder / f"weather_delayed_by_type_{draw}.csv")
        df_all_years_weather_cancelled_mean.to_csv(output_folder / f"weather_cancelled_by_type_{draw}.csv")

        # Accumulate data across all draws
        all_draws_treatments_mean.append(pd.Series(df_all_years_treatments_mean.sum(), name=f'Draw {draw}'))
        all_draws_never_ran_mean.append(pd.Series(df_all_years_never_ran_mean.sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_mean.append(pd.Series(df_all_years_weather_delayed_mean.sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_mean.append(
            pd.Series(df_all_years_weather_cancelled_mean.sum(), name=f'Draw {draw}'))

        all_draws_treatments_lower.append(pd.Series(df_all_years_treatments_lower.sum(), name=f'Draw {draw}'))
        all_draws_never_ran_lower.append(pd.Series(df_all_years_never_ran_lower.sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_lower.append(pd.Series(df_all_years_weather_delayed_lower.sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_lower.append(
            pd.Series(df_all_years_weather_cancelled_lower.sum(), name=f'Draw {draw}'))

        all_draws_treatments_upper.append(pd.Series(df_all_years_treatments_upper.sum(), name=f'Draw {draw}'))
        all_draws_never_ran_upper.append(pd.Series(df_all_years_never_ran_upper.sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_upper.append(pd.Series(df_all_years_weather_delayed_upper.sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_upper.append(
            pd.Series(df_all_years_weather_cancelled_upper.sum(), name=f'Draw {draw}'))

        # Per 1000 for final year only
        all_draws_treatments_mean_1000.append(pd.Series(df_treatments_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_never_ran_mean_1000.append(pd.Series(df_never_ran_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_weather_delayed_mean_1000.append(
            pd.Series(df_weather_delayed_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_weather_cancelled_mean_1000.append(
            pd.Series(df_weather_cancelled_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))

        if draw == 0:
            baseline_treatments_by_year = df_all_years_treatments_mean.copy()
            baseline_never_ran_by_year = df_all_years_never_ran_mean.copy()
            baseline_weather_delayed_by_year = df_all_years_weather_delayed_mean.copy()
            baseline_weather_cancelled_by_year = df_all_years_weather_cancelled_mean.copy()
            baseline_population = df_all_years_data_population_mean.copy()

    # Combine all draws
    df_treatments_all_draws_mean = pd.concat(all_draws_treatments_mean, axis=1)
    df_never_ran_all_draws_mean = pd.concat(all_draws_never_ran_mean, axis=1)
    df_weather_delayed_all_draws_mean = pd.concat(all_draws_weather_delayed_mean, axis=1)
    df_weather_cancelled_all_draws_mean = pd.concat(all_draws_weather_cancelled_mean, axis=1)

    df_treatments_all_draws_lower = pd.concat(all_draws_treatments_lower, axis=1)
    df_never_ran_all_draws_lower = pd.concat(all_draws_never_ran_lower, axis=1)
    df_weather_delayed_all_draws_lower = pd.concat(all_draws_weather_delayed_lower, axis=1)
    df_weather_cancelled_all_draws_lower = pd.concat(all_draws_weather_cancelled_lower, axis=1)

    df_treatments_all_draws_upper = pd.concat(all_draws_treatments_upper, axis=1)
    df_never_ran_all_draws_upper = pd.concat(all_draws_never_ran_upper, axis=1)
    df_weather_delayed_all_draws_upper = pd.concat(all_draws_weather_delayed_upper, axis=1)
    df_weather_cancelled_all_draws_upper = pd.concat(all_draws_weather_cancelled_upper, axis=1)

    df_treatments_all_draws_mean_1000 = pd.concat(all_draws_treatments_mean_1000, axis=1)
    df_never_ran_all_draws_mean_1000 = pd.concat(all_draws_never_ran_mean_1000, axis=1)
    df_weather_delayed_all_draws_mean_1000 = pd.concat(all_draws_weather_delayed_mean_1000, axis=1)
    df_weather_cancelled_all_draws_mean_1000 = pd.concat(all_draws_weather_cancelled_mean_1000, axis=1)

    # Final summary plots across all scenarios
    treatments_totals_mean = df_treatments_all_draws_mean.sum()
    never_ran_totals_mean = df_never_ran_all_draws_mean.sum()
    weather_delayed_totals_mean = df_weather_delayed_all_draws_mean.sum()
    weather_cancelled_totals_mean = df_weather_cancelled_all_draws_mean.sum()

    treatments_totals_lower = df_treatments_all_draws_lower.sum()
    treatments_totals_upper = df_treatments_all_draws_upper.sum()
    never_ran_totals_lower = df_never_ran_all_draws_lower.sum()
    never_ran_totals_upper = df_never_ran_all_draws_upper.sum()
    weather_delayed_totals_lower = df_weather_delayed_all_draws_lower.sum()
    weather_delayed_totals_upper = df_weather_delayed_all_draws_upper.sum()
    weather_cancelled_totals_lower = df_weather_cancelled_all_draws_lower.sum()
    weather_cancelled_totals_upper = df_weather_cancelled_all_draws_upper.sum()

    treatments_totals_err = np.array([
        treatments_totals_mean - treatments_totals_lower,
        treatments_totals_upper - treatments_totals_mean
    ])

    never_ran_totals_err = np.array([
        never_ran_totals_mean - never_ran_totals_lower,
        never_ran_totals_upper - never_ran_totals_mean
    ])

    weather_delayed_totals_err = np.array([
        weather_delayed_totals_mean - weather_delayed_totals_lower,
        weather_delayed_totals_upper - weather_delayed_totals_mean
    ])

    weather_cancelled_totals_err = np.array([
        weather_cancelled_totals_mean - weather_cancelled_totals_lower,
        weather_cancelled_totals_upper - weather_cancelled_totals_mean
    ])


    target_year_final = max_year
    target_period_final = (Date(2025, 1, 1), Date(target_year_final, 12, 31))
    scenario_labels_final = ["Baseline", "SSP2-4.5"]
    scenario_indices_final = [0, 1]

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

    # --- Get full data with confidence intervals for HSI by treatment type

    # --- Plot: stacked bar chart (Baseline vs SSP2.45) with significance stars
    fig_final, ax_final = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(2)

    # Store positions for significance testing
    treatment_positions = {}

    for treatment in df_final.index:
        print(treatment)
        print(get_color_short_treatment_id(treatment))
        values = df_final.loc[treatment]
        ax_final.bar(scenario_labels_final, values, bottom=bottom,
                     color=get_color_short_treatment_id(treatment),
                     label=treatment)

        # Store the center position of each treatment segment for significance stars
        treatment_positions[treatment] = bottom.copy()
        bottom += values.values

    ax_final.set_ylabel("Total Number of HSIs", fontsize=12)
    ax_final.set_xlabel("Scenario", fontsize=12)

    handles, labels = ax_final.get_legend_handles_labels()
    labels = [l.replace("*", "") for l in labels]

    ax_final.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Treatment Type",
    )
    ax_final.tick_params(axis='both', labelsize=11)

    # Calculate significance for each treatment type and store in dictionary
    treatment_significance = {}
    for treatment in df_final.index:
        # Get data for this treatment type from both scenarios
        baseline_mean = final_data_with_ci['Baseline']['mean'].get(treatment, 0)
        baseline_lower = final_data_with_ci['Baseline']['lower'].get(treatment, 0)
        baseline_upper = final_data_with_ci['Baseline']['upper'].get(treatment, 0)

        climate_mean = final_data_with_ci['SSP2-4.5']['mean'].get(treatment, 0)
        climate_lower = final_data_with_ci['SSP2-4.5']['lower'].get(treatment, 0)
        climate_upper = final_data_with_ci['SSP2-4.5']['upper'].get(treatment, 0)

        # Calculate standard errors (assuming 95% CI)
        baseline_se = (baseline_upper - baseline_lower) / (2 * 1.96) if baseline_upper > baseline_lower else 0
        climate_se = (climate_upper - climate_lower) / (2 * 1.96) if climate_upper > climate_lower else 0

        # Calculate difference and pooled SE
        diff = abs(climate_mean - baseline_mean)
        pooled_se = np.sqrt(baseline_se ** 2 + climate_se ** 2)

        if pooled_se > 0 and diff > 0:
            z_score = diff / pooled_se
            p_value = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed test

            # Determine significance level and star
            if p_value < 0.001:
                star = '***'
            elif p_value < 0.01:
                star = '**'
            elif p_value < 0.05:
                star = '*'
            else:
                star = ''

            treatment_significance[treatment] = star
        else:
            treatment_significance[treatment] = ''

    # Update legend with significance indicators
    handles, labels = ax_final.get_legend_handles_labels()
    labels = [l.replace("*", "") for l in labels]

    handles, labels = ax_final.get_legend_handles_labels()
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

    ax_final.legend(handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left',
                    title='Treatment Type\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=10)

    # Add overall total significance as text annotation
    total_baseline_mean = df_final['Baseline'].sum()
    total_baseline_lower = sum([final_data_with_ci['Baseline']['lower'].get(t, 0) for t in df_final.index])
    total_baseline_upper = sum([final_data_with_ci['Baseline']['upper'].get(t, 0) for t in df_final.index])

    total_climate_mean = df_final['SSP2-4.5'].sum()
    total_climate_lower = sum([final_data_with_ci['SSP2-4.5']['lower'].get(t, 0) for t in df_final.index])
    total_climate_upper = sum([final_data_with_ci['SSP2-4.5']['upper'].get(t, 0) for t in df_final.index])

    total_baseline_se = (total_baseline_upper - total_baseline_lower) / (2 * 1.96)
    total_climate_se = (total_climate_upper - total_climate_lower) / (2 * 1.96)

    total_diff = abs(total_climate_mean - total_baseline_mean)
    total_pooled_se = np.sqrt(total_baseline_se ** 2 + total_climate_se ** 2)

    if total_pooled_se > 0:
        total_z_score = total_diff / total_pooled_se
        total_p_value = 2 * (1 - stats.norm.cdf(total_z_score))

        if total_p_value < 0.001:
            total_star = '***'
        elif total_p_value < 0.01:
            total_star = '**'
        elif total_p_value < 0.05:
            total_star = '*'
        else:
            total_star = ''

        if total_star:
            # Add text annotation showing overall significance
            ax_final.text(0.5, 1.02, f'Overall difference: {total_star}',
                          ha='center', va='bottom', fontsize=11, fontweight='bold',
                          color='darkred', transform=ax_final.transAxes)

    fig_final.tight_layout()
    fig_final.savefig(
        output_folder / f"{PREFIX_ON_FILENAME}_Final_Treatments_StackedBar_all_years_{suffix}_with_significance.png",
        dpi=300)
    plt.close(fig_final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
