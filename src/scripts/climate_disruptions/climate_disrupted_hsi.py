
import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    summarize,
    get_color_short_treatment_id
)

min_year = 2026
max_year = 2041
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'
climate_sensitivity_analysis = False
parameter_sensitivity_analysis = False
main_text = True
scenario_names_all = ["Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low",
                  "SSP 2.45 Mean", "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]

if climate_sensitivity_analysis:
    scenario_names = ["Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",  "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]
    suffix = "climate_SA"
    scenarios_of_interest = range(len(scenario_names))
if parameter_sensitivity_analysis:
    scenario_names_all = range(0,10,1)
    scenario_names = scenario_names_all
    suffix = "parameter_SA"

if main_text:
    scenario_names = ["Baseline", "SSP 2.45 Mean", ]
    suffix = "main_text"
    scenarios_of_interest = [0,6]


precipitation_files  = {
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


scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167' ] *4

def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the healthcare system utilization across scenarios.
    - We estimate the healthcare system impact through total treatments and never-ran appointments.
    - Now includes weather-delayed and weather-cancelled appointments.
    """

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions
    def get_num_treatments_total(_df):
        _df['date'] = pd.to_datetime(_df['date'])

        # filter to target period
        _df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        total = {}

        for d in _df['hsi_event_key_to_counts']:
            for k, v in d.items():
                total[k] = 0
                total[k] += total.get(k, 0) + v
        return pd.Series(sum(total.values()), name="total_treatments")

    def get_num_treatments_never_ran(_df):
        _df['date'] = pd.to_datetime(_df['date'])

        # filter to target period
        _df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        total = {}

        for d in _df['never_ran_hsi_event_key_to_counts']:
            for k, v in d.items():
                total[k] = 0
                total[k] += total.get(k, 0) + v
        return pd.Series(sum(total.values()), name="total_treatments")


    def get_num_treatments_total_delayed(_df):
        _df['date'] = pd.to_datetime(_df['date'])

        # filter to target period
        _df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        total = {}

        for d in _df['weather_delayed_hsi_event_key_to_counts']:
            for k, v in d.items():
                total[k] = 0
                total[k] += total.get(k, 0) + v
        return pd.Series(sum(total.values()), name="total_treatments")

    def get_num_treatments_total_cancelled(_df):
        _df['date'] = pd.to_datetime(_df['date'])

        # filter to target period
        _df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        total = {}

        for d in _df['weather_cancelled_hsi_event_key_to_counts']:
            for k, v in d.items():
                total[k] = 0
                total[k] += total.get(k, 0) + v
        return pd.Series(sum(total.values()), name="total_treatments")


    def get_population_for_year(_df):
        """Returns the population in the year of interest"""
        _df['date'] = pd.to_datetime(_df['date'])
        filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=['female', 'male'], errors='ignore')
        population_sum = numeric_df.sum(numeric_only=True)
        return population_sum

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
                custom_generate_series=get_population_for_year,
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
            print(num_never_ran_appts)
            all_years_data_never_ran_mean[target_year] = num_never_ran_appts['mean']
            all_years_data_never_ran_lower[target_year] = num_never_ran_appts['lower']
            all_years_data_never_ran_upper[target_year] = num_never_ran_appts['upper']

            if draw == 0:
                all_years_data_weather_delayed_mean[target_year] = pd.Series([0], name='mean')
                all_years_data_weather_delayed_lower[target_year] = pd.Series([0], name='lower')
                all_years_data_weather_delayed_upper[target_year] = pd.Series([0], name='upper')

                all_years_data_weather_cancelled_mean[target_year] = pd.Series([0], name='mean')
                all_years_data_weather_cancelled_lower[target_year] = pd.Series([0], name='lower')
                all_years_data_weather_cancelled_upper[target_year] = pd.Series([0], name='upper')


            else:

                # Weather delayed appointments

                num_weather_delayed_appointments = summarize(
                    extract_results(
                        results_folder,
                        module='tlo.methods.healthsystem.summary',
                        key='weather_delayed_hsi_event_counts',
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
                        key='weather_cancelled_hsi_event_counts',
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
                    custom_generate_series=get_population_for_year,
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

        # PER 1000 POPULATION
        fig, axes = plt.subplots(2, 2, figsize=(25, 20))
        # Calculate per 1000 rates
        df_treatments_per_1000_mean = df_all_years_treatments_mean / df_all_years_data_population_mean.iloc[0, 0] * 1000
        df_never_ran_per_1000_mean = df_all_years_never_ran_mean / df_all_years_data_population_mean.iloc[0, 0] * 1000
        df_weather_delayed_per_1000_mean = df_all_years_weather_delayed_mean / df_all_years_data_population_mean.iloc[0, 0] * 1000
        df_weather_cancelled_per_1000_mean = df_all_years_weather_cancelled_mean / df_all_years_data_population_mean.iloc[0, 0] * 1000

        # Panel A: Treatments per 1000
        df_treatments_per_1000_mean.T.plot.bar(stacked=True, ax=axes[0,0])
        axes[0,0].set_title('Panel A: Healthcare Treatments per 1000 Population')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Treatments per 1000 people')
        axes[0,0].grid(True)
        axes[0,0].legend().set_visible(False)

        # Panel B: Never ran per 1000
        df_never_ran_per_1000_mean.T.plot.bar(stacked=True, ax=axes[0,1])
        axes[0,1].set_title('Panel B: Never Ran Appointments per 1000 Population')
        axes[0,1].set_ylabel('Never Ran Appointments per 1000 people')
        axes[0,1].set_xlabel('Year')
        axes[0,1].grid(True)
        axes[0,1].legend(title='Appointment Type', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Panel C: Weather delayed per 1000
        df_weather_delayed_per_1000_mean.T.plot.bar(stacked=True, ax=axes[1,0])
        axes[1,0].set_title('Panel C: Weather Delayed Appointments per 1000 Population')
        axes[1,0].set_ylabel('Weather Delayed Appointments per 1000 people')
        axes[1,0].set_xlabel('Year')
        axes[1,0].grid(True)
        axes[1,0].legend(title='Appointment Type', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Panel D: Weather cancelled per 1000
        df_weather_cancelled_per_1000_mean.T.plot.bar(stacked=True, ax=axes[1,1])
        axes[1,1].set_title('Panel D: Weather Cancelled Appointments per 1000 Population')
        axes[1,1].set_ylabel('Weather Cancelled Appointments per 1000 people')
        axes[1,1].set_xlabel('Year')
        axes[1,1].grid(True)
        axes[1,1].legend(title='Appointment Type', bbox_to_anchor=(1.05, 1), loc='upper left')

        fig.tight_layout()
        fig.savefig(make_graph_file_name(f'Healthcare_System_Utilization_Per_1000_With_{suffix}'))
        plt.close(fig)

        # Save data to CSV
        df_all_years_treatments_mean.to_csv(output_folder / f"treatments_by_type_{draw}.csv")
        df_all_years_never_ran_mean.to_csv(output_folder / f"never_ran_by_type_{draw}.csv")
        df_all_years_weather_delayed_mean.to_csv(output_folder / f"weather_delayed_by_type_{draw}.csv")
        df_all_years_weather_cancelled_mean.to_csv(output_folder / f"weather_cancelled_by_type_{draw}.csv")

        # Accumulate data across all draws
        all_draws_treatments_mean.append(pd.Series(df_all_years_treatments_mean.sum(), name=f'Draw {draw}'))
        all_draws_never_ran_mean.append(pd.Series(df_all_years_never_ran_mean.sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_mean.append(pd.Series(df_all_years_weather_delayed_mean.sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_mean.append(pd.Series(df_all_years_weather_cancelled_mean.sum(), name=f'Draw {draw}'))

        all_draws_treatments_lower.append(pd.Series(df_all_years_treatments_lower.sum(), name=f'Draw {draw}'))
        all_draws_never_ran_lower.append(pd.Series(df_all_years_never_ran_lower.sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_lower.append(pd.Series(df_all_years_weather_delayed_lower.sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_lower.append(pd.Series(df_all_years_weather_cancelled_lower.sum(), name=f'Draw {draw}'))

        all_draws_treatments_upper.append(pd.Series(df_all_years_treatments_upper.sum(), name=f'Draw {draw}'))
        all_draws_never_ran_upper.append(pd.Series(df_all_years_never_ran_upper.sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_upper.append(pd.Series(df_all_years_weather_delayed_upper.sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_upper.append(pd.Series(df_all_years_weather_cancelled_upper.sum(), name=f'Draw {draw}'))

        # Per 1000 for final year only
        all_draws_treatments_mean_1000.append(pd.Series(df_treatments_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_never_ran_mean_1000.append(pd.Series(df_never_ran_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_weather_delayed_mean_1000.append(pd.Series(df_weather_delayed_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_weather_cancelled_mean_1000.append(pd.Series(df_weather_cancelled_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))

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

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    width = 0.35

    # --------------------------
    # total treatments
    # --------------------------
    x = np.arange(len(treatments_totals_mean.index))
    axes[0].bar(x, treatments_totals_mean.values, width,
                color=scenario_colours, yerr=treatments_totals_err, capsize=10)
    axes[0].text(-0.0, 1.05, '(A)', transform=axes[0].transAxes,
                   fontsize=14, va='top', ha='right')
    axes[0].set_title(f'Total Health System Interactions (2020–2040)')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Total HSIs')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenario_names)
    axes[0].grid(False)

    # --------------------------
    # weather disruptions
    # --------------------------
    x = np.arange(2)  # only two bars: Delayed and Cancelled
    axes[1].bar(x[0], weather_delayed_totals_mean, width,
                label='Weather Delayed', color='#FEB95F',
                yerr=weather_delayed_totals_err, capsize=10)
    axes[1].bar(x[1], weather_cancelled_totals_mean, width,
                label='Weather Cancelled', color="#f07167",
                yerr=weather_cancelled_totals_err, capsize=10)
    axes[1].text(-0.0, 1.05, '(B)', transform=axes[1].transAxes,
                   fontsize=14, va='top', ha='right')
    axes[1].set_title(f'Weather-Disrupted Health System Interactions (2020–2040)')
    axes[1].set_xlabel('Disruption Type')
    axes[1].set_ylabel('')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Delayed', 'Cancelled'])
    axes[1].grid(False)

    fig.tight_layout()
    fig.savefig(output_folder / f"treatments_and_weather_disruptions_{suffix}.png")
    plt.close(fig)

    # Per 1000
    fig, axes = plt.subplots(2, 2, figsize=(25, 20))

    df_treatments_all_draws_mean_1000.T.plot.bar(stacked=True, ax=axes[0,0])
    axes[0,0].set_title(f'Healthcare Treatments per 1,000 ({max_year})')
    axes[0,0].set_xlabel('Scenario')
    axes[0,0].set_ylabel('Treatments per 1,000')
    axes[0,0].set_xticklabels(scenario_names, rotation=45)
    axes[0,0].legend().set_visible(False)

    df_never_ran_all_draws_mean_1000.T.plot.bar(stacked=True, ax=axes[0,1])
    axes[0,1].set_title(f'Never Ran Appointments per 1,000 ({max_year})')
    axes[0,1].set_xlabel('Scenario')
    axes[0,1].set_ylabel('Never Ran Appointments per 1,000')
    axes[0,1].set_xticklabels(scenario_names, rotation=45)
    axes[0,1].legend(title='Type', bbox_to_anchor=(1., 1), loc='upper left')

    df_weather_delayed_all_draws_mean_1000.T.plot.bar(stacked=True, ax=axes[1,0])
    axes[1,0].set_title(f'Weather Delayed Appointments per 1,000 ({max_year})')
    axes[1,0].set_xlabel('Scenario')
    axes[1,0].set_ylabel('Weather Delayed Appointments per 1,000')
    axes[1,0].set_xticklabels(scenario_names, rotation=45)
    axes[1,0].legend(title='Type', bbox_to_anchor=(1., 1), loc='upper left')

    df_weather_cancelled_all_draws_mean_1000.T.plot.bar(stacked=True, ax=axes[1,1])
    axes[1,1].set_title(f'Weather Cancelled Appointments per 1,000 ({max_year})')
    axes[1,1].set_xlabel('Scenario')
    axes[1,1].set_ylabel('Weather Cancelled Appointments per 1,000')
    axes[1,1].set_xticklabels(scenario_names, rotation=45)
    axes[1,1].legend(title='Type', bbox_to_anchor=(1., 1), loc='upper left')

    fig.tight_layout()
    fig.savefig(output_folder / f"treatments_and_appointments_per_1000_all_draws_with_weather_{max_year}_{suffix}.png")
    plt.close(fig)


    target_year_final = max_year
    target_period_final = (Date(2026, 1, 1), Date(target_year_final, 12, 31))
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
    for i, draw in enumerate(scenario_indices_final):
        result_data = summarize(
            extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_counts_of_hsi_by_short_treatment_id,
                do_scaling=True
            ),
            only_mean=True,
            collapse_columns=True,
        )[draw]
        final_data[scenario_labels_final[i]] = result_data['mean']

    df_final = pd.DataFrame(final_data).fillna(0)

    # --- Plot: stacked bar chart (Baseline vs SSP2.45)
    fig_final, ax_final = plt.subplots(figsize=(10, 7))
    bottom = np.zeros(len(scenario_labels_final))

    for treatment in df_final.index:
        values = df_final.loc[treatment]
        ax_final.bar(scenario_labels_final, values, bottom=bottom,
                     color=get_color_short_treatment_id(treatment),
                     label=treatment)
        bottom += values.values

    ax_final.set_ylabel("Total Number of HSIs ", fontsize=12)
    ax_final.set_xlabel("Scenario", fontsize=12)
    ax_final.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Treatment Type')
    ax_final.tick_params(axis='both', labelsize=11)
    fig_final.tight_layout()
    fig_final.savefig(output_folder / f"{PREFIX_ON_FILENAME}_Final_Treatments_StackedBar_all_years.png")
    plt.close(fig_final)

        # Calculate differences relative to baseline
    df_treatments_diff = df_all_years_treatments_mean - baseline_treatments_by_year
    df_never_ran_diff = df_all_years_never_ran_mean - baseline_never_ran_by_year
    # others cannot be different to baseline
    # Calculate percentage change relative to baseline
    df_treatments_pct_change = (
            (df_all_years_treatments_mean - baseline_treatments_by_year) / baseline_treatments_by_year * 100)
    df_never_ran_pct_change = (
            (df_all_years_never_ran_mean - baseline_never_ran_by_year) / baseline_never_ran_by_year * 100)

    # Save relative data
    df_treatments_diff.to_csv(output_folder / f"treatments_diff_from_baseline_{draw}.csv")
    df_never_ran_diff.to_csv(output_folder / f"never_ran_diff_from_baseline_{draw}.csv")


    df_treatments_pct_change.to_csv(output_folder / f"treatments_pct_change_from_baseline_{draw}.csv")
    df_never_ran_pct_change.to_csv(output_folder / f"never_ran_pct_change_from_baseline_{draw}.csv")

    # Save summary data
    df_treatments_all_draws_mean.to_csv(output_folder / "treatments_summary_all_draws.csv")
    df_never_ran_all_draws_mean.to_csv(output_folder / "never_ran_summary_all_draws.csv")
    df_weather_delayed_all_draws_mean.to_csv(output_folder / "weather_delayed_summary_all_draws.csv")
    df_weather_cancelled_all_draws_mean.to_csv(output_folder / "weather_cancelled_summary_all_draws.csv")

    # dont include delayed in denominator as included in treatments delivered
    ((df_weather_delayed_all_draws_mean / (df_treatments_all_draws_mean + df_weather_cancelled_all_draws_mean )*100).to_csv(
        output_folder / f"percentage_weather_delayed_by_all_draws.csv"))
    ((df_weather_cancelled_all_draws_mean / (df_treatments_all_draws_mean + df_weather_cancelled_all_draws_mean )*100).to_csv(
        output_folder / f"percentage_weather_cancelled_by_all_draws.csv"))

    (((df_weather_cancelled_all_draws_mean  + df_weather_delayed_all_draws_mean)/ (df_treatments_all_draws_mean + df_weather_cancelled_all_draws_mean)*100).to_csv(
        output_folder / f"percentage_weather_disrupted_by_all_draws.csv"))

    ((df_weather_delayed_all_draws_mean.sum() / (df_treatments_all_draws_mean.sum() + df_weather_cancelled_all_draws_mean.sum() )*100).to_csv(
        output_folder / f"percentage_weather_delayed_by_all_draws_total_across_years.csv"))
    ((df_weather_cancelled_all_draws_mean.sum() / (df_treatments_all_draws_mean.sum() + df_weather_cancelled_all_draws_mean.sum() )*100).to_csv(
        output_folder / f"percentage_weather_cancelled_by_all_draws_total_across_years.csv"))

    (((df_weather_cancelled_all_draws_mean.sum()  + df_weather_delayed_all_draws_mean.sum())/ (df_treatments_all_draws_mean.sum() + df_weather_cancelled_all_draws_mean.sum())*100).to_csv(
        output_folder / f"percentage_weather_disrupted_by_all_draws_total_across_years.csv"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
