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
    load_pickled_dataframes
)

min_year = 2026
max_year = 2044
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'

scenario_names = [
    "Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean",
    "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",
    "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"
]
scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167'] * 4


def plot_time_series(ax, df, title, ylabel, legend_title=None):
    """Helper to plot a time series with markers and legend."""
    for series in df.index:
        ax.plot(df.columns, df.loc[series], marker='o', label=series)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    if legend_title:
        ax.legend(title=legend_title, bbox_to_anchor=(1., 1), loc='upper left')
    ax.grid(True)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce simplified set of plots describing healthcare utilization across scenarios.
    - Includes treatments, never-ran, weather-delayed, and weather-cancelled appointments.
    """

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    def get_facility_level_data(_df):
        _df['date'] = pd.to_datetime(_df['date'])
        _df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        return (
            _df.drop(columns=['Appt_Type', 'TREATMENT_ID', 'Event_Name', 'Facility_Level', 'Person_ID', 'priority'])
               .melt(id_vars=['date'], var_name='Facility_Level', value_name='Num')
               .groupby(by=['date', 'Facility_Level'])['Num'].sum()
        )

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    all_draws_treatments_mean, all_draws_never_ran_mean = [], []
    all_draws_weather_delayed_mean, all_draws_weather_cancelled_mean = [], []
    all_draws_treatments_lower, all_draws_treatments_upper = [], []
    all_draws_never_ran_lower, all_draws_never_ran_upper = [], []
    all_draws_weather_delayed_lower, all_draws_weather_delayed_upper = [], []
    all_draws_weather_cancelled_lower, all_draws_weather_cancelled_upper = [], []

    for draw in range(len(scenario_names)):
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"

        all_years_data_treatments_mean, all_years_data_treatments_lower, all_years_data_treatments_upper = {}, {}, {}
        all_years_data_never_ran_mean, all_years_data_never_ran_lower, all_years_data_never_ran_upper = {}, {}, {}
        all_years_data_weather_delayed_mean, all_years_data_weather_delayed_lower, all_years_data_weather_delayed_upper = {}, {}, {}
        all_years_data_weather_cancelled_mean, all_years_data_weather_cancelled_lower, all_years_data_weather_cancelled_upper = {}, {}, {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

            if draw == 0:
                # Placeholder for baseline run
                for dct in [
                    all_years_data_treatments_mean, all_years_data_treatments_lower, all_years_data_treatments_upper,
                    all_years_data_never_ran_mean, all_years_data_never_ran_lower, all_years_data_never_ran_upper,
                    all_years_data_weather_delayed_mean, all_years_data_weather_delayed_lower, all_years_data_weather_delayed_upper,
                    all_years_data_weather_cancelled_mean, all_years_data_weather_cancelled_lower, all_years_data_weather_cancelled_upper,
                ]:
                    dct[target_year] = pd.Series([0], name='placeholder')

            else:
                # Total treatments
                num_treatments_total = summarize(
                    extract_results(results_folder, module='tlo.methods.healthsystem',
                                    key='HSI_Counts',
                                    custom_generate_series=get_facility_level_data,
                                    do_scaling=True),
                    only_mean=False, collapse_columns=True
                )[draw]
                all_years_data_treatments_mean[target_year] = num_treatments_total['mean']
                all_years_data_treatments_lower[target_year] = num_treatments_total['lower']
                all_years_data_treatments_upper[target_year] = num_treatments_total['upper']

                # Never ran
                num_never_ran_appts = summarize(
                    extract_results(results_folder, module='tlo.methods.healthsystem',
                                    key='Never_ran_HSI_Event',
                                    custom_generate_series=get_facility_level_data,
                                    do_scaling=True),
                    only_mean=False, collapse_columns=True
                )[draw]
                all_years_data_never_ran_mean[target_year] = num_never_ran_appts['mean']
                all_years_data_never_ran_lower[target_year] = num_never_ran_appts['lower']
                all_years_data_never_ran_upper[target_year] = num_never_ran_appts['upper']

                # Weather delayed
                num_weather_delayed = summarize(
                    extract_results(results_folder, module='tlo.methods.healthsystem.summary',
                                    key='Weather_cancelled_HSI_Event_full_info',
                                    custom_generate_series=get_facility_level_data,
                                    do_scaling=True),
                    only_mean=False, collapse_columns=True
                )[draw]
                all_years_data_weather_delayed_mean[target_year] = num_weather_delayed['mean']
                all_years_data_weather_delayed_lower[target_year] = num_weather_delayed['lower']
                all_years_data_weather_delayed_upper[target_year] = num_weather_delayed['upper']

                # Weather cancelled
                num_weather_cancelled = summarize(
                    extract_results(results_folder, module='tlo.methods.healthsystem.summary',
                                    key='Weather_delayed_HSI_Event_full_info',
                                    custom_generate_series=get_facility_level_data,
                                    do_scaling=True),
                    only_mean=False, collapse_columns=True
                )[draw]
                all_years_data_weather_cancelled_mean[target_year] = num_weather_cancelled['mean']
                all_years_data_weather_cancelled_lower[target_year] = num_weather_cancelled['lower']
                all_years_data_weather_cancelled_upper[target_year] = num_weather_cancelled['upper']

        # Convert to DataFrames
        df_all_years_treatments_mean = pd.DataFrame(all_years_data_treatments_mean)
        df_all_years_never_ran_mean = pd.DataFrame(all_years_data_never_ran_mean)
        df_all_years_weather_delayed_mean = pd.DataFrame(all_years_data_weather_delayed_mean)
        df_all_years_weather_cancelled_mean = pd.DataFrame(all_years_data_weather_cancelled_mean)

        # --- Line plots (time series, 4 panels) ---
        fig, axes = plt.subplots(2, 2, figsize=(25, 20))
        plot_time_series(axes[0, 0], df_all_years_treatments_mean,
                         "Panel A: Healthcare Treatments by Type", "Number of Treatments", "Treatment Type")
        plot_time_series(axes[0, 1], df_all_years_never_ran_mean,
                         "Panel B: Never Ran Appointments by Type", "Number of Never Ran Appointments", "Appointment Type")
        plot_time_series(axes[1, 0], df_all_years_weather_delayed_mean,
                         "Panel C: Weather Delayed Appointments by Type", "Number of Weather Delayed Appointments", "Appointment Type")
        plot_time_series(axes[1, 1], df_all_years_weather_cancelled_mean,
                         "Panel D: Weather Cancelled Appointments by Type", "Number of Weather Cancelled Appointments", "Appointment Type")
        fig.tight_layout()
        fig.savefig(make_graph_file_name('Healthcare_System_Utilization_TimeSeries'))
        plt.close(fig)

        # Save data
        df_all_years_treatments_mean.to_csv(output_folder / f"treatments_by_type_{draw}.csv")
        df_all_years_never_ran_mean.to_csv(output_folder / f"never_ran_by_type_{draw}.csv")
        df_all_years_weather_delayed_mean.to_csv(output_folder / f"weather_delayed_by_type_{draw}.csv")
        df_all_years_weather_cancelled_mean.to_csv(output_folder / f"weather_cancelled_by_type_{draw}.csv")

        # Accumulate totals
        all_draws_treatments_mean.append(pd.Series(df_all_years_treatments_mean.sum(), name=f'Draw {draw}'))
        all_draws_never_ran_mean.append(pd.Series(df_all_years_never_ran_mean.sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_mean.append(pd.Series(df_all_years_weather_delayed_mean.sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_mean.append(pd.Series(df_all_years_weather_cancelled_mean.sum(), name=f'Draw {draw}'))

        all_draws_treatments_lower.append(pd.Series(pd.DataFrame(all_years_data_treatments_lower).sum(), name=f'Draw {draw}'))
        all_draws_never_ran_lower.append(pd.Series(pd.DataFrame(all_years_data_never_ran_lower).sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_lower.append(pd.Series(pd.DataFrame(all_years_data_weather_delayed_lower).sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_lower.append(pd.Series(pd.DataFrame(all_years_data_weather_cancelled_lower).sum(), name=f'Draw {draw}'))

        all_draws_treatments_upper.append(pd.Series(pd.DataFrame(all_years_data_treatments_upper).sum(), name=f'Draw {draw}'))
        all_draws_never_ran_upper.append(pd.Series(pd.DataFrame(all_years_data_never_ran_upper).sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_upper.append(pd.Series(pd.DataFrame(all_years_data_weather_delayed_upper).sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_upper.append(pd.Series(pd.DataFrame(all_years_data_weather_cancelled_upper).sum(), name=f'Draw {draw}'))

    # --- Final summary bar plots (across scenarios) ---
    df_treatments_all_draws_mean = pd.concat(all_draws_treatments_mean, axis=1)
    df_never_ran_all_draws_mean = pd.concat(all_draws_never_ran_mean, axis=1)
    df_weather_delayed_all_draws_mean = pd.concat(all_draws_weather_delayed_mean, axis=1)
    df_weather_cancelled_all_draws_mean = pd.concat(all_draws_weather_cancelled_mean, axis=1)

    df_treatments_all_draws_lower = pd.concat(all_draws_treatments_lower, axis=1)
    df_treatments_all_draws_upper = pd.concat(all_draws_treatments_upper, axis=1)
    df_never_ran_all_draws_lower = pd.concat(all_draws_never_ran_lower, axis=1)
    df_never_ran_all_draws_upper = pd.concat(all_draws_never_ran_upper, axis=1)
    df_weather_delayed_all_draws_lower = pd.concat(all_draws_weather_delayed_lower, axis=1)
    df_weather_delayed_all_draws_upper = pd.concat(all_draws_weather_delayed_upper, axis=1)
    df_weather_cancelled_all_draws_lower = pd.concat(all_draws_weather_cancelled_lower, axis=1)
    df_weather_cancelled_all_draws_upper = pd.concat(all_draws_weather_cancelled_upper, axis=1)

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

    treatments_totals_err = np.array([treatments_totals_mean - treatments_totals_lower,
                                      treatments_totals_upper - treatments_totals_mean])
    never_ran_totals_err = np.array([never_ran_totals_mean - never_ran_totals_lower,
                                     never_ran_totals_upper - never_ran_totals_mean])
    weather_delayed_totals_err = np.array([weather_delayed_totals_mean - weather_delayed_totals_lower,
                                           weather_delayed_totals_upper - weather_delayed_totals_mean])
    weather_cancelled_totals_err = np.array([weather_cancelled_totals_mean - weather_cancelled_totals_lower,
                                             weather_cancelled_totals_upper - weather_cancelled_totals_mean])

    fig, axes = plt.subplots(2, 2, figsize=(25, 20))
    axes[0, 0].bar(treatments_totals_mean.index, treatments_totals_mean.values,
                   color=scenario_colours, yerr=treatments_totals_err, capsize=20)
    axes[0, 0].set_title(f'Total Healthcare Treatments (2020-{max_year})')
    axes[0, 0].set_xlabel('Scenario')
    axes[0, 0].set_ylabel('Total Treatments')
    axes[0, 0].set_xticklabels(scenario_names, rotation=45)

    axes[0, 1].bar(never_ran_totals_mean.index, never_ran_totals_mean.values,
                   color=scenario_colours, yerr=never_ran_totals_err, capsize=20)
    axes[0, 1].set_title(f'Total Never Ran Appointments (2020-{max_year})')
    axes[0, 1].set_xlabel('Scenario')
    axes[0, 1].set_ylabel('Total Never Ran Appointments')
    axes[0, 1].set_xticklabels(scenario_names, rotation=45)

    axes[1, 0].bar(weather_delayed_totals_mean.index, weather_delayed_totals_mean.values,
                   color=scenario_colours, yerr=weather_delayed_totals_err, capsize=20)
    axes[1, 0].set_title(f'Total Weather Delayed Appointments (2020-{max_year})')
    axes[1, 0].set_xlabel('Scenario')
    axes[1, 0].set_ylabel('Total Weather Delayed Appointments')
    axes[1, 0].set_xticklabels(scenario_names, rotation=45)

    axes[1, 1].bar(weather_cancelled_totals_mean.index, weather_cancelled_totals_mean.values,
                   color=scenario_colours, yerr=weather_cancelled_totals_err, capsize=20)
    axes[1, 1].set_title(f'Total Weather Cancelled Appointments (2020-{max_year})')
    axes[1, 1].set_xlabel('Scenario')
    axes[1, 1].set_ylabel('Total Weather Cancelled Appointments')
    axes[1, 1].set_xticklabels(scenario_names, rotation=45)

    fig.tight_layout()
    fig.savefig(output_folder / "total_treatments_and_appointments_all_draws_with_weather.png")
    plt.close(fig)

    # Save summary
    df_treatments_all_draws_mean.to_csv(output_folder / "treatments_summary_all_draws.csv")
    df_never_ran_all_draws_mean.to_csv(output_folder / "never_ran_summary_all_draws.csv")
    df_weather_delayed_all_draws_mean.to_csv(output_folder / "weather_delayed_summary_all_draws.csv")
    df_weather_cancelled_all_draws_mean.to_csv(output_folder / "weather_cancelled_summary_all_draws.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
