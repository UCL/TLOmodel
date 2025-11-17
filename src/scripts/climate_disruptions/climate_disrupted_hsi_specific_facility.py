import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

min_year = 2026
max_year = 2041
spacing_of_years = 1
PREFIX_ON_FILENAME = "1"

scenario_names = ["Baseline", "SSP 2.45 Mean", "SSP 5.85 High"]
scenario_colours = ["#0081a7", "#00afb9", "#FEB95F", "#fed9b7", "#f07167"] * 4

# CHANGE THIS: Set to a single facility ID instead of a list
facility_of_interest = 4  # Single facility


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the healthcare system utilization across scenarios.
    - We estimate the healthcare system impact through total treatments and climate-disrupted appointments.
    - Modified to track non-blank HSIs for a specific facility.
    """
    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"  # noqa: E731

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions

    def get_num_treatments_total(_df):
        """Get count of non-blank HSIs for the specific facility"""
        if _df is None or _df.empty:
            return pd.Series(0, name="total_treatments")

        _df["date"] = pd.to_datetime(_df["date"], errors="coerce")

        # Filter to target period
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]

        # Filter to specific facility
        if "Facility_ID" in _df.columns:
            _df = _df.loc[_df["Facility_ID"] == facility_of_interest]

        # Handle empty result
        if _df.empty:
            return pd.Series(0, name="total_treatments")

        return pd.Series(len(_df), name="total_treatments")

    def get_num_treatments_total_disrupted(_df):
        """Get count of disrupted HSIs for the specific facility"""
        if _df is None or _df.empty:
            return pd.Series(0, name="total_treatments")

        _df["date"] = pd.to_datetime(_df["date"], errors="coerce")
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]

        # Filter to specific facility
        if "Facility_ID" in _df.columns:
            _df = _df.loc[_df["Facility_ID"] == facility_of_interest]
        else:
            # If no Facility_ID column, return 0
            return pd.Series(0, name="total_treatments")

        # Handle empty result
        if _df.empty:
            return pd.Series(0, name="total_treatments")

        return pd.Series(len(_df), name="total_treatments")

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    all_draws_treatments_mean = []
    all_draws_treatments_lower = []
    all_draws_treatments_upper = []

    all_draws_weather_delayed_mean = []
    all_draws_weather_delayed_lower = []
    all_draws_weather_delayed_upper = []

    all_draws_weather_cancelled_mean = []
    all_draws_weather_cancelled_lower = []
    all_draws_weather_cancelled_upper = []

    for draw in range(len(scenario_names)):
        all_years_data_treatments_mean = {}
        all_years_data_treatments_upper = {}
        all_years_data_treatments_lower = {}

        all_years_data_weather_delayed_mean = {}
        all_years_data_weather_delayed_upper = {}
        all_years_data_weather_delayed_lower = {}

        all_years_data_weather_cancelled_mean = {}
        all_years_data_weather_cancelled_upper = {}
        all_years_data_weather_cancelled_lower = {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

            # Total non-blank treatments for specific facility
            num_treatments_total = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="HSI_Event_non_blank_full_details",
                    custom_generate_series=get_num_treatments_total,
                    do_scaling=True,
                ),
                only_mean=False,
                collapse_columns=True,
            )[draw]
            all_years_data_treatments_mean[target_year] = num_treatments_total["mean"]
            all_years_data_treatments_lower[target_year] = num_treatments_total["lower"]
            all_years_data_treatments_upper[target_year] = num_treatments_total["upper"]

            # Weather delayed appointments - only for non-baseline scenarios
            if draw == 0:
                # Baseline has no weather disruptions
                all_years_data_weather_delayed_mean[target_year] = pd.Series([0], name="mean")
                all_years_data_weather_delayed_lower[target_year] = pd.Series([0], name="lower")
                all_years_data_weather_delayed_upper[target_year] = pd.Series([0], name="upper")
            else:
                num_weather_delayed_appointments = summarize(
                    extract_results(
                        results_folder,
                        module="tlo.methods.healthsystem.summary",
                        key="Weather_delayed_HSI_Event_full_info",
                        custom_generate_series=get_num_treatments_total_disrupted,
                        do_scaling=True,
                    ),
                    only_mean=False,
                    collapse_columns=True,
                )[draw]

                all_years_data_weather_delayed_mean[target_year] = num_weather_delayed_appointments["mean"]
                all_years_data_weather_delayed_lower[target_year] = num_weather_delayed_appointments["lower"]
                all_years_data_weather_delayed_upper[target_year] = num_weather_delayed_appointments["upper"]

            # Weather cancelled appointments - only for non-baseline scenarios
            if draw == 0:
                # Baseline has no weather disruptions
                all_years_data_weather_cancelled_mean[target_year] = pd.Series([0], name="mean")
                all_years_data_weather_cancelled_lower[target_year] = pd.Series([0], name="lower")
                all_years_data_weather_cancelled_upper[target_year] = pd.Series([0], name="upper")
            else:
                num_weather_cancelled_appointments = summarize(
                    extract_results(
                        results_folder,
                        module="tlo.methods.healthsystem.summary",
                        key="Weather_cancelled_HSI_Event_full_info",
                        custom_generate_series=get_num_treatments_total_disrupted,
                        do_scaling=True,
                    ),
                    only_mean=False,
                    collapse_columns=True,
                )[draw]

                all_years_data_weather_cancelled_mean[target_year] = num_weather_cancelled_appointments["mean"]
                all_years_data_weather_cancelled_lower[target_year] = num_weather_cancelled_appointments["lower"]
                all_years_data_weather_cancelled_upper[target_year] = num_weather_cancelled_appointments["upper"]

        # Convert the accumulated data into DataFrames for plotting
        df_all_years_treatments_mean = pd.DataFrame(all_years_data_treatments_mean)
        df_all_years_treatments_lower = pd.DataFrame(all_years_data_treatments_lower)
        df_all_years_treatments_upper = pd.DataFrame(all_years_data_treatments_upper)

        df_all_years_weather_delayed_mean = pd.DataFrame(all_years_data_weather_delayed_mean)
        df_all_years_weather_delayed_lower = pd.DataFrame(all_years_data_weather_delayed_lower)
        df_all_years_weather_delayed_upper = pd.DataFrame(all_years_data_weather_delayed_upper)

        df_all_years_weather_cancelled_mean = pd.DataFrame(all_years_data_weather_cancelled_mean)
        df_all_years_weather_cancelled_lower = pd.DataFrame(all_years_data_weather_cancelled_lower)
        df_all_years_weather_cancelled_upper = pd.DataFrame(all_years_data_weather_cancelled_upper)

        # Calculate disruption rates
        total_hsis = df_all_years_treatments_mean.sum(axis=0)
        delayed_hsis = df_all_years_weather_delayed_mean.sum(axis=0)
        cancelled_hsis = df_all_years_weather_cancelled_mean.sum(axis=0)

        # Calculate percentages (avoiding division by zero)
        pct_delayed = (delayed_hsis / total_hsis * 100).fillna(0).replace([np.inf, -np.inf], 0)
        pct_cancelled = (cancelled_hsis / total_hsis * 100).fillna(0).replace([np.inf, -np.inf], 0)
        pct_total_disrupted = ((delayed_hsis + cancelled_hsis) / total_hsis * 100).fillna(0).replace([np.inf, -np.inf],
                                                                                                     0)

        # New plot showing HSI counts and disruption rates for specific facility
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))

        # Panel A: Absolute counts
        years = df_all_years_treatments_mean.columns
        axes[0].plot(years, total_hsis, marker='o', label='Total Non-Blank HSIs', linewidth=2)
        axes[0].plot(years, delayed_hsis, marker='s', label='Weather Delayed', linewidth=2)
        axes[0].plot(years, cancelled_hsis, marker='^', label='Weather Cancelled', linewidth=2)
        axes[0].set_title(f'Panel A: HSI Counts at Facility {facility_of_interest} - {scenario_names[draw]}')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of HSIs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Panel B: Disruption rates
        axes[1].plot(years, pct_delayed, marker='s', label='% Delayed', linewidth=2)
        axes[1].plot(years, pct_cancelled, marker='^', label='% Cancelled', linewidth=2)
        axes[1].plot(years, pct_total_disrupted, marker='o', label='% Total Disrupted', linewidth=2, linestyle='--')
        axes[1].set_title(
            f'Panel B: Climate Disruption Rates at Facility {facility_of_interest} - {scenario_names[draw]}')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Percentage of Total HSIs (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(make_graph_file_name(f'Facility_{facility_of_interest}_HSI_and_Disruption'))
        plt.close(fig)

        # Save facility-specific summary data
        facility_summary = pd.DataFrame({
            'Year': years,
            'Total_NonBlank_HSIs': total_hsis.values,
            'Weather_Delayed': delayed_hsis.values,
            'Weather_Cancelled': cancelled_hsis.values,
            'Pct_Delayed': pct_delayed.values,
            'Pct_Cancelled': pct_cancelled.values,
            'Pct_Total_Disrupted': pct_total_disrupted.values
        })
        facility_summary.to_csv(output_folder / f'facility_{facility_of_interest}_summary_{draw}.csv', index=False)

        # Plotting - Healthcare System Utilization by Type
        fig, axes = plt.subplots(1, 3, figsize=(30, 8))

        # Panel A: Total Treatments
        for i, treatment_type in enumerate(df_all_years_treatments_mean.index):
            axes[0].plot(
                df_all_years_treatments_mean.columns,
                df_all_years_treatments_mean.loc[treatment_type],
                marker="o",
                label=treatment_type,
            )
        axes[0].set_title(f"Panel A: Healthcare Treatments by Type - {scenario_names[draw]}")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Number of Treatments")
        axes[0].grid(True)
        axes[0].legend(title="Treatment Type", bbox_to_anchor=(1.0, 1), loc="upper left")

        # Panel B: Weather Delayed Appointments
        for i, appt_type in enumerate(df_all_years_weather_delayed_mean.index):
            axes[1].plot(
                df_all_years_weather_delayed_mean.columns,
                df_all_years_weather_delayed_mean.loc[appt_type],
                marker="o",
                label=appt_type,
            )
        axes[1].set_title(f"Panel B: Weather Delayed Appointments by Type - {scenario_names[draw]}")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Number of Weather Delayed Appointments")
        axes[1].legend(title="Appointment Type", bbox_to_anchor=(1.0, 1), loc="upper left")
        axes[1].grid(True)

        # Panel C: Weather Cancelled Appointments
        for i, appt_type in enumerate(df_all_years_weather_cancelled_mean.index):
            axes[2].plot(
                df_all_years_weather_cancelled_mean.columns,
                df_all_years_weather_cancelled_mean.loc[appt_type],
                marker="o",
                label=appt_type,
            )
        axes[2].set_title(f"Panel C: Weather Cancelled Appointments by Type - {scenario_names[draw]}")
        axes[2].set_xlabel("Year")
        axes[2].set_ylabel("Number of Weather Cancelled Appointments")
        axes[2].legend(title="Appointment Type", bbox_to_anchor=(1.0, 1), loc="upper left")
        axes[2].grid(True)

        fig.tight_layout()
        fig.savefig(make_graph_file_name("Healthcare_System_Utilization_All_Years_With_Weather"))
        plt.close(fig)

        # STACKED BAR PLOTS
        fig, axes = plt.subplots(1, 3, figsize=(30, 8))

        df_all_years_treatments_mean.T.plot.bar(stacked=True, ax=axes[0])
        axes[0].set_title(f"Panel A: Healthcare Treatments by Type - {scenario_names[draw]}")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Number of Treatments")
        axes[0].legend(title="Treatment Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0].grid(True)

        df_all_years_weather_delayed_mean.T.plot.bar(stacked=True, ax=axes[1])
        axes[1].set_title(f"Panel B: Weather Delayed Appointments by Type - {scenario_names[draw]}")
        axes[1].set_ylabel("Number of Weather Delayed Appointments")
        axes[1].set_xlabel("Year")
        axes[1].legend(title="Appointment Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[1].grid(True)

        df_all_years_weather_cancelled_mean.T.plot.bar(stacked=True, ax=axes[2])
        axes[2].set_title(f"Panel C: Weather Cancelled Appointments by Type - {scenario_names[draw]}")
        axes[2].set_ylabel("Number of Weather Cancelled Appointments")
        axes[2].set_xlabel("Year")
        axes[2].legend(title="Appointment Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[2].grid(True)

        fig.tight_layout()
        fig.savefig(make_graph_file_name("Healthcare_System_Utilization_Stacked_With_Weather"))
        plt.close(fig)

        # STACKED AREA PLOTS
        fig, axes = plt.subplots(1, 3, figsize=(30, 8))

        # Panel A: Treatments (Stacked area plot)
        years_treatments = df_all_years_treatments_mean.columns
        treatment_types = df_all_years_treatments_mean.index
        axes[0].stackplot(years_treatments, df_all_years_treatments_mean.values, labels=treatment_types)
        axes[0].set_title(f"Panel A: Healthcare Treatments by Type - {scenario_names[draw]}")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Number of Treatments")
        axes[0].grid(True)

        # Panel B: Weather delayed appointments (Stacked area plot)
        years_weather_delayed = df_all_years_weather_delayed_mean.columns
        weather_delayed_types = df_all_years_weather_delayed_mean.index
        axes[1].stackplot(
            years_weather_delayed, df_all_years_weather_delayed_mean.values, labels=weather_delayed_types
        )
        axes[1].set_title(f"Panel B: Weather Delayed Appointments by Type - {scenario_names[draw]}")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Number of Weather Delayed Appointments")
        axes[1].legend(title="Appointment Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[1].grid(True)

        # Panel C: Weather cancelled appointments (Stacked area plot)
        years_weather_cancelled = df_all_years_weather_cancelled_mean.columns
        weather_cancelled_types = df_all_years_weather_cancelled_mean.index
        axes[2].stackplot(
            years_weather_cancelled, df_all_years_weather_cancelled_mean.values, labels=weather_cancelled_types
        )
        axes[2].set_title(f"Panel C: Weather Cancelled Appointments by Type - {scenario_names[draw]}")
        axes[2].set_xlabel("Year")
        axes[2].set_ylabel("Number of Weather Cancelled Appointments")
        axes[2].legend(title="Appointment Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[2].grid(True)

        fig.tight_layout()
        fig.savefig(make_graph_file_name("Healthcare_System_Utilization_Area_With_Weather"))
        plt.close(fig)

        # Save data to CSV
        df_all_years_treatments_mean.to_csv(output_folder / f"treatments_by_type_{draw}.csv")
        df_all_years_weather_delayed_mean.to_csv(output_folder / f"weather_delayed_by_type_{draw}.csv")
        df_all_years_weather_cancelled_mean.to_csv(output_folder / f"weather_cancelled_by_type_{draw}.csv")

        # Accumulate data across all draws
        all_draws_treatments_mean.append(pd.Series(df_all_years_treatments_mean.sum(), name=f"Draw {draw}"))
        all_draws_weather_delayed_mean.append(pd.Series(df_all_years_weather_delayed_mean.sum(), name=f"Draw {draw}"))
        all_draws_weather_cancelled_mean.append(
            pd.Series(df_all_years_weather_cancelled_mean.sum(), name=f"Draw {draw}")
        )

        all_draws_treatments_lower.append(pd.Series(df_all_years_treatments_lower.sum(), name=f"Draw {draw}"))
        all_draws_weather_delayed_lower.append(pd.Series(df_all_years_weather_delayed_lower.sum(), name=f"Draw {draw}"))
        all_draws_weather_cancelled_lower.append(
            pd.Series(df_all_years_weather_cancelled_lower.sum(), name=f"Draw {draw}")
        )

        all_draws_treatments_upper.append(pd.Series(df_all_years_treatments_upper.sum(), name=f"Draw {draw}"))
        all_draws_weather_delayed_upper.append(pd.Series(df_all_years_weather_delayed_upper.sum(), name=f"Draw {draw}"))
        all_draws_weather_cancelled_upper.append(
            pd.Series(df_all_years_weather_cancelled_upper.sum(), name=f"Draw {draw}")
        )

    # Combine all draws
    df_treatments_all_draws_mean = pd.concat(all_draws_treatments_mean, axis=1)
    df_weather_delayed_all_draws_mean = pd.concat(all_draws_weather_delayed_mean, axis=1)
    df_weather_cancelled_all_draws_mean = pd.concat(all_draws_weather_cancelled_mean, axis=1)

    df_treatments_all_draws_lower = pd.concat(all_draws_treatments_lower, axis=1)
    df_weather_delayed_all_draws_lower = pd.concat(all_draws_weather_delayed_lower, axis=1)
    df_weather_cancelled_all_draws_lower = pd.concat(all_draws_weather_cancelled_lower, axis=1)

    df_treatments_all_draws_upper = pd.concat(all_draws_treatments_upper, axis=1)
    df_weather_delayed_all_draws_upper = pd.concat(all_draws_weather_delayed_upper, axis=1)
    df_weather_cancelled_all_draws_upper = pd.concat(all_draws_weather_cancelled_upper, axis=1)

    # Final summary plots across all scenarios
    treatments_totals_mean = df_treatments_all_draws_mean.sum()
    weather_delayed_totals_mean = df_weather_delayed_all_draws_mean.sum()
    weather_cancelled_totals_mean = df_weather_cancelled_all_draws_mean.sum()

    treatments_totals_lower = df_treatments_all_draws_lower.sum()
    treatments_totals_upper = df_treatments_all_draws_upper.sum()
    weather_delayed_totals_lower = df_weather_delayed_all_draws_lower.sum()
    weather_delayed_totals_upper = df_weather_delayed_all_draws_upper.sum()
    weather_cancelled_totals_lower = df_weather_cancelled_all_draws_lower.sum()
    weather_cancelled_totals_upper = df_weather_cancelled_all_draws_upper.sum()

    treatments_totals_err = np.array(
        [treatments_totals_mean - treatments_totals_lower, treatments_totals_upper - treatments_totals_mean]
    )

    weather_delayed_totals_err = np.array(
        [
            weather_delayed_totals_mean - weather_delayed_totals_lower,
            weather_delayed_totals_upper - weather_delayed_totals_mean,
        ]
    )

    weather_cancelled_totals_err = np.array(
        [
            weather_cancelled_totals_mean - weather_cancelled_totals_lower,
            weather_cancelled_totals_upper - weather_cancelled_totals_mean,
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Panel A: Total Treatments
    axes[0].bar(
        treatments_totals_mean.index,
        treatments_totals_mean.values,
        color=scenario_colours,
        yerr=treatments_totals_err,
        capsize=20,
    )
    axes[0].set_title(f"Total Healthcare Treatments ({min_year}-{max_year})")
    axes[0].set_xlabel("Scenario")
    axes[0].set_ylabel("Total Treatments")
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].grid(True)

    # Panel B: Total Weather Delayed Appointments
    axes[1].bar(
        weather_delayed_totals_mean.index,
        weather_delayed_totals_mean.values,
        color=scenario_colours,
        yerr=weather_delayed_totals_err,
        capsize=20,
    )
    axes[1].set_title(f"Total Weather Delayed Appointments ({min_year}-{max_year})")
    axes[1].set_xlabel("Scenario")
    axes[1].set_ylabel("Total Weather Delayed Appointments")
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].grid(True)

    # Panel C: Total Weather Cancelled Appointments
    axes[2].bar(
        weather_cancelled_totals_mean.index,
        weather_cancelled_totals_mean.values,
        color=scenario_colours,
        yerr=weather_cancelled_totals_err,
        capsize=20,
    )
    axes[2].set_title(f"Total Weather Cancelled Appointments ({min_year}-{max_year})")
    axes[2].set_xlabel("Scenario")
    axes[2].set_ylabel("Total Weather Cancelled Appointments")
    axes[2].set_xticklabels(scenario_names, rotation=45)
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(output_folder / "total_treatments_and_appointments_all_draws_with_weather.png")
    plt.close(fig)

    # Save summary data
    df_treatments_all_draws_mean.to_csv(output_folder / "treatments_summary_all_draws.csv")
    df_weather_delayed_all_draws_mean.to_csv(output_folder / "weather_delayed_summary_all_draws.csv")
    df_weather_cancelled_all_draws_mean.to_csv(output_folder / "weather_cancelled_summary_all_draws.csv")

    (df_weather_delayed_all_draws_mean / df_treatments_all_draws_mean).to_csv(
        output_folder / "percentage_weather_delayed_by_all_draws.csv"
    )
    (df_weather_cancelled_all_draws_mean / df_treatments_all_draws_mean).to_csv(
        output_folder / "percentage_weather_cancelled_by_all_draws.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
