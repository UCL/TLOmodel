import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import extract_results, summarize, get_color_short_treatment_id

min_year = 2025
max_year = 2029
spacing_of_years = 1
PREFIX_ON_FILENAME = "1"
climate_sensitivity_analysis = False
parameter_sensitivity_analysis = False
main_text = True

if climate_sensitivity_analysis:
    scenario_names = [
        "Baseline",
        "SSP 1.26 High",
        "SSP 1.26 Low",
        "SSP 1.26 Mean",
        "SSP 2.45 High",
        "SSP 2.45 Low",
        "SSP 2.45 Mean",
        "SSP 5.85 High",
        "SSP 5.85 Low",
        "SSP 5.85 Mean",
    ]

    suffix = "climate_SA"
    scenarios_of_interest = range(len(scenario_names))
if parameter_sensitivity_analysis:
    scenario_names_all = range(0, 10, 1)
    scenario_names = scenario_names_all
    suffix = "parameter_SA"
    scenarios_of_interest = [0, 1, 2, 3, 4, 7, 8, 9]  # range(0, 10, 1)

if main_text:
    scenario_names = [
        "Baseline",
        "SSP 2.45 Mean",
        "Worst Case"
    ]
    suffix = "main_text"
    scenarios_of_interest = [0, 1, 2]

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

scenario_colours = [
    "#823038",  # Baseline

    # SSP 1.26 (Teal)
    "#00566f",  # High
    "#0081a7",  # Low
    "#5ab4c6",  # Mean

    # SSP 2.45 (Purple/Lavender - more distinct)
    "#5b3f8c",  # High
    "#8e7cc3",  # Low
    "#c7b7ec",  # Mean

    # SSP 5.85 (Coral)
    "#c65a52",  # High
    "#f07167",  # Low
    "#f59e96",  # Mean
]


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the healthcare system utilization across scenarios.
    - We estimate the healthcare system impact through total treatments and never-ran appointments.
    - Now includes weather-delayed and weather-cancelled appointments.
    - Refactored to extract results once per draw instead of per year.
    """

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Simplified helper functions that just sum counts
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
        print(pd.Series(len(_df), name="total"))
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

    # Storage for all draws
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

    baseline_treatments_total = None
    baseline_never_ran_total = None
    baseline_population = None

    for draw in range(len(scenario_names)):
        print(draw)
        if draw not in scenarios_of_interest:
            continue
        print(draw)

        print(f"Processing draw {draw}...")
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"

        # Extract all results ONCE per draw
        print(f"  Extracting total treatments...")
        num_treatments_total = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.healthsystem.summary",
                key="hsi_event_counts",
                custom_generate_series=get_num_treatments_total,
                do_scaling=False,
            ),
            only_mean=False,
            collapse_columns=True,
        )[draw]

        print(f"  Extracting population...")
        population_data = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="population",
                custom_generate_series=get_population_total,
                do_scaling=False,
            ),
            only_mean=False,
            collapse_columns=True,
        )[draw]

        print(f"  Extracting never-ran appointments...")
        num_never_ran_appts = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.healthsystem.summary",
                key="never_ran_hsi_event_counts",
                custom_generate_series=get_num_treatments_never_ran,
                do_scaling=False,
            ),
            only_mean=False,
            collapse_columns=True,
        )[draw]

        if scenario_names[draw] == 'Baseline':
            # Baseline: no weather disruptions
            num_weather_delayed = {"mean": pd.Series([0]), "lower": pd.Series([0]), "upper": pd.Series([0])}
            num_weather_cancelled = {"mean": pd.Series([0]), "lower": pd.Series([0]), "upper": pd.Series([0])}

            # Store baseline values
            baseline_treatments_total = num_treatments_total["mean"].values[0]
            baseline_never_ran_total = num_never_ran_appts["mean"].values[0]
            baseline_population = population_data["mean"].values[0]
        elif main_text:
            num_weather_delayed = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="Weather_delayed_HSI_Event_full_info",
                    custom_generate_series=get_num_treatments_total_delayed,
                    do_scaling=False,
                ),
                only_mean=False,
                collapse_columns=True,
            )[draw]

            print(f"  Extracting weather-cancelled appointments...")
            num_weather_cancelled = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="Weather_cancelled_HSI_Event_full_info",
                    custom_generate_series=get_num_treatments_total_cancelled,
                    do_scaling=False,
                ),
                only_mean=False,
                collapse_columns=True,
            )[draw]

        else:
            print(f"  Extracting weather-delayed appointments...")
            num_weather_delayed = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="Weather_delayed_HSI_Event_full_info",
                    custom_generate_series=get_num_treatments_total_delayed,
                    do_scaling=False,
                ),
                only_mean=False,
                collapse_columns=True,
            )[draw]

            print(f"  Extracting weather-cancelled appointments...")
            num_weather_cancelled = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="Weather_cancelled_HSI_Event_full_info",
                    custom_generate_series=get_num_treatments_total_cancelled,
                    do_scaling=False,
                ),
                only_mean=False,
                collapse_columns=True,
            )[draw]

            print(num_weather_cancelled)
        # Store results
        all_draws_treatments_mean.append(num_treatments_total["mean"])
        all_draws_treatments_lower.append(num_treatments_total["lower"])
        all_draws_treatments_upper.append(num_treatments_total["upper"])

        all_draws_never_ran_mean.append(num_never_ran_appts["mean"])
        all_draws_never_ran_lower.append(num_never_ran_appts["lower"])
        all_draws_never_ran_upper.append(num_never_ran_appts["upper"])

        all_draws_weather_delayed_mean.append(num_weather_delayed["mean"])
        all_draws_weather_delayed_lower.append(num_weather_delayed["lower"])
        all_draws_weather_delayed_upper.append(num_weather_delayed["upper"])

        all_draws_weather_cancelled_mean.append(num_weather_cancelled["mean"])
        all_draws_weather_cancelled_lower.append(num_weather_cancelled["lower"])
        all_draws_weather_cancelled_upper.append(num_weather_cancelled["upper"])

        # Save individual draw data
        pd.DataFrame({
            "treatments_mean": num_treatments_total["mean"].values[0],
            "never_ran_mean": num_never_ran_appts["mean"].values[0],
            "weather_delayed_mean": num_weather_delayed["mean"].values[0],
            "weather_cancelled_mean": num_weather_cancelled["mean"].values[0],
            "population": population_data["mean"].values[0],
        }, index=[0]).to_csv(output_folder / f"summary_draw_{draw}.csv", index=False)

        print(f"Draw {draw} complete.")

    # Combine all draws
    print("\nCombining all draws...")

    def series_to_value(series_list):
        """Convert list of Series to array of values"""
        return np.array([s.values[0] if len(s.values) > 0 else 0 for s in series_list])

    treatments_mean_values = series_to_value(all_draws_treatments_mean)
    treatments_lower_values = series_to_value(all_draws_treatments_lower)
    treatments_upper_values = series_to_value(all_draws_treatments_upper)

    never_ran_mean_values = series_to_value(all_draws_never_ran_mean)
    never_ran_lower_values = series_to_value(all_draws_never_ran_lower)
    never_ran_upper_values = series_to_value(all_draws_never_ran_upper)

    weather_delayed_mean_values = series_to_value(all_draws_weather_delayed_mean)
    weather_delayed_lower_values = series_to_value(all_draws_weather_delayed_lower)
    weather_delayed_upper_values = series_to_value(all_draws_weather_delayed_upper)

    weather_cancelled_mean_values = series_to_value(all_draws_weather_cancelled_mean)
    weather_cancelled_lower_values = series_to_value(all_draws_weather_cancelled_lower)
    weather_cancelled_upper_values = series_to_value(all_draws_weather_cancelled_upper)

    # Create summary DataFrames
    print(scenarios_of_interest)
    print(treatments_mean_values)
    print(never_ran_mean_values)
    print(weather_delayed_mean_values)
    print(weather_cancelled_mean_values)
    summary_df = pd.DataFrame({'Scenario': [scenario_names[i] if climate_sensitivity_analysis
                                            else f"Draw {draw}" if parameter_sensitivity_analysis
    else f"Draw {i}"
                                            for i, draw in enumerate(scenarios_of_interest)],
                               'treatments_mean': treatments_mean_values,
                               'treatments_lower': treatments_lower_values,
                               'treatments_upper': treatments_upper_values,
                               'never_ran_mean': never_ran_mean_values,
                               'never_ran_lower': never_ran_lower_values,
                               'never_ran_upper': never_ran_upper_values,
                               'weather_delayed_mean': weather_delayed_mean_values,
                               'weather_delayed_lower': weather_delayed_lower_values,
                               'weather_delayed_upper': weather_delayed_upper_values,
                               'weather_cancelled_mean': weather_cancelled_mean_values,
                               'weather_cancelled_lower': weather_cancelled_lower_values,
                               'weather_cancelled_upper': weather_cancelled_upper_values,
    })

    summary_df.to_csv(output_folder / "summary_all_draws.csv", index=False)

    # Calculate error bars
    treatments_err = np.array([
        treatments_mean_values - treatments_lower_values,
        treatments_upper_values - treatments_mean_values
    ])

    weather_delayed_err = np.array([
        weather_delayed_mean_values - weather_delayed_lower_values,
        weather_delayed_upper_values - weather_delayed_mean_values
    ])

    weather_cancelled_err = np.array([
        weather_cancelled_mean_values - weather_cancelled_lower_values,
        weather_cancelled_upper_values - weather_cancelled_mean_values
    ])

    # Create main visualization
    print("\nCreating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    width = 0.35

    # Panel A: Total treatments
    x = np.arange(len(treatments_mean_values))
    axes[0].bar(x, treatments_mean_values, width, color=scenario_colours[:len(x)],
                yerr=treatments_err, capsize=6)
    axes[0].text(-0.0, 1.05, "(A)", transform=axes[0].transAxes, fontsize=14, va="top", ha="right")
    axes[0].set_title(f"Total Health System Interactions ({min_year}–{max_year})")
    axes[0].set_xlabel("Scenario")
    axes[0].set_ylabel("Total HSIs")
    axes[0].set_xticks(x)
    if climate_sensitivity_analysis:
        axes[0].set_xticklabels(scenario_names, rotation=45, ha='right')
    else:
        axes[0].set_xticklabels([f"Draw {i}" for i in scenarios_of_interest])
    axes[0].grid(False)

    # Panel B: Weather disruptions (excluding baseline)
    if len(x) > 1:
        x_weather = np.arange(len(weather_delayed_mean_values[1:]))
        bar_width = width / 2

        axes[1].bar(
            x_weather - bar_width / 2,
            weather_delayed_mean_values[1:],
            bar_width,
            label="Weather Delayed",
            color="#FEB95F",
            yerr=weather_delayed_err[:, 1:],
            capsize=6,
        )
        axes[1].bar(
            x_weather + bar_width / 2,
            weather_cancelled_mean_values[1:],
            bar_width,
            label="Weather Cancelled",
            color="#f07167",
            yerr=weather_cancelled_err[:, 1:],
            capsize=6,
        )
        axes[1].text(-0.0, 1.05, "(B)", transform=axes[1].transAxes, fontsize=14, va="top", ha="right")
        axes[1].set_title(f"Weather-Disrupted Health System Interactions ({min_year}–{max_year})")
        axes[1].set_xlabel("Scenario")
        axes[1].set_ylabel("Total Weather-Disrupted HSIs")
        axes[1].set_xticks(x_weather)
        if climate_sensitivity_analysis:
            axes[1].set_xticklabels(scenario_names[1:], rotation=45, ha='right')
        else:
            axes[1].set_xticklabels([f"Draw {i}" for i in list(scenarios_of_interest)[1:]])
        axes[1].grid(False)
        axes[1].legend(loc='upper left', frameon=False)

    fig.tight_layout()
    fig.savefig(output_folder / f"treatments_and_weather_disruptions_{suffix}.png", dpi=300)
    plt.close(fig)

    # Calculate percentages
    print("\nCalculating disruption percentages...")

    # Percentage calculations (excluding baseline where appropriate)
    total_potential_hsis = treatments_mean_values + weather_cancelled_mean_values

    pct_delayed = (weather_delayed_mean_values / total_potential_hsis * 100)
    pct_cancelled = (weather_cancelled_mean_values / total_potential_hsis * 100)
    pct_disrupted = ((weather_delayed_mean_values + weather_cancelled_mean_values) / total_potential_hsis * 100)

    percentage_df = pd.DataFrame({
        'Scenario': summary_df['Scenario'],
        'pct_delayed': pct_delayed,
        'pct_cancelled': pct_cancelled,
        'pct_disrupted': pct_disrupted,
    })

    percentage_df.to_csv(output_folder / "percentage_disruptions_by_scenario.csv", index=False)

    # Calculate differences from baseline (if baseline exists)
    if baseline_treatments_total is not None:
        print("\nCalculating differences from baseline...")

        diff_df = pd.DataFrame({
            'Scenario': summary_df['Scenario'],
            'treatments_diff': treatments_mean_values - baseline_treatments_total,
            'treatments_pct_change': ((treatments_mean_values - baseline_treatments_total) /
                                      baseline_treatments_total * 100),
            'never_ran_diff': never_ran_mean_values - baseline_never_ran_total,
            'never_ran_pct_change': ((never_ran_mean_values - baseline_never_ran_total) /
                                     baseline_never_ran_total * 100),
        })

        diff_df.to_csv(output_folder / "differences_from_baseline.csv", index=False)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
