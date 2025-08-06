import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    summarize,
    parse_log_file
)
import geopandas as gpd

min_year = 2020
max_year = 2028
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'

scenario_names = ["Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",  "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]
scenario_names = ["Baseline"]#, "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",  "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]

scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167']*4
def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce a standard set of plots describing the effect of each climate scenario on appointment delivery
    """
    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))
    target_year_sequence = range(min_year, max_year, spacing_of_years)

    # Function for getting missing appointments
    def get_counts_of_HSIs(_df):
        """Get the counts of the short TREATMENT_IDs occurring"""
        _counts= _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD)] \
            .apply(pd.Series) \
            .sum() \
            .astype(int)
        return _counts.groupby(level=0).sum()

    # Storage
    all_scenarios_cancelled = {}
    all_scenarios_delayed = {}

    for draw in range(len(scenario_names)):
        scenario_name = scenario_names[draw]
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"

        all_years_data_cancelled_mean = {}
        all_years_data_cancelled_upper = {}
        all_years_data_cancelled_lower = {}

        all_years_data_delayed_mean = {}
        all_years_data_delayed_upper = {}
        all_years_data_delayed_lower = {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (
                Date(target_year, 1, 1), Date(target_year, 12, 31))


            result_cancelled = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='weather_cancelled_hsi_event_counts',
                    custom_generate_series=get_counts_of_HSIs,
                    do_scaling=True
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            result_delayed = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='weather_delayed_hsi_event_counts',
                    custom_generate_series=get_counts_of_HSIs,
                    do_scaling=True
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            all_years_data_cancelled_mean[target_year] = result_cancelled['mean']
            all_years_data_delayed_mean[target_year] = result_delayed['mean']

            all_years_data_cancelled_lower[target_year] = result_cancelled['lower']
            all_years_data_delayed_lower[target_year] = result_delayed['lower']

            all_years_data_cancelled_upper[target_year] = result_cancelled['upper']
            all_years_data_delayed_upper[target_year] = result_delayed['upper']

            # Convert the accumulated data into a DataFrame for plotting
            df_all_years_cancelled_mean = pd.DataFrame(all_years_data_cancelled_mean)
            df_all_years_cancelled_lower = pd.DataFrame(all_years_data_cancelled_lower)
            df_all_years_cancelled_upper = pd.DataFrame(all_years_data_cancelled_upper)
            df_all_years_delayed_mean = pd.DataFrame(all_years_data_delayed_mean)
            df_all_years_delayed_lower = pd.DataFrame(all_years_data_delayed_lower)
            df_all_years_delayed_upper = pd.DataFrame(all_years_data_delayed_upper)


        # save across all scenarios
        cancelled_total = all_years_data_cancelled_mean.mean(axis=1)  # Average across years
        delayed_total = all_years_data_delayed_mean.mean(axis=1)

        all_scenarios_cancelled[scenario_name] = cancelled_total
        all_scenarios_delayed[scenario_name] = delayed_total

    df_cancelled_all_scenarios = pd.DataFrame(all_scenarios_cancelled)
    df_delayed_all_scenarios = pd.DataFrame(all_scenarios_delayed)

    # Plot disrupted appointments for each scenario
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Panel A: Cancelled appointments for each scenario
    df_cancelled_all_scenarios.plot(kind='bar', ax=axes[0], color=scenario_colours[:len(scenario_names)])
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Cancelled HSIs')
    axes[0].legend().set_visible(False)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

    # Panel B: Delayed appointments for each scenario
    df_delayed_all_scenarios.plot(kind='bar', ax=axes[1], color=scenario_colours[:len(scenario_names)])
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Delayed HSIs')
    axes[1].legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / "delayed_and_cancelled_HSIs_all_scenarios.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


    # Compute absolute difference from baseline (first row)
    df_cancelled_diff = df_cancelled_all_scenarios.subtract(df_cancelled_all_scenarios.iloc[0])
    df_delayed_diff = df_delayed_all_scenarios.subtract(df_delayed_all_scenarios.iloc[0])

    # Plot disrupted appointments for each scenario
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Panel A: Cancelled appointments (difference from baseline)
    df_cancelled_diff.plot(kind='bar', ax=axes[0], color=scenario_colours[:len(scenario_names)])
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Cancelled HSIs (difference from baseline)')
    axes[0].legend().set_visible(False)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].grid(True, alpha=0.3)

    # Panel B: Delayed appointments (difference from baseline)
    df_delayed_diff.plot(kind='bar', ax=axes[1], color=scenario_colours[:len(scenario_names)])
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Delayed HSIs (difference from baseline)')
    axes[1].legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_folder / "delayed_and_cancelled_HSIs_all_scenarios_relative_to_baseline.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()


    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
