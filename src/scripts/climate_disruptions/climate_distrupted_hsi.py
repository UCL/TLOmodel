
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
)

min_year = 2020
max_year = 2027
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'

scenario_names = ["Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",  "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]
scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167' ] *4


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the healthcare system utilization across scenarios.
    - We estimate the healthcare system impact through total treatments and never-ran appointments.
    """

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    def get_num_treatments_total(_df):
        """Return the number of treatments in total of all treatments (total within the TARGET_PERIOD)"""
        _df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), 'TREATMENT_ID'].apply(pd.Series).sum()
        _df.index = _df.index.map(lambda x: x.split('_')[0] + "*")
        _df = _df.groupby(level=0).sum().sum()
        return pd.Series(_df)

    def get_num_appts(_df):
        """Return total number of appointments (never ran) within the TARGET_PERIOD"""
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .drop(columns=['date'], errors='ignore') \
            .sum(numeric_only=True)

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

    all_draws_treatments_mean_1000 = []
    all_draws_treatments_lower_1000 = []
    all_draws_treatments_upper_1000 = []

    all_draws_never_ran_mean_1000 = []
    all_draws_never_ran_lower_1000 = []
    all_draws_never_ran_upper_1000 = []

    for draw in range(len(scenario_names)):
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"  # noqa: E731

        all_years_data_treatments_mean = {}
        all_years_data_treatments_upper = {}
        all_years_data_treatments_lower = {}

        all_years_data_never_ran_mean = {}
        all_years_data_never_ran_upper = {}
        all_years_data_never_ran_lower = {}

        all_years_data_population_mean = {}
        all_years_data_population_lower = {}
        all_years_data_population_upper = {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (
                Date(target_year, 1, 1), Date(target_year + spacing_of_years, 12, 31))

            # Extract healthcare system utilization data

            # Total treatments
            num_treatments_total =  summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_num_treatments_total,
                do_scaling=True
            ),
                only_mean=False,
                collapse_columns=True,
            )[draw]
            print(num_treatments_total)
            all_years_data_treatments_mean[target_year] = num_treatments_total['mean']
            all_years_data_treatments_lower[target_year] = num_treatments_total['lower']
            all_years_data_treatments_upper[target_year] = num_treatments_total['upper']

            # Never ran appointments
            num_never_ran_appts = extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Never_ran_HSI_Event',
                custom_generate_series=get_num_appts,
                do_scaling=True
            )
            result_data_never_ran = summarize(
                num_never_ran_appts,
                only_mean=True,
                collapse_columns=True,
            )[draw]

            all_years_data_never_ran_mean[target_year] = result_data_never_ran['mean']
            all_years_data_never_ran_lower[target_year] = result_data_never_ran['lower']
            all_years_data_never_ran_upper[target_year] = result_data_never_ran['upper']

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
        df_all_years_treatments_lower = pd.DataFrame(all_years_data_treatments_lower)
        df_all_years_treatments_upper = pd.DataFrame(all_years_data_treatments_upper)

        df_all_years_never_ran_mean = pd.DataFrame(all_years_data_never_ran_mean)
        df_all_years_never_ran_lower = pd.DataFrame(all_years_data_never_ran_lower)
        df_all_years_never_ran_upper = pd.DataFrame(all_years_data_never_ran_upper)

        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)
        df_all_years_data_population_lower = pd.DataFrame(all_years_data_population_lower)
        df_all_years_data_population_upper = pd.DataFrame(all_years_data_population_upper)

        # Plotting - Healthcare System Utilization
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))

        # Panel A: Total Treatments
        for i, treatment_type in enumerate(df_all_years_treatments_mean.index):
            axes[0].plot(df_all_years_treatments_mean.columns, df_all_years_treatments_mean.loc[treatment_type],
                         marker='o', label=treatment_type)
        axes[0].set_title('Panel A: Healthcare Treatments by Type')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of Treatments')
        axes[0].grid(True)
        axes[0].legend(title='Treatment Type', bbox_to_anchor=(1., 1), loc='upper left')

        # Panel B: Never Ran Appointments
        for i, appt_type in enumerate(df_all_years_never_ran_mean.index):
            axes[1].plot(df_all_years_never_ran_mean.columns, df_all_years_never_ran_mean.loc[appt_type],
                         marker='o', label=appt_type)
        axes[1].set_title('Panel B: Never Ran Appointments by Type')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Number of Never Ran Appointments')
        axes[1].legend(title='Appointment Type', bbox_to_anchor=(1., 1), loc='upper left')
        axes[1].grid(True)

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Healthcare_System_Utilization_All_Years'))
        plt.close(fig)

        # NORMALIZED TO 2020
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))

        df_treatments_normalized_mean = df_all_years_treatments_mean.div(df_all_years_treatments_mean.iloc[:, 0], axis=0)
        df_never_ran_normalized_mean = df_all_years_never_ran_mean.div(df_all_years_never_ran_mean.iloc[:, 0], axis=0)

        df_treatments_normalized_mean.to_csv(output_folder / f"treatments_normalized_2020_{draw}.csv")
        df_never_ran_normalized_mean.to_csv(output_folder / f"never_ran_normalized_2020_{draw}.csv")

        # Panel A: Treatments normalized
        for i, treatment_type in enumerate(df_treatments_normalized_mean.index):
            axes[0].plot(df_treatments_normalized_mean.columns, df_treatments_normalized_mean.loc[treatment_type],
                         marker='o', label=treatment_type)
        axes[0].set_title('Panel A: Healthcare Treatments (Normalized to 2020)')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Fold change compared to 2020')
        axes[0].grid(True)

        # Panel B: Never ran normalized
        for i, appt_type in enumerate(df_never_ran_normalized_mean.index):
            axes[1].plot(df_never_ran_normalized_mean.columns, df_never_ran_normalized_mean.loc[appt_type],
                         marker='o', label=appt_type)
        axes[1].set_title('Panel B: Never Ran Appointments (Normalized to 2020)')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Fold change compared to 2020')
        axes[1].legend(title='Appointment Type', bbox_to_anchor=(1., 1), loc='upper left')
        axes[1].grid(True)

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Healthcare_System_Utilization_Normalized'))
        plt.close(fig)

        # STACKED BAR PLOTS
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))

        df_all_years_treatments_mean.T.plot.bar(stacked=True, ax=axes[0])
        axes[0].set_title('Panel A: Healthcare Treatments by Type')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of Treatments')
        axes[0].legend(title='Treatment Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True)

        df_all_years_never_ran_mean.T.plot.bar(stacked=True, ax=axes[1])
        axes[1].set_title('Panel B: Never Ran Appointments by Type')
        axes[1].set_ylabel('Number of Never Ran Appointments')
        axes[1].set_xlabel('Year')
        axes[1].legend(title='Appointment Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True)

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Healthcare_System_Utilization_Stacked'))
        plt.close(fig)

        # STACKED AREA PLOTS
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))

        # Panel A: Treatments (Stacked area plot)
        years_treatments = df_all_years_treatments_mean.columns
        treatment_types = df_all_years_treatments_mean.index

        axes[0].stackplot(years_treatments, df_all_years_treatments_mean.values, labels=treatment_types)
        axes[0].set_title('Panel A: Healthcare Treatments by Type')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of Treatments')
        axes[0].grid(True)

        # Panel B: Never ran appointments (Stacked area plot)
        years_never_ran = df_all_years_never_ran_mean.columns
        appt_types = df_all_years_never_ran_mean.index

        axes[1].stackplot(years_never_ran, df_all_years_never_ran_mean.values, labels=appt_types)
        axes[1].set_title('Panel B: Never Ran Appointments by Type')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Number of Never Ran Appointments')
        axes[1].legend(title='Appointment Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True)

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Healthcare_System_Utilization_Area'))
        plt.close(fig)

        # PER 1000 POPULATION
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))

        # Calculate per 1000 rates
        population_base = df_all_years_data_population_mean.iloc[0, 0] / 1000  # Convert to per 1000
        df_treatments_per_1000_mean = df_all_years_treatments_mean / population_base
        df_never_ran_per_1000_mean = df_all_years_never_ran_mean / population_base

        # Panel A: Treatments per 1000
        df_treatments_per_1000_mean.T.plot.bar(stacked=True, ax=axes[0])
        axes[0].set_title('Panel A: Healthcare Treatments per 1000 Population')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Treatments per 1000 people')
        axes[0].grid(True)
        axes[0].legend().set_visible(False)

        # Panel B: Never ran per 1000
        df_never_ran_per_1000_mean.T.plot.bar(stacked=True, ax=axes[1])
        axes[1].set_title('Panel B: Never Ran Appointments per 1000 Population')
        axes[1].set_ylabel('Never Ran Appointments per 1000 people')
        axes[1].set_xlabel('Year')
        axes[1].grid(True)
        axes[1].legend(title='Appointment Type', bbox_to_anchor=(1.05, 1), loc='upper left')

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Healthcare_System_Utilization_Per_1000'))
        plt.close(fig)

        # Save data to CSV
        df_all_years_treatments_mean.to_csv(output_folder / f"treatments_by_type_{draw}.csv")
        df_all_years_never_ran_mean.to_csv(output_folder / f"never_ran_by_type_{draw}.csv")

        # Accumulate data across all draws
        all_draws_treatments_mean.append(pd.Series(df_all_years_treatments_mean.sum(), name=f'Draw {draw}'))
        all_draws_never_ran_mean.append(pd.Series(df_all_years_never_ran_mean.sum(), name=f'Draw {draw}'))

        all_draws_treatments_lower.append(pd.Series(df_all_years_treatments_lower.sum(), name=f'Draw {draw}'))
        all_draws_never_ran_lower.append(pd.Series(df_all_years_never_ran_lower.sum(), name=f'Draw {draw}'))

        all_draws_treatments_upper.append(pd.Series(df_all_years_treatments_upper.sum(), name=f'Draw {draw}'))
        all_draws_never_ran_upper.append(pd.Series(df_all_years_never_ran_upper.sum(), name=f'Draw {draw}'))

        # Per 1000 for final year only
        all_draws_treatments_mean_1000.append(pd.Series(df_treatments_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_never_ran_mean_1000.append(pd.Series(df_never_ran_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))

    # Combine all draws
    df_treatments_all_draws_mean = pd.concat(all_draws_treatments_mean, axis=1)
    df_never_ran_all_draws_mean = pd.concat(all_draws_never_ran_mean, axis=1)
    df_treatments_all_draws_lower = pd.concat(all_draws_treatments_lower, axis=1)
    df_never_ran_all_draws_lower = pd.concat(all_draws_never_ran_lower, axis=1)
    df_treatments_all_draws_upper = pd.concat(all_draws_treatments_upper, axis=1)
    df_never_ran_all_draws_upper = pd.concat(all_draws_never_ran_upper, axis=1)

    df_treatments_all_draws_mean_1000 = pd.concat(all_draws_treatments_mean_1000, axis=1)
    df_never_ran_all_draws_mean_1000 = pd.concat(all_draws_never_ran_mean_1000, axis=1)

    # Final summary plots across all scenarios
    treatments_totals_mean = df_treatments_all_draws_mean.sum()
    never_ran_totals_mean = df_never_ran_all_draws_mean.sum()
    treatments_totals_lower = df_treatments_all_draws_lower.sum()
    treatments_totals_upper = df_treatments_all_draws_upper.sum()
    never_ran_totals_lower = df_never_ran_all_draws_lower.sum()
    never_ran_totals_upper = df_never_ran_all_draws_upper.sum()

    treatments_totals_err = np.array([
        treatments_totals_mean - treatments_totals_lower,
        treatments_totals_upper - treatments_totals_mean
    ])

    never_ran_totals_err = np.array([
        never_ran_totals_mean - never_ran_totals_lower,
        never_ran_totals_upper - never_ran_totals_mean
    ])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Panel A: Total Treatments
    axes[0].bar(treatments_totals_mean.index, treatments_totals_mean.values,
                color=scenario_colours, yerr=treatments_totals_err, capsize=20)
    axes[0].set_title(f'Total Healthcare Treatments (2020-{max_year})')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Total Treatments')
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].grid(False)

    # Panel B: Total Never Ran Appointments
    axes[1].bar(never_ran_totals_mean.index, never_ran_totals_mean.values,
                color=scenario_colours, yerr=never_ran_totals_err, capsize=20)
    axes[1].set_title(f'Total Never Ran Appointments (2020-{max_year})')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Total Never Ran Appointments')
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].grid(False)

    fig.tight_layout()
    fig.savefig(output_folder / "total_treatments_and_never_ran_all_draws.png")
    plt.close(fig)

    # Per 1000 in final year
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    df_treatments_all_draws_mean_1000.T.plot.bar(stacked=True, ax=axes[0])
    axes[0].set_title(f'Healthcare Treatments per 1,000 ({max_year})')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Treatments per 1,000')
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].legend().set_visible(False)

    df_never_ran_all_draws_mean_1000.T.plot.bar(stacked=True, ax=axes[1])
    axes[1].set_title(f'Never Ran Appointments per 1,000 ({max_year})')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Never Ran Appointments per 1,000')
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].legend(title='Type', bbox_to_anchor=(1., 1), loc='upper left')

    fig.tight_layout()
    fig.savefig(output_folder / f"treatments_and_never_ran_per_1000_all_draws_{max_year}.png")
    plt.close(fig)

    # Save summary data
    df_treatments_all_draws_mean.to_csv(output_folder / "treatments_summary_all_draws.csv")
    df_never_ran_all_draws_mean.to_csv(output_folder / "never_ran_summary_all_draws.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
