import argparse
from pathlib import Path
from scipy.stats import ttest_rel

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    get_color_cause_of_death_or_daly_label,
    make_age_grp_lookup,
    summarize,
    get_scenario_info,
    load_pickled_dataframes,
)

min_year = 2025
max_year = 2031
spacing_of_years = 1

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

climate_sensitivity_analysis = False
parameter_sensitivity_analysis = True  # Changed to True
main_text = False  # Changed to False
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
    num_draws = 200  # Number of parameter scan draws
    scenario_names = [f"Draw_{i}" for i in range(num_draws)]
    scenarios_of_interest = range(num_draws)
    suffix = "parameter_SA"
if main_text:
    scenario_names = ["No disruptions", "Baseline", "Worst case"]
    suffix = "main_text"
    scenarios_of_interest = [0, 1, 2]

PREFIX_ON_FILENAME = "1"


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each climate scenario.
    - We estimate the epidemiological impact as the EXTRA DALYs that would occur under climate disruption.
    """

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def get_num_dalys_by_cause_label(_df):
        """Return total number of DALYS (Stacked) by label (total by age-group within the TARGET_PERIOD)"""
        return (
            _df.loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
            .drop(columns=["date", "sex", "age_range", "year"])
            .sum()
        )

    def get_population_for_year(_df):
        """Returns the population in the year of interest"""
        _df["date"] = pd.to_datetime(_df["date"])

        # Filter the DataFrame based on the target period
        filtered_df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=["female", "male"], errors="ignore")
        population_sum = numeric_df.sum(numeric_only=True)

        return population_sum

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    all_draws_dalys_mean = []
    all_draws_dalys_lower = []
    all_draws_dalys_upper = []

    normalized_DALYs = []

    all_draws_dalys_mean_1000 = []
    all_draws_dalys_lower_1000 = []
    all_draws_dalys_upper_1000 = []

    for draw in scenarios_of_interest:
        print(f"Processing draw {draw}...")

        all_years_data_dalys_mean = {}
        all_years_data_dalys_upper = {}
        all_years_data_dalys_lower = {}

        all_years_data_population_mean = {}
        all_years_data_population_lower = {}
        all_years_data_population_upper = {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (
                Date(target_year, 1, 1),
                Date(target_year + spacing_of_years, 12, 31),
            )

            # Absolute Number of DALYs
            result_data_dalys = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthburden",
                    key="dalys_stacked_by_age_and_time",
                    custom_generate_series=get_num_dalys_by_cause_label,
                    do_scaling=True,
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_dalys_mean[target_year] = result_data_dalys['mean']
            all_years_data_dalys_lower[target_year] = result_data_dalys['lower']
            all_years_data_dalys_upper[target_year] = result_data_dalys['upper']

            result_data_population = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.demography",
                    key="population",
                    custom_generate_series=get_population_for_year,
                    do_scaling=True,
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_population_mean[target_year] = result_data_population['mean']
            all_years_data_population_lower[target_year] = result_data_population['lower']
            all_years_data_population_upper[target_year] = result_data_population['upper']

        # Convert the accumulated data into a DataFrame
        df_all_years_DALYS_mean = pd.DataFrame(all_years_data_dalys_mean)
        df_all_years_DALYS_lower = pd.DataFrame(all_years_data_dalys_lower)
        df_all_years_DALYS_upper = pd.DataFrame(all_years_data_dalys_upper)

        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)
        df_all_years_data_population_lower = pd.DataFrame(all_years_data_population_lower)
        df_all_years_data_population_upper = pd.DataFrame(all_years_data_population_upper)

        # Calculate per 1000 rates
        df_daly_per_1000_mean = df_all_years_DALYS_mean.div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000
        df_daly_per_1000_lower = (
            df_all_years_DALYS_lower.div(df_all_years_data_population_lower.iloc[0, 0], axis=0) * 1000
        )
        df_daly_per_1000_upper = (
            df_all_years_DALYS_upper.div(df_all_years_data_population_upper.iloc[0, 0], axis=0) * 1000
        )

        # Store data for cross-draw comparisons
        all_years_data_dalys_mean = df_all_years_DALYS_mean.sum()
        all_years_data_dalys_lower = df_all_years_DALYS_lower.sum()
        all_years_data_dalys_upper = df_all_years_DALYS_upper.sum()

        all_draws_dalys_mean.append(pd.Series(all_years_data_dalys_mean, name=f"Draw {draw}"))
        all_draws_dalys_lower.append(pd.Series(all_years_data_dalys_lower, name=f"Draw {draw}"))
        all_draws_dalys_upper.append(pd.Series(all_years_data_dalys_upper, name=f"Draw {draw}"))

        all_draws_dalys_mean_1000.append(pd.Series(df_daly_per_1000_mean.mean(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_lower_1000.append(pd.Series(df_daly_per_1000_lower.mean(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_upper_1000.append(pd.Series(df_daly_per_1000_upper.mean(axis=1), name=f"Draw {draw}"))

    # Concatenate all draws
    df_dalys_all_draws_mean = pd.concat(all_draws_dalys_mean, axis=1)
    df_dalys_all_draws_lower = pd.concat(all_draws_dalys_lower, axis=1)
    df_dalys_all_draws_upper = pd.concat(all_draws_dalys_upper, axis=1)

    df_dalys_all_draws_mean_1000 = pd.concat(all_draws_dalys_mean_1000, axis=1)
    df_dalys_all_draws_lower_1000 = pd.concat(all_draws_dalys_lower_1000, axis=1)
    df_dalys_all_draws_upper_1000 = pd.concat(all_draws_dalys_upper_1000, axis=1)

    # Save summary data to CSV
    df_dalys_all_draws_mean.to_csv(output_folder / f"dalys_by_cause_all_draws_{suffix}.csv")
    df_dalys_all_draws_mean_1000.to_csv(output_folder / f"dalys_per_1000_by_cause_all_draws_{suffix}.csv")

    # ============================================================================
    # SUMMARY FIGURES FOR ALL DRAWS
    # ============================================================================

    # 1. Total DALYs distribution across draws (box plot by cause)
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Prepare data for box plot
    box_data = []
    box_labels = []
    box_colors = []

    for condition in df_dalys_all_draws_mean.index:
        box_data.append(df_dalys_all_draws_mean.loc[condition].values)
        box_labels.append(condition)
        box_colors.append(get_color_cause_of_death_or_daly_label(condition))

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(
        f"Distribution of Total DALYs Across {len(scenarios_of_interest)} Parameter Draws ({min_year}-{max_year})")
    ax.set_ylabel("Total DALYs")
    ax.set_xlabel("Cause")
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / f"total_dalys_distribution_by_cause_{suffix}.png", dpi=300)
    plt.close(fig)

    # 2. Total DALYs per 1000 distribution (box plot by cause)
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    box_data = []
    box_labels = []
    box_colors = []

    for condition in df_dalys_all_draws_mean_1000.index:
        box_data.append(df_dalys_all_draws_mean_1000.loc[condition].values)
        box_labels.append(condition)
        box_colors.append(get_color_cause_of_death_or_daly_label(condition))

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True)

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(
        f"Distribution of DALYs per 1,000 Across {len(scenarios_of_interest)} Parameter Draws (Mean {min_year}-{max_year})")
    ax.set_ylabel("DALYs per 1,000")
    ax.set_xlabel("Cause")
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / f"dalys_per_1000_distribution_by_cause_{suffix}.png", dpi=300)
    plt.close(fig)

    # 3. Overall total DALYs distribution (all causes summed)
    total_dalys_all_draws = df_dalys_all_draws_mean.sum(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax1.hist(total_dalys_all_draws.values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(total_dalys_all_draws.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {total_dalys_all_draws.mean():.0f}')
    ax1.axvline(total_dalys_all_draws.median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {total_dalys_all_draws.median():.0f}')
    ax1.set_title(f"Distribution of Total DALYs Across {len(scenarios_of_interest)} Draws")
    ax1.set_xlabel("Total DALYs")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Box plot
    ax2.boxplot([total_dalys_all_draws.values], labels=['All Draws'], patch_artist=True)
    ax2.set_title("Total DALYs Summary")
    ax2.set_ylabel("Total DALYs")
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / f"total_dalys_overall_distribution_{suffix}.png", dpi=300)
    plt.close(fig)

    # 4. Summary statistics table
    summary_stats = pd.DataFrame({
        'Mean': df_dalys_all_draws_mean.mean(axis=1),
        'Median': df_dalys_all_draws_mean.median(axis=1),
        'Std': df_dalys_all_draws_mean.std(axis=1),
        'Min': df_dalys_all_draws_mean.min(axis=1),
        'Max': df_dalys_all_draws_mean.max(axis=1),
        'Q25': df_dalys_all_draws_mean.quantile(0.25, axis=1),
        'Q75': df_dalys_all_draws_mean.quantile(0.75, axis=1),
    })
    summary_stats.to_csv(output_folder / f"summary_statistics_dalys_{suffix}.csv")

    summary_stats_1000 = pd.DataFrame({
        'Mean': df_dalys_all_draws_mean_1000.mean(axis=1),
        'Median': df_dalys_all_draws_mean_1000.median(axis=1),
        'Std': df_dalys_all_draws_mean_1000.std(axis=1),
        'Min': df_dalys_all_draws_mean_1000.min(axis=1),
        'Max': df_dalys_all_draws_mean_1000.max(axis=1),
        'Q25': df_dalys_all_draws_mean_1000.quantile(0.25, axis=1),
        'Q75': df_dalys_all_draws_mean_1000.quantile(0.75, axis=1),
    })
    summary_stats_1000.to_csv(output_folder / f"summary_statistics_dalys_per_1000_{suffix}.csv")

    # 5. Stacked area chart showing contribution of each cause (mean across draws)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    mean_dalys_by_cause = df_dalys_all_draws_mean_1000.mean(axis=1).sort_values(ascending=False)
    colors_sorted = [get_color_cause_of_death_or_daly_label(label) for label in mean_dalys_by_cause.index]

    ax.barh(range(len(mean_dalys_by_cause)), mean_dalys_by_cause.values, color=colors_sorted)
    ax.set_yticks(range(len(mean_dalys_by_cause)))
    ax.set_yticklabels(mean_dalys_by_cause.index)
    ax.set_xlabel("Mean DALYs per 1,000 (across all parameter draws)")
    ax.set_title(f"Mean DALYs per 1,000 by Cause (Average across {len(scenarios_of_interest)} draws)")
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / f"mean_dalys_per_1000_by_cause_{suffix}.png", dpi=300)
    plt.close(fig)

    # 6. Coefficient of variation (CV) plot - shows which causes have most variability
    cv_by_cause = (df_dalys_all_draws_mean_1000.std(axis=1) / df_dalys_all_draws_mean_1000.mean(axis=1)) * 100
    cv_by_cause = cv_by_cause.sort_values(ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    colors_cv = [get_color_cause_of_death_or_daly_label(label) for label in cv_by_cause.index]

    ax.barh(range(len(cv_by_cause)), cv_by_cause.values, color=colors_cv, alpha=0.7)
    ax.set_yticks(range(len(cv_by_cause)))
    ax.set_yticklabels(cv_by_cause.index)
    ax.set_xlabel("Coefficient of Variation (%)")
    ax.set_title(f"Parameter Uncertainty by Cause (CV across {len(scenarios_of_interest)} draws)")
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / f"cv_by_cause_{suffix}.png", dpi=300)
    plt.close(fig)

    print(f"Summary figures saved to {output_folder}")
    print(f"\nSummary Statistics:")
    print(f"Total DALYs - Mean: {total_dalys_all_draws.mean():.0f}, Median: {total_dalys_all_draws.median():.0f}")
    print(f"Total DALYs - Range: {total_dalys_all_draws.min():.0f} to {total_dalys_all_draws.max():.0f}")
    print(f"Total DALYs - Std: {total_dalys_all_draws.std():.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
