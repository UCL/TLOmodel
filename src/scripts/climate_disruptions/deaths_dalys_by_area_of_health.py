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
    scenario_names = range(0, 9, 1)
    scenarios_of_interest = scenario_names

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

    for draw in range(len(scenario_names)):
        if draw not in scenarios_of_interest:
            continue
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"  # noqa: E731

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

        # Convert the accumulated data into a DataFrame for plotting
        df_all_years_DALYS_mean = pd.DataFrame(all_years_data_dalys_mean)
        df_all_years_DALYS_lower = pd.DataFrame(all_years_data_dalys_lower)
        df_all_years_DALYS_upper = pd.DataFrame(all_years_data_dalys_upper)

        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)
        df_all_years_data_population_lower = pd.DataFrame(all_years_data_population_lower)
        df_all_years_data_population_upper = pd.DataFrame(all_years_data_population_upper)

        # Plotting - Line plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        for i, condition in enumerate(df_all_years_DALYS_mean.index):
            ax.plot(
                df_all_years_DALYS_mean.columns,
                df_all_years_DALYS_mean.loc[condition],
                marker="o",
                label=condition,
                color=[get_color_cause_of_death_or_daly_label(_label) for _label in df_all_years_DALYS_mean.index][i],
            )
        ax.set_title("DALYs by Cause")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of DALYs")
        ax.legend(title="Condition", bbox_to_anchor=(1.0, 1), loc="upper left")
        ax.grid(False)

        fig.savefig(make_graph_file_name("Trend_DALYs_by_condition_All_Years"))
        plt.close(fig)

        # NORMALIZED DALYS - TO 2020
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        df_DALY_normalized_mean = df_all_years_DALYS_mean.div(df_all_years_DALYS_mean.iloc[:, 0], axis=0)
        df_DALY_normalized_mean.to_csv(output_folder / f"cause_of_dalys_normalized_2020_{draw}.csv")

        for i, condition in enumerate(df_DALY_normalized_mean.index):
            ax.plot(
                df_DALY_normalized_mean.columns,
                df_DALY_normalized_mean.loc[condition],
                marker="o",
                label=condition,
                color=[get_color_cause_of_death_or_daly_label(_label) for _label in df_DALY_normalized_mean.index][i],
            )
        ax.set_title("DALYs by Cause (Normalized to 2020)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Fold change in DALYs compared to 2020")
        ax.legend(title="Condition", bbox_to_anchor=(1.0, 1), loc="upper left")
        ax.grid(False)

        fig.tight_layout()
        fig.savefig(make_graph_file_name("Trend_DALYs_by_condition_All_Years_Normalized"))
        plt.close(fig)

        ## BARPLOTS STACKED PER 1000
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        df_daly_per_1000_mean = df_all_years_DALYS_mean.div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000
        df_daly_per_1000_lower = (
            df_all_years_DALYS_lower.div(df_all_years_data_population_lower.iloc[0, 0], axis=0) * 1000
        )
        df_daly_per_1000_upper = (
            df_all_years_DALYS_upper.div(df_all_years_data_population_upper.iloc[0, 0], axis=0) * 1000
        )

        # DALYs (Stacked bar plot)
        df_daly_per_1000_mean.T.plot.bar(
            stacked=True,
            ax=ax,
            color=[get_color_cause_of_death_or_daly_label(_label) for _label in df_daly_per_1000_mean.index],
            label=[label for label in df_daly_per_1000_mean.index],
        )
        ax.axhline(0.0, color="black")
        ax.set_title("DALYs by Cause")
        ax.set_ylabel("Number of DALYs per 1000 people")
        ax.set_xlabel("Year")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.tight_layout()
        fig.savefig(make_graph_file_name("Trend_DALYs_by_condition_All_Years_Stacked_Rate"))
        plt.close(fig)

        # save cause of DALYs to csv
        df_all_years_DALYS_mean.to_csv(output_folder / f"dalys_by_cause_rate_{draw}.csv")

        normalized_DALYs.append(pd.Series(df_DALY_normalized_mean.iloc[:, -1], name=f"Draw {draw}"))
        print(df_all_years_DALYS_mean)
        all_years_data_dalys_mean = df_all_years_DALYS_mean.sum()
        print(all_years_data_dalys_mean)
        all_years_data_dalys_lower = df_all_years_DALYS_lower.sum()
        all_years_data_dalys_upper = df_all_years_DALYS_upper.sum()
        all_draws_dalys_mean.append(pd.Series(all_years_data_dalys_mean, name=f"Draw {draw}"))
        all_draws_dalys_lower.append(pd.Series(all_years_data_dalys_lower, name=f"Draw {draw}"))
        all_draws_dalys_upper.append(pd.Series(all_years_data_dalys_upper, name=f"Draw {draw}"))

        print(df_daly_per_1000_mean)
        all_draws_dalys_mean_1000.append(pd.Series(df_daly_per_1000_mean.mean(axis=1), name=f"Draw {draw}"))
        print(all_draws_dalys_mean_1000)
        all_draws_dalys_lower_1000.append(pd.Series(df_daly_per_1000_lower.mean(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_upper_1000.append(pd.Series(df_daly_per_1000_upper.mean(axis=1), name=f"Draw {draw}"))

    df_dalys_all_draws_mean = pd.concat(all_draws_dalys_mean, axis=1)
    df_dalys_all_draws_lower = pd.concat(all_draws_dalys_lower, axis=1)
    df_dalys_all_draws_upper = pd.concat(all_draws_dalys_upper, axis=1)

    df_dalys_all_draws_mean_1000 = pd.concat(all_draws_dalys_mean_1000, axis=1)
    df_dalys_all_draws_lower_1000 = pd.concat(all_draws_dalys_lower_1000, axis=1)
    df_dalys_all_draws_upper_1000 = pd.concat(all_draws_dalys_upper_1000, axis=1)

    # Compute mean, lower, and upper total DALYs for each draw
    total_dalys_mean = df_dalys_all_draws_mean.sum(axis=0)
    total_dalys_lower = df_dalys_all_draws_lower.sum(axis=0)
    total_dalys_upper = df_dalys_all_draws_upper.sum(axis=0)

    # Baseline reference (first draw)
    baseline_mean = total_dalys_mean.iloc[0]
    baseline_lower = total_dalys_lower.iloc[0]
    baseline_upper = total_dalys_upper.iloc[0]

    # Compute percentage change relative to baseline
    mean_change = ((total_dalys_mean - baseline_mean) / baseline_mean) * 100
    lower_change = ((total_dalys_lower - baseline_upper) / baseline_upper) * 100
    upper_change = ((total_dalys_upper - baseline_lower) / baseline_lower) * 100

    # Drop baseline (since change = 0)
    mean_change = mean_change.iloc[1:]
    lower_change = lower_change.iloc[1:]
    upper_change = upper_change.iloc[1:]

    # Compute CI error bars
    yerr_lower = mean_change - lower_change
    yerr_upper = upper_change - mean_change
    yerr = np.vstack([yerr_lower, yerr_upper])

    # Plot with CI error bars
    if main_text:
        fig, ax = plt.subplots(figsize=(12, 6))
        mean_change.plot(
            kind="bar",
            color=scenario_colours[1: len(mean_change) + 1],
            ax=ax,
            yerr=yerr,
            capsize=5,
            error_kw=dict(linewidth=1, alpha=0.8),
        )

        ax.set_title("Percentage Change in Total DALYs (with 95% CI)")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Percentage change in DALYs")
        ax.set_xticklabels(scenario_names[1:], rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=1)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(output_folder / f"relative_change_in_total_DALYs_across_draws_with_CI_{suffix}.png")
        plt.close(fig)

    # Plotting as bar charts
    dalys_totals_mean = df_dalys_all_draws_mean.sum()
    dalys_totals_lower = df_dalys_all_draws_lower.sum()
    dalys_totals_upper = df_dalys_all_draws_upper.sum()
    dalys_totals_err = np.array([dalys_totals_mean - dalys_totals_lower, dalys_totals_upper - dalys_totals_mean])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    width = 0.35

    # Total DALYs
    x = np.arange(len(dalys_totals_mean.index))
    ax.bar(x, dalys_totals_mean.values, width, color=scenario_colours, yerr=dalys_totals_err, capsize=6)
    ax.set_title(f"Total DALYs ({min_year}-{max_year})")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Total DALYs")
    ax.set_xticks(x)
    if climate_sensitivity_analysis:
        ax.set_xticks(ax.get_xticks(), labels=scenario_names, rotation=45, ha='right')
    else:
        ax.set_xticklabels(scenario_names)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_folder / f"total_dalys_all_draws_{suffix}.png")
    plt.close(fig)

    # Scatter plot across all causes
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    x_positions = np.arange(len(df_dalys_all_draws_mean_1000.columns))
    jitter_strength = 0.05

    # DALYs
    for i, condition in enumerate(df_dalys_all_draws_mean_1000.index):
        colour = get_color_cause_of_death_or_daly_label(condition)
        y_means = df_dalys_all_draws_mean_1000.loc[condition].values
        y_lower = df_dalys_all_draws_lower_1000.loc[condition].values
        y_upper = df_dalys_all_draws_upper_1000.loc[condition].values

        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(x_positions))
        x_jittered = x_positions + jitter
        ax.scatter(x_jittered, y_means, color=colour, s=50)
        ax.errorbar(
            x_jittered,
            y_means,
            yerr=[y_means - y_lower, y_upper - y_means],
            fmt="none",
            ecolor=colour,
            capsize=3,
            alpha=0.7,
        )
        ax.plot(x_jittered, y_means, color=colour, linestyle="-", alpha=0.5)

    ax.set_title(f"DALYs per 1,000 ({max_year})")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_dalys_all_draws_mean_1000.columns)
    ax.set_xlabel("Draw")
    ax.set_ylabel("DALYs per 1,000")
    ax.legend(df_dalys_all_draws_mean_1000.index, title="Cause", bbox_to_anchor=(1.0, 1), loc="upper left")
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(output_folder / f"dalys_per_1000_all_cause_all_draws_{max_year}_{suffix}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
