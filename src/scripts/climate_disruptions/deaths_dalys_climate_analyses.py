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


min_year = 2026
max_year = 2041
spacing_of_years = 1
scenario_names_all = [
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
climate_sensitivity_analysis = True
parameter_sensitivity_analysis = False
main_text = True
mode_2 = False
if climate_sensitivity_analysis:

    suffix = "climate_SA"
    scenarios_of_interest = range(len(scenario_names_all) -1)
    scenario_names = scenario_names_all[1:]
if parameter_sensitivity_analysis:
    scenario_names = range(0, 9, 1)
    scenarios_of_interest = scenario_names

    suffix = "parameter_SA"
if main_text:
    scenario_names = [
        "Baseline",
        "SSP 2.45 Mean",
    ]
    suffix = "main_text"
    scenarios_of_interest = [0, 1]

if mode_2:
    scenario_names = [
        "Baseline",
        "SSP 5.85 Mean",
    ]
    suffix = "mode_2"
    scenarios_of_interest = [0, 1]

PREFIX_ON_FILENAME = "1"


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def get_num_deaths_by_cause_label(_df):
        """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)"""
        return _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)].groupby(_df["label"]).size()

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
    all_draws_deaths_mean = []
    all_draws_deaths_lower = []
    all_draws_deaths_upper = []

    all_draws_dalys_mean = []
    all_draws_dalys_lower = []
    all_draws_dalys_upper = []

    normalized_DALYs = []
    all_draws_deaths_mean_1000 = []
    all_draws_deaths_lower_1000 = []
    all_draws_deaths_upper_1000 = []

    all_draws_dalys_mean_1000 = []
    all_draws_dalys_lower_1000 = []
    all_draws_dalys_upper_1000 = []
    for draw in range(len(scenario_names_all)):
        if draw not in scenarios_of_interest:
            continue
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"  # noqa: E731

        all_years_data_deaths_mean = {}
        all_years_data_deaths_upper = {}
        all_years_data_deaths_lower = {}

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
            )  # Corrected the year range to cover 5 years.

            # %% Quantify the health gains associated with all interventions combined.

            # Absolute Number of Deaths and DALYs
            result_data_deaths = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.demography",
                    key="death",
                    custom_generate_series=get_num_deaths_by_cause_label,
                    do_scaling=True,
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            all_years_data_deaths_mean[target_year] = result_data_deaths["mean"]
            all_years_data_deaths_lower[target_year] = result_data_deaths["lower"]
            all_years_data_deaths_upper[target_year] = result_data_deaths["upper"]

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
            all_years_data_dalys_mean[target_year] = result_data_dalys["mean"]
            all_years_data_dalys_lower[target_year] = result_data_dalys["lower"]
            all_years_data_dalys_upper[target_year] = result_data_dalys["upper"]

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
            all_years_data_population_mean[target_year] = result_data_population["mean"]
            all_years_data_population_lower[target_year] = result_data_population["lower"]
            all_years_data_population_upper[target_year] = result_data_population["upper"]

        # Convert the accumulated data into a DataFrame for plotting
        df_all_years_DALYS_mean = pd.DataFrame(all_years_data_dalys_mean)
        df_all_years_DALYS_lower = pd.DataFrame(all_years_data_dalys_lower)
        df_all_years_DALYS_upper = pd.DataFrame(all_years_data_dalys_upper)
        df_all_years_deaths_mean = pd.DataFrame(all_years_data_deaths_mean)
        df_all_years_deaths_lower = pd.DataFrame(all_years_data_deaths_lower)
        df_all_years_deaths_upper = pd.DataFrame(all_years_data_deaths_upper)

        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)
        df_all_years_data_population_lower = pd.DataFrame(all_years_data_population_lower)
        df_all_years_data_population_upper = pd.DataFrame(all_years_data_population_upper)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side
        print(df_all_years_deaths_mean.index)
        # Panel A: Deaths
        for i, condition in enumerate(df_all_years_deaths_mean.index):
            axes[0].plot(
                df_all_years_deaths_mean.columns,
                df_all_years_deaths_mean.loc[condition],
                marker="o",
                label=condition,
                color=[get_color_cause_of_death_or_daly_label(_label) for _label in df_all_years_deaths_mean.index][i],
            )
        axes[0].set_title("Panel A: Deaths by Cause")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Number of deaths")
        axes[0].grid(False)
        # Panel B: DALYs
        for i, condition in enumerate(df_all_years_DALYS_mean.index):
            axes[1].plot(
                df_all_years_DALYS_mean.columns,
                df_all_years_DALYS_mean.loc[condition],
                marker="o",
                label=condition,
                color=[get_color_cause_of_death_or_daly_label(_label) for _label in df_all_years_DALYS_mean.index][i],
            )
        axes[1].set_title("Panel B: DALYs by cause")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Number of DALYs")
        axes[1].legend(title="Condition", bbox_to_anchor=(1.0, 1), loc="upper left")
        axes[1].grid()

        fig.savefig(make_graph_file_name("Trend_Deaths_and_DALYs_by_condition_All_Years_Panel_A_and_B"))
        plt.close(fig)

        # NORMALIZED DEATHS AND DALYS - TO 2020
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side

        df_death_normalized_mean = df_all_years_deaths_mean.div(df_all_years_deaths_mean.iloc[:, 0], axis=0)
        df_DALY_normalized_mean = df_all_years_DALYS_mean.div(df_all_years_DALYS_mean.iloc[:, 0], axis=0)
        df_death_normalized_mean.to_csv(output_folder / f"cause_of_death_normalized_2020_{draw}.csv")
        df_DALY_normalized_mean.to_csv(output_folder / f"cause_of_dalys_normalized_2020_{draw}.csv")
        for i, condition in enumerate(df_death_normalized_mean.index):
            axes[0].plot(
                df_death_normalized_mean.columns,
                df_death_normalized_mean.loc[condition],
                marker="o",
                label=condition,
                color=[get_color_cause_of_death_or_daly_label(_label) for _label in df_all_years_deaths_mean.index][i],
            )
        axes[0].set_title("Panel A: Deaths by Cause")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Fold change in deaths compared to 2020")
        axes[0].grid()

        # Panel B: DALYs
        for i, condition in enumerate(df_DALY_normalized_mean.index):
            axes[1].plot(
                df_DALY_normalized_mean.columns,
                df_DALY_normalized_mean.loc[condition],
                marker="o",
                label=condition,
                color=[get_color_cause_of_death_or_daly_label(_label) for _label in df_DALY_normalized_mean.index][i],
            )
        axes[1].set_title("Panel B: DALYs by cause")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Fold change in DALYs compared to 2020")
        axes[1].legend(title="Condition", bbox_to_anchor=(1.0, 1), loc="upper left")
        axes[1].grid()

        fig.tight_layout()
        fig.savefig(make_graph_file_name("Trend_Deaths_and_DALYs_by_condition_All_Years_Normalized_Panel_A_and_B"))
        plt.close(fig)

        ## BARPLOTS STACKED PER 1000
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side
        df_death_per_1000_mean = (
            df_all_years_deaths_mean.div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000
        )
        df_daly_per_1000_mean = df_all_years_DALYS_mean.div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000
        df_death_per_1000_lower = (
            df_all_years_deaths_lower.div(df_all_years_data_population_lower.iloc[0, 0], axis=0) * 1000
        )
        df_daly_per_1000_lower = (
            df_all_years_DALYS_lower.div(df_all_years_data_population_lower.iloc[0, 0], axis=0) * 1000
        )
        df_death_per_1000_upper = (
            df_all_years_deaths_upper.div(df_all_years_data_population_upper.iloc[0, 0], axis=0) * 1000
        )
        df_daly_per_1000_upper = (
            df_all_years_DALYS_upper.div(df_all_years_data_population_upper.iloc[0, 0], axis=0) * 1000
        )

        # Panel A: Deaths (Stacked bar plot)
        df_death_per_1000_mean.T.plot.bar(
            stacked=True,
            ax=axes[0],
            color=[get_color_cause_of_death_or_daly_label(_label) for _label in df_death_per_1000_mean.index],
        )
        axes[0].set_title("Panel A: Deaths by Cause")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Number of deaths per 1000 people")
        axes[0].grid()
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].legend().set_visible(False)

        # Panel B: DALYs (Stacked bar plot)
        df_daly_per_1000_mean.T.plot.bar(
            stacked=True,
            ax=axes[1],
            color=[get_color_cause_of_death_or_daly_label(_label) for _label in df_daly_per_1000_mean.index],
            label=[label for label in df_daly_per_1000_mean.index],
        )
        axes[1].axhline(0.0, color="black")
        axes[1].set_title("Panel B: DALYs")
        axes[1].set_ylabel("Number of DALYs per 1000 people")
        axes[1].set_xlabel("Year")
        axes[1].grid()
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].legend(ncol=3, fontsize=8, loc="upper right")
        axes[1].legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.tight_layout()
        fig.savefig(make_graph_file_name("Trend_Deaths_and_DALYs_by_condition_All_Years_Panel_A_and_B_Stacked_Rate"))

        # save cause of death to csv
        df_all_years_DALYS_mean.to_csv(output_folder / f"dalys_by_cause_rate_{draw}.csv")
        df_all_years_deaths_mean.to_csv(output_folder / f"deaths_by_cause_rate_{draw}.csv")

        normalized_DALYs.append(pd.Series(df_DALY_normalized_mean.iloc[:, -1], name=f"Draw {draw}"))
        print(df_all_years_DALYS_mean)
        all_years_data_dalys_mean = df_all_years_DALYS_mean.sum()
        print(all_years_data_dalys_mean)
        all_years_data_deaths_mean = df_all_years_deaths_mean.sum()
        all_years_data_dalys_lower = df_all_years_DALYS_lower.sum()
        all_years_data_deaths_lower = df_all_years_deaths_lower.sum()
        all_years_data_dalys_upper = df_all_years_DALYS_upper.sum()
        all_years_data_deaths_upper = df_all_years_deaths_upper.sum()
        all_draws_deaths_mean.append(pd.Series(all_years_data_deaths_mean, name=f"Draw {draw}"))
        all_draws_dalys_mean.append(pd.Series(all_years_data_dalys_mean, name=f"Draw {draw}"))
        all_draws_deaths_lower.append(pd.Series(all_years_data_deaths_lower, name=f"Draw {draw}"))
        all_draws_dalys_lower.append(pd.Series(all_years_data_dalys_lower, name=f"Draw {draw}"))
        all_draws_deaths_upper.append(pd.Series(all_years_data_deaths_upper, name=f"Draw {draw}"))
        all_draws_dalys_upper.append(pd.Series(all_years_data_dalys_upper, name=f"Draw {draw}"))
        # only include 2070 as can't have cumulative per 1000?
        print(df_daly_per_1000_mean)
        all_draws_dalys_mean_1000.append(pd.Series(df_daly_per_1000_mean.mean(axis=1), name=f"Draw {draw}"))
        print(all_draws_dalys_mean_1000)
        all_draws_dalys_lower_1000.append(pd.Series(df_daly_per_1000_lower.mean(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_upper_1000.append(pd.Series(df_daly_per_1000_upper.mean(axis=1), name=f"Draw {draw}"))
        all_draws_deaths_mean_1000.append(pd.Series(df_death_per_1000_mean.mean(axis=1), name=f"Draw {draw}"))
        all_draws_deaths_lower_1000.append(pd.Series(df_death_per_1000_lower.mean(axis=1), name=f"Draw {draw}"))
        all_draws_deaths_upper_1000.append(pd.Series(df_death_per_1000_upper.mean(axis=1), name=f"Draw {draw}"))

    df_deaths_all_draws_mean = pd.concat(all_draws_deaths_mean, axis=1)
    df_dalys_all_draws_mean = pd.concat(all_draws_dalys_mean, axis=1)
    df_deaths_all_draws_lower = pd.concat(all_draws_deaths_lower, axis=1)
    df_dalys_all_draws_lower = pd.concat(all_draws_dalys_lower, axis=1)
    df_deaths_all_draws_upper = pd.concat(all_draws_deaths_upper, axis=1)
    df_dalys_all_draws_upper = pd.concat(all_draws_dalys_upper, axis=1)

    df_deaths_all_draws_mean_1000 = pd.concat(all_draws_deaths_mean_1000, axis=1)
    df_dalys_all_draws_mean_1000 = pd.concat(all_draws_dalys_mean_1000, axis=1)
    df_deaths_all_draws_lower_1000 = pd.concat(all_draws_deaths_lower_1000, axis=1)
    df_dalys_all_draws_lower_1000 = pd.concat(all_draws_dalys_lower_1000, axis=1)
    df_deaths_all_draws_upper_1000 = pd.concat(all_draws_deaths_upper_1000, axis=1)
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
            color=scenario_colours[1 : len(mean_change) + 1],
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
    deaths_totals_mean = df_deaths_all_draws_mean.sum()
    dalys_totals_mean = df_dalys_all_draws_mean.sum()
    deaths_totals_lower = df_deaths_all_draws_lower.sum()
    deaths_totals_upper = df_deaths_all_draws_upper.sum()
    dalys_totals_lower = df_dalys_all_draws_lower.sum()
    dalys_totals_upper = df_dalys_all_draws_upper.sum()
    deaths_totals_err = np.array([deaths_totals_mean - deaths_totals_lower, deaths_totals_upper - deaths_totals_mean])
    dalys_totals_err = np.array([dalys_totals_mean - dalys_totals_lower, dalys_totals_upper - dalys_totals_mean])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    width = 0.35

    # --------------------------
    # Panel A: Total Deaths
    # --------------------------
    x = np.arange(len(deaths_totals_mean.index))
    axes[0].bar(x, deaths_totals_mean.values, width, color=scenario_colours, yerr=deaths_totals_err, capsize=6)
    axes[0].text(-0.0, 1.05, "(A)", transform=axes[0].transAxes, fontsize=14, va="top", ha="right")
    axes[0].set_title(f"Total Deaths (2025-{max_year})")
    axes[0].set_xlabel("Scenario")
    axes[0].set_ylabel("Total Deaths")
    axes[0].set_xticks(x)
    if climate_sensitivity_analysis:
        axes[0].set_xticks(axes[0].get_xticks(), labels=scenario_names, rotation=45, ha='right')
    else:
        axes[0].set_xticklabels(scenario_names)
    axes[0].grid(False)

    # --------------------------
    # Panel B: Total DALYs
    # --------------------------
    x = np.arange(len(dalys_totals_mean.index))
    axes[1].bar(x, dalys_totals_mean.values, width, color=scenario_colours, yerr=dalys_totals_err, capsize=6)
    axes[1].text(-0.0, 1.05, "(B)", transform=axes[1].transAxes, fontsize=14, va="top", ha="right")
    axes[1].set_title(f"Total DALYs (2025-{max_year})")
    axes[1].set_xlabel("Scenario")
    axes[1].set_ylabel("Total DALYs")
    axes[1].set_xticks(x)
    if climate_sensitivity_analysis:
        axes[1].set_xticks(axes[1].get_xticks(), labels=scenario_names, rotation=45, ha='right')
    else:
        axes[1].set_xticklabels(scenario_names)
    axes[1].grid(False)

    fig.tight_layout()
    fig.savefig(output_folder / f"total_deaths_and_dalys_all_draws_{suffix}.png")
    plt.close(fig)
    # Scatter plot across all causes

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    x_positions = np.arange(len(df_deaths_all_draws_mean_1000.columns))
    jitter_strength = 0.05

    # --- Deaths ---
    for i, condition in enumerate(df_deaths_all_draws_mean_1000.index):
        colour = get_color_cause_of_death_or_daly_label(condition)
        y_means = df_deaths_all_draws_mean_1000.loc[condition].values
        y_lower = df_deaths_all_draws_lower_1000.loc[condition].values
        y_upper = df_deaths_all_draws_upper_1000.loc[condition].values

        # Jitter for scatter
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(x_positions))
        x_jittered = x_positions + jitter
        axes[0].scatter(x_jittered, y_means, color=colour, s=50)
        axes[0].errorbar(
            x_jittered,
            y_means,
            yerr=[y_means - y_lower, y_upper - y_means],
            fmt="none",
            ecolor=colour,
            capsize=3,
            alpha=0.7,
        )
        axes[0].plot(x_jittered, y_means, color=colour, linestyle="-", alpha=0.5)

        # --- t-test for significance ---

    axes[0].set_title(f"Deaths per 1,000 ({max_year})")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(df_deaths_all_draws_mean_1000.columns)
    axes[0].set_xlabel("Draw")
    axes[0].set_ylabel("Deaths per 1,000")

    # --- DALYs ---
    for i, condition in enumerate(df_dalys_all_draws_mean_1000.index):
        colour = get_color_cause_of_death_or_daly_label(condition)
        y_means = df_dalys_all_draws_mean_1000.loc[condition].values
        y_lower = df_dalys_all_draws_lower_1000.loc[condition].values
        y_upper = df_dalys_all_draws_upper_1000.loc[condition].values

        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(x_positions))
        x_jittered = x_positions + jitter
        axes[1].scatter(x_jittered, y_means, color=colour, s=50)
        axes[1].errorbar(
            x_jittered,
            y_means,
            yerr=[y_means - y_lower, y_upper - y_means],
            fmt="none",
            ecolor=colour,
            capsize=3,
            alpha=0.7,
        )
        axes[1].plot(x_jittered, y_means, color=colour, linestyle="-", alpha=0.5)

    axes[1].set_title(f"DALYS per 1,000 ({max_year})")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(df_dalys_all_draws_mean_1000.columns)
    axes[1].set_xlabel("Draw")
    axes[1].set_ylabel("DALYS per 1,000")
    axes[1].legend(df_dalys_all_draws_mean_1000.index, title="Cause", bbox_to_anchor=(1.0, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(output_folder / f"deaths_and_dalys_per_1000_all_cause_all_draws_{max_year}_{suffix}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
