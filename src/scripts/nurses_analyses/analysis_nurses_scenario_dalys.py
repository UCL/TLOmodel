"""Plot DALYs and Deaths across nurse staffing scenarios.

This script produces two figures for the Nurse Shortages analysis:

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario
from tlo.analysis.utils import extract_results, load_pickled_dataframes, summarize


DALY_METADATA_COLUMNS = {"date", "year", "sex", "age_range", "li_wealth", "district_of_residence"}


def find_difference_relative_to_comparison_series(
    _ser: pd.Series,
    comparison: str,
    scaled: bool = False,
    drop_comparison: bool = True,
):
    return (
        _ser
        .unstack(level=0)
        .apply(
            lambda x: (
                (x - x[comparison]) /
                (x[comparison] if scaled else 1.0)
            ),
            axis=1,
        )
        .drop(
            columns=([comparison] if drop_comparison else [])
        )
        .stack()
    )


def find_difference_relative_to_comparison_series_dataframe(
    _df: pd.DataFrame,
    **kwargs,
):
    return pd.concat(
        {
            idx: find_difference_relative_to_comparison_series(
                row,
                **kwargs,
            )
            for idx, row in _df.iterrows()
        },
        axis=1,
    ).T


def set_param_names_as_column_index_level_0(_df, param_names):
    """Set column index level 0 (draw numbers) to scenario names."""
    ordered_param_names = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [
        ordered_param_names.get(col)
        for col in _df.columns.levels[0]
    ]
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


def extract_annual_dalys(results_folder):

    def get_num_dalys_yearly(df: pd.DataFrame) -> pd.Series:
        """Return total DALYs for each year."""

        # Add year if it isn't already present
        if "year" not in df.columns:
            df = df.assign(year=df["date"].dt.year)

        cause_cols = [
            c
            for c in df.columns
            if c not in DALY_METADATA_COLUMNS
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        yearly = (
            df.groupby("year")[cause_cols]
              .sum()
              .sum(axis=1)
        )

        return yearly

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_num_dalys_yearly,
        do_scaling=True,
    )


# Extract annual Deaths
def extract_annual_deaths(results_folder):
    def get_num_deaths_yearly(df: pd.DataFrame) -> pd.Series:
        """Return total deaths for each year."""
        yearly = (
            df.assign(year=df["date"].dt.year)
            .groupby("year")["person_id"]
            .count()
        )
        return yearly

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_num_deaths_yearly,
        do_scaling=True,
    )


# Plot: Annual DALYs over time
def plot_annual_dalys(summarized_annual_dalys, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_names = summarized_annual_dalys.columns.get_level_values(0).unique()

    # Short labels for legend
    label_map = {
        "Baseline Nurses / Default Healthsystem Function": "Baseline",
        "Fewer Nurses / Default Healthsystem Function": "Fewer nurses",
        "More Nurses / Default Healthsystem Function": "More nurses",
        "More CNP staff / Default Healthsystem Function": "More CNP",
        "More Nurses by District / Default Healthsystem Function": "More nurses by district",
        "More CNP staff by District / Default Healthsystem Function": "More CNP by district",

        "Baseline Nurses / Improved Healthsystem Function": "Baseline",
        "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
        "More Nurses / Improved Healthsystem Function": "More nurses",
        "More CNP staff / Improved Healthsystem Function": "More CNP",
        "More Nurses by District / Improved Healthsystem Function": "More nurses by district",
        "More CNP staff by District / Improved Healthsystem Function": "More CNP by district",
    }

    color_map = {
        "Baseline Nurses / Default Healthsystem Function": "black",
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function": "orange",

        "Baseline Nurses / Improved Healthsystem Function": "black",
        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function": "orange",
    }

    for scenario in scenario_names:
        years = summarized_annual_dalys.index.astype(int)
        means = summarized_annual_dalys[(scenario, "mean")].values
        lowers = summarized_annual_dalys[(scenario, "lower")].values
        uppers = summarized_annual_dalys[(scenario, "upper")].values

        color = color_map.get(scenario, "gray")
        ax.plot(years, means, linewidth=2, color=color, label=label_map.get(scenario, scenario))
        ax.fill_between(years, lowers, uppers, color=color, alpha=0.2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual DALYs")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(2025, 2034)
    ax.set_ylim(bottom=8e6)
    # ax.set_ylim(bottom=0.8)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    # fig.suptitle(title, fontsize=14)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, ax


# Plot: Annual Deaths over time
def plot_annual_deaths(summarized_annual_deaths, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_names = (
        summarized_annual_deaths.columns
        .get_level_values(0)
        .unique()
    )

    label_map = {
        "Baseline Nurses / Default Healthsystem Function": "Baseline",
        "Fewer Nurses / Default Healthsystem Function": "Fewer nurses",
        "More Nurses / Default Healthsystem Function": "More nurses",
        "More CNP staff / Default Healthsystem Function": "More CNP",
        "More Nurses by District / Default Healthsystem Function": "More nurses by district",
        "More CNP staff by District / Default Healthsystem Function": "More CNP by district",

        "Baseline Nurses / Improved Healthsystem Function": "Baseline",
        "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
        "More Nurses / Improved Healthsystem Function": "More nurses",
        "More CNP staff / Improved Healthsystem Function": "More CNP",
        "More Nurses by District / Improved Healthsystem Function": "More nurses by district",
        "More CNP staff by District / Improved Healthsystem Function": "More CNP by district",
    }

    color_map = {
        "Baseline Nurses / Default Healthsystem Function": "black",
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function": "orange",

        "Baseline Nurses / Improved Healthsystem Function": "black",
        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function": "orange",
    }

    for scenario in scenario_names:
        years = summarized_annual_deaths.index.astype(int)
        means = summarized_annual_deaths[(scenario, "mean")].values
        lowers = summarized_annual_deaths[(scenario, "lower")].values

        uppers = summarized_annual_deaths[(scenario, "upper")].values

        color = color_map.get(scenario, "gray")
        ax.plot(years, means, linewidth=2, color=color, label=label_map.get(scenario, scenario))
        ax.fill_between(years, lowers, uppers, color=color, alpha=0.2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual deaths")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(2025, 2034)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    # fig.suptitle(title, fontsize=14)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, ax


# Extract deaths by cause
def extract_deaths_by_cause(results_folder):
    def get_deaths_by_cause(df: pd.DataFrame) -> pd.Series:
        """
        Return deaths by cause aggregated across 2027–2034.
        """
        # Add year
        df = df.assign(year=df["date"].dt.year)
        # Restrict years
        df = df[df["year"].between(2027, 2034)]
        # Changed to "label" in order to capture group causes
        # cause_col = "cause"
        cause_col = "label"
        deaths_by_cause = (df.groupby(cause_col)["person_id"].count())
        return deaths_by_cause

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_deaths_by_cause,
        do_scaling=True,
    )


# Extract deaths by age group
def extract_deaths_by_age_group(results_folder):

    def get_deaths_by_age_group(df: pd.DataFrame) -> pd.Series:
        """
        Return deaths by age group aggregated across 2027–2034.
        """
        df = df.assign(year=df["date"].dt.year)
        df = df[df["year"].between(2027, 2034)]

        # Create age groups
        age_bins = [
            0, 5, 10, 15, 20, 25, 30, 35,
            40, 45, 50, 55, 60, 65, 70,
            75, 80, np.inf
        ]

        age_labels = [
            "0-4",
            "5-9",
            "10-14",
            "15-19",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80+",
        ]

        df["age_group"] = pd.cut(
            df["age"],
            bins=age_bins,
            labels=age_labels,
            right=False,
        )
        # Aggregate deaths by age group
        deaths_by_age = (df.groupby("age_group")["person_id"].count())
        return deaths_by_age

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_deaths_by_age_group,
        do_scaling=True,
    )


# Extract DALYs by cause
def extract_dalys_by_cause(results_folder):
    def get_dalys_by_cause(df: pd.DataFrame) -> pd.Series:
        """
        Return DALYs by cause aggregated across 2027–2034.
        """
        df = df.assign(year=df["date"].dt.year)
        df = df[df["year"].between(2027, 2034)]
        # Removing metadata columns
        cause_cols = [
            c for c in df.columns
            if c not in DALY_METADATA_COLUMNS
               and pd.api.types.is_numeric_dtype(df[c])
        ]
        # Sum DALYs for each cause
        return df[cause_cols].sum()

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_dalys_by_cause,
        do_scaling=True,
    )


# Extract DALYs by age group
def extract_dalys_by_age_group(results_folder):

    def get_dalys_by_age_group(df: pd.DataFrame) -> pd.Series:
        """
        Return DALYs by age group aggregated across 2027–2034.
        """
        df = df.assign(year=df["date"].dt.year)
        df = df[df["year"].between(2027, 2034)]

        cause_cols = [
            c for c in df.columns
            if c not in DALY_METADATA_COLUMNS
               and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Sum DALYs across causes first
        df["total_dalys"] = df[cause_cols].sum(axis=1)
        # Aggregating by age group
        dalys_by_age = (
            df.groupby("age_range")["total_dalys"]
            .sum()
        )
        return dalys_by_age

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_dalys_by_age_group,
        do_scaling=True,
    )


# Plot: Percent DALYs averted relative to baseline (2027–2034)
def calculate_percent_dalys_averted(
    annual_dalys,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate % DALYs averted using run-to-run differences.
    """
    years = annual_dalys.index.astype(int)
    year_mask = np.isin(years, list(comparison_years))

    annual_dalys = annual_dalys.loc[year_mask]
    annual_dalys_agg = annual_dalys.sum(axis=0)

    pct_diff = pd.DataFrame(
        -100.0
        * find_difference_relative_to_comparison_series(
            annual_dalys_agg,
            comparison=baseline_scenario,
            scaled=True,
        )
    ).T

    summarized = summarize(pct_diff)
    results = {}

    scenario_names = (
        summarized.columns
        .get_level_values(0)
        .unique()
    )

    for scenario in scenario_names:
        results[scenario] = {
            "mean": summarized[(scenario, "mean")].iloc[0],
            "lower": summarized[(scenario, "lower")].iloc[0],
            "upper": summarized[(scenario, "upper")].iloc[0],
        }

    return pd.DataFrame(results).T


def calculate_percent_deaths_averted(
    annual_deaths,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate % deaths averted using run-to-run differences.
    """
    years = annual_deaths.index.astype(int)
    year_mask = np.isin(years, list(comparison_years))

    annual_deaths = annual_deaths.loc[year_mask]
    annual_deaths_agg = annual_deaths.sum(axis=0)

    pct_diff = pd.DataFrame(
        -100.0
        * find_difference_relative_to_comparison_series(
            annual_deaths_agg,
            comparison=baseline_scenario,
            scaled=True,
        )
    ).T

    summarized = summarize(pct_diff)
    results = {}

    scenario_names = (
        summarized.columns
        .get_level_values(0)
        .unique()
    )

    for scenario in scenario_names:
        results[scenario] = {
            "mean": summarized[(scenario, "mean")].iloc[0],
            "lower": summarized[(scenario, "lower")].iloc[0],
            "upper": summarized[(scenario, "upper")].iloc[0],
        }

    return pd.DataFrame(results).T


# Calculate % deaths averted by cause
def calculate_percent_deaths_averted_by_cause(
    deaths_by_cause,
    baseline_scenario,
):

    pct_diff = (
        -100.0
        * find_difference_relative_to_comparison_series_dataframe(
            deaths_by_cause,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    results = {}
    scenario_names = (summarized.columns.get_level_values(0).unique())

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame({
            "mean": summarized[(scenario, "mean")],
            "lower": summarized[(scenario, "lower")],
            "upper": summarized[(scenario, "upper")],
        })

    return results


def calculate_percent_dalys_averted_by_cause(
    dalys_by_cause,
    baseline_scenario,
):

    pct_diff = (
        -100.0
        * find_difference_relative_to_comparison_series_dataframe(
            dalys_by_cause,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    results = {}
    scenario_names = (summarized.columns.get_level_values(0).unique())

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame({
            "mean": summarized[(scenario, "mean")],
            "lower": summarized[(scenario, "lower")],
            "upper": summarized[(scenario, "upper")],
        })

    return results


# Calculate % DALYs averted by age group
def calculate_percent_dalys_averted_by_age_group(
    dalys_by_age_group,
    baseline_scenario,
):
    """
    Run-level comparison first,
    then summarize.
    """

    pct_diff = (
        -100.0
        * find_difference_relative_to_comparison_series_dataframe(
            dalys_by_age_group,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    results = {}
    scenario_names = (summarized.columns.get_level_values(0).unique())

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame({
            "mean": summarized[(scenario, "mean")],
            "lower": summarized[(scenario, "lower")],
            "upper": summarized[(scenario, "upper")],
        })

    return results


# Calculate % deaths averted by age group
def calculate_percent_deaths_averted_by_age_group(
    deaths_by_age_group,
    baseline_scenario,
):

    pct_diff = (
        -100.0
        * find_difference_relative_to_comparison_series_dataframe(
            deaths_by_age_group,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    results = {}
    scenario_names = (summarized.columns.get_level_values(0).unique())

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame({
            "mean": summarized[(scenario, "mean")],
            "lower": summarized[(scenario, "lower")],
            "upper": summarized[(scenario, "upper")],
        })

    return results


def plot_percent_dalys_averted_comparison(default_df, improved_df):
    # Default Healthsystem
    ordered_scenarios_default = [
        "Fewer Nurses / Default Healthsystem Function",
        "More Nurses / Default Healthsystem Function",
        "More CNP staff / Default Healthsystem Function",
        "More Nurses by District / Default Healthsystem Function",
        "More CNP staff by District / Default Healthsystem Function",
    ]

    label_map_default = {
        "Fewer Nurses / Default Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Default Healthsystem Function": "More\nnurses",
        "More CNP staff / Default Healthsystem Function": "More\nCNP",
        "More Nurses by District / Default Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Default Healthsystem Function": "More CNP\nby district",
    }

    color_map_default = {
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function": "orange",
    }

    labels_default = [label_map_default[s] for s in ordered_scenarios_default]
    means_default = default_df.loc[ordered_scenarios_default, "mean"].values
    lowers_default = default_df.loc[ordered_scenarios_default, "lower"].values
    uppers_default = default_df.loc[ordered_scenarios_default, "upper"].values

    yerr_default = np.vstack([means_default - lowers_default, uppers_default - means_default,])

    colors_default = [color_map_default[s] for s in ordered_scenarios_default]

    fig_default, ax_default = plt.subplots(figsize=(7, 6))

    ax_default.bar(
        labels_default,
        means_default,
        yerr=yerr_default,
        capsize=6,
        color=colors_default,
        width=0.55,
    )

    ax_default.axhline(0, color="black", linewidth=1)
    # ax_default.set_title("Default Healthsystem")
    ax_default.set_ylabel("% DALYs averted")
    ax_default.grid(axis="y", alpha=0.3)

    # fig_default.suptitle(
    #     "% DALYs averted relative to baseline (2027–2034)",
    #     fontsize=14,
    # )
    fig_default.tight_layout()


    # Improved Healthsystem
    ordered_scenarios_improved = [
        "Fewer Nurses / Improved Healthsystem Function",
        "More Nurses / Improved Healthsystem Function",
        "More CNP staff / Improved Healthsystem Function",
        "More Nurses by District / Improved Healthsystem Function",
        "More CNP staff by District / Improved Healthsystem Function",
    ]

    label_map_improved = {
        "Fewer Nurses / Improved Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Improved Healthsystem Function": "More\nnurses",
        "More CNP staff / Improved Healthsystem Function": "More\nCNP",
        "More Nurses by District / Improved Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Improved Healthsystem Function": "More CNP\nby district",
    }

    color_map_improved = {
        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function": "orange",
    }

    labels_improved = [label_map_improved[s] for s in ordered_scenarios_improved]
    means_improved = improved_df.loc[ordered_scenarios_improved, "mean"].values
    lowers_improved = improved_df.loc[ordered_scenarios_improved, "lower"].values
    uppers_improved = improved_df.loc[ordered_scenarios_improved, "upper"].values

    yerr_improved = np.vstack([means_improved - lowers_improved, uppers_improved - means_improved,])
    colors_improved = [color_map_improved[s] for s in ordered_scenarios_improved]

    fig_improved, ax_improved = plt.subplots(figsize=(7, 6))

    ax_improved.bar(
        labels_improved,
        means_improved,
        yerr=yerr_improved,
        capsize=6,
        color=colors_improved,
        width=0.55,
    )

    ax_improved.axhline(0, color="black", linewidth=1)
    # ax_improved.set_title("Improved Healthsystem")
    ax_improved.set_ylabel("% DALYs averted")
    ax_improved.grid(axis="y", alpha=0.3)
    # fig_improved.suptitle(
    #     "% DALYs averted relative to baseline (2027–2034)",
    #     fontsize=14,
    # )
    fig_improved.tight_layout()
    return fig_default, ax_default, fig_improved, ax_improved


def plot_percent_deaths_averted_comparison(default_df, improved_df):
    # Default Healthsystem
    ordered_scenarios_default = [
        "Fewer Nurses / Default Healthsystem Function",
        "More Nurses / Default Healthsystem Function",
        "More CNP staff / Default Healthsystem Function",
        "More Nurses by District / Default Healthsystem Function",
        "More CNP staff by District / Default Healthsystem Function",
    ]

    label_map_default = {
        "Fewer Nurses / Default Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Default Healthsystem Function": "More\nnurses",
        "More CNP staff / Default Healthsystem Function": "More\nCNP",
        "More Nurses by District / Default Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Default Healthsystem Function": "More CNP\nby district",
    }

    color_map_default = {
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function": "orange",
    }

    labels_default = [label_map_default[s] for s in ordered_scenarios_default]
    means_default = default_df.loc[ordered_scenarios_default, "mean"].values
    lowers_default = default_df.loc[ordered_scenarios_default, "lower"].values
    uppers_default = default_df.loc[ordered_scenarios_default, "upper"].values

    yerr_default = np.vstack([means_default - lowers_default, uppers_default - means_default,])
    colors_default = [color_map_default[s] for s in ordered_scenarios_default]

    fig_default, ax_default = plt.subplots(figsize=(7, 6))

    ax_default.bar(
        labels_default,
        means_default,
        yerr=yerr_default,
        capsize=6,
        color=colors_default,
        width=0.55,
    )

    ax_default.axhline(0, color="black", linewidth=1)
    # ax_default.set_title("Default Healthsystem")
    ax_default.set_ylabel("% Deaths averted")
    ax_default.grid(axis="y", alpha=0.3)

    # fig_default.suptitle(
    #     "% deaths averted relative to baseline (2027–2034)",
    #     fontsize=14,
    # )
    fig_default.tight_layout()


    # Improved Healthsystem
    ordered_scenarios_improved = [
        "Fewer Nurses / Improved Healthsystem Function",
        "More Nurses / Improved Healthsystem Function",
        "More CNP staff / Improved Healthsystem Function",
        "More Nurses by District / Improved Healthsystem Function",
        "More CNP staff by District / Improved Healthsystem Function",
    ]

    label_map_improved = {
        "Fewer Nurses / Improved Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Improved Healthsystem Function": "More\nnurses",
        "More CNP staff / Improved Healthsystem Function": "More\nCNP",
        "More Nurses by District / Improved Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Improved Healthsystem Function": "More CNP\nby district",
    }

    color_map_improved = {
        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function": "orange",
    }

    labels_improved = [label_map_improved[s] for s in ordered_scenarios_improved]
    means_improved = improved_df.loc[ordered_scenarios_improved, "mean"].values
    lowers_improved = improved_df.loc[ordered_scenarios_improved, "lower"].values
    uppers_improved = improved_df.loc[ordered_scenarios_improved, "upper"].values

    yerr_improved = np.vstack([means_improved - lowers_improved, uppers_improved - means_improved,])
    colors_improved = [color_map_improved[s] for s in ordered_scenarios_improved]

    fig_improved, ax_improved = plt.subplots(figsize=(7, 6))

    ax_improved.bar(
        labels_improved,
        means_improved,
        yerr=yerr_improved,
        capsize=6,
        color=colors_improved,
        width=0.55,
    )

    ax_improved.axhline(0, color="black", linewidth=1)
    ax_improved.set_title("Improved Healthsystem")
    ax_improved.set_ylabel("% Deaths averted\n")
    ax_improved.grid(axis="y", alpha=0.3)

    fig_improved.suptitle(
        "% deaths averted relative to baseline (2027–2034)",
        fontsize=14,
    )
    fig_improved.tight_layout()
    return fig_default, ax_default, fig_improved, ax_improved



# Shared plotting function for percentage DALYs/deaths averted by cause
def _plot_percent_averted_by_cause(
    default_df,
    improved_df,
    cause_order,
    outcome_label,
    top_n=30,
):
    """
    Plot percentage DALYs or deaths averted by cause.

    Causes are on the x-axis and percentage averted is on the y-axis.
    Default and Improved Healthsystem figures are plotted separately.
    """

    # Keep the existing cause order and limit the number of causes plotted.
    available_causes = [
        cause
        for cause in cause_order
        if cause in default_df["Fewer Nurses / Default Healthsystem Function"].index
        and cause in improved_df["Fewer Nurses / Improved Healthsystem Function"].index
    ]

    available_causes = sorted(available_causes, key=str.lower)
    causes = available_causes[:top_n]

    # Scenario order and colours
    scenarios = [
        ("Fewer nurses", "Fewer Nurses", "indianred"),
        ("More nurses", "More Nurses", "steelblue"),
        ("More CNP", "More CNP staff", "darkgreen"),
        ("More nurses by district", "More Nurses by District", "mediumpurple"),
        ("More CNP by district", "More CNP staff by District", "orange"),
    ]

    # Make each figure wider as the number of causes increases.
    # A larger width also makes the individual bars easier to distinguish.
    figure_width = max(20, len(causes) * 0.8)
    figure_height = 10

    # Total width allocated to the five bars for each cause.
    group_width = 0.95
    bar_width = group_width / len(scenarios)

    x = np.arange(len(causes))
    offsets = (
        np.arange(len(scenarios))
        - (len(scenarios) - 1) / 2
    ) * bar_width

    def plot_healthsystem(data, healthsystem):
        fig, ax = plt.subplots(
            figsize=(figure_width, figure_height)
        )

        for i, (label, scenario_prefix, color) in enumerate(scenarios):
            scenario_name = (
                f"{scenario_prefix} / {healthsystem} Healthsystem Function"
            )

            # Retrieve values and put causes in the intended order.
            scenario_df = data[scenario_name].reindex(causes)

            means = scenario_df["mean"].to_numpy()
            lower = scenario_df["lower"].to_numpy()
            upper = scenario_df["upper"].to_numpy()

            # Draw vertical bars, with causes on the x-axis.
            xpos = x + offsets[i]

            ax.bar(
                xpos,
                means,
                width=bar_width * 0.95,
                color=color,
                label=label,
                yerr=np.vstack([
                    means - lower,
                    upper - means,
                ]),
                capsize=2,
                error_kw={"elinewidth": 0.8},
            )

        ax.axhline(0, color="black", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(
            causes,
            rotation=60,
            ha="right",
            rotation_mode="anchor",
        )

        ax.set_xlabel("", labelpad=10)
        ax.set_ylabel(f"% {outcome_label} averted")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        # Put the legend below the x-axis label.
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=5,
            frameon=True,
        )

        # Leave room for long cause labels and the bottom legend.
        fig.subplots_adjust(
            left=0.06,
            right=0.99,
            top=0.97,
            bottom=0.30,
        )

        return fig, ax

    # Produce separate figures for each healthsystem.
    fig_default, ax_default = plot_healthsystem(
        default_df,
        "Default",
    )

    fig_improved, ax_improved = plot_healthsystem(
        improved_df,
        "Improved",
    )

    return fig_default, ax_default, fig_improved, ax_improved


# Plot % DALYs averted by cause
def plot_percent_dalys_averted_by_cause(
    default_df,
    improved_df,
    top_n=30,
):
    return _plot_percent_averted_by_cause(
        default_df=default_df,
        improved_df=improved_df,
        cause_order=cause_order,
        outcome_label="DALYs",
        top_n=top_n,
    )


# Plot % deaths averted by cause
def plot_percent_deaths_averted_by_cause(
    default_df,
    improved_df,
    top_n=30,
):
    return _plot_percent_averted_by_cause(
        default_df=default_df,
        improved_df=improved_df,
        cause_order=death_order,
        outcome_label="Deaths",
        top_n=top_n,
    )


# Plot % DALYs averted by age group
def plot_percent_dalys_averted_by_age_group(default_df,improved_df,):
    # Default Healthsystem
    default_more = default_df["More Nurses / Default Healthsystem Function"]
    default_cnp = default_df["More CNP staff / Default Healthsystem Function"]
    default_more_district = default_df["More Nurses by District / Default Healthsystem Function"]
    default_cnp_district = default_df["More CNP staff by District / Default Healthsystem Function"]
    default_fewer = default_df["Fewer Nurses / Default Healthsystem Function"]

    # Improved Healthsystem
    improved_more = improved_df["More Nurses / Improved Healthsystem Function"]
    improved_cnp = improved_df["More CNP staff / Improved Healthsystem Function"]
    improved_more_district = improved_df["More Nurses by District / Improved Healthsystem Function"]
    improved_cnp_district = improved_df["More CNP staff by District / Improved Healthsystem Function"]
    improved_fewer = improved_df["Fewer Nurses / Improved Healthsystem Function"]

    # Ordering age groups
    age_order = [
        "0-4",
        "5-9",
        "10-14",
        "15-19",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80+",
    ]

    for df in [default_more, default_fewer, improved_more, improved_fewer,]:
        df = df.reindex(age_order)

    default_more = default_more.reindex(age_order)
    default_cnp = default_cnp.reindex(age_order)
    default_more_district = default_more_district.reindex(age_order)
    default_cnp_district = default_cnp_district.reindex(age_order)
    default_fewer = default_fewer.reindex(age_order)

    improved_more = improved_more.reindex(age_order)
    improved_cnp = improved_cnp.reindex(age_order)
    improved_more_district = improved_more_district.reindex(age_order)
    improved_cnp_district = improved_cnp_district.reindex(age_order)
    improved_fewer = improved_fewer.reindex(age_order)

    # Reverse so oldest ages appear at top
    default_more = default_more.iloc[::-1]
    default_cnp = default_cnp.iloc[::-1]
    default_more_district = default_more_district.iloc[::-1]
    default_cnp_district = default_cnp_district.iloc[::-1]
    default_fewer = default_fewer.iloc[::-1]

    improved_more = improved_more.iloc[::-1]
    improved_cnp = improved_cnp.iloc[::-1]
    improved_more_district = improved_more_district.iloc[::-1]
    improved_cnp_district = improved_cnp_district.iloc[::-1]
    improved_fewer = improved_fewer.iloc[::-1]

    # Plot Default Healthsystem
    scenarios_default = [
        ("Fewer nurses", default_fewer, "indianred"),
        ("More nurses", default_more, "steelblue"),
        ("More CNP", default_cnp, "darkgreen"),
        ("More nurses by district", default_more_district, "mediumpurple"),
        ("More CNP by district", default_cnp_district, "orange"),
    ]

    fig_default, ax_default = plt.subplots(figsize=(12, 10))
    y_default = np.arange(len(scenarios_default[0][1]))

    offsets = [-0.32, -0.16, 0.0, 0.16, 0.32]

    for offset, (label, df, color) in zip(offsets, scenarios_default):
        ax_default.barh(
            y_default + offset,
            df["mean"],
            height=0.18,
            color=color,
            label=label,
        )

        ax_default.errorbar(
            df["mean"],
            y_default + offset,
            xerr=[
                df["mean"] - df["lower"],
                df["upper"] - df["mean"],
            ],
            fmt="none",
            elinewidth=1,
            capsize=1.5,
            color="black",
            alpha=0.4,
        )

    ax_default.axvline(0, color="black")
    ax_default.set_yticks(y_default)
    ax_default.set_yticklabels(scenarios_default[0][1].index)
    ax_default.set_xlabel("% DALYs averted")
    # ax_default.set_title("Default Healthsystem")
    ax_default.grid(axis="x", alpha=0.3)

    handles_default, labels_default = ax_default.get_legend_handles_labels()
    ax_default.legend(
        handles_default,
        labels_default,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
    )

    # fig_default.suptitle(
    #     "% DALYs averted by age group on national level\n"
    #     "(2027–2034)"
    # )
    fig_default.tight_layout(rect=[0, 0.08, 1, 1])

    # Plot Improved Healthsystem
    scenarios_improved = [
        ("Fewer nurses", improved_fewer, "indianred"),
        ("More nurses", improved_more, "steelblue"),
        ("More CNP", improved_cnp, "darkgreen"),
        ("More nurses by district", improved_more_district, "mediumpurple"),
        ("More CNP by district", improved_cnp_district, "orange"),
    ]

    fig_improved, ax_improved = plt.subplots(figsize=(12, 10))
    y_improved = np.arange(len(scenarios_improved[0][1]))

    for offset, (label, df, color) in zip(offsets, scenarios_improved):
        ax_improved.barh(
            y_improved + offset,
            df["mean"],
            height=0.18,
            color=color,
            label=label,
        )

        ax_improved.errorbar(
            df["mean"],
            y_improved + offset,
            xerr=[
                df["mean"] - df["lower"],
                df["upper"] - df["mean"],
            ],
            fmt="none",
            elinewidth=1,
            capsize=1.5,
            color="black",
            alpha=0.4,
        )

    ax_improved.axvline(0, color="black")
    ax_improved.set_yticks(y_improved)
    ax_improved.set_yticklabels(scenarios_improved[0][1].index)
    ax_improved.set_xlabel("% DALYs averted")
    # ax_improved.set_title("Improved Healthsystem")
    ax_improved.grid(axis="x", alpha=0.3)

    handles_improved, labels_improved = ax_improved.get_legend_handles_labels()
    ax_improved.legend(
        handles_improved,
        labels_improved,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
    )

    # fig_improved.suptitle(
    #     "% DALYs averted by age group on national level\n"
    #     "(2027–2034)"
    # )
    fig_improved.tight_layout(rect=[0, 0.08, 1, 1])

    return fig_default, ax_default, fig_improved, ax_improved


# Plot % deaths averted by age group
def plot_percent_deaths_averted_by_age_group(default_df,improved_df,):
    # Default Healthsystem
    default_more = default_df["More Nurses / Default Healthsystem Function"]
    default_cnp = default_df["More CNP staff / Default Healthsystem Function"]
    default_more_district = default_df["More Nurses by District / Default Healthsystem Function"]
    default_cnp_district = default_df["More CNP staff by District / Default Healthsystem Function"]
    default_fewer = default_df["Fewer Nurses / Default Healthsystem Function"]

    # Improved Healthsystem
    improved_more = improved_df["More Nurses / Improved Healthsystem Function"]
    improved_cnp = improved_df["More CNP staff / Improved Healthsystem Function"]
    improved_more_district = improved_df["More Nurses by District / Improved Healthsystem Function"]
    improved_cnp_district = improved_df["More CNP staff by District / Improved Healthsystem Function"]
    improved_fewer = improved_df["Fewer Nurses / Improved Healthsystem Function"]

    age_order = [
        "0-4",
        "5-9",
        "10-14",
        "15-19",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80+",
    ]

    default_more = default_more.reindex(age_order)
    default_cnp = default_cnp.reindex(age_order)
    default_more_district = default_more_district.reindex(age_order)
    default_cnp_district = default_cnp_district.reindex(age_order)
    default_fewer = default_fewer.reindex(age_order)

    improved_more = improved_more.reindex(age_order)
    improved_cnp = improved_cnp.reindex(age_order)
    improved_more_district = improved_more_district.reindex(age_order)
    improved_cnp_district = improved_cnp_district.reindex(age_order)
    improved_fewer = improved_fewer.reindex(age_order)

    # Reverse so oldest ages appear at top
    default_more = default_more.iloc[::-1]
    default_cnp = default_cnp.iloc[::-1]
    default_more_district = default_more_district.iloc[::-1]
    default_cnp_district = default_cnp_district.iloc[::-1]
    default_fewer = default_fewer.iloc[::-1]

    improved_more = improved_more.iloc[::-1]
    improved_cnp = improved_cnp.iloc[::-1]
    improved_more_district = improved_more_district.iloc[::-1]
    improved_cnp_district = improved_cnp_district.iloc[::-1]
    improved_fewer = improved_fewer.iloc[::-1]

    # Plot Default Healthsystem
    scenarios_default = [
        ("Fewer nurses", default_fewer, "indianred"),
        ("More nurses", default_more, "steelblue"),
        ("More CNP", default_cnp, "darkgreen"),
        ("More nurses by district", default_more_district, "mediumpurple"),
        ("More CNP by district", default_cnp_district, "orange"),
    ]

    fig_default, ax_default = plt.subplots(figsize=(12, 10))
    y_default = np.arange(len(scenarios_default[0][1]))

    offsets = [-0.32, -0.16, 0.0, 0.16, 0.32]

    for offset, (label, df, color) in zip(offsets, scenarios_default):
        ax_default.barh(
            y_default + offset,
            df["mean"],
            height=0.18,
            color=color,
            label=label,
        )

        ax_default.errorbar(
            df["mean"],
            y_default + offset,
            xerr=[
                df["mean"] - df["lower"],
                df["upper"] - df["mean"],
            ],
            fmt="none",
            elinewidth=1,
            capsize=1.5,
            color="black",
            alpha=0.4,
        )

    ax_default.axvline(0, color="black")
    ax_default.set_yticks(y_default)
    ax_default.set_yticklabels(scenarios_default[0][1].index)
    ax_default.set_xlabel("% Deaths averted")
    # ax_default.set_title("Default Healthsystem")
    ax_default.grid(axis="x", alpha=0.3)

    handles_default, labels_default = ax_default.get_legend_handles_labels()
    ax_default.legend(
        handles_default,
        labels_default,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
    )

    # fig_default.suptitle(
    #     "% deaths averted by age group on national level\n"
    #     "(2027–2034)"
    # )
    fig_default.tight_layout(rect=[0, 0.08, 1, 1])

    # Plot Improved Healthsystem
    scenarios_improved = [
        ("Fewer nurses", improved_fewer, "indianred"),
        ("More nurses", improved_more, "steelblue"),
        ("More CNP", improved_cnp, "darkgreen"),
        ("More nurses by district", improved_more_district, "mediumpurple"),
        ("More CNP by district", improved_cnp_district, "orange"),
    ]

    fig_improved, ax_improved = plt.subplots(figsize=(12, 10))
    y_improved = np.arange(len(scenarios_improved[0][1]))

    for offset, (label, df, color) in zip(offsets, scenarios_improved):
        ax_improved.barh(
            y_improved + offset,
            df["mean"],
            height=0.18,
            color=color,
            label=label,
        )

        ax_improved.errorbar(
            df["mean"],
            y_improved + offset,
            xerr=[
                df["mean"] - df["lower"],
                df["upper"] - df["mean"],
            ],
            fmt="none",
            elinewidth=1,
            capsize=1.5,
            color="black",
            alpha=0.4,
        )

    ax_improved.axvline(0, color="black")
    ax_improved.set_yticks(y_improved)
    ax_improved.set_yticklabels(scenarios_improved[0][1].index)
    ax_improved.set_xlabel("% deaths averted")
    # ax_improved.set_title("Improved Healthsystem")
    ax_improved.grid(axis="x", alpha=0.3)

    handles_improved, labels_improved = ax_improved.get_legend_handles_labels()
    ax_improved.legend(
        handles_improved,
        labels_improved,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
    )

    # fig_improved.suptitle(
    #     "% deaths averted by age group on national level\n"
    #     "(2027–2034)"
    # )
    fig_improved.tight_layout(rect=[0, 0.08, 1, 1])

    return fig_default, ax_default, fig_improved, ax_improved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Analyse DALYs/Deaths across nurse staffing scenarios"
    )
    parser.add_argument(
        "--scenario-outputs-folder",
        type=Path,
        required=True,
        help="Path to folder containing scenario outputs",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Whether to interactively show figures",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Whether to save figures to results folder",
    )
    args = parser.parse_args()

    # Use command-line folder
    results_folder = args.scenario_outputs_folder

    # Optional: load logs
    log = load_pickled_dataframes(results_folder)

    # Getting scenario names from scenario class
    param_names = tuple(StaffingScenario()._scenarios.keys())

    # Scnarios to keep (Default Healthsystem Function only)
    default_hs_scenarios = [
        "Baseline Nurses / Default Healthsystem Function",
        "Fewer Nurses / Default Healthsystem Function",
        "More Nurses / Default Healthsystem Function",
        "More CNP staff / Default Healthsystem Function",
        "More Nurses by District / Default Healthsystem Function",
        "More CNP staff by District / Default Healthsystem Function",
    ]

    baseline_scenario = "Baseline Nurses / Default Healthsystem Function"

    improved_hs_scenarios = [
        "Baseline Nurses / Improved Healthsystem Function",
        "Fewer Nurses / Improved Healthsystem Function",
        "More Nurses / Improved Healthsystem Function",
        "More CNP staff / Improved Healthsystem Function",
        "More Nurses by District / Improved Healthsystem Function",
        "More CNP staff by District / Improved Healthsystem Function",
    ]

    baseline_improved_scenario = ("Baseline Nurses / Improved Healthsystem Function")

    # Extract annual DALYs
    annual_dalys = extract_annual_dalys(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # Summarize across runs
    # Filter to Default Healthsystem Function scenarios only
    summarized_annual_dalys = summarize(annual_dalys)

    # Filter to Default Healthsystem Function scenarios only
    summarized_annual_dalys_default = summarized_annual_dalys.loc[
                                      :,
                                      summarized_annual_dalys.columns.get_level_values(0).isin(
                                          default_hs_scenarios
                                      ),
                                      ]

    # Filter to Improved Healthsystem Function scenarios only
    summarized_annual_dalys_improved = summarized_annual_dalys.loc[
                                       :,
                                       summarized_annual_dalys.columns.get_level_values(0).isin(
                                           improved_hs_scenarios
                                       ),
                                       ]

    # Plot 1: Annual DALYs over time
    fig_1, ax_1 = plot_annual_dalys(
        summarized_annual_dalys_default,
        title="DALYs at national level: Default Healthsystem",
    )

    # Plot 2: Percent DALYs averted relative to baseline (2027–2034)
    percent_dalys_averted = calculate_percent_dalys_averted(
        annual_dalys.loc[
            :,
            annual_dalys.columns.get_level_values(0).isin(default_hs_scenarios)
        ],
        baseline_scenario=baseline_scenario,
        comparison_years=range(2027, 2035),
    )

    percent_dalys_averted_improved = calculate_percent_dalys_averted(
        annual_dalys.loc[
            :,
            annual_dalys.columns.get_level_values(0).isin(improved_hs_scenarios)
        ],
        baseline_scenario=baseline_improved_scenario,
        comparison_years=range(2027, 2035),
    )

    fig_2_default, ax_2_default, fig_2_improved, ax_2_improved = (
        plot_percent_dalys_averted_comparison(
            percent_dalys_averted,
            percent_dalys_averted_improved,
        )
    )

    # Sensitivity analysis: DALYs under Improved Healthsystem Function
    fig_5, ax_5 = plot_annual_dalys(
        summarized_annual_dalys_improved,
        title="DALYs at national level: Improved Healthsystem",
    )

    # Extract annual deaths
    annual_deaths = extract_annual_deaths(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    summarized_annual_deaths = summarize(annual_deaths)

    # Default Healthsystem Function deaths
    summarized_annual_deaths_default = summarized_annual_deaths.loc[
                                       :,
                                       summarized_annual_deaths.columns.get_level_values(0).isin(
                                           default_hs_scenarios
                                       ),
                                       ]

    # Improved Healthsystem Function deaths
    summarized_annual_deaths_improved = summarized_annual_deaths.loc[
                                        :,
                                        summarized_annual_deaths.columns.get_level_values(0).isin(
                                            improved_hs_scenarios
                                        ),
                                        ]

    # Plot annual deaths
    fig_3, ax_3 = plot_annual_deaths(
        summarized_annual_deaths_default,
        title="Deaths at national level: Default Healthsystem",
    )

    # Plot % deaths averted
    percent_deaths_averted = calculate_percent_deaths_averted(
        annual_deaths.loc[
            :,
            annual_deaths.columns.get_level_values(0).isin(default_hs_scenarios)
        ],
        baseline_scenario=baseline_scenario,
        comparison_years=range(2027, 2035),
    )

    percent_deaths_averted_improved = calculate_percent_deaths_averted(
        annual_deaths.loc[
            :,
            annual_deaths.columns.get_level_values(0).isin(improved_hs_scenarios)
        ],
        baseline_scenario=baseline_improved_scenario,
        comparison_years=range(2027, 2035),
    )

    fig_4_default, ax_4_default, fig_4_improved, ax_4_improved = (
        plot_percent_deaths_averted_comparison(
            percent_deaths_averted,
            percent_deaths_averted_improved,
        )
    )

    # Sensitivity analysis: deaths under Improved Healthsystem Function
    fig_7, ax_7 = plot_annual_deaths(
        summarized_annual_deaths_improved,
        title="Deaths at national level: Improved Healthsystem",
    )

    # Extract deaths by cause
    deaths_by_cause = extract_deaths_by_cause(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # check that total deaths equal to sum of deaths by cause
    total_deaths = annual_deaths.loc[
        (annual_deaths.index >= 2027) & (annual_deaths.index <= 2034)
        ].sum(axis=0)
    total_deaths_cause = deaths_by_cause.sum(axis=0)
    assert (total_deaths.index == total_deaths_cause.index).all()
    assert (abs(total_deaths.values - total_deaths_cause.values) < 1e-7).all()

    # Alphabetical order of causes
    death_order = sorted(deaths_by_cause.index.tolist(), key=str.lower, reverse=True)

    deaths_by_cause_default = (
        deaths_by_cause.loc[
            :,
            deaths_by_cause.columns
            .get_level_values(0)
            .isin(default_hs_scenarios)
        ]
    )

    percent_deaths_by_cause_default = (
        calculate_percent_deaths_averted_by_cause(
            deaths_by_cause_default,
            baseline_scenario=baseline_scenario,
        )
    )

    deaths_by_cause_improved = (
        deaths_by_cause.loc[
            :,
            deaths_by_cause.columns
            .get_level_values(0)
            .isin(improved_hs_scenarios)
        ]
    )

    percent_deaths_by_cause_improved = (
        calculate_percent_deaths_averted_by_cause(
            deaths_by_cause_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_10_default, ax_10_default, fig_10_improved, ax_10_improved = (
        plot_percent_deaths_averted_by_cause(
            percent_deaths_by_cause_default,
            percent_deaths_by_cause_improved,
            top_n=30,
        )
    )

    # Extract deaths by age group
    deaths_by_age_group = extract_deaths_by_age_group(
        results_folder
    ).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # check that total deaths equal to sum of deaths by age group
    total_deaths_age = deaths_by_age_group.sum(axis=0)
    assert (total_deaths.index == total_deaths_age.index).all()
    assert (abs(total_deaths.values - total_deaths_age.values) < 1e-7).all()

    deaths_by_age_group_default = (
        deaths_by_age_group.loc[
            :,
            deaths_by_age_group.columns
            .get_level_values(0)
            .isin(default_hs_scenarios)
        ]
    )

    percent_deaths_by_age_default = (
        calculate_percent_deaths_averted_by_age_group(
            deaths_by_age_group_default,
            baseline_scenario=baseline_scenario,
        )
    )

    deaths_by_age_group_improved = (
        deaths_by_age_group.loc[
            :,
            deaths_by_age_group.columns
            .get_level_values(0)
            .isin(improved_hs_scenarios)
        ]
    )

    percent_deaths_by_age_improved = (
        calculate_percent_deaths_averted_by_age_group(
            deaths_by_age_group_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_12_default, ax_12_default, fig_12_improved, ax_12_improved = (
        plot_percent_deaths_averted_by_age_group(
            percent_deaths_by_age_default,
            percent_deaths_by_age_improved,
        )
    )

    # Extract DALYs by cause
    dalys_by_cause = extract_dalys_by_cause(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # check that total dalys equal to sum of dalys by cause
    total_dalys = annual_dalys.loc[
        (annual_dalys.index >= 2027) & (annual_dalys.index <= 2034)
        ].sum(axis=0)
    total_dalys_cause = dalys_by_cause.sum(axis=0)
    assert (total_dalys.index == total_dalys_cause.index).all()
    assert (abs(total_dalys.values - total_dalys_cause.values) < 1e-7).all()

    # Alphabetical order of causes
    cause_order = sorted(dalys_by_cause.index.tolist(), key=str.lower, reverse=True)

    # Default Healthsystem
    dalys_by_cause_default = (
        dalys_by_cause.loc[
            :,
            dalys_by_cause.columns
            .get_level_values(0)
            .isin(default_hs_scenarios)
        ]
    )

    percent_by_cause_default = (
        calculate_percent_dalys_averted_by_cause(
            dalys_by_cause_default,
            baseline_scenario=baseline_scenario,
        )
    )

    # Improved Healthsystem
    dalys_by_cause_improved = (
        dalys_by_cause.loc[
            :,
            dalys_by_cause.columns
            .get_level_values(0)
            .isin(improved_hs_scenarios)
        ]
    )

    percent_by_cause_improved = (
        calculate_percent_dalys_averted_by_cause(
            dalys_by_cause_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_9_default, ax_9_default, fig_9_improved, ax_9_improved = (
        plot_percent_dalys_averted_by_cause(
            percent_by_cause_default,
            percent_by_cause_improved,
            top_n=30,
        )
    )

    # Extract DALYs by age group
    dalys_by_age_group = extract_dalys_by_age_group(
        results_folder
    ).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # check that total dalys equal to sum of dalys by age groups
    total_dalys_age = dalys_by_age_group.sum(axis=0)
    assert (total_dalys.index == total_dalys_age.index).all()
    assert (abs(total_dalys.values - total_dalys_age.values) < 1e-7).all()

    dalys_by_age_group_default = (
        dalys_by_age_group.loc[
            :,
            dalys_by_age_group.columns
            .get_level_values(0)
            .isin(default_hs_scenarios)
        ]
    )

    percent_dalys_by_age_default = (
        calculate_percent_dalys_averted_by_age_group(
            dalys_by_age_group_default,
            baseline_scenario=baseline_scenario,
        )
    )

    dalys_by_age_group_improved = (
        dalys_by_age_group.loc[
            :,
            dalys_by_age_group.columns
            .get_level_values(0)
            .isin(improved_hs_scenarios)
        ]
    )

    percent_dalys_by_age_improved = (
        calculate_percent_dalys_averted_by_age_group(
            dalys_by_age_group_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_11_default, ax_11_default, fig_11_improved, ax_11_improved = (
        plot_percent_dalys_averted_by_age_group(
            percent_dalys_by_age_default,
            percent_dalys_by_age_improved,
        )
    )

    # Showing figures
    if args.show_figures:
        plt.show()

    # Saving figures
    if args.save_figures:
        fig_1.savefig(
            results_folder / "annual_dalys_across_scenarios_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_2_default.savefig(
            results_folder /
            "percent_dalys_averted_vs_baseline_2027_2034_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_2_improved.savefig(
            results_folder /
            "percent_dalys_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_3.savefig(
            results_folder / "annual_deaths_across_scenarios_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_4_default.savefig(
            results_folder /
            "percent_deaths_averted_vs_baseline_2027_2034_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_4_improved.savefig(
            results_folder /
            "percent_deaths_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        # Sensitivity-analysis DALY figures
        fig_5.savefig(
            results_folder /
            "annual_dalys_across_scenarios_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        # fig_6.savefig(
        #     results_folder /
        #     "percent_dalys_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
        #     bbox_inches="tight",
        # )

        # Sensitivity-analysis death figures
        fig_7.savefig(
            results_folder /
            "annual_deaths_across_scenarios_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        # fig_8.savefig(
        #     results_folder /
        #     "percent_deaths_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
        #     bbox_inches="tight",
        # )

        fig_9_default.savefig(
            results_folder /
            "percent_dalys_averted_by_cause_national_level_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_9_improved.savefig(
            results_folder /
            "percent_dalys_averted_by_cause_national_level_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_10_default.savefig(
            results_folder /
            "percent_deaths_averted_by_cause_national_level_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_10_improved.savefig(
            results_folder /
            "percent_deaths_averted_by_cause_national_level_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_11_default.savefig(
            results_folder /
            "percent_dalys_averted_by_age_group_national_level_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_11_improved.savefig(
            results_folder /
            "percent_dalys_averted_by_age_group_national_level_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_12_default.savefig(
            results_folder /
            "percent_deaths_averted_by_age_group_national_level_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_12_improved.savefig(
            results_folder /
            "percent_deaths_averted_by_age_group_national_level_improved_healthsystem.pdf",
            bbox_inches="tight",
        )
