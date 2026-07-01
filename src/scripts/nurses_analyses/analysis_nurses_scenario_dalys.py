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
def plot_annual_dalys(summarized_annual_dalys):
    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_names = summarized_annual_dalys.columns.get_level_values(0).unique()

    # Short labels for legend
    label_map = {
        "Baseline Nurses / Default Healthsystem Function": "Baseline",
        "Fewer Nurses / Default Healthsystem Function": "Fewer nurses",
        "More Nurses / Default Healthsystem Function": "More nurses",

        "Baseline Nurses / Improved Healthsystem Function": "Baseline",
        "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
        "More Nurses / Improved Healthsystem Function": "More nurses",
    }

    for scenario in scenario_names:
        years = summarized_annual_dalys.index.astype(int)
        means = summarized_annual_dalys[(scenario, "mean")].values
        lowers = summarized_annual_dalys[(scenario, "lower")].values
        uppers = summarized_annual_dalys[(scenario, "upper")].values

        print(means.min(), means.max())

        ax.plot(
            years,
            means,
            linewidth=2,
            label=label_map.get(scenario, scenario),
        )

        ax.fill_between(
            years,
            lowers,
            uppers,
            alpha=0.2,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual DALYs")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(2025, 2034)
    ax.set_ylim(bottom=8e6)
    # ax.set_ylim(bottom=0.8)
    fig.tight_layout()

    return fig, ax


# Plot: Annual Deaths over time
def plot_annual_deaths(summarized_annual_deaths):
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

        "Baseline Nurses / Improved Healthsystem Function": "Baseline",
        "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
        "More Nurses / Improved Healthsystem Function": "More nurses",
    }

    for scenario in scenario_names:
        years = summarized_annual_deaths.index.astype(int)
        means = summarized_annual_deaths[(scenario, "mean")].values
        lowers = summarized_annual_deaths[(scenario, "lower")].values

        uppers = summarized_annual_deaths[(scenario, "upper")].values

        ax.plot(
            years,
            means,
            linewidth=2,
            label=label_map.get(scenario, scenario),
        )

        ax.fill_between(
            years,
            lowers,
            uppers,
            alpha=0.2,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual deaths")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(2025, 2034)
    fig.tight_layout()
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


def plot_percent_dalys_averted_comparison(default_df, improved_df,):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharey=True,)

    panel_data = [
        (axes[0], default_df, "Default Healthsystem"),
        (axes[1], improved_df, "Improved Healthsystem"),
    ]

    for ax, df, title in panel_data:
        ordered_scenarios = [
            s for s in df.index
            if "More Nurses" in s
        ] + [
            s for s in df.index
            if "Fewer Nurses" in s
        ]

        labels = ["More nurses" if "More Nurses" in s else "Fewer nurses" for s in ordered_scenarios]

        means = df.loc[ordered_scenarios, "mean"].values
        lowers = df.loc[ordered_scenarios, "lower"].values
        uppers = df.loc[ordered_scenarios, "upper"].values

        yerr = np.vstack([
            means - lowers,
            uppers - means,
        ])

        colors = ["steelblue" if "More Nurses" in s else "indianred" for s in ordered_scenarios]

        ax.bar(labels, means, yerr=yerr, capsize=6, color=colors, width=0.55,)
        ax.axhline(0, color="black", linewidth=1,)
        ax.set_title(title)
        ax.grid(axis="y",alpha=0.3,)

    axes[0].set_ylabel(
        "% DALYs averted compared to Baseline\n"
        "(total between 2027 and 2034)"
    )

    fig.suptitle(
        "% DALYs averted relative to baseline (2027–2034)",
        fontsize=14,
    )
    fig.tight_layout()
    return fig, axes


def plot_percent_deaths_averted_comparison(default_df,improved_df,):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharey=True,)

    panel_data = [
        (axes[0], default_df, "Default Healthsystem"),
        (axes[1], improved_df, "Improved Healthsystem"),
    ]

    for ax, df, title in panel_data:
        ordered_scenarios = [
            s for s in df.index
            if "More Nurses" in s
        ] + [
            s for s in df.index
            if "Fewer Nurses" in s
        ]

        labels = ["More nurses" if "More Nurses" in s else "Fewer nurses" for s in ordered_scenarios]

        means = df.loc[ordered_scenarios, "mean"].values
        lowers = df.loc[ordered_scenarios, "lower"].values
        uppers = df.loc[ordered_scenarios, "upper"].values

        yerr = np.vstack([
            means - lowers,
            uppers - means,
        ])

        colors = ["steelblue" if "More Nurses" in s else "indianred" for s in ordered_scenarios]

        ax.bar(labels, means, yerr=yerr, capsize=6, color=colors, width=0.55,)
        ax.axhline(0, color="black", linewidth=1,)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3,)

    axes[0].set_ylabel(
        "% deaths averted compared to Baseline\n"
        "(total between 2027 and 2034)"
    )

    fig.suptitle(
        "% deaths averted relative to baseline (2027–2034)",
        fontsize=14,
    )
    fig.tight_layout()
    return fig, axes


# Plot % DALYs averted by cause
def plot_percent_dalys_averted_by_cause(default_df, improved_df, top_n=30):

    # Extracting scenario dataframes
    default_more = default_df[
        "More Nurses / Default Healthsystem Function"
    ]

    default_fewer = default_df[
        "Fewer Nurses / Default Healthsystem Function"
    ]

    improved_more = improved_df[
        "More Nurses / Improved Healthsystem Function"
    ]

    improved_fewer = improved_df[
        "Fewer Nurses / Improved Healthsystem Function"
    ]

    # Using sum
    # total_dalys = (
    #     dalys_by_cause
    #     .xs(baseline_scenario, level="draw", axis=1)
    #     .sum(axis=1)
    #     .sort_values(ascending=False)
    # )
    #
    # top_causes = total_dalys.head(10).index.tolist()
    #
    # default_more = default_more.loc[top_causes]
    # default_fewer = default_fewer.loc[top_causes]
    #
    # improved_more = improved_more.loc[top_causes]
    # improved_fewer = improved_fewer.loc[top_causes]
    #
    # # Reverse so largest appears at top
    # default_more = default_more.iloc[::-1]
    # default_fewer = default_fewer.iloc[::-1]
    #
    # improved_more = improved_more.iloc[::-1]
    # improved_fewer = improved_fewer.iloc[::-1]

    # Top causes for DEFAULT healthsystem
    default_top = (
        default_more["mean"]
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    # default_more = (
    #     default_more.loc[default_top]
    #     .sort_values("mean", ascending=True)
    # )
    default_more = default_more.reindex(cause_order)

    # default_fewer = (
    #     default_fewer.loc[default_top]
    #     .reindex(default_more.index)
    # )
    default_fewer = default_fewer.reindex(cause_order)

    # Top causes for IMPROVED healthsystem
    improved_top = (
        improved_more["mean"]
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    # improved_more = (
    #     improved_more.loc[improved_top]
    #     .sort_values("mean", ascending=True)
    # )
    improved_more = improved_more.reindex(cause_order)

    # improved_fewer = (
    #     improved_fewer.loc[improved_top]
    #     .reindex(improved_more.index)
    # )
    improved_fewer = improved_fewer.reindex(cause_order)

    # Plot
    fig, axes = plt.subplots(ncols=2, figsize=(14, 10), sharey=True)

    panel_data = [
        (
            axes[0],
            default_more,
            default_fewer,
            "Default Healthsystem",
        ),
        (
            axes[1],
            improved_more,
            improved_fewer,
            "Improved Healthsystem",
        ),
    ]

    for ax, more, fewer, title in panel_data:
        y = np.arange(len(more))
        ax.barh(y - 0.2, more["mean"], height=0.35, color="steelblue", label="More nurses",)

        ax.barh(y + 0.2, fewer["mean"], height=0.35, color="indianred", label="Fewer nurses",)

        # CI bars: More nurses
        ax.errorbar(
            more["mean"],
            y - 0.2,
            xerr=[
                more["mean"] - more["lower"],
                more["upper"] - more["mean"],
            ],
            fmt="none",
            capsize=2,
            color="black",
            alpha=0.5,
        )

        # CI bars: Fewer nurses
        ax.errorbar(
            fewer["mean"],
            y + 0.2,
            xerr=[
                fewer["mean"] - fewer["lower"],
                fewer["upper"] - fewer["mean"],
            ],
            fmt="none",
            capsize=2,
            color="black",
            alpha=0.5,
        )

        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(more.index)
        ax.set_xlabel("% DALYs averted")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),)

    fig.suptitle(
        "% DALYs averted by causes on national level\n(2027–2034)"
    )
    fig.tight_layout()
    return fig, axes


# Plot % deaths averted by cause
def plot_percent_deaths_averted_by_cause(default_df, improved_df, top_n=30):

    # Extracting scenario dataframes
    default_more = default_df[
        "More Nurses / Default Healthsystem Function"
    ]

    default_fewer = default_df[
        "Fewer Nurses / Default Healthsystem Function"
    ]

    improved_more = improved_df[
        "More Nurses / Improved Healthsystem Function"
    ]

    improved_fewer = improved_df[
        "Fewer Nurses / Improved Healthsystem Function"
    ]

    # Top causes for DEFAULT healthsystem
    default_top = (
        default_more["mean"]
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    # default_more = (
    #     default_more.loc[default_top]
    #     .sort_values("mean", ascending=True)
    # )
    default_more = default_more.reindex(death_order)

    # default_fewer = (
    #     default_fewer.loc[default_top]
    #     .reindex(default_more.index)
    # )
    default_fewer = default_fewer.reindex(death_order)

    # Top causes for IMPROVED healthsystem
    improved_top = (
        improved_more["mean"]
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    # improved_more = (
    #     improved_more.loc[improved_top]
    #     .sort_values("mean", ascending=True)
    # )
    improved_more = improved_more.reindex(death_order)

    # improved_fewer = (
    #     improved_fewer.loc[improved_top]
    #     .reindex(improved_more.index)
    # )
    improved_fewer = improved_fewer.reindex(death_order)

    # Plot
    fig, axes = plt.subplots(ncols=2, figsize=(14, 10), sharey=True)

    panel_data = [
        (
            axes[0],
            default_more,
            default_fewer,
            "Default Healthsystem",
        ),
        (
            axes[1],
            improved_more,
            improved_fewer,
            "Improved Healthsystem",
        ),
    ]

    for ax, more, fewer, title in panel_data:
        y = np.arange(len(more))

        ax.barh(y - 0.2, more["mean"], height=0.35, color="steelblue", label="More nurses",)
        ax.barh(y + 0.2, fewer["mean"], height=0.35, color="indianred", label="Fewer nurses",)

        ax.errorbar(
            more["mean"],
            y - 0.2,
            xerr=[
                more["mean"] - more["lower"],
                more["upper"] - more["mean"],
            ],
            fmt="none",
            capsize=2,
            color="black",
            alpha=0.5,
        )

        ax.errorbar(
            fewer["mean"],
            y + 0.2,
            xerr=[
                fewer["mean"] - fewer["lower"],
                fewer["upper"] - fewer["mean"],
            ],
            fmt="none",
            capsize=2,
            color="black",
            alpha=0.5
        )

        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(more.index)
        ax.set_xlabel("% deaths averted")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),)
    fig.suptitle(
        "% deaths averted by causes on national level\n(2027–2034)"
    )
    fig.tight_layout()
    return fig, axes


# Plot % DALYs averted by age group
def plot_percent_dalys_averted_by_age_group(default_df,improved_df,):

    default_more = default_df[
        "More Nurses / Default Healthsystem Function"
    ]

    default_fewer = default_df[
        "Fewer Nurses / Default Healthsystem Function"
    ]

    improved_more = improved_df[
        "More Nurses / Improved Healthsystem Function"
    ]

    improved_fewer = improved_df[
        "Fewer Nurses / Improved Healthsystem Function"
    ]

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
    default_fewer = default_fewer.reindex(age_order)

    improved_more = improved_more.reindex(age_order)
    improved_fewer = improved_fewer.reindex(age_order)

    # Reverse so oldest ages appear at top
    default_more = default_more.iloc[::-1]
    default_fewer = default_fewer.iloc[::-1]

    improved_more = improved_more.iloc[::-1]
    improved_fewer = improved_fewer.iloc[::-1]

    # Plot
    fig, axes = plt.subplots(ncols=2, figsize=(14, 8), sharey=True,)

    panel_data = [
        (
            axes[0],
            default_more,
            default_fewer,
            "Default Healthsystem",
        ),
        (
            axes[1],
            improved_more,
            improved_fewer,
            "Improved Healthsystem",
        ),
    ]

    for ax, more, fewer, title in panel_data:
        y = np.arange(len(more))

        # More nurses
        ax.barh(y - 0.2, more["mean"], height=0.35, color="steelblue", label="More Nurses",)

        # Fewer nurses
        ax.barh(y + 0.2, fewer["mean"], height=0.35, color="indianred", label="Fewer Nurses",)

        # CI for More Nurses
        ax.errorbar(
            more["mean"],
            y - 0.2,
            xerr=[
                more["mean"] - more["lower"],
                more["upper"] - more["mean"],
            ],
            fmt="none",
            color="black",
            capsize=3,
        )

        # CI for Fewer Nurses
        ax.errorbar(
            fewer["mean"],
            y + 0.2,
            xerr=[
                fewer["mean"] - fewer["lower"],
                fewer["upper"] - fewer["mean"],
            ],
            fmt="none",
            color="black",
            capsize=3,
        )

        ax.axvline(0, color="black")
        ax.set_yticks(y)
        ax.set_yticklabels(more.index)
        ax.set_xlabel("% DALYs averted")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "% DALYs averted by age group on national level\n(2027–2034)"
    )

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,)
    fig.tight_layout()
    return fig, axes


# Plot % deaths averted by age group
def plot_percent_deaths_averted_by_age_group(default_df,improved_df,):

    default_more = default_df[
        "More Nurses / Default Healthsystem Function"
    ]

    default_fewer = default_df[
        "Fewer Nurses / Default Healthsystem Function"
    ]

    improved_more = improved_df[
        "More Nurses / Improved Healthsystem Function"
    ]

    improved_fewer = improved_df[
        "Fewer Nurses / Improved Healthsystem Function"
    ]

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
    default_fewer = default_fewer.reindex(age_order)

    improved_more = improved_more.reindex(age_order)
    improved_fewer = improved_fewer.reindex(age_order)

    # Reverse so oldest age groups appear at top
    default_more = default_more.iloc[::-1]
    default_fewer = default_fewer.iloc[::-1]

    improved_more = improved_more.iloc[::-1]
    improved_fewer = improved_fewer.iloc[::-1]

    fig, axes = plt.subplots(ncols=2, figsize=(14, 8), sharey=True,)

    panel_data = [
        (
            axes[0],
            default_more,
            default_fewer,
            "Default Healthsystem",
        ),
        (
            axes[1],
            improved_more,
            improved_fewer,
            "Improved Healthsystem",
        ),
    ]

    for ax, more, fewer, title in panel_data:
        y = np.arange(len(more))

        ax.barh(y - 0.2, more["mean"], height=0.35, color="steelblue", label="More nurses",)
        ax.barh(y + 0.2, fewer["mean"], height=0.35, color="indianred", label="Fewer nurses",)

        # More nurses CI
        ax.errorbar(
            more["mean"],
            y - 0.2,
            xerr=[
                more["mean"] - more["lower"],
                more["upper"] - more["mean"],
            ],
            fmt="none",
            capsize=4,
            color="black",
        )

        # Fewer nurses CI
        ax.errorbar(
            fewer["mean"],
            y + 0.2,
            xerr=[
                fewer["mean"] - fewer["lower"],
                fewer["upper"] - fewer["mean"],
            ],
            fmt="none",
            capsize=4,
            color="black",
        )

        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(more.index)
        ax.set_xlabel("% deaths averted")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),)
    fig.suptitle(
        "% deaths averted by age group on national level\n(2027–2034)"
    )
    fig.tight_layout()
    return fig, axes


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
    ]

    baseline_scenario = "Baseline Nurses / Default Healthsystem Function"

    improved_hs_scenarios = [
        "Baseline Nurses / Improved Healthsystem Function",
        "Fewer Nurses / Improved Healthsystem Function",
        "More Nurses / Improved Healthsystem Function",
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
    fig_1, ax_1 = plot_annual_dalys(summarized_annual_dalys_default)

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

    fig_2, ax_2 = plot_percent_dalys_averted_comparison(
        percent_dalys_averted,
        percent_dalys_averted_improved,
    )

    # Sensitivity analysis: DALYs under Improved Healthsystem Function
    fig_5, ax_5 = plot_annual_dalys(
        summarized_annual_dalys_improved
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
        summarized_annual_deaths_default
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

    fig_4, ax_4 = plot_percent_deaths_averted_comparison(
        percent_deaths_averted,
        percent_deaths_averted_improved,
    )

    # Sensitivity analysis: deaths under Improved Healthsystem Function
    fig_7, ax_7 = plot_annual_deaths(
        summarized_annual_deaths_improved
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

    # find the descending order of causes in terms of total deaths in baseline scenario
    mean_deaths_by_cause = deaths_by_cause.groupby(axis=1, level="draw").mean().sort_values(
        by="Baseline Nurses / Default Healthsystem Function",
        ascending=True,
    )
    death_order = mean_deaths_by_cause.index.tolist()

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

    fig_10, ax_10 = plot_percent_deaths_averted_by_cause(
        percent_deaths_by_cause_default,
        percent_deaths_by_cause_improved,
        top_n=30,
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

    fig_12, ax_12 = plot_percent_deaths_averted_by_age_group(
        percent_deaths_by_age_default,
        percent_deaths_by_age_improved,
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

    # find the descending order of causes in terms of total dalys in baseline scenario
    mean_dalys_by_cause = dalys_by_cause.groupby(axis=1, level="draw").mean().sort_values(
        by="Baseline Nurses / Default Healthsystem Function",
        ascending=True,
    )
    cause_order = mean_dalys_by_cause.index.tolist()

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

    fig_9, ax_9 = plot_percent_dalys_averted_by_cause(
        percent_by_cause_default,
        percent_by_cause_improved,
        top_n=30,
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

    fig_11, ax_11 = plot_percent_dalys_averted_by_age_group(
        percent_dalys_by_age_default,
        percent_dalys_by_age_improved,
    )

    # Showing figures
    if args.show_figures:
        plt.show()

    # Saving figures
    if args.save_figures:
        fig_1.savefig(
            results_folder / "annual_dalys_across_scenarios.pdf",
            bbox_inches="tight",
        )

        fig_2.savefig(
            results_folder / "percent_dalys_averted_vs_baseline_2027_2034_comparison.pdf",
            bbox_inches="tight",
        )

        fig_3.savefig(
            results_folder / "annual_deaths_across_scenarios.pdf",
            bbox_inches="tight",
        )

        fig_4.savefig(
            results_folder / "percent_deaths_averted_vs_baseline_2027_2034_comparison.pdf",
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

        fig_9.savefig(
            results_folder /
            "percent_dalys_averted_by_cause_national_level.pdf",
            bbox_inches="tight",
        )

        fig_10.savefig(
            results_folder /
            "percent_deaths_averted_by_cause_national_level.pdf",
            bbox_inches="tight",
        )

        fig_11.savefig(
            results_folder /
            "percent_dalys_averted_by_age_group_national_level.pdf",
            bbox_inches="tight",
        )

        fig_12.savefig(
            results_folder /
            "percent_deaths_averted_by_age_group_national_level.pdf",
            bbox_inches="tight",
        )
