"""Plot DALYs and Deaths across nurse staffing scenarios.

This script figures for the Nurse Shortages analysis at district level:

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
        _ser.unstack(level=0)
        .apply(
            lambda x: (
                (x - x[comparison]) /
                (x[comparison] if scaled else 1.0)
            ),
            axis=1,
        )
        .drop(columns=([comparison] if drop_comparison else []))
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

        yearly = (df.groupby("year")[cause_cols].sum().sum(axis=1))

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
        yearly = (df.assign(year=df["date"].dt.year).groupby("year")["person_id"].count())
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

        ax.plot(years, means, linewidth=2, label=label_map.get(scenario, scenario),)
        ax.fill_between(years, lowers, uppers, alpha=0.2,)

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

        ax.fill_between(years, lowers, uppers, alpha=0.2,)

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

        age_labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
                      "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+",
                    ]

        df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False,)
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
        dalys_by_age = (df.groupby("age_range")["total_dalys"].sum())
        return dalys_by_age

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_dalys_by_age_group,
        do_scaling=True,
    )


def extract_dalys_by_district(results_folder):

    def get_dalys_by_district(df):

        df = df.assign(year=df["date"].dt.year)

        cause_cols = [
            c for c in df.columns
            if c not in DALY_METADATA_COLUMNS
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        df["total_dalys"] = df[cause_cols].sum(axis=1)

        return (
            df.groupby(["year", "district_of_residence"])["total_dalys"].sum()
        )

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_dalys_by_district,
        do_scaling=True,
    )


def extract_deaths_by_district(results_folder):

    # def get_deaths_by_district(df: pd.DataFrame) -> pd.Series:
    #     """Return annual deaths by district."""
    #
    #     df = df.assign(year=df["date"].dt.year)
    #
    #     return (df.groupby(["year", "district_of_residence"])["person_id"].count())
    #
    # return extract_results(
    #     results_folder,
    #     module="tlo.methods.demography",
    #     key="death",
    #     custom_generate_series=get_deaths_by_district,
    #     do_scaling=True,
    # )
    def get_deaths_by_district(df):
        print("===Looking for Death Cols")
        print(df.columns)
        print(df.head())

        raise SystemExit


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

    # NATIONAL DALYs
    annual_dalys = extract_annual_dalys(results_folder)

    print("\nNational DALYs")
    print(annual_dalys)

    # DISTRICT DALYs
    dalys_by_district = extract_dalys_by_district(results_folder)

    print("\nDALYs by district")
    print(dalys_by_district)

    # VALIDATION
    # Sum DALYs across districts and compare to national totals
    district_totals = dalys_by_district.groupby(level="year").sum()

    comparison = pd.DataFrame({
        "National DALYs": annual_dalys.squeeze(),
        "District DALYs": district_totals.squeeze(),
    })

    comparison["Difference"] = (
        comparison["National DALYs"]
        - comparison["District DALYs"]
    ).round(6)

    print("\nComparison of National vs District DALYs")
    print(comparison)

    assert np.allclose(
        comparison["National DALYs"],
        comparison["District DALYs"],
    ), "District DALYs do not sum to national DALYs."

    print("\n DALY validation passed.")

    # NATIONAL DEATHS
    annual_deaths = extract_annual_deaths(results_folder)

    print("\nNational Deaths")
    print(annual_deaths)

    # DISTRICT DEATHS

    deaths_by_district = extract_deaths_by_district(results_folder)

    print("\nDeaths by district")
    print(deaths_by_district)

    # VALIDATION
    # Sum deaths across districts and compare to national totals

    district_death_totals = deaths_by_district.groupby(level="year").sum()

    death_comparison = pd.DataFrame({
        "National Deaths": annual_deaths.squeeze(),
        "District Deaths": district_death_totals.squeeze(),
    })

    death_comparison["Difference"] = (
        death_comparison["National Deaths"]
        - death_comparison["District Deaths"]
    ).round(6)

    print("\nComparison of National vs District Deaths")
    print(death_comparison)

    assert np.allclose(
        death_comparison["National Deaths"],
        death_comparison["District Deaths"],
    ), "District deaths do not sum to national deaths."

    print("\n✓ Death validation passed.")

