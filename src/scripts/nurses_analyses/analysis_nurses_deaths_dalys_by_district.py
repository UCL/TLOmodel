"""Plot DALYs and Deaths across nurse staffing scenarios.

This script figures for the Nurse Shortages analysis at district level:

"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario
from tlo.analysis.utils import extract_results, load_pickled_dataframes, summarize

DALY_DEATH_METADATA_COLUMNS = {"date", "year", "sex", "age_range", "li_wealth", "district_of_residence"}


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
            if c not in DALY_DEATH_METADATA_COLUMNS
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
    def get_num_deaths_yearly(df):
        if "year" not in df.columns:
            df = df.assign(year=df["date"].dt.year)

        return (
            df.groupby("year")["person_id"]
            .count()
        )

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
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,)
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
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,)
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
            if c not in DALY_DEATH_METADATA_COLUMNS
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
            if c not in DALY_DEATH_METADATA_COLUMNS
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
            if c not in DALY_DEATH_METADATA_COLUMNS
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
    def get_deaths_by_district(df):
        if "year" not in df.columns:
            df = df.assign(year=df["date"].dt.year)

        return (
            df.groupby(
                ["year", "district_of_residence"]
            )["person_id"]
            .count()
        )

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_deaths_by_district,
        do_scaling=True,
    )


# For maps
def extract_total_dalys_by_district(results_folder):
    def get_total_dalys(df):
        df = df.assign(year=df["date"].dt.year)
        df = df[df["year"].between(2027, 2034)]

        cause_cols = [
            c
            for c in df.columns
            if c not in DALY_DEATH_METADATA_COLUMNS
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        df["total_dalys"] = df[cause_cols].sum(axis=1)

        return (
            df.groupby("district_of_residence")["total_dalys"]
            .sum()
        )

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_total_dalys,
        do_scaling=True,
    )

def extract_total_deaths_by_district(results_folder):
    def get_total_deaths(df):
        df = df.assign(year=df["date"].dt.year)
        df = df[df["year"].between(2027, 2034)]

        return (
            df.groupby("district_of_residence")["person_id"]
            .count()
        )

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_total_deaths,
        do_scaling=True,
    )


def plot_district_maps(gdf, scenario_names, title):
    vmax = np.nanmax(np.abs(gdf[scenario_names].values))

    fig, axes = plt.subplots(1, len(scenario_names), figsize=(6 * len(scenario_names), 8))

    if len(scenario_names) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenario_names):
        gdf.plot(column=scenario, cmap="coolwarm", edgecolor="black", linewidth=0.4,
                legend=False, vmin=-vmax, vmax=vmax, ax=ax)

        # ax.set_title(scenario.replace(" / Default Healthsystem Function", ""), fontsize=11,)

        label_map = {
            "More Nurses / Default Healthsystem Function":
                "More nurses",

            "Fewer Nurses / Default Healthsystem Function":
                "Fewer nurses",

            "More CNP staff / Default Healthsystem Function":
                "More CNP",

            "More Nurses by District / Default Healthsystem Function":
                "More nurses by district",

            "More CNP staff by District / Default Healthsystem Function":
                "More CNP by district",

            "More Nurses / Improved Healthsystem Function":
                "More nurses",

            "Fewer Nurses / Improved Healthsystem Function":
                "Fewer nurses",

            "More CNP staff / Improved Healthsystem Function":
                "More CNP",

            "More Nurses by District / Improved Healthsystem Function":
                "More nurses by district",

            "More CNP staff by District / Improved Healthsystem Function":
                "More CNP by district",
        }

        ax.set_title(label_map[scenario], fontsize=12)

        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-vmax, vmax),)
    # sm._A = []
    # fig.subplots_adjust(right=0.86)
    # fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.07, label="% change in DALYs (vs Baseline)")
    # plt.suptitle(title)
    # plt.tight_layout()
    # fig.tight_layout()
    # return fig
    sm.set_array([])
    # Create an independent axis for the colour bar
    cax = fig.add_axes([0.87, 0.20, 0.02, 0.60])
    #                     ^left ^bottom ^width ^height
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("% change in DALYs (vs Baseline)")
    fig.suptitle(title, fontsize=14)
    return fig


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

    # Malawi district boundaries
    district_map = gpd.read_file(
        Path(
            "resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
        )
    )
    print("---District Columns---\n")
    print(district_map.head())

    print("\nDistricts in shapefile:")
    print(sorted(district_map["ADM2_EN"].tolist()))

    # Optional: load logs
    log = load_pickled_dataframes(results_folder)
    param_names = tuple(StaffingScenario()._scenarios.keys())

    # NATIONAL DALYs
    annual_dalys = extract_annual_dalys(results_folder)

    print("\nNational DALYs")
    print(annual_dalys)

    dalys_by_district = extract_total_dalys_by_district(results_folder)

    print("\n---Show scenario names---")
    print(dalys_by_district.columns.to_list())

    # DALYs Default Healthsystem
    dalys_by_district = set_param_names_as_column_index_level_0(
        dalys_by_district,
        param_names
    )

    district_mean = (dalys_by_district.groupby(level=0, axis=1).mean())

    print("\nDistricts in DALY results:")
    print(sorted(district_mean.index.tolist()))

    print("\nNumber of Districts:")
    print(sorted(district_mean.index.tolist()))

    default_baseline = (
        "Baseline Nurses / Default Healthsystem Function"
    )

    default_pct = (
        district_mean
        .subtract(district_mean[default_baseline], axis=0)
        .divide(district_mean[default_baseline], axis=0)
        * 100
    )

    default_pct = default_pct.drop(columns=default_baseline)

    district_map_default = district_map.merge(
        default_pct,
        left_on="ADM2_EN",
        right_index=True,
        how="left",
    )

    fig_dalys_default_maps = plot_district_maps(
        district_map_default,
        [
            "More Nurses / Default Healthsystem Function",
            "Fewer Nurses / Default Healthsystem Function",
            "More CNP staff / Default Healthsystem Function",
            "More Nurses by District / Default Healthsystem Function",
            "More CNP staff by District / Default Healthsystem Function",
        ],
        "% DALYs averted (vs Baseline): Default Healthsystem",
    )

    # DALYs Improved Healthsystem
    improved_baseline = (
        "Baseline Nurses / Improved Healthsystem Function"
    )

    improved_pct = (
        district_mean
        .subtract(district_mean[improved_baseline], axis=0)
        .divide(district_mean[improved_baseline], axis=0)
        * 100
    )

    improved_pct = improved_pct.drop(columns=improved_baseline)

    district_map_improved = district_map.merge(
        improved_pct,
        left_on="ADM2_EN",
        right_index=True,
        how="left",
    )

    fig_dalys_improved_maps = plot_district_maps(
        district_map_improved,
        [
            "More Nurses / Improved Healthsystem Function",
            "Fewer Nurses / Improved Healthsystem Function",
            "More CNP staff / Improved Healthsystem Function",
            "More Nurses by District / Improved Healthsystem Function",
            "More CNP staff by District / Improved Healthsystem Function",
        ],
        "% DALYs averted (vs Baseline): Improved Healthsystem",
    )


    # VALIDATION

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

    death_comparison = pd.concat(
        {
            "National Deaths": annual_deaths,
            "District Deaths": district_death_totals,
        },
        axis=1,
    )

    print(death_comparison)

    difference = annual_deaths - district_death_totals

    print("\nComparison of National vs District Deaths")
    print(death_comparison)
    print("\nComparison Death Difference")
    print(difference)

    assert np.allclose(annual_deaths.values, district_death_totals.values,)

    print("\nDeath validation passed.")

    if args.save_figures:
        fig_dalys_default_maps.savefig(
            results_folder /
            "district_dalys_default.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig_dalys_improved_maps.savefig(
            results_folder /
            "district_dalys_improved.png",
            dpi=300,
            bbox_inches="tight",
        )

