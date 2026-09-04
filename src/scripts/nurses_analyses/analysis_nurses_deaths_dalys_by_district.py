"""
This script plots maps for the Nurse Shortages analysis at district level:
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

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


def get_yearly_hr_count(_df):

    if 'GenericClinic' not in _df.columns:
        return None

    years = _df['date'].dt.year.rename("year")

    # Expand facility dictionary
    staff_df = _df['GenericClinic'].apply(pd.Series)

    # Extract facility IDs
    facility_ids = [
        int(c.split("FacilityID_")[1].split("_")[0])
        for c in staff_df.columns
    ]

    # Extract cadre names
    cadres = [
        c.split("Officer_")[-1]
        for c in staff_df.columns
    ]

    # Load Master Facility List
    mfl = pd.read_csv(
        Path("./resources/healthsystem/organisation/ResourceFile_Master_Facilities_List.csv")
    ).set_index("Facility_ID")

    # Add district info for facilities at levels 3+ that have nan district info,
    # to avoid these facilities being dropped
    for fid in {128, 129, 130, 131, 132}:
        mfl.loc[fid, "District"] = mfl.loc[fid, "Facility_Name"]

    # Map facilities to districts
    districts = [
        mfl.loc[fid, "District"] if fid in mfl.index else "Unknown"
        for fid in facility_ids
    ]

    # Create MultiIndex columns
    staff_df.columns = pd.MultiIndex.from_arrays(
        [districts, cadres],
        names=["District", "Cadre"]
    )

    # Sum yearly
    staff_df = staff_df.groupby(years).sum()

    # Sum facilities within district/cadre
    staff_df = staff_df.T.groupby(level=[0, 1]).sum().T

    # POP_SCALE = 145.39609
    # staff_df = staff_df * POP_SCALE

    # Convert columns to index
    return staff_df.stack([0, 1])


def extract_staff_counts(results_folder):
    return extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="number_of_hcw_staff",
        custom_generate_series=get_yearly_hr_count,
        do_scaling=False
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
        df["district_of_residence"] = df["district_of_residence"].replace(
            {
                "Likoma": "Nkhata Bay",
            }
        )

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

        df["district_of_residence"] = df["district_of_residence"].replace(
            {
                "Likoma": "Nkhata Bay",
            }
        )

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


# For maps and causes
def extract_total_dalys_by_district(results_folder):
    def get_total_dalys(df):
        df = df.assign(year=df["date"].dt.year)
        df = df[df["year"].between(2027, 2034)]

        df["district_of_residence"] = df["district_of_residence"].replace(
            {
                "Likoma": "Nkhata Bay",
            }
        )

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

        # Combining Likoma and Nkhata Bay
        df["district_of_residence"] = df["district_of_residence"].replace(
            {
                "Likoma": "Nkhata Bay",
            }
        )

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


def calculate_percent_dalys_averted_by_district(
    dalys_by_district,
    baseline_scenario,
):
    pct_diff = (
        -100.0
        * find_difference_relative_to_comparison_series_dataframe(
            dalys_by_district,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    results = {}
    scenario_names = (summarized.columns.get_level_values(0).unique())

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame(
            {
                "mean": summarized[(scenario, "mean")],
                "lower": summarized[(scenario, "lower")],
                "upper": summarized[(scenario, "upper")],
            }
        )

    return results


def calculate_percent_deaths_averted_by_district(
    deaths_by_district,
    baseline_scenario,
):

    pct_diff = (
        -100.0
        * find_difference_relative_to_comparison_series_dataframe(
            deaths_by_district,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)

    results = {}

    scenario_names = (
        summarized.columns.get_level_values(0).unique()
    )

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame({
            "mean": summarized[(scenario, "mean")],
            "lower": summarized[(scenario, "lower")],
            "upper": summarized[(scenario, "upper")],
        })

    return results




def validate_dalys_by_district(
    annual_dalys,
    dalys_by_district,
):
    """
    Check that total DALYs summed across districts equal the
    national DALYs for every scenario, draw and run.
    """

    district_totals = (
        dalys_by_district
        .groupby(level="year")
        .sum()
    )

    print("\n-- DALY VALIDATION --\n")

    comparison = pd.concat(
        {
            "National": annual_dalys,
            "District Total": district_totals,
            "Difference": annual_dalys - district_totals,
        },
        axis=1,
    )

    print(comparison)

    assert np.allclose(
        annual_dalys.values,
        district_totals.values,
    )

    print("\nDALY validation passed.\n")

    return comparison


def validate_deaths_by_district(
    annual_deaths,
    deaths_by_district,
):
    """
    Check that total deaths summed across districts equal the
    national deaths for every scenario, draw and run.
    """

    district_totals = (
        deaths_by_district
        .groupby(level="year")
        .sum()
    )

    print("\n-- DEATH VALIDATION --\n")

    comparison = pd.concat(
        {
            "National": annual_deaths,
            "District Total": district_totals,
            "Difference": annual_deaths - district_totals,
        },
        axis=1,
    )

    print(comparison)

    assert np.allclose(
        annual_deaths.values,
        district_totals.values,
    )

    print("\nDeath validation passed.\n")

    return comparison


def plot_district_maps(gdf, scenario_names, title, colorbar_label):
    vmax = np.nanmax(np.abs(gdf[scenario_names].values))
    ncols = 3
    nrows = math.ceil(len(scenario_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9), constrained_layout=True)
    axes = np.array(axes).flatten()

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

    # Plot maps
    for ax, scenario in zip(axes, scenario_names):
        gdf.plot(column=scenario, cmap="coolwarm_r", edgecolor="black", linewidth=0.4, legend=False,
                            vmin=-vmax, vmax=vmax, ax=ax)
        ax.set_title(label_map[scenario], fontsize=12)
        ax.axis("off")

    # Hide any unused subplot(s)
    for ax in axes[len(scenario_names):]:
        ax.axis("off")

    # Shared colour bar
    sm = plt.cm.ScalarMappable(cmap="coolwarm_r", norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, location="right", shrink=0.85, pad=0.02, label=colorbar_label,)
    fig.suptitle(title, fontsize=15)
    return fig


def plot_percent_dalys_averted_by_district(default_df, improved_df, top_n=30):
    # Default Healthsystem
    default_more = default_df[
        "More Nurses / Default Healthsystem Function"
    ]

    default_cnp = default_df[
        "More CNP staff / Default Healthsystem Function"
    ]

    default_more_district = default_df[
        "More Nurses by District / Default Healthsystem Function"
    ]

    default_cnp_district = default_df[
        "More CNP staff by District / Default Healthsystem Function"
    ]

    default_fewer = default_df[
        "Fewer Nurses / Default Healthsystem Function"
    ]

    # Improved Healthsystem
    improved_more = improved_df[
        "More Nurses / Improved Healthsystem Function"
    ]

    improved_cnp = improved_df[
        "More CNP staff / Improved Healthsystem Function"
    ]

    improved_more_district = improved_df[
        "More Nurses by District / Improved Healthsystem Function"
    ]

    improved_cnp_district = improved_df[
        "More CNP staff by District / Improved Healthsystem Function"
    ]

    improved_fewer = improved_df[
        "Fewer Nurses / Improved Healthsystem Function"
    ]

    default_more = default_more.reindex(district_order)
    default_cnp = default_cnp.reindex(district_order)
    default_more_district = default_more_district.reindex(district_order)
    default_cnp_district = default_cnp_district.reindex(district_order)
    default_fewer = default_fewer.reindex(district_order)

    improved_more = improved_more.reindex(district_order)
    improved_cnp = improved_cnp.reindex(district_order)
    improved_more_district = improved_more_district.reindex(district_order)
    improved_cnp_district = improved_cnp_district.reindex(district_order)
    improved_fewer = improved_fewer.reindex(district_order)

    # Plot
    fig, axes = plt.subplots(ncols=2, figsize=(15, 13), sharey=True)

    panel_data = [
        (
            axes[0],
            [
                ("More nurses", default_more, "steelblue"),
                ("More CNP", default_cnp, "darkgreen"),
                ("More nurses by district", default_more_district, "mediumpurple"),
                ("More CNP by district", default_cnp_district, "orange"),
                ("Fewer nurses", default_fewer, "indianred"),
            ],
            "Default Healthsystem",
        ),
        (
            axes[1],
            [
                ("More nurses", improved_more, "steelblue"),
                ("More CNP", improved_cnp, "darkgreen"),
                ("More nurses by district", improved_more_district, "mediumpurple"),
                ("More CNP by district", improved_cnp_district, "orange"),
                ("Fewer nurses", improved_fewer, "indianred"),
            ],
            "Improved Healthsystem",
        ),
    ]

    offsets = [-0.32, -0.16, 0.0, 0.16, 0.32]

    for ax, scenarios, title in panel_data:
        y = np.arange(len(scenarios[0][1]))
        for offset, (label, df, color) in zip(offsets, scenarios):
            ax.barh(
                y + offset,
                df["mean"],
                height=0.12,
                color=color,
                label=label,
            )

            ax.errorbar(df["mean"], y + offset, xerr=[df["mean"] - df["lower"], df["upper"] - df["mean"],],
                        fmt="none", elinewidth=1, capsize=1.5, color="black", alpha=0.4,)

        ax.axvline(0, color="black")
        ax.set_yticks(y)
        ax.set_yticklabels(scenarios[0][1].index)
        ax.set_xlabel("% DALYs averted")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08),)

    fig.suptitle("% DALYs averted by district \n(2027–2034)")
    fig.tight_layout()
    return fig, axes


def plot_percent_deaths_averted_by_district(default_df, improved_df, top_n=30):
    # Default Healthsystem
    default_more = default_df[
        "More Nurses / Default Healthsystem Function"
    ]

    default_cnp = default_df[
        "More CNP staff / Default Healthsystem Function"
    ]

    default_more_district = default_df[
        "More Nurses by District / Default Healthsystem Function"
    ]

    default_cnp_district = default_df[
        "More CNP staff by District / Default Healthsystem Function"
    ]

    default_fewer = default_df[
        "Fewer Nurses / Default Healthsystem Function"
    ]

    # Improved Healthsystem
    improved_more = improved_df[
        "More Nurses / Improved Healthsystem Function"
    ]

    improved_cnp = improved_df[
        "More CNP staff / Improved Healthsystem Function"
    ]

    improved_more_district = improved_df[
        "More Nurses by District / Improved Healthsystem Function"
    ]

    improved_cnp_district = improved_df[
        "More CNP staff by District / Improved Healthsystem Function"
    ]

    improved_fewer = improved_df[
        "Fewer Nurses / Improved Healthsystem Function"
    ]

    default_more = default_more.reindex(district_order)
    default_cnp = default_cnp.reindex(district_order)
    default_more_district = default_more_district.reindex(district_order)
    default_cnp_district = default_cnp_district.reindex(district_order)
    default_fewer = default_fewer.reindex(district_order)

    improved_more = improved_more.reindex(district_order)
    improved_cnp = improved_cnp.reindex(district_order)
    improved_more_district = improved_more_district.reindex(district_order)
    improved_cnp_district = improved_cnp_district.reindex(district_order)
    improved_fewer = improved_fewer.reindex(district_order)

    # Plot
    fig, axes = plt.subplots(ncols=2, figsize=(15, 13), sharey=True)

    panel_data = [
        (
            axes[0],
            [
                ("More nurses", default_more, "steelblue"),
                ("More CNP", default_cnp, "darkgreen"),
                ("More nurses by district", default_more_district, "mediumpurple"),
                ("More CNP by district", default_cnp_district, "orange"),
                ("Fewer nurses", default_fewer, "indianred"),
            ],
            "Default Healthsystem",
        ),
        (
            axes[1],
            [
                ("More nurses", improved_more, "steelblue"),
                ("More CNP", improved_cnp, "darkgreen"),
                ("More nurses by district", improved_more_district, "mediumpurple"),
                ("More CNP by district", improved_cnp_district, "orange"),
                ("Fewer nurses", improved_fewer, "indianred"),
            ],
            "Improved Healthsystem",
        ),
    ]

    offsets = [-0.32, -0.16, 0.0, 0.16, 0.32]

    for ax, scenarios, title in panel_data:
        y = np.arange(len(scenarios[0][1]))
        for offset, (label, df, color) in zip(offsets, scenarios):
            ax.barh(
                y + offset,
                df["mean"],
                height=0.12,
                color=color,
                label=label,
            )

            ax.errorbar(df["mean"], y + offset, xerr=[df["mean"] - df["lower"], df["upper"] - df["mean"],],
                        fmt="none", elinewidth=1, capsize=1.5, color="black", alpha=0.4,)

        ax.axvline(0, color="black")
        ax.set_yticks(y)
        ax.set_yticklabels(scenarios[0][1].index)
        ax.set_xlabel("% Deaths averted")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08),)

    fig.suptitle("% Deaths averted by district \n(2027–2034)")
    fig.tight_layout()
    return fig, axes


def plot_staff_scaleup_factors_by_district(scaleup_plot_data):
    """
    Plot staff scale-up factors by district.
    Only Default Healthsystem scenarios are plotted.
    Improved scenarios are excluded because their staff scale-up
    factors are the same as the Default scenarios.
    """

    # Scenario/cadre combinations to plot.
    plot_combinations = [
        (
            "Baseline Nurses / Default Healthsystem Function",
            "Nursing_and_Midwifery",
            "Baseline nurses",
            "black",
        ),
        (
            "Fewer Nurses / Default Healthsystem Function",
            "Nursing_and_Midwifery",
            "Fewer nurses",
            "indianred",
        ),
        (
            "More Nurses / Default Healthsystem Function",
            "Nursing_and_Midwifery",
            "More nurses",
            "steelblue",
        ),
        (
            "More Nurses by District / Default Healthsystem Function",
            "Nursing_and_Midwifery",
            "More nurses by district",
            "mediumpurple",
        ),
        (
            "More CNP staff / Default Healthsystem Function",
            "Clinical",
            "More CNP - Clinical",
            "darkgreen",
        ),
        (
            "More CNP staff by District / Default Healthsystem Function",
            "Clinical",
            "More CNP by district - Clinical",
            "limegreen",
        ),
        (
            "More CNP staff / Default Healthsystem Function",
            "Pharmacy",
            "More CNP - Pharmacy",
            "darkorange",
        ),
        (
            "More CNP staff by District / Default Healthsystem Function",
            "Pharmacy",
            "More CNP by district - Pharmacy",
            "gold",
        ),
    ]

    figures = {}

    for scaleup_column, ylabel, title in [
        (
            "Scale2027_2024",
            "Staff scale-up factor (2027/2024)",
            "Staff scale-up factor by district: 2027 vs 2024",
        ),
        (
            "Scale2027_2019",
            "Staff scale-up factor (2027/2019)",
            "Staff scale-up factor by district: 2027 vs 2019",
        ),
    ]:

        fig, ax = plt.subplots(figsize=(16, 8))

        for (scenario, cadre, label, color,) in plot_combinations:
            # Select exactly the requested scenario and cadre.
            scenario_data = scaleup_plot_data[
                (scaleup_plot_data["Scenario"] == scenario)
                & (scaleup_plot_data["Cadre"] == cadre)
            ].copy()

            # Skip if this scenario/cadre combination is absent.
            if scenario_data.empty:
                continue

            scenario_data = scenario_data.sort_values("District")

            ax.scatter(
                scenario_data["District"],
                scenario_data[scaleup_column],
                color=color,
                marker="o",
                label=label,
                alpha=0.7,
            )

        # Reference line: no scale-up relative to the comparison year.
        ax.axhline(1.0, linestyle="--", linewidth=1,)
        ax.set_xlabel("District")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=90,)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",)
        fig.tight_layout()
        figures[scaleup_column] = fig

    return figures


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

    district_order = district_map["ADM2_EN"].tolist()

    district_order = [
        district for district in district_order
        if district != "Likoma"
    ]

    # Optional: load logs
    log = load_pickled_dataframes(results_folder)
    param_names = tuple(StaffingScenario()._scenarios.keys())

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

    # For staff counts
    staff_counts = extract_staff_counts(results_folder)
    staff_counts = set_param_names_as_column_index_level_0(
        staff_counts,
        param_names
    )
    staff_counts_summary = summarize(staff_counts)

    # Staff scale-up factors for all districts
    # Use mean staff counts
    nurses_all_mean = staff_counts_summary.xs("mean", axis=1, level=1)

    scaleup_rows = []

    # Get only Default scenarios.
    scaleup_scenario_cadres = {
        "Baseline Nurses / Default Healthsystem Function":
            ["Nursing_and_Midwifery"],

        "Fewer Nurses / Default Healthsystem Function":
            ["Nursing_and_Midwifery"],

        "More Nurses / Default Healthsystem Function":
            ["Nursing_and_Midwifery"],

        "More Nurses by District / Default Healthsystem Function":
            ["Nursing_and_Midwifery"],

        "More CNP staff / Default Healthsystem Function":
            ["Clinical", "Pharmacy"],

        "More CNP staff by District / Default Healthsystem Function":
            ["Clinical", "Pharmacy"],
    }

    for scenario, cadres in scaleup_scenario_cadres.items():
        # Select the scenario and calculate the yearly staff counts.
        scenario_data = nurses_all_mean[scenario].unstack(
            level="year"
        )

        # Keep only the years required for the scale-up factors.
        scenario_data = scenario_data[[2019, 2024, 2027]].copy()

        # Keep only the relevant cadre(s).
        scenario_data = scenario_data[
            scenario_data.index.get_level_values("Cadre").isin(cadres)
        ].copy()

        # Calculate scale-up factors separately for each
        # District + Cadre combination.
        scenario_data["Scale2027_2024"] = (
            scenario_data[2027] /
            scenario_data[2024]
        )

        scenario_data["Scale2027_2019"] = (
            scenario_data[2027] /
            scenario_data[2019]
        )

        # Store scenario as a column.
        scenario_data["Scenario"] = scenario

        # Convert District and Cadre from the MultiIndex to columns.
        scenario_data = scenario_data.reset_index()

        scaleup_rows.append(scenario_data)

    scaleup_plot_data = pd.concat(
        scaleup_rows,
        ignore_index=True,
    )

    # Keep only districts used in the district plots.
    scaleup_plot_data = scaleup_plot_data[
        scaleup_plot_data["District"].isin(district_order)
    ].copy()

    # Make District categorical so that the plotting order is fixed.
    scaleup_plot_data["District"] = pd.Categorical(
        scaleup_plot_data["District"],
        categories=district_order,
        ordered=True,
    )

    # Sort the final plotting data.
    scaleup_plot_data = scaleup_plot_data.sort_values(
        ["District", "Cadre", "Scenario"]
    )

    # Keeping exactly the same districts and order used in the
    # DALYs and deaths district bar plots
    scaleup_plot_data = scaleup_plot_data[
        scaleup_plot_data["District"].isin(district_order)
    ].copy()

    scaleup_plot_data["District"] = pd.Categorical(
        scaleup_plot_data["District"],
        categories=district_order,
        ordered=True,
    )

    scaleup_plot_data = scaleup_plot_data.sort_values(
        ["District", "Cadre", "Scenario"]
    )

    # Plot staff scale-up factors by district
    scaleup_figures = plot_staff_scaleup_factors_by_district(
        scaleup_plot_data
    )

    # Staff scaling: nurses

    nurses = staff_counts_summary.xs(
        "Nursing_and_Midwifery",
        level="Cadre"
    )

    nurses = nurses.loc[
        nurses.index.get_level_values("year").isin(
            [2019, 2024, 2027]
        )
    ]

    nurses_mean = nurses.xs("mean", axis=1, level=1)

    nurse_scenarios = [
        "More Nurses / Default Healthsystem Function",
        "More Nurses by District / Default Healthsystem Function",
        "More Nurses / Improved Healthsystem Function",
        "More Nurses by District / Improved Healthsystem Function",
    ]

    nurses_mean = nurses_mean[nurse_scenarios]

    nurse_rows = []

    for scenario in nurse_scenarios:
        scenario_data = (
            nurses_mean[[scenario]]
            .unstack(level="year")
        )

        # Remove scenario level from columns
        scenario_data.columns = scenario_data.columns.droplevel(0)

        scenario_data = scenario_data.rename(
            columns={
                2019: "Staff2019",
                2024: "Staff2024",
                2027: "Staff2027",
            }
        )

        scenario_data["Scale2027_2024"] = (
            scenario_data["Staff2027"] /
            scenario_data["Staff2024"]
        )

        scenario_data["Scale2027_2019"] = (
            scenario_data["Staff2027"] /
            scenario_data["Staff2019"]
        )

        scenario_data["Scenario"] = scenario

        scenario_data = scenario_data.reset_index()

        nurse_rows.append(scenario_data)

    nurse_summary = pd.concat(
        nurse_rows,
        ignore_index=True,
    )

    nurse_summary["Staff_Type"] = "Nurses"

    # Staff scaling: CNP
    cnp_cadres = [
        "Clinical",
        "Nursing_and_Midwifery",
        "Pharmacy",
    ]

    cnp_scenarios = [
        "More CNP staff / Default Healthsystem Function",
        "More CNP staff by District / Default Healthsystem Function",
        "More CNP staff / Improved Healthsystem Function",
        "More CNP staff by District / Improved Healthsystem Function",
    ]

    # Select only the three CNP cadres
    cnp = staff_counts_summary.loc[
        staff_counts_summary.index.get_level_values("Cadre").isin(cnp_cadres)
    ].copy()

    cnp = cnp.loc[
        cnp.index.get_level_values("year").isin(
            [2019, 2024, 2027]
        )
    ]

    # Select mean across draws
    cnp_mean = cnp.xs("mean", axis=1, level=1)

    # Keep only the CNP scenarios
    cnp_mean = cnp_mean[cnp_scenarios]

    # Sum the three CNP cadres within each district/year/scenario
    cnp_mean = (
        cnp_mean
        .groupby(
            level=["District", "year"]
        )
        .sum()
    )

    cnp_rows = []

    for scenario in cnp_scenarios:
        scenario_data = (
            cnp_mean[[scenario]]
            .unstack(level="year")
        )

        # Remove scenario level from columns
        scenario_data.columns = scenario_data.columns.droplevel(0)

        scenario_data = scenario_data.rename(
            columns={
                2019: "Staff2019",
                2024: "Staff2024",
                2027: "Staff2027",
            }
        )

        scenario_data["Scale2027_2024"] = (
            scenario_data["Staff2027"] /
            scenario_data["Staff2024"]
        )

        scenario_data["Scale2027_2019"] = (
            scenario_data["Staff2027"] /
            scenario_data["Staff2019"]
        )

        scenario_data["Scenario"] = scenario

        scenario_data = scenario_data.reset_index()
        cnp_rows.append(scenario_data)

    cnp_summary = pd.concat(cnp_rows, ignore_index=True,)

    cnp_summary["Staff_Type"] = "CNP"

    # Combine nurse + CNP staff summaries
    staff_summary = pd.concat(
        [
            nurse_summary,
            cnp_summary,
        ],
        ignore_index=True,
    )

    # Arrange columns
    staff_summary = staff_summary[
        [
            "District",
            "Staff_Type",
            "Staff2019",
            "Staff2024",
            "Staff2027",
            "Scale2027_2024",
            "Scale2027_2019",
            "Scenario",
        ]
    ]

    staff_summary.to_excel(
        results_folder / "district_staff_scaling.xlsx",
        index=False,
    )

    # National DALYs
    annual_dalys = extract_annual_dalys(results_folder)

    dalys_by_district = extract_total_dalys_by_district(results_folder)
    dalys_by_district_for_validation = extract_dalys_by_district(results_folder)

    dalys_validation = validate_dalys_by_district(
        annual_dalys,
        dalys_by_district_for_validation,
    )

    # Sum district DALYs for each year
    district_daly_totals = (
        dalys_by_district_for_validation
        .groupby(level="year")
        .sum()
    )

    # DALYs Default Healthsystem
    dalys_by_district = set_param_names_as_column_index_level_0(
        dalys_by_district,
        param_names
    )

    # Keep only Default Healthsystem scenarios
    dalys_by_district_default = dalys_by_district.loc[
                                :,
                                dalys_by_district.columns.get_level_values(0).isin(default_hs_scenarios)
                                ]

    # Calculate % DALYs averted using run-to-run comparisons
    percent_dalys_default = calculate_percent_dalys_averted_by_district(
        dalys_by_district_default,
        baseline_scenario=baseline_scenario,
    )

    # Convert dictionary of dataframes into one dataframe containing the means
    default_pct = pd.DataFrame(
        {
            scenario: df["mean"]
            for scenario, df in percent_dalys_default.items()
        }
    )

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
            "More CNP staff / Default Healthsystem Function",
            "Fewer Nurses / Default Healthsystem Function",
            "More Nurses by District / Default Healthsystem Function",
            "More CNP staff by District / Default Healthsystem Function",
        ],
        "% DALYs averted (vs Baseline): Default Healthsystem",
        "DALYs averted in percentage",
    )

    # Keep only Improved Healthsystem scenarios
    dalys_by_district_improved = dalys_by_district.loc[
                                 :,
                                 dalys_by_district.columns.get_level_values(0).isin(improved_hs_scenarios)
                                 ]

    # Calculate % DALYs averted using run-to-run comparisons
    percent_dalys_improved = calculate_percent_dalys_averted_by_district(
        dalys_by_district_improved,
        baseline_scenario=baseline_improved_scenario,
    )

    # Convert dictionary of dataframes into one dataframe containing the means
    improved_pct = pd.DataFrame(
        {
            scenario: df["mean"]
            for scenario, df in percent_dalys_improved.items()
        }
    )

    # Create DALY summary for ALL scenarios
    # Combine Default and Improved Healthsystem results
    all_pct_dalys = pd.concat(
        [
            default_pct,
            improved_pct,
        ],
        axis=1,
    )

    # Convert from wide format to long format
    dalys_summary = (
        all_pct_dalys
        .stack()
        .reset_index()
    )

    dalys_summary.columns = [
        "District",
        "Scenario",
        "Percent_DALYs_Averted",
    ]

    print("\n--- ALL DALY SCENARIOS ---")
    print(dalys_summary.head(20))

    print("\n--- DALY scenarios ---")
    print(dalys_summary["Scenario"].unique())

    # Merge all DALY scenarios with staff scaling data
    staff_summary = staff_summary.merge(
        dalys_summary,
        on=["District", "Scenario"],
        how="left",
    )

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
            "More CNP staff / Improved Healthsystem Function",
            "Fewer Nurses / Improved Healthsystem Function",
            "More Nurses by District / Improved Healthsystem Function",
            "More CNP staff by District / Improved Healthsystem Function",
        ],
        "% DALYs averted (vs Baseline): Improved Healthsystem",
        "DALYs averted in percentage",
    )

    fig_dalys_bar, ax_dalys_bar = (
        plot_percent_dalys_averted_by_district(
            percent_dalys_default,
            percent_dalys_improved,
        )
    )

    annual_deaths = extract_annual_deaths(results_folder)

    # Deaths Default Healthsystem
    deaths_by_district = extract_total_deaths_by_district(results_folder)
    deaths_by_district_for_validation = extract_deaths_by_district(results_folder)

    deaths_validation = validate_deaths_by_district(
        annual_deaths,
        deaths_by_district_for_validation,
    )

    # Sum district deaths for each year
    district_death_totals = (
        deaths_by_district_for_validation
        .groupby(level="year")
        .sum()
    )

    deaths_by_district = set_param_names_as_column_index_level_0(
        deaths_by_district,
        param_names
    )

    # Deaths: Default Heathsystem
    # Keep only Default Healthsystem scenarios
    deaths_by_district_default = deaths_by_district.loc[
                                 :,
                                 deaths_by_district.columns.get_level_values(0).isin(
                                     default_hs_scenarios
                                 )
                                 ]

    # Calculate % deaths averted using run-to-run comparisons
    percent_deaths_default = calculate_percent_deaths_averted_by_district(
        deaths_by_district_default,
        baseline_scenario=baseline_scenario,
    )

    # Convert dictionary of dataframes into one dataframe containing the means
    default_pct_deaths = pd.DataFrame(
        {
            scenario: df["mean"]
            for scenario, df in percent_deaths_default.items()
        }
    )

    # Deaths: Improved Healthsystem
    deaths_by_district_improved = deaths_by_district.loc[
                                  :,
                                  deaths_by_district.columns.get_level_values(0).isin(
                                      improved_hs_scenarios
                                  )
                                  ]

    percent_deaths_improved = calculate_percent_deaths_averted_by_district(
        deaths_by_district_improved,
        baseline_scenario=baseline_improved_scenario,
    )

    improved_pct_deaths = pd.DataFrame(
        {
            scenario: df["mean"]
            for scenario, df in percent_deaths_improved.items()
        }
    )

    # Table for Excel create deaths summary for all scenarios
    # Default + Improved Healthsystem scenarios
    all_pct_deaths = pd.concat(
        [
            default_pct_deaths,
            improved_pct_deaths,
        ],
        axis=1,
    )

    # Convert from wide format to long format
    deaths_summary = (all_pct_deaths.stack().reset_index())

    deaths_summary.columns = [
        "District",
        "Scenario",
        "Percent_Deaths_Averted",
    ]

    # Merge DALYs + Death with staff scaling

    staff_summary = staff_summary.merge(
        deaths_summary,
        on=["District", "Scenario"],
        how="left",
    )

    # Short scenario names for final Excel output
    staff_summary["Scenario"] = staff_summary["Scenario"].replace({
        "More Nurses / Default Healthsystem Function":
            "More Nurses / Default",

        "More Nurses by District / Default Healthsystem Function":
            "More Nurses by District / Default",

        "More Nurses / Improved Healthsystem Function":
            "More Nurses / Improved",

        "More Nurses by District / Improved Healthsystem Function":
            "More Nurses by District / Improved",

        "More CNP staff / Default Healthsystem Function":
            "More CNP / Default",

        "More CNP staff by District / Default Healthsystem Function":
            "More CNP by District / Default",

        "More CNP staff / Improved Healthsystem Function":
            "More CNP / Improved",

        "More CNP staff by District / Improved Healthsystem Function":
            "More CNP by District / Improved",
    })

    staff_summary.to_excel(
        results_folder / "district_staff_scaling_health_outcomes.xlsx",
        index=False,
    )

    district_map_default_deaths = district_map.merge(
        default_pct_deaths,
        left_on="ADM2_EN",
        right_index=True,
        how="left",
    )

    fig_deaths_default_maps = plot_district_maps(
        district_map_default_deaths,
        [
            "More Nurses / Default Healthsystem Function",
            "More CNP staff / Default Healthsystem Function",
            "Fewer Nurses / Default Healthsystem Function",
            "More Nurses by District / Default Healthsystem Function",
            "More CNP staff by District / Default Healthsystem Function",
        ],
        "% Deaths averted (vs Baseline): Default Healthsystem",
        "Deaths averted in percentage",
    )

    district_map_improved_deaths = district_map.merge(
        improved_pct_deaths,
        left_on="ADM2_EN",
        right_index=True,
        how="left",
    )

    fig_deaths_improved_maps = plot_district_maps(
        district_map_improved_deaths,
        [
            "More Nurses / Improved Healthsystem Function",
            "More CNP staff / Improved Healthsystem Function",
            "Fewer Nurses / Improved Healthsystem Function",
            "More Nurses by District / Improved Healthsystem Function",
            "More CNP staff by District / Improved Healthsystem Function",
        ],
        "% Deaths averted (vs Baseline): Improved Healthsystem",
        "Deaths averted in percentage",
    )

    fig_deaths_bar, ax_deaths_bar = (
        plot_percent_deaths_averted_by_district(
            percent_deaths_default,
            percent_deaths_improved,
        )
    )

    if args.save_figures:
        fig_dalys_default_maps.savefig(
            results_folder /
            "district_dalys_default.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        fig_dalys_improved_maps.savefig(
            results_folder /
            "district_dalys_improved.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        fig_deaths_default_maps.savefig(
            results_folder /
            "district_deaths_default.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        fig_deaths_improved_maps.savefig(
            results_folder /
            "district_deaths_improved.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        fig_dalys_bar.savefig(
            results_folder /
            "district_dalys_barplots.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        fig_deaths_bar.savefig(
            results_folder /
            "district_deaths_barplots.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        scaleup_figures["Scale2027_2024"].savefig(
            results_folder /
            "staff_scaleup_2027_2024_by_district.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        scaleup_figures["Scale2027_2019"].savefig(
            results_folder /
            "staff_scaleup_2027_2019_by_district.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        validation_file = (
            results_folder /
            "district_vs_national_validation.xlsx"
        )

        with pd.ExcelWriter(
            validation_file,
            engine="openpyxl",
        ) as writer:
            # DALYs
            annual_dalys.to_excel(
                writer,
                sheet_name="National_DALYs",
            )

            dalys_by_district_for_validation.to_excel(
                writer,
                sheet_name="District_DALYs",
            )

            district_daly_totals.to_excel(
                writer,
                sheet_name="District_DALY_Totals",
            )

            dalys_validation.to_excel(
                writer,
                sheet_name="DALY_Comparison",
            )

            # Deaths

            annual_deaths.to_excel(
                writer,
                sheet_name="National_Deaths",
            )

            deaths_by_district_for_validation.to_excel(
                writer,
                sheet_name="District_Deaths",
            )

            district_death_totals.to_excel(
                writer,
                sheet_name="District_Death_Totals",
            )

            deaths_validation.to_excel(
                writer,
                sheet_name="Death_Comparison",
            )
