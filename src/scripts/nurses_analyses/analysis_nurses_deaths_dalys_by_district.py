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
        gdf.plot(column=scenario, cmap="coolwarm", edgecolor="black", linewidth=0.4, legend=False,
                            vmin=-vmax, vmax=vmax, ax=ax)
        ax.set_title(label_map[scenario], fontsize=12)
        ax.axis("off")

    # Hide any unused subplot(s)
    for ax in axes[len(scenario_names):]:
        ax.axis("off")

    # Shared colour bar
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, location="right", shrink=0.85, pad=0.02, label=colorbar_label,)
    fig.suptitle(title, fontsize=15)
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

    # Optional: load logs
    log = load_pickled_dataframes(results_folder)
    param_names = tuple(StaffingScenario()._scenarios.keys())

    # For staff counts
    staff_counts = extract_staff_counts(results_folder)
    staff_counts = set_param_names_as_column_index_level_0(
        staff_counts,
        param_names
    )
    staff_counts_summary = summarize(staff_counts)
    # Get nurses only
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

    nurses_mean = nurses_mean[
        [
            "More Nurses / Default Healthsystem Function",
            "More Nurses by District / Default Healthsystem Function",
        ]
    ]

    more_nurses = nurses_mean[
        ["More Nurses / Default Healthsystem Function"]
    ].unstack(level="year")

    more_nurses_district = nurses_mean[
        ["More Nurses by District / Default Healthsystem Function"]
    ].unstack(level="year")

    more_nurses.columns = [
        "Staff2019",
        "Staff2024",
        "Staff2027",
    ]
    print("\n---Inspecting renamed More Nurses---")
    print(more_nurses.head())

    more_nurses_district.columns = [
        "Staff2019",
        "Staff2024",
        "Staff2027",
    ]
    print("\n---Inspecting renamed More Nurses by District---")
    print(more_nurses_district.head())

    # Scaling factors
    more_nurses["Scale2027_2024"] = (
        more_nurses["Staff2027"] /
        more_nurses["Staff2024"]
    )

    more_nurses["Scale2027_2019"] = (
        more_nurses["Staff2027"] /
        more_nurses["Staff2019"]
    )

    more_nurses_district["Scale2027_2024"] = (
        more_nurses_district["Staff2027"] /
        more_nurses_district["Staff2024"]
    )

    more_nurses_district["Scale2027_2019"] = (
        more_nurses_district["Staff2027"] /
        more_nurses_district["Staff2019"]
    )

    more_nurses["Scenario"] = "More Nurses"
    more_nurses_district["Scenario"] = "More Nurses by District"

    staff_summary = pd.concat(
        [
            more_nurses,
            more_nurses_district,
        ]
    )

    staff_summary.to_excel(
        results_folder / "district_staff_scaling.xlsx",
        index=True,
    )

    # NATIONAL DALYs
    annual_dalys = extract_annual_dalys(results_folder)

    dalys_by_district = extract_total_dalys_by_district(results_folder)

    # DALYs Default Healthsystem
    dalys_by_district = set_param_names_as_column_index_level_0(
        dalys_by_district,
        param_names
    )

    district_mean = (dalys_by_district.groupby(level=0, axis=1).mean())

    default_baseline = (
        "Baseline Nurses / Default Healthsystem Function"
    )

    default_cols = [
        c for c in district_mean.columns
        if "Default Healthsystem Function" in c
    ]

    default_df_dalys = district_mean[default_cols]

    default_pct = (
        default_df_dalys
        .subtract(default_df_dalys[default_baseline], axis=0)
        .divide(default_df_dalys[default_baseline], axis=0)
        * 100
    )

    default_pct = default_pct.drop(columns=default_baseline)

    # DALYs table for the two nurse expansion scenarios
    dalys_summary = default_pct[
        [
            "More Nurses / Default Healthsystem Function",
            "More Nurses by District / Default Healthsystem Function",
        ]
    ].copy()
    # Renaming the columns
    dalys_summary.columns = [
        "More Nurses",
        "More Nurses by District",
    ]

    dalys_summary = (
        dalys_summary
        .stack()
        .reset_index()
    )

    dalys_summary.columns = [
        "District",
        "Scenario",
        "Percent_DALYs_Averted",
    ]
    print("\n---DALYs Summary after formatting and renaming---")
    print(dalys_summary.head())

    staff_summary = staff_summary.merge(
        dalys_summary,
        on=["District", "Scenario"],
        how="left",
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
            "Fewer Nurses / Default Healthsystem Function",
            "More CNP staff / Default Healthsystem Function",
            "More Nurses by District / Default Healthsystem Function",
            "More CNP staff by District / Default Healthsystem Function",
        ],
        "% DALYs averted (vs Baseline): Default Healthsystem",
        "% change in DALYs (vs Baseline)",
    )

    # DALYs Improved Healthsystem
    improved_baseline = (
        "Baseline Nurses / Improved Healthsystem Function"
    )

    improved_cols = [
        c for c in district_mean.columns
        if "Improved Healthsystem Function" in c
    ]

    improved_df_dalys = district_mean[improved_cols]

    improved_pct = (
        improved_df_dalys
        .subtract(improved_df_dalys[improved_baseline], axis=0)
        .divide(improved_df_dalys[improved_baseline], axis=0)
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
        "% change in DALYs (vs Baseline)",
    )

    # Deaths Default Healthsystem
    deaths_by_district = extract_total_deaths_by_district(results_folder)
    deaths_by_district = set_param_names_as_column_index_level_0(deaths_by_district, param_names)
    district_mean_deaths = (deaths_by_district.groupby(level=0, axis=1).mean())

    # default_baseline = (
    #     "Baseline Nurses / Default Healthsystem Function"
    # )

    default_cols_deaths = [
        c for c in district_mean.columns
        if "Default Healthsystem Function" in c
    ]

    default_df_deaths = district_mean_deaths[default_cols_deaths]

    default_pct_deaths = (
        default_df_deaths
        .subtract(default_df_deaths[default_baseline], axis=0)
        .divide(default_df_deaths[default_baseline], axis=0)
        * 100
    )

    default_pct_deaths = default_pct_deaths.drop(columns=default_baseline)

    deaths_summary = default_pct_deaths[
        [
            "More Nurses / Default Healthsystem Function",
            "More Nurses by District / Default Healthsystem Function",
        ]
    ].copy()

    deaths_summary.columns = [
        "More Nurses",
        "More Nurses by District",
    ]

    deaths_summary = (
        deaths_summary
        .stack()
        .reset_index()
    )

    deaths_summary.columns = [
        "District",
        "Scenario",
        "Percent_Deaths_Averted",
    ]

    staff_summary = staff_summary.merge(
        deaths_summary,
        on=["District", "Scenario"],
        how="left",
    )
    print("\n---Table with DALYs and Deaths---")
    print(staff_summary.head())

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
            "Fewer Nurses / Default Healthsystem Function",
            "More CNP staff / Default Healthsystem Function",
            "More Nurses by District / Default Healthsystem Function",
            "More CNP staff by District / Default Healthsystem Function",
        ],
        "% Deaths averted (vs Baseline): Default Healthsystem",
        "% change in Deaths (vs Baseline)",
    )

    # Deaths Improved Healthsystem
    improved_cols_deaths = [
        c for c in district_mean.columns
        if "Improved Healthsystem Function" in c
    ]

    improved_df_deaths = district_mean_deaths[improved_cols_deaths]

    improved_pct_deaths = (
        improved_df_deaths
        .subtract(improved_df_deaths[improved_baseline], axis=0)
        .divide(improved_df_deaths[improved_baseline], axis=0)
        * 100
    )

    improved_pct_deaths = improved_pct_deaths.drop(columns=improved_baseline)

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
            "Fewer Nurses / Improved Healthsystem Function",
            "More CNP staff / Improved Healthsystem Function",
            "More Nurses by District / Improved Healthsystem Function",
            "More CNP staff by District / Improved Healthsystem Function",
        ],
        "% Deaths averted (vs Baseline): Improved Healthsystem",
        "% change in Deaths (vs Baseline)",
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
