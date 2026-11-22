import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario
from tlo.analysis.utils import extract_results, load_pickled_dataframes, summarize

DALY_METADATA_COLUMNS = {"date", "year", "sex", "age_range", "li_wealth", "district_of_residence"}

# Group individual causes into broader health areas.
CAUSE_GROUP = {
    "AIDS": "HIV/AIDS",
    "TB (non-AIDS)": "TB (non-AIDS)",
    "Malaria": "Malaria",
    "Childhood Diarrhoea": "RMNCH",
    "Childhood Undernutrition": "RMNCH",
    "Congenital birth defects": "RMNCH",
    "Lower respiratory infections": "RMNCH",
    "Maternal Disorders": "RMNCH",
    "Measles": "RMNCH",
    "Neonatal Disorders": "RMNCH",
    "Schistosomiasis": "RMNCH",
    "COPD": "NCDs",
    "Cancer (Bladder)": "NCDs",
    "Cancer (Breast)": "NCDs",
    "Cancer (Cervix)": "RMNCH",
    "Cancer (Oesophagus)": "NCDs",
    "Cancer (Other)": "NCDs",
    "Cancer (Prostate)": "NCDs",
    "Depression / Self-harm": "NCDs",
    "Diabetes": "NCDs",
    "Epilepsy": "NCDs",
    "Heart Disease": "NCDs",
    "Kidney Disease": "NCDs",
    "Lower Back Pain": "NCDs",
    "Stroke": "NCDs",
    "Transport Injuries": "Transport Injuries",
    "Other": "Other",
}

CAUSE_GROUP_ORDER = [
    "HIV/AIDS",
    "TB (non-AIDS)",
    "Malaria",
    "RMNCH",
    "NCDs",
    "Transport Injuries",
    "Other",
]

CAUSE_GROUP_COLOR = {
    "HIV/AIDS": "deepskyblue",
    "TB (non-AIDS)": "mediumslateblue",
    "Malaria": "khaki",
    "RMNCH": "mediumaquamarine",
    "NCDs": "violet",
    "Transport Injuries": "lightsalmon",
    "Other": "dimgrey",
}


def prepare_plot_data_for_excel(data, category_name="Category"):
    """
    Prepare the mean and error-bar values used in a plot
    for export to Excel.
    For baseline-comparison plots, data is a DataFrame indexed
    by scenario.
    For cause and age-group plots, data is a dictionary of
    scenario names and DataFrames indexed by cause or age group.
    """
    if isinstance(data, pd.DataFrame):
        # Baseline-comparison plots
        result = data.copy()
        result.index.name = "Scenario"
        result = result.reset_index()

    else:
        # Cause and age-group plots
        frames = []

        for scenario, scenario_df in data.items():
            temp = scenario_df.copy()
            temp.index.name = category_name
            temp = temp.reset_index()
            temp.insert(0, "Scenario", scenario)
            frames.append(temp)

        result = pd.concat(frames, ignore_index=True)

    # Include the numerical size of each error bar.
    if {"mean", "lower", "upper"}.issubset(result.columns):
        result["Lower error"] = result["mean"] - result["lower"]
        result["Upper error"] = result["upper"] - result["mean"]

        # Arrange the columns in a useful order.
        result = result[
            [
                col for col in [
                "Scenario",
                category_name,
                "mean",
                "lower",
                "upper",
                "Lower error",
                "Upper error",
            ]
                if col in result.columns
            ]
        ]

        result = result.rename(
            columns={
                "mean": "Mean",
                "lower": "Lower bound",
                "upper": "Upper bound",
            }
        )

    return result


def export_nurse_plot_data_to_excel(results_folder, plot_data):
    """
    Export the data underlying the selected nurse-scenario
    plots into separate worksheets in one Excel workbook.
    """
    output_file = results_folder / "nurses_scenario_plot_data.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, data, category_name in plot_data:
            excel_df = prepare_plot_data_for_excel(
                data,
                category_name=category_name,
            )

            excel_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            # Make columns easier to read in Excel.
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                column_letter = column_cells[0].column_letter
                worksheet.column_dimensions[column_letter].width = min(
                    max_length + 2, 45
                )

    print(f"Excel plot data saved to: {output_file}")


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


def aggregate_causes_by_health_area(data):
    """
    Aggregate individual causes into broader health areas.
    """

    data = data.copy()

    # Identify causes not explicitly included in the mapping
    unmapped_causes = [
        cause for cause in data.index
        if cause not in CAUSE_GROUP
    ]

    if unmapped_causes:
        print(
            "WARNING: The following causes were not explicitly "
            "mapped and will be assigned to Other:",
            unmapped_causes,
        )

    # Mapping individual causes to broader health areas
    data.index = data.index.map(
        lambda cause: CAUSE_GROUP.get(cause, "Other")
    )

    # Summing individual causes belonging to the same health area
    data = data.groupby(level=0).sum()

    # Keep only health areas that actually exist in the data,
    # while preserving the desired order
    available_groups = [
        group
        for group in CAUSE_GROUP_ORDER
        if group in data.index
    ]

    data = data.reindex(available_groups)
    return data


# Extract DALYs by cause and district
def extract_dalys_by_cause_and_district(
    results_folder,
    key="dalys_stacked",
    district_col="district_of_residence",
    years=(2027, 2034),
):
    """
    Return DALYs by district and cause, aggregated across `years`
    (inclusive), for every draw and run.

    Result: rows are a (District, Cause) MultiIndex; columns are the usual
    (draw, run) MultiIndex from extract_results.

    key: the healthburden log to read. It must contain a district column
        (`district_col`). If your `dalys_stacked` log has no district,
        use the log that does (check the error message for its columns).
    """

    def get_dalys_by_cause_and_district(df: pd.DataFrame) -> pd.Series:
        if district_col not in df.columns:
            # Raise ValueError, not KeyError: extract_results treats a KeyError
            # as a failed run and would silently drop every run.
            raise ValueError(
                f"Column '{district_col}' not found in the '{key}' log. "
                f"Columns available: {list(df.columns)}"
            )

        if "year" not in df.columns:
            df = df.assign(year=df["date"].dt.year)
        df = df[df["year"].between(*years)].copy()

        # Count Likoma as part of Nkhata Bay.
        df[district_col] = df[district_col].replace(
            {
                "Likoma": "Nkhata Bay",
            }
        )

        cause_cols = [
            c for c in df.columns
            if c not in DALY_METADATA_COLUMNS
               and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Sum DALYs for each district and cause.
        by_district = df.groupby(district_col)[cause_cols].sum()
        out = by_district.stack()
        out.index.names = ["District", "Cause"]
        return out

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key=key,
        custom_generate_series=get_dalys_by_cause_and_district,
        do_scaling=True,
    )


def aggregate_causes_by_health_area_by_district(data):
    """
    Aggregate individual causes into broader health areas within each
    district. Input rows: (District, Cause); output rows: (District, Health area).
    """
    data = data.copy()
    causes = data.index.get_level_values("Cause")

    unmapped_causes = sorted(set(causes) - set(CAUSE_GROUP))
    if unmapped_causes:
        print(
            "WARNING: The following causes were not explicitly "
            "mapped and will be assigned to Other:",
            unmapped_causes,
        )

    areas = [CAUSE_GROUP.get(cause, "Other") for cause in causes]
    data.index = pd.MultiIndex.from_arrays(
        [data.index.get_level_values("District"), areas],
        names=["District", "Health area"],
    )
    return data.groupby(level=["District", "Health area"]).sum()


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


def calculate_dalys_averted(
    annual_dalys,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate DALYs averted using run-to-run differences.
    """
    years = annual_dalys.index.astype(int)
    year_mask = np.isin(years, list(comparison_years))

    annual_dalys = annual_dalys.loc[year_mask]
    annual_dalys_agg = annual_dalys.sum(axis=0)

    num_diff = pd.DataFrame(
        -1.0
        * find_difference_relative_to_comparison_series(
            annual_dalys_agg,
            comparison=baseline_scenario,
            scaled=False,
        )
    ).T

    summarized = summarize(num_diff)
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


def calculate_dalys_averted_by_cause(
    dalys_by_cause,
    baseline_scenario,
):
    num_diff = (
        -1.0
        * find_difference_relative_to_comparison_series_dataframe(
        dalys_by_cause,
        comparison=baseline_scenario,
        scaled=False,
    )
    )

    summarized = summarize(num_diff)
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

    yerr_default = np.vstack([means_default - lowers_default, uppers_default - means_default, ])

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

    yerr_improved = np.vstack([means_improved - lowers_improved, uppers_improved - means_improved, ])
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


def plot_dalys_averted_comparison(default_df, improved_df):
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

    yerr_default = np.vstack([means_default - lowers_default, uppers_default - means_default, ])

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
    ax_default.set_ylabel("DALYs averted")
    ax_default.grid(axis="y", alpha=0.3)

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

    yerr_improved = np.vstack([means_improved - lowers_improved, uppers_improved - means_improved, ])
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
    ax_improved.set_ylabel("DALYs averted")
    ax_improved.grid(axis="y", alpha=0.3)
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

    yerr_default = np.vstack([means_default - lowers_default, uppers_default - means_default, ])
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

    yerr_improved = np.vstack([means_improved - lowers_improved, uppers_improved - means_improved, ])
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


# Shared plotting function for percentage DALYs/deaths averted by health area
def _plot_percent_averted_by_cause(
    default_df,
    improved_df,
    outcome_label,
):
    """
    Plot percentage DALYs or deaths averted by health area
    """

    # Health areas available in the results.
    # health_areas = [
    #     group
    #     for group in CAUSE_GROUP_ORDER
    #     if group in default_df[
    #         "Fewer Nurses / Default Healthsystem Function"
    #     ].index
    #     and group in improved_df[
    #         "Fewer Nurses / Improved Healthsystem Function"
    #     ].index
    # ]

    health_areas = sorted(
        [
            area
            for area in CAUSE_GROUP_ORDER
            if area != "Other"
               and area in default_df[
                   "Fewer Nurses / Default Healthsystem Function"
               ].index
               and area in improved_df[
                   "Fewer Nurses / Improved Healthsystem Function"
               ].index
        ],
        key=str.lower,
    )

    # Scenario order and colours.
    scenarios = [
        ("Fewer nurses", "Fewer Nurses", "indianred"),
        ("More nurses", "More Nurses", "steelblue"),
        ("More CNP", "More CNP staff", "darkgreen"),
        ("More nurses by district", "More Nurses by District", "mediumpurple"),
        ("More CNP by district", "More CNP staff by District", "orange"),
    ]

    # Figure dimensions.
    figure_width = 15
    figure_height = max(10, len(health_areas) * 1.8)

    # Spacing between health areas.
    group_spacing = 1.5

    y = np.arange(len(health_areas)) * group_spacing

    # Width/height allocated to the five scenario bars.
    bar_height = 0.14

    offsets = (
                  np.arange(len(scenarios))
                  - (len(scenarios) - 1) / 2
              ) * bar_height

    def plot_healthsystem(data, healthsystem):

        fig, ax = plt.subplots(
            figsize=(figure_width, figure_height)
        )

        for i, (label, scenario_prefix, color) in enumerate(scenarios):
            scenario_name = (
                f"{scenario_prefix} / "
                f"{healthsystem} Healthsystem Function"
            )

            # Retrieve values in the intended health-area order.
            scenario_df = data[scenario_name].reindex(
                health_areas
            )

            means = scenario_df["mean"].to_numpy()
            lower = scenario_df["lower"].to_numpy()
            upper = scenario_df["upper"].to_numpy()

            # Position of each scenario bar.
            ypos = y + offsets[i]

            # Draw horizontal bars.
            ax.barh(
                ypos,
                means,
                height=bar_height * 0.9,
                color=color,
                label=label,
                xerr=np.vstack([
                    means - lower,
                    upper - means,
                ]),
                capsize=3,
                error_kw={
                    "elinewidth": 0.9,
                    "capthick": 0.9,
                },
            )

        # Reference line at zero.
        ax.axvline(0, color="black", linewidth=1, )

        # Health-area labels.
        ax.set_yticks(y)
        ax.set_yticklabels(health_areas)

        # Colour health-area labels according to their groups.
        for tick, group in zip(
            ax.get_yticklabels(),
            health_areas,
        ):
            tick.set_color(CAUSE_GROUP_COLOR[group])
            tick.set_fontweight("bold")

        # Separate health areas with dashed horizontal lines.
        for i in range(len(health_areas) - 1):
            separator_position = (
                                     y[i] + y[i + 1]
                                 ) / 2

            ax.axhline(separator_position, color="grey", linestyle="--", linewidth=1, alpha=0.7, )

        # Put the first health area at the top.
        ax.invert_yaxis()

        # Axis labels.
        ax.set_xlabel(f"% {outcome_label} averted", fontsize=12, )
        ax.set_ylabel("Health area", fontsize=12, )

        # Grid and formatting.
        ax.grid(axis="x", alpha=0.3, )

        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=11, length=0, pad=12, )
        ax.tick_params(axis="x", labelsize=10, )
        ax.set_yticks(y)
        ax.set_yticklabels(health_areas)
        ax.tick_params(axis="y", labelcolor="black")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=True, )
        # Leave space for labels and legend
        fig.subplots_adjust(left=0.20, right=0.98, top=0.97, bottom=0.15, )
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

    return (
        fig_default,
        ax_default,
        fig_improved,
        ax_improved,
    )


def _plot_averted_by_cause(
    default_df,
    improved_df,
    outcome_label,
):
    """
    Plot DALYs or deaths averted by health area
    """

    available = (
        set(default_df["Fewer Nurses / Default Healthsystem Function"].index)
        & set(improved_df["Fewer Nurses / Improved Healthsystem Function"].index)
    )

    health_areas = sorted(
        (area for area in CAUSE_GROUP_ORDER if area in available and area != "Other"),
        key=str.lower,
    )

    # Scenario order and colours.
    scenarios = [
        ("Fewer nurses", "Fewer Nurses", "indianred"),
        ("More nurses", "More Nurses", "steelblue"),
        ("More CNP", "More CNP staff", "darkgreen"),
        ("More nurses by district", "More Nurses by District", "mediumpurple"),
        ("More CNP by district", "More CNP staff by District", "orange"),
    ]

    # Figure dimensions.
    figure_width = 15
    figure_height = max(10, len(health_areas) * 1.8)

    # Spacing between health areas.
    group_spacing = 1.5

    y = np.arange(len(health_areas)) * group_spacing

    # Width/height allocated to the five scenario bars.
    bar_height = 0.14

    offsets = (
                  np.arange(len(scenarios))
                  - (len(scenarios) - 1) / 2
              ) * bar_height

    def plot_healthsystem(data, healthsystem):

        fig, ax = plt.subplots(
            figsize=(figure_width, figure_height)
        )

        for i, (label, scenario_prefix, color) in enumerate(scenarios):
            scenario_name = (
                f"{scenario_prefix} / "
                f"{healthsystem} Healthsystem Function"
            )

            # Retrieve values in the intended health-area order.
            scenario_df = data[scenario_name].reindex(
                health_areas
            )

            means = scenario_df["mean"].to_numpy()
            lower = scenario_df["lower"].to_numpy()
            upper = scenario_df["upper"].to_numpy()

            # Position of each scenario bar.
            ypos = y + offsets[i]

            # Draw horizontal bars.
            ax.barh(
                ypos,
                means,
                height=bar_height * 0.9,
                color=color,
                label=label,
                xerr=np.vstack([
                    means - lower,
                    upper - means,
                ]),
                capsize=3,
                error_kw={
                    "elinewidth": 0.9,
                    "capthick": 0.9,
                },
            )

        # Reference line at zero.
        ax.axvline(0, color="black", linewidth=1, )

        # Health-area labels.
        ax.set_yticks(y)
        ax.set_yticklabels(health_areas)

        # Colour health-area labels according to their groups.
        for tick, group in zip(
            ax.get_yticklabels(),
            health_areas,
        ):
            tick.set_color(CAUSE_GROUP_COLOR[group])
            tick.set_fontweight("bold")

        # Separate health areas with dashed horizontal lines.
        for i in range(len(health_areas) - 1):
            separator_position = (
                                     y[i] + y[i + 1]
                                 ) / 2

            ax.axhline(separator_position, color="grey", linestyle="--", linewidth=1, alpha=0.7, )

        # Put the first health area at the top.
        ax.invert_yaxis()

        # Axis labels.
        ax.set_xlabel(f"{outcome_label} averted", fontsize=12, )
        ax.set_ylabel("Health area", fontsize=12, )

        # Grid and formatting.
        ax.grid(axis="x", alpha=0.3, )

        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=11, length=0, pad=12, )
        ax.tick_params(axis="x", labelsize=10, )
        ax.set_yticks(y)
        ax.set_yticklabels(health_areas)
        ax.tick_params(axis="y", labelcolor="black")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=True, )
        # Leave space for labels and legend
        fig.subplots_adjust(left=0.20, right=0.98, top=0.97, bottom=0.15, )
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

    return (
        fig_default,
        ax_default,
        fig_improved,
        ax_improved,
    )


# Plot % DALYs averted by health area
def plot_percent_dalys_averted_by_cause(
    default_df,
    improved_df,
):
    return _plot_percent_averted_by_cause(
        default_df=default_df,
        improved_df=improved_df,
        outcome_label="DALYs",
    )


def plot_dalys_averted_by_cause(
    default_df,
    improved_df,
):
    return _plot_averted_by_cause(
        default_df=default_df,
        improved_df=improved_df,
        outcome_label="DALYs",
    )


# Plot % deaths averted by health area
def plot_percent_deaths_averted_by_cause(
    default_df,
    improved_df,
):
    return _plot_percent_averted_by_cause(
        default_df=default_df,
        improved_df=improved_df,
        outcome_label="Deaths",
    )


# Plot % DALYs averted by age group
def plot_percent_dalys_averted_by_age_group(default_df, improved_df, ):
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

    for df in [default_more, default_fewer, improved_more, improved_fewer, ]:
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

    offsets = [0.32, 0.16, 0.0, -0.16, -0.32]

    for offset, (label, df, color) in zip(offsets, scenarios_default):
        ax_default.barh(
            y_default + offset,
            df["mean"],
            height=0.20,
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
        ncol=5,
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
        ncol=5,
        frameon=True,
    )

    # fig_improved.suptitle(
    #     "% DALYs averted by age group on national level\n"
    #     "(2027–2034)"
    # )
    fig_improved.tight_layout(rect=[0, 0.08, 1, 1])

    return fig_default, ax_default, fig_improved, ax_improved


# Plot % deaths averted by age group
def plot_percent_deaths_averted_by_age_group(default_df, improved_df, ):
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

    offsets = [0.32, 0.16, 0.0, -0.16, -0.32]

    for offset, (label, df, color) in zip(offsets, scenarios_default):
        ax_default.barh(
            y_default + offset,
            df["mean"],
            height=0.20,
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
        ncol=5,
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
        ncol=5,
        frameon=True,
    )

    # fig_improved.suptitle(
    #     "% deaths averted by age group on national level\n"
    #     "(2027–2034)"
    # )
    fig_improved.tight_layout(rect=[0, 0.08, 1, 1])

    return fig_default, ax_default, fig_improved, ax_improved


# ---------------------------------------------------------------------------
# Convergence check: how do results change with the number of runs?
# ---------------------------------------------------------------------------

# Legend label and colour for each scenario, keyed by the part before " / ".
CONVERGENCE_SCENARIO_STYLE = {
    "Fewer Nurses": ("Fewer nurses", "indianred"),
    "More Nurses": ("More nurses", "steelblue"),
    "More CNP staff": ("More CNP", "darkgreen"),
    "More Nurses by District": ("More nurses by district", "mediumpurple"),
    "More CNP staff by District": ("More CNP by district", "orange"),
}


def select_runs(results, runs):
    """
    Keep only some runs from a results frame with (draw, run) columns.
    runs: an int n for the first n runs (0 .. n-1), or an iterable of run indices.
    """
    if isinstance(runs, int):
        runs = range(runs)
    run_level = results.columns.get_level_values("run")
    return results.loc[:, run_level.isin(list(runs))]


def runs_per_draw(results):
    """Number of runs present for each draw (failed runs are dropped by extract_results)."""
    return (
        results.columns.to_frame(index=False)
        .groupby("draw", sort=False)["run"]
        .count()
    )


def _runs_to_calculate(n_values):
    """
    The requested numbers of runs, plus one fewer than the smallest
    (e.g. 4 when n_values starts at 5). The extra one is calculated only so
    the first requested row also gets a change from the previous number of
    runs; it is dropped from the returned table and the plots.
    """
    n_values = sorted(n_values)
    extra = [n_values[0] - 1] if n_values[0] > 1 else []
    return n_values, extra + n_values


def convergence_of_averted(
    annual_outcome,
    baseline_scenario,
    calculate_fn,
    n_values=None,
    comparison_years=range(2027, 2035),
):
    """
    Repeat an 'averted' calculation (e.g. calculate_dalys_averted or
    calculate_percent_dalys_averted) using only the first n runs, for each n.

    Returns a long table with columns:
    n_runs, Scenario, mean, lower, upper, change_from_previous_n, pct_change_from_previous_n
    """
    if n_values is None:
        max_runs = int(annual_outcome.columns.get_level_values("run").max()) + 1
        n_values = range(2, max_runs + 1)  # one run gives no interval

    n_values, n_to_calculate = _runs_to_calculate(n_values)

    frames = []
    for n in n_to_calculate:
        res = calculate_fn(
            select_runs(annual_outcome, n),
            baseline_scenario=baseline_scenario,
            comparison_years=comparison_years,
        )
        res = res.astype(float)
        res.index.name = "Scenario"
        res = res.reset_index()
        res.insert(0, "n_runs", n)
        frames.append(res)

    conv = pd.concat(frames, ignore_index=True)

    # How much the estimate moves when one more run is added.
    conv = conv.sort_values(["Scenario", "n_runs"])
    previous_mean = conv.groupby("Scenario")["mean"].shift()
    conv["change_from_previous_n"] = conv["mean"] - previous_mean
    conv["pct_change_from_previous_n"] = (
        100 * conv["change_from_previous_n"] / previous_mean.abs()
    )

    # Keep only the requested numbers of runs.
    conv = conv[conv["n_runs"].isin(n_values)]

    return conv.sort_values(["n_runs", "Scenario"]).reset_index(drop=True)


def _order_convergence_scenarios(scenarios):
    """Sort scenario names in the same order as the other figures."""
    style_order = list(CONVERGENCE_SCENARIO_STYLE)

    def key(scenario):
        prefix = scenario.split(" / ")[0]
        return style_order.index(prefix) if prefix in style_order else len(style_order)

    return sorted(scenarios, key=key)


def _legend_across_rows(handles, labels, ncol):
    """Reorder legend entries so they read across rows (matplotlib fills by column)."""
    order = [i for c in range(ncol) for i in range(c, len(labels), ncol)]
    return [handles[i] for i in order], [labels[i] for i in order]


def plot_convergence(conv, ylabel, title=None):
    """Plot mean and interval against number of runs, one line per scenario."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for scenario in _order_convergence_scenarios(conv["Scenario"].unique()):
        d = conv[conv["Scenario"] == scenario].sort_values("n_runs")
        prefix = scenario.split(" / ")[0]
        label, color = CONVERGENCE_SCENARIO_STYLE.get(prefix, (scenario, "gray"))

        ax.plot(d["n_runs"], d["mean"], marker="o", color=color, label=label)
        ax.fill_between(d["n_runs"], d["lower"], d["upper"], color=color, alpha=0.15)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Number of runs")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(conv["n_runs"].unique()))
    ax.grid(axis="y", alpha=0.3)
    if title is not None:
        ax.set_title(title)

    # Legend below the plot, reading across rows in scenario order.
    ncol = 3
    handles, labels = _legend_across_rows(*ax.get_legend_handles_labels(), ncol)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=ncol,
        numpoints=1,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.25)

    return fig, ax


def convergence_of_averted_by_cause(
    dalys_by_cause,
    baseline_scenario,
    calculate_fn,
    n_values=None,
    row_name="Health area",
):
    """
    Repeat a by-cause 'averted' calculation (e.g. calculate_dalys_averted_by_cause
    or calculate_percent_dalys_averted_by_cause) using only the first n runs,
    for each n.

    The rows of `dalys_by_cause` can be health areas or anything else, e.g.
    districts (then use row_name="District").

    Returns a long table with columns:
    n_runs, Scenario, <row_name>, mean, lower, upper,
    change_from_previous_n, pct_change_from_previous_n
    """
    if n_values is None:
        max_runs = int(dalys_by_cause.columns.get_level_values("run").max()) + 1
        n_values = range(2, max_runs + 1)  # one run gives no interval

    n_values, n_to_calculate = _runs_to_calculate(n_values)

    frames = []
    for n in n_to_calculate:
        res = calculate_fn(
            select_runs(dalys_by_cause, n),
            baseline_scenario=baseline_scenario,
        )
        for scenario, scenario_df in res.items():
            temp = scenario_df.astype(float).copy()
            temp.index.name = row_name
            temp = temp.reset_index()
            temp.insert(0, "Scenario", scenario)
            temp.insert(0, "n_runs", n)
            frames.append(temp)

    conv = pd.concat(frames, ignore_index=True)

    # How much the estimate moves when one more run is added.
    conv = conv.sort_values(["Scenario", row_name, "n_runs"])
    previous_mean = conv.groupby(["Scenario", row_name])["mean"].shift()
    conv["change_from_previous_n"] = conv["mean"] - previous_mean
    conv["pct_change_from_previous_n"] = (
        100 * conv["change_from_previous_n"] / previous_mean.abs()
    )

    # Keep only the requested numbers of runs.
    conv = conv[conv["n_runs"].isin(n_values)]

    return conv.sort_values(["n_runs", "Scenario", row_name]).reset_index(drop=True)


def plot_convergence_by_cause(
    conv,
    ylabel,
    title=None,
    ncols=3,
    panel_col="Health area",
    exclude=("Other",),
    panel_size=(5.5, 3.8),
    title_fontsize=None,
):
    """
    One panel per health area (excluding Other, alphabetical as in the
    by-cause plot), one line per scenario, against number of runs.

    panel_col: column of `conv` that defines the panels, e.g. "District".
    exclude: panel values to leave out.
    panel_size: (width, height) of each panel in inches.
    """
    health_areas = sorted(
        (area for area in conv[panel_col].unique() if area not in exclude),
        key=str.lower,
    )
    scenarios = _order_convergence_scenarios(conv["Scenario"].unique())
    n_values = sorted(conv["n_runs"].unique())

    nrows = int(np.ceil(len(health_areas) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )

    for ax, area in zip(axes.flat, health_areas):
        area_df = conv[conv[panel_col] == area]

        for scenario in scenarios:
            d = area_df[area_df["Scenario"] == scenario].sort_values("n_runs")
            prefix = scenario.split(" / ")[0]
            label, color = CONVERGENCE_SCENARIO_STYLE.get(prefix, (scenario, "gray"))

            ax.plot(
                d["n_runs"],
                d["mean"],
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color=color,
                label=label,
            )
            ax.fill_between(d["n_runs"], d["lower"], d["upper"], color=color, alpha=0.12)

        ax.axhline(0, color="black", linewidth=0.8)
        if title_fontsize is None:
            ax.set_title(area, fontweight="bold")
        else:
            ax.set_title(area, fontweight="bold", fontsize=title_fontsize)
        ax.set_xticks(n_values)
        ax.grid(axis="y", alpha=0.3)
        # Full numbers with thousands separators (no "1e6" offset); small values as-is.
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}")
        )

    # Hide any unused panels.
    for ax in axes.flat[len(health_areas):]:
        ax.set_visible(False)

    fig.supylabel(ylabel)
    if title is not None:
        fig.suptitle(title)

    # Leave room at the bottom for the x-axis label and the shared legend.
    fig.tight_layout(rect=[0.01, 0.10, 1, 1])
    fig.supxlabel("Number of runs", y=0.055)

    # One shared legend below all panels.
    legend_ncol = 5
    handles, labels = _legend_across_rows(
        *axes.flat[0].get_legend_handles_labels(), legend_ncol
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=legend_ncol,
        numpoints=1,
        frameon=False,
    )

    return fig, axes


# ---------------------------------------------------------------------------
# District staff check: are district DALYs averted related to staff changes?
# ---------------------------------------------------------------------------

# Cadres changed by the scenarios (nurse scenarios only change nursing; the CNP
# scenarios change all three), summed to give one staff measure per district.
CNP_CADRES = ("Clinical", "Nursing_and_Midwifery", "Pharmacy")


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


def extract_staff_counts(results_folder, custom_generate_series=get_yearly_hr_count):
    """
    Staff counts by (year, District, Cadre) for every draw and run.

    custom_generate_series: turns the number_of_hcw_staff log into a Series
        indexed by (year, District, Cadre); get_yearly_hr_count by default.
    """
    return extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="number_of_hcw_staff",
        custom_generate_series=custom_generate_series,
        do_scaling=False,
    )


def staff_counts_to_long(staff_counts, districts=None):
    """
    Turn extract_staff_counts output (with scenario names as draws) into the
    long table used by plot_staff_change_lines_by_district:
    columns Scenario, Cadre, District, Year, Staff (mean over runs).

    Likoma is counted as part of Nkhata Bay. districts: if given, keep only
    these districts (e.g. the districts in the DALY data).
    """
    staff = staff_counts.T.groupby(level="draw").mean().T
    staff = staff.rename(index={"Likoma": "Nkhata Bay"}, level="District")
    staff = staff.groupby(level=["year", "District", "Cadre"]).sum()

    staff_df = (
        staff.stack()
        .rename("Staff")
        .reset_index()
        .rename(columns={"year": "Year", "draw": "Scenario"})
    )
    staff_df["Year"] = staff_df["Year"].astype(int)

    if districts is not None:
        dropped = sorted(set(staff_df["District"]) - set(districts))
        if dropped:
            print("NOTE: staff locations not in the DALY data (left out):", dropped)
        staff_df = staff_df[staff_df["District"].isin(list(districts))]

    return staff_df[["Scenario", "Cadre", "District", "Year", "Staff"]]


def staff_change_by_district(
    staff_df,
    scenarios,
    baseline_scenario,
    cadres=CNP_CADRES,
    years=range(2027, 2035),
    relative=False,
):
    """
    Change in staff in each district for each scenario compared with the
    baseline: average yearly staff over `years`, summed over `cadres`.

    staff_df: output of staff_counts_to_long.
    relative=False: change in number of staff (scenario - baseline);
    relative=True:  % change (100 * (scenario - baseline) / baseline).

    Returns: rows District, one column per scenario (baseline excluded).
    """
    selected = staff_df[staff_df["Cadre"].isin(cadres) & staff_df["Year"].isin(list(years))]
    per_district = (
        selected.groupby(["Scenario", "District", "Year"])["Staff"].sum()  # sum over cadres
        .groupby(["Scenario", "District"]).mean()                          # average per year
        .unstack("Scenario")
    )

    baseline = per_district[baseline_scenario]
    alternatives = [s for s in scenarios if s != baseline_scenario and s in per_district.columns]
    change = per_district[alternatives].sub(baseline, axis=0)
    if relative:
        change = (100 * change.div(baseline, axis=0)).replace([np.inf, -np.inf], np.nan)
    return change


# Scenario/cadre combinations to plot: (scenario prefix, cadre, legend label, colour).
# The order here is the legend order (two rows of four, reading across).
CHANGE_LINE_COMBINATIONS = [
    ("Baseline Nurses", "Nursing_and_Midwifery", "Baseline nurses", "black"),
    ("Fewer Nurses", "Nursing_and_Midwifery", "Fewer nurses", "indianred"),
    ("More Nurses", "Nursing_and_Midwifery", "More nurses", "steelblue"),
    ("More CNP staff", "Clinical", "More CNP - Clinical", "darkgreen"),
    ("More CNP staff", "Pharmacy", "More CNP - Pharmacy", "darkorange"),
    ("More Nurses by District", "Nursing_and_Midwifery", "More nurses by district", "mediumpurple"),
    ("More CNP staff by District", "Clinical", "More CNP by district - Clinical", "limegreen"),
    ("More CNP staff by District", "Pharmacy", "More CNP by district - Pharmacy", "gold"),
]


def plot_staff_change_lines_by_district(
    staff_df,
    healthsystem="Default",
    year_before=2026,
    year_after=2027,
    combinations=CHANGE_LINE_COMBINATIONS,
    symlog_threshold=20,
    connect_points="faint",
    line_style="--",
    title=None,
):
    """
    Change in number of staff (year_after - year_before) by district, one
    series per scenario/cadre combination.

    staff_df: long-format DataFrame with columns
        Scenario, Cadre, District, Year, Staff (see staff_counts_to_long).
    healthsystem: "Default" or "Improved"; selects scenarios ending in
        " / <healthsystem> Healthsystem Function".
    symlog_threshold: the y-axis is linear between -threshold and +threshold
        and compressed (log-like) beyond, so small changes such as Pharmacy
        stay readable next to large nurse changes. None for a linear axis.
    connect_points: "faint" shows solid dots joined by thin, light lines;
        True draws full lines; False shows dots only.
    line_style: style of the connecting lines: "--" (dashed), "-" (solid)
        or ":" (dotted).

    Figure caption note (with the default settings):
        The y-axis is linear between -20 and +20 staff and logarithmic beyond,
        so that small and large changes can be shown together. Dashed lines
        are a visual guide for comparing scenarios only.
    """
    suffix = f" / {healthsystem} Healthsystem Function"

    # Fixed district order shared by every series.
    districts = sorted(staff_df["District"].unique())
    x = np.arange(len(districts))

    fig, ax = plt.subplots(figsize=(16, 8))

    for prefix, cadre, label, color in combinations:
        scen_df = staff_df[
            (staff_df["Scenario"] == f"{prefix}{suffix}")
            & (staff_df["Cadre"] == cadre)
        ]
        if scen_df.empty:
            continue

        by_year = scen_df.pivot_table(
            index="District", columns="Year", values="Staff", aggfunc="sum"
        ).reindex(districts)
        change = by_year[year_after] - by_year[year_before]

        if connect_points == "faint":
            # Solid dots joined by a light line: the line colour carries the
            # transparency, the marker colours stay solid.
            ax.plot(
                x,
                change,
                color=to_rgba(color, 0.3 if line_style == "-" else 0.5),
                linewidth=0.8,
                linestyle=line_style,
                marker="o",
                markersize=5,
                markerfacecolor=to_rgba(color, 0.85),
                markeredgecolor=to_rgba(color, 0.85),
                label=label,
            )
        else:
            ax.plot(
                x,
                change,
                color=color,
                marker="o",
                linestyle=line_style if connect_points else "none",
                linewidth=1.2,
                markersize=5,
                label=label,
                alpha=0.7,
            )

    # Reference line: no change.
    ax.axhline(0, linestyle="--", linewidth=1)

    if symlog_threshold is not None:
        ax.set_yscale("symlog", linthresh=symlog_threshold, linscale=0.8)
        ticks = [-500, -200, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 200, 500]
        low, high = ax.get_ylim()
        ax.set_yticks([t for t in ticks if low <= t <= high])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_locator(plt.NullLocator())

    ax.set_xticks(x)
    ax.set_xticklabels(districts, rotation=90)
    ax.set_xlabel("District")
    ax.set_ylabel(f"Change in number of staff ({year_after} - {year_before})")
    ax.grid(axis="y", alpha=0.3)
    if title is None:
        title = f"Change in workforce size by cadre and district, {year_before} to {year_after}"
    ax.set_title(title)

    # Legend below the plot, two rows reading across.
    ncol = 4
    handles, labels = ax.get_legend_handles_labels()
    order = [i for c in range(ncol) for i in range(c, len(labels), ncol)]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=ncol,
        numpoints=1,
        handlelength=2.5,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.38, right=0.98)

    return fig, ax


def dalys_averted_vs_staff_change(
    area_by_district,
    staff_change,
    baseline_scenario,
    relative=True,
):
    """
    DALYs averted (run-by-run vs baseline, then mean and 2.5–97.5% range across
    runs) per district, next to the district's staff change, for each scenario.

    area_by_district: DALYs for one health area, rows District, columns (draw, run).
    staff_change: output of staff_change_by_district (rows District, columns scenario).
    relative: True -> % DALYs averted; False -> number of DALYs averted.

    Returns a long table: Scenario, District, mean, lower, upper, staff_change.
    """
    districts = area_by_district.index.intersection(staff_change.index)
    calculate_fn = (
        calculate_percent_dalys_averted_by_cause if relative
        else calculate_dalys_averted_by_cause
    )
    averted = calculate_fn(area_by_district.loc[districts], baseline_scenario=baseline_scenario)

    table = pd.concat(
        {s: df.astype(float) for s, df in averted.items() if s in staff_change.columns},
        names=["Scenario", "District"],
    ).reset_index()
    table["staff_change"] = [
        staff_change.at[d, s] for s, d in zip(table["Scenario"], table["District"])
    ]
    return table


def plot_dalys_averted_vs_staff_change(
    table,
    xlabel,
    ylabel,
    title=None,
    label_districts=True,
    ncols=3,
):
    """
    One panel per scenario: each dot is a district, placed by its staff change
    (x) and DALYs averted (y, mean with 2.5–97.5% range across runs).
    The Spearman rank correlation is shown where staff changes differ
    between districts.
    """
    scenarios = _order_convergence_scenarios(table["Scenario"].unique())
    nrows = int(np.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.6 * nrows),
        squeeze=False,
        sharey=True,
    )

    for ax, scenario in zip(axes.flat, scenarios):
        d = table[table["Scenario"] == scenario].dropna(subset=["staff_change"])
        prefix = scenario.split(" / ")[0]
        label, color = CONVERGENCE_SCENARIO_STYLE.get(prefix, (scenario, "gray"))

        ax.errorbar(
            d["staff_change"],
            d["mean"],
            yerr=[d["mean"] - d["lower"], d["upper"] - d["mean"]],
            fmt="o",
            markersize=5,
            color=color,
            ecolor=to_rgba(color, 0.35),
            elinewidth=0.8,
            capsize=0,
        )

        if label_districts:
            for _, row in d.iterrows():
                ax.annotate(
                    row["District"],
                    (row["staff_change"], row["mean"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color="dimgrey",
                )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":")

        # Ignore tiny rounding differences when deciding whether districts differ.
        if d["staff_change"].round(6).nunique() > 1:
            rho = d["staff_change"].corr(d["mean"], method="spearman")
            note = f"Spearman ρ = {rho:.2f}"
        else:
            note = "Same staff change in every district"
        ax.text(0.02, 0.98, note, transform=ax.transAxes, ha="left", va="top", fontsize=9)

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def split_dalys_averted_by_staff_change(
    area_by_district,
    staff_change,
    baseline_scenario,
):
    """
    Split the national DALYs averted for one health area into districts that
    gain staff, lose staff, or have no change, for each scenario.

    Within each run, DALYs averted (baseline - scenario, same run) are summed
    over the districts in each group; the mean and 2.5–97.5% range are then
    taken across runs.

    staff_change: number-of-staff change from staff_change_by_district
        (relative=False); only its sign is used.

    Returns a table: Scenario, Group, n_districts, staff_change, mean, lower, upper.
    """
    districts = area_by_district.index.intersection(staff_change.index)

    # Run-level DALYs averted: rows District, columns (run, draw).
    run_averted = -1.0 * find_difference_relative_to_comparison_series_dataframe(
        area_by_district.loc[districts],
        comparison=baseline_scenario,
        scaled=False,
    )

    rows = []
    for scenario in staff_change.columns:
        if scenario not in run_averted.columns.get_level_values("draw"):
            continue
        scenario_runs = run_averted.xs(scenario, axis=1, level="draw")  # columns = run
        change = staff_change.loc[districts, scenario]

        sign = np.sign(change.mask(np.isclose(change, 0), 0))  # tiny changes count as 0
        groups = {
            "Districts gaining staff": sign > 0,
            "Districts losing staff": sign < 0,
            "Districts with no change": sign == 0,
            "All districts": sign == sign,
        }

        for group, mask in groups.items():
            members = mask[mask].index
            totals = scenario_runs.loc[members].sum(axis=0)  # one value per run
            rows.append({
                "Scenario": scenario,
                "Group": group,
                "n_districts": len(members),
                "staff_change": change[members].sum(),
                "mean": totals.mean(),
                "lower": totals.quantile(0.025),
                "upper": totals.quantile(0.975),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run-level check: DALYs and DALYs averted in each individual run
# ---------------------------------------------------------------------------

def run_level_tables(area_by_district, baseline_scenario):
    """
    For one health area (rows District, columns (draw, run)), return:

    dalys_by_run:     national DALYs, rows run, one column per scenario (incl. baseline)
    averted_by_run:   national DALYs averted (baseline - scenario, same run),
                      rows run, one column per scenario
    averted_by_district_and_run: DALYs averted per district,
                      rows District, columns (Scenario, run)
    """
    dalys_by_run = area_by_district.sum(axis=0).unstack("draw")
    averted_by_run = dalys_by_run.rsub(dalys_by_run[baseline_scenario], axis=0).drop(
        columns=baseline_scenario
    )

    baseline = area_by_district.xs(baseline_scenario, axis=1, level="draw")
    averted_by_district_and_run = pd.concat(
        {
            scenario: baseline - area_by_district.xs(scenario, axis=1, level="draw")
            for scenario in area_by_district.columns.unique(level="draw")
            if scenario != baseline_scenario
        },
        axis=1,
        names=["Scenario", "run"],
    )

    return dalys_by_run, averted_by_run, averted_by_district_and_run


def plot_averted_by_run(averted_by_run, ylabel, title=None, zero_line=True):
    """
    One column per scenario: each dot is one run, labelled with its run number;
    the black dash is the mean across runs.

    Also used for DALYs by run (plot_dalys_by_run), where the baseline is
    included and drawn first, in black.
    """
    style = {"Baseline Nurses": ("Baseline", "black"), **CONVERGENCE_SCENARIO_STYLE}
    scenarios = sorted(
        averted_by_run.columns,
        key=lambda s: list(style).index(s.split(" / ")[0]) if s.split(" / ")[0] in style else len(style),
    )
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, scenario in enumerate(scenarios):
        _, color = style.get(scenario.split(" / ")[0], (scenario, "gray"))
        values = averted_by_run[scenario]

        ax.scatter([i] * len(values), values, color=color, s=30, zorder=3)
        for run, value in values.items():
            ax.annotate(str(run), (i, value), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color="dimgrey")
        ax.plot([i - 0.25, i + 0.25], [values.mean()] * 2, color="black", linewidth=2)

    if zero_line:
        ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(
        [style.get(s.split(" / ")[0], (s, None))[0].replace(" by ", "\nby ") for s in scenarios]
    )
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

    return fig, ax


def plot_dalys_by_run(dalys_by_run, ylabel, title=None):
    """
    DALYs in each run for every scenario, including the baseline, in the same
    format as plot_averted_by_run. The y-axis does not start at zero, so the
    run-to-run variation is visible.
    """
    return plot_averted_by_run(dalys_by_run, ylabel, title=title, zero_line=False)


def plot_dalys_by_run_by_district(area_by_district, ylabel, title=None, ncols=6, zero_line=False):
    """
    DALYs in each run for each district: one panel per district (own y-axis),
    one column of dots per scenario (each dot one run, baseline first in black),
    black dash = mean across runs. Scenario colours are given in the legend.

    area_by_district: DALYs for one health area and one health system,
        rows District, columns (scenario, run). Also used for DALYs averted
        by district and run (see plot_averted_by_run_by_district).
    zero_line: draw a horizontal line at zero.
    """
    style = {"Baseline Nurses": ("Baseline", "black"), **CONVERGENCE_SCENARIO_STYLE}
    order = list(style)
    scenarios = sorted(
        area_by_district.columns.unique(level=0),
        key=lambda s: order.index(s.split(" / ")[0]) if s.split(" / ")[0] in order else len(order),
    )
    districts = sorted(area_by_district.index, key=str.lower)

    nrows = int(np.ceil(len(districts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)

    for ax, district in zip(axes.flat, districts):
        district_values = area_by_district.loc[district]
        for i, scenario in enumerate(scenarios):
            color = style.get(scenario.split(" / ")[0], (scenario, "gray"))[1]
            values = district_values.xs(scenario, level=0)
            ax.scatter([i] * len(values), values, color=color, s=10, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [values.mean()] * 2, color="black", linewidth=1.2)

        if zero_line:
            ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(district, fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(scenarios) - 0.4)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    for ax in axes.flat[len(districts):]:
        ax.set_visible(False)

    fig.supylabel(ylabel)
    if title is not None:
        fig.suptitle(title)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=style.get(s.split(" / ")[0], (s, "gray"))[1],
               label=style.get(s.split(" / ")[0], (s, None))[0])
        for s in scenarios
    ]
    handles.append(Line2D([], [], color="black", linewidth=1.2, label="Mean across runs"))
    fig.tight_layout(rect=[0.01, 0.04, 1, 1])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), frameon=False)

    return fig, axes


def plot_averted_by_run_by_district(averted_by_district_and_run, ylabel, title=None, ncols=6):
    """
    DALYs averted (baseline - scenario, same run) in each run for each district,
    in the same format as plot_dalys_by_run_by_district, with a line at zero.

    averted_by_district_and_run: rows District, columns (Scenario, run),
        from run_level_tables.
    """
    return plot_dalys_by_run_by_district(
        averted_by_district_and_run, ylabel, title=title, ncols=ncols, zero_line=True
    )


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
    parser.add_argument(
        "--convergence-check",
        action="store_true",
        help="Whether to check how DALYs averted change with the number of runs",
    )
    parser.add_argument(
        "--convergence-health-area",
        default="HIV/AIDS",
        help="Health area to check district by district "
             "(used with --convergence-check and --district-staff-check)",
    )
    parser.add_argument(
        "--district-staff-check",
        action="store_true",
        help="Whether to compare district DALYs averted (for the health area above) "
             "with district staff changes",
    )
    parser.add_argument(
        "--run-level-check",
        action="store_true",
        help="Whether to save DALYs and DALYs averted in each run (for the health "
             "area above), nationally and by district",
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

    # Plot 2: DALYs averted relative to baseline (2027–2034)
    num_dalys_averted = calculate_dalys_averted(
        annual_dalys.loc[
            :,
            annual_dalys.columns.get_level_values(0).isin(default_hs_scenarios)
        ],
        baseline_scenario=baseline_scenario,
        comparison_years=range(2027, 2035),
    )

    num_dalys_averted_improved = calculate_dalys_averted(
        annual_dalys.loc[
            :,
            annual_dalys.columns.get_level_values(0).isin(improved_hs_scenarios)
        ],
        baseline_scenario=baseline_improved_scenario,
        comparison_years=range(2027, 2035),
    )

    fig_2_1_default, ax_2_1_default, fig_2_1_improved, ax_2_1_improved = (
        plot_dalys_averted_comparison(
            num_dalys_averted,
            num_dalys_averted_improved,
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

    # Aggregate individual causes into broader health areas.
    dalys_by_cause = aggregate_causes_by_health_area(
        dalys_by_cause
    )
    # Alphabetical order of causes
    # cause_order = sorted(dalys_by_cause.index.tolist(), key=str.lower, reverse=True)

    # Default Healthsystem
    dalys_by_cause_default = (
        dalys_by_cause.loc[
            :,
            dalys_by_cause.columns
            .get_level_values(0)
            .isin(default_hs_scenarios)
        ]
    )

    num_by_cause_default = (
        calculate_dalys_averted_by_cause(
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

    num_by_cause_improved = (
        calculate_dalys_averted_by_cause(
            dalys_by_cause_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_9_1_default, ax_9_1_default, fig_9_1_improved, ax_9_1_improved = (
        plot_dalys_averted_by_cause(
            num_by_cause_default,
            num_by_cause_improved,
        )
    )

    # District-level DALYs for one health area (e.g. HIV/AIDS), used by the
    # convergence check and the district staff check.
    if args.convergence_check or args.district_staff_check or args.run_level_check:
        area = args.convergence_health_area
        area_slug = re.sub(r"[^A-Za-z0-9]+", "_", area).strip("_").lower()  # "HIV/AIDS" -> "hiv_aids"

        dalys_by_area_district = aggregate_causes_by_health_area_by_district(
            extract_dalys_by_cause_and_district(results_folder).pipe(
                set_param_names_as_column_index_level_0,
                param_names=param_names,
            )
        )

        # Rows: districts; columns: (draw, run).
        area_by_district = dalys_by_area_district.xs(area, level="Health area")

        # Check: in every run, the district DALYs for this health area must
        # add up to the national DALYs for the same health area.
        area_total_from_districts = area_by_district.sum(axis=0)
        area_total_national = dalys_by_cause.loc[area]
        assert (area_total_from_districts.index == area_total_national.index).all(), (
            "District and national results do not have the same (draw, run) columns"
        )
        assert np.allclose(
            area_total_from_districts.to_numpy(),
            area_total_national.to_numpy(),
            rtol=1e-9,
            atol=1e-6,
        ), (
            f"{area} DALYs summed over districts do not match the national total in "
            f"every run; largest difference: "
            f"{(area_total_from_districts - area_total_national).abs().max():,.3f}"
        )

    # Convergence check: DALYs averted and % DALYs averted vs number of runs
    convergence_figures = {}
    convergence_tables = {}

    # Numbers of runs to compare in the convergence check.
    convergence_runs = range(5, 11)  # 5, 6, 7, 8, 9, 10 runs

    if args.convergence_check:
        print("\nRuns available per scenario:")
        print(runs_per_draw(annual_dalys).to_string())

        for hs_label, hs_scenarios, hs_baseline in [
            ("default", default_hs_scenarios, baseline_scenario),
            ("improved", improved_hs_scenarios, baseline_improved_scenario),
        ]:
            hs_annual_dalys = annual_dalys.loc[
                :,
                annual_dalys.columns.get_level_values(0).isin(hs_scenarios)
            ]

            for metric, calculate_fn, ylabel in [
                ("dalys_averted", calculate_dalys_averted, "DALYs averted (2027–2034)"),
                ("pct_dalys_averted", calculate_percent_dalys_averted, "% DALYs averted (2027–2034)"),
            ]:
                key = f"{metric}_{hs_label}"

                conv = convergence_of_averted(
                    hs_annual_dalys,
                    baseline_scenario=hs_baseline,
                    calculate_fn=calculate_fn,
                    n_values=convergence_runs,
                    comparison_years=range(2027, 2035),
                )
                convergence_tables[key] = conv

                fig_conv, _ = plot_convergence(
                    conv,
                    ylabel=ylabel,
                    title=f"{hs_label.capitalize()} Healthsystem Function",
                )
                convergence_figures[key] = fig_conv

                print(f"\nConvergence: {key}")
                print(conv.round(3).to_string(index=False))

        # By health area (2027–2034), matching the DALYs-averted-by-cause plot
        for hs_label, hs_dalys_by_cause, hs_baseline in [
            ("default", dalys_by_cause_default, baseline_scenario),
            ("improved", dalys_by_cause_improved, baseline_improved_scenario),
        ]:
            for metric, calculate_fn, ylabel in [
                ("dalys_by_cause", calculate_dalys_averted_by_cause, "DALYs averted (2027–2034)"),
                ("pct_dalys_by_cause", calculate_percent_dalys_averted_by_cause, "% DALYs averted (2027–2034)"),
            ]:
                key = f"{metric}_{hs_label}"

                conv = convergence_of_averted_by_cause(
                    hs_dalys_by_cause,
                    baseline_scenario=hs_baseline,
                    calculate_fn=calculate_fn,
                    n_values=convergence_runs,
                )
                convergence_tables[key] = conv

                fig_conv, _ = plot_convergence_by_cause(
                    conv,
                    ylabel=ylabel,
                    title=f"{hs_label.capitalize()} Healthsystem Function",
                )
                convergence_figures[key] = fig_conv

                print(f"\nConvergence: {key}")
                print(conv.round(3).to_string(index=False))

        # By district, for one health area (extracted above)
        for hs_label, hs_scenarios, hs_baseline in [
            ("default", default_hs_scenarios, baseline_scenario),
            ("improved", improved_hs_scenarios, baseline_improved_scenario),
        ]:
            hs_area_by_district = area_by_district.loc[
                :,
                area_by_district.columns.get_level_values(0).isin(hs_scenarios)
            ]

            for metric, calculate_fn, ylabel in [
                ("dalys", calculate_dalys_averted_by_cause, f"{area} DALYs averted (2027–2034)"),
                ("pct_dalys", calculate_percent_dalys_averted_by_cause, f"% {area} DALYs averted (2027–2034)"),
            ]:
                key = f"{metric}_{area_slug}_by_district_{hs_label}"

                conv = convergence_of_averted_by_cause(
                    hs_area_by_district,
                    baseline_scenario=hs_baseline,
                    calculate_fn=calculate_fn,
                    n_values=convergence_runs,
                    row_name="District",
                )
                convergence_tables[key] = conv

                fig_conv, _ = plot_convergence_by_cause(
                    conv,
                    ylabel=ylabel,
                    title=f"{area} by district, {hs_label.capitalize()} Healthsystem Function",
                    ncols=6,
                    panel_col="District",
                    exclude=(),
                    panel_size=(3.4, 2.5),
                    title_fontsize=9,
                )
                convergence_figures[key] = fig_conv

    # District staff check: district DALYs averted vs district staff change
    district_staff_figures = {}
    district_staff_tables = {}

    if args.district_staff_check:
        staff_counts = extract_staff_counts(results_folder).pipe(
            set_param_names_as_column_index_level_0,
            param_names=param_names,
        )

        staff_label = "clinical, nursing and pharmacy staff vs baseline\n(average per year, 2027–2034)"

        # Long table of staff by scenario, cadre, district and year, for the
        # districts in the DALY data.
        staff_df = staff_counts_to_long(staff_counts, districts=area_by_district.index)

        for hs_label, hs_scenarios, hs_baseline in [
            ("default", default_hs_scenarios, baseline_scenario),
            ("improved", improved_hs_scenarios, baseline_improved_scenario),
        ]:
            # First: change in staff by district (2027 - 2026) in each scenario.
            fig_staff, _ = plot_staff_change_lines_by_district(
                staff_df,
                healthsystem=hs_label.capitalize(),
            )
            district_staff_figures[f"staff_change_by_district_2027_vs_2026_{hs_label}"] = fig_staff

            hs_area_by_district = area_by_district.loc[
                :,
                area_by_district.columns.get_level_values(0).isin(hs_scenarios)
            ]
            staff_change_num = staff_change_by_district(staff_df, hs_scenarios, hs_baseline)
            staff_change_pct = staff_change_by_district(
                staff_df, hs_scenarios, hs_baseline, relative=True
            )

            # Check 2: DALYs averted vs staff change, district by district.
            for measure, staff_change, relative, xlabel, ylabel in [
                ("pct", staff_change_pct, True,
                 f"% change in {staff_label}", f"% {area} DALYs averted (2027–2034)"),
                ("num", staff_change_num, False,
                 f"Change in number of {staff_label}", f"{area} DALYs averted (2027–2034)"),
            ]:
                key = f"{area_slug}_dalys_averted_vs_staff_change_{measure}_{hs_label}"

                table = dalys_averted_vs_staff_change(
                    hs_area_by_district,
                    staff_change,
                    baseline_scenario=hs_baseline,
                    relative=relative,
                )
                district_staff_tables[key] = table

                fig_staff, _ = plot_dalys_averted_vs_staff_change(
                    table,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=f"{area}: DALYs averted vs staff change by district, "
                          f"{hs_label.capitalize()} Healthsystem Function",
                )
                district_staff_figures[key] = fig_staff

            # Check 3: national DALYs averted split by districts gaining/losing staff.
            key = f"{area_slug}_dalys_averted_split_by_staff_change_{hs_label}"
            split = split_dalys_averted_by_staff_change(
                hs_area_by_district,
                staff_change_num,
                baseline_scenario=hs_baseline,
            )
            district_staff_tables[key] = split

            print(f"\n{area} DALYs averted (2027–2034) by districts gaining/losing staff, "
                  f"{hs_label.capitalize()} Healthsystem Function:")
            print(split.round(1).to_string(index=False))

    # Run-level check: DALYs and DALYs averted in each individual run
    run_level_figures = {}
    run_level_tables_out = {}

    if args.run_level_check:
        for hs_label, hs_scenarios, hs_baseline in [
            ("default", default_hs_scenarios, baseline_scenario),
            ("improved", improved_hs_scenarios, baseline_improved_scenario),
        ]:
            hs_area_by_district = area_by_district.loc[
                :,
                area_by_district.columns.get_level_values(0).isin(hs_scenarios)
            ]
            dalys_by_run, averted_by_run, averted_by_district_and_run = run_level_tables(
                hs_area_by_district, hs_baseline
            )

            run_level_tables_out[f"{area_slug}_dalys_by_run_{hs_label}"] = dalys_by_run
            run_level_tables_out[f"{area_slug}_dalys_averted_by_run_{hs_label}"] = averted_by_run
            run_level_tables_out[f"{area_slug}_dalys_averted_by_district_and_run_{hs_label}"] = (
                averted_by_district_and_run
            )

            fig_runs, _ = plot_averted_by_run(
                averted_by_run,
                ylabel=f"{area} DALYs averted (2027–2034)",
                title=f"{area} DALYs averted in each run, {hs_label.capitalize()} Healthsystem Function",
            )
            run_level_figures[f"{area_slug}_dalys_averted_by_run_{hs_label}"] = fig_runs

            fig_dalys_runs, _ = plot_dalys_by_run(
                dalys_by_run,
                ylabel=f"{area} DALYs (2027–2034)",
                title=f"{area} DALYs in each run, {hs_label.capitalize()} Healthsystem Function",
            )
            run_level_figures[f"{area_slug}_dalys_by_run_{hs_label}"] = fig_dalys_runs

            # DALYs in each run, district by district.
            fig_district_runs, _ = plot_dalys_by_run_by_district(
                hs_area_by_district,
                ylabel=f"{area} DALYs (2027–2034)",
                title=f"{area} DALYs in each run by district, {hs_label.capitalize()} Healthsystem Function",
            )
            run_level_figures[f"{area_slug}_dalys_by_district_and_run_{hs_label}"] = fig_district_runs
            run_level_tables_out[f"{area_slug}_dalys_by_district_and_run_{hs_label}"] = hs_area_by_district

            # DALYs averted vs baseline in each run, district by district.
            fig_district_averted, _ = plot_averted_by_run_by_district(
                averted_by_district_and_run,
                ylabel=f"{area} DALYs averted (2027–2034)",
                title=f"{area} DALYs averted in each run by district, "
                      f"{hs_label.capitalize()} Healthsystem Function",
            )
            run_level_figures[f"{area_slug}_dalys_averted_by_district_and_run_{hs_label}"] = (
                fig_district_averted
            )

            print(f"\n{area} DALYs (2027–2034) in each run, {hs_label.capitalize()}:")
            print(dalys_by_run.round(0).to_string())
            print(f"\n{area} DALYs averted (2027–2034) in each run, {hs_label.capitalize()}:")
            print(averted_by_run.round(0).to_string())

    # Showing figures
    if args.show_figures:
        plt.show()

    # Saving figures
    if args.save_figures:

        fig_2_1_default.savefig(
            results_folder /
            "dalys_averted_vs_baseline_2027_2034_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_2_1_improved.savefig(
            results_folder /
            "dalys_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_9_1_default.savefig(
            results_folder /
            "dalys_averted_by_cause_national_level_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_9_1_improved.savefig(
            results_folder /
            "dalys_averted_by_cause_national_level_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        # Convergence check outputs
        for key, fig_conv in convergence_figures.items():
            fig_conv.savefig(
                results_folder / f"convergence_{key}.pdf",
                bbox_inches="tight",
            )

        for key, conv in convergence_tables.items():
            convergence_file = results_folder / f"convergence_{key}.csv"
            conv.to_csv(convergence_file, index=False)
            print(f"Convergence table saved to: {convergence_file}")

        # District staff check outputs
        for key, fig_staff in district_staff_figures.items():
            fig_staff.savefig(results_folder / f"{key}.pdf", bbox_inches="tight")

        for key, table in district_staff_tables.items():
            table_file = results_folder / f"{key}.csv"
            table.to_csv(table_file, index=False)
            print(f"District staff check table saved to: {table_file}")

        # Run-level check outputs
        for key, fig_runs in run_level_figures.items():
            fig_runs.savefig(results_folder / f"{key}.pdf", bbox_inches="tight")

        for key, table in run_level_tables_out.items():
            table_file = results_folder / f"{key}.csv"
            table.to_csv(table_file)
            print(f"Run-level table saved to: {table_file}")
